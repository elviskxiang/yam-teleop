"""MDF-lite denoiser: a causal transformer over (time x modality) tokens, each token with its own noise level.

Token (t, m) = Linear_m(x_{t,m}) + modality embedding + time-step embedding + embedding of its noise level k_{t,m}.
Attention is causal in time: a token sees every modality at the same and earlier time steps (as in Diffusion Forcing,
so the model can be rolled out step by step). The network predicts the clean value x0 of every token.

Noise-level sampling during training (the ablation in the project plan):
  both      k ~ U{0..K} independently per (time step, modality)        <- MDF (multimodal diffusion forcing)
  timestep  k per time step, shared by the modalities at that step     <- original Diffusion Forcing
  modality  k per modality, shared over the window
  sequence  one k for the whole window                                 <- ordinary full-sequence diffusion
"""
import math

import torch
import torch.nn as nn

NOISE_MODES = ("both", "timestep", "modality", "sequence")


def cosine_alpha_bar(K, s=0.008):
    t = torch.linspace(0, 1, K + 1)
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 1.0)                      # alpha_bar[0] = 1 (clean) .. alpha_bar[K] ~ 0


def level_embedding(k, K, dim):
    """Sinusoidal embedding of the noise level k in [0, K]."""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=k.device) / half)
    a = (k.float() / K * 1000.0)[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], -1)


class MDF(nn.Module):
    def __init__(self, dims, length, d_model=256, layers=6, heads=8, K=100, causal=True, dropout=0.1):
        super().__init__()
        self.mods = list(dims)
        self.dims, self.L, self.K, self.D, self.causal = dict(dims), length, K, d_model, causal
        self.inp = nn.ModuleDict({m: nn.Linear(d, d_model) for m, d in dims.items()})
        self.out = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d)) for m, d in dims.items()})
        self.mod_emb = nn.Parameter(torch.randn(len(dims), d_model) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(length, d_model) * 0.02)
        self.level_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        M = len(dims)
        t_of = torch.arange(length).repeat_interleave(M)        # token order: time-major, modalities within a step
        self.register_buffer("mask", (t_of[None, :] > t_of[:, None]) if causal else torch.zeros(length * M, length * M, dtype=torch.bool),
                             persistent=False)
        self.register_buffer("alpha_bar", cosine_alpha_bar(K), persistent=False)

    def forward(self, x, k):
        """x: {m: (B, L, d_m)} noisy tokens; k: (B, L, M) noise levels -> {m: (B, L, d_m)} predicted clean tokens."""
        B, L = k.shape[:2]
        h = torch.stack([self.inp[m](x[m]) for m in self.mods], 2)               # (B, L, M, D)
        h = h + self.mod_emb + self.time_emb[:L, None] + self.level_mlp(level_embedding(k, self.K, self.D))
        h = self.tf(h.reshape(B, L * len(self.mods), self.D), mask=self.mask[: L * len(self.mods), : L * len(self.mods)])
        h = h.reshape(B, L, len(self.mods), self.D)
        return {m: self.out[m](h[:, :, i]) for i, m in enumerate(self.mods)}

    # ---- diffusion
    def sample_levels(self, B, L, mode, device):
        M = len(self.mods)
        shape = {"both": (B, L, M), "timestep": (B, L, 1), "modality": (B, 1, M), "sequence": (B, 1, 1)}[mode]
        return torch.randint(0, self.K + 1, shape, device=device).expand(B, L, M)

    def noise(self, x0, k):
        ab = self.alpha_bar[k]                                                    # (B, L, M)
        eps = {m: torch.randn_like(x0[m]) for m in self.mods}
        xk = {m: ab[..., i, None].sqrt() * x0[m] + (1 - ab[..., i, None]).sqrt() * eps[m] for i, m in enumerate(self.mods)}
        return xk

    def loss(self, x0, mode, weights=None):
        B, L = x0[self.mods[0]].shape[:2]
        k = self.sample_levels(B, L, mode, x0[self.mods[0]].device)
        pred = self(self.noise(x0, k), k)
        per = {m: ((pred[m] - x0[m]) ** 2).mean() for m in self.mods}            # each modality counts equally
        w = weights or {}
        total = sum(w.get(m, 1.0) * per[m] for m in self.mods) / sum(w.get(m, 1.0) for m in self.mods)
        return total, per

    @torch.no_grad()
    def sample(self, known, known_level, steps=20):
        """Denoise every unknown token jointly from level K to 0 (DDIM, deterministic).
        known: {m: (B, L, d_m)} values (anything where unknown); known_level: (B, L, M) long, -1 = unknown token,
        0 = clean observation, k > 0 = an observation already noisy at level k (MDF's noise-aware conditioning;
        its value must be given as sqrt(alpha_bar_k) * x0 + sqrt(1 - alpha_bar_k) * noise)."""
        unknown = known_level < 0                                                # (B, L, M)
        x = {m: torch.where(unknown[..., i, None], torch.randn_like(known[m]), known[m]) for i, m in enumerate(self.mods)}
        levels = torch.linspace(self.K, 0, steps + 1, device=unknown.device).round().long()
        x0 = None
        for a, b in zip(levels[:-1], levels[1:]):
            k = torch.where(unknown, a, known_level.clamp(min=0))
            x0 = self(x, k)
            ab_a, ab_b = self.alpha_bar[a], self.alpha_bar[b]
            for i, m in enumerate(self.mods):
                eps = (x[m] - ab_a.sqrt() * x0[m]) / (1 - ab_a).clamp(min=1e-8).sqrt()
                nxt = ab_b.sqrt() * x0[m] + (1 - ab_b).sqrt() * eps
                x[m] = torch.where(unknown[..., i, None], nxt, x[m])
        return {m: torch.where(unknown[..., i, None], x0[m], known[m]) for i, m in enumerate(self.mods)}
