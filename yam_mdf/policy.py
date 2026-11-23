"""Use a trained MDF-lite checkpoint as a policy: observation history in, a chunk of actions out (raw units).

  from yam_mdf.policy import MDFPolicy
  pol = MDFPolicy("outputs/mdf/screw_from_rest_timestep/ckpt_best.pt")
  actions = pol.act(states, actions_past, images)      # (n_exec, 7): joint1-6 (rad) + gripper (0..1)

  states        (n_ctx, 21)  observation.state of the last n_ctx steps at 10 Hz, the last one = now
  actions_past  (n_ctx-1, 7) the actions sent at the steps before now
  images        {camera: (n_ctx, H, W, 3) uint8 RGB} for the cameras the model was trained on
  obs_noise     optional: standard deviation (standardised units) of noise known to be in the observations; the
                model is then told their noise level (MDF's noise-aware conditioning)

A closed loop encodes each new frame once with encode() and keeps the features: act_features() takes the history of
features instead of images (yam_mdf.run does this on the arm; yam_mdf.demo_plot on recorded episodes).
"""
import numpy as np
import torch

from yam_mdf.cache import Encoder
from yam_mdf.data import MODALITIES, Normalizer
from yam_mdf.model import MDF


class MDFPolicy:
    def __init__(self, ckpt, device="cuda", sample_steps=None, schedule=None, uncertainty=None, n_exec=None):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = c = ck["cfg"]
        self.norm = Normalizer.from_state_dict(ck["norm"])
        self.model = MDF(c["dims"], c["L"], c["d_model"], c["layers"], c["heads"], c["K"], causal=not c["no_causal"]).to(device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.enc = None
        self.cams = c["meta"]["cameras"]
        self.state_names, self.action_names = c["meta"]["state_names"], c["meta"]["action_names"]
        self.hz = float(c["meta"]["hz"])
        self.device, self.steps = device, sample_steps or c.get("sample_steps", 10)
        self.schedule = schedule or c.get("sample_schedule", "full")     # full | pyramid (see MDF.sample)
        self.uncertainty = uncertainty if uncertainty is not None else c.get("uncertainty", 1.0)
        self.n_ctx, self.L = c["n_ctx"], c["L"]
        self.n_exec = n_exec or c["n_exec"]
        self.step = ck.get("step")

    @torch.no_grad()
    def encode(self, images):
        """{camera: (H, W, 3) uint8 RGB} for one time step -> (C, 512) features, cameras in training order."""
        if self.enc is None:
            self.enc = Encoder().to(self.device)
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(images[c])) for c in self.cams]).to(self.device)
        return self.enc(x.permute(0, 3, 1, 2).float() / 255).float().cpu().numpy()

    @torch.no_grad()
    def act(self, states, actions_past, images, obs_noise=0.0):
        vis = np.stack([self.encode({c: images[c][i] for c in self.cams}) for i in range(self.n_ctx)])
        return self.act_features(states, actions_past, vis, obs_noise)

    @torch.no_grad()
    def act_features(self, states, actions_past, vis, obs_noise=0.0, full=False, trace=None):
        """vis: (n_ctx, C, 512) image features of the same steps as states. Returns the n_exec actions from now, or with
        full=True the whole predicted future in raw units: {m: (L - n_ctx + 1, d)} from now on (action) / (L - n_ctx, d)
        after now (observations). trace: see MDF.sample."""
        n, d = self.n_ctx, self.device
        states, actions_past = np.asarray(states, np.float32), np.asarray(actions_past, np.float32)
        assert states.shape == (n, 21) and actions_past.shape == (n - 1, 7), (states.shape, actions_past.shape)
        ep = dict(state=states, action=np.concatenate([actions_past, actions_past[-1:]]), vis=np.asarray(vis, np.float32))
        f = self.norm.transform(ep)
        M = len(self.model.mods)
        known = {m: torch.zeros(1, self.L, self.norm.dims()[m], device=d) for m in MODALITIES}
        lvl = torch.full((1, self.L, M), -1, dtype=torch.long, device=d)
        k = 0
        if obs_noise > 0:
            ab = self.model.alpha_bar
            k = int(torch.argmin(((1 - ab) / ab).sqrt().sub(obs_noise).abs()))
        for i, m in enumerate(self.model.mods):
            x = torch.from_numpy(f[m]).to(d)
            if m == "action":
                known[m][0, : n - 1] = x[: n - 1]; lvl[0, : n - 1, i] = 0
            else:
                known[m][0, :n] = x * (self.model.alpha_bar[k].sqrt() if k else 1.0); lvl[0, :n, i] = k
        out = self.model.sample(known, lvl, steps=self.steps, schedule=self.schedule, uncertainty=self.uncertainty, trace=trace)
        if not full:
            return self.norm.unnorm("action", out["action"][0, n - 1 : n - 1 + self.n_exec].float().cpu().numpy())
        res = {"action": self.norm.unnorm("action", out["action"][0, n - 1:].float().cpu().numpy())}
        for m in ("proprio", "force"):
            res[m] = self.norm.unnorm(m, out[m][0, n:].float().cpu().numpy())
        return res
