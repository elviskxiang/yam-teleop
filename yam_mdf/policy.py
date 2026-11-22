"""Use a trained MDF-lite checkpoint as a policy: observation history in, a chunk of actions out (raw units).

  from yam_mdf.policy import MDFPolicy
  pol = MDFPolicy("outputs/mdf/screw_from_rest_both/ckpt_best.pt")
  actions = pol.act(states, actions_past, images)      # (n_exec, 7): joint1-6 (rad) + gripper (0..1)

  states        (n_ctx, 21)  observation.state of the last n_ctx steps at 10 Hz, the last one = now
  actions_past  (n_ctx-1, 7) the actions sent at the steps before now
  images        {camera: (n_ctx, H, W, 3) uint8 RGB} for the cameras the model was trained on
  obs_noise     optional: standard deviation (standardised units) of noise known to be in the observations; the
                model is then told their noise level (MDF's noise-aware conditioning)

Closed-loop use on the arm still needs a runner that steps at 10 Hz, keeps the history, and sends the actions through
the yam_follower safety layer (lerobot_robot_yam): see README "Next".
"""
import numpy as np
import torch

from yam_mdf.cache import Encoder
from yam_mdf.data import MODALITIES, Normalizer
from yam_mdf.model import MDF


class MDFPolicy:
    def __init__(self, ckpt, device="cuda", sample_steps=10):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = c = ck["cfg"]
        self.norm = Normalizer.from_state_dict(ck["norm"])
        self.model = MDF(c["dims"], c["L"], c["d_model"], c["layers"], c["heads"], c["K"], causal=not c["no_causal"]).to(device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.enc = Encoder().to(device)
        self.cams = c["meta"]["cameras"]
        self.device, self.steps = device, sample_steps
        self.n_ctx, self.n_exec, self.L = c["n_ctx"], c["n_exec"], c["L"]

    @torch.no_grad()
    def act(self, states, actions_past, images, obs_noise=0.0):
        n, d = self.n_ctx, self.device
        states, actions_past = np.asarray(states, np.float32), np.asarray(actions_past, np.float32)
        assert states.shape == (n, 21) and actions_past.shape == (n - 1, 7)
        vis = torch.stack([self.enc(torch.from_numpy(np.asarray(images[c])).to(d).permute(0, 3, 1, 2).float() / 255) for c in self.cams], 1)
        ep = dict(state=states, action=np.concatenate([actions_past, actions_past[-1:]]), vis=vis.cpu().numpy())
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
        out = self.model.sample(known, lvl, steps=self.steps)
        a = out["action"][0, n - 1 : n - 1 + self.n_exec].float().cpu().numpy()
        return self.norm.unnorm("action", a)
