"""Modalities, normalisation and training windows for MDF-lite.

Modalities per 10 Hz time step (from the yam_follower dataset, see README "Dataset features"):
  action  (7)   leader joint1-6 + gripper target
  proprio (14)  follower joint1-6 + gripper position and velocity
  force   (7)   follower effort: torque beyond the URDF gravity model (joint1-6) + gripper motor effort
  vision  (C*P) per camera: ResNet18 feature (512) -> PCA to P dims, fitted on the training episodes
Every modality is standardised (mean / std over the training episodes) so the diffusion noise means the same in each.
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODALITIES = ("action", "proprio", "force", "vision")


def load_cache(dataset):
    d = ROOT / "data" / "mdf_cache" / dataset
    meta = json.loads((d / "meta.json").read_text())
    eps = []
    for f in sorted(d.glob("ep_*.npz")):
        z = np.load(f)
        eps.append(dict(index=int(f.stem.split("_")[1]), state=z["state"], action=z["action"], vis=z["vis"].astype(np.float32),
                        task=str(z["task"])))
    if not eps:
        raise SystemExit(f"no cached episodes in {d}: run python -m yam_mdf.cache --dataset {dataset}")
    return meta, eps


def split(eps, holdout=2, holdout_episodes=""):
    """Hold out the last `holdout` episodes (by index), or the explicit comma-separated episode indices."""
    if holdout_episodes:
        ids = {int(x) for x in holdout_episodes.split(",")}
        return [e for e in eps if e["index"] not in ids], [e for e in eps if e["index"] in ids]
    n = int(holdout)
    eps = sorted(eps, key=lambda e: e["index"])
    return (eps[:-n], eps[-n:]) if n > 0 else (eps, [])


class Normalizer:
    """Per-camera PCA of the image features + per-dimension standardisation of every modality."""

    def __init__(self, pca_dim=32):
        self.pca_dim = pca_dim

    def raw(self, e):
        return dict(action=e["action"], proprio=e["state"][:, :14], force=e["state"][:, 14:21], vision=e["vis"])

    def fit(self, eps):
        vis = np.concatenate([e["vis"] for e in eps])               # (N, C, 512)
        self.pca_mean = vis.mean(0)                                 # (C, 512)
        self.pca_comp = np.stack([np.linalg.svd(vis[:, c] - self.pca_mean[c], full_matrices=False)[2][: self.pca_dim]
                                  for c in range(vis.shape[1])])     # (C, P, 512)
        feats = [self._features(e) for e in eps]
        self.mean = {m: np.concatenate([f[m] for f in feats]).mean(0) for m in MODALITIES}
        self.std = {m: np.maximum(np.concatenate([f[m] for f in feats]).std(0), 1e-3) for m in MODALITIES}
        return self

    def _features(self, e):
        r = self.raw(e)
        v = np.einsum("ncd,cpd->ncp", r["vision"] - self.pca_mean, self.pca_comp).reshape(len(r["vision"]), -1)
        return dict(action=r["action"], proprio=r["proprio"], force=r["force"], vision=v)

    def transform(self, e):
        f = self._features(e)
        return {m: ((f[m] - self.mean[m]) / self.std[m]).astype(np.float32) for m in MODALITIES}

    def unnorm(self, m, x):
        return x * self.std[m] + self.mean[m]

    def dims(self):
        return {m: int(self.mean[m].shape[0]) for m in MODALITIES}

    def state_dict(self):
        return dict(pca_dim=self.pca_dim, pca_mean=self.pca_mean, pca_comp=self.pca_comp, mean=self.mean, std=self.std)

    @classmethod
    def from_state_dict(cls, s):
        n = cls(s["pca_dim"])
        for k in ("pca_mean", "pca_comp", "mean", "std"):
            setattr(n, k, s[k])
        return n


class Windows(torch.utils.data.Dataset):
    """Random windows of `length` consecutive steps from the normalised episodes (episodes weighted by length)."""

    def __init__(self, seqs, length, samples_per_epoch=100_000):
        self.seqs = [s for s in seqs if len(s["action"]) >= length]
        if not self.seqs:
            raise SystemExit(f"no episode has {length} steps")
        self.L, self.n = length, samples_per_epoch
        w = np.array([len(s["action"]) - length + 1 for s in self.seqs], float)
        self.p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        s = self.seqs[rng.choice(len(self.seqs), p=self.p)]
        t = rng.integers(0, len(s["action"]) - self.L + 1)
        return {m: torch.from_numpy(s[m][t:t + self.L]) for m in MODALITIES}
