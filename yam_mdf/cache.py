"""Turn a LeRobot dataset (recorded with lerobot-record / yam_follower) into a compact per-episode feature cache.

  source env.sh && PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.cache --dataset screw_from_rest [--hz 10]

Per episode, every (30 / hz)-th frame: observation.state (21: pos, vel, effort for joint1-6 + gripper), action (7), and
for each camera a 512-d feature from a frozen ImageNet ResNet18 (images resized to 240x320). Written to
data/mdf_cache/<dataset>/ep_NNNN.npz; an episode already cached with the same length and settings is skipped, so after
recording more episodes only the new ones are encoded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
ENCODER = "resnet18_imagenet_240x320_avgpool"


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        self.m = m.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):                          # (B, 3, H, W) in [0, 1]
        x = F.interpolate(x, size=(240, 320), mode="bilinear", align_corners=False, antialias=True)
        return self.m((x - self.mean) / self.std)


def build(dataset, hz=10, batch=32, workers=8, device="cuda"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = ROOT / "data" / "lerobot" / dataset
    ds = LeRobotDataset(f"local/{dataset}", root=root)
    stride = int(round(ds.fps / hz))
    cams = sorted(k for k in ds.meta.features if k.startswith("observation.images."))
    out = ROOT / "data" / "mdf_cache" / dataset
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=dataset, fps=ds.fps, hz=ds.fps / stride, stride=stride, encoder=ENCODER, cameras=[c.split(".")[-1] for c in cams],
                state_names=ds.meta.features["observation.state"]["names"], action_names=ds.meta.features["action"]["names"])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    enc = None
    eps = ds.meta.episodes
    for e in range(ds.meta.total_episodes):
        rec = eps[e]
        lo, hi, n = int(rec["dataset_from_index"]), int(rec["dataset_to_index"]), int(rec["length"])
        f = out / f"ep_{e:04d}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            if int(z["length"]) == n and int(z["stride"]) == stride and str(z["encoder"]) == ENCODER and list(z["cameras"]) == meta["cameras"]:
                continue
        idx = list(range(lo, hi, stride))
        if enc is None:
            enc = Encoder().to(device)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch, num_workers=workers, shuffle=False)
        S, A, V, FI = [], [], [], []
        for b in loader:
            S.append(b["observation.state"].numpy()); A.append(b["action"].numpy()); FI.append(b["frame_index"].numpy())
            V.append(torch.stack([enc(b[c].to(device, non_blocking=True)) for c in cams], 1).float().cpu().numpy())
        np.savez(f, state=np.concatenate(S).astype(np.float32), action=np.concatenate(A).astype(np.float32),
                 vis=np.concatenate(V).astype(np.float16), frame_index=np.concatenate(FI), length=n, stride=stride,
                 encoder=ENCODER, cameras=np.array(meta["cameras"]), task=str(rec["tasks"][0]) if len(rec["tasks"]) else "")
        print(f"episode {e}: {len(idx)} steps at {meta['hz']:.0f} Hz cached", flush=True)
    print(json.dumps(dict(cache=str(out), episodes=ds.meta.total_episodes, **{k: meta[k] for k in ("hz", "cameras")})))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LeRobot dataset -> MDF-lite feature cache")
    ap.add_argument("--dataset", default="screw_from_rest")
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(a.dataset, a.hz, workers=a.workers)
