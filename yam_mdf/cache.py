"""Turn a LeRobot dataset (recorded with lerobot-record / yam_follower) into a compact per-episode feature cache.

  source env.sh && PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.cache --dataset screw_from_rest [--hz 10]

Per episode, every (30 / hz)-th frame: observation.state (21: pos, vel, effort for joint1-6 + gripper), action (7), and
for each camera a 512-d feature from a frozen ImageNet ResNet18 (images resized to 240x320). Written to
data/mdf_cache/<dataset>/ep_NNNN.npz; an episode already cached with the same length and settings is skipped, so after
recording more episodes only the new ones are encoded.

Safe to run while lerobot-record is still adding to the dataset: LeRobot keeps the newest data parquet open (its footer is
written only when the file is closed) and buffers the newest episodes' metadata, so the dataset is read through a view
(data/mdf_cache/<dataset>/_view, symlinks) holding only the finished files, and only the episodes fully in them are cached.
The rest are picked up by the next run, after the session ends.
"""
import argparse
import json
import os
import shutil
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


def _readable_parquet(f):
    import pyarrow.parquet as pq
    try:
        pq.read_metadata(f)
        return True
    except Exception:
        return False


def finished_view(root, view):
    """A read-only view of a LeRobot dataset with only its finished files (see the module doc): symlinks under `view`.
    Returns (view, episodes fully on disk, episodes left out)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    if view.exists():
        shutil.rmtree(view)
    (view / "meta").mkdir(parents=True)
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"{root} is not a LeRobot dataset (no meta/info.json)")
    files = []
    for top in ("meta", "data", "videos"):
        for d, _, names in os.walk(root / top, followlinks=True):
            files += [Path(d) / n for n in names]
    for f in sorted(files):
        rel = f.relative_to(root)
        if f.suffix == ".parquet" and not _readable_parquet(f):
            continue                                        # still being written
        (view / rel).parent.mkdir(parents=True, exist_ok=True)
        (view / rel).symlink_to(f.resolve())
    meta = LeRobotDatasetMetadata("local/view", root=view)
    listed = [int(i) for i in meta.episodes["episode_index"]]
    ok, left = [], [i for i in range(meta.total_episodes) if i not in set(listed)]
    for e in listed:
        files = [meta.get_data_file_path(e)] + [meta.get_video_file_path(e, k) for k in meta.video_keys]
        (ok if all((view / p).exists() for p in files) else left).append(e)
    return view, ok, sorted(left)


def build(dataset, hz=10, batch=32, workers=8, device="cuda"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = ROOT / "data" / "lerobot" / dataset
    out = ROOT / "data" / "mdf_cache" / dataset
    out.mkdir(parents=True, exist_ok=True)
    view, ready, left = finished_view(root, out / "_view")
    if left:
        print(f"episodes {left} are still being recorded or written (their files are open): not cached this time", flush=True)
    if not ready:
        raise SystemExit(f"no finished episodes in {root} yet")
    ds = LeRobotDataset(f"local/{dataset}", root=view, episodes=ready)
    stride = int(round(ds.fps / hz))
    cams = sorted(k for k in ds.meta.features if k.startswith("observation.images."))
    meta = dict(dataset=dataset, fps=ds.fps, hz=ds.fps / stride, stride=stride, encoder=ENCODER, cameras=[c.split(".")[-1] for c in cams],
                state_names=ds.meta.features["observation.state"]["names"], action_names=ds.meta.features["action"]["names"])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    enc = None
    eps = {int(r["episode_index"]): r for r in ds.meta.episodes}
    for e in ready:
        rec = eps[e]
        lo, hi, n = int(rec["dataset_from_index"]), int(rec["dataset_to_index"]), int(rec["length"])
        f = out / f"ep_{e:04d}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            if int(z["length"]) == n and int(z["stride"]) == stride and str(z["encoder"]) == ENCODER and list(z["cameras"]) == meta["cameras"]:
                continue
        idx = [ds.reader._absolute_to_relative_idx[i] for i in range(lo, hi, stride)]   # rows of the episode subset
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
    print(json.dumps(dict(cache=str(out), episodes=len(ready), not_yet=left, **{k: meta[k] for k in ("hz", "cameras")})))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LeRobot dataset -> MDF-lite feature cache")
    ap.add_argument("--dataset", default="screw_from_rest")
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(a.dataset, a.hz, workers=a.workers)
