"""Train MDF-lite on a recorded dataset and evaluate it on held-out episodes.

  source env.sh && PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.train --dataset screw_from_rest --noise-mode both
  (./train_mdf.sh does the cache + training; --noise-mode timestep|modality|sequence for the ablation)

Window: n_ctx steps of history (observations up to and including "now", actions before now) + n_pred future steps, at
10 Hz. As a policy the model gets the history clean and denoises the action now and the n_pred future steps of every
modality together (it also predicts future proprio / force / vision: a dynamics model).

Held-out evaluation (EMA weights, every --eval-every steps; outputs/mdf/<run>/log.jsonl):
  act_rmse        RMSE (rad) of joints 1-6 over the next n_exec actions, history clean, sampled with --sample-schedule
  act_rmse_<other> the same with the other schedule (full: all future tokens denoised together; pyramid: Diffusion
                  Forcing's schedule, each further step lagging --uncertainty sampling steps, so the near future is clean first)
  hold_rmse       the same for "repeat the last action": a model worth having beats it
  grip_mae        gripper target error (0..1)
  noisy_s<σ>_unaware / _aware  act_rmse with Gaussian noise σ (in standardised units) added to the observed history,
                  given as clean (unaware) or with its matching noise level (aware: MDF's noise-aware conditioning)
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from yam_mdf.data import MODALITIES, Normalizer, Windows, load_cache, split
from yam_mdf.model import MDF, NOISE_MODES

ROOT = Path(__file__).resolve().parent.parent
OBS = ("proprio", "force", "vision")


def eval_windows(seq, L, stride):
    T = len(seq["action"])
    return [t for t in range(0, T - L + 1, stride)]


@torch.no_grad()
def evaluate(model, val, norm, n_ctx, n_exec, L, device, steps=10, sigmas=(0.25, 0.5, 1.0), stride=5, max_windows=400,
             schedule="full", uncertainty=1.0):
    model.eval()
    M = len(model.mods); c = n_ctx - 1
    batch = {m: [] for m in MODALITIES}
    for s in val:
        for t in eval_windows(s, L, stride):
            for m in MODALITIES:
                batch[m].append(s[m][t:t + L])
    if not batch["action"]:
        return {}
    idx = np.random.default_rng(0).permutation(len(batch["action"]))[:max_windows]
    x0 = {m: torch.from_numpy(np.stack(batch[m])[idx]).to(device) for m in MODALITIES}
    B = x0["action"].shape[0]
    lvl = torch.full((B, L, M), -1, dtype=torch.long, device=device)
    for i, m in enumerate(model.mods):
        if m == "action":
            lvl[:, :c, i] = 0                                         # past actions known; now + future unknown
        else:
            lvl[:, :c + 1, i] = 0                                     # observations up to now known
    gt = norm.unnorm("action", x0["action"][:, c:c + n_exec].cpu().numpy())

    def act_err(known, level, sched=schedule):
        out = model.sample(known, level, steps=steps, schedule=sched, uncertainty=uncertainty)
        pred = norm.unnorm("action", out["action"][:, c:c + n_exec].float().cpu().numpy())
        return float(np.sqrt(((pred[..., :6] - gt[..., :6]) ** 2).mean())), float(np.abs(pred[..., 6] - gt[..., 6]).mean())

    res = {}
    res["act_rmse"], res["grip_mae"] = act_err(x0, lvl)
    other = "pyramid" if schedule == "full" else "full"
    res[f"act_rmse_{other}"] = act_err(x0, lvl, other)[0]                  # the other sampling schedule, for comparison
    hold = norm.unnorm("action", x0["action"][:, c - 1:c].cpu().numpy()).repeat(n_exec, 1)
    res["hold_rmse"] = float(np.sqrt(((hold[..., :6] - gt[..., :6]) ** 2).mean()))
    ab = model.alpha_bar
    g = torch.Generator(device=device).manual_seed(0)
    for sg in sigmas:
        noisy = {m: (x0[m] + sg * torch.randn(x0[m].shape, generator=g, device=device)) if m in OBS else x0[m] for m in MODALITIES}
        res[f"noisy_s{sg}_unaware"] = act_err(noisy, lvl)[0]
        k = int(torch.argmin(((1 - ab) / ab).sqrt().sub(sg).abs()))   # level whose noise matches sigma
        aware_lvl = lvl.clone()
        known = dict(noisy)
        for i, m in enumerate(model.mods):
            if m in OBS:
                aware_lvl[:, :c + 1, i] = k
                known[m] = ab[k].sqrt() * noisy[m]                     # x_k = sqrt(ab) (x0 + sigma eps)
        res[f"noisy_s{sg}_aware"] = act_err(known, aware_lvl)[0]
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser(description="train MDF-lite")
    ap.add_argument("--dataset", default="screw_from_rest")
    ap.add_argument("--noise-mode", choices=NOISE_MODES, default="both")
    ap.add_argument("--run", default=None, help="output name (default: <dataset>_<noise-mode>)")
    ap.add_argument("--holdout", type=int, default=2, help="hold out the last N episodes for evaluation")
    ap.add_argument("--holdout-episodes", default="", help="or: comma-separated episode indices to hold out")
    ap.add_argument("--n-ctx", type=int, default=8, help="history steps incl. now (10 Hz)")
    ap.add_argument("--n-pred", type=int, default=16, help="future steps predicted")
    ap.add_argument("--n-exec", type=int, default=8, help="actions evaluated / executed per prediction")
    ap.add_argument("--pca", type=int, default=32, help="PCA dims per camera")
    ap.add_argument("--d-model", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--K", type=int, default=100, help="diffusion levels")
    # 9 episodes: held-out error was best at ~4k steps and rose after (overfitting); the best checkpoint is kept anyway
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--sample-schedule", choices=("full", "pyramid"), default="full",
                    help="held-out evaluation: denoise the future all together, or as a Diffusion Forcing pyramid (near first)")
    ap.add_argument("--uncertainty", type=float, default=1.0, help="pyramid: sampling steps each further time step lags")
    ap.add_argument("--no-causal", action="store_true", help="bidirectional attention over the window")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda"
    run = a.run or f"{a.dataset}_{a.noise_mode}"
    out = ROOT / "outputs" / "mdf" / run
    out.mkdir(parents=True, exist_ok=True)

    meta, eps = load_cache(a.dataset)
    tr, va = split(eps, a.holdout, a.holdout_episodes)
    norm = Normalizer(a.pca).fit(tr)
    tr_s, va_s = [norm.transform(e) for e in tr], [norm.transform(e) for e in va]
    L = a.n_ctx + a.n_pred
    dims = norm.dims()
    cfg = dict(vars(a), run=run, L=L, dims=dims, train_episodes=[e["index"] for e in tr], val_episodes=[e["index"] for e in va],
               train_steps_total=int(sum(len(e["action"]) for e in tr)), meta=meta)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[{run}] train episodes {cfg['train_episodes']} ({cfg['train_steps_total']} steps at {meta['hz']:.0f} Hz), "
          f"held out {cfg['val_episodes']}; modalities {dims}; window {L}", flush=True)

    model = MDF(dims, L, a.d_model, a.layers, a.heads, a.K, causal=not a.no_causal).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    loader = torch.utils.data.DataLoader(Windows(tr_s, L, a.steps * a.batch), batch_size=a.batch, num_workers=4, persistent_workers=True)
    print(f"[{run}] model {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params", flush=True)

    logf = open(out / "log.jsonl", "a")
    best, t0, acc = float("inf"), time.time(), {}
    for step, batch in enumerate(loader, 1):
        x0 = {m: v.to(device, non_blocking=True) for m, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, per = model.loss(x0, a.noise_mode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        for m, v in per.items():
            acc[m] = acc.get(m, 0.0) + float(v.detach())
        acc["loss"] = acc.get("loss", 0.0) + float(loss.detach())
        if step % a.log_every == 0:
            rec = dict(step=step, lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1), **{k: v / a.log_every for k, v in acc.items()})
            acc = {}
            if step % a.eval_every == 0 or step == a.steps:
                if va_s:
                    rec.update(evaluate(ema, va_s, norm, a.n_ctx, a.n_exec, L, device, a.sample_steps,
                                        schedule=a.sample_schedule, uncertainty=a.uncertainty))
                ck = dict(model=ema.state_dict(), cfg=cfg, norm=norm.state_dict(), step=step)
                torch.save(ck, out / "ckpt_last.pt")
                if va_s and rec["act_rmse"] < best:
                    best = rec["act_rmse"]; torch.save(ck, out / "ckpt_best.pt"); rec["best"] = True
                print(f"[{run}] step {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in rec.items() if isinstance(v, float) and k != "lr"), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step >= a.steps:
            break
    print(json.dumps(dict(run=run, best_act_rmse=best if va_s else None, out=str(out))), flush=True)


if __name__ == "__main__":
    main()
