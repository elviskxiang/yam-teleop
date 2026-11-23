"""Demo figures for an MDF-lite run on a held-out episode (the backup for the live demo): what the policy predicts vs what
the operator did, and how the Diffusion Forcing pyramid denoises the future.

  PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.demo_plot outputs/mdf/screw_from_rest_timestep/ckpt_best.pt [--episode N]

Writes to outputs/mdf/<run>/demo/:
  rollout_ep<N>.png      every n_exec steps (0.8 s) the policy sees the recorded history and predicts the future actions;
                         per joint: the recorded action (black), each prediction (colour; the executed part bold)
  pyramid_ep<N>_t<T>.png one prediction: (left) the noise level of every future action step at each denoising pass,
                         (right) the model's estimate of the most-moving joint after successive passes vs the truth
The episode defaults to the run's first held-out episode (never seen in training).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from yam_mdf.data import load_cache
from yam_mdf.policy import MDFPolicy

JOINTS = ["joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6", "gripper"]


def predict(pol, e, t, trace=None):
    """Prediction at step t (= now) of episode e from its recorded history."""
    n = pol.n_ctx
    return pol.act_features(e["state"][t - n + 1:t + 1], e["action"][t - n + 1:t], e["vis"][t - n + 1:t + 1], full=True, trace=trace)


def rollout_figure(pol, e, path):
    n, T = pol.n_ctx, len(e["action"])
    ts = list(range(n - 1, T - 1, pol.n_exec))
    preds = [predict(pol, e, t)["action"] for t in ts]
    hz = pol.hz
    err = []
    for t, p in zip(ts, preds):
        h = min(pol.n_exec, T - t)
        err.append(p[:h, :6] - e["action"][t:t + h, :6])
    rmse = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
    fig, axs = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    axs = axs.ravel()
    cmap = plt.get_cmap("viridis")
    time = np.arange(T) / hz
    for j in range(7):
        ax = axs[j]
        ax.plot(time, e["action"][:, j], color="black", lw=1.6, label="recorded (operator)")
        for i, (t, p) in enumerate(zip(ts, preds)):
            c = cmap(i / max(1, len(ts) - 1))
            tt = (t + np.arange(len(p))) / hz
            ax.plot(tt, p[:, j], color=c, lw=0.8, alpha=0.45)
            ax.plot(tt[:pol.n_exec], p[:pol.n_exec, j], color=c, lw=2.0)
        ax.set_title(JOINTS[j], fontsize=10, loc="left")
        ax.set_ylabel("rad" if j < 6 else "0 closed .. 1 open", fontsize=8)
        ax.grid(alpha=0.25)
    axs[0].legend(loc="best", fontsize=8)
    ax = axs[7]
    ax.axis("off")
    ax.text(0.0, 0.95, f"episode {e['index']} (held out: never seen in training)\n"
                       f"model: {pol.cfg['noise_mode']} noise, {pol.schedule} schedule, step {pol.step}\n"
                       f"a prediction every {pol.n_exec / hz:.1f} s from the last {pol.n_ctx / hz:.1f} s of history\n"
                       f"bold = the {pol.n_exec} actions executed, thin = the rest of the predicted future\n"
                       f"RMSE of the executed actions (joints 1-6): {rmse:.3f} rad",
            va="top", fontsize=10, family="monospace")
    for ax in axs[5:7]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Diffusion Forcing policy vs the recorded demonstration", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return rmse


def pyramid_figure(pol, e, t, path):
    trace = []
    pred = predict(pol, e, t, trace)["action"]
    n = pol.n_ctx
    ai = pol.model.mods.index("action")
    lv = np.stack([lvl[0, n - 1:, ai].cpu().numpy() for lvl, _ in trace])            # (passes, future steps)
    est = np.stack([pol.norm.unnorm("action", x0["action"][0, n - 1:].float().cpu().numpy()) for _, x0 in trace])
    truth = e["action"][t:t + lv.shape[1]]
    j = int(np.argmax(np.ptp(truth[:, :6], 0)))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
    im = a0.imshow(lv, aspect="auto", cmap="magma", origin="upper", vmin=0, vmax=pol.model.K,
                   extent=[-0.5, lv.shape[1] - 0.5, lv.shape[0] + 0.5, 0.5])
    a0.set_xlabel("future action step (0 = now, 10 Hz)"); a0.set_ylabel("denoising pass")
    a0.set_title(f"noise level per step ({pol.schedule} schedule)", fontsize=10, loc="left")
    fig.colorbar(im, ax=a0, label="noise level (0 = clean)")
    hz = pol.hz
    tt = np.arange(lv.shape[1]) / hz
    picks = sorted(set(np.linspace(0, len(trace) - 1, 6).round().astype(int)))
    cmap = plt.get_cmap("plasma")
    for i, p in enumerate(picks):
        a1.plot(tt, est[p, :, j], color=cmap(i / max(1, len(picks) - 1)), lw=1.4, label=f"after pass {p + 1}/{len(trace)}")
    a1.plot(tt[:len(truth)], truth[:, j], color="black", lw=2.2, ls="--", label="recorded")
    a1.set_xlabel("time ahead (s)"); a1.set_ylabel("rad")
    a1.set_title(f"{JOINTS[j]}: the estimate settles near-first", fontsize=10, loc="left")
    a1.grid(alpha=0.25); a1.legend(fontsize=8, loc="best")
    fig.suptitle(f"Diffusion Forcing sampling, episode {e['index']}, t = {t / hz:.1f} s", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="demo figures for an MDF-lite run on a held-out episode")
    ap.add_argument("ckpt")
    ap.add_argument("--episode", type=int, default=None, help="default: the run's first held-out episode")
    ap.add_argument("--t", type=float, default=None, help="time (s) of the pyramid figure; default: where the arm moves most")
    ap.add_argument("--schedule", choices=("full", "pyramid"), default="pyramid")
    a = ap.parse_args()
    pol = MDFPolicy(a.ckpt, schedule=a.schedule)
    _, eps = load_cache(pol.cfg["dataset"])
    want = a.episode if a.episode is not None else (pol.cfg["val_episodes"] or [eps[-1]["index"]])[0]
    e = next((x for x in eps if x["index"] == want), None)
    if e is None:
        raise SystemExit(f"episode {want} is not in the cache")
    if want in pol.cfg["train_episodes"]:
        print(f"note: episode {want} was used in training")
    out = Path(a.ckpt).resolve().parent / "demo"
    out.mkdir(exist_ok=True)
    rmse = rollout_figure(pol, e, out / f"rollout_ep{want}.png")
    n, T = pol.n_ctx, len(e["action"])
    if a.t is not None:
        t = int(round(a.t * pol.hz))
    else:                                                     # where the recorded action moves most over the horizon
        h = pol.L - n
        speed = [np.abs(e["action"][t + h, :6] - e["action"][t, :6]).sum() for t in range(n - 1, T - h)]
        t = n - 1 + int(np.argmax(speed))
    t = int(np.clip(t, n - 1, T - 2))
    pyramid_figure(pol, e, t, out / f"pyramid_ep{want}_t{t}.png")
    print(f"episode {want}: executed-action RMSE {rmse:.3f} rad; figures in {out}")


if __name__ == "__main__":
    main()
