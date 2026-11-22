"""Compare MDF-lite runs: the best held-out evaluation of each (outputs/mdf/<dataset>_*/log.jsonl).

  PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.report screw_from_rest
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(ds="screw_from_rest"):
    rows = []
    for d in sorted((ROOT / "outputs" / "mdf").glob(f"{ds}_*")):
        recs = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()] if (d / "log.jsonl").exists() else []
        ev = [r for r in recs if "act_rmse" in r]
        if ev:
            b = min(ev, key=lambda r: r["act_rmse"]); rows.append((d.name, b))
    if not rows:
        print("no evaluated runs yet"); return
    keys = ["act_rmse", "act_rmse_full", "act_rmse_pyramid", "hold_rmse", "grip_mae"] + sorted(k for k in rows[0][1] if k.startswith("noisy_"))
    print(f"{'run':34s} {'step':>6s} " + " ".join(f"{k.replace('noisy_', ''):>14s}" for k in keys))
    for name, b in rows:
        print(f"{name:34s} {b['step']:6d} " + " ".join(f"{b.get(k, float('nan')):14.4f}" for k in keys))
    print("\nact_rmse: joints 1-6, rad, next n_exec actions on held-out episodes (lower is better); hold_rmse: repeat-last-action "
          "baseline; act_rmse_full/_pyramid: the other sampling schedule (blank = not evaluated); noisy_s<sigma>_aware/_unaware: observation history with noise, model told its level or not")


if __name__ == "__main__":
    main(*sys.argv[1:])
