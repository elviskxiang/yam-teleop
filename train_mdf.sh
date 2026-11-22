#!/usr/bin/env bash
# MDF-lite on a recorded dataset: update the feature cache (only new episodes are encoded), then train.
#   ./train_mdf.sh [dataset=screw_from_rest] [noise-mode=both|timestep|modality|sequence|ablation] [extra yam_mdf.train args]
# e.g. ./train_mdf.sh                                   MDF (independent noise per time step x modality), ~12 min
#      ./train_mdf.sh screw_from_rest ablation          all four noise modes in turn (the project plan's ablation)
#      ./train_mdf.sh screw_from_rest both --steps 60000 --holdout 3
# Results: outputs/mdf/<dataset>_<mode>/{log.jsonl, ckpt_best.pt, config.json}; ./train_mdf.sh ... then
#   $LR_PY -m yam_mdf.report <dataset>     compares the runs
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
ds="${1:-screw_from_rest}"; mode="${2:-both}"; shift || true; shift || true
export PYTHONPATH="$LR_PYTHONPATH"
busy=$(pgrep -af "record_session|lerobot-record" | grep -v pgrep || true)
[ -n "$busy" ] && echo "note: a recording session is running; training uses the GPU only, the session is unaffected"
"$LR_PY" -m yam_mdf.cache --dataset "$ds"
if [ "$mode" = ablation ]; then
  for m in both timestep modality sequence; do "$LR_PY" -m yam_mdf.train --dataset "$ds" --noise-mode "$m" "$@"; done
  "$LR_PY" -m yam_mdf.report "$ds"
else
  "$LR_PY" -m yam_mdf.train --dataset "$ds" --noise-mode "$mode" "$@"
fi
