#!/usr/bin/env bash
# Run a trained MDF-lite policy on the follower (right) arm, closed loop (yam_mdf/run.py).
#   ./run_mdf.sh [checkpoint=outputs/mdf/screw_from_rest_timestep/ckpt_best.pt] [--seconds 60] [--max-vel 1.0] [--exec 8]
# Before: left arm at rest, scene as in the recordings, hand ready. Stop: Ctrl+C or Esc (the arm parks, torque off).
# Arm + camera page: http://127.0.0.1:8781
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
ckpt="${1:-outputs/mdf/screw_from_rest_timestep/ckpt_best.pt}"; [ $# -gt 0 ] && shift
[ -f "$ckpt" ] || { echo "no checkpoint at $ckpt (train one: ./train_mdf.sh screw_from_rest timestep --sample-schedule pyramid)"; exit 1; }
busy=$(pgrep -af "yam_teleop.control|record_session|lerobot-record|cam_wall.py|yam_teleop.leader_follower|yam_mdf.run " | grep -v pgrep || true)
if [ -n "$busy" ]; then
  echo "These hold the arms or cameras; stop them first (the control page's camera preview holds the cameras too):"; echo "$busy"; exit 1
fi
exec env PYTHONPATH="$LR_PYTHONPATH" HF_HUB_OFFLINE=1 "$LR_PY" -m yam_mdf.run "$ckpt" "$@"
