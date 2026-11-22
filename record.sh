#!/usr/bin/env bash
# Record demonstrations with lerobot-record: the left YAM leads (move it by hand), the right YAM follows.
#   ./record.sh <dataset-name> [num-episodes=10] [extra lerobot-record args...]
# e.g. ./record.sh screw_from_rest 10
# Appends to data/lerobot/<dataset-name> if it exists (--resume), else creates it. Task label = the dataset name
# (override with TASK="..."). Keys: right arrow = end episode early, left arrow = discard and re-record it, Esc = stop;
# gripper V open, B cube width, C squeeze, Z close. A live view of the cameras and joints opens in rerun.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
name="${1:?dataset name, e.g. screw_from_rest}"; n="${2:-10}"; shift; [ $# -gt 0 ] && shift
root="data/lerobot/$name"; resume=false; [ -d "$root" ] && resume=true
busy=$(pgrep -af "cam_wall.py|yam_teleop.leader_follower|teleop/se3_jog.py|yam_teleop.record|rig_logger.py" | grep -v pgrep || true)
if [ -n "$busy" ]; then
  echo "These hold the arms or cameras; stop them first (park the arms from their page):"; echo "$busy"; exit 1
fi
echo "recording $n episode(s) of '${TASK:-$name}' into $root (resume=$resume)"
exec env PYTHONPATH="$LR_PYTHONPATH" "$(dirname "$LR_PY")/lerobot-record" \
  --robot.type=yam_follower --robot.id=right \
  --teleop.type=yam_leader --teleop.id=left \
  --dataset.repo_id="local/$name" --dataset.root="$root" --dataset.single_task="${TASK:-$name}" \
  --dataset.fps=30 --dataset.episode_time_s=90 --dataset.reset_time_s=30 --dataset.num_episodes="$n" \
  --dataset.push_to_hub=false --display_data=true --resume="$resume" \
  "$@"
