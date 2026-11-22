#!/usr/bin/env bash
# Train a policy on a converted dataset (yam_teleop.to_lerobot) with LeRobot.
#   ./train.sh <dataset-name> [policy=act] [extra lerobot-train args...]
# e.g.  ./train.sh pick_cube             -> ACT on data/lerobot/pick_cube, outputs/train/act_pick_cube
#       ./train.sh pick_cube diffusion --steps=200000
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
name="${1:?dataset name (directory under data/lerobot/)}"; policy="${2:-act}"; shift; [ $# -gt 0 ] && shift
out="outputs/train/${policy}_${name}"
[ -e "$out" ] && { echo "$out exists: pick another name or remove it (or pass --resume=true --config_path=$out/checkpoints/last/pretrained_model/train_config.json)"; exit 1; }
exec conda run --no-capture-output -n "$LEROBOT_ENV" lerobot-train \
  --dataset.repo_id="local/$name" --dataset.root="data/lerobot/$name" \
  --policy.type="$policy" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$out" --job_name="${policy}_${name}" \
  --batch_size=8 --steps=100000 --save_freq=10000 --log_freq=100 --num_workers=8 --wandb.enable=false \
  "$@"
