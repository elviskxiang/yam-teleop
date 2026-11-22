# Source from the repo root:  source env.sh
# yam-teleop runs on the rig PC next to a checkout of robotics-rubiks-cube, which provides the I2RT SDK (vendor/i2rt), its
# hardware deps (vendor/hwdeps: mcap, python-can), the rig's gripper force cap (scripts/hw/yam_io.py: LeashGripper) and the
# rig scene used for collision checks (poc/generated/scene.xml).
export RUBIKS="${RUBIKS:-$HOME/Desktop/experiments/rubiks_cube/robotics-rubiks-cube}"
export YT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$YT_ROOT:$RUBIKS:$RUBIKS/scripts/hw:$RUBIKS/vendor/i2rt:$RUBIKS/vendor/hwdeps"
export HW_PY="$RUBIKS/.venv-warp/bin/python"   # mujoco + numpy + scipy; the SDK comes from PYTHONPATH
export REC_PY="$RUBIKS/.venv-hw/bin/python"    # mcap writer
export CAM_PY="$RUBIKS/../.venv-ir/bin/python" # OpenCV with V4L2 for the cameras
export LEROBOT_ENV="${LEROBOT_ENV:-lerobot-libero}"   # conda env with lerobot 0.5.x (training)
[ -d "$RUBIKS/vendor/i2rt" ] || echo "env.sh: RUBIKS=$RUBIKS has no vendor/i2rt; set RUBIKS to the robotics-rubiks-cube checkout" >&2
