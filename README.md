# yam-teleop

Teleoperate the **right** YAM arm by moving the **left** YAM arm by hand, record demonstrations with LeRobot's standard
`lerobot-record`, and train policies on them with `lerobot-train`.

```
left arm (leader, by hand) --lerobot_teleoperator_yam--> action --lerobot_robot_yam (100 Hz safety layer)--> right arm
                                    4 rig cameras ----------------> lerobot-record --> data/lerobot/<name>  (LeRobot v3 dataset)
                                                                                    --> train.sh (ACT by default)
```

The YAM arms are LeRobot plugins (`plugins/`): robot type `yam_follower`, teleoperator type `yam_leader`, so every
LeRobot tool works with them (`lerobot-record`, `lerobot-teleoperate`, `lerobot-replay`, and `lerobot-record
--policy.path=...` to run a trained policy on the right arm).

## Requirements

Runs on the rig PC, next to a checkout of `robotics-rubiks-cube` (private), which supplies the I2RT SDK, the gripper force
cap (`LeashGripper`) and the rig model used for collision checks. `env.sh` finds it at
`~/Desktop/experiments/rubiks_cube/robotics-rubiks-cube`; set `RUBIKS=...` before sourcing it if yours is elsewhere.

LeRobot env: conda `lerobot-libero` (LeRobot 0.5.x, CUDA; override with `LEROBOT_ENV`). One-time setup in it:

```bash
pip install "python-can==4.5.0" "crcmod-plus>=2.3.1" pynput tyro "dm-env==1.6" mcap "mcap-ros2-support>=0.5.5" "rerun-sdk>=0.24,<0.27"
pip install -e plugins/lerobot_robot_yam -e plugins/lerobot_teleoperator_yam
```

Both arms on CAN (`can_left`, `can_right`, 1 Mbit/s). **Plug the two CAN adapters straight into the PC**, not into the
hub with the cameras: on Nov 22 that hub dropped out twice in three minutes and took both arms down with it.
Only one program may use the arms and cameras at a time (`record.sh` checks).

## Recording demonstrations

```bash
./record.sh screw_from_rest 10       # 10 more episodes, appended to data/lerobot/screw_from_rest
```

- Both arms start **floating** (gravity compensation from the URDF, all six joints). Start with both at the rest pose:
  the right arm engages on the first frame with a slow (0.35 rad/s) ramp to the left arm's pose.
- Move the left arm; the right arm follows. Gripper keys: **V** open, **B** cube width, **C** squeeze, **Z** close
  (13 N cap).
- Each episode lasts up to 90 s: **→** ends it early (saved), **←** discards and re-records it, **Esc** stops the session.
  Between episodes there is a 30 s reset period (**→** skips it).
- At the end both arms ramp to the rest pose before torque is cut, the leader too: let go of it when Esc is pressed.
- A rerun window shows the cameras and joint values live.

Follower safety (`plugins/lerobot_robot_yam`, every 10 ms): targets clipped to the joint limits and slewed at ≤ 1.5
rad/s (`--robot.max_vel`) with a velocity feed-forward (`--robot.vff`); a target the rig model says hits the table, the
floor or the left arm, or puts the grasp point below the table top, is not sent; if the arm falls 0.3 rad behind its
command it holds for 1 s and re-engages slowly; if either arm's motor chain dies (a motor off CAN), recording stops
instead of saving frozen data.

Tips: about 40–60 s per demo from rest (reach + 3–4 screw cycles), the same task name every time, 50–100 episodes plus
~10 held out; vary the nut's start angle; keep the cameras fixed.

## Training

```bash
./train.sh screw_from_rest                 # ACT, 100k steps, batch 8 -> outputs/train/act_screw_from_rest (~4.5 h on a 4090)
./train.sh screw_from_rest diffusion       # or any LeRobot policy type; extra args go to lerobot-train
```

Dataset features (what `lerobot-record` writes for `yam_follower`):

| Feature | Shape | Contents |
|---|---|---|
| `observation.state` | 21 | `joint1..6, gripper` × `.pos` (rad; gripper 0 closed..1 open), `.vel`, `.effort` |
| `action` | 7 | `joint1..6.pos, gripper.pos`: the leader's joint angles and the gripper target (LeRobot stores the teleoperator's action) |
| `observation.images.{c922, d415_tripod_top, microdia_left_side, microdia_right_base_plate}` | 480×640×3 | video, 30 fps |

`.effort` is the rig's force signal: there is no force/torque sensor, so it is the motor torque beyond the URDF gravity
model (joints 1-6, N m) and the gripper motor's effort. It also contains tracking and friction torque. On the first screw
demo the gripper effort separated holding from open clearly (0.43 vs 0.11); the wrist roll's contact torque was weak
(0.23 vs 0.15 N m holding-and-turning vs holding-still, noise 0.09).

## Earlier recordings (MCAP)

Demos recorded before the switch to `lerobot-record` are MCAP files (`data/episodes/...`, from `yam_teleop.record` with
the web-page teleop `yam_teleop.leader_follower`). `yam_teleop.to_lerobot` converts them into exactly the
`lerobot-record` format, so new episodes append to the converted dataset:

```bash
source env.sh
PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_teleop.to_lerobot data/episodes/right_arm_20261122 --task screw_from_rest \
    --repo-id local/screw_from_rest --root data/lerobot/screw_from_rest
```

`yam_teleop.relabel` fixes an MCAP episode's task label without touching its data.

## Files

| File | What it does |
|---|---|
| `plugins/lerobot_robot_yam` | `yam_follower`: right arm + cameras, 100 Hz safety layer, observation/action features |
| `plugins/lerobot_teleoperator_yam` | `yam_leader`: left arm floating, gripper keys |
| `yam_teleop/arms.py` | Opening an arm, park, dead-chain check, leader pose shared with the follower |
| `yam_teleop/gravity.py` | Gravity torques from I2RT's `yam.urdf` (incl. gripper), I2RT's per-joint factors |
| `record.sh`, `train.sh` | `lerobot-record` / `lerobot-train` with this rig's defaults |
| `yam_teleop/to_lerobot.py`, `relabel.py` | Earlier MCAP episodes → LeRobot dataset; relabel an MCAP episode |
| `yam_teleop/leader_follower.py`, `record.py`, `cam_wall.py` | The earlier web-page teleop, MCAP recorder and camera server |

`--robot.mock=true --teleop.mock=true --robot.cameras="{}"` runs the whole pipeline without hardware.
