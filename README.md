# yam-teleop

Teleoperate the **right** YAM arm by moving the **left** YAM arm by hand. Record demonstrations (video from all rig cameras
plus arm state and commands) and train an imitation-learning policy on them with [LeRobot](https://github.com/huggingface/lerobot).

```
left arm (leader)  --you move it-->  joint angles  --copied, checked, rate-limited-->  right arm (follower)
                                                                                         |
cam_wall (4 cameras) --------------------------------------------------> record.py  -> data/episodes/<name>/episode_*.mcap
                                                                                         |
                                              to_lerobot.py -> data/lerobot/<name>  -> train.sh (ACT by default)
```

## Requirements

Runs on the rig PC, next to a checkout of `robotics-rubiks-cube` (private). That checkout supplies the I2RT SDK, the
gripper force cap (`LeashGripper`) and the rig model used for collision checks. `env.sh` finds it at
`~/Desktop/experiments/rubiks_cube/robotics-rubiks-cube`; set `RUBIKS=...` before sourcing it if yours is elsewhere.
Training uses the conda env `lerobot-libero` (LeRobot 0.5.x, CUDA); override it with `LEROBOT_ENV`.

Both arms on CAN (`can_left`, `can_right` up at 1 Mbit/s). Only one program may drive the arms at a time: stop
`teleop/se3_jog.py` (its page's **Park and release**) before starting the leader-follower.

## Collecting demonstrations

Use three terminals, all from this directory after `source env.sh`:

```bash
$CAM_PY -m yam_teleop.cam_wall                    # cameras -> http://localhost:8090 (skip if already running)
```

```bash
$HW_PY -m yam_teleop.leader_follower --task "pick up the cube and put it on the tape"
```

```bash
$REC_PY -m yam_teleop.record --out data/episodes/pick_cube
```

Then open http://127.0.0.1:8780:

1. Both arms start **floating** (gravity compensation from the URDF, all six joints). Hold the left arm.
2. **Engage** (`E`): the right arm moves slowly (0.35 rad/s) to the left arm's pose, then follows it.
3. Type the task. **Start** (`space`) records an episode; `space` again saves it; **Discard** (`X`) throws it away.
4. Gripper from the page or keys: **Open** `V`, **Cube width** `B`, **Squeeze** `C`, **Close** fully `Z` (all force-capped at 13 N).
5. **Float** (`Q`) to let go of the right arm; **Park and release** at the end of the session.

What the follower does to stay safe, every 10 ms: targets clipped to the joint limits and slewed at ≤ 1.5 rad/s
(`--max-vel`); a target that the rig model says hits the table, the floor or the leader, or puts the grasp point below
the table top, is not sent (the page shows *blocked: …*); if it falls 0.3 rad behind its command it stops, holds, and
the current episode is discarded.

Tips for good data: 50+ episodes per task to start; same task text for the same task; vary the start positions; avoid
reaching into the right arm's camera views with your hands; keep each episode to one attempt (discard fumbles).

## Training

```bash
source env.sh
conda run --no-capture-output -n $LEROBOT_ENV python -m yam_teleop.to_lerobot data/episodes/pick_cube \
    --repo-id local/pick_cube --root data/lerobot/pick_cube
./train.sh pick_cube                 # ACT, 100k steps, batch 8 -> outputs/train/act_pick_cube (~4.5 h on a 4090)
./train.sh pick_cube diffusion       # or any LeRobot policy type; extra args go to lerobot-train
```

Dataset features: `observation.state` (follower joint 1-6 + gripper), `action` (follower command), one
`observation.images.<camera>` video per camera (640×480 @ 30 fps), `task`. `--cams c922,microdia_right_base_plate` keeps
only some cameras.

## Files

| File | What it does |
|---|---|
| `yam_teleop/leader_follower.py` | Drives both arms; web page; telemetry stream (`/telemetry/stream`) with the episode state |
| `yam_teleop/record.py` | One MCAP per episode: camera JPEGs, leader/follower joint states, `/action`, full telemetry |
| `yam_teleop/to_lerobot.py` | MCAP episodes → LeRobot v3 dataset, resampled to a fixed frame rate |
| `yam_teleop/gravity.py` | Gravity torques from I2RT's `yam.urdf` (incl. gripper), I2RT's per-joint factors |
| `yam_teleop/cam_wall.py` | Camera server; `/streams` carries all cameras on one connection with capture timestamps |
| `train.sh` | `lerobot-train` with this rig's defaults |

`--mock` on `leader_follower` runs without hardware (a self-moving leader and a simulated follower) for testing the
recorder and the converter.

## Not done yet

Running a trained policy on the right arm (closed-loop deployment) is the next piece: it needs the same safety layer as
the follower (limits, slew, collision checks) between the policy output and the arm.
