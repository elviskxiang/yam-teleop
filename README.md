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
- **Session page: http://127.0.0.1:8780** (same layout as the old teleop page): the cameras pinned at the top,
  follower state (floating / engaging / following / held / FAULT, and *blocked: …* when the safety check stops a move),
  episode state read from lerobot-record (RECORDING episode N + elapsed time, RESET), buttons **Save & next** (→),
  **Re-record** (←), **Stop session** (Esc) and the four gripper buttons. A rerun window also shows cameras and joints.

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
| `observation.images.{c922, d415_tripod_top, microdia_right_base_plate}` | 480×640×3 | video, 30 fps |

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

## Control page (recommended)

```bash
source env.sh && PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_teleop.control      # -> http://127.0.0.1:8780
```

Always on: live camera previews while idle; **Start recording** launches `record.sh` (lerobot-record) for the dataset
named on the page and the first episode begins. **Stop episode** ends it and **saves it immediately** (a few seconds of
video encoding); **Stop & discard** ends it without saving. The session then waits (arms keep following) until
**Start next episode** or **End session** (click twice: both arms park, torque is cut). Gripper buttons, follower state
and the session log are on the same page. Episodes still end (and save) on their own after 90 s. `--mock` uses
simulated arms for testing.

Immediate saving: LeRobot 0.5's `record()` saves an episode only after its reset period; `record.sh` runs
`yam_teleop.record_session`, which is `lerobot-record` unchanged except that it saves when the reset period begins.

Cameras: `microdia_left_side` was dropped on Nov 22 (its view was disturbed). It is no longer recorded or converted, and
it was removed from `data/lerobot/screw_from_rest` with `lerobot-edit-dataset --operation.type remove_feature`; the
4-camera original is in `data/lerobot/_backup_screw_from_rest_4cam`.

## MDF-lite training (diffusion forcing, project plan)

After recording, one command updates the feature cache (only new episodes are encoded) and trains:

```bash
./train_mdf.sh                               # MDF: independent noise per (time step x modality), ~12 min on the 4090
./train_mdf.sh screw_from_rest ablation      # all four noise modes in turn, then a comparison table (~50 min)
./train.sh screw_from_rest                   # baseline policy (LeRobot ACT) on the same dataset
```

| Piece | What it is |
|---|---|
| `yam_mdf/cache.py` | dataset → `data/mdf_cache/<dataset>/ep_NNNN.npz`: 10 Hz steps, state (21), action (7), frozen ImageNet ResNet18 feature (512) per camera |
| `yam_mdf/data.py` | modalities **action** (7), **proprio** (pos + vel, 14), **force** (effort, 7), **vision** (PCA 32 per camera); standardised on the training episodes |
| `yam_mdf/model.py` | causal transformer over (time × modality) tokens; each token has its own noise level k ∈ [0, 100]; predicts clean tokens; DDIM sampling with any mix of known / noisy / unknown tokens |
| `yam_mdf/train.py` | `--noise-mode both` (MDF) / `timestep` (Diffusion Forcing) / `modality` / `sequence` (full-sequence diffusion); window = 8 history + 16 future steps; EMA weights; held-out evaluation |
| `yam_mdf/report.py` | best held-out numbers of every run of a dataset, side by side |
| `yam_mdf/policy.py` | `MDFPolicy(ckpt).act(states, past_actions, images)` → next 8 actions (raw units), optionally told the observation noise level |
| `yam_mdf/run.py` | closed-loop runner on the follower arm (`./run_mdf.sh`) |
| `yam_mdf/demo_plot.py` | demo figures on a held-out episode |

Held-out metrics (last 2 episodes by default, `--holdout N` / `--holdout-episodes 3,7`): `act_rmse` (rad, joints 1-6,
next 8 actions), `hold_rmse` (repeat-the-last-action baseline: the model has to beat it), `grip_mae`, and
`noisy_s{0.25,0.5,1.0}_{unaware,aware}`: action error when Gaussian noise is added to the observation history, given to
the model as clean (unaware) or with its matching noise level (aware) — the plan's "robustness to noisy observations".

Sampling: `--sample-schedule full` denoises the whole future together; `pyramid` is Diffusion Forcing's schedule (the
near future is clean first, each further step lags `--uncertainty` passes). Every evaluation logs both.

### Demo: the policy on the arm, and the backup figures

```bash
./train_mdf.sh screw_from_rest timestep --sample-schedule pyramid   # the paper's Diffusion Forcing (noise per time step)
./run_mdf.sh outputs/mdf/screw_from_rest_timestep/ckpt_best.pt --seconds 60
PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.demo_plot outputs/mdf/screw_from_rest_timestep/ckpt_best.pt
```

`run_mdf.sh` (`yam_mdf/run.py`) runs the policy closed loop on the follower: at 10 Hz it reads the arm and cameras and
keeps the last 0.8 s; every 8 steps it denoises the next 1.6 s and executes the first 0.8 s (interpolated to 30 Hz).
Actions go through the `yam_follower` plugin, so the safety layer is the same as when recording (slow engage, speed
limit `--max-vel` 1.0 rad/s by default, collision checks, blocked joints, dead-arm detection); its page is on
http://127.0.0.1:8781. Close the control page first (its camera preview holds the cameras). Before starting: the left
arm at rest (the collision check assumes it there), the scene as in the recordings, a hand ready. Ctrl+C or Esc stops:
the arm parks, then torque is cut. Each run is saved to `outputs/mdf/<run>/rollouts/`.

`yam_mdf.demo_plot` writes to `outputs/mdf/<run>/demo/`: `rollout_ep<N>.png` (on a held-out episode, a prediction every
0.8 s against what the operator did, per joint) and `pyramid_ep<N>_t<T>.png` (the pyramid's noise level per future step
and pass, and one joint's estimate settling near-first).
