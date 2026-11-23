# yam-teleop — complete handoff (Sunday, Nov 22 2026)

This document records everything that was built, changed, recorded, trained, broken and fixed on the rig on
Nov 22 2026, in enough detail that someone who was not there can understand the current state, reproduce every
step, and carry on. It is long on purpose. Times are local (US Eastern, UTC−5) and come from logs and commits.

Contents

1. [What this project is](#1-what-this-project-is)
2. [The physical rig](#2-the-physical-rig)
3. [The computer and software environments](#3-the-computer-and-software-environments)
4. [The two repositories](#4-the-two-repositories)
5. [Timeline of the day](#5-timeline-of-the-day)
6. [Gravity compensation (what it is, what changed, the bug)](#6-gravity-compensation)
7. [The camera setup](#7-the-camera-setup)
8. [Teleoperation: left arm leads, right arm follows](#8-teleoperation)
9. [Recording, stage 1: MCAP files (superseded)](#9-recording-stage-1-mcap-files-superseded)
10. [Recording, stage 2: LeRobot plugins + lerobot-record (current)](#10-recording-stage-2-lerobot-plugins--lerobot-record-current)
11. [The control page (how to record now)](#11-the-control-page-how-to-record-now)
12. [Datasets: what exists, where, and what is in it](#12-datasets)
13. [Training: MDF-lite (diffusion forcing)](#13-training-mdf-lite)
14. [Running a trained model on the arm, and replaying demos](#14-running-a-trained-model-on-the-arm-and-replaying-demos)
15. [Hardware incidents and their fixes](#15-hardware-incidents-and-their-fixes)
16. [Every command, in order, to rebuild from scratch](#16-rebuild-from-scratch)
17. [Known issues, open questions, next steps](#17-known-issues-open-questions-next-steps)
18. [Reference: every file](#18-reference-every-file)

---

## 1. What this project is

EECS 553 (Team 16: Fitzie Hung, Yash Shah, Ken Zheng, Christ Xu) is writing a reproducibility report on
**Diffusion Forcing** (Chen et al., NeurIPS 2024, arXiv 2407.01392). The final-report extension (see
`docs/EECS553_Team16_Project_Plan.pdf`, section 5) is **MDF-lite**: following *Multimodal Diffusion Forcing for
Forceful Manipulation* (Huang, Hou, Berenson, arXiv 2511.04812), train a diffusion-forcing model in which every
(time step, modality) pair gets its own noise level, on our own robot-arm **screw task** (screwing a large
3D-printed nut onto a bolt), and:

- ablate noise **per time step only** vs **per modality only** vs **both**;
- evaluate **task success over repeated rollouts vs a baseline policy**;
- optionally test **robustness to noisy observations**.

The plan's deadlines: arm demos Nov 19 (slipped to Nov 22), MDF-lite trained by Dec 3, evaluation before Dec 9,
final presentation Wed Dec 9, 7–9 PM.

What exists after today:

| Piece | Status |
|---|---|
| Teleoperation of the right arm by hand-guiding the left arm | working (LeRobot plugins) |
| Recording demonstrations into a LeRobot dataset with a web control page | working |
| Dataset `screw_from_rest` | **20 episodes**, 3 cameras, 30 fps |
| MDF-lite training pipeline (cache, model, training, evaluation, ablation, dashboard) | working |
| A trained model (`screw_from_rest_timestep`) | 0.125 rad held-out action error vs 0.244 rad baseline |
| Closed-loop runner of the model on the real arm | working (5-min and 30-s runs done) |
| Task-success evaluation vs a baseline (ACT) | **not done** |
| MDF `both`-mode model on 20 episodes; full ablation | **not done** (deliberately not retrained) |

---

## 2. The physical rig

### 2.1 Arms

- Two **I2RT YAM** 6-DOF arms with the **linear 4310 gripper** (motor 7 on each chain), mounted identically
  0.61 m apart along y. Same joint angles on both arms ⇒ the right gripper is exactly 0.61 m toward −y from the
  left one, same orientation (verified with the rig's MuJoCo model).
- **Left arm = leader** (you move it by hand; it floats under gravity compensation).
  **Right arm = follower** (copies the leader's joint angles).
- Motors per arm (CAN ids): 1–3 Damiao DM4340 (shoulder/elbow), 4–6 DM4310 (wrist pitch, wrist yaw/pitch, wrist
  roll), 7 the gripper motor. Joint 6 = wrist roll (the joint that screws), range ±2.007 rad (±115°).
- Joint limits used by the safety layer (rad):
  `J_LO = [-2.61799, 0, 0, -1.5708, -1.5708, -2.0944+5°]`, `J_HI = [3.05433, 3.66519, 3.66519, 1.5708, 1.5708, 2.0944-5°]`.
- Rest pose = all joints 0 (arm folded). Arms always park there slowly before torque is cut.
- Motor LEDs: on Damiao motors red generally means powered but disabled; **flashing red** usually means a
  latched fault. Seen after the USB-hub dropouts (section 15). A power cycle of the arms clears latched faults;
  the SDK also clears motor errors when it connects.

### 2.2 CAN

- Two USB-CAN adapters: **CANable 2.5 (candleLight firmware)**, USB id `1d50:606f`, driver `gs_usb`.
- A udev rule names them by serial and brings them up at 1 Mbit/s on every plug-in:
  `/etc/udev/rules.d/99-yam-can.rules`

  ```
  SUBSYSTEM=="net", ACTION=="add", KERNEL=="can*", ATTRS{serial}=="208C336D594E5018", NAME="can_left",  RUN+="/usr/sbin/ip link set can_left up type can bitrate 1000000"
  SUBSYSTEM=="net", ACTION=="add", KERNEL=="can*", ATTRS{serial}=="20A3378145465006", NAME="can_right", RUN+="/usr/sbin/ip link set can_right up type can bitrate 1000000"
  ```
  (install: `sudo cp 99-yam-can.rules /etc/udev/rules.d/ && sudo udevadm control --reload`, replug). Because it
  matches the serial, the adapters can go in any USB port.
- Check: `ip -br link | grep can_` → both `UP`; `ip -details link show can_left` → `ERROR-ACTIVE`, `bitrate 1000000`.

### 2.3 Cameras (after today's change: 3 used)

| Name in datasets | Device | Path used | Status |
|---|---|---|---|
| `c922` | Logitech C922 (`046d:085c`) | `/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_A579FDAF-video-index0` | used |
| `d415_tripod_top` | Intel RealSense D415, colour node | `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:12:1.3-video-index0` | used |
| `microdia_right_base_plate` | Microdia/SONiX "Streaming Camera" | `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0` | used |
| `microdia_left_side` | Microdia/SONiX (same serial SN0001 as the other) | `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.1:1.0-video-index0` | **dropped** (its view was disturbed; removed from all datasets, section 12) |

All are read at 640×480, MJPG, 30 fps. The two Microdias share a serial, so they must be addressed by USB port path;
`/dev/videoN` numbers change on every replug. The C922's view was also bumped once today (between the first and
later demos the nut moved from mid-frame to the upper left).

**C922 frame-rate trap:** with default settings it has `exposure_auto_priority = 1`, which lets auto-exposure halve
its frame rate to **15 fps** in dim light (a whole demo was recorded at 15 fps that way). Fix: V4L2 control
`V4L2_CID_EXPOSURE_AUTO_PRIORITY` (0x009A0903) = 0 (auto-exposure stays on, 30 fps held, a bit more gain). It
**resets on every replug**, so the software sets it every time it opens the camera (`yam_teleop/cam_wall.py:
set_v4l2_ctrl`, called by the camera server, the control page and the follower plugin). `v4l2-ctl` is not
installed on this machine; the code uses the ioctl directly from Python.

### 2.4 The USB hub (important)

A bus-powered 4-port hub, `1a40:0101` (Terminus), in PC port **1-2**, carries:

| Hub port | Device |
|---|---|
| 1-2.1 | Microdia (the dropped left-side camera) |
| 1-2.2 | CAN adapter → `can_left` |
| 1-2.3 | C922 |
| 1-2.4 | CAN adapter → `can_right` |

This hub **dropped out three times today** (16:57:27, 16:59:49, 17:19:13), taking both arms down; at 17:19 it
stopped enumerating altogether (`device not accepting address`, error −71) until it was physically replugged
(back at ~17:51; during that replug one CAN adapter briefly enumerated as an STM32 DFU bootloader, `0483:df11`).
See section 15. **Recommendation: replace it with a powered hub, or plug the two CAN adapters straight into the
PC.** The D415 (port 12) and the right Microdia (port 10) are directly on the PC and were never affected.

### 2.5 The task scene

A large 3D-printed bolt taped (yellow tape) to the table, with a large nut. Demos: from the rest pose, reach, grasp
the nut, turn it with the wrist roll (~230° per grasp, the wrist's full range), release, turn back, regrasp, repeat.
Measured in the demos: one grasp-turn-regrasp cycle takes ~10 s, first grasp ~12 s after start.

---

## 3. The computer and software environments

- Ubuntu 24.04.5 LTS, kernel 7.0.0-43-generic, 24 cores, NVIDIA RTX 4090 (24 GB, driver 580.178.04).
- **No passwordless sudo** for the agent (it was needed once; see section 15).
- Repos:
  - `~/Desktop/ken-fitzie` = **yam-teleop** (this repo; GitHub `elviskxiang/yam-teleop`, private).
  - `~/Desktop/experiments/rubiks_cube/robotics-rubiks-cube` = the older **robotics-rubiks-cube** repo
    (GitHub `elviskxiang/robotics-rubiks-cube`, private, branch `tidy`). yam-teleop depends on it (section 4.2).

Python environments:

| Name | Python | Used for |
|---|---|---|
| conda env **`lerobot-libero`** (`~/miniconda3/envs/lerobot-libero`) | 3.12.13 | **everything current**: lerobot-record, the plugins, the control page, MDF training, running models. `$LR_PY` in `env.sh` |
| `robotics-rubiks-cube/.venv-warp` | 3.13.12 | the first-generation teleop (`se3_jog.py`, `yam_teleop.leader_follower`) — `$HW_PY` |
| `robotics-rubiks-cube/.venv-hw` | 3.12.3 | MCAP writer for the first-generation recorder / rig logger — `$REC_PY` |
| `~/Desktop/experiments/rubiks_cube/.venv-ir` | 3.12.3 | the old camera server (`cam_wall.py`) — `$CAM_PY` |

Versions in `lerobot-libero` today: lerobot **0.5.2** (editable source at `~/Desktop/experiments/libero/lerobot`),
torch 2.11.0+cu130, torchvision 0.26.0, numpy 2.2.6, mujoco 3.8.1, opencv-python 4.13.0.92.

**Packages installed into `lerobot-libero` today** (all with `pip install`, run from `~` — see the pitfall below):

```bash
P=~/miniconda3/envs/lerobot-libero/bin/python
cd ~
$P -m pip install "python-can==4.5.0" "crcmod-plus>=2.3.1" pynput      # I2RT SDK CAN deps; pynput for LeRobot's keys
$P -m pip install tyro "dm-env==1.6" mcap "mcap-ros2-support>=0.5.5"    # further I2RT SDK imports
$P -m pip install "rerun-sdk>=0.24.0,<0.27.0"                           # lerobot-record --display_data
$P -m pip install deepdiff                                              # lerobot-record --resume compatibility check
$P -m pip install -e ~/Desktop/ken-fitzie/plugins/lerobot_robot_yam -e ~/Desktop/ken-fitzie/plugins/lerobot_teleoperator_yam
```

Resulting versions: python-can 4.5.0, crcmod-plus 2.3.6, pynput 1.8.2, tyro 1.0.17, dm-env 1.6, mcap 1.6.0,
mcap-ros2-support 0.6.0, rerun-sdk 0.26.2, deepdiff 9.2.1, lerobot_robot_yam 0.1.0, lerobot_teleoperator_yam 0.1.0.
The full I2RT SDK package was **not** installed (its pins, e.g. numpy==2.2.6 and others, could break the env); it is
imported from source via `PYTHONPATH` instead.

**Pitfall:** never run `pip` (or anything else) with the working directory inside `lerobot/src/lerobot`: that folder
has a `types.py` that shadows Python's stdlib `types` and everything fails with a circular-import error.

**`env.sh`** (source it from the repo root) defines: `RUBIKS` (path of the rubiks repo), `YT_ROOT`,
`PYTHONPATH` (for the old venvs: this repo + rubiks + `rubiks/scripts/hw` + `rubiks/vendor/i2rt` +
`rubiks/vendor/hwdeps`), `HW_PY`, `REC_PY`, `CAM_PY`, `LEROBOT_ENV=lerobot-libero`, `LR_PY` (that env's
python) and **`LR_PYTHONPATH`** = this repo + `rubiks/scripts/hw` + `rubiks/vendor/i2rt` (deliberately **not**
`vendor/hwdeps`, whose packages are built for Python 3.13 and would shadow the 3.12 env's cv2 / pydantic).

---

## 4. The two repositories

### 4.1 yam-teleop (`~/Desktop/ken-fitzie`, GitHub `elviskxiang/yam-teleop`, private)

**Branches:** `main` ends at 3a429a0 (the MDF-lite pipeline). Everything after it — the gravity fix, pyramid
sampling, the runner, the dashboard, the demo model, the tracked runs/cache and this document — is on branch
**`follower-fix-pyramid-cache`** (created by the second session; pushed to GitHub at the end of the day; **not yet
merged into `main`**). Check it out to get the current state: `git checkout follower-fix-pyramid-cache`.

Created today at 15:55 from this session. Access: owner `elviskxiang`; collaborator **`fhung1`** (accepted);
**`BobyWoby`** invited (still pending at the time of writing). Both were asked to be "owners"; a personal-account
repository cannot have other owners or admins (GitHub refused `permission=admin`: "Cannot assign … permission of
admin"), so they were invited with the only collaborator level a personal repo has (push/merge, no settings).
Real owner/admin rights need the repo moved into a GitHub organisation.

Two Claude Code sessions worked in this repo today, both committing as `elviskxiang` (the same git identity: name
`elviskxiang`, email = the GitHub no-reply address). This document was written by the first; the second added the
pyramid sampling, the closed-loop runner, the dashboard and the trained model (marked "second session" below).

Commit history (oldest first):

| Commit | Time | What |
|---|---|---|
| 87593b0 | 15:55 | Right-arm YAM teleop (left leads), MCAP episode recording, LeRobot conversion, training script |
| ba76046 | 16:06 | Gripper "Close" (Z) to the calibrated mechanical stop, still force-capped |
| 4440817 / f4b71c8 | 16:19 (merge of PR #1, by `fhung1`) | `docs/EECS553_Team16_Project_Plan.pdf` |
| 16f152b | 16:32 | Velocity feed-forward on the follower; torque + velocity in the dataset; C922 holds 30 fps |
| d17b0db | 16:52 | `relabel`: change an MCAP episode's task label without touching its data |
| f8b9ecd | 16:58 | Detect a dead arm (motor off CAN), Reconnect button |
| c7f0f1b | 17:14 | **LeRobot plugins** `yam_follower` / `yam_leader`, `record.sh`, converter to the lerobot-record format |
| 4f92781 | 17:14 | ignore `*.egg-info` |
| 22cc733 | 17:22 | Session page inside the follower plugin (cameras, state, episode buttons) |
| 6dd8ef0 | 18:05 | Always-on **control page** (`yam_teleop.control`) |
| d713882 | 18:18 | Save each episode when it stops; End session fix; **drop `microdia_left_side`** |
| fbe7748 | 18:25 | Blocked wrist joint gives way instead of freezing the arm; gripper obeys during a hold |
| 3a429a0 | 18:44 | **MDF-lite training pipeline** |
| 097bbfc | 18:52 | (second session) **Gravity model ignores joint limits**; crashed SDK thread counts as a dead arm |
| b85f819 | 18:52 | (second session) Diffusion Forcing **pyramid sampling**; cache works while recording continues |
| 79d640e | 19:06 | (second session) **closed-loop runner** `run_mdf.sh` / `yam_mdf/run.py`, demo figures |
| 29f3a06 | 19:07 | (second session) training **dashboard** (`yam_mdf.dashboard`, port 8790) |
| 6f59e48 | 19:29 | (second session) keep the demo model `outputs/mdf/screw_from_rest_timestep/ckpt_best.pt` + run records in git |

What is in git (as of the commit that added this document): all code; **all MDF-lite training runs in full**
(`outputs/mdf/*`: config, training log, `ckpt_best.pt` + `ckpt_last.pt` of every run, demo figures, all 7 real-arm
rollouts, the training and dashboard logs); the **MDF-lite feature cache** `data/mdf_cache/screw_from_rest/`
(20 episodes, 28 MB — enough to retrain MDF-lite without the dataset); and **all collected data as a split zip**
in `archive/` (below).

**`archive/` — every dataset and raw recording, zipped.** Several data files exceed GitHub's 100 MB per-file limit
(largest 908 MB) and Git LFS is not installed (its free quota is 1 GB), so the data is stored as **one plain zip
per item, each cut into 95 MB pieces** with `split` (`archive/<item>.zip.part000`, `.part001`, …; made by
`archive/make_zip.sh`, deflate level 1, every zip under 4 GB so no ZIP64). A single split zip of everything
(`zip -s 95m`, 8.9 GB) was tried first and **failed its restore test** (Info-ZIP 3.0 / unzip 6.0 mishandle split
archives over 4 GB: "invalid compressed data", "overlapped components"), so it was not committed.

| Zip (pieces in `archive/`) | Restores to | What | Size |
|---|---|---|---|
| `screw_from_rest.zip.part*` | `data/lerobot/screw_from_rest/` | the LeRobot dataset, 20 episodes, 3 cameras | 1.2 GB |
| `backup_screw_from_rest_4cam.zip.part*` | `data/lerobot/_backup_screw_from_rest_4cam/` | the dataset at 8 episodes with `microdia_left_side` (before removal) | 0.6 GB |
| `episodes_right_arm_20261122.zip.part*` | `data/episodes/right_arm_20261122/` | MCAP episodes 00000–00004 | 1.9 GB |
| `episodes_pre_fix.zip.part*` | `data/episodes/_pre_fix/` | the first, pre-fix MCAP demo | 0.8 GB |
| `rig_log_20261122_154451.zip.part*`, `rig_log_20261122_155452.zip.part*` | `data/rubiks_rig_logs/` | the continuous rig logs (on the rig PC in `robotics-rubiks-cube/data/real/logs/`) | 2.8 + 1.7 GB |

`archive/MANIFEST.sha256` lists the SHA-256 of every original file. To get the data back:

```bash
archive/restore_zip.sh                 # into this repo's data/ (joins the pieces, unzips, checks every hash)
archive/restore_zip.sh /some/dir       # or somewhere else
# by hand, one item: cat archive/screw_from_rest.zip.part* > screw_from_rest.zip && unzip screw_from_rest.zip
```

The restore was tested before committing (every file's hash matched). The archive was pushed in commits of
≤ 1.4 GB (`archive/commit_batches.sh`), because GitHub refuses a single push over 2 GB; the repository is therefore
~9 GB and every clone downloads all of it (`git clone --filter=blob:none` + checking out only what you need avoids
that).

### 4.2 robotics-rubiks-cube (dependency; `~/Desktop/experiments/rubiks_cube/robotics-rubiks-cube`)

yam-teleop needs this checkout for:

- `vendor/i2rt` — the **I2RT SDK** (source; `get_yam_robot`, `MotorChainRobot`, the DM motor driver, the YAM URDF
  at `vendor/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf`, arm config `i2rt/robots/config/yam_v1.yml`). Git-ignored
  in that repo.
- `scripts/hw/yam_io.py` — `LeashGripper` (the gripper squeeze-force cap; the stock servo burst the cube twice in
  October), `GripperMap`, `MockArm`, gains `SIM_KP/SIM_KD/MATCH_KP_MULT`.
- `poc/generated/scene.xml` — the rig's MuJoCo scene (both arms, table, floor), used for collision checks.
- `vendor/hwdeps` — pure-Python mcap readers used by the MCAP converter (appended to `sys.path`).

**Uncommitted work left in that repo today** (all untracked; `git status` shows `?? teleop/`,
`?? scripts/hw/cam_wall.py`, `?? scripts/hw/rig_logger.py`; `teleop/` and `cam_wall.py` were already untracked
before today):

- `teleop/se3_jog.py` — modified: URDF gravity model, I2RT's per-joint gravity factors, separate IK instances for
  readers (race fix), `/telemetry` + `/telemetry/stream` + `/config` endpoints, pinned 2×2 camera layout, single
  connection camera stream, `--grav-factor`, `--friction`.
- `teleop/urdf_gravity.py` — new; **still has the joint-limit bug fixed in yam-teleop's `gravity.py`** (section 6.3).
- `scripts/hw/cam_wall.py` — modified: capture timestamps, `/streams` (all cameras on one connection),
  `/streams.js`, C922 exposure-priority fix (it is a copy of `yam_teleop/cam_wall.py`).
- `scripts/hw/rig_logger.py` — new MCAP rig logger (section 9.1).

---

## 5. Timeline of the day

(Approximate where marked ~; exact times come from logs, commits and kernel messages.)

| Time | Event |
|---|---|
| ~15:20 | Started in the empty folder `~/Desktop/ken-fitzie`. Asked to "enable gravity compensation on the teleop arms". Found the arms: two YAMs on `can_left` / `can_right`, code in the rubiks repo. |
| ~15:30 | Started `teleop/se3_jog.py` (rubiks repo): both arms floating under gravity compensation. Opened the camera wall (`scripts/hw/cam_wall.py`, port 8090, already running). |
| ~15:40 | Requests: wrist gravity comp, all cameras visible, logging of all video + arm measurements in MCAP, gravity from URDF. Found: the browser's 6-connections-per-host limit starved 2 of 4 camera streams; a thread race in `se3_jog.py`; wrist gravity factor overridden to 1.0. Built `urdf_gravity.py`, `rig_logger.py`, single-connection camera stream. |
| 15:44 | Restarted `se3_jog.py` with URDF gravity: wrist pitch torque 1.57 → 1.90 N·m. Rig logger recording started (`rig_20261122_154451.mcap`, then `..._155452.mcap`, 4.5 GB together). |
| 15:55 | Asked: "how do we start teleop training (right arm only), make a repo, add BobyWoby and fhung1 as owners". Chose: new private repo `yam-teleop`, left arm as leader. Built the leader-follower teleop, MCAP episode recorder, LeRobot converter, `train.sh`. Pushed; invited both users. |
| ~16:01 | Parked/stopped `se3_jog`; started `yam_teleop.leader_follower` on the real arms; recorder writing `data/episodes/right_arm_20261122/`. |
| 16:06 | Added gripper "Close" (Z). |
| 16:11–16:14 | **First demo**, "screw the nut", 185 s. Verified: file perfect, but C922 at 15 fps and the follower's wrist roll lagged its command by 0.26 rad (auto-stop at 0.30). |
| 16:19 | Team merged the project plan PDF (PR #1). Assessed feasibility vs the plan (force = estimated torque only; wrist lag; C922 fps; no rollout runner). |
| 16:32 | Fixed: velocity feed-forward (wrist error 0.26 → 0.05 rad), effort/velocity in dataset, C922 30 fps. Moved the first demo to `data/episodes/_pre_fix/`. |
| 16:40–16:55 | Demos: `episode_00000` "screw the nut (pre-aligned and closed)" (183 s), then 00001–00004 from the rest pose (41–66 s); 00001 relabelled to `screw_from_rest` (relabel tool). |
| ~16:55 | Left wrist motor 6 dropped off CAN; then **hub dropout 16:57:27**. Added dead-arm detection + Reconnect. Reconnected. **Hub dropout again 16:59:49.** |
| ~17:03 | User: "use the standard way to record and process robot data" → switch to LeRobot's `lerobot-record`. Converted the four `screw_from_rest` MCAP demos into `data/lerobot/screw_from_rest` (lerobot-record format). Built the plugins, `record.sh`. |
| 17:14 | Plugins committed. Old teleop/recorder/camera wall stopped (arms parked). |
| 17:19:13 | **Hub dropout #3; hub stopped enumerating.** |
| ~17:25 | Agent tried to power-cycle the port via sysfs (needs sudo; none) → user replugged the hub; one CAN adapter came up in DFU mode, then everything enumerated (~17:51). |
| 17:22–18:05 | Session page, then the always-on control page (port 8780), tested with simulated arms. |
| ~18:05–18:15 | User recorded with the control page; "End session" did not work (blocked browser pop-up); the session was ended from the API: dataset 8 episodes. |
| 18:16 | `microdia_left_side` removed from the dataset (LeRobot `remove_feature`); 4-camera original kept as backup. |
| 18:18 | Episodes now save when stopped; End session = click twice. |
| ~18:20 | "Follower is not following": joint 6 tracking-hold loop. Fixed the hold logic (wrist yields). Real root cause found by the second session: gravity model + joint limit (section 6.3). |
| 18:26 | Session ended; 9 episodes. |
| 18:44 | MDF-lite pipeline committed; first model `check9_both` (9 episodes) trained: 0.099 rad vs 0.180 baseline. |
| 18:52–19:07 | Second session: gravity fix, pyramid sampling, runner, dashboard. User recorded up to **20 episodes**. Second session trained **`screw_from_rest_timestep`** (18 train episodes) and ran it on the arm (5 rollouts 19:10–19:19). |
| 19:24 | Replayed demo episode 19 on the right arm (`lerobot-replay`) to see where the screw used to be (it had been moved). |
| 19:26–19:31 | 5-minute closed-loop run of `screw_from_rest_timestep` on the arm. |
| 19:33–19:34 | 30-second run. |

---

## 6. Gravity compensation

### 6.1 How it works on the YAM

In "floating" mode (I2RT SDK `zero_gravity_mode=True`, idle) every motor gets torque = gravity torque of the arm
at its current pose × a per-joint factor (+ optional friction feed-forward), so the arm holds itself up and gives
way to your hand. The SDK computes the gravity torque with a model of the arm (`robot.kdl.compute_inverse_dynamics(q, 0, 0)`).

### 6.2 What was changed

- The model is now built from I2RT's **URDF** (`vendor/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf`, which includes
  the gripper body and fingers), loaded into MuJoCo with visual/collision meshes stripped
  (`yam_teleop/gravity.py: UrdfGravity`, swapped in as `robot.kdl` after `get_yam_robot`). It agrees with the SDK's
  own MJCF model to <0.1 % on every joint (checked on random poses).
- The per-joint factors are I2RT's own tuning for the YAM: `[1.0, 1.1, 1.1, 1.2, 1.0, 1.0]`
  (`i2rt/robots/config/yam_v1.yml`). The old teleop had overridden all six to 1.0, which under-compensated the
  **wrist pitch (joint 4)** — the "wrist has no gravity comp" symptom. Measured at rest after the change:
  joint 2 1.98 N·m, joint 3 7.30 N·m, joint 4 **1.90 N·m** (was 1.57); joints 5 and 6 ≈ 0 at rest (their axes
  carry no gravity load folded; they do when the arm is extended).
- Both the leader and the follower use it (`yam_teleop/arms.py: open_arm`).

### 6.3 The joint-limit bug (fixed in yam-teleop by commit 097bbfc; still present in the rubiks repo copy)

The URDF has joint limits. MuJoCo's `mj_inverse` adds **limit-constraint forces** when a joint reads past its limit.
An arm resting on its joint-2 stop reads a hair past it (−0.009 rad), and the "gravity" torque on joint 2 became
−30 N·m instead of +1.8. The SDK's 25 N·m sanity guard then raised **inside its server thread**: that arm's control
loop died, no command reached its motors (the arm went limp) and its readings froze, while the CAN chain thread
kept running — so the first dead-arm check (`motor_chain.running`) saw nothing. That is why "the follower is not
following" at ~18:20. Fix: `self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT` in
`UrdfGravity` (values inside the range unchanged), and `chain_dead()` also reports an arm whose SDK server thread
(`arm._server_thread`) is no longer alive. `teleop/urdf_gravity.py` in the rubiks repo still lacks this fix.

---

## 7. The camera setup

- Old camera server (`yam_teleop/cam_wall.py`, port 8090; also `rubiks/scripts/hw/cam_wall.py`): 2×2 browser wall,
  every camera re-encoded as JPEG; `/stream/<i>` per camera, **`/streams` = all cameras on one connection**
  (multipart parts tagged `X-Cam: <index>` and `X-Timestamp-Ns: <capture time>`) read by `/streams.js`.
  Reason for `/streams`: browsers allow only **6 connections per host**, so one `<img>` per camera starved the 5th+
  stream as soon as two camera pages were open (reproduced today).
- Current pages (control page, session page, runner page) serve the same `/streams` from LeRobot's own camera
  objects (`cam.read_latest()`), so only one program at a time can hold the cameras: the control page's idle
  preview **hands the cameras over** to a recording session when you press Start and takes them back after.
- Pages pin the camera grid to the top of the viewport (`position: sticky`) so controls can scroll under it.

---

## 8. Teleoperation

Left arm (leader) floats under gravity compensation; you move it. The right arm (follower) copies its joint angles.
The follower's safety layer (`plugins/lerobot_robot_yam/lerobot_robot_yam/yam_follower.py`, a 100 Hz thread;
the same logic existed first in `yam_teleop/leader_follower.py`):

1. Starts **floating**. The first action engages it with a slow ramp (0.35 rad/s) from where it is to the target.
2. Targets are clipped to the joint limits and **slewed at ≤ max_vel** (1.5 rad/s while recording, 1.0 in the
   model runner) per joint.
3. **Velocity feed-forward**: the motors run MIT mode, τ = kp(q* − q) + kd(v* − v) + gravity. With v* = 0 the
   damping makes the arm trail a moving target by ≈ kd·v/kp (the wrist roll trailed 0.26 rad at 1.5 rad/s). v* is
   now the command's own velocity, low-passed (α = 0.25). Gains (from the rubiks repo's sim matching):
   `KP = [80, 80, 80, 30, 15, 15]`, `KD = [11, 11, 11, 1.5, 2.25, 2.25]`. Result on a screw demo: wrist-roll
   tracking error p95 0.257 → 0.046 rad, max 0.275 → 0.067 rad. The command is swapped in **atomically** under the
   SDK's command lock (the SDK's own `command_joint_state` resets the command to zeros outside its lock → a possible
   zero-stiffness tick).
4. **Collision check** every tick with the rig model (`poc/generated/scene.xml`): a target that touches the table,
   the floor or the left arm (pose published by the leader plugin; assumed at rest when no leader runs), or puts the
   grasp point below the table top (z < 0.735 m), is not sent (page shows "blocked: …"). Check costs 0.1 ms.
5. **Tracking hold**: if a joint falls > 0.30 rad behind its command:
   - shoulder/elbow (joints 1–3): hold the whole arm 1 s, then re-engage slowly;
   - wrist (joints 4–6): that joint **gives way** (commanded where it physically is) while the rest keeps following,
     and rejoins when the leader's joint comes back within 0.05 rad. (Before 18:25 a stuck wrist froze the whole arm
     in a 2-second hold/re-engage loop.)
   - the gripper always obeys its target, even during a hold (so a clamped gripper can be opened).
6. **Dead arm**: if an arm's SDK chain or server thread dies, recording stops (raises) instead of saving frozen data.
7. **Gripper**: commanded from keys/buttons (V open 1.0, B cube width 0.62, C squeeze 0.45, Z close 0.0), slewed at
   1.5 /s, squeeze force capped at **13 N** by `LeashGripper` (the leader's lead-screw gripper can't be driven by
   hand). Gripper position 0 = the calibrated closed stop (the SDK drives to the hard stop at connect).
8. At the end both arms ramp to rest (≥ 5 s, 0.3 rad/s) before torque is cut — **including the leader, so let go of
   it**.

---

## 9. Recording, stage 1: MCAP files (superseded)

Kept for the history and because some raw data is in this format.

### 9.1 Rig logger (rubiks repo, `scripts/hw/rig_logger.py`)

Continuous MCAP of everything: per camera `sensor_msgs/msg/CompressedImage` (JPEG as served, capture-time stamped),
per arm `sensor_msgs/msg/JointState` + full JSON telemetry at 100 Hz (pos/vel/eff, gripper, motor temperatures,
URDF gravity torque, motor torque command, joint command, end-effector pose), `/rig/status` once a second; URDF as
attachment; new file every 10 min; LZ4 chunks. Output today: `rubiks/data/real/logs/rig_20261122_154451.mcap`
(2.8 GB, 15:44–15:54) and `rig_20261122_155452.mcap` (1.7 GB, 15:54–16:01). (One early stop attempt signalled the
shell wrapper instead of the Python process; the logger kept writing; files were fine once it stopped.)

### 9.2 Episode recorder (`yam_teleop/record.py`) with the web teleop (`yam_teleop/leader_follower.py`, port 8780)

One MCAP per episode in `data/episodes/<name>/episode_NNNNN.mcap` (`.part` until saved; discarded ones deleted;
numbering continues after the highest existing). Topics: `/cam/<name>/image/compressed`, `/follower/joint_states`,
`/leader/joint_states`, `/action` (follower command), `/telemetry` (JSON). Metadata `episode` (task, start time,
cameras) and `rig` (gains, gravity model, link specs). Uncompressed chunks.

Files recorded this way (`data/episodes/`):

| File | Task label | Notes |
|---|---|---|
| `_pre_fix/right_arm_20261122_episode_00000.mcap` | "screw the nut" | 185 s, before the fixes (wrist lag, C922 15 fps); moved out of the dataset |
| `right_arm_20261122/episode_00000.mcap` | "screw the nut (pre-aligned and closed)" | 183 s, starts holding the nut |
| `right_arm_20261122/episode_00001.mcap` | `screw_from_rest` (relabelled from "screw the nut (from park) 1") | 41 s |
| `right_arm_20261122/episode_00002–00004.mcap` | `screw_from_rest` | 60, 43, 66 s |

`yam_teleop/relabel.py` changes an episode's task label in its metadata without touching data (copies record by
record, checks per-topic counts, keeps the old text as `task_as_recorded`; byte-identical payloads verified).

### 9.3 Converter (`yam_teleop/to_lerobot.py`)

Converts these MCAP episodes into exactly the lerobot-record format (features built with LeRobot's own functions
from the plugin's feature lists; **action = the leader's joint angles**, as lerobot-record stores the
teleoperator's action; gripper = commanded target), resampled to 30 fps from the latest sample at or before each
frame. Cameras not in the rig config (now `microdia_left_side`) are skipped.

```bash
source env.sh
PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_teleop.to_lerobot data/episodes/right_arm_20261122 --task screw_from_rest \
    --repo-id local/screw_from_rest --root data/lerobot/screw_from_rest
```

This is how episodes 0–3 of `screw_from_rest` were made (from MCAP 00001–00004).

---

## 10. Recording, stage 2: LeRobot plugins + lerobot-record (current)

LeRobot discovers third-party devices only from **installed** distributions whose names start with
`lerobot_robot_` / `lerobot_teleoperator_`; hence the two packages in `plugins/` (editable-installed, section 3).

- **`yam_follower`** (`plugins/lerobot_robot_yam`): right arm + cameras. Config fields: `can_channel=can_right`,
  `cameras` (default = the 3 rig cameras), `max_vel=1.5`, `vff=1.0`, `grip_force_N=13`, `grav_factor`,
  `scene_xml`, `park_on_disconnect=True`, `web_port=8780` (0 = off; the session page), `mock=False`.
  - `observation_features` (21 scalars + cameras): `joint1..6, gripper` × `.pos`, `.vel`, `.effort`
    (effort = measured motor torque minus URDF gravity torque × factor, N·m; gripper = motor effort).
  - `action_features`: `joint1..6.pos, gripper.pos`.
  - Applies the C922 exposure fix before opening cameras.
- **`yam_leader`** (`plugins/lerobot_teleoperator_yam`): left arm floating; `get_action()` = its joint angles +
  the shared gripper target; publishes its pose for the follower's collision check; keyboard V/B/C/Z (pynput);
  parks on disconnect. Config: `can_channel=can_left`, `grav_factor`, `friction`, `gripper_keys`,
  `park_on_disconnect`, `mock`.
- Shared code: `yam_teleop/arms.py` (open an arm, park, dead-chain check, leader pose / gripper target channel).

**`record.sh <dataset> [num_episodes]`** runs **`yam_teleop.record_session`** = LeRobot's `lerobot-record`
unchanged except that each episode is **saved as soon as it stops** (LeRobot 0.5's `record()` saves only after its
reset period; the wrapper saves when the reset begins and skips LeRobot's own save once). Args it passes:
`--robot.type=yam_follower --robot.id=right --teleop.type=yam_leader --teleop.id=left
--dataset.repo_id=local/<name> --dataset.root=data/lerobot/<name> --dataset.single_task=<TASK or name>
--dataset.fps=30 --dataset.episode_time_s=$EPISODE_S(90) --dataset.reset_time_s=$RESET_S(30)
--dataset.push_to_hub=false --display_data=$DISPLAY_DATA(true) --resume=<true if the folder exists>
--robot.web_port=$WEB_PORT(8780) --robot.mock=$MOCK --teleop.mock=$MOCK`. It refuses to start while another
program holds the arms or cameras.

LeRobot's keys (pynput, global — work from any window): **→** end the episode (save), **←** discard and re-record,
**Esc** stop the session.

Tested end to end with simulated arms (`MOCK=true`) and with the real cameras, including `--resume` on a copy of the
real dataset.

---

## 11. The control page (how to record now)

```bash
cd ~/Desktop/ken-fitzie
source env.sh && PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_teleop.control     # → http://127.0.0.1:8780
```

- **Idle**: live preview of the 3 cameras; dataset name and task fields (default `screw_from_rest`); episodes in the
  dataset.
- **● Start recording**: hands the cameras over and launches `record.sh <dataset> 200` with `RESET_S=3600`,
  `WEB_PORT=8781`, `DISPLAY_DATA=false`; ~10 s to connect the arms (each gripper opens and closes once to
  calibrate), then episode N records and the right arm ramps to the left arm's pose.
- **■ Stop episode**: saves it immediately (a few seconds of video encoding, "SAVING"), then **WAITING** (arms keep
  following) until **▶ Start next episode**. **✕ Stop & discard**: not saved; the same index is recorded next.
- **End session (park arms)**: click **twice** (the first click arms it for 4 s). Saves any episode in progress,
  parks both arms, cuts torque; the page returns to idle preview. (The first version used a `confirm()` pop-up, which
  embedded browsers suppress — the button silently did nothing.)
- Gripper buttons; follower state (FLOATING / ENGAGING / FOLLOWING / HELD / FAULT, "blocked: …"); the last lines of
  the session log.
- Episodes also end on their own after 90 s (and are saved).
- `--mock` runs sessions with simulated arms (cameras real) for testing.

The page relays the session page that the follower plugin serves on 8781 (episode state is parsed from
lerobot-record's log lines: "Recording episode N", "Reset the environment", "Saving episode N", "Episode N saved",
"Re-record episode", "Stop recording"; buttons press the arrow/Esc keys through pynput).

---

## 12. Datasets

All under `~/Desktop/ken-fitzie/data/` (git-ignored as files; all of it is in git inside the split zip
`archive/*.zip.part*` — restore with `archive/restore_zip.sh`, section 4.1).

| Path | What |
|---|---|
| **`data/lerobot/screw_from_rest/`** | **The dataset.** LeRobot v3.0, `robot_type=yam_follower`, 30 fps, **20 episodes, 26,470 frames**, task `screw_from_rest`, 3 cameras, 2.2 GB (with the backup). |
| `data/lerobot/_backup_screw_from_rest_4cam/` | The same dataset as it was at 18:16 (8 episodes) **with** `microdia_left_side`; kept as a backup of the camera removal; deletable. |
| `data/episodes/right_arm_20261122/` | MCAP episodes 00000–00004 (section 9.2) |
| `data/episodes/_pre_fix/` | the first, pre-fix demo |
| `data/mdf_cache/screw_from_rest/` | MDF-lite feature cache (section 13), 28 MB |

`screw_from_rest` features:

| Feature | Shape | Content |
|---|---|---|
| `observation.state` | 21 | names `joint1.pos … joint6.pos, gripper.pos, joint1.vel … gripper.vel, joint1.effort … gripper.effort` (rad, rad/s, N·m; gripper 0 closed … 1 open) |
| `action` | 7 | `joint1.pos … joint6.pos, gripper.pos` — leader joint angles + gripper target |
| `observation.images.c922`, `.d415_tripod_top`, `.microdia_right_base_plate` | 480×640×3 | AV1 video, 30 fps |

Episode provenance: 0–3 converted from MCAP (`right_arm_20261122/episode_00001–00004`); 4–19 recorded with
lerobot-record on the control page (episode 7 was saved by ending a session; one discarded attempt at index 9 was
re-recorded). Episodes are ~36–66 s. Episodes 16–19 each start from rest and use the wrist's full range (±2.1 rad).

**Camera removal** (18:16): `lerobot-edit-dataset --repo_id local/screw_from_rest --root data/lerobot/screw_from_rest
--new_repo_id local/screw_from_rest --new_root data/lerobot/_tmp… --operation.type remove_feature
--operation.feature_names "['observation.images.microdia_left_side']" --push_to_hub=false`, verified (same
episodes, frames, state, action, images), then swapped in; original moved to `_backup_screw_from_rest_4cam`. The
episode metadata still lists the old camera's video columns (harmless).

LeRobot quirk: while a session is recording, the newest data parquet has no footer and the newest episodes' metadata
is buffered; reading the dataset then fails ("Parquet magic bytes not found"). Read it only when no session runs
(the MDF cache works around this, section 13).

---

## 13. Training: MDF-lite

### 13.1 One command

```bash
cd ~/Desktop/ken-fitzie
./train_mdf.sh                                         # cache (new episodes only) + train MDF (noise mode both)
./train_mdf.sh screw_from_rest timestep --sample-schedule pyramid   # how the demo model was trained
./train_mdf.sh screw_from_rest ablation                # both, timestep, modality, sequence in turn + table
PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.report screw_from_rest     # compare runs
PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.dashboard --port 8790      # live curves at http://127.0.0.1:8790
./train.sh screw_from_rest                             # LeRobot ACT baseline (100k steps, ~4.5 h)
```

`train_mdf.sh` sets `HF_HUB_OFFLINE=1` (local datasets only).

### 13.2 Data pipeline

1. **Cache** (`yam_mdf/cache.py`): read the LeRobot dataset; every 3rd frame (10 Hz); per frame state (21),
   action (7) and, per camera, a **512-d feature from a frozen ImageNet ResNet18** (torchvision
   `IMAGENET1K_V1`, `fc` removed, images resized to 240×320, ImageNet-normalised). Writes
   `data/mdf_cache/<dataset>/ep_NNNN.npz` + `meta.json`; an episode already cached with the same length/settings is
   skipped. Reads through a symlink view of finished files only, so it works while a session is recording
   (unfinished episodes are cached next time). The ResNet18 weights were already in `~/.cache/torch/hub/checkpoints`.
2. **Modalities** (`yam_mdf/data.py`), per 10 Hz step:
   `action` (7), `proprio` (state[0:14] = pos + vel), `force` (state[14:21] = effort), `vision` (per camera PCA of
   the 512-d feature to 32 dims, fitted on the training episodes → 96). Every modality standardised with
   training-episode mean/std (std floor 1e-3).
3. **Split**: last `--holdout 2` episodes held out (or `--holdout-episodes 3,7`).
4. **Windows**: random windows of L = n_ctx + n_pred = 8 + 16 = 24 steps (0.8 s history incl. "now", 1.6 s future),
   episodes weighted by length.

### 13.3 Model (`yam_mdf/model.py`, ~4.94 M parameters)

- Tokens: one per (time step, modality): 24 × 4 = 96. Token = `Linear_m(x)` + learned modality embedding + learned
  time-step embedding + MLP(sinusoidal embedding of its **own noise level k ∈ [0, 100]**).
- Transformer encoder: 6 layers, d_model 256, 8 heads, FFN 1024, GELU, pre-norm, dropout 0.1.
  **Causal in time** (a token attends to all modalities at its own and earlier steps), as in Diffusion Forcing.
- Output: per modality LayerNorm + Linear → **predicted clean value x0**. Loss: MSE per modality, averaged with
  equal weight per modality.
- Noise: cosine ᾱ schedule, K = 100 levels (ᾱ₀ = 1).
- **Noise modes** (`--noise-mode`): `both` = k independent per (time step, modality) — MDF; `timestep` = per time
  step, shared by modalities — original Diffusion Forcing; `modality` = per modality, shared over the window;
  `sequence` = one k for the window — ordinary full-sequence diffusion.
- **Sampling** (`MDF.sample`): deterministic DDIM over known (level 0), noisy-known (level k > 0 = MDF's
  noise-aware conditioning, value given as √ᾱ_k·x) and unknown (−1) tokens. Schedules: `full` (all unknown tokens
  from K to 0 together) or `pyramid` (Diffusion Forcing: each further future step lags by `uncertainty` sampling
  steps, so the near future is clean first), `--sample-steps 10`.
- As a policy: history (8 steps) clean for observations, past actions clean; the action "now" and 16 future steps of
  every modality are generated; the first `n_exec = 8` actions are executed.

### 13.4 Training loop (`yam_mdf/train.py`)

AdamW lr 3e-4, wd 1e-4, 500 warmup steps then cosine decay, batch 128, bf16 autocast, grad clip 1.0, EMA 0.999 (the
EMA weights are evaluated and saved), 10,000 steps by default (~40 steps/s → ~4–5 min), evaluation every 1,000
steps. Outputs in `outputs/mdf/<dataset>_<mode>/`: `config.json` (all args, dims, split, cache meta),
`log.jsonl`, `ckpt_best.pt` (lowest held-out `act_rmse`), `ckpt_last.pt`. Checkpoints contain the EMA weights,
config and the **normaliser + PCA**, so a checkpoint is self-contained.

Held-out metrics (EMA, on windows every 5 steps of the held-out episodes, ≤ 400 windows):
`act_rmse` (rad, joints 1–6, next 8 actions, history clean), `act_rmse_full` / pyramid (the other schedule),
`hold_rmse` (repeat-the-last-action baseline — a useful model must beat it), `grip_mae`, and
`noisy_s{0.25,0.5,1.0}_{unaware,aware}` (Gaussian noise of that std, in standardised units, added to the observed
history; given as clean vs with its matching noise level).

### 13.5 Runs that exist

| Run | Data | Mode / sampling | Best held-out act_rmse | Hold baseline | Notes |
|---|---|---|---|---|---|
| `check9_both` | episodes 0–8 (7 train: 0,1,3–8; held out 2 — a default-argument bug at the time held out episode index 2) | both / full, 30k steps | **0.099** rad at step 4,000 (rose to 0.133 by 30k: overfitting) | 0.180 | weights, config, log, demo figures in git |
| **`screw_from_rest_timestep`** (the demo model; trained by the second session) | 20 episodes: train 0–17 (8,058 steps), held out 18, 19 | timestep / **pyramid**, 10k steps | **0.125** rad at 10,000 (full schedule 0.127) | 0.244 | `ckpt_best.pt` + `ckpt_last.pt` in git; noisy-obs: σ=1.0 unaware 0.161 vs aware 0.119 |

An MDF `both` run on all 20 episodes was started at ~19:20 and **stopped and deleted on purpose** (the user did not
want retraining).

### 13.6 Policy API

`yam_mdf/policy.py: MDFPolicy(ckpt, sample_steps=None, schedule=None, uncertainty=None, n_exec=None)`;
`encode(images_one_step)` → (C, 512); `act_features(states (8,21), actions_past (7,7), vis (8,C,512), obs_noise=0,
full=False)` → next `n_exec` actions (raw units) or the whole predicted future; `act(states, actions_past,
images)` encodes then calls it. 46 ms per call measured with the early version; planning in the runner is 126–155 ms
(median 131 ms) with pyramid + 10 steps.

---

## 14. Running a trained model on the arm, and replaying demos

### 14.1 Closed-loop runner (second session: `run_mdf.sh` → `yam_mdf/run.py`)

```bash
cd ~/Desktop/ken-fitzie
./run_mdf.sh outputs/mdf/screw_from_rest_timestep/ckpt_best.pt --seconds 300      # options: --max-vel 1.0 --exec 8
                                          # --schedule full|pyramid --sample-steps N --web-port 8781 --mock
```

Before: left arm at rest (collision check assumes it there), scene as in the recordings, a hand ready, nothing else
holding the arms or cameras (it refuses otherwise; stop the control page first). It: loads the checkpoint, opens the
3 cameras and the right arm through the `yam_follower` plugin (same safety layer, max 1.0 rad/s), reads arm + cameras
at 10 Hz, encodes each frame once, plans every `n_exec` steps from the last 8 steps (the next plan starts one step
early in a background thread), executes the actions interpolated to 30 Hz, **Esc / Ctrl+C** → park, torque off.
Page with cameras and arm state: http://127.0.0.1:8781. Each run is saved to
`outputs/mdf/<run>/rollouts/<YYYYmmdd_HHMMSS>.npz` (t, state, action, plan_t, plans, plan_ms, names, schedule).

Runs today with `screw_from_rest_timestep`: 19:10 (20 s), 19:12 (45 s), 19:15 and 19:18 (90 s), 19:19 (5 min,
stopped after a few seconds) — by the second session, all tracked in git; **19:26:54–19:31:54 (5 min: 3,001 steps,
376 plans, no blocks/faults)** → `rollouts/20261122_193154.npz`; **19:33:58–19:34:28 (30 s: 300 steps, 38 plans)**
→ `rollouts/20261122_193428.npz` (all seven rollouts are in git). Issue: median planning 131 ms > the 100 ms a plan
has, so 172 of 376 plans arrived 30–60 ms late (a short hesitation every 0.8 s); try `--sample-steps 6` or
`--schedule full`. Whether the screw got screwed was not observed by the agent (no camera review).

### 14.2 Replaying a recorded demo (LeRobot's standard `lerobot-replay`)

Sends exactly the recorded actions (leader joints + gripper) at 30 fps through the follower's safety layer, no model,
no cameras. Used at 19:24 on episode 19 (41 s, clean) to see where the screw had been (it had been moved since).

```bash
cd ~/Desktop/ken-fitzie && source env.sh
HF_HUB_OFFLINE=1 PYTHONPATH=$LR_PYTHONPATH $(dirname $LR_PY)/lerobot-replay --robot.type=yam_follower --robot.id=right \
  --robot.cameras="{}" --robot.web_port=8781 --dataset.repo_id=local/screw_from_rest \
  --dataset.root=data/lerobot/screw_from_rest --dataset.episode=19 --play_sounds=false
```

### 14.3 Generalisation (answers given today)

- **Moved screw:** probably not handled — the images enter as globally pooled ResNet features (little position
  information) and every demo had the screw in the same place; the model learned the motion toward the old spot.
  Fix: demos with the screw at several positions (hold one out), and/or spatial image features.
- **Nut height as it threads down:** within the demonstrated range yes (the descending regrasp pattern is in every
  demo, proprio + gripper effort give feedback; a few mm per turn); a different starting height probably not.
- **Generating without cameras:** possible in principle with an MDF `both` model (vision can be marked unknown);
  the demo model is `timestep` mode, which never saw mixed known/unknown modalities, so not with it.

---

## 15. Hardware incidents and their fixes

| When | What happened | Cause | Fix / state |
|---|---|---|---|
| 15:40 | 2 of 4 camera streams black with two camera pages open | browser 6-connections-per-host limit | `/streams` single connection |
| all day | C922 at 15 fps | `exposure_auto_priority=1` | set to 0 on every open |
| 16:14 | wrist roll 0.26 rad behind (auto-stop 0.30) | damping with zero velocity target | velocity feed-forward |
| ~16:55 | left wrist motor 6 "fail to communicate"; then **16:57:27 hub dropout** (both CAN adapters, C922, Microdia) | the hub (see below); first event's cause unclear | dead-arm detection + Reconnect; reconnected |
| 16:59:49 | hub dropout #2 (both chains die, motor 6 left / motor 1 right first to error) | the hub | — |
| 17:19:13 | hub dropout #3, then it would not enumerate (`device not accepting address`, −71) | hub failed | sysfs port power-cycle needs **root** (`/sys/bus/usb/devices/usb1/1-0:1.0/usb1-port2/disable`), no sudo; `systemctl suspend` was the only non-root reset (not used); user **replugged the hub** (~17:25–17:51), one CAN adapter came up once as DFU `0483:df11`, then all fine |
| after the dropouts | motor LEDs flashing red | latched motor faults after comms loss | power-cycle arms or reconnect (SDK clears errors) |
| ~18:20 | follower not following; joint 6 in a hold loop | gravity model limit-constraint torque → SDK guard killed the follower's control thread (section 6.3) + the hold logic froze the arm | 097bbfc (gravity), fbe7748 (yielding wrist) |
| 18:15 | "End session" did nothing | `confirm()` pop-up blocked in the embedded browser | click-twice button |
| shutdowns | `file descriptor cannot be a negative integer (-1)`, `DM Error in control loop` after "Robot closed" | SDK threads touching the CAN socket after close | harmless noise |

Process-control pitfall: a background `cmd | grep` started from a shell is a `bash` wrapper + the Python process;
signalling the wrapper's PID leaves the Python process running (happened once with the rig logger). Also
`pkill -f <pattern>` can match its own shell. Send signals to the Python process's PID.

---

## 16. Rebuild from scratch

On a fresh machine with the two arms, the cameras and this repo:

0. **Data**: `git checkout follower-fix-pyramid-cache && archive/restore_zip.sh` puts every dataset and recording
   back under `data/` (checks all hashes).
1. **Rubiks repo** (needs `vendor/i2rt` and `vendor/hwdeps`, which are git-ignored there — copy them from this
   machine): place at `~/Desktop/experiments/rubiks_cube/robotics-rubiks-cube` or set `RUBIKS=…`.
2. **CAN**: install `99-yam-can.rules` (section 2.2), plug the adapters (preferably directly into the PC), check
   `ip -br link | grep can_`.
3. **Cameras**: check the paths in section 2.3 exist (`ls /dev/v4l/by-id /dev/v4l/by-path`); if ports differ, edit
   `rig_cameras()` in `plugins/lerobot_robot_yam/lerobot_robot_yam/config_yam_follower.py`.
4. **Conda env** with LeRobot 0.5.x (`lerobot-libero`), then the pip installs of section 3 (from `~`).
5. `cd ~/Desktop/ken-fitzie && source env.sh` and check imports:
   `PYTHONPATH=$LR_PYTHONPATH $LR_PY -c "from lerobot_robot_yam import YamFollower; from lerobot_teleoperator_yam import YamLeader; import i2rt, yam_io; print('ok')"`.
6. **Dry run without hardware**: `PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_teleop.control --mock`, open
   http://127.0.0.1:8780, Start with a throwaway dataset name, stop/save/discard/end, delete `data/lerobot/<name>`.
7. **Record**: start the control page without `--mock` (section 11), both arms at rest, Start, demonstrate, Stop,
   Start next, …, End session (twice).
8. **Train**: `./train_mdf.sh` (or the timestep/pyramid variant, or the ablation); watch
   `python -m yam_mdf.dashboard`; compare with `python -m yam_mdf.report screw_from_rest`.
9. **Run**: stop the control page, `./run_mdf.sh outputs/mdf/<run>/ckpt_best.pt --seconds 30`.
10. **Baseline**: `./train.sh screw_from_rest` (ACT), then run it with
    `lerobot-record --policy.path=outputs/train/act_screw_from_rest/checkpoints/last/pretrained_model …` (not
    done yet).

---

## 17. Known issues, open questions, next steps

1. **Replace the USB hub** / put the CAN adapters directly on the PC.
2. **Task-success evaluation** (the plan's main result): repeated rollouts of MDF-lite vs a baseline (train ACT with
   `./train.sh`), with a fixed success criterion (e.g. N turns or nut reaches the tape), not done.
3. **Ablation** (`both / timestep / modality / sequence`) on the 20 episodes: not run. Note `check9_both` held out a
   different episode than intended (fixed bug).
4. **Planning latency** 131 ms vs the 100 ms budget: fewer sampling steps or the full schedule.
5. **Generalisation**: screw position and nut height are fixed in all demos; record varied ones (with held-out
   positions/heights) if the evaluation should cover them; consider spatial image features.
6. **Force signal** is estimated torque, not a force/torque sensor: gripper effort is informative (0.43 vs 0.11 N·m
   holding vs open), wrist-roll contact torque weak (0.23 vs 0.15 N·m, noise 0.09). Describe it that way in the
   report.
7. **Rubiks repo**: `teleop/urdf_gravity.py` still has the joint-limit bug; the rubiks changes are uncommitted.
8. The datasets and raw recordings are in git only as the split zip in `archive/` (section 4.1). New recordings
   after Nov 22 are **not** in it: rerun `archive/make_zip.sh` (add sources to it) and `archive/commit_batches.sh`,
   or move to a private Hugging Face dataset repo (LeRobot's standard, `LeRobotDataset.push_to_hub`) as the data
   grows.
9. `BobyWoby` has not accepted the repo invitation; "owner" rights need an organisation.
10. Untested: the `--no-vision` idea (not implemented); closed-loop success of the demo model (not observed).

---

## 18. Reference: every file

yam-teleop (`~/Desktop/ken-fitzie`):

| File | Purpose |
|---|---|
| `env.sh` | paths and interpreters (section 3) |
| `record.sh` | lerobot-record (via `record_session`) with the rig defaults |
| `train.sh` | LeRobot `lerobot-train` (ACT default) on `data/lerobot/<name>` |
| `train_mdf.sh` | MDF-lite cache + train / ablation |
| `run_mdf.sh` | run a checkpoint on the arm |
| `plugins/lerobot_robot_yam/…/config_yam_follower.py`, `yam_follower.py` | follower robot plugin |
| `plugins/lerobot_teleoperator_yam/…/config_yam_leader.py`, `yam_leader.py` | leader teleoperator plugin |
| `yam_teleop/arms.py` | open/park arms, dead-chain check, shared leader pose + gripper target |
| `yam_teleop/gravity.py` | URDF gravity model, factors |
| `yam_teleop/control.py` | always-on control page (8780) |
| `yam_teleop/web.py` | session page inside the follower plugin (8781 when launched by the control page) |
| `yam_teleop/record_session.py` | lerobot-record with save-on-stop |
| `yam_teleop/cam_wall.py` | old camera server; `set_v4l2_ctrl`, `STREAMS_JS` used by the pages |
| `yam_teleop/leader_follower.py`, `record.py`, `to_lerobot.py`, `relabel.py` | stage-1 web teleop, MCAP recorder, converter, relabel |
| `yam_mdf/cache.py`, `data.py`, `model.py`, `train.py`, `policy.py`, `report.py` | MDF-lite pipeline |
| `yam_mdf/run.py`, `demo_plot.py`, `dashboard.py` | runner, demo figures, live dashboard (second session) |
| `outputs/mdf/screw_from_rest_timestep/` | demo model (`ckpt_best.pt`), config, log, figures, rollouts |
| `outputs/mdf/check9_both/` | first check run (config, log, figures) |
| `docs/EECS553_Team16_Project_Plan.pdf` | the project plan |
| `docs/HANDOFF.md` | this document |
| `archive/*.zip.part*` | all datasets and raw recordings: one zip per item, cut into 95 MB pieces |
| `archive/MANIFEST.sha256` | SHA-256 of every original file in the zip |
| `archive/make_zip.sh`, `restore_zip.sh`, `commit_batches.sh` | build the zip; restore and verify it; commit/push it in ≤ 1.4 GB batches |

Ports: 8780 control page (and the stage-1 teleop page), 8781 session/runner page, 8090 old camera wall,
8790 training dashboard (was running at the time of writing), 8770 the old `se3_jog` page.
