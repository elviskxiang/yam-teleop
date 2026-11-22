"""Convert recorded episodes (yam_teleop.record MCAP files) into a LeRobot v3 dataset for training.

  source env.sh && conda run --no-capture-output -n $LEROBOT_ENV python -m yam_teleop.to_lerobot \
      data/episodes/<name> --repo-id local/<name> --root data/lerobot/<name> [--fps 30] [--cams c922,microdia_right_base_plate]

One dataset frame per 1/fps s, from the latest sample at or before that instant:
  observation.state           (7,)  follower joint1-6 (rad) + gripper (0 closed .. 1 open), measured
  observation.velocity        (7,)  follower joint1-6 + gripper velocity, measured
  observation.effort          (7,)  torque beyond the URDF gravity model on joints 1-6 (N m) + gripper motor effort: the
                                    rig's contact-force signal (no F/T sensor; includes tracking and friction torque)
  action                      (7,)  follower command at that instant (what the policy must output)
  observation.images.<cam>    (480, 640, 3) RGB video per camera
  task                        the episode's task text
The span is where every stream has data (first sample of the slowest stream .. last sample of the earliest-ending one).

Runs in the LeRobot env, which has no mcap reader: the pure-Python mcap / mcap_ros2 packages are taken from the rubiks
checkout's vendor/hwdeps, appended to sys.path so the env's own packages (cv2, numpy, ...) still win.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.join(os.environ.get("RUBIKS", ""), "vendor", "hwdeps"))
import cv2  # noqa: E402
from mcap.reader import make_reader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402

NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper"]


def read_episode(path):
    """-> task, {cam: (t[n], [jpeg bytes])}, follower (t, q7), action (t, a7), dyn (t, vel7, effort7)

    effort = the torque the motors apply beyond the URDF gravity model (joint_eff - grav_torque, N m) on joints 1-6,
    plus the gripper motor's effort: the rig has no force/torque sensor, so this is its contact-force signal. It still
    contains the PD tracking and friction torque, not only contact."""
    with open(path, "rb") as f:
        r = make_reader(f)
        meta = {m.name: m.metadata for m in r.iter_metadata()}
        task = meta.get("episode", {}).get("task", "")
        dfac, decs = DecoderFactory(), {}
        cams, fol, act, dyn = {}, ([], []), ([], []), ([], [], [])
        for sch, ch, msg in r.iter_messages(topics=None):
            if ch.topic == "/telemetry":
                x = json.loads(msg.data); F = x["follower"]
                dyn[0].append(msg.log_time)
                dyn[1].append(F["joint_vel"] + (F.get("gripper_vel") or [0.0])[:1])
                dyn[2].append([e - g for e, g in zip(F["joint_eff"], F["grav_torque"])] + (F.get("gripper_eff") or [0.0])[:1])
                continue
            if ch.message_encoding != "cdr":
                continue
            dec = decs.get(sch.id) or decs.setdefault(sch.id, dfac.decoder_for("cdr", sch))
            m = dec(msg.data)
            if ch.topic.startswith("/cam/"):
                c = cams.setdefault(ch.topic.split("/")[2], ([], []))
                c[0].append(msg.log_time); c[1].append(bytes(m.data))
            elif ch.topic == "/follower/joint_states":
                fol[0].append(msg.log_time); fol[1].append(list(m.position))
            elif ch.topic == "/action":
                act[0].append(msg.log_time); act[1].append(list(m.position))
    cams = {k: (np.array(t), j) for k, (t, j) in cams.items()}
    return (task, cams, (np.array(fol[0]), np.array(fol[1], np.float32)), (np.array(act[0]), np.array(act[1], np.float32)),
            (np.array(dyn[0]), np.array(dyn[1], np.float32), np.array(dyn[2], np.float32)))


def latest(ts, t):
    """index of the latest sample at or before each t (ts sorted)"""
    return np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)


def main():
    ap = argparse.ArgumentParser(description="MCAP episodes -> LeRobot dataset")
    ap.add_argument("episodes", help="directory of episode_*.mcap files")
    ap.add_argument("--repo-id", required=True, help="e.g. local/pick_cube (not pushed anywhere)")
    ap.add_argument("--root", required=True, help="output directory of the dataset (must not exist yet)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cams", default="", help="comma-separated camera names to keep (default: all)")
    ap.add_argument("--min-seconds", type=float, default=1.0, help="skip shorter episodes")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = sorted(Path(a.episodes).glob("episode_*.mcap"))
    if not files:
        raise SystemExit(f"no episode_*.mcap in {a.episodes}")
    if Path(a.root).exists():
        raise SystemExit(f"{a.root} exists; pick a new --root (LeRobot creates it)")
    keep = [c for c in a.cams.split(",") if c]
    ds, cam_list, n_ok = None, None, 0
    for p in files:
        task, cams, (tf, qf), (ta, qa), (td, vd, ed) = read_episode(p)
        if keep:
            cams = {k: v for k, v in cams.items() if k in keep}
        if not cams or len(tf) == 0 or len(ta) == 0 or len(td) == 0:
            print(f"skip {p.name}: missing streams"); continue
        if cam_list is None:
            cam_list = sorted(cams)
            h, w = cv2.imdecode(np.frombuffer(cams[cam_list[0]][1][0], np.uint8), cv2.IMREAD_COLOR).shape[:2]
            features = {"observation.state": dict(dtype="float32", shape=(7,), names=NAMES),
                        "observation.velocity": dict(dtype="float32", shape=(7,), names=NAMES),
                        "observation.effort": dict(dtype="float32", shape=(7,), names=NAMES),
                        "action": dict(dtype="float32", shape=(7,), names=NAMES)}
            for c in cam_list:
                features[f"observation.images.{c}"] = dict(dtype="video", shape=(h, w, 3), names=["height", "width", "channels"])
            ds = LeRobotDataset.create(repo_id=a.repo_id, fps=a.fps, features=features, root=a.root, robot_type="yam_right",
                                       use_videos=True, image_writer_threads=4)
        if sorted(cams) != cam_list:
            print(f"skip {p.name}: cameras {sorted(cams)} != {cam_list}"); continue
        t0 = max([tf[0], ta[0]] + [cams[c][0][0] for c in cam_list])
        t1 = min([tf[-1], ta[-1]] + [cams[c][0][-1] for c in cam_list])
        grid = np.arange(t0, t1, int(1e9 / a.fps))
        if (t1 - t0) / 1e9 < a.min_seconds:
            print(f"skip {p.name}: {(t1 - t0) / 1e9:.1f} s"); continue
        i_f, i_a, i_d = latest(tf, grid), latest(ta, grid), latest(td, grid)
        i_c = {c: latest(cams[c][0], grid) for c in cam_list}
        for k in range(len(grid)):
            frame = {"observation.state": qf[i_f[k]], "observation.velocity": vd[i_d[k]], "observation.effort": ed[i_d[k]],
                     "action": qa[i_a[k]], "task": task}
            for c in cam_list:
                bgr = cv2.imdecode(np.frombuffer(cams[c][1][i_c[c][k]], np.uint8), cv2.IMREAD_COLOR)
                frame[f"observation.images.{c}"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ds.add_frame(frame)
        ds.save_episode()
        n_ok += 1
        print(f"{p.name}: {len(grid)} frames ({len(grid) / a.fps:.1f} s)  task {task!r}", flush=True)
    if ds is None:
        raise SystemExit("no usable episodes")
    ds.finalize()
    print(json.dumps(dict(dataset=a.root, repo_id=a.repo_id, episodes=n_ok, cameras=cam_list, fps=a.fps)))


if __name__ == "__main__":
    main()
