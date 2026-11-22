"""Convert episodes recorded as MCAP (the earlier yam_teleop.record pipeline) into a LeRobot dataset in exactly the format
lerobot-record writes with the YAM plugins, so new demos can be appended to it with `lerobot-record --resume=true`.

  source env.sh && PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_teleop.to_lerobot data/episodes/<name> \
      --repo-id local/screw_from_rest --root data/lerobot/screw_from_rest [--task screw_from_rest]

Same features as lerobot-record (built with the same LeRobot functions from the plugin's feature lists):
  observation.state (21)      joint1-6 + gripper: .pos, .vel, .effort (follower, measured; effort = torque beyond the
                              URDF gravity model, gripper = motor effort)
  action (7)                  joint1-6 + gripper .pos as lerobot-record stores it: the TELEOPERATOR's action, i.e. the
                              leader's joint angles (not the slewed follower command); gripper = the commanded target
  observation.images.<cam>    the four rig cameras, 640x480 video
One frame per 1/fps s from the latest sample at or before that instant, over the span where every stream has data.
Only episodes whose task (metadata) matches --task are converted (default: all).

Needs the mcap readers, which the LeRobot env lacks: they are taken from the rubiks checkout's vendor/hwdeps, APPENDED to
sys.path so the env's own packages win.
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
    with open(path, "rb") as f:
        r = make_reader(f)
        meta = {m.name: m.metadata for m in r.iter_metadata()}
        task = meta.get("episode", {}).get("task", "")
        dfac, decs = DecoderFactory(), {}
        cams, tel = {}, []
        for sch, ch, msg in r.iter_messages():
            if ch.topic == "/telemetry":
                x = json.loads(msg.data)
                if x.get("action") is None:
                    continue
                F, L = x["follower"], x["leader"]
                one = lambda v: (v or [0.0])[:1]
                tel.append((msg.log_time,
                            {**{f"{n}.pos": v for n, v in zip(NAMES, F["joint_pos"] + one(F["gripper_pos"]))},
                             **{f"{n}.vel": v for n, v in zip(NAMES, F["joint_vel"] + one(F.get("gripper_vel")))},
                             **{f"{n}.effort": v for n, v in zip(NAMES, [e - g for e, g in zip(F["joint_eff"], F["grav_torque"])]
                                                                 + one(F.get("gripper_eff")))}},
                            {**{f"{n}.pos": v for n, v in zip(NAMES[:6], L["joint_pos"])}, "gripper.pos": x["action"][6]}))
            elif ch.topic.startswith("/cam/"):
                dec = decs.get(sch.id) or decs.setdefault(sch.id, dfac.decoder_for("cdr", sch))
                c = cams.setdefault(ch.topic.split("/")[2], ([], []))
                c[0].append(msg.log_time); c[1].append(bytes(dec(msg.data).data))
    return task, {k: (np.array(t), j) for k, (t, j) in cams.items()}, tel


def latest(ts, t):
    return np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)


def dataset_features(cameras):
    """Exactly what lerobot-record builds for the yam_follower robot (default processors)."""
    from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.utils.feature_utils import combine_feature_dicts
    from lerobot_robot_yam import YamFollower, YamFollowerConfig
    from lerobot_robot_yam.config_yam_follower import rig_cameras
    cfg = YamFollowerConfig(mock=True, cameras={k: v for k, v in rig_cameras().items() if k in cameras})
    robot = YamFollower(cfg)
    t, _, o = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=t, initial_features=create_initial_features(action=robot.action_features), use_videos=True),
        aggregate_pipeline_dataset_features(pipeline=o, initial_features=create_initial_features(observation=robot.observation_features), use_videos=True),
    ), robot.name


def main():
    ap = argparse.ArgumentParser(description="MCAP episodes -> LeRobot dataset (lerobot-record format)")
    ap.add_argument("episodes", nargs="+", help="episode_*.mcap files or directories of them")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True, help="output dataset directory (must not exist)")
    ap.add_argument("--task", default=None, help="only convert episodes with this task label")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = []
    for e in a.episodes:
        p = Path(e)
        files += sorted(p.glob("episode_*.mcap")) if p.is_dir() else [p]
    if Path(a.root).exists():
        raise SystemExit(f"{a.root} exists; pick a new --root")
    ds = None; cam_list = None; n_ok = 0
    for p in files:
        task, cams, tel = read_episode(p)
        if a.task is not None and task != a.task:
            print(f"skip {p.name}: task {task!r}"); continue
        if not tel or not cams:
            print(f"skip {p.name}: no data"); continue
        if ds is None:
            cam_list = sorted(cams)
            features, robot_type = dataset_features(cam_list)
            ds = LeRobotDataset.create(repo_id=a.repo_id, fps=a.fps, features=features, root=a.root, robot_type=robot_type,
                                       use_videos=True, image_writer_threads=4)
            state_names = features["observation.state"]["names"]; act_names = features["action"]["names"]
        if sorted(cams) != cam_list:
            print(f"skip {p.name}: cameras {sorted(cams)} != {cam_list}"); continue
        tt = np.array([x[0] for x in tel])
        t0 = max([tt[0]] + [cams[c][0][0] for c in cam_list]); t1 = min([tt[-1]] + [cams[c][0][-1] for c in cam_list])
        grid = np.arange(t0, t1, int(1e9 / a.fps))
        it = latest(tt, grid); ic = {c: latest(cams[c][0], grid) for c in cam_list}
        for k in range(len(grid)):
            _, st, ac = tel[it[k]]
            frame = {"observation.state": np.array([st[n] for n in state_names], np.float32),
                     "action": np.array([ac[n] for n in act_names], np.float32), "task": task}
            for c in cam_list:
                frame[f"observation.images.{c}"] = cv2.cvtColor(cv2.imdecode(np.frombuffer(cams[c][1][ic[c][k]], np.uint8),
                                                                             cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            ds.add_frame(frame)
        ds.save_episode(); n_ok += 1
        print(f"{p.name}: {len(grid)} frames ({len(grid) / a.fps:.1f} s)  task {task!r}", flush=True)
    if ds is None:
        raise SystemExit("no usable episodes")
    ds.finalize()
    print(json.dumps(dict(dataset=a.root, repo_id=a.repo_id, episodes=n_ok, cameras=cam_list, fps=a.fps, robot_type=robot_type)))


if __name__ == "__main__":
    main()
