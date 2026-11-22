"""Write one MCAP file per teleop episode: all camera video + leader / follower state + the follower command (the action).

  source env.sh && $REC_PY -m yam_teleop.record [--out data/episodes/<dataset>]

Follows the episode state of yam_teleop.leader_follower (:8780 /telemetry/stream, 100 Hz) and takes video from cam_wall
(:8090 /streams, all cameras over one connection, each frame stamped with its capture time). Nothing is written between
episodes. A file is written as episode_NNNNN.mcap.part and renamed when the episode is saved; a discarded episode's file
is deleted. Numbering continues after the highest episode already in --out, so restarting either program never overwrites.

Topics (log_time = capture / sample time, ns since epoch):
  /cam/<name>/image/compressed  sensor_msgs/msg/CompressedImage  JPEG as captured (640x480)
  /follower/joint_states        sensor_msgs/msg/JointState       joint1-6 + gripper, measured
  /leader/joint_states          sensor_msgs/msg/JointState
  /action                       sensor_msgs/msg/JointState       follower command (position only): what a policy outputs
  /telemetry                    JSON    the full sample (efforts, temperatures, gravity torque, blocked reason, mode)
Metadata 'episode' (task, index, start time, cameras) and 'rig' (gains, gravity model, link specs). Chunks are
uncompressed (JPEG dominates) so any mcap reader opens them without lz4 / zstd.
"""
import argparse
import http.client
import json
import os
import queue
import re
import signal
import threading
import time
from pathlib import Path

from mcap.writer import CompressionType, Writer
from mcap_ros2._dynamic import serialize_dynamic

HEADER = ("================================================================================\nMSG: std_msgs/Header\n"
          "builtin_interfaces/Time stamp\nstring frame_id\n"
          "================================================================================\nMSG: builtin_interfaces/Time\n"
          "int32 sec\nuint32 nanosec")
IMG_DEF = "std_msgs/Header header\nstring format\nuint8[] data\n" + HEADER
JS_DEF = "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n" + HEADER
NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper"]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def stamp(t):
    return {"sec": int(t // 1_000_000_000), "nanosec": int(t % 1_000_000_000)}


def get_json(host, port, path):
    c = http.client.HTTPConnection(host, port, timeout=3)
    try:
        c.request("GET", path); r = c.getresponse()
        return json.loads(r.read()) if r.status == 200 else None
    finally:
        c.close()


class Episode:
    def __init__(self, path, meta):
        self.path, self.part = path, path.with_suffix(".mcap.part")
        self.f = open(self.part, "wb")
        self.w = Writer(self.f, compression=CompressionType.NONE, chunk_size=4 * 1024 * 1024)
        self.w.start(profile="", library="yam_teleop.record")
        self.s_img = self.w.register_schema("sensor_msgs/msg/CompressedImage", "ros2msg", IMG_DEF.encode())
        self.s_js = self.w.register_schema("sensor_msgs/msg/JointState", "ros2msg", JS_DEF.encode())
        self.e_img = serialize_dynamic("sensor_msgs/msg/CompressedImage", IMG_DEF)["sensor_msgs/msg/CompressedImage"]
        self.e_js = serialize_dynamic("sensor_msgs/msg/JointState", JS_DEF)["sensor_msgs/msg/JointState"]
        self.ch, self.seq, self.n = {}, {}, {}
        for k, v in meta.items():
            self.w.add_metadata(k, {kk: vv if isinstance(vv, str) else json.dumps(vv) for kk, vv in v.items()})

    def _chan(self, topic, schema, enc):
        if topic not in self.ch:
            self.ch[topic] = self.w.register_channel(topic=topic, message_encoding=enc, schema_id=schema); self.seq[topic] = 0
        self.seq[topic] += 1; self.n[topic] = self.n.get(topic, 0) + 1
        return self.ch[topic], self.seq[topic]

    def put(self, topic, t, data, schema, enc):
        ch, seq = self._chan(topic, schema, enc)
        self.w.add_message(channel_id=ch, log_time=t, publish_time=t, sequence=seq, data=data)

    def image(self, cam, t, jpg):
        self.put(f"/cam/{cam}/image/compressed", t,
                 self.e_img({"header": {"stamp": stamp(t), "frame_id": cam}, "format": "jpeg", "data": jpg}), self.s_img, "cdr")

    def joints(self, topic, t, pos, vel=None, eff=None, frame=""):
        self.put(topic, t, self.e_js({"header": {"stamp": stamp(t), "frame_id": frame}, "name": NAMES, "position": pos,
                                      "velocity": vel or [], "effort": eff or []}), self.s_js, "cdr")

    def json(self, topic, t, obj):
        self.put(topic, t, json.dumps(obj).encode(), 0, "json")

    def close(self, keep):
        self.w.finish(); self.f.close()
        if keep:
            os.replace(self.part, self.path)
        else:
            os.remove(self.part)
        return self.n


def camera_reader(host, port, q, stop, names):
    """cam_wall /streams: every camera on one connection; parts tagged X-Cam (index) and X-Timestamp-Ns."""
    while not stop.is_set():
        try:
            c = http.client.HTTPConnection(host, port, timeout=5); c.request("GET", "/streams"); r = c.getresponse()
            while not stop.is_set():
                line = r.fp.readline()
                if not line:
                    break
                if not line.startswith(b"--f"):
                    continue
                h = {}
                while True:
                    x = r.fp.readline().strip()
                    if not x:
                        break
                    k, _, v = x.decode().partition(":"); h[k.strip().lower()] = v.strip()
                jpg = r.fp.read(int(h["content-length"]))
                i = int(h.get("x-cam", -1))
                if 0 <= i < len(names):
                    q.put(("img", int(h.get("x-timestamp-ns") or time.time_ns()), (names[i], jpg)))
            c.close()
        except (OSError, http.client.HTTPException, ValueError, KeyError):
            pass
        stop.wait(1.0)


def telemetry_reader(host, port, q, stop):
    while not stop.is_set():
        try:
            c = http.client.HTTPConnection(host, port, timeout=5); c.request("GET", "/telemetry/stream?hz=100"); r = c.getresponse()
            while not stop.is_set():
                line = r.fp.readline()
                if not line:
                    break
                s = json.loads(line); q.put(("tel", int(s["t_ns"]), s))
            c.close()
        except (OSError, http.client.HTTPException, ValueError, KeyError):
            pass
        q.put(("lost", time.time_ns(), None))
        stop.wait(1.0)


def next_index(out):
    nums = [int(m.group(1)) for p in out.glob("episode_*.mcap*") if (m := re.match(r"episode_(\d+)\.mcap", p.name))]
    return max(nums, default=-1) + 1


def main():
    ap = argparse.ArgumentParser(description="record teleop episodes to MCAP")
    ap.add_argument("--out", default="data/episodes/default")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cam-port", type=int, default=8090)
    ap.add_argument("--teleop-port", type=int, default=8780)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.mcap.part"):                       # an episode cut off by a crash is incomplete: never train on it
        print(f"removing incomplete {p.name}", flush=True); p.unlink()

    while True:
        try:
            cams = [slug(c["name"]) for c in get_json(a.host, a.cam_port, "/status")]
            rig = get_json(a.host, a.teleop_port, "/config") or {}
            break
        except OSError:
            print(f"waiting for cam_wall (:{a.cam_port}) and leader_follower (:{a.teleop_port}) ...", flush=True); time.sleep(2)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set()); signal.signal(signal.SIGTERM, lambda *_: stop.set())
    q = queue.Queue(maxsize=10000)
    threading.Thread(target=camera_reader, args=(a.host, a.cam_port, q, stop, cams), daemon=True).start()
    threading.Thread(target=telemetry_reader, args=(a.host, a.teleop_port, q, stop), daemon=True).start()
    print(f"recorder ready: cameras {cams}; episodes go to {out}/ (next episode_{next_index(out):05d})", flush=True)

    ep, ep_id, t_last_tel = None, None, 0

    def finish(how):
        nonlocal ep, ep_id
        n = ep.close(keep=(how == "saved"))
        frames = {k.split("/")[2]: v for k, v in n.items() if k.startswith("/cam/")}
        print(f"episode {ep.path.name}: {how}  ({n.get('/action', 0)} actions, frames {frames})"
              if how == "saved" else f"episode {ep.path.name}: {how} (deleted)", flush=True)
        ep, ep_id = None, None

    try:
        while not stop.is_set():
            try:
                kind, t, x = q.get(timeout=0.2)
            except queue.Empty:
                if ep is not None and time.time_ns() - t_last_tel > 2e9:
                    finish("discarded (teleop telemetry lost for 2 s)")
                continue
            if kind == "lost":
                if ep is not None:
                    finish("discarded (teleop connection lost)")
                continue
            if kind == "tel":
                t_last_tel = time.time_ns()
                e = x["episode"]
                if ep is not None and (e["state"] != "recording" or e["id"] != ep_id):
                    last = e.get("last") or {}
                    finish(last.get("how", "saved") if last.get("id") == ep_id else "saved")
                if ep is None and e["state"] == "recording":
                    ep_id = e["id"]
                    path = out / f"episode_{next_index(out):05d}.mcap"
                    ep = Episode(path, {"episode": dict(task=e["task"], teleop_episode_id=ep_id, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                        cameras=cams, fps_cameras="30"), "rig": rig})
                    print(f"recording {path.name}: {e['task']!r}", flush=True)
                if ep is not None and x.get("action") is not None:
                    F, L = x["follower"], x["leader"]
                    ep.joints("/follower/joint_states", t, F["joint_pos"] + F["gripper_pos"][:1],
                              F["joint_vel"] + (F["gripper_vel"] or [0.0])[:1], F["joint_eff"] + (F["gripper_eff"] or [0.0])[:1], "follower")
                    ep.joints("/leader/joint_states", t, L["joint_pos"] + (L["gripper_pos"] or [0.0])[:1], L["joint_vel"], L["joint_eff"], "leader")
                    ep.joints("/action", t, x["action"], frame="follower_cmd")
                    ep.json("/telemetry", t, x)
            elif kind == "img" and ep is not None:
                ep.image(x[0], t, x[1])
    finally:
        if ep is not None:
            finish("discarded (recorder stopped mid-episode)")


if __name__ == "__main__":
    main()
