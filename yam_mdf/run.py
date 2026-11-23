"""Run a trained MDF-lite policy on the follower (right) arm, closed loop.

  ./run_mdf.sh outputs/mdf/screw_from_rest_timestep/ckpt_best.pt [--seconds 60] [--max-vel 1.0] [--exec 8]

Every 1/hz s (10 Hz) it reads the arm and cameras, encodes the frames once, and keeps the last n_ctx steps. When the
current plan is used up (every n_exec steps) it gives the model that history and denoises the future (by default with
the schedule the model was evaluated with, e.g. Diffusion Forcing's pyramid), then executes the first n_exec actions,
interpolated to 30 Hz (receding horizon). Actions go through the yam_follower plugin, so the arm keeps its safety layer:
slow engage ramp from where it is, speed limit, joint limits, collision checks against the table / floor / left arm,
blocked-joint handling, dead-arm detection. Its page (cameras, arm state, command) is on http://127.0.0.1:<web-port>.

Before you run: the left (leader) arm at its rest pose (the collision check assumes it there, no leader is running),
the scene set up as in the recordings, a hand ready. Stop: Ctrl+C or Esc (the arm parks at rest, then torque is cut),
or --seconds runs out. Each run is saved to outputs/mdf/<run>/rollouts/<time>.npz (states, actions, plans, timings).
"""
import argparse
import logging
import signal
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from yam_mdf.policy import MDFPolicy

log = logging.getLogger("yam_mdf.run")
FPS = 30


def make_robot(cams, web_port, max_vel, mock):
    from lerobot_robot_yam import YamFollower, YamFollowerConfig
    from lerobot_robot_yam.config_yam_follower import rig_cameras
    rc = rig_cameras()
    missing = [c for c in cams if c not in rc]
    if missing:
        raise SystemExit(f"the model was trained on cameras {missing} that the rig config does not have")
    cfg = YamFollowerConfig(id="right", cameras={c: rc[c] for c in cams}, web_port=web_port, mock=mock,
                            **({"max_vel": max_vel} if max_vel else {}))
    return YamFollower(cfg)


class Runner:
    def __init__(self, policy, robot, seconds, out=None):
        self.p, self.robot, self.seconds, self.out = policy, robot, seconds, out
        self.stop = threading.Event()
        self.sub = int(round(FPS / policy.hz))                          # 30 Hz ticks per model step (3)
        n = policy.n_ctx
        self.states, self.vis, self.acts = deque(maxlen=n), deque(maxlen=n), deque(maxlen=n - 1)
        self.rec = dict(t=[], state=[], action=[], plan_t=[], plan=[], plan_ms=[])

    def observe(self, obs):
        s = np.array([obs[k] for k in self.p.state_names], np.float32)
        v = self.p.encode({c: obs[c] for c in self.p.cams})
        if not self.states:                                             # no history yet: as if it had been still
            self.states.extend([s] * self.p.n_ctx); self.vis.extend([v] * self.p.n_ctx)
            self.acts.extend([s[:7].copy()] * (self.p.n_ctx - 1))      # holding where it is (gripper target = position)
        else:
            self.states.append(s); self.vis.append(v)
        return s

    def history(self):
        return np.stack(self.states), np.stack(self.acts), np.stack(self.vis)

    def plan(self, skip=0, hist=None):
        """n_exec actions planned from the history (default: as it is now); skip=1 starts them one step after now (the
        plan was started one step early, see run)."""
        t = time.perf_counter()
        a = self.p.act_features(*(hist or self.history()), full=True)["action"][skip:skip + self.p.n_exec]
        ms = (time.perf_counter() - t) * 1000
        self.rec["plan_t"].append(time.time()); self.rec["plan"].append(a); self.rec["plan_ms"].append(ms)
        return a, ms

    def plan_async(self):
        """Start planning in the background from the history up to now; the result covers the steps after now."""
        box, hist = {}, self.history()                                   # snapshot: the loop keeps appending
        th = threading.Thread(target=lambda: box.update(r=self.plan(skip=1, hist=hist)), daemon=True)
        th.start()
        return th, box

    def send(self, a):
        self.robot.send_action({n: float(v) for n, v in zip(self.p.action_names, a)})

    def run(self):
        t0, plan, k, step, pending = time.time(), None, 0, 0, None
        log.info(f"running for {self.seconds:.0f} s: {self.p.schedule} schedule, {self.p.steps} sampling steps, "
                 f"{self.p.n_exec} actions per plan ({self.p.n_exec / self.p.hz:.1f} s)")
        nxt = time.time()
        while not self.stop.is_set() and time.time() - t0 < self.seconds:
            obs = self.robot.get_observation()                          # raises if the arm faulted
            s = self.observe(obs)
            if plan is None or k >= len(plan):
                if pending is None:                                     # first plan: synchronous
                    plan, ms = self.plan()
                else:                                                   # started one step ago, during the last action
                    t_wait = time.perf_counter()
                    pending[0].join()
                    plan, ms = pending[1]["r"]; pending = None
                    wait = (time.perf_counter() - t_wait) * 1000
                    if wait > 30:
                        log.info(f"waited {wait:.0f} ms for plan {len(self.rec['plan_ms'])} ({ms:.0f} ms to compute)")
                k = 0
                if len(self.rec["plan_ms"]) <= 2:
                    log.info(f"plan {len(self.rec['plan_ms'])}: {ms:.0f} ms")
            if k == len(plan) - 1 and pending is None:                  # plan the next chunk while this action runs
                pending = self.plan_async()
            a, a_next = plan[k], plan[k + 1] if k + 1 < len(plan) else plan[k]
            self.rec["t"].append(time.time()); self.rec["state"].append(s); self.rec["action"].append(a)
            for j in range(self.sub):                                   # interpolate this step's action to the next one
                self.send(a + (a_next - a) * (j / self.sub))
                nxt += 1.0 / FPS
                time.sleep(max(0.0, nxt - time.time()))
                if time.time() - nxt > 0.2:
                    nxt = time.time()
            self.acts.append(a)
            k += 1; step += 1
        why = "stopped" if self.stop.is_set() else f"{self.seconds:.0f} s up"
        log.info(f"{why}: {step} steps, {len(self.rec['plan_ms'])} plans, median planning {np.median(self.rec['plan_ms'] or [0]):.0f} ms")

    def save(self):
        if self.out is None or not self.rec["t"]:
            return None
        self.out.mkdir(parents=True, exist_ok=True)
        f = self.out / time.strftime("%Y%m%d_%H%M%S.npz")
        np.savez(f, **{k: np.asarray(v) for k, v in self.rec.items()}, state_names=self.p.state_names,
                 action_names=self.p.action_names, schedule=self.p.schedule, n_exec=self.p.n_exec, hz=self.p.hz)
        return f


def main():
    ap = argparse.ArgumentParser(description="run an MDF-lite policy on the follower arm")
    ap.add_argument("ckpt")
    ap.add_argument("--seconds", type=float, default=60.0, help="stop (park) after this long")
    ap.add_argument("--exec", type=int, default=None, help="actions executed per plan (default: the run's n_exec)")
    ap.add_argument("--schedule", choices=("full", "pyramid"), default=None, help="default: the run's --sample-schedule")
    ap.add_argument("--uncertainty", type=float, default=None)
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--max-vel", type=float, default=1.0, help="follower speed limit, rad/s (recording uses 1.5)")
    ap.add_argument("--web-port", type=int, default=8781, help="arm + camera page (0 = off)")
    ap.add_argument("--mock", action="store_true", help="simulated arm (cameras are still real)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    pol = MDFPolicy(a.ckpt, sample_steps=a.sample_steps, schedule=a.schedule, uncertainty=a.uncertainty, n_exec=a.exec)
    pol.encode({c: np.zeros((480, 640, 3), np.uint8) for c in pol.cams})          # load the encoder + warm up CUDA
    log.info(f"policy {a.ckpt} (step {pol.step}, {pol.cfg['noise_mode']} noise): cameras {pol.cams}")
    robot = make_robot(pol.cams, a.web_port, a.max_vel, a.mock)
    run = Runner(pol, robot, a.seconds, Path(a.ckpt).resolve().parent / "rollouts")
    signal.signal(signal.SIGINT, lambda *_: run.stop.set())
    signal.signal(signal.SIGTERM, lambda *_: run.stop.set())
    try:
        from pynput import keyboard
        keyboard.Listener(on_press=lambda k: run.stop.set() if k == keyboard.Key.esc else None, daemon=True).start()
    except Exception as e:
        log.info(f"Esc to stop unavailable ({e}): use Ctrl+C")
    robot.connect()
    try:
        run.observe(robot.get_observation())                         # history from the arm at rest; plans warm up
        run.plan()
        run.rec = {k: [] for k in run.rec}
        run.run()
    finally:
        f = run.save()
        log.info("parking the arm, then cutting torque")
        robot.disconnect()
        if f:
            log.info(f"rollout saved to {f}")


if __name__ == "__main__":
    main()
