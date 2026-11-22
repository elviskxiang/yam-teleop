"""YAM follower (right arm) as a LeRobot Robot.

lerobot-record calls get_observation / send_action at the dataset rate (30 Hz). A 100 Hz thread in here does the actual
control, with the same safety layer as yam_teleop.leader_follower:
  * the arm starts floating; the first action engages it with a 0.35 rad/s ramp from where it is;
  * targets are clipped to the joint limits and slewed at <= max_vel rad/s, with a velocity feed-forward;
  * a target the rig model says hits the table, the floor or the left arm (pose from the leader plugin, or rest), or that
    puts the grasp point below the table top, is not sent;
  * if the arm falls > 0.3 rad behind its command it holds for 1 s, then re-engages slowly;
  * if the arm's motor chain dies (a motor off CAN) it raises, so lerobot-record stops instead of recording frozen data.
Observation: joint1-6 + gripper position, velocity and effort (torque beyond the URDF gravity model; the gripper motor's
effort), plus the cameras. Action: joint1-6 + gripper position targets.
"""
import logging
import threading
import time
from functools import cached_property

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot

from yam_teleop import arms as A

from .config_yam_follower import YamFollowerConfig

log = logging.getLogger(__name__)
DT = 0.01
NAMES = A.JOINTS + [A.GRIPPER]
YIELD_REJOIN = 0.05             # rad: a yielded wrist joint rejoins once the leader's is this close to it


class YamFollower(Robot):
    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.arm = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._target = None             # (7,) latest action, clipped
        self.web = None
        self.cmd = None                 # (7,) what is being commanded now
        self.vff = np.zeros(6)
        self.mode = "floating"
        self.blocked = None
        self.fault = None
        self._hold_until = 0.0
        self._yield = set()             # wrist joints (index 3-5) giving way after being blocked
        self._state = None
        self._factor = np.asarray(config.grav_factor, float)

    # ---- features
    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return {f"{n}.{k}": float for k in ("pos", "vel", "effort") for n in NAMES}

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {c: (self.config.cameras[c].height, self.config.cameras[c].width, 3) for c in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{n}.pos": float for n in NAMES}

    @property
    def is_connected(self) -> bool:
        return self.arm is not None and all(c.is_connected for c in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True                     # the SDK calibrates the gripper on connect; joints are absolute encoders

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ---- connection
    def connect(self, calibrate: bool = True) -> None:
        if "c922" in self.config.cameras:   # keep auto-exposure from halving the C922 to 15 fps (resets on replug)
            try:
                from yam_teleop.cam_wall import V4L2_CID_EXPOSURE_AUTO_PRIORITY, set_v4l2_ctrl
                set_v4l2_ctrl(str(self.config.cameras["c922"].index_or_path), V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)
            except Exception as e:
                log.warning(f"C922 exposure priority not set: {e}")
        for cam in self.cameras.values():
            cam.connect()
        self.col = A.Collision(self.config.scene_xml or A.DEFAULT_SCENE)
        self.gmodel = A.UrdfGravity()
        self.arm = A.open_arm(self.config.can_channel.removeprefix("can_"), follower=True, grav_factor=self._factor,
                              grip_N=self.config.grip_force_N, mock=self.config.mock)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="yam_follower_100hz")
        self._thread.start()
        t0 = time.time()
        while self._state is None and time.time() - t0 < 2.0:
            time.sleep(0.01)
        self.web = None
        if self.config.web_port:
            try:
                from yam_teleop.web import WebPage
                self.web = WebPage(self, self.config.web_port); self.web.start()
            except Exception as e:
                log.warning(f"session page not started: {e}")
        log.info(f"{self} connected: floating until the first action")

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.arm is not None:
            try:
                if self.config.park_on_disconnect and not A.chain_dead(self.arm):
                    o = self.arm.get_observations()
                    q = self.cmd if self.cmd is not None else np.r_[o["joint_pos"], np.ravel(o["gripper_pos"])[:1]]
                    log.info(f"{self}: parking at the rest pose before cutting torque")
                    A.park(self.arm, q)
            finally:
                if hasattr(self.arm, "close"):
                    self.arm.close()
                self.arm = None
        if getattr(self, "web", None) is not None:
            self.web.stop()
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

    # ---- control (100 Hz thread)
    def _loop(self):
        nxt = time.time()
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._tick()
            except Exception as e:
                self.fault = self.fault or f"control loop error: {type(e).__name__}: {e}"
                log.exception("yam_follower control loop")
            nxt += DT
            time.sleep(max(0.0, nxt - time.time()))
            if time.time() - nxt > 0.1:
                nxt = time.time()

    def _send(self, cmd, vff):
        a = self.arm
        if not hasattr(a, "_command_lock"):          # mock arm
            a.command_joint_pos(cmd); return
        from i2rt.robots.motor_chain_robot import JointCommands
        c = JointCommands.init_all_zero(len(a.motor_chain))
        c.pos = a._joint_to_motor_pos(a.remapper.to_robot_joint_pos_space(a._clip_robot_joint_pos_command(cmd)))
        c.vel = a._joint_to_motor_vel(a.remapper.to_robot_joint_vel_space(np.r_[vff, 0.0]))
        c.kp, c.kd = a._kp, a._kd
        with a._command_lock:                        # swapped in whole (the SDK's command_joint_state is not atomic)
            a._commands = c

    def _tick(self):
        if A.chain_dead(self.arm):
            self.fault = self.fault or "the follower (right) arm stopped responding (a motor dropped off CAN)"
            return
        o = self.arm.get_observations()
        q = np.asarray(o["joint_pos"], float); g = float(np.ravel(o["gripper_pos"])[0])
        eff = np.asarray(o["joint_eff"], float) - self.gmodel.gravity(q) * self._factor
        self._state = dict(pos=np.r_[q, g], vel=np.r_[np.asarray(o["joint_vel"], float), np.ravel(o.get("gripper_vel", [0.0]))[0]],
                           effort=np.r_[eff, np.ravel(o.get("gripper_eff", [0.0]))[0]])
        tgt = self._target
        if tgt is None:
            return
        now = time.time()
        g_tgt = float(np.clip(tgt[6], 0, 1))
        grip_step = lambda c: c + float(np.clip(g_tgt - c, -A.GRIP_VEL * DT, A.GRIP_VEL * DT))
        if self.mode == "floating" or (self.mode == "held" and now >= self._hold_until):
            self.cmd = np.r_[q, g if self.cmd is None or self.mode == "floating" else self.cmd[6]]
            self.vff = np.zeros(6); self.mode = "engaging"; self._yield = set()
            self.arm.command_joint_pos(self.cmd)
            return
        if self.mode == "held":                      # joints hold until _hold_until; the gripper still obeys (to let go)
            self.cmd[6] = grip_step(self.cmd[6])
            self.arm.command_joint_pos(self.cmd)
            return
        lq = A.leader_q()
        target = np.clip(tgt[:6], A.J_LO, A.J_HI)
        # a blocked wrist joint gives way (commanded where it is) until the leader's comes back to it
        for j in sorted(self._yield):
            if abs(target[j] - q[j]) < YIELD_REJOIN:
                self._yield.discard(j); log.info(f"{self}: joint {j + 1} rejoined")
        vel = A.ENGAGE_VEL if self.mode == "engaging" else self.config.max_vel
        q_new = self.cmd[:6] + np.clip(target - self.cmd[:6], -vel * DT, vel * DT)
        for j in self._yield:
            q_new[j] = q[j]
        hit = self.col.check(q_new, lq)
        self.blocked = hit
        if hit:
            q_new = self.cmd[:6]
        v_raw = (q_new - self.cmd[:6]) / DT
        self.vff = np.zeros(6) if hit else (1 - A.VFF_ALPHA) * self.vff + A.VFF_ALPHA * v_raw
        for j in self._yield:
            self.vff[j] = 0.0
        self.cmd = np.r_[q_new, grip_step(self.cmd[6])]
        self._send(self.cmd, self.vff * self.config.vff)
        if self.mode == "engaging" and np.abs(np.delete(target - q_new, list(self._yield))).max(initial=0.0) < 0.02:
            self.mode = "following"; log.info(f"{self}: following")
        lag = np.abs(self.cmd[:6] - q)
        wrist = [j for j in (3, 4, 5) if lag[j] > A.TRACK_STOP and j not in self._yield]
        for j in wrist:                              # wrist: give way, the rest of the arm keeps following
            self._yield.add(j); self.cmd[j] = q[j]; self.vff[j] = 0.0
            log.warning(f"{self}: joint {j + 1} blocked ({lag[j]:.2f} rad behind): it gives way until the leader's joint "
                        f"{j + 1} comes back to it; open the gripper if it is clamped on something")
        if lag[:3].max() > A.TRACK_STOP:             # shoulder / elbow: hold the whole arm, then re-engage slowly
            self.cmd = np.r_[q, self.cmd[6]]; self.vff = np.zeros(6)
            self.arm.command_joint_pos(self.cmd)
            self.mode, self._hold_until = "held", now + 1.0
            log.warning(f"{self}: joint {int(lag[:3].argmax()) + 1} fell {lag[:3].max():.2f} rad behind (touching something?); "
                        f"holding 1 s, then re-engaging slowly")
        if self._yield and not self.blocked:
            self.blocked = (f"wrist joint {', '.join(str(j + 1) for j in sorted(self._yield))} blocked and giving way: bring the "
                            f"leader's wrist back to it (or open the gripper)")

    # ---- LeRobot API
    def get_observation(self) -> dict:
        if self.fault:
            raise RuntimeError(f"{self}: {self.fault}")
        with self._lock:
            s = self._state
        if s is None:
            raise RuntimeError(f"{self}: no arm state yet")
        obs = {}
        for k in ("pos", "vel", "effort"):
            for n, v in zip(NAMES, s[k]):
                obs[f"{n}.{k}"] = float(v)
        for name, cam in self.cameras.items():
            obs[name] = cam.read_latest()
        return obs

    def send_action(self, action: dict) -> dict:
        if self.fault:
            raise RuntimeError(f"{self}: {self.fault}")
        a = np.array([float(action[f"{n}.pos"]) for n in NAMES])
        a[:6] = np.clip(a[:6], A.J_LO, A.J_HI); a[6] = float(np.clip(a[6], 0.0, 1.0))
        with self._lock:
            self._target = a
        return {f"{n}.pos": float(v) for n, v in zip(NAMES, a)}
