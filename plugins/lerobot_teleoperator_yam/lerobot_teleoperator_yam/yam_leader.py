"""YAM leader (left arm) as a LeRobot Teleoperator: it floats under URDF gravity compensation and you move it by hand.

Action: joint1-6 = the leader's joint angles (the arms are mounted identically 0.61 m apart, so the follower copying them
makes the same motion in the world), gripper = a target set from the keyboard (V open, B cube width, C squeeze, Z close;
LeRobot's own keys: right arrow ends the episode, left arrow re-records it, Esc stops).
The leader's pose is also published for the follower's collision check (yam_teleop.arms.publish_leader).
"""
import logging
import os

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator

from yam_teleop import arms as A

from .config_yam_leader import YamLeaderConfig

log = logging.getLogger(__name__)
NAMES = A.JOINTS + [A.GRIPPER]
GRIP_KEYS = {"v": 1.0, "b": 0.62, "c": 0.45, "z": 0.0}


class YamLeader(Teleoperator):
    config_class = YamLeaderConfig
    name = "yam_leader"

    def __init__(self, config: YamLeaderConfig):
        super().__init__(config)
        self.config = config
        self.arm = None
        self.grip_target = 1.0
        self._listener = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{n}.pos": float for n in NAMES}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.arm is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        self.arm = A.open_arm(self.config.can_channel.removeprefix("can_"), follower=False,
                              grav_factor=self.config.grav_factor, friction=self.config.friction, mock=self.config.mock)
        if self.config.gripper_keys and os.environ.get("DISPLAY"):
            try:
                from pynput import keyboard

                def on_press(key):
                    ch = getattr(key, "char", None)
                    if ch and ch.lower() in GRIP_KEYS:
                        self.grip_target = GRIP_KEYS[ch.lower()]
                        log.info(f"gripper target {self.grip_target:.2f}")
                self._listener = keyboard.Listener(on_press=on_press)
                self._listener.start()
            except Exception as e:
                log.warning(f"gripper keys unavailable: {e}")
        log.info(f"{self} connected: floating, move it by hand; gripper keys V open, B cube, C squeeze, Z close")

    def get_action(self) -> dict[str, float]:
        if A.chain_dead(self.arm):
            raise RuntimeError(f"{self}: the leader (left) arm stopped responding (a motor dropped off CAN)")
        q = np.asarray(self.arm.get_observations()["joint_pos"], float)
        A.publish_leader(q)
        return {**{f"{n}.pos": float(v) for n, v in zip(A.JOINTS, q)}, f"{A.GRIPPER}.pos": float(self.grip_target)}

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        if self.arm is not None:
            try:
                if self.config.park_on_disconnect and not A.chain_dead(self.arm):
                    o = self.arm.get_observations()
                    log.info(f"{self}: parking at the rest pose before cutting torque")
                    A.park(self.arm, np.r_[o["joint_pos"], np.ravel(o["gripper_pos"])[:1]])
            finally:
                if hasattr(self.arm, "close"):
                    self.arm.close()
                self.arm = None
