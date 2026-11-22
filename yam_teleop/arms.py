"""Shared by the LeRobot plugins (lerobot_robot_yam, lerobot_teleoperator_yam): opening a YAM arm, the follower's safety
constants, and the leader pose the follower checks collisions against.

Both plugins live in the same lerobot-record process, so the leader publishes its joint angles here and the follower
reads them (without a leader, e.g. when a policy drives the follower, the left arm is assumed at rest).
"""
import os
import threading
import time

import numpy as np

from yam_teleop.gravity import GRAV_FACTOR, UrdfGravity
from yam_teleop.leader_follower import (ENGAGE_VEL, FLOOR_Z, GRIP_VEL, J_HI, J_LO, KD, KP, TRACK_STOP, VFF_ALPHA,
                                        Collision, MockLeader)

JOINTS = [f"joint{i}" for i in range(1, 7)]
GRIPPER = "gripper"
DEFAULT_SCENE = os.path.join(os.environ.get("RUBIKS", os.path.expanduser("~/Desktop/experiments/rubiks_cube/robotics-rubiks-cube")),
                             "poc", "generated", "scene.xml")

_leader = {"q": None, "t": 0.0}
_leader_lock = threading.Lock()


def publish_leader(q):
    with _leader_lock:
        _leader["q"], _leader["t"] = np.array(q, float), time.time()


_grip = {"target": 1.0}


def set_grip_target(g):
    """Gripper target (0 closed .. 1 open), from the keyboard (leader plugin) or the web page (follower plugin)."""
    _grip["target"] = float(min(1.0, max(0.0, g)))


def grip_target():
    return _grip["target"]


def leader_q(max_age=0.5):
    """Latest leader joint angles, or the rest pose (zeros) if no leader is running / its data is stale."""
    with _leader_lock:
        if _leader["q"] is not None and time.time() - _leader["t"] < max_age:
            return _leader["q"].copy()
    return np.zeros(6)


def open_arm(side, follower, grav_factor=GRAV_FACTOR, friction=False, grip_N=13.0, mock=False):
    """Connect one YAM: the SDK checks every motor, calibrates the gripper (it opens and closes) and starts the arm
    floating under gravity compensation from the URDF model (all six joints, incl. the wrist)."""
    if mock:
        import yam_io as Y
        return Y.MockArm(np.r_[np.zeros(6), 1.0]) if follower else MockLeader()
    import yam_io as Y
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType
    r = get_yam_robot(channel=f"can_{side}", arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=True,
                      gravity_comp_factor=np.asarray(grav_factor, float), enable_auto_recovery=True, gripper_kp=None, gripper_kd=0.1)
    r.kdl = UrdfGravity()
    r.temp_record_flag = True
    r.use_coulomb_friction = bool(friction) and not follower
    if follower:
        # the stock gripper servo burst the cube twice; LeashGripper caps the squeeze force (robotics-rubiks-cube yam_io)
        r._gripper_force_limiter = Y.LeashGripper(r, grip_N, Y.GripperMap(), 0.031); r._limit_gripper_force = grip_N
        r.update_kp_kd(np.r_[KP, r._kp[6]], np.r_[KD, r._kd[6]])
    return r


def chain_dead(arm):
    """The SDK stopped this arm's control loop (a motor stopped answering on CAN): its readings are frozen."""
    return hasattr(arm, "motor_chain") and not arm.motor_chain.running


def park(arm, q_from, seconds_min=5.0, speed=0.3):
    """Ramp an arm (q6 + gripper) slowly to the rest pose (all joints 0), gripper as it is, then float."""
    q_from = np.asarray(q_from, float)
    seconds = max(seconds_min, float(np.abs(q_from[:6]).max()) / speed)
    n = int(seconds * 100)
    for i in range(1, n + 1):
        u = i / n; u = u * u * (3 - 2 * u)
        c = q_from.copy(); c[:6] = (1 - u) * q_from[:6]
        arm.command_joint_pos(c)
        time.sleep(seconds / n)
    time.sleep(0.5)
    arm.enter_gravity_comp_idle()


__all__ = ["JOINTS", "GRIPPER", "DEFAULT_SCENE", "publish_leader", "leader_q", "open_arm", "chain_dead", "park", "Collision",
           "UrdfGravity", "GRAV_FACTOR", "KP", "KD", "J_LO", "J_HI", "ENGAGE_VEL", "TRACK_STOP", "GRIP_VEL", "VFF_ALPHA", "FLOOR_Z"]
