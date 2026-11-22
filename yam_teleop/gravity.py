"""Gravity model of one YAM + linear-4310 gripper from I2RT's URDF (link masses, COMs, inertias as the URDF states them).

Drop-in for the SDK's MuJoCoKDL (robot.kdl): the SDK builds its gravity model from the MJCF; this parses
i2rt/robot_models/arm/yam/v1/yam.urdf, which includes the gripper body and both fingers. Meshes are stripped: only the
kinematic tree and the inertials matter. The two models agree to < 0.1 % on every joint. Each instance owns its MjData, so
it is not thread-safe: give every thread its own instance.
"""
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

# Per-joint multiplier on the model torque: I2RT's own tuning for the YAM (i2rt/robots/config/yam_v1.yml). The wrist pitch
# (joint 4) needs 1.2 to make up for motor torque-constant error; at 1.0 it droops.
GRAV_FACTOR = np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


def default_urdf():
    import i2rt
    return os.path.join(os.path.dirname(i2rt.__file__), "robot_models", "arm", "yam", "v1", "yam.urdf")


class UrdfGravity:
    def __init__(self, path=None):
        self.path = os.path.abspath(path or default_urdf())
        root = ET.parse(self.path).getroot()
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for e in link.findall(tag):
                    link.remove(e)
        root.insert(0, ET.fromstring('<mujoco><compiler fusestatic="false" discardvisual="true"/></mujoco>'))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        # gravity only: the URDF's joint limits would otherwise add MuJoCo's limit-constraint force to mj_inverse. An arm
        # resting on its stop reads a hair past it (joint 2 at -0.009 rad gave -30 N m instead of +1.8), the SDK's 25 N m
        # guard then raised in its server thread, and that arm went limp with frozen readings
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.vadr = np.array([self.model.joint(n).dofadr[0] for n in ARM_JOINTS])
        assert self.model.body("gripper").mass[0] > 0.3, "URDF lacks the gripper body"

    def compute_inverse_dynamics(self, q, qdot, qdotdot):
        d = self.data
        d.qpos[:] = 0.0; d.qvel[:] = 0.0; d.qacc[:] = 0.0
        d.qpos[self.qadr] = np.asarray(q, float)[:6]
        d.qvel[self.vadr] = np.broadcast_to(qdot, 6)
        d.qacc[self.vadr] = np.broadcast_to(qdotdot, 6)
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse[self.vadr].copy()

    def gravity(self, q):
        return self.compute_inverse_dynamics(q, 0.0, 0.0)

    def specs(self):
        m = self.model
        return {m.body(i).name: dict(mass=round(float(m.body_mass[i]), 6), com=[round(float(v), 6) for v in m.body_ipos[i]])
                for i in range(1, m.nbody)}
