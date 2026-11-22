from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig

_BY_PATH = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:"


def rig_cameras() -> dict[str, CameraConfig]:
    """The four rig cameras at 640x480 MJPG 30 fps, addressed by USB port path / serial (videoN numbers move on replug).
    Names match the datasets converted from the earlier MCAP episodes (yam_teleop.to_lerobot)."""
    cam = lambda p: OpenCVCameraConfig(index_or_path=p, fps=30, width=640, height=480, fourcc="MJPG")
    return {
        "c922": cam("/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_A579FDAF-video-index0"),
        "d415_tripod_top": cam(_BY_PATH + "12:1.3-video-index0"),
        "microdia_left_side": cam(_BY_PATH + "2.1:1.0-video-index0"),
        "microdia_right_base_plate": cam(_BY_PATH + "10:1.0-video-index0"),
    }


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YamFollowerConfig(RobotConfig):
    can_channel: str = "can_right"
    cameras: dict[str, CameraConfig] = field(default_factory=rig_cameras)
    max_vel: float = 1.5            # follower joint speed limit (rad/s); the leader may move faster, the follower trails
    vff: float = 1.0                # velocity feed-forward gain (0 = position only)
    grip_force_N: float = 13.0      # LeashGripper squeeze cap
    grav_factor: list[float] = field(default_factory=lambda: [1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
    scene_xml: str = ""             # rig model for collision checks; default: $RUBIKS/poc/generated/scene.xml
    park_on_disconnect: bool = True # ramp to the rest pose before cutting torque
    web_port: int = 8780            # session page (cameras, state, episode + gripper buttons); 0 = off
    mock: bool = False              # no hardware: a simulated arm (for testing the pipeline)
