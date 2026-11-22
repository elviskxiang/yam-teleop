from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("yam_leader")
@dataclass
class YamLeaderConfig(TeleoperatorConfig):
    can_channel: str = "can_left"
    grav_factor: list[float] = field(default_factory=lambda: [1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
    friction: bool = False          # Coulomb friction feed-forward: lighter by hand, may drift
    gripper_keys: bool = True       # V open, B cube width, C squeeze, Z close (the leader's lead-screw gripper can't be hand-driven)
    park_on_disconnect: bool = True # ramp to the rest pose before cutting torque (closing it while raised would drop it)
    mock: bool = False              # no hardware: a leader that moves by itself
