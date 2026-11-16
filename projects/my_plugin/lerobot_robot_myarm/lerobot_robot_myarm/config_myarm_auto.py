from dataclasses import dataclass
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig

@TeleoperatorConfig.register_subclass("my_arm_auto")
@dataclass
class MyArmAutoConfig(KeyboardEndEffectorTeleopConfig):
    camera_name: str = "wrist"
    camera_fovy_deg: float = 60.0
    # Tag 中心到 TCP 抓取/放置点的偏移，均在 base 坐标系中，单位米。
    pick_tcp_offset_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    place_tcp_offset_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clearance: float = 0.10
    position_tolerance: float = 0.005
    max_step: float = 0.002
    grasp_wait_cycles: int = 15
    release_wait_cycles: int = 15
