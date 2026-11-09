from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig


@TeleoperatorConfig.register_subclass("my_arm_keyboard")
@dataclass
class MyArmKeyboardConfig(KeyboardEndEffectorTeleopConfig):
    translation_step: float = 0.002
    gripper_step: float = 0.05
    # auto 优先读取当前终端，stdin 不是 TTY 时再回退到 pynput/X11。
    input_backend: str = "auto"
