from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig


@TeleoperatorConfig.register_subclass("my_arm_keyboard")
@dataclass
class MyArmKeyboardConfig(KeyboardEndEffectorTeleopConfig):
    translation_step: float = 0.002
    gripper_step: float = 0.05