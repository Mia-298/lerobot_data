from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop

from .config_myarm_keyboard import MyArmKeyboardConfig
from .myarm import MyArm

class MyArmKeyboard(KeyboardEndEffectorTeleop):
    config_class = MyArmKeyboardConfig
    name = "my_arm_keyboard"

    def __init__(self, config: MyArmKeyboardConfig):
        super().__init__(config)
    def get_action(self):
        # 先执行父类方法：
        # 更新按键状态，并保留父类的夹爪按键处理
        action = super().get_action()

        keys = self.current_pressed

        # 再用自己的映射覆盖三个移动分量
        action["delta_x"] = int(keys.get("d", False)) - int(keys.get("a", False))
        action["delta_y"] = int(keys.get("w", False)) - int(keys.get("s", False))
        action["delta_z"] = int(keys.get("e", False)) - int(keys.get("c", False))

        return action
