import logging
import sys
from queue import Empty, Queue

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop
from lerobot.utils.keyboard_input import TerminalKeyListener

from .config_myarm_keyboard import MyArmKeyboardConfig


class MyArmKeyboard(KeyboardEndEffectorTeleop):
    config_class = MyArmKeyboardConfig
    name = "my_arm_keyboard"

    def __init__(self, config: MyArmKeyboardConfig):
        super().__init__(config)
        self._terminal_listener: TerminalKeyListener | None = None
        self._terminal_keys: Queue[str] = Queue()

    @property
    def is_connected(self) -> bool:
        if self._terminal_listener is not None:
            return self._terminal_listener._running
        return super().is_connected

    def connect(self, calibrate: bool = True) -> None:
        backend = self.config.input_backend.lower()
        if backend not in {"auto", "terminal", "x11"}:
            raise ValueError("input_backend 必须是 auto、terminal 或 x11")

        if backend in {"auto", "terminal"} and sys.stdin.isatty():
            self._terminal_listener = TerminalKeyListener(self._terminal_keys.put)
            self._terminal_listener.start()
            if self._terminal_listener._running:
                logging.info(
                    "终端键盘控制已启动：W/S=Y，A/D=X，E/C=Z；保持当前终端焦点。"
                )
                return
            self._terminal_listener = None

        if backend == "terminal":
            raise RuntimeError("当前标准输入不是交互终端，无法启用终端键盘控制")

        super().connect(calibrate=calibrate)

    def disconnect(self) -> None:
        if self._terminal_listener is not None:
            self._terminal_listener.stop()
            self._terminal_listener = None
            return
        super().disconnect()

    def _get_terminal_action(self) -> dict[str, float]:
        # 终端只有 key-down，没有 key-up。清空本周期积压并按最后一个方向键
        # 产生单步动作，避免松键后仍继续移动。
        keys: list[str] = []
        while True:
            try:
                keys.append(self._terminal_keys.get_nowait().lower())
            except Empty:
                break

        action = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "gripper": 1.0,
        }
        for key in keys:
            if key in {"d", "right"}:
                action["delta_x"] = 1.0
            elif key in {"a", "left"}:
                action["delta_x"] = -1.0
            elif key in {"w", "up"}:
                action["delta_y"] = 1.0
            elif key in {"s", "down"}:
                action["delta_y"] = -1.0
            elif key == "e":
                action["delta_z"] = 1.0
            elif key == "c":
                action["delta_z"] = -1.0
        return action

    def get_action(self):
        if self._terminal_listener is not None:
            return self._get_terminal_action()

        # 先执行父类方法：
        # 更新按键状态，并保留父类的夹爪按键处理
        action = super().get_action()

        keys = self.current_pressed

        # 再用自己的映射覆盖三个移动分量
        action["delta_x"] = int(keys.get("d", False)) - int(keys.get("a", False))
        action["delta_y"] = int(keys.get("w", False)) - int(keys.get("s", False))
        action["delta_z"] = int(keys.get("e", False)) - int(keys.get("c", False))

        return action
