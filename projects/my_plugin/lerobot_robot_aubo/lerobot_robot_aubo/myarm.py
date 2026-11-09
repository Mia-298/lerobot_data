import time

import numpy as np
import pyaubo_sdk
from camsdk_py import Camera
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_myarm import MyArmConfig


def wait_arrival(
    robot_interface,
    *,
    start_timeout_s: float,
    motion_timeout_s: float,
) -> None:
    """Wait until an accepted move command finishes."""
    motion_control = robot_interface.getMotionControl()
    start_deadline = time.monotonic() + start_timeout_s

    exec_id = motion_control.getExecId()
    while exec_id == -1:
        if time.monotonic() >= start_deadline:
            # The target may already equal the measured pose, in which case the
            # controller can accept the command without creating an execution ID.
            return
        time.sleep(0.05)
        exec_id = motion_control.getExecId()

    finish_deadline = time.monotonic() + motion_timeout_s
    while motion_control.getExecId() != -1:
        if time.monotonic() >= finish_deadline:
            raise TimeoutError(f"机械臂运动超过 {motion_timeout_s:.1f} 秒仍未完成")
        time.sleep(0.05)


def _wait_for_robot_mode(robot_state, expected_mode, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        current_mode = robot_state.getRobotModeType()
        if current_mode == expected_mode:
            return
        if current_mode == pyaubo_sdk.RobotModeType.Error:
            raise RuntimeError("AUBO 进入 Error 状态，请先在示教器上排除故障")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"等待 AUBO 状态 {expected_mode} 超时，当前状态：{current_mode}"
            )
        time.sleep(0.1)


class MyArm(Robot):
    config_class = MyArmConfig
    name = "my_arm"

    def __init__(self, config: MyArmConfig):
        # 父类初始化 id、标定目录等通用属性，只调用一次。
        super().__init__(config)
        self.config = config
        self.robot_rpc_client = pyaubo_sdk.RpcClient()

        # 观测格式在连接前就必须确定，供 LeRobot 创建数据集字段。
        if not isinstance(config.camera_names, (list, tuple)):
            raise ValueError("camera_names 必须是相机名称列表，例如 ['front', 'wrist']")
        self.camera_names = tuple(config.camera_names)

        if any(not isinstance(name, str) or not name.strip() for name in self.camera_names):
            raise ValueError("每个相机名称必须是非空字符串")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names 不能包含重复名称")
        if len(self.camera_names) > 1:
            raise ValueError("当前 CamSDK 适配器只支持一台相机")
        if set(self.camera_names) & {f"{name}.pos" for name in self.config.joint_names}:
            raise ValueError("相机名称不能与关节观测字段重名")

        self.camera = Camera() if self.camera_names else None
        self.cameras = {
            name: self.camera
            for name in self.camera_names
        }
        for name in ("image_width", "image_height"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        self.image_width = config.image_width
        self.image_height = config.image_height

        # 创建 Python 对象不等于连接设备；网络连接由 connect() 建立。
        self._is_connected = False
        self.robot_interface = None
        self.ctrl_min = None
        self.ctrl_max = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已连接")

        self.robot_rpc_client.setRequestTimeout(self.config.rpc_timeout_ms)
        self.robot_rpc_client.connect(self.config.robot_ip, self.config.robot_port)
        if not self.robot_rpc_client.hasConnected():
            raise ConnectionError(
                f"无法连接 AUBO RPC {self.config.robot_ip}:{self.config.robot_port}"
            )

        try:
            self.robot_rpc_client.login(
                self.config.robot_name,
                self.config.robot_password,
            )
            if not self.robot_rpc_client.hasLogined():
                raise PermissionError("AUBO RPC 登录失败，请检查用户名和密码")

            robot_names = self.robot_rpc_client.getRobotNames()
            if not robot_names:
                raise RuntimeError("AUBO RPC 未返回任何机器人名称")

            self.robot_name = robot_names[0]
            self.robot_interface = self.robot_rpc_client.getRobotInterface(self.robot_name)
            motion_control = self.robot_interface.getMotionControl()
            robot_config = self.robot_interface.getRobotConfig()
            motion_control.setSpeedFraction(self.config.robot_speed)

            self.ctrl_max = np.asarray(
                robot_config.getJointMaxPositions(), dtype=np.float64
            )
            self.ctrl_min = np.asarray(
                robot_config.getJointMinPositions(), dtype=np.float64
            )
            expected_shape = (len(self.config.joint_names),)
            if self.ctrl_min.shape != expected_shape or self.ctrl_max.shape != expected_shape:
                raise RuntimeError(
                    "AUBO 返回的关节限位数量与 joint_names 不一致："
                    f"min={self.ctrl_min.shape}, max={self.ctrl_max.shape}"
                )

            if self.camera is not None:
                self.camera.open(
                    self.config.camera_ip,
                    discover_timeout_ms=2000,
                )
            self._is_connected = True
        except Exception:
            self._cleanup_connections()
            raise

    def disconnect(self) -> None:
        self._cleanup_connections()

    def _cleanup_connections(self) -> None:
        if self.camera is not None and self.camera.is_open:
            self.camera.close()
        if self.robot_rpc_client.hasLogined():
            self.robot_rpc_client.logout()
        if self.robot_rpc_client.hasConnected():
            self.robot_rpc_client.disconnect()
        self.robot_interface = None
        self.ctrl_min = None
        self.ctrl_max = None
        self._is_connected = False

    @property
    def is_calibrated(self) -> bool:
        # AUBO 的零位和标定由控制器管理。
        return True

    def calibrate(self) -> None:
        """AUBO 的标定由控制器管理；保留 Robot 接口。"""
        pass

    def configure(self) -> None:
        """当前 AUBO 参数由控制器配置，无需额外设置。"""
        pass

    @property
    def robot_mode(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("请先调用 MyArm.connect()")
        return self.robot_interface.getRobotState().getRobotModeType()

    def power_on_and_start(self) -> None:
        """显式上电并松开制动器，使机械臂进入 Running 状态。"""
        if not self.is_connected:
            raise DeviceNotConnectedError("请先调用 MyArm.connect()")

        robot_state = self.robot_interface.getRobotState()
        robot_manage = self.robot_interface.getRobotManage()
        mode = robot_state.getRobotModeType()
        if mode == pyaubo_sdk.RobotModeType.Running:
            return

        if mode == pyaubo_sdk.RobotModeType.PowerOff:
            result = robot_manage.poweron()
            if result != 0:
                raise RuntimeError(f"AUBO 上电失败，返回码：{result}")

        _wait_for_robot_mode(
            robot_state,
            pyaubo_sdk.RobotModeType.Idle,
            self.config.power_on_timeout_s,
        )

        result = robot_manage.startup()
        if result != 0:
            raise RuntimeError(f"AUBO 松刹车/启动失败，返回码：{result}")
        _wait_for_robot_mode(
            robot_state,
            pyaubo_sdk.RobotModeType.Running,
            self.config.startup_timeout_s,
        )

    # 声明观测字段包含关节值，夹爪状态，相机输出
    @property
    def observation_features(self) -> dict:
        """声明数据格式，不读取设备，因此未连接时也能调用。"""
        # 存入每个关节值的名称和类型float
        features = {f"{name}.pos": float for name in self.config.joint_names}
        features["gripper.pos"] = float
        # 存入每个观测图像的名称和类型
        features.update({
            name: (self.image_height, self.image_width, 3)
            for name in self.camera_names
        })
        return features
    # smolvla并不限制action的类型，这里设置的动作是6个关节值+一个夹爪值
    @property
    def action_features(self) -> dict:
        features = {
            f"{name}.pos": float
            for name in self.config.joint_names
        }

        features["gripper.pos"] = float

        return features

    def get_observation(self) -> dict:
        """读取当前六个关节位置，单位为 rad。"""
        if not self.is_connected:
            raise DeviceNotConnectedError("请先调用 MyArm.connect()")

        joint_positions = (
            self.robot_interface
            .getRobotState()
            .getJointPositions()
        )

        if len(joint_positions) != len(self.config.joint_names):
            raise RuntimeError(
                f"关节数量异常：期望 {len(self.config.joint_names)}，"
                f"实际 {len(joint_positions)}"
            )

        observation = {
            f"{name}.pos": float(position)
            for name, position in zip(
                self.config.joint_names,
                joint_positions,
                strict=True,
            )
        }

        # 暂未接入夹爪状态,先输入0
        observation["gripper.pos"] = 0.0
        observation.update({
            name: self.get_images(name)
            for name in self.camera_names
        })
        return observation

    def send_action(self, action: dict) -> dict:
        if not self.is_connected:
            raise DeviceNotConnectedError("请先调用 MyArm.connect()")
        mode = self.robot_mode
        if mode != pyaubo_sdk.RobotModeType.Running:
            raise RuntimeError(
                f"AUBO 当前状态为 {mode}，不能运动；"
                "请先在确认现场安全后调用 power_on_and_start()"
            )
        expected_keys = set(self.action_features)
        received_keys = set(action)
        # 对字典使用set会得到key的集合，不保留value
        if received_keys != expected_keys:
            missing = expected_keys - received_keys
            extra = received_keys - expected_keys
            raise ValueError(
                f"动作字段不匹配，缺少：{missing}，多余：{extra}"
            )
        # 取出六关节值
        q_target = np.array(
            [action[f"{name}.pos"] for name in self.config.joint_names],
            dtype=np.float64,
        )

        gripper = float(action["gripper.pos"])

        if not np.all(np.isfinite(q_target)):
            raise ValueError("机械臂动作不能包含 NaN 或无穷大")

        if not np.isfinite(gripper):
            raise ValueError("夹爪动作不能是 NaN 或无穷大")
        q_target = np.clip(q_target, self.ctrl_min, self.ctrl_max)
        gripper = float(np.clip(gripper, 0.0, 1.0))

        result = self.robot_interface.getMotionControl().moveJoint(
            q_target.tolist(),
            self.config.move_speed_rad_s,
            self.config.move_accel_rad_s2,
            0,
            0,
        )
        if result != 0:
            raise RuntimeError(f"moveJoint 下发失败，AUBO 返回码：{result}")

        wait_arrival(
            self.robot_interface,
            start_timeout_s=self.config.motion_start_timeout_s,
            motion_timeout_s=self.config.motion_timeout_s,
        )
        sent_action = {
            f"{name}.pos": float(value)
            for name, value in zip(
                self.config.joint_names,
                q_target,
                strict=True,
            )
        }
        sent_action["gripper.pos"] = gripper
        return sent_action

    def get_images(self, camera_name: str) -> np.ndarray:
        """读取一个已配置相机，返回独立的 H×W×3 uint8 RGB 数组。"""
        if not self.is_connected:
            raise DeviceNotConnectedError("请先调用 MyArm.connect()")
        if camera_name not in self.camera_names:
            raise ValueError(f"相机 {camera_name!r} 未启用，请检查 config.camera_names")
        if self.camera is None:
            raise RuntimeError("相机未配置")

        frame = self.camera.capture()
        image = frame["rgb"]
        if image is None:
            raise RuntimeError("CamSDK 没有返回 RGB 图像")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"CamSDK RGB 数据格式异常：{image.shape}, {image.dtype}")

        if image.shape[:2] != (self.image_height, self.image_width):
            source_height, source_width = image.shape[:2]
            rows = np.linspace(
                0, source_height - 1, self.image_height, dtype=np.intp
            )
            columns = np.linspace(
                0, source_width - 1, self.image_width, dtype=np.intp
            )
            image = image[rows][:, columns]

        return np.ascontiguousarray(image)

