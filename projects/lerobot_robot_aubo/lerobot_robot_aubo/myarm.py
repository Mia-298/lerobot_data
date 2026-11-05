from lerobot.robots import Robot
from .config_myarm import MyArmConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from pathlib import Path
import sys
import pyaubo_sdk
import numpy as np
import time
import math
def wait_arrival(robot_interface):
    max_retry_count = 5
    cnt = 0

    # 接口调用: 获取当前的运动指令 ID
    exec_id = robot_interface.getMotionControl().getExecId()

    # 等待机械臂开始运动
    while exec_id == -1:
        if cnt > max_retry_count:
            return -1
        time.sleep(0.05)
        cnt += 1
        exec_id = robot_interface.getMotionControl().getExecId()

    # 等待机械臂运动完成
    while robot_interface.getMotionControl().getExecId() != -1:
        time.sleep(0.05)

    return 0


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
        if set(self.camera_names) & {f"{name}.pos" for name in self.config.joint_names}:
            raise ValueError("相机名称不能与关节观测字段重名")
        # lerobt录制器通过cameras来获取相机名称，需要适配
        self.cameras = self.camera_names
        for name in ("image_width", "image_height"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        self.image_width = config.image_width
        self.image_height = config.image_height

        # 创建 Python 对象不等于连接仿真；模型由 connect() 加载。
        self._is_connected = False

        self.control_cycles = 5
        self.max_episode_steps = 200

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:

        
        self.robot_rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 超时
        self.robot_rpc_client.connect(self.config.robot_ip, self.config.robot_port)  # 接口调用: 连接 RPC 服务
        if self.robot_rpc_client.hasConnected():
            print("RPC客户端连接成功!")
            self.robot_rpc_client.login(self.config.robot_name, self.config.robot_password)  # 接口调用: 登录
            self.robot_name = self.robot_rpc_client.getRobotNames()[0]

            self.robot_interface = self.robot_rpc_client.getRobotInterface(self.robot_name)
            self.robot_interface.getMotionControl().setSpeedFraction(self.config.robot_speed)
            self._is_connected = True
        else:
            print("RPC客户端连接失败!")

    def disconnect(self) -> None:
        self.robot_rpc_client.logout()  # 退出登录
        self.robot_rpc_client.disconnect()  # 断开连接
        self._is_connected = False

    @property
    def is_calibrated(self) -> bool:
        # 纯仿真不需要真实电机标定。
        return True

    def calibrate(self) -> None:
        """纯仿真无需标定；保留 Robot 接口。"""
        pass

    def configure(self) -> None:
        """当前模型的执行器参数由 XML 配置，无需额外设置。"""
        pass
    # 声明观测字段包含关节值，夹爪状态，相机输出
    @property
    def observation_features(self) -> dict:
        """声明数据格式，不读取仿真，因此未连接时也能调用。"""
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
        """读取当前六个关节位置（rad）和 RGB 图像；不推进仿真。"""
        if not self.is_connected:
            raise DeviceNotConnectedError("请先调用 MyArm.connect()")
        
        # for camera_name in self.camera_names:
        #     observation[camera_name] = self.get_images(camera_name)
        # return observation

    def send_action(self, action: dict) -> dict:
        if not self.is_connected:
            raise DeviceNotConnectedError("请先调用 MyArm.connect()")
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
        q_target = np.clip(
                    q_target,
                    self.ctrl_min,
                    self.ctrl_max,
                )
        gripper = float(np.clip(gripper, 0.0, 1.0))
        # finger_target = 0.04 * gripper
        # 暂时还没写夹爪控制，
        ret =self.robot_interface.getMotionControl() \
        .moveJoint(q_target, 80 * (math.pi / 180), 60 * (math.pi / 180), 0, 0)
        ret = wait_arrival(self.robot_interface)
        if ret == 0:
            print("关节运动到路点1成功")
        else:
            raise RuntimeError(f"moveJoint 下发未成功，返回码：{ret}")
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
        # 还没有接入相机
        """读取一个已配置相机，返回独立的 H×W×3 uint8 RGB 数组。"""
        if not self.is_connected:
            raise DeviceNotConnectedError("请先调用 MyArm.connect()")
        if camera_name not in self.camera_names:
            raise ValueError(f"相机 {camera_name!r} 未启用，请检查 config.camera_names")
        image = 0
        return image
    # 用于episode之间的复位
    def reset_simulation(self) -> None:
        if not self.is_connected:
            raise RuntimeError("请先连接机器人，再复位仿真")

        # 恢复 XML 中名为 home 的 keyframe
        mujoc
    

