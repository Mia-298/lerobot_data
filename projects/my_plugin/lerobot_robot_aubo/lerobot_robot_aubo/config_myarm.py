from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots import RobotConfig

# 当前目录布局：projects/lerobot_robot_myarm/lerobot_robot_myarm/config_myarm.py
# 默认模型位于 projects/models/ur5e.xml，也可以创建配置时传入其他绝对路径。
DEFAULT_XML_PATH = Path(__file__).resolve().parents[2] / "models" / "ur5e.xml"


@RobotConfig.register_subclass("my_arm")
@dataclass
class MyArmConfig(RobotConfig):
    xml_path: str = str(DEFAULT_XML_PATH)
    reset_noise: float = 0.0  # 预留给后续 reset 使用，单位 rad；connect 始终恢复 home。

    # 相机名称必须与 XML 中的 <camera name="..."> 一致；[] 表示只读关节。
    camera_names: list[str] = field(default_factory=lambda: ["front"])
    image_width: int = 640
    image_height: int = 480
    joint_names: list[str] = field(default_factory=lambda: [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ])

    actuator_names: list[str] = field(default_factory=lambda: [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ])
    gripper_joint_names: list[str] = field(default_factory=lambda: [
        "left_finger_joint",
        "right_finger_joint",
    ])

    gripper_actuator_names: list[str] = field(default_factory=lambda: [
        "left_gripper",
        "right_gripper",
    ])
    ee_name: str = "attachment_site"
    robot_ip: str = "192.168.193.130"
    robot_port: int = 30004
    robot_name: str = "rob1"
    robot_password: str = "aubo123"
    robot_speed: float = 0.75
    rpc_timeout_ms: int = 1000
    camera_ip: str = "10.10.10.18"

    move_speed_rad_s: float = 1.3962634015954636  # 80 deg/s
    move_accel_rad_s2: float = 1.0471975511965976  # 60 deg/s^2
    power_on_timeout_s: float = 30.0
    startup_timeout_s: float = 30.0
    motion_start_timeout_s: float = 1.0
    motion_timeout_s: float = 30.0


