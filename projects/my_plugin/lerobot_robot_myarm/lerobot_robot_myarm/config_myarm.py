"""robot_config子类"""
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots import RobotConfig
from .task_tags import default_tasklist
import numpy as np

# 当前目录布局：projects/lerobot_robot_myarm/lerobot_robot_myarm/config_myarm.py
# 默认模型位于 projects/models/ur5e.xml，也可以创建配置时传入其他绝对路径。
DEFAULT_XML_PATH = Path(__file__).resolve().parents[3] / "models" / "world.xml"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "models" / "wrist_camera_to_gripper.yaml"

@RobotConfig.register_subclass("my_arm")
@dataclass
class MyArmConfig(RobotConfig):
    xml_path: str = str(DEFAULT_XML_PATH)
    # LeRobot calibration_dir 是目录；手眼标定文件单独配置，不能混用。
    calibration_dir: Path | None = None
    handeye_calibration_path: str = str(DEFAULT_CALIBRATION_PATH)
    reset_noise: float = 0.0  # 预留给后续 reset 使用，单位 rad；connect 始终恢复 home。

    # 通用机器人默认关闭；record_my_arm.py 默认开启两种随机化。
    # 随机化发生在 connect，当前录制入口每段 episode 都会重新 connect。
    randomize_objects: bool = False
    randomize_lighting: bool = False
    scene_randomization_seed: int | None = None
    scene_randomization_index: int = 0  # 固定 seed 时，每段递增以生成可复现序列。
    light_brightness_range: list[float] = field(default_factory=lambda: [0.75, 1.15])
    # 相对 XML 灯光位置，各轴在 [-jitter, +jitter] 内均匀采样，单位米。
    light_position_jitter: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.3])

    # 相机名称必须与 XML 中的 <camera name="..."> 一致；[] 表示只读关节。
    camera_names: list[str] = field(default_factory=lambda: ["front", "wrist"])
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
    # 物体高度
    OBJECT_TAG_TO_BOTTOM_M = {
        0: 0.0465,  # 小方块：本体高 0.045
        1: 0.0760,  # 大方块：本体高 0.075
        4: 0.0760,  # 小杯子：本体高 0.075
        5: 0.1010,  # 大杯子：本体高 0.100
    }

    tasklist: list[str] = field(default_factory=default_tasklist)
    task_id: int = 0 # 0 表示尚未选择；有效任务编号为 1～16
    T_G_C: np.ndarray = field(default_factory=lambda: np.eye(4), init=False, repr=False)
    T_G_F: np.ndarray = field(default_factory=lambda: np.eye(4), init=False, repr=False)
    distortion_coefficients: np.ndarray = field(
        default_factory=lambda: np.zeros((5, 1), dtype=np.float32),
        init=False,
        repr=False,
    )
    episode_done = False
