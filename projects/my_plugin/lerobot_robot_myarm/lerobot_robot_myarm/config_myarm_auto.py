from dataclasses import dataclass
from pathlib import Path
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig

@TeleoperatorConfig.register_subclass("my_arm_auto")
@dataclass
class MyArmAutoConfig(KeyboardEndEffectorTeleopConfig):
    camera_name: str = "wrist"
    camera_fovy_deg: float = 60.0
    # 独立高分辨率渲染供 AprilTag PnP 使用，不改变数据集 observation 的尺寸。
    detection_image_width: int = 1280
    detection_image_height: int = 960
    # Tag 中心到 TCP 抓取/放置点的偏移，均在 base 坐标系中，单位米。
    pick_tcp_offset_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    place_tcp_offset_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clearance: float = 0.10
    position_tolerance: float = 0.005
    max_step: float = 0.002
    grasp_wait_cycles: int = 15
    release_wait_cycles: int = 15
    # 从模型中读取 base 坐标系表面高度，不使用有测量误差的 AprilTag 推桌高。
    # 取最高面作为全局保守下限；仅保护桌面可设置为 ("table_top",)。
    surface_geom_names: tuple[str, ...] = ("table_top",)
    table_safety_margin: float = 0.010

    # 首次在物体零旋转场景测得的 Tag/base 与 EE/base 姿态会持久保存。
    # 之后：目标姿态 = Rz(Tag 相对基准 yaw + 杯子偏置) @ EE 基准姿态。
    apriltag_rotation_reference_path: str = str(
        Path(__file__).resolve().parents[3] / "models" / "apriltag_rotation_reference.json"
    )
    follow_tag_yaw: bool = True
    cup_yaw_offset_deg: float = 90.0
    rotation_step_rad: float = 0.02
    orientation_tolerance_rad: float = 0.03
