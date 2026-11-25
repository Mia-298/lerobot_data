"""teleoperator子类"""
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop

from .config_myarm_auto import MyArmAutoConfig
from .detect_apriltags import AprilTagDetection, detect_apriltags, detection_to_T
from .task_tags import resolve_task_tags
from .myarm import get_site_pose_in_base
from .grasp_safety import TableClearance
from .grasp_orientation import (
    AprilTagRotationReference, CUP_TAG_IDS, ROTATION_TARGET_KEY, rotation_error,
)
import logging
import numpy as np
import time

from .pick_place_state import (
    PickPlaceState,
    PickPlaceStateMachine,
)

class MyArmAuto(KeyboardEndEffectorTeleop):
    config_class = MyArmAutoConfig
    name = "my_arm_auto"
    

    def __init__(self, config):
        super().__init__(config)
        if not np.isfinite(config.orientation_tolerance_rad) or config.orientation_tolerance_rad <= 0:
            raise ValueError("姿态到位容差必须为正有限数")
        if not np.isfinite(config.cup_yaw_offset_deg):
            raise ValueError("杯子旋转角必须为有限数")
        self.rotation_reference = AprilTagRotationReference(config.apriltag_rotation_reference_path)
        self.grasp_rotation_base = None
        self._orientation_active = False
        self.controller = PickPlaceStateMachine(
            clearance=config.clearance,
            tolerance=config.position_tolerance,
            max_step=config.max_step,
        )
        self.object_baseT = None
        self.plate_baseT = None
        self.object_detection = None
        self.plate_detection = None
        self.targets_locked = False
        self._state_ticks = 0

    @property
    def state(self):
        return self.controller.state

    def bind_robot(self, robot):
        targets = resolve_task_tags(robot.config.task_id, robot.config.tasklist)
        self.robot = robot
        # LeRobot 在 robot.connect() 之前绑定，此时 model/data 仍为 None。
        self.table_clearance = None
        self._clearance_model = None
        self.task_id = robot.config.task_id
        self.task_text = robot.config.tasklist[self.task_id - 1]
        self.object_tag_id = targets.object_tag_id
        self.plate_tag_id = targets.plate_tag_id
        self.controller.reset()
        self.grasp_rotation_base = None
        self._orientation_active = False
        self.targets_locked = False
        self._state_ticks = 0

    def _ensure_table_clearance(self):
        robot = getattr(self, "robot", None)
        if robot is None or robot.model is None or robot.data is None:
            raise RuntimeError("自动抓取需要先绑定机器人并完成 robot.connect()，模型尚未就绪")
        if self.table_clearance is None or self._clearance_model is not robot.model:
            self.table_clearance = TableClearance(
                robot.model, robot.config.ee_name, robot.config.gripper_joint_names,
                self.config.surface_geom_names, margin=self.config.table_safety_margin,
            )
            self._clearance_model = robot.model
        return self.table_clearance

    def _camera_matrix(self, image: np.ndarray) -> np.ndarray:
        """根据图像尺寸和配置的视场角计算相机内参；不读取相机位姿。"""
        fovy = float(self.config.camera_fovy_deg)
        if not 0.0 < fovy < 180.0:
            raise ValueError("camera_fovy_deg 必须在 0 到 180 度之间")
        # 必须使用本次检测图像的真实尺寸，不能沿用低分辨率 observation 内参。
        height, width = image.shape[:2]
        fy = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        fx = fy
        return np.array(
            [[fx, 0.0, (width - 1) / 2.0],
             [0.0, fy, (height - 1) / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _base_camera_pose_from_ee(self) -> np.ndarray:
        """使用 T_base_ee 和 YAML 的 T_ee_camera，不读取仿真相机位姿。"""
        T_base_ee = get_site_pose_in_base(
            self.robot.model,
            self.robot.data,
            self.robot.config.ee_name,
        )
        T_ee_camera = np.asarray(self.robot.config.T_G_C, dtype=float)
        if T_ee_camera.shape != (4, 4) or not np.all(np.isfinite(T_ee_camera)):
            raise ValueError("YAML 手眼矩阵 T_G_C 必须是有限的 4×4 矩阵")
        return T_base_ee @ T_ee_camera

    def get_target_detections(self, timeout_s=10.0):
        self._ensure_table_clearance()
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            image = self.robot.get_images(
                self.config.camera_name,
                hide_tags=False,
                width=self.config.detection_image_width,
                height=self.config.detection_image_height,
            )

            detections = {
                item.tag_id: item
                for item in detect_apriltags(image)
            }

            object_detection = detections.get(self.object_tag_id)
            plate_detection = detections.get(self.plate_tag_id)

            if object_detection is not None and plate_detection is not None:
                self.object_detection = object_detection
                self.plate_detection = plate_detection
    
                T_base_camera = self._base_camera_pose_from_ee()
                camera_matrix = self._camera_matrix(image)
                distortion = np.asarray(
                    self.robot.config.distortion_coefficients,
                    dtype=np.float32,
                )
    
                # 同一张图、同一手眼标定：记录所有可见任务 Tag 的 base 姿态。
                # 不把 Tag 当前姿态直接当夹爪姿态；保留首次测量为零参考。
                tag_base_poses = {
                    tag_id: T_base_camera @ detection_to_T(detection, camera_matrix, distortion)
                    for tag_id, detection in detections.items() if tag_id in range(6)
                }
                self.object_baseT = tag_base_poses[self.object_tag_id]
                self.plate_baseT = tag_base_poses[self.plate_tag_id]

                pick_offset = np.asarray(self.config.pick_tcp_offset_base, dtype=float)
                place_offset = np.asarray(self.config.place_tcp_offset_base, dtype=float)
                if pick_offset.shape != (3,) or place_offset.shape != (3,):
                    raise ValueError("TCP 偏移必须是长度为 3 的向量")
                if not np.all(np.isfinite(pick_offset)) or not np.all(np.isfinite(place_offset)):
                    raise ValueError("TCP 偏移必须是有限数值")
    
                T_base_ee = get_site_pose_in_base(
                    self.robot.model, self.robot.data, self.robot.config.ee_name,
                )
                self.rotation_reference.capture_missing(
                    {tag_id: pose[:3, :3] for tag_id, pose in tag_base_poses.items()},
                    T_base_ee[:3, :3],
                )
                self.grasp_rotation_base, relative_yaw = self.rotation_reference.grasp_rotation(
                    self.object_tag_id, self.object_baseT[:3, :3],
                    cup_yaw_offset_deg=self.config.cup_yaw_offset_deg,
                    follow_tag_yaw=self.config.follow_tag_yaw,
                )
                logging.info(
                    "抓取姿态（base Z）：Tag %s，相对基准 yaw=%.2f°，杯子偏置=%.2f°；基准文件=%s",
                    self.object_tag_id, np.rad2deg(relative_yaw),
                    self.config.cup_yaw_offset_deg if self.object_tag_id in CUP_TAG_IDS else 0.0,
                    self.rotation_reference.path,
                )
                logging.info("Tag %s 当前 R_base_tag=%s；基准 R_base_tag=%s；目标 R_base_ee=%s",
                             self.object_tag_id, self.object_baseT[:3, :3].tolist(),
                             self.rotation_reference.tags[str(self.object_tag_id)].tolist(),
                             self.grasp_rotation_base.tolist())
                target_T_base_ee = T_base_ee.copy()
                target_T_base_ee[:3, :3] = self.grasp_rotation_base
                minimum_tcp_z = max(
                    self.table_clearance.minimum_tcp_z(T_base_ee, self.robot.config.T_G_F),
                    self.table_clearance.minimum_tcp_z(target_T_base_ee, self.robot.config.T_G_F),
                )
                self.controller.set_targets(
                    pick_tcp=self.object_baseT[:3, 3] + pick_offset,
                    place_tcp=self.plate_baseT[:3, 3] + place_offset,
                    obj_height=self.robot.config.OBJECT_TAG_TO_BOTTOM_M[self.object_tag_id],
                    minimum_tcp_z=minimum_tcp_z,
                )

                targets = self.controller.targets
                print(
                    f"AprilTag 检测：{image.shape[1]}×{image.shape[0]}，"
                    f"物体 Tag {self.object_tag_id} 边长像素="
                    f"{object_detection.side_lengths.round(1).tolist()}"
                )
                print(
                    f"限高诊断（base Z，米）：Tag 抓取目标="
                    f"{(self.object_baseT[:3, 3] + pick_offset)[2]:.4f}，"
                    f"安全下限={minimum_tcp_z:.4f}，"
                    f"实际抓取目标={targets[PickPlaceState.MOVE_TO_PICK][2]:.4f}，"
                    f"实际放置目标={targets[PickPlaceState.MOVE_TO_PLACE][2]:.4f}"
                )

                self.pick_position = targets[
                    PickPlaceState.MOVE_TO_PICK
                ].copy()

                self.above_pick_position = targets[
                    PickPlaceState.MOVE_ABOVE_PICK
                ].copy()

                self.place_position = targets[
                    PickPlaceState.MOVE_TO_PLACE
                ].copy()

                self.above_place_position = targets[
                    PickPlaceState.MOVE_ABOVE_PLACE
                ].copy()

                self.targets_locked = True
                return object_detection, plate_detection

            time.sleep(0.02)

        return None, None

    # 数据的存储和状态更新在get_target_detections，controller.step中已实现
    def get_action(self):
        self._ensure_table_clearance()
        # 1. episode 开始时检测目标并初始化状态机。
        if self.state == PickPlaceState.WAIT_TARGET:
            object_detection, plate_detection = self.get_target_detections()
            if object_detection is None or plate_detection is None:
                return self._hold_action()
            self._state_ticks = 0

        # 2. 每个控制周期读取当前末端位置。
        current_T_base_ee = get_site_pose_in_base(
            self.robot.model,
            self.robot.data,
            self.robot.config.ee_name,
        )
        current_T_base_fc = current_T_base_ee@self.robot.config.T_G_F
        current_tcp = current_T_base_fc[:3, 3]

        state_before = self.state
        minimum_tcp_z = self.table_clearance.minimum_tcp_z(
            current_T_base_ee, self.robot.config.T_G_F,
        )
        # 安全下限会随实测姿态改变。和 controller.step 使用同一个实时目标，
        # 不能用检测时保存的 above_pick_position，否则抬高目标后可能永远无法转腕。
        above_pick = self.controller.targets.get(PickPlaceState.MOVE_ABOVE_PICK)
        if above_pick is not None:
            above_pick = above_pick.copy()
            above_pick[2] = max(above_pick[2], minimum_tcp_z)
        # 到达抓取点正上方后再转腕，避免在接近物体/把手时横扫。
        # 用实测 EE 姿态判断是否到位，不能用已下发的 IK 目标代替反馈。
        if (state_before == PickPlaceState.MOVE_ABOVE_PICK
                and self.grasp_rotation_base is not None
                and above_pick is not None
                and np.linalg.norm(current_tcp - above_pick) <= self.config.position_tolerance):
            self._orientation_active = True
        orientation_ready = (
            self.grasp_rotation_base is not None
            and self._orientation_active
            and rotation_error(current_T_base_ee[:3, :3], self.grasp_rotation_base)
                <= self.config.orientation_tolerance_rad
        )

        # 3. 暂时使用等待周期作为夹爪动作完成的条件。
        grasp_confirmed = (
            state_before == PickPlaceState.GRASP
            and self._state_ticks >= self.config.grasp_wait_cycles
        )
        release_confirmed = (
            state_before == PickPlaceState.RELEASE
            and self._state_ticks >= self.config.release_wait_cycles
        )

        # 4. 状态机判断到位、切换状态，并生成本周期动作。
        action = self.controller.step(
            current_tcp,
            gripper_open=(self.controller.gripper_target == 1.0),
            grasp_confirmed=grasp_confirmed,
            release_confirmed=release_confirmed,
            orientation_ready=orientation_ready,
            minimum_tcp_z=minimum_tcp_z,
        )

        if self._orientation_active:
            # 绝对姿态元数据仅交给自定义 IK processor，最终输出仍只有关节/夹爪动作。
            action[ROTATION_TARGET_KEY] = self.grasp_rotation_base.copy()

        # 5. 记录当前状态持续了多少个周期。
        if self.state != state_before:
            print(f"状态切换：{state_before.name} -> {self.state.name}")
            self._state_ticks = 0
        else:
            self._state_ticks += 1

        return action

    @staticmethod
    def _hold_action():
        return {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
        }
