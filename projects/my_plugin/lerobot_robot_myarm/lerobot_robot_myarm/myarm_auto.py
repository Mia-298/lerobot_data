"""teleoperator子类"""
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop

from .config_myarm_auto import MyArmAutoConfig
from .detect_apriltags import AprilTagDetection, detect_apriltags, detection_to_T
from .task_tags import resolve_task_tags
from .myarm import get_site_pose_in_base
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
        self.task_id = robot.config.task_id
        self.task_text = robot.config.tasklist[self.task_id - 1]
        self.object_tag_id = targets.object_tag_id
        self.plate_tag_id = targets.plate_tag_id
        self.controller.reset()
        self.targets_locked = False
        self._state_ticks = 0

    def _camera_matrix(self) -> np.ndarray:
        """根据图像尺寸和配置的视场角计算相机内参；不读取相机位姿。"""
        fovy = float(self.config.camera_fovy_deg)
        if not 0.0 < fovy < 180.0:
            raise ValueError("camera_fovy_deg 必须在 0 到 180 度之间")
        width = self.robot.image_width
        height = self.robot.image_height
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
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            image = self.robot.get_images(
                self.config.camera_name,
                hide_tags=False,
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
                camera_matrix = self._camera_matrix()
                distortion = np.asarray(
                    self.robot.config.distortion_coefficients,
                    dtype=np.float32,
                )
    
                T_camera_object = detection_to_T(
                    object_detection,
                    camera_matrix,
                    distortion,
                )
                T_camera_plate = detection_to_T(
                    plate_detection,
                    camera_matrix,
                    distortion,
                )
    
                self.object_baseT = T_base_camera @ T_camera_object
                self.plate_baseT = T_base_camera @ T_camera_plate
    
                pick_offset = np.asarray(self.config.pick_tcp_offset_base, dtype=float)
                place_offset = np.asarray(self.config.place_tcp_offset_base, dtype=float)
                if pick_offset.shape != (3,) or place_offset.shape != (3,):
                    raise ValueError("TCP 偏移必须是长度为 3 的向量")
                if not np.all(np.isfinite(pick_offset)) or not np.all(np.isfinite(place_offset)):
                    raise ValueError("TCP 偏移必须是有限数值")
    
                self.controller.set_targets(
                    pick_tcp=self.object_baseT[:3, 3] + pick_offset,
                    place_tcp=self.plate_baseT[:3, 3] + place_offset,
                    obj_height=self.robot.config.OBJECT_TAG_TO_BOTTOM_M[self.object_tag_id]
                )

                targets = self.controller.targets

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
        )

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
