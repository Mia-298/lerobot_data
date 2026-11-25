from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

from lerobot.lerobot_types import RobotAction, TransitionKey
from .config_myarm import MyArmConfig
from .myarm import FK_learn, IK_learn, get_joint_initial_info, get_T_joint_to_site
from .grasp_safety import TableClearance
from .grasp_orientation import ROTATION_TARGET_KEY, rotation_matrix, step_rotation

class KeyboardIKActionStep(RobotActionProcessorStep):
    def __init__(
            self,
            joint_names,
            transforms_relative,
            axes_relative,
            T6ee,
            q_min,
            q_max,
            table_clearance=None,
            tcp_offset_ee=None,
            rotation_step_rad=0.02,):
        super().__init__()
        self.joint_names = list(joint_names)

        self.transforms_relative = [
            transform.copy()
            for transform in transforms_relative
        ]

        self.axes_relative = [
            axis.copy()
            for axis in axes_relative
        ]

        self.T6ee = T6ee.copy()
        self.q_min = np.asarray(q_min, dtype=float).copy()
        self.q_max = np.asarray(q_max, dtype=float).copy()
        self.table_clearance = table_clearance
        self.tcp_offset_ee = np.zeros(3) if tcp_offset_ee is None else np.asarray(tcp_offset_ee, dtype=float).copy()
        if self.tcp_offset_ee.shape != (3,) or not np.all(np.isfinite(self.tcp_offset_ee)):
            raise ValueError("TCP 偏移必须为三个有限数")
        if not np.isfinite(rotation_step_rad) or rotation_step_rad <= 0:
            raise ValueError("单周期旋转步长必须为正有限数")
        self.rotation_step_rad = rotation_step_rad
        # 仿真测试用：限制每个控制周期的关节目标变化量，单位 rad
        self.max_joint_step = 0.1
        # 保持上一次下发的目标，避免把重力造成的位置误差变成新的目标。
        self._joint_target: np.ndarray | None = None
        self._gripper_target: float | None = None
        self._previous_q: np.ndarray | None = None
        # 仿真回到 keyframe 时关节位置会瞬间跳变，用它识别外部 reset。
        self._reset_jump_threshold = 2.0 * self.max_joint_step

    def _ee_pose(self, q):
        T_joints, _ = FK_learn.Tbn(q, self.transforms_relative, self.axes_relative)
        return T_joints[-1] @ self.T6ee

    def _safe_joint_segment(self, start, end):
        # 离散检查关节插值；不是连续碰撞证明，也不包含全机械臂几何。
        for fraction in np.linspace(0.0, 1.0, 21):
            try:
                self.table_clearance.require_safe_ee(
                    self._ee_pose(start + fraction * (end - start))
                )
            except RuntimeError:
                return False
        return True

    def _sync_targets(self, q_current: np.ndarray, gripper_current: float) -> None:
        reset_detected = (
            self._previous_q is not None
            and np.max(np.abs(q_current - self._previous_q)) > self._reset_jump_threshold
        )
        if self._joint_target is None or reset_detected:
            self._joint_target = q_current.copy()
            self._gripper_target = gripper_current
        self._previous_q = q_current.copy()

    def action(self, action: RobotAction) -> RobotAction:
        obs = self.transition[TransitionKey.OBSERVATION]
        q_current = np.array(
            [
                obs[f"{name}.pos"]
                for name in self.joint_names
            ],
            dtype=float,
        )
        gripper_current = float(obs["gripper.pos"])
        if not np.all(np.isfinite(q_current)) or not np.isfinite(gripper_current):
            raise ValueError("关节反馈与夹爪反馈必须是有限数")
        self._sync_targets(q_current, gripper_current)
        assert self._joint_target is not None
        assert self._gripper_target is not None

        # 必须先检查再处理零位移/夹合；不能继续下发已低于下限的旧目标。
        if self.table_clearance is not None:
            self.table_clearance.require_safe_ee(self._ee_pose(q_current))
            self.table_clearance.require_safe_ee(self._ee_pose(self._joint_target))
            if not self._safe_joint_segment(q_current, self._joint_target):
                raise RuntimeError("反馈到旧目标的关节插值低于夹爪安全下限，停止自动下发")

        # 自动状态机可下发归一化夹爪绝对目标，即使本周期没有 XYZ 移动。
        if "gripper.pos" in action:
            gripper_target = float(action["gripper.pos"])
            if not np.isfinite(gripper_target) or not 0.0 <= gripper_target <= 1.0:
                raise ValueError("gripper.pos 必须在 0～1 之间")
            self._gripper_target = gripper_target

        movement = np.array(
            [
                action["delta_x"],
                action["delta_y"],
                action["delta_z"],
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(movement)):
            raise ValueError("XYZ 位移动作必须是有限数")
        rotation_target = None
        if ROTATION_TARGET_KEY in action:
            rotation_target = rotation_matrix(action[ROTATION_TARGET_KEY])
        if not np.any(movement) and rotation_target is None:
            processed_action = {
                f"{name}.pos": float(self._joint_target[i])
                for i, name in enumerate(self.joint_names)
            }
            processed_action["gripper.pos"] = self._gripper_target
            return processed_action

        print("执行到我自己写的 action() 了")
        print("收到动作：", action)
        T_ee = self._ee_pose(self._joint_target)
        rotation_current = T_ee[:3, :3]
        rotation_next = rotation_current if rotation_target is None else step_rotation(
            rotation_current, rotation_target, self.rotation_step_rad,
        )
        # 绕 TCP 转动，不绕 attachment_site 平移 TCP；兼容非轴向的标定偏移。
        tcp_current = T_ee[:3, 3] + rotation_current @ self.tcp_offset_ee
        delta = 0.002 * movement
        pos_target = tcp_current + delta - rotation_next @ self.tcp_offset_ee
        if self.table_clearance is not None:
            # 目标从旧命令累积，不能只限制带滞后反馈的状态机高度。
            # 留 1mm 数值缓冲；候选 IK 仍须按它自己的真实姿态复查。
            pos_target[2] = max(
                pos_target[2],
                self.table_clearance.minimum_ee_z(rotation_next) + 0.001,
            )
        rpy_target = Rotation.from_matrix(rotation_next).as_euler("xyz")
        q_target, success, iterations = IK_learn.inverse_kinematics(
            q_init=self._joint_target,
            pos_target=pos_target,
            rpy_target=rpy_target,
            transforms_relative=self.transforms_relative,
            axes_relative=self.axes_relative,
            T6ee=self.T6ee,
            pos_tolerance=0.0001,
            rot_tolerance=0.001,
        )
        q_target = np.asarray(q_target, dtype=float)

        valid = (
            success
            and q_target.shape == q_current.shape
            and np.all(np.isfinite(q_target))
            and np.all(q_target >= self.q_min)
            and np.all(q_target <= self.q_max)
            and np.all(np.abs(q_target - self._joint_target) <= self.max_joint_step)
        )
        if valid and self.table_clearance is not None:
            valid = (
                self._safe_joint_segment(self._joint_target, q_target)
                and self._safe_joint_segment(q_current, q_target)
            )
        if not success:
            print("IK求解失败,本次不移动")
            q_target = self._joint_target.copy()
        elif not valid:
            print("目标超出测试限制，本次移动被限制")
            q_target = self._joint_target.copy()
        else:
            self._joint_target = q_target.copy()

        print("IK 是否成功：", success, "迭代次数：", iterations)
        processed_action = {
            f"{name}.pos": float(q_target[i])
            for i, name in enumerate(self.joint_names)
        }
        processed_action["gripper.pos"] = self._gripper_target
        print("处理后：", processed_action)
        return processed_action

    def reset(self) -> None:
        self._joint_target = None
        self._gripper_target = None
        self._previous_q = None

    def transform_features(self, features):
        # 暂时不改变特征声明
        return features


def make_myarm_processor(robot_cfg: MyArmConfig, *, auto_config=None):
    # 这里只读取模型、计算固定参数，不启动另一套控制循环
    xml_path = Path(robot_cfg.xml_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    _, transforms_relative, axes_relative = get_joint_initial_info(
        model,
        robot_cfg.joint_names,
    )

    T6ee = get_T_joint_to_site(
        model,
        robot_cfg.joint_names[-1],
        robot_cfg.ee_name,
    )

    actuator_ids = np.array([
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in robot_cfg.actuator_names
    ], dtype=int)

    if np.any(actuator_ids < 0):
        raise ValueError("找不到机械臂执行器，请检查配置中的名称与 XML")
    tcp_offset_ee = np.zeros(3)
    table_clearance = None
    if auto_config is not None:
        # 工厂在 robot.connect() 前执行，不能读取尚未加载的 T_G_F 默认单位矩阵。
        calibration_path = Path(robot_cfg.handeye_calibration_path).expanduser()
        calibration = yaml.safe_load(calibration_path.read_text())
        T_ee_tcp = np.asarray(calibration["fc"]["transformation_matrix"], dtype=float)
        if T_ee_tcp.shape != (4, 4) or not np.all(np.isfinite(T_ee_tcp)):
            raise ValueError("YAML TCP 标定必须是有限的 4×4 矩阵")
        rotation_matrix(T_ee_tcp[:3, :3])
        if not np.allclose(T_ee_tcp[3], [0, 0, 0, 1]):
            raise ValueError("YAML TCP 标定必须是齐次变换")
        tcp_offset_ee = T_ee_tcp[:3, 3]
        table_clearance = TableClearance(
            model, robot_cfg.ee_name, robot_cfg.gripper_joint_names,
            auto_config.surface_geom_names, margin=auto_config.table_safety_margin,
        )
    # 在record中没有把robot传入，所以需要从同一份 XML 和配置读取静态 IK 参数
    ik_step = KeyboardIKActionStep(
        joint_names=robot_cfg.joint_names,
        transforms_relative=transforms_relative,
        axes_relative=axes_relative,
        T6ee=T6ee,
        q_min=model.actuator_ctrlrange[actuator_ids, 0].copy(),
        q_max=model.actuator_ctrlrange[actuator_ids, 1].copy(),
        table_clearance=table_clearance,
        tcp_offset_ee=tcp_offset_ee,
        rotation_step_rad=auto_config.rotation_step_rad if auto_config is not None else 0.02,
    )
    # 这里是返回一个流水线对象，里面包含一个step对象
    return RobotProcessorPipeline(
        steps=[ik_step],
        # 这一步会把action和observation
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
