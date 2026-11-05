from lerobot.robots import Robot
from .config_myarm import MyArmConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from pathlib import Path
import sys

from mia_kinematics import FK_learn, IK_learn

import numpy as np
import mujoco
import time
import mujoco.viewer



def get_site_pose_in_base(
    model,
    data,
    site_name,
    base_body_name="base",
):
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    base_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_name,
    )

    if site_id == -1:
        raise ValueError(
            f"不存在 site：{site_name}"
        )

    if base_id == -1:
        raise ValueError(
            f"不存在 body：{base_body_name}"
        )

    p_world_ee = (
        data.site_xpos[site_id]
        .copy()
    )

    R_world_ee = (
        data.site_xmat[site_id]
        .reshape(3, 3)
        .copy()
    )

    p_world_base = (
        data.xpos[base_id]
        .copy()
    )

    R_world_base = (
        data.xmat[base_id]
        .reshape(3, 3)
        .copy()
    )

    p_base_ee = (
        R_world_base.T
        @ (p_world_ee - p_world_base)
    )

    R_base_ee = (
        R_world_base.T
        @ R_world_ee
    )

    T_base_ee = np.eye(4)
    T_base_ee[:3, :3] = R_base_ee
    T_base_ee[:3, 3] = p_base_ee

    return T_base_ee
def get_T_joints_from_mujoco(
    model,
    data,
    joint_names,
    base_body_name="base",
):
    """
    返回当前状态下，每个 joint 相对于 base 的齐次变换矩阵。

    Returns
    -------
    T_joints : ndarray, shape (n, 4, 4)
        T_joints[i] 表示 base -> joint_i。
    """

    # 根据当前 qpos 更新所有笛卡尔量
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_name,
    )

    if base_id == -1:
        raise ValueError(
            f"模型中不存在 body：{base_body_name}"
        )

    # base 在 world 下的位姿
    p_world_base = (
        data.xpos[base_id]
        .copy()
    )

    R_world_base = (
        data.xmat[base_id]
        .reshape(3, 3)
        .copy()
    )

    T_joints = np.tile(
        np.eye(4),
        (len(joint_names), 1, 1),
    )

    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id == -1:
            raise ValueError(
                f"模型中不存在 joint：{joint_name}"
            )

        # 当前 joint 所属的 body
        body_id = model.jnt_bodyid[joint_id]

        # joint anchor 在 world 下的位置
        p_world_joint = (
            data.xanchor[joint_id]
            .copy()
        )

        # joint 所属 body 在 world 下的旋转矩阵
        R_world_joint = (
            data.xmat[body_id]
            .reshape(3, 3)
            .copy()
        )

        # world 表达转换为 base 表达
        R_base_joint = (
            R_world_base.T
            @ R_world_joint
        )

        p_base_joint = (
            R_world_base.T
            @ (
                p_world_joint
                - p_world_base
            )
        )

        T_joints[i, :3, :3] = (
            R_base_joint
        )

        T_joints[i, :3, 3] = (
            p_base_joint
        )

    return T_joints
def get_joint_initial_info(
    model,
    joint_names,
    base_body_name="base",
):
    data_zero = mujoco.MjData(model)
    n = len(joint_names)

    zero_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "zero",
    )

    if zero_id == -1:
        raise ValueError(
            "MuJoCo 模型中不存在 zero keyframe"
        )
    mujoco.mj_resetDataKeyframe(
        model,
        data_zero,
        zero_id,
    )

    mujoco.mj_forward(
        model,
        data_zero,
    )

    joint_ids = np.empty(
        n,
        dtype=int,
    )

    positions_relative = np.empty((n, 3))
    rotations_relative = np.empty((n, 3, 3))
    axes_relative = np.empty((n, 3))

    transforms_relative = np.tile(
        np.eye(4),
        (n, 1, 1),
    )

    base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_name,
    )

    if base_body_id == -1:
        raise ValueError(
            f"不存在 body：{base_body_name}"
        )

    p_previous = (
        data_zero.xpos[base_body_id]
        .copy()
    )

    R_previous = (
        data_zero.xmat[base_body_id]
        .reshape(3, 3)
        .copy()
    )

    for i, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id == -1:
            raise ValueError(
                f"MuJoCo 模型中不存在关节：{name}"
            )

        body_id = model.jnt_bodyid[joint_id]

        p_world = (
            data_zero.xanchor[joint_id]
            .copy()
        )

        R_world = (
            data_zero.xmat[body_id]
            .reshape(3, 3)
            .copy()
        )

        axis_world = (
            data_zero.xaxis[joint_id]
            .copy()
        )

        p_relative = (
            R_previous.T
            @ (p_world - p_previous)
        )

        R_relative = (
            R_previous.T
            @ R_world
        )

        axis_relative = (
            R_previous.T
            @ axis_world
        )

        joint_ids[i] = joint_id
        positions_relative[i] = p_relative
        rotations_relative[i] = R_relative
        axes_relative[i] = axis_relative

        transforms_relative[i, :3, :3] = (
            R_relative
        )

        transforms_relative[i, :3, 3] = (
            p_relative
        )

        # 这两行不能遗漏
        p_previous = p_world
        R_previous = R_world

    return (
        joint_ids,
        transforms_relative,
        axes_relative,
    )
def get_T_joint_to_site(
    model,
    joint_name,
    site_name,
):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if joint_id == -1:
        raise ValueError(
            f"不存在 joint：{joint_name}"
        )

    if site_id == -1:
        raise ValueError(
            f"不存在 site：{site_name}"
        )

    joint_body_id = model.jnt_bodyid[joint_id]
    site_body_id = model.site_bodyid[site_id]

    if joint_body_id != site_body_id:
        raise ValueError(
            "joint 和 site 不在同一个 body，"
            "不能直接使用此方法"
        )

    # 两者的位置都相对于同一个 body
    p_joint_body = model.jnt_pos[joint_id].copy()
    p_site_body = model.site_pos[site_id].copy()

    p_joint_site = (
        p_site_body - p_joint_body
    )

    # site 相对于所属 body 的旋转矩阵
    R_joint_site_flat = np.empty(9)

    mujoco.mju_quat2Mat(
        R_joint_site_flat,
        model.site_quat[site_id],
    )

    R_joint_site = (
        R_joint_site_flat.reshape(3, 3)
    )

    T_joint_site = np.eye(4)
    T_joint_site[:3, :3] = R_joint_site
    T_joint_site[:3, 3] = p_joint_site

    return T_joint_site

def limit_vector_norm(vector, max_norm):
    vector = np.asarray(
        vector,
        dtype=float,
    )

    vector_norm = np.linalg.norm(vector)

    if vector_norm > max_norm:
        vector = (
            vector
            / vector_norm
            * max_norm
        )

    return vector

def cartesian_closed_loop(
    model,
    data,
    joint_names,
    actuator_names,
    transforms_relative,
    axes_relative,
    T6ee,
    target_position,
    target_rpy,
    control_dt=0.01,
    max_time=10.0,
    position_tolerance=0.002,
    rotation_tolerance=0.01,
    kp_position=1.5,
    kp_rotation=1.0,
    linear_velocity_max=0.08,
    angular_velocity_max=0.3,
    joint_velocity_max=0.5,
):
    target_position = np.asarray(
        target_position,
        dtype=float,
    )

    target_rpy = np.asarray(
        target_rpy,
        dtype=float,
    )

    joint_ids = np.array([
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        for name in joint_names
    ])

    actuator_ids = np.array([
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in actuator_names
    ])

    if np.any(joint_ids == -1):
        raise ValueError(
            "存在无效 joint 名称"
        )

    if np.any(actuator_ids == -1):
        raise ValueError(
            "存在无效 actuator 名称"
        )

    qpos_addresses = (
        model.jnt_qposadr[joint_ids]
    )

    ctrl_min = model.actuator_ctrlrange[
        actuator_ids, 0
    ]

    ctrl_max = model.actuator_ctrlrange[
        actuator_ids, 1
    ]

    simulation_dt = float(
        model.opt.timestep
    )

    # 例如仿真周期 0.002 s，控制周期 0.01 s
    # 则一个控制周期执行 5 次 mj_step
    substeps = max(
        1,
        round(control_dt / simulation_dt),
    )

    actual_control_dt = (
        substeps * simulation_dt
    )

    max_cycles = int(
        max_time / actual_control_dt
    )

    K = np.diag([
        kp_position,
        kp_position,
        kp_position,
        kp_rotation,
        kp_rotation,
        kp_rotation,
    ])

    logs = {
        "time": [],
        "position": [],
        "position_error": [],
        "rotation_error_norm": [],
        "q": [],
        "q_dot": [],
        "q_command": [],
    }

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        q_command = (
            data.qpos[qpos_addresses]
            .copy()
        )
        for cycle in range(max_cycles):
            wall_start = time.perf_counter()

            # 1. 读取 MuJoCo 实际关节角
            q_current = (
                data.qpos[qpos_addresses]
                .copy()
            )

            # 2. 根据实际 q 重新计算 FK
            T_joints, axes_base = (
                FK_learn.Tbn(
                    q_current,
                    transforms_relative,
                    axes_relative,
                )
            )

            T_ee = FK_learn.compose_transform(
                T_joints[-1],
                T6ee,
            )

            current_position = (
                T_ee[:3, 3].copy()
            )

            # 3. 计算当前位姿误差
            error = IK_learn.pose_error(
                T_ee,
                target_position,
                target_rpy,
            )

            position_error_norm = (
                np.linalg.norm(error[:3])
            )

            rotation_error_norm = (
                np.linalg.norm(error[3:])
            )

            # 4. 判断实际状态是否到达目标
            if (
                position_error_norm
                < position_tolerance
                and rotation_error_norm
                < rotation_tolerance
            ):
                print(
                    f"闭环收敛：cycle={cycle}, "
                    f"position_error="
                    f"{position_error_norm:.6f} m, "
                    f"rotation_error="
                    f"{rotation_error_norm:.6f} rad"
                )

                return (
                    q_current,
                    True,
                    logs,
                )

            # 5. 根据当前实际状态重新计算 Jacobian
            J = IK_learn.geometric_jacobian(
                T_joints,
                T_ee,
                axes_base,
            )

            J_pinv = (
                IK_learn.damped_pseudoinverse(J)
            )

            # 6. 位姿误差转换为期望笛卡尔速度
            x_dot_command = K @ error

            x_dot_command[:3] = (
                limit_vector_norm(
                    x_dot_command[:3],
                    linear_velocity_max,
                )
            )

            x_dot_command[3:] = (
                limit_vector_norm(
                    x_dot_command[3:],
                    angular_velocity_max,
                )
            )

            # 7. DLS 计算关节速度
            q_dot = (
                J_pinv
                @ x_dot_command
            )

            q_dot = np.clip(
                q_dot,
                -joint_velocity_max,
                joint_velocity_max,
            )

            # 8. 积分得到位置执行器目标
            q_command = (
                q_command
                + q_dot * actual_control_dt
            )

            q_command = np.clip(
                q_command,
                ctrl_min,
                ctrl_max,
            )

            data.ctrl[actuator_ids] = (
                q_command
            )

            # 9. 执行 MuJoCo
            for _ in range(substeps):
                mujoco.mj_step(
                    model,
                    data,
                )

            viewer.sync()

            # 10. 记录当前控制周期数据
            logs["time"].append(
                float(data.time)
            )

            logs["position"].append(
                current_position
            )

            logs["position_error"].append(
                error[:3].copy()
            )

            logs["rotation_error_norm"].append(
                rotation_error_norm
            )

            logs["q"].append(
                q_current
            )

            logs["q_dot"].append(
                q_dot.copy()
            )

            logs["q_command"].append(
                q_command.copy()
            )

            if cycle % 50 == 0:
                print(
                    f"cycle={cycle}, "
                    f"position_error="
                    f"{position_error_norm:.6f} m, "
                    f"rotation_error="
                    f"{rotation_error_norm:.6f} rad"
                )

            elapsed = (
                time.perf_counter()
                - wall_start
            )

            if elapsed < actual_control_dt:
                time.sleep(
                    actual_control_dt - elapsed
                )

    q_final = (
        data.qpos[qpos_addresses]
        .copy()
    )

    return q_final, False, logs





class MyArm(Robot):
    config_class = MyArmConfig
    name = "my_arm"

    def __init__(self, config: MyArmConfig):
        # 父类初始化 id、标定目录等通用属性，只调用一次。
        super().__init__(config)
        self.config = config
        self.xml_path = Path(config.xml_path).expanduser().resolve()
        self.reset_noise = float(config.reset_noise)
        if not np.isfinite(self.reset_noise) or self.reset_noise < 0:
            raise ValueError("reset_noise 必须是非负有限数")

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
        self.model = None
        self.data = None
        self._renderer = None
        self.control_cycles = 5
        self.max_episode_steps = 200

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """加载并检查 MuJoCo 模型，恢复 home 姿态；暂不打开窗口。"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("MyArm 已连接，请先 disconnect()")
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"找不到 MuJoCo 模型：{self.xml_path}")

        # 先使用局部变量检查模型，全部成功后才保存到 self。
        # 这样加载中途出错时，不会留下“连接成功”的状态。
        model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        data = mujoco.MjData(model)
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0:
            raise ValueError("MuJoCo 模型中不存在 home keyframe")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        mujoco.mj_forward(model, data)

        # 此辅助函数使用单独的 MjData 读取 zero 姿态，不改变 data 的 home 姿态。
        joint_ids, transforms_relative, axes_relative = get_joint_initial_info(
            model, self.config.joint_names
        )
        if np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("当前 MyArm 实现要求六个关节都是 hinge 转动关节")
        qpos_addresses = model.jnt_qposadr[joint_ids].copy()
        qvel_addresses = model.jnt_dofadr[joint_ids].copy()

        actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.config.actuator_names
        ], dtype=int)
        if np.any(actuator_ids < 0):
            raise ValueError("MuJoCo 模型中缺少所需的 actuator，请检查 actuator_names")
        if (
            np.any(model.actuator_trntype[actuator_ids] != mujoco.mjtTrn.mjTRN_JOINT)
            or not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids)
        ):
            raise ValueError("actuator 必须按 joint_names 的顺序驱动对应关节")
        if len(self.config.gripper_actuator_names) != 2:
            raise ValueError("当前夹爪实现需要两个执行器")
        gripper_actuator_ids = np.array([
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            for name in self.config.gripper_actuator_names
        ], dtype=int)
        if np.any(gripper_actuator_ids < 0):
            raise ValueError(
                "找不到夹爪执行器，请检查 config 和 XML 中的名称"
            )
        if len(self.config.gripper_joint_names) != 2:
            raise ValueError("当前夹爪实现需要两个关节")

        gripper_joint_ids = np.array([
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in self.config.gripper_joint_names
        ], dtype=int)

        if np.any(gripper_joint_ids < 0):
            raise ValueError("找不到夹爪关节，请检查 config 和 XML 中的名称")

        if np.any(
            model.jnt_type[gripper_joint_ids]
            != mujoco.mjtJoint.mjJNT_SLIDE
        ):
            raise ValueError("当前夹爪实现要求两个关节都是 slide 滑动关节")

        # 检查两个执行器是否按配置顺序驱动对应的夹爪关节。
        if (
            np.any(
                model.actuator_trntype[gripper_actuator_ids]
                != mujoco.mjtTrn.mjTRN_JOINT
            )
            or not np.array_equal(
                model.actuator_trnid[gripper_actuator_ids, 0],
                gripper_joint_ids,
            )
        ):
            raise ValueError("夹爪执行器与夹爪关节的对应关系不正确")

        gripper_qpos_addresses = model.jnt_qposadr[gripper_joint_ids].copy()
        T6ee = get_T_joint_to_site(
            model, joint_name=self.config.joint_names[-1], site_name=self.config.ee_name
        )
        substeps = max(1, round(0.01 / float(model.opt.timestep)))

        # 连接时检查相机存在；渲染器留到第一次读取图像时再创建。
        for camera_name in self.camera_names:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(f"MuJoCo 模型中不存在相机：{camera_name}")
        if self.camera_names:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.image_width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, self.image_height)

        # 全部检查通过后，提交连接所需的成员属性。
        self.model = model
        self.data = data
        self.home_id = home_id
        self.joint_ids = joint_ids
        self.transforms_relative = transforms_relative
        self.axes_relative = axes_relative
        self.qpos_addresses = qpos_addresses
        self.qvel_addresses = qvel_addresses
        self.actuator_ids = actuator_ids
        self.gripper_actuator_ids = gripper_actuator_ids
        self.gripper_qpos_addresses = gripper_qpos_addresses
        self.ctrl_min = model.actuator_ctrlrange[actuator_ids, 0].copy()
        self.ctrl_max = model.actuator_ctrlrange[actuator_ids, 1].copy()
        self.T6ee = T6ee
        self.substeps = substeps
        self.control_dt = substeps * float(model.opt.timestep)
        self._is_connected = True

    def disconnect(self) -> None:
        """释放当前适配器的仿真资源；重复调用也是安全的。"""
        try:
            if self._renderer is not None:
                self._renderer.close()
        finally:
            self._renderer = None
            self.data = None
            self.model = None
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
        # 更新由 qpos 推导的位姿，保证相机图像与当前关节状态一致。
        # mj_forward 不是 mj_step，不会增加仿真时间。
        mujoco.mj_forward(self.model, self.data)
        observation = {
            f"{name}.pos": float(self.data.qpos[address])
            for name, address in zip(self.config.joint_names, self.qpos_addresses, strict=True)
        }
        finger_positions = self.data.qpos[self.gripper_qpos_addresses]

        # 两个夹指的平均位移，归一化为 0～1。
        gripper_opening = float(np.mean(finger_positions) / 0.04)

        observation["gripper.pos"] = float(
            np.clip(gripper_opening, 0.0, 1.0)
        )
        for camera_name in self.camera_names:
            observation[camera_name] = self.get_images(camera_name)
        return observation

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
        finger_target = 0.04 * gripper
        self.data.ctrl[self.actuator_ids] = q_target
        self.data.ctrl[self.gripper_actuator_ids] = finger_target

        # 7. 推进一个控制周期
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
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
        # 同一个渲染器可以切换相机，不需要每一帧都重新创建。
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.image_height,
                width=self.image_width,
            )
        self._renderer.update_scene(self.data, camera=camera_name)
        # copy 防止下一次渲染覆盖已经返回给采集程序的图像。
        return self._renderer.render().copy()
    # 用于episode之间的复位
    def reset_simulation(self) -> None:
        if not self.is_connected:
            raise RuntimeError("请先连接机器人，再复位仿真")

        # 恢复 XML 中名为 home 的 keyframe
        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.home_id,
        )

        # 根据复位后的状态，重新计算位置等派生数据
        mujoco.mj_forward(self.model, self.data)

        print("仿真已恢复 home 初始状态")
    

