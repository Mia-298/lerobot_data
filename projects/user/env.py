# env.py
import gymnasium as gym

def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg=None,
):
    # 找到模型文件
    root = Path(__file__).resolve().parents[1]
    xml_path = root / "models" / "ur5e.xml"

    if n_envs < 1:
        raise ValueError("n_envs 必须大于等于 1")

    # 每调用一次，就创建一个独立的 UR5 仿真环境
    def _make_single_env():
        return UR5ReachEnv(xml_path=xml_path)

    # 注意：这里放的是函数，不是已经创建好的环境对象
    env_fns = [_make_single_env for _ in range(n_envs)]

    if use_async_envs:
        return gym.vector.AsyncVectorEnv(
            env_fns,
            context="spawn",
        )

    return gym.vector.SyncVectorEnv(env_fns)

from pathlib import Path
import sys

# 当前文件：
TEST_DIR = Path(__file__).resolve().parent.parent
MC_DIR = "/home/mia/data/openvla_work/projects/openvla-data/test/MC"

# 告诉 Python 到 MC 文件夹里查找模块
sys.path.insert(0, str(MC_DIR))

# 必须放在添加搜索路径之后
import FK_learn
import IK_learn

import numpy as np
import mujoco
import time
import mujoco.viewer
import time
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


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

    print("zero qpos:")
    print(data_zero.qpos.copy())

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
MODEL_DIR = Path(__file__).resolve().parent
XML_PATH = MODEL_DIR / "models/ur5e.xml"



joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
    ]
actuator_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
ee_name="attachment_site"

class UR5ReachEnv(gym.Env):
    def __init__(self, xml_path, reset_noise=0.0):
        super().__init__()
        # 默认初态扰动幅度（rad），单次 reset 的 options 可覆盖。
        self.reset_noise = float(reset_noise)
        if not np.isfinite(self.reset_noise) or self.reset_noise < 0:
            raise ValueError("reset_noise 必须是非负有限数")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # 加载 XML 中定义的 home 姿态
        self.home_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        mujoco.mj_forward(self.model, self.data)


        print("模型加载成功")
        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "attachment_site",
        )
        self.joint_ids,self.transforms_relative,self.axes_relative = get_joint_initial_info(self.model, joint_names)
        self.qpos_addresses = (
            self.model.jnt_qposadr[self.joint_ids].copy()
        )
        self.qvel_addresses = (
            self.model.jnt_dofadr[self.joint_ids].copy()
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )
        # 执行器索引，顺序与 joint_names 对应
        self.actuator_ids = np.array([
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            for name in actuator_names
        ])

        if np.any(self.actuator_ids < 0):
            raise ValueError("存在无效 actuator 名称")

        # 当前 XML 中这些执行器的控制量是关节位置指令
        self.ctrl_min = (
            self.model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        )
        self.ctrl_max = (
            self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        )

        # 末端相对于最后一个关节的固定变换
        self.T6ee = get_T_joint_to_site(
            self.model,
            joint_name=joint_names[-1],
            site_name=ee_name,
        )

        # 一个控制周期约 0.01 秒
        self.substeps = max(
            1,
            round(0.01 / self.model.opt.timestep),
        )
        self.control_dt = (
            self.substeps * float(self.model.opt.timestep)
        )

        # 每个策略动作保持 5 个控制周期
        self.control_cycles = 5
        self.max_episode_steps = 200
        self._renderer = None
    def get_images(self):
        # 第一次获取图像时创建，之后重复使用
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=480,
                width=640,
            )

        # 根据当前仿真状态，切换到 XML 中名为 wrist 的相机
        self._renderer.update_scene(
            self.data,
            camera="wrist",
        )

        # 获取 RGB 图像
        rgb = self._renderer.render().copy()

        return {"wrist": rgb}
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 1. 恢复初始姿态
        # 2. 设置本回合最终任务目标
        # 3. 清零步数，重置控制器内部状态
        # 4. 返回 obs, info
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        mujoco.mj_forward(self.model, self.data)
        T_home = get_site_pose_in_base(
            self.model,
            self.data,
            site_name=ee_name,
        )
        home_pos, home_rpy = FK_learn.transfotm2Cartesian(T_home)
        # 示例任务：沿 base 的 x 方向移动 3 cm，姿态不变
        # 这个偏移需要先用你的控制器验证可达、无碰撞
        self.goal_pos = home_pos + np.array([0.03, 0.0, 0.0])
        self.goal_rpy = home_rpy.copy()

        # 动作映射的固定中心：这里采用 home 位姿
        self.command_pos_center = home_pos.copy()
        self.command_rpy_center = home_rpy.copy()
        # 默认使用环境配置的扰动幅度，单次 options 可以覆盖
        options = {} if options is None else options
        reset_noise = float(options.get("reset_noise", self.reset_noise))

        if not np.isfinite(reset_noise) or reset_noise < 0:
            raise ValueError("reset_noise 必须是非负有限数")

        # 此时仍然是原始 home 关节位置
        q_home = self.data.qpos[self.qpos_addresses].copy()

        # 每个关节独立添加随机扰动，单位 rad
        q_initial = q_home + self.np_random.uniform(
            low=-reset_noise,
            high=reset_noise,
            size=q_home.shape,
        )

        # 同时满足关节位置限制和位置执行器指令范围
        joint_limited = self.model.jnt_limited[self.joint_ids].astype(bool)
        joint_ranges = self.model.jnt_range[self.joint_ids]

        q_min = np.maximum(
            self.ctrl_min,
            np.where(joint_limited, joint_ranges[:, 0], -np.inf),
        )
        q_max = np.minimum(
            self.ctrl_max,
            np.where(joint_limited, joint_ranges[:, 1], np.inf),
        )
        q_initial = np.clip(q_initial, q_min, q_max)

        # 同步实际位置与控制器指令，初始速度清零
        self.data.qpos[self.qpos_addresses] = q_initial
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.actuator_ids] = q_initial

        # 根据新的关节位置，重新计算末端位置等派生状态
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps = 0

        return self._get_obs(), {
            "goal_pos": self.goal_pos.copy(),
            "goal_rpy": self.goal_rpy.copy(),
        }
    def _advance_control(self, target_pos, target_rpy):
        for _ in range(self.control_cycles):
            # 1. 读取实际关节位置
            q = self.data.qpos[self.qpos_addresses].copy()

            # 2. FK：计算当前末端位姿与关节轴
            T_joints, axes_base = FK_learn.Tbn(
                q,
                self.transforms_relative,
                self.axes_relative,
            )
            T_ee = T_joints[-1] @ self.T6ee

            # 3. 相对于本步控制指令的位姿误差
            error = IK_learn.pose_error(
                T_ee,
                target_pos,
                target_rpy,
            )

            # 4. 误差转换成期望笛卡尔速度
            velocity = error * np.array([
                1.5, 1.5, 1.5,
                1.0, 1.0, 1.0,
            ])

            velocity[:3] = limit_vector_norm(
                velocity[:3], 0.08
            )
            velocity[3:] = limit_vector_norm(
                velocity[3:], 0.3
            )

            # 5. 雅可比 + 阻尼伪逆：笛卡尔速度 → 关节速度
            J = IK_learn.geometric_jacobian(
                T_joints,
                T_ee,
                axes_base,
            )

            dq = IK_learn.damped_pseudoinverse(J) @ velocity
            dq = np.clip(dq, -0.5, 0.5)

            # 6. 在上一条关节位置指令上积分
            q_command = (
                self.data.ctrl[self.actuator_ids]
                + dq * self.control_dt
            )

            self.data.ctrl[self.actuator_ids] = np.clip(
                q_command,
                self.ctrl_min,
                self.ctrl_max,
            )

            # 7. 推进一个控制周期
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
    def step(self, action):
        # 1. 检查并限制动作
        action = np.asarray(action, dtype=np.float32)

        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("action 必须是有限数值组成的 (6,) 数组")

        action = np.clip(action, -1.0, 1.0)

        # 2. 归一化动作 → 本步末端位姿指令
        ee_target_pos = (
            self.command_pos_center + 0.05 * action[:3]
        )
        ee_target_rpy = (
            self.command_rpy_center + 0.2 * action[3:]
        )

        # 3. 固定时长执行，不等待完全到位
        self._advance_control(ee_target_pos, ee_target_rpy)

        # 4. 读取执行后的实际末端位姿
        T_actual = get_site_pose_in_base(
            self.model,
            self.data,
            site_name=ee_name,
        )

        # 注意：奖励比较的是最终任务目标，不是本步指令
        error = IK_learn.pose_error(
            T_actual,
            self.goal_pos,
            self.goal_rpy,
        )

        position_error = float(np.linalg.norm(error[:3]))
        rotation_error = float(np.linalg.norm(error[3:]))

        # 5. 计算即时奖励
        reward = (
            -position_error / 0.05
            -0.2 * rotation_error / 0.2
        )

        # 6. 更新步数，判断结束
        self.elapsed_steps += 1

        terminated = bool(
            position_error < 0.01
            and rotation_error < 0.1
        )

        truncated = bool(
            self.elapsed_steps >= self.max_episode_steps
            and not terminated
        )

        info = {
            "is_success": terminated,
            "position_error": position_error,
            "rotation_error": rotation_error,
        }

        # 7. 返回动作执行后的结果
        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _get_obs(self):
        q = self.data.qpos[self.qpos_addresses].copy()
        dq_actual = self.data.qvel[self.qvel_addresses].copy()

        q_command = self.data.ctrl[self.actuator_ids].copy()
        tracking_error = q_command - q

        return np.concatenate([
            q,               # 6维：实际关节位置
            dq_actual,       # 6维：实际关节速度
            tracking_error,  # 6维：指令与实际位置的差
        ]).astype(np.float32)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

