from pathlib import Path

import mujoco
import numpy as np


from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

from lerobot.lerobot_types import RobotAction, TransitionKey
from .config_myarm import MyArmConfig
from .myarm import FK_learn, IK_learn,get_joint_initial_info,get_T_joint_to_site
import numpy as np
class KeyboardIKActionStep(RobotActionProcessorStep):
    def __init__(
            self,
            joint_names,
            transforms_relative,
            axes_relative,
            T6ee,
            q_min,
            q_max,):
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
        # 仿真测试用：限制每个控制周期的关节目标变化量，单位 rad
        self.max_joint_step = 0.1
    def action(self, action: RobotAction) -> RobotAction:
        print("执行到我自己写的 action() 了")
        print("收到动作：", action)
        obs = self.transition[TransitionKey.OBSERVATION]
        q_current = np.array(
            [
                obs[f"{name}.pos"]
                for name in self.joint_names
            ],
            dtype=float,
        )
        T_joints, _ = FK_learn.Tbn(
            q_current,
            self.transforms_relative,
            self.axes_relative,
        )

        T_ee = T_joints[-1] @ self.T6ee
        pos_current, rpy_current = FK_learn.transfotm2Cartesian(T_ee)
        delta = 0.002 * np.array(
            [
                action["delta_x"],
                action["delta_y"],
                action["delta_z"],
            ],
            dtype=float,
        )
        pos_target = pos_current + delta
        rpy_target = rpy_current
        q_target, success, iterations = IK_learn.inverse_kinematics(
            q_init=q_current,
            pos_target=pos_target,
            rpy_target=rpy_target,
            transforms_relative=self.transforms_relative,
            axes_relative=self.axes_relative,
            T6ee=self.T6ee,
            pos_tolerance=0.0001,
            rot_tolerance=0.001,
        )
        q_target = np.asarray(q_target, dtype=float)

        
        if not success :
            print("IK求解失败,本次不移动")
            q_target = q_current.copy()

        valid = (
                    q_target.shape == q_current.shape
                    and np.all(np.isfinite(q_target))
                    and np.all(q_target >= self.q_min)
                    and np.all(q_target <= self.q_max)
                    and np.all(
                        np.abs(q_target - q_current) <= self.max_joint_step
                    )
                )
        if not valid:
            print("目标超出测试限制，本次移动被限制")
            q_target = q_target.clip(self.q_min ,self.q_max)

        print("IK 是否成功：", success, "迭代次数：", iterations)
        processed_action = {
            f"{name}.pos": float(q_target[i])
            for i, name in enumerate(self.joint_names)
        }
        processed_action["gripper.pos"] = float(obs["gripper.pos"])
        print("处理后：",processed_action)
        return processed_action
    def transform_features(self, features):
        # 暂时不改变特征声明
        return features

def make_myarm_processor(robot_cfg: MyArmConfig):
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
    # 在record中没有把robot传入，所以需要从同一份 XML 和配置读取静态 IK 参数
    ik_step = KeyboardIKActionStep(
        joint_names=robot_cfg.joint_names,
        transforms_relative=transforms_relative,
        axes_relative=axes_relative,
        T6ee=T6ee,
        q_min=model.actuator_ctrlrange[actuator_ids, 0].copy(),
        q_max=model.actuator_ctrlrange[actuator_ids, 1].copy(),
    )
    # 这里是返回一个流水线对象，里面包含一个step对象
    return RobotProcessorPipeline(
        steps=[ik_step],
        # 这一步会把action和observation
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )