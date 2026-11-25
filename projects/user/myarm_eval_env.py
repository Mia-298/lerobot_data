import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
import mujoco.viewer
from lerobot_robot_myarm import MyArm, MyArmConfig
from lerobot_robot_myarm.task_tags import resolve_task_tags

@register_env("myarm_eval_env")  # 注册给LeRobot识别
class MyArmEvalEnv(gym.Env):
    """第一步：只实现环境初始化、reset、观测和资源释放。"""
    metadata = {
        "render_modes": ["rgb_array"],
    }
    render_mode = "rgb_array"
    def __init__(self, task_id=1, xml_path=None, episode_length=3000):
        super().__init__()

        self.robot = MyArm(
            MyArmConfig(
                **({"xml_path": xml_path} if xml_path is not None else {}),
                camera_names=["front", "wrist"],
                image_width=640,
                image_height=480,
                randomize_objects=False,
                randomize_lighting=False,
            )
        )

        # 与训练数据 observation.state 的顺序一致。
        self.state_names = [
            f"{name}.pos"
            for name in self.robot.config.joint_names
        ] + ["gripper.pos"]

        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(
                low=np.array([-np.inf] * 6 + [0.0], dtype=np.float32),
                high=np.array([np.inf] * 6 + [1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "pixels": spaces.Dict({
                camera: spaces.Box(
                    low=0,
                    high=255,
                    shape=(480, 640, 3),
                    dtype=np.uint8,
                )
                for camera in ("front", "wrist")
            }),
        })
        # 加载模型
        self.robot.connect()
                # 复用采集数据时的任务编号与语言指令。
        targets = resolve_task_tags(
            task_id,
            self.robot.config.tasklist,
        )
        self.task_description = (
            self.robot.config.tasklist[task_id - 1]
        )

        object_names = {
            0: "cube_small",
            1: "cube_large",
            4: "cup_small",
            5: "cup_large",
        }
        plate_names = {
            2: "plate_small",
            3: "plate_large",
        }

        model = self.robot.model
        object_name = object_names[targets.object_tag_id]
        plate_name = plate_names[targets.plate_tag_id]

        self._object_body_id = model.body(object_name).id
        self._plate_geom_id = model.geom(
            f"{plate_name}_geom"
        ).id

        # 当前 XML 中这些物体的碰撞 geom 都属于自身 body。
        self._object_geom_ids = set(
            np.flatnonzero(
                model.geom_bodyid == self._object_body_id
            ).tolist()
        )

        # 用于排除物体仍与夹指接触的情况。
        finger_body_ids = model.jnt_bodyid[
            [
                model.joint(name).id
                for name in self.robot.config.gripper_joint_names
            ]
        ]
        self._finger_geom_ids = set(
            np.flatnonzero(
                np.isin(model.geom_bodyid, finger_body_ids)
            ).tolist()
        )

        # 连续满足 0.2 秒仿真时间才确认放置。
        self._placement_steps = 0
        self._placement_required_steps = max(
            1,
            int(np.ceil(0.2 / self.robot.control_dt)),
        )
        # 视频帧率对应每次 step 推进的仿真时间。
        self.metadata = {
            **type(self).metadata,
            "render_fps": round(1.0 / self.robot.control_dt),
        }
        self.action_names = self.state_names.copy()

        self.action_space = spaces.Box(
            low=np.array(
                [*self.robot.ctrl_min, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [*self.robot.ctrl_max, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        if episode_length <= 0:
            raise ValueError("episode_length 必须大于 0")
        self._max_episode_steps = episode_length
        self._elapsed_steps = 0
        self._needs_reset = True

    def _check_placement(self):
        model = self.robot.model
        data = self.robot.data

        # 将物体中心转换到盘子 geom 的局部坐标系。
        plate_position = data.geom_xpos[self._plate_geom_id]
        plate_rotation = data.geom_xmat[
            self._plate_geom_id
        ].reshape(3, 3)

        object_position = data.xpos[self._object_body_id]
        relative = plate_rotation.T @ (
            object_position - plate_position
        )

        # 当前 XML 的盘子是 cylinder：
        # size[0] 是半径，size[1] 是半高。
        radius, half_height = model.geom_size[
            self._plate_geom_id, :2
        ]

        # 第一版采用“物体中心在盘面范围内”的宽松标准。
        centered = np.linalg.norm(relative[:2]) < radius
        above_plate = relative[2] > half_height

        touching_plate = False
        touching_finger = False

        for i in range(data.ncon):
            contact = data.contact[i]

            # 不将正距离的近接触算成实际接触。
            if contact.dist > 0:
                continue

            g1, g2 = int(contact.geom1), int(contact.geom2)

            if g1 in self._object_geom_ids:
                other = g2
            elif g2 in self._object_geom_ids:
                other = g1
            else:
                continue

            if other == self._plate_geom_id:
                touching_plate = True

            if other in self._finger_geom_ids:
                touching_finger = True

        placed_now = bool(
            centered
            and above_plate
            and touching_plate
            and not touching_finger
        )

        if placed_now:
            self._placement_steps += 1
        else:
            self._placement_steps = 0

        success = (
            self._placement_steps >= self._placement_required_steps
        )

        details = {
            "centered_on_plate": bool(centered),
            "touching_plate": touching_plate,
            "touching_finger": touching_finger,
            "placement_steps": self._placement_steps,
        }
        return bool(success), details
    def render(self):
            """返回前置相机的 RGB 图像，不推进仿真。"""
            if not self.robot.is_connected:
                raise RuntimeError("环境未连接，无法渲染")
    
            return self.robot.get_images(
                "front",
                hide_tags=True,
            )
    def _get_obs(self):
        raw = self.robot.get_observation()

        return {
            "agent_pos": np.array(
                [raw[name] for name in self.state_names],
                dtype=np.float32,
            ),
            "pixels": {
                "front": raw["front"],
                "wrist": raw["wrist"],
            },
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if not self.robot.is_connected:
            self.robot.connect()
        else:
            self.robot.reset_simulation()

        self._elapsed_steps = 0
        self._placement_steps = 0
        self._needs_reset = False

        observation = self._get_obs()
        info = {}
        return observation, info

    def close(self):
        self.robot.disconnect()
    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("请先调用 reset() 开始新的 episode")

        action = np.asarray(action, dtype=np.float32)

        if action.shape != self.action_space.shape:
            raise ValueError(
                f"动作应为 {self.action_space.shape}，实际是 {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("动作中不能包含 NaN 或无穷大")

        # 限制到执行器允许的范围。
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        action_dict = {
            name: float(value)
            for name, value in zip(
                self.action_names, action, strict=True
            )
        }

        # 内部已经调用 mj_step，不要在这里额外推进仿真。
        self.robot.send_action(action_dict)
        self._elapsed_steps += 1

        observation = self._get_obs()

        
        success, placement_info = self._check_placement()

        reward = float(success)
        terminated = success
        truncated = (
            self._elapsed_steps >= self._max_episode_steps
            and not success
        )

        self._needs_reset = terminated or truncated

        info = {
            "is_success": success,
            "elapsed_steps": self._elapsed_steps,
            **placement_info,
        }

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    env = MyArmEvalEnv()

    try:
        # 延长到约 10 秒仿真时间，方便观察。
        env._max_episode_steps = round(
            10.0 / env.robot.control_dt
        )

        obs, info = env.reset(seed=42)

        # 检查评估视频使用的画面。
        image = env.render()
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8
        print("render() 正常：", image.shape, image.dtype)

        # 继续使用保持初始位置的动作。
        hold_action = np.clip(
            obs["agent_pos"].copy(),
            env.action_space.low,
            env.action_space.high,
        )

        with mujoco.viewer.launch_passive(
            env.robot.model,
            env.robot.data,
        ) as viewer:

            while viewer.is_running():
                start = time.perf_counter()

                obs, reward, terminated, truncated, info = (
                    env.step(hold_action)
                )
                print("评估结果：", info)
                # 把当前仿真状态同步到窗口。
                viewer.sync()

                if terminated or truncated:
                    print(
                        f"本轮结束：steps={env._elapsed_steps}, "
                        f"sim_time={env.robot.data.time:.3f}"
                    )
                    break

                # 计算较快时适当等待；计算慢时不额外等待。
                remaining = (
                    env.robot.control_dt
                    - (time.perf_counter() - start)
                )
                if remaining > 0:
                    time.sleep(remaining)

        print("第三步完成：图像渲染和窗口显示正常。")
    finally:
        env.close()
