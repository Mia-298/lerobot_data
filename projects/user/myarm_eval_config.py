"""导入本模块即注册 --env.type=myarm_sim；不会创建仿真。"""
from dataclasses import dataclass, field
from functools import partial

import gymnasium as gym
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig


@EnvConfig.register_subclass("myarm_sim")
@dataclass
class MyArmEvalConfig(EnvConfig):
    task: str | None = "pick_place"
    task_id: int = 1
    xml_path: str | None = None
    episode_length: int = 3000
    fps: int = 100
    features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "pixels/front": PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3)),
        "pixels/wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3)),
    })
    features_map: dict[str, str] = field(default_factory=lambda: {
        "action": "action",
        "agent_pos": "observation.state",
        "pixels/front": "observation.images.front",
        "pixels/wrist": "observation.images.wrist",
    })

    @property
    def gym_kwargs(self):
        return dict(task_id=self.task_id, xml_path=self.xml_path,
                    episode_length=self.episode_length)

    def create_envs(self, n_envs, use_async_envs=False):
        from myarm_eval_env import MyArmEvalEnv
        # MuJoCo 渲染先采用单环境，避免多进程 OpenGL 上下文问题。
        if n_envs != 1:
            raise ValueError("当前 myarm_sim 请设置 eval.batch_size=1")
        vec = gym.vector.SyncVectorEnv(
            [partial(MyArmEvalEnv, **self.gym_kwargs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        return {self.type: {self.task_id: vec}}
