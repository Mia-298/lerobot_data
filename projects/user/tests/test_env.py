import os
from pathlib import Path

# 允许在无显示器的机器上使用 MuJoCo 离屏渲染。
os.environ.setdefault("MUJOCO_GL", "egl")

from PIL import Image

from lerobot.envs.utils import _load_module_from_path, _call_make_env, _normalize_hub_result

# Load your module
module = _load_module_from_path("./env.py")

# Test the make_env function
result = _call_make_env(module, n_envs=2, use_async_envs=False, cfg=None)
normalized = _normalize_hub_result(result)

# Verify it works
suite_name = next(iter(normalized))
env = normalized[suite_name][0]
obs, info = env.reset()
print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

# SyncVectorEnv 的第一个子环境在 reset 后处于 XML 的 home 位姿（默认 reset_noise=0）。
# 直接调用底层环境的 wrist 相机渲染，保存一张用于检查夹爪遮挡情况的图片。
base_env = env.envs[0] if hasattr(env, "envs") else getattr(env, "unwrapped", env)
if not hasattr(base_env, "get_images"):
    raise AttributeError("当前测试环境没有 get_images()，无法渲染 wrist 视角")

images = base_env.get_images()
wrist_home = images["wrist"]
output_dir = Path(__file__).resolve().parents[3] / "outputs" / "test"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "wrist_home.png"
Image.fromarray(wrist_home).save(output_path)
print(f"Wrist home 图像已保存：{output_path}")

env.close()
