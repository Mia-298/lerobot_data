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
env.close()