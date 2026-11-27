"""在新加载、恢复 home 的场景上随机化；不修改 XML 或官方录制循环。"""
import logging
import math
import secrets

import mujoco
import numpy as np

from .config_myarm import MyArmConfig


OBJECT_NAMES = (
    "cube_small", "cube_large", "plate_small", "plate_large", "cup_small", "cup_large",
)
LIGHT_NAMES = ("key_light", "fill_light")
logger = logging.getLogger(__name__)


def _unrank_permutation(index: int, size: int) -> np.ndarray:
    """把 0..size!-1 的组合序号映射为一个唯一的字典序排列。"""
    total = math.factorial(size)
    if not 0 <= index < total:
        raise ValueError(f"物体组合序号必须在 0～{total - 1} 之间，当前为 {index}")

    remaining = list(range(size))
    permutation = []
    for width in range(size, 0, -1):
        block_size = math.factorial(width - 1)
        choice, index = divmod(index, block_size)
        permutation.append(remaining.pop(choice))
    return np.asarray(permutation, dtype=int)


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")


def randomize_scene(model, data, config: MyArmConfig) -> dict:
    """每次 connect 调用一次。返回可追溯的随机种子、槽位和灯光参数。

    六个槽位取自 XML 的初始 XY，而 Z/四元数保留 home 中各物体自身的值。
    调用者必须传入新加载的模型，避免灯光参数在多次调用时累计偏移。
    """
    if not (config.randomize_objects or config.randomize_lighting):
        return {}

    _nonnegative_integer(config.scene_randomization_index, "scene_randomization_index")
    seed = config.scene_randomization_seed
    if seed is None:
        seed = secrets.randbits(64)
    _nonnegative_integer(seed, "scene_randomization_seed")
    # 灯光使用独立随机流，切换灯光开关不会改变同一种子的物体排列。
    _, light_seed = np.random.SeedSequence(
        [seed, config.scene_randomization_index]
    ).spawn(2)
    light_rng = np.random.default_rng(light_seed)
    result = {"seed": seed, "index": config.scene_randomization_index}

    # 先验证所有目标和范围，再写入 model/data，避免一半成功一半失败。
    qpos_addresses, dof_addresses = [], []
    for name in OBJECT_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0 or model.body_jntnum[body_id] != 1:
            raise ValueError(f"随机场景要求物体 {name} 存在且只有一个 freejoint")
        joint_id = model.body_jntadr[body_id]
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"随机场景要求物体 {name} 使用 freejoint")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    slots = np.array([model.qpos0[adr:adr + 2] for adr in qpos_addresses])
    if len(np.unique(slots, axis=0)) != len(OBJECT_NAMES):
        raise ValueError("六个物体的 XML 初始 XY 槽位必须互不相同")

    light_updates = []
    if config.randomize_lighting:
        bounds = np.asarray(config.light_brightness_range, dtype=float)
        if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
                or not 0 < bounds[0] <= bounds[1]):
            raise ValueError("light_brightness_range 必须是正数区间 [最小倍率, 最大倍率]")
        jitter = np.asarray(config.light_position_jitter, dtype=float)
        if jitter.shape != (3,) or not np.all(np.isfinite(jitter)) or np.any(jitter < 0):
            raise ValueError("light_position_jitter 必须是三个非负有限数，单位米")
        # 照向六个槽位的中心、桌面附近；保持 XML 中的平行光类型。
        target = np.r_[slots.mean(axis=0), np.mean([
            model.qpos0[adr + 2] for adr in qpos_addresses
        ])]
        for name in LIGHT_NAMES:
            light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
            if light_id < 0 or model.light_bodyid[light_id] != 0:
                raise ValueError(f"随机场景要求灯光 {name} 存在且固定在 worldbody 下")
            if model.light_mode[light_id] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
                raise ValueError(f"随机灯光 {name} 必须使用 fixed 模式")
            if model.light_pos[light_id, 2] - jitter[2] <= target[2]:
                raise ValueError("灯光最低高度必须高于物体；请减小 light_position_jitter[2]")
            pos = model.light_pos[light_id] + light_rng.uniform(-jitter, jitter)
            direction = target - pos
            direction /= np.linalg.norm(direction)
            multiplier = float(light_rng.uniform(*bounds))
            light_updates.append((light_id, name, pos, direction, multiplier))

    if config.randomize_objects:
        # seed 在这里作为组合序号使用；index 仍保留为额外的可复现偏移，
        # 兼容旧配置中通过 scene_randomization_index 生成不同场景的行为。
        combination_count = math.factorial(len(OBJECT_NAMES))
        combination_index = (
            seed + config.scene_randomization_index
        ) % combination_count
        permutation = _unrank_permutation(combination_index, len(OBJECT_NAMES))
        result["object_combination_index"] = int(combination_index)
        result["objects"] = {}
        for i, name in enumerate(OBJECT_NAMES):
            adr, dof = qpos_addresses[i], dof_addresses[i]
            slot = int(permutation[i])
            data.qpos[adr:adr + 2] = slots[slot]
            data.qvel[dof:dof + 6] = 0
            result["objects"][name] = {"slot": slot, "xy": slots[slot].tolist()}

    if config.randomize_lighting:
        result["lights"] = {}
        for light_id, name, pos, direction, multiplier in light_updates:
            model.light_pos[light_id] = pos
            model.light_dir[light_id] = direction
            model.light_diffuse[light_id] *= multiplier
            model.light_specular[light_id] *= multiplier
            result["lights"][name] = {"position": pos.tolist(), "brightness": multiplier}

    mujoco.mj_forward(model, data)
    logger.info("场景随机化: %s", result)
    return result
