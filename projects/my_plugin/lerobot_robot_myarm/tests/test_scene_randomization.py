"""真实 MuJoCo 模型离线测试，不开启窗口、不录制数据集。"""
import importlib
from itertools import permutations
from pathlib import Path
import sys
import types
import unittest

import mujoco
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1] / "lerobot_robot_myarm"
# 发现同目录离线测试时复用已加载的包，避免重复注册 my_arm 配置。
pkg = sys.modules.get("_myarm_safety_offline")
if pkg is None:
    pkg = types.ModuleType("_myarm_scene_offline")
    pkg.__path__ = [str(PACKAGE)]
    sys.modules[pkg.__name__] = pkg
config = importlib.import_module(f"{pkg.__name__}.config_myarm")
scene = importlib.import_module(f"{pkg.__name__}.scene_randomization")
myarm = importlib.import_module(f"{pkg.__name__}.myarm")


class SceneRandomizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config.MyArmConfig(camera_names=[], randomize_objects=True,
                                      randomize_lighting=True, scene_randomization_seed=42)
        self.model, self.data = self.fresh()
        self.addresses = [int(self.model.joint(name + "_free").qposadr[0])
                          for name in scene.OBJECT_NAMES]
        self.slots = np.array([self.model.qpos0[a:a + 2] for a in self.addresses])

    def fresh(self):
        model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        return model, data

    def test_disabled_leaves_scene_unchanged(self):
        self.cfg.randomize_objects = self.cfg.randomize_lighting = False
        qpos = self.data.qpos.copy()
        pos, diffuse = self.model.light_pos.copy(), self.model.light_diffuse.copy()
        self.assertEqual(scene.randomize_scene(self.model, self.data, self.cfg), {})
        np.testing.assert_array_equal(self.data.qpos, qpos)
        np.testing.assert_array_equal(self.model.light_pos, pos)
        np.testing.assert_array_equal(self.model.light_diffuse, diffuse)

    def test_permutation_preserves_height_orientation_robot_and_tags(self):
        before, geom_bodyids = self.data.qpos.copy(), self.model.geom_bodyid.copy()
        result = scene.randomize_scene(self.model, self.data, self.cfg)
        self.assertEqual(sorted(o["slot"] for o in result["objects"].values()), list(range(6)))
        expected = before.copy()
        for name, adr in zip(scene.OBJECT_NAMES, self.addresses):
            slot = result["objects"][name]["slot"]
            expected[adr:adr + 2] = self.slots[slot]
            np.testing.assert_allclose(self.data.xpos[self.model.body(name).id, :2], self.slots[slot])
        np.testing.assert_array_equal(self.data.qpos, expected)
        np.testing.assert_array_equal(self.model.geom_bodyid, geom_bodyids)
        np.testing.assert_array_equal(self.data.qvel, 0)

    def test_seed_and_index_replay_and_vary(self):
        first = scene.randomize_scene(self.model, self.data, self.cfg)
        m, d = self.fresh()
        self.assertEqual(first, scene.randomize_scene(m, d, self.cfg))
        self.cfg.scene_randomization_index += 1
        m, d = self.fresh()
        second = scene.randomize_scene(m, d, self.cfg)
        self.assertNotEqual(first["objects"], second["objects"])
        self.assertNotEqual(first["lights"], second["lights"])

    def test_object_permutation_independent_of_lighting_toggle(self):
        first = scene.randomize_scene(self.model, self.data, self.cfg)
        self.cfg.randomize_lighting = False
        m, d = self.fresh()
        self.assertEqual(first["objects"], scene.randomize_scene(m, d, self.cfg)["objects"])
        np.testing.assert_array_equal(m.light_pos, self.fresh()[0].light_pos)

    def test_only_lights_does_not_move_objects_and_respects_bounds(self):
        self.cfg.randomize_objects = False
        before = self.data.qpos.copy()
        pos = self.model.light_pos.copy()
        diffuse, specular = self.model.light_diffuse.copy(), self.model.light_specular.copy()
        result = scene.randomize_scene(self.model, self.data, self.cfg)
        np.testing.assert_array_equal(self.data.qpos, before)
        for name, sample in result["lights"].items():
            i = self.model.light(name).id
            self.assertTrue(np.all(np.abs(self.model.light_pos[i] - pos[i]) <= self.cfg.light_position_jitter))
            self.assertGreaterEqual(sample["brightness"], .75)
            self.assertLessEqual(sample["brightness"], 1.15)
            np.testing.assert_allclose(self.model.light_diffuse[i], diffuse[i] * sample["brightness"])
            np.testing.assert_allclose(self.model.light_specular[i], specular[i] * sample["brightness"])
            self.assertAlmostEqual(np.linalg.norm(self.model.light_dir[i]), 1)
            self.assertLess(self.model.light_dir[i, 2], 0)
            np.testing.assert_allclose(self.data.light_xpos[i], self.model.light_pos[i])

    def test_unseeded_sample_can_be_replayed_from_report(self):
        self.cfg.scene_randomization_seed = None
        first = scene.randomize_scene(self.model, self.data, self.cfg)
        self.cfg.scene_randomization_seed = first["seed"]
        m, d = self.fresh()
        self.assertEqual(first, scene.randomize_scene(m, d, self.cfg))

    def test_invalid_settings_fail_before_mutation(self):
        for field, bad in (("scene_randomization_seed", -1),
                           ("scene_randomization_index", True),
                           ("light_brightness_range", [1, .5]),
                           ("light_brightness_range", [0, 1]),
                           ("light_brightness_range", [1, float("nan")]),
                           ("light_position_jitter", [-1, 0, 0]),
                           ("light_position_jitter", [1, 2]),
                           ("light_position_jitter", [1, 1, 5])):
            with self.subTest(field=field, value=bad):
                original = getattr(self.cfg, field)
                setattr(self.cfg, field, bad)
                before = self.data.qpos.copy()
                pos = self.model.light_pos.copy()
                with self.assertRaises(ValueError):
                    scene.randomize_scene(self.model, self.data, self.cfg)
                np.testing.assert_array_equal(self.data.qpos, before)
                np.testing.assert_array_equal(self.model.light_pos, pos)
                setattr(self.cfg, field, original)

    def test_missing_light_fails_without_moving_objects(self):
        # 使用缺少第二盏灯的真实小模型，保证失败时不写入任何物体位置。
        bodies = ''.join(f'<body name="{n}" pos="{i} 0 1"><freejoint/>'
                         '<geom type="sphere" size=".01" mass="1"/></body>'
                         for i, n in enumerate(scene.OBJECT_NAMES))
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light name="key_light" '
                                          'pos="0 0 3"/>' + bodies + '</worldbody></mujoco>')
        d = mujoco.MjData(m)
        before = d.qpos.copy()
        with self.assertRaisesRegex(ValueError, "fill_light"):
            scene.randomize_scene(m, d, self.cfg)
        np.testing.assert_array_equal(d.qpos, before)

    def test_all_720_permutations_have_no_initial_object_object_penetration(self):
        # 逐一验证所有排列，不凭物体中心距离猜测杯把/盘子的碰撞。
        bodyids = {self.model.body(name).id for name in scene.OBJECT_NAMES}
        for perm in permutations(range(6)):
            for adr, slot in zip(self.addresses, perm):
                self.data.qpos[adr:adr + 2] = self.slots[slot]
            mujoco.mj_forward(self.model, self.data)
            for contact in self.data.contact:
                b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if b1 in bodyids and b2 in bodyids and b1 != b2:
                    self.assertGreaterEqual(contact.dist, -1e-8, (perm, b1, b2))

    def test_connect_uses_randomization_and_keeps_robot_home(self):
        robot = myarm.MyArm(self.cfg)
        try:
            robot.connect()
            self.assertTrue(robot.is_connected)
            self.assertIn("objects", robot.scene_randomization)
            np.testing.assert_array_equal(robot.data.qpos[robot.qpos_addresses],
                                          self.data.qpos[robot.qpos_addresses])
            first = robot.scene_randomization
            robot.disconnect()
            self.cfg.scene_randomization_index += 1
            robot.connect()
            self.assertNotEqual(first["objects"], robot.scene_randomization["objects"])
        finally:
            robot.disconnect()


if __name__ == "__main__":
    unittest.main()
