"""离线回归：只加载模型/计算 FK、IK，从不 connect、启动 viewer 或控制机器人。

运行：python -m unittest discover -s projects/my_plugin/lerobot_robot_myarm/tests -v
"""
import contextlib
import importlib
import io
from itertools import product
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import yaml

# 避免插件 __init__ 导入键盘监听器；保持真实子模块、FK/IK 和模型不变。
PACKAGE = Path(__file__).resolve().parents[1] / "lerobot_robot_myarm"
pkg = types.ModuleType("_myarm_safety_offline")
pkg.__path__ = [str(PACKAGE)]
sys.modules[pkg.__name__] = pkg
safety = importlib.import_module(f"{pkg.__name__}.grasp_safety")
state = importlib.import_module(f"{pkg.__name__}.pick_place_state")
processor = importlib.import_module(f"{pkg.__name__}.processor_myarm")
config = importlib.import_module(f"{pkg.__name__}.config_myarm")
myarm = importlib.import_module(f"{pkg.__name__}.myarm")
auto = importlib.import_module(f"{pkg.__name__}.myarm_auto")


def transform(rotation=None, z=0.0):
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    T[2, 3] = z
    return T


DOWN = np.diag([-1.0, 1.0, -1.0])


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config.MyArmConfig()
        cls.model = mujoco.MjModel.from_xml_path(cls.cfg.xml_path)
        cls.guard = cls.make_guard()
        cls.tcp = np.asarray(yaml.safe_load(Path(cls.cfg.handeye_calibration_path).read_text())["fc"]["transformation_matrix"])

    @classmethod
    def make_guard(cls, surfaces=("table_top", "work_platform"), margin=0.01):
        return safety.TableClearance(cls.model, cls.cfg.ee_name,
                                     cls.cfg.gripper_joint_names, surfaces, margin)

    def test_palm_tcp_matches_model(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        geom = self.model.geom("gripper_palm").id
        site = self.model.site(self.cfg.ee_name).id
        # 掌部 local +Y 面中心（指尖所在一面），独立从模型几何求 TCP。
        point_world = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ [0, self.model.geom_size[geom, 1], 0]
        point_ee = data.site_xmat[site].reshape(3, 3).T @ (point_world - data.site_xpos[site])
        np.testing.assert_allclose(self.tcp[:3, 3], point_ee, atol=1e-12)
        raw = yaml.safe_load(Path(self.cfg.handeye_calibration_path).read_text())["fc"]["translation"]
        np.testing.assert_allclose([raw[k] for k in ("x", "y", "z")], point_ee, atol=1e-12)

    def test_downward_floor_table_and_platform(self):
        table = self.make_guard(("table_top",))
        self.assertAlmostEqual(table.surface_z_base, -0.05)
        self.assertAlmostEqual(self.guard.surface_z_base, -0.04)
        self.assertAlmostEqual(table.minimum_tcp_z(transform(DOWN), self.tcp), 0.037)
        self.assertAlmostEqual(self.guard.minimum_tcp_z(transform(DOWN), self.tcp), 0.047)

    def test_tilted_open_close_envelope_against_mujoco(self):
        rng = np.random.default_rng(471)
        data = mujoco.MjData(self.model)
        site = self.model.site(self.cfg.ee_name).id
        base = self.model.body("base").id
        signs = np.asarray(list(product((-1., 1.), repeat=3)))
        for _ in range(60):
            for name in self.cfg.joint_names + self.cfg.gripper_joint_names:
                j = self.model.joint(name).id
                data.qpos[self.model.jnt_qposadr[j]] = rng.uniform(*self.model.jnt_range[j])
            mujoco.mj_forward(self.model, data)
            R_base = data.xmat[base].reshape(3, 3)
            R_ee = R_base.T @ data.site_xmat[site].reshape(3, 3)
            ee = R_base.T @ (data.site_xpos[site] - data.xpos[base])
            floor = self.guard.minimum_ee_z(R_ee)
            # 把真实几何平移到计算出的 EE floor，再独立检查所有角点。
            for name in ("gripper_palm", "left_pad", "right_pad"):
                g = self.model.geom(name).id
                corners_world = (signs * self.model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                corners_base = (corners_world - data.xpos[base]) @ R_base
                lowest_z = corners_base[:, 2].min() + floor - ee[2]
                self.assertGreaterEqual(lowest_z, self.guard.surface_z_base + self.guard.margin - 1e-12)

    def test_build_guard_does_not_change_existing_data(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = np.arange(self.model.nq) * 0.01
        before = data.qpos.copy()
        self.make_guard()
        np.testing.assert_array_equal(data.qpos, before)

    def test_invalid_geometry_inputs_fail_closed(self):
        for margin in (-0.01, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.make_guard(margin=margin)
        with self.assertRaises(ValueError):
            self.make_guard(())
        for R in (np.zeros((3, 3)), np.eye(3) * np.nan, np.diag([-1., 1., 1.])):
            with self.assertRaises(ValueError):
                self.guard.minimum_ee_z(R)
        with self.assertRaises(RuntimeError):
            self.guard.require_safe_ee(transform(DOWN, 0.084))

    def test_real_fk_matches_mujoco(self):
        _, ts, axes = processor.get_joint_initial_info(self.model, self.cfg.joint_names)
        end = processor.get_T_joint_to_site(self.model, self.cfg.joint_names[-1], self.cfg.ee_name)
        step = processor.KeyboardIKActionStep(self.cfg.joint_names, ts, axes, end,
                                              np.full(6, -7.), np.full(6, 7.), self.guard)
        data = mujoco.MjData(self.model)
        rng = np.random.default_rng(7)
        for _ in range(12):
            q = rng.uniform(-2, 2, 6)
            for n, value in zip(self.cfg.joint_names, q):
                data.qpos[self.model.jnt_qposadr[self.model.joint(n).id]] = value
            expected = myarm.get_site_pose_in_base(self.model, data, self.cfg.ee_name)
            np.testing.assert_allclose(step._ee_pose(q), expected, atol=1e-10)


class StateMachineTests(unittest.TestCase):
    def test_max_both_branches_pick_and_place(self):
        sm = state.PickPlaceStateMachine()
        sm.set_targets([0, 0, -0.03], [1, 0, -0.04], .0465, minimum_tcp_z=.047)
        self.assertAlmostEqual(sm.targets[state.PickPlaceState.MOVE_TO_PICK][2], .047)
        self.assertAlmostEqual(sm.targets[state.PickPlaceState.MOVE_TO_PLACE][2], .047)
        sm.set_targets([0, 0, .08], [1, 0, .06], .0465, minimum_tcp_z=.047)
        self.assertAlmostEqual(sm.targets[state.PickPlaceState.MOVE_TO_PICK][2], .08)
        self.assertAlmostEqual(sm.targets[state.PickPlaceState.MOVE_TO_PLACE][2], .1065)

    def test_unsafe_feedback_cannot_close_or_move(self):
        sm = state.PickPlaceStateMachine()
        sm.set_targets([0, 0, 0], [1, 0, 0], .045, minimum_tcp_z=.047)
        sm.state = state.PickPlaceState.MOVE_TO_PICK
        with self.assertRaises(RuntimeError):
            sm.step([0, 0, .046])
        self.assertEqual(sm.state, state.PickPlaceState.FAILED)
        self.assertEqual(sm.gripper_target, 1.)

    def test_floor_update_never_lowers_existing_targets(self):
        sm = state.PickPlaceStateMachine()
        sm.set_targets([0, 0, 0], [1, 0, 0], .045, minimum_tcp_z=.047)
        sm.step([0, 0, .2], minimum_tcp_z=.060)
        sm.step([0, 0, .2], minimum_tcp_z=.050)
        self.assertGreaterEqual(sm.targets[state.PickPlaceState.MOVE_TO_PICK][2], .060)


class ProcessorTests(unittest.TestCase):
    """可控笛卡尔 FK/IK：检验反馈滞后、候选姿态和插值保护。"""
    @classmethod
    def setUpClass(cls):
        # 此测试类也可以单独运行，不依赖 unittest 的类执行顺序。
        GeometryTests.setUpClass()

    def setUp(self):
        self.guard = GeometryTests.guard
        self.names = [f"q{i}" for i in range(6)]
        self.step = processor.KeyboardIKActionStep(self.names, [], [], np.eye(4),
                                                   np.full(6, -7.), np.full(6, 7.), self.guard)
        self.step._ee_pose = lambda q: transform(DOWN, q[2])
        self.q = np.array([0., 0., .10, 0., 0., 0.])
        self.set_feedback(self.q)

    def set_feedback(self, q):
        obs = {f"{n}.pos": v for n, v in zip(self.names, q)}
        obs["gripper.pos"] = 1.
        self.step._current_transition = {processor.TransitionKey.OBSERVATION: obs}

    @staticmethod
    def fake_ik(**kwargs):
        q = kwargs["q_init"].copy()
        q[:3] = kwargs["pos_target"]
        return q, True, 1

    def act(self, dz=0., gripper=1.):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.step.action(dict(delta_x=0., delta_y=0., delta_z=dz, **{"gripper.pos": gripper}))

    def test_repeated_descent_with_stale_feedback_is_clamped(self):
        with patch.object(processor.IK_learn, "inverse_kinematics", side_effect=self.fake_ik):
            for _ in range(25):
                output = self.act(-1.)  # 反馈固定 100mm，命令持续下降。
                self.assertGreaterEqual(output["q2.pos"], .085 - 1e-9)
            self.assertAlmostEqual(output["q2.pos"], .086)  # 1mm IK 数值缓冲

    def test_unsafe_ik_solution_is_rejected(self):
        def bad_ik(**kwargs):
            q = kwargs["q_init"].copy()
            q[2] = .084
            return q, True, 1
        with patch.object(processor.IK_learn, "inverse_kinematics", side_effect=bad_ik):
            output = self.act(-1.)
        self.assertEqual(output["q2.pos"], .1)

    def test_unsafe_zero_motion_and_gripper_command_raise(self):
        self.q[2] = .084
        self.set_feedback(self.q)
        with self.assertRaises(RuntimeError):
            self.act(gripper=0.)
        self.assertEqual(self.step._gripper_target, 1.)

    def test_safe_endpoints_with_unsafe_interpolation_are_rejected(self):
        self.step._ee_pose = lambda q: transform(DOWN, .1 - .03 * np.sin(np.pi * q[0] / .1))
        def ik(**kwargs):
            q = kwargs["q_init"].copy()
            q[0] = .1
            return q, True, 1
        with patch.object(processor.IK_learn, "inverse_kinematics", side_effect=ik):
            output = self.act(-1.)
        self.assertEqual(output["q0.pos"], 0.)

    def test_candidate_tilt_requires_its_own_floor(self):
        self.q[2] = .086
        self.set_feedback(self.q)
        tilt = .4
        Ry = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
        self.step._ee_pose = lambda q: transform(DOWN if q[0] == 0 else Ry @ DOWN, q[2])
        def ik(**kwargs):
            q = kwargs["q_init"].copy()
            q[0] = .001
            return q, True, 1
        with patch.object(processor.IK_learn, "inverse_kinematics", side_effect=ik):
            output = self.act(-1.)
        self.assertEqual(output["q0.pos"], 0.)

    def test_manual_mode_keeps_unclamped_behavior(self):
        self.step.table_clearance = None
        with patch.object(processor.IK_learn, "inverse_kinematics", side_effect=self.fake_ik):
            for _ in range(12):
                output = self.act(-1.)
        self.assertAlmostEqual(output["q2.pos"], .076)

    def test_nonfinite_action_and_feedback_are_rejected(self):
        with self.assertRaises(ValueError):
            self.act(np.nan)
        self.q[0] = np.nan
        self.set_feedback(self.q)
        with self.assertRaises(ValueError):
            self.act()

    def test_real_ik_descent_stays_safe(self):
        cfg = GeometryTests.cfg
        model = GeometryTests.model
        _, ts, axes = processor.get_joint_initial_info(model, cfg.joint_names)
        end = processor.get_T_joint_to_site(model, cfg.joint_names[-1], cfg.ee_name)
        step = processor.KeyboardIKActionStep(cfg.joint_names, ts, axes, end,
                                              np.full(6, -7.), np.full(6, 7.), self.guard)
        q = model.key_qpos[model.key("home").id, :6].copy()
        previous_z = step._ee_pose(q)[2, 3]
        initial_z = previous_z
        for _ in range(12):
            obs = {f"{n}.pos": v for n, v in zip(cfg.joint_names, q)}
            obs["gripper.pos"] = 1.
            step._current_transition = {processor.TransitionKey.OBSERVATION: obs}
            with contextlib.redirect_stdout(io.StringIO()):
                output = step.action(dict(delta_x=0., delta_y=0., delta_z=-1.))
            q = np.array([output[f"{n}.pos"] for n in cfg.joint_names])
            pose = step._ee_pose(q)
            self.guard.require_safe_ee(pose)
            self.assertLessEqual(pose[2, 3], previous_z + .0001)
            previous_z = pose[2, 3]
        self.assertLess(previous_z, initial_z - .02)


class StartupAndDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(config.MyArmConfig().xml_path)

    def setUp(self):
        # 只实例化适配器；注入独立模型，不调用 connect 或访问运行中的仿真。
        self.robot = myarm.MyArm(config.MyArmConfig(task_id=15))
        self.reference_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.reference_dir.cleanup)
        self.teleop = auto.MyArmAuto(auto.MyArmAutoConfig(
            apriltag_rotation_reference_path=str(Path(self.reference_dir.name) / "reference.json"),
        ))
        self.teleop.bind_robot(self.robot)

    def ready(self):
        self.robot.model = self.model
        self.robot.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.robot.data)

    def test_bind_before_model_load_defers_guard(self):
        self.assertIsNone(self.teleop.table_clearance)
        with self.assertRaisesRegex(RuntimeError, "模型尚未就绪"):
            self.teleop._ensure_table_clearance()
        self.ready()
        guard = self.teleop._ensure_table_clearance()
        self.assertIs(guard, self.teleop._ensure_table_clearance())
        self.teleop.bind_robot(self.robot)
        self.assertIsNone(self.teleop.table_clearance)
        self.assertIsNot(guard, self.teleop._ensure_table_clearance())

    def test_guard_rebuilds_for_reloaded_model(self):
        self.ready()
        guard = self.teleop._ensure_table_clearance()
        self.robot.model = mujoco.MjModel.from_xml_path(self.robot.config.xml_path)
        self.robot.data = mujoco.MjData(self.robot.model)
        self.assertIsNot(guard, self.teleop._ensure_table_clearance())

    def test_guard_none_model_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "已加载"):
            safety.TableClearance(None, "unused", [], [])

    def test_intrinsics_follow_actual_image_size(self):
        low = self.teleop._camera_matrix(np.empty((480, 640, 3)))
        high = self.teleop._camera_matrix(np.empty((960, 1280, 3)))
        self.assertAlmostEqual(high[0, 0], 2 * low[0, 0])
        self.assertEqual(high[0, 2], 639.5)
        self.assertEqual(high[1, 2], 479.5)
        for fovy in (0, 180, float("nan")):
            self.teleop.config.camera_fovy_deg = fovy
            with self.assertRaises(ValueError):
                self.teleop._camera_matrix(np.empty((480, 640, 3)))

    def test_detection_requests_highres_and_uses_matching_intrinsics(self):
        self.ready()
        image = np.zeros((960, 1280, 3), dtype=np.uint8)
        self.robot.get_images = Mock(return_value=image)
        detections = [types.SimpleNamespace(tag_id=n, side_lengths=np.full(4, 40.))
                      for n in (self.teleop.object_tag_id, self.teleop.plate_tag_id)]
        with patch.object(auto, "detect_apriltags", return_value=detections), \
             patch.object(auto, "detection_to_T", return_value=transform(z=.1)) as pnp, \
             patch.object(self.teleop, "_base_camera_pose_from_ee", return_value=np.eye(4)), \
             contextlib.redirect_stdout(io.StringIO()):
            result = self.teleop.get_target_detections()
        self.assertEqual(result, tuple(detections))
        self.robot.get_images.assert_called_once_with("wrist", hide_tags=False, width=1280, height=960)
        for call in pnp.call_args_list:
            np.testing.assert_allclose(call.args[1], self.teleop._camera_matrix(image))
        self.assertTrue(self.teleop.targets_locked)

    def test_renderer_caches_are_separate_and_resize_closes_old(self):
        self.ready()
        self.robot._is_connected = True
        def make_renderer(model, *, height, width):
            renderer = Mock()
            renderer.render.return_value = np.zeros((height, width, 3), dtype=np.uint8)
            return renderer
        with patch.object(myarm.mujoco, "Renderer", side_effect=make_renderer) as factory:
            low = self.robot.get_images("wrist")
            low_renderer = self.robot._renderer
            high = self.robot.get_images("wrist", width=1280, height=960)
            high_renderer = self.robot._detection_renderer
            self.robot.get_images("wrist", width=1280, height=960)
            self.robot.get_images("wrist")
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(low.shape, (480, 640, 3))
            self.assertEqual(high.shape, (960, 1280, 3))
            self.assertIs(self.robot._renderer, low_renderer)
            self.robot.get_images("wrist", width=1920, height=1440)
            high_renderer.close.assert_called_once()
            newest = self.robot._detection_renderer
            self.robot.disconnect()
            low_renderer.close.assert_called_once()
            newest.close.assert_called_once()
            self.assertFalse(self.robot.is_connected)

    def test_invalid_detection_render_dimensions_rejected(self):
        self.ready()
        self.robot._is_connected = True
        for width in (0, -1, True, 1280.0):
            with self.assertRaises(ValueError):
                self.robot.get_images("wrist", width=width, height=960)

    def test_disconnect_cleans_both_renderers_even_if_first_close_fails(self):
        self.robot._renderer = Mock()
        self.robot._renderer.close.side_effect = RuntimeError("close failed")
        detection = self.robot._detection_renderer = Mock()
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            self.robot.disconnect()
        detection.close.assert_called_once()
        self.assertIsNone(self.robot.model)
        self.assertIsNone(self.robot._renderer)
        self.assertIsNone(self.robot._detection_renderer)


if __name__ == "__main__":
    unittest.main()
