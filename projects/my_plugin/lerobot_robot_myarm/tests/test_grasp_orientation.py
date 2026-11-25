"""Offline orientation regressions: no connection, viewer, recording or robot I/O.

Run with unittest discover from this directory. Reference writes use temporary dirs.
"""
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import test_grasp_safety as s

o = importlib.import_module(f"{s.pkg.__name__}.grasp_orientation")


def rz(degrees):
    return Rotation.from_euler("z", degrees, degrees=True).as_matrix()


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "reference.json"
        self.reference = o.AprilTagRotationReference(self.path)
        # Intentionally noncommuting with Rz: distinguish base Z from tool Z.
        self.ee = Rotation.from_euler("xyz", [165, 20, -12], degrees=True).as_matrix()
        self.tag = Rotation.from_euler("xyz", [180, 0, 24], degrees=True).as_matrix()
        self.reference.capture_missing({0: self.tag, 4: self.tag, 5: self.tag}, self.ee)

    def test_cups_base_z_90_and_non_cup_zero(self):
        for tag_id in (4, 5):
            target, yaw = self.reference.grasp_rotation(tag_id, self.tag)
            np.testing.assert_allclose(target, rz(90) @ self.ee, atol=1e-12)
            self.assertAlmostEqual(yaw, 0.)
            self.assertGreater(o.rotation_error(target, self.ee @ rz(90)), 1.)
        target, _ = self.reference.grasp_rotation(0, self.tag)
        np.testing.assert_allclose(target, self.ee, atol=1e-12)

    def test_relative_tag_yaw_is_added_to_cup_offset(self):
        for tag_id, degrees in ((0, 30), (4, 120), (5, 120)):
            target, yaw = self.reference.grasp_rotation(tag_id, rz(30) @ self.tag)
            np.testing.assert_allclose(target, rz(degrees) @ self.ee, atol=1e-12)
            self.assertAlmostEqual(np.rad2deg(yaw), 30.)
        target, yaw = self.reference.grasp_rotation(4, rz(30) @ self.tag, follow_tag_yaw=False)
        np.testing.assert_allclose(target, rz(90) @ self.ee, atol=1e-12)
        self.assertAlmostEqual(yaw, 0.)
        target, _ = self.reference.grasp_rotation(4, self.tag, cup_yaw_offset_deg=-90)
        np.testing.assert_allclose(target, rz(-90) @ self.ee, atol=1e-12)

    def test_reload_does_not_rezero_or_accumulate_cup_offset(self):
        saved = self.path.read_bytes()
        goal, _ = self.reference.grasp_rotation(4, self.tag)
        for _ in range(3):
            reference = o.AprilTagRotationReference(self.path)
            reference.capture_missing({4: rz(30) @ self.tag}, goal)
            self.assertEqual(self.path.read_bytes(), saved)
            again, _ = reference.grasp_rotation(4, self.tag)
            np.testing.assert_allclose(again, goal, atol=1e-12)
        reference.capture_missing({1: self.tag}, goal)
        np.testing.assert_allclose(reference.ee_reference, self.ee, atol=1e-12)
        self.assertIn("1", o.AprilTagRotationReference(self.path).tags)

    def test_invalid_or_missing_reference_fails_closed(self):
        with self.assertRaises(ValueError):
            self.reference.grasp_rotation(3, self.tag)
        for invalid in (np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(4)):
            with self.assertRaises(ValueError):
                self.reference.capture_missing({1: invalid}, self.ee)
            self.assertNotIn("1", self.reference.tags)
        self.path.write_text(json.dumps({"version": 1, "frame": "tool", "ee_base": self.ee.tolist(), "tags_base": {}}))
        with self.assertRaises(ValueError):
            o.AprilTagRotationReference(self.path)
        empty = o.AprilTagRotationReference(Path(self.directory.name) / "missing.json")
        with self.assertRaises(ValueError):
            empty.grasp_rotation(4, self.tag)

    def test_shortest_rotation_step_is_bounded_in_base(self):
        target = rz(90) @ self.ee
        next_rotation = o.step_rotation(self.ee, target, .02)
        self.assertAlmostEqual(o.rotation_error(self.ee, next_rotation), .02)
        np.testing.assert_allclose(next_rotation, rz(np.rad2deg(.02)) @ self.ee, atol=1e-12)
        np.testing.assert_allclose(o.step_rotation(self.ee, target, 2.), target, atol=1e-12)
        for step in (0, -.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                o.step_rotation(self.ee, target, step)


class ProcessorOrientationTests(unittest.TestCase):
    def test_zero_xyz_rotation_reaches_ik_and_preserves_nonaxial_tcp(self):
        names = [f"q{i}" for i in range(6)]
        offset = np.array([.025, -.018, .038])
        step = s.processor.KeyboardIKActionStep(
            names, [], [], np.eye(4), np.full(6, -7.), np.full(6, 7.),
            tcp_offset_ee=offset, rotation_step_rad=.02)
        q = np.array([.2, -.3, .5, .3, -.2, .1])

        def fk(joints):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("xyz", joints[3:]).as_matrix()
            T[:3, 3] = joints[:3]
            return T

        def ik(**kwargs):
            return np.r_[kwargs["pos_target"], kwargs["rpy_target"]], True, 1

        initial = fk(q)
        goal = rz(90) @ initial[:3, :3]
        step._current_transition = {s.processor.TransitionKey.OBSERVATION: {
            **{f"{name}.pos": float(value) for name, value in zip(names, q)}, "gripper.pos": 1.}}
        with patch.object(step, "_ee_pose", side_effect=fk), \
             patch.object(s.processor.IK_learn, "inverse_kinematics", side_effect=ik) as solver, \
             contextlib.redirect_stdout(io.StringIO()):
            output = step.action({"delta_x": 0., "delta_y": 0., "delta_z": 0., o.ROTATION_TARGET_KEY: goal})
        solver.assert_called_once()
        next_q = np.array([output[f"{name}.pos"] for name in names])
        actual = fk(next_q)
        self.assertGreater(np.linalg.norm(next_q - q), .001)
        self.assertAlmostEqual(o.rotation_error(initial[:3, :3], actual[:3, :3]), .02)
        np.testing.assert_allclose(actual[:3, 3] + actual[:3, :3] @ offset,
                                   initial[:3, 3] + initial[:3, :3] @ offset, atol=1e-12)
        self.assertNotIn(o.ROTATION_TARGET_KEY, output)
        self.assertEqual(set(output), {f"{name}.pos" for name in names} | {"gripper.pos"})

    def test_real_pipeline_zero_xyz_converges_to_90_degrees(self):
        cfg = s.config.MyArmConfig()
        model = s.mujoco.MjModel.from_xml_path(cfg.xml_path)
        auto_config = s.auto.MyArmAutoConfig(rotation_step_rad=.02)
        pipeline = s.processor.make_myarm_processor(cfg, auto_config=auto_config)
        step = pipeline.steps[0]
        self.assertEqual(step.rotation_step_rad, auto_config.rotation_step_rad)
        calibration = s.yaml.safe_load(Path(cfg.handeye_calibration_path).read_text())
        np.testing.assert_allclose(step.tcp_offset_ee, np.array(calibration["fc"]["transformation_matrix"])[:3, 3])
        q = model.key_qpos[model.key("home").id, :6].copy()
        initial = step._ee_pose(q)
        tcp = initial[:3, 3] + initial[:3, :3] @ step.tcp_offset_ee
        goal = rz(90) @ initial[:3, :3]
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(110):
                observation = {f"{name}.pos": float(value) for name, value in zip(cfg.joint_names, q)}
                observation["gripper.pos"] = 1.
                output = pipeline(({"delta_x": 0., "delta_y": 0., "delta_z": 0., o.ROTATION_TARGET_KEY: goal}, observation))
                next_q = np.array([output[f"{name}.pos"] for name in cfg.joint_names])
                self.assertLessEqual(np.max(np.abs(next_q - q)), step.max_joint_step + 1e-12)
                self.assertNotIn(o.ROTATION_TARGET_KEY, output)
                q = next_q
        final = step._ee_pose(q)
        self.assertLess(o.rotation_error(final[:3, :3], goal), .003)
        self.assertLess(np.linalg.norm(final[:3, 3] + final[:3, :3] @ step.tcp_offset_ee - tcp), .001)


class HoverGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = s.mujoco.MjModel.from_xml_path(s.config.MyArmConfig().xml_path)

    def test_live_hover_target_and_measured_rotation_gate_descent(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = s.myarm.MyArm(s.config.MyArmConfig(task_id=15))
            teleop = s.auto.MyArmAuto(s.auto.MyArmAutoConfig(
                apriltag_rotation_reference_path=str(Path(directory) / "reference.json")))
            teleop.bind_robot(robot)
            robot.model = self.model
            robot.data = s.mujoco.MjData(self.model)
            teleop.table_clearance = Mock()
            teleop.table_clearance.minimum_tcp_z.return_value = .25
            teleop._clearance_model = self.model
            teleop.controller.set_targets([.4, -.3, .05], [.5, -.3, .05], .05)
            # Cached detection-time hover is stale; only the live controller target is relevant.
            teleop.above_pick_position = np.array([.4, -.3, .1])
            teleop.grasp_rotation_base = rz(90) @ s.DOWN

            def pose(tcp, rotation):
                T = s.transform(rotation)
                T[:3, 3] = np.array(tcp) - rotation @ robot.config.T_G_F[:3, 3]
                return T

            with patch.object(s.auto, "get_site_pose_in_base", return_value=pose([.4, -.3, .241], s.DOWN)) as measured, \
                 contextlib.redirect_stdout(io.StringIO()):
                approaching = teleop.get_action()
                self.assertNotIn(o.ROTATION_TARGET_KEY, approaching)
                self.assertGreater(approaching["delta_z"], 0.)
                measured.return_value = pose([.4, -.3, .25], s.DOWN)
                for _ in range(100):  # A timer cannot substitute for orientation feedback.
                    turning = teleop.get_action()
                    self.assertEqual(teleop.state, s.state.PickPlaceState.MOVE_ABOVE_PICK)
                    self.assertEqual(turning["delta_z"], 0.)
                    np.testing.assert_allclose(turning[o.ROTATION_TARGET_KEY], teleop.grasp_rotation_base)
                measured.return_value = pose([.4, -.3, .25], teleop.grasp_rotation_base)
                teleop.get_action()
                self.assertEqual(teleop.state, s.state.PickPlaceState.MOVE_TO_PICK)
                # Lower the test guard to allow a descent target below the hover.
                teleop.table_clearance.minimum_tcp_z.return_value = .03
                teleop.controller.targets[s.state.PickPlaceState.MOVE_TO_PICK][2] = .05
                descending = teleop.get_action()
                self.assertLess(descending["delta_z"], 0.)
                self.assertIn(o.ROTATION_TARGET_KEY, descending)
            teleop.bind_robot(robot)
            self.assertFalse(teleop._orientation_active)
            self.assertIsNone(teleop.grasp_rotation_base)


if __name__ == "__main__":
    unittest.main()
