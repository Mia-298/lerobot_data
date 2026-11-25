"""用真实 MuJoCo 接触验证放置评分；不加载策略，不打开窗口。"""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from myarm_eval_env import MyArmEvalEnv


class PlacementTests(unittest.TestCase):
    def setUp(self):
        # task 1: large cup -> large plate
        self.env = MyArmEvalEnv(task_id=1)
        self.addCleanup(self.env.close)
        obs, _ = self.env.reset(seed=42)
        self.action = np.clip(obs['agent_pos'],
                              self.env.action_space.low,
                              self.env.action_space.high)

    def put_cup(self, plate_name, gap=0.001):
        """只设置测试初始状态；之后由真实物理和 env.step 推进。"""
        model, data = self.env.robot.model, self.env.robot.data
        plate = model.geom(plate_name + '_geom').id
        joint = model.joint('cup_large_free').id
        qadr = model.jnt_qposadr[joint]
        vadr = model.jnt_dofadr[joint]
        half_height = model.geom_size[model.geom('cup_large_body').id, 1]
        position = data.geom_xpos[plate].copy()
        position[2] += model.geom_size[plate, 1] + half_height + gap
        data.qpos[qadr:qadr + 3] = position
        data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        data.qvel[vadr:vadr + 6] = 0
        mujoco.mj_forward(model, data)

    def test_correct_plate_without_lift_history(self):
        self.put_cup('plate_large')
        for _ in range(self.env._max_episode_steps):
            _, reward, terminated, truncated, info = self.env.step(self.action)
            if terminated or truncated:
                break
        self.assertTrue(info['is_success'], info)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0)
        self.assertGreaterEqual(info['placement_steps'],
                                self.env._placement_required_steps)
        self.env.reset(seed=42)
        self.assertEqual(self.env._placement_steps, 0)
        self.assertFalse(self.env._needs_reset)
        _, reward, terminated, _, info = self.env.step(self.action)
        self.assertFalse(info['is_success'])
        self.assertFalse(terminated)
        self.assertEqual(reward, 0.0)

    def test_wrong_plate(self):
        self.put_cup('plate_small')
        for _ in range(self.env._max_episode_steps):
            _, reward, terminated, truncated, info = self.env.step(self.action)
            self.assertFalse(info['is_success'], info)
            if terminated or truncated:
                break
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 0.0)

    def test_hovering_and_interrupted_hold(self):
        self.put_cup('plate_large', gap=0.10)
        # 仅推进一小步，物体仍悬空；长时间运行会自然落到盘子上。
        self.env._placement_steps = self.env._placement_required_steps - 1
        _, reward, terminated, _, info = self.env.step(self.action)
        self.assertTrue(info['centered_on_plate'])
        self.assertFalse(info['touching_plate'])
        self.assertFalse(info['is_success'])
        self.assertEqual(info['placement_steps'], 0)
        self.assertFalse(terminated)
        self.assertEqual(reward, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
