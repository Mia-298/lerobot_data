"""录制入口配置回归测试；不连接机器人、不启动录制。"""

import contextlib
import io
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import draccus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projects.user.record_datasets.record_my_arm import (  # noqa: E402
    MyDatasetRecordConfig,
    MyRecordConfig,
    RecordConfig,
    _print_recording_summary,
)


class RecordingConfigTests(unittest.TestCase):
    def assert_dataset_defaults(self, dataset, **overrides):
        expected = {
            'repo_id': 'mia/pick_place_16',
            'root': '/home/mia/data/smolvla_work/datasets/pick_place_16',
            'episode_time_s': 120,
            'fps': 30,
            'num_episodes': 1,
            'no_stamp': True,
            'push_to_hub': False,
        }
        expected.update(overrides)
        self.assertIsInstance(dataset, MyDatasetRecordConfig)
        for key, value in expected.items():
            self.assertEqual(getattr(dataset, key), value, key)

    def test_no_cli_overrides(self):
        self.assert_dataset_defaults(draccus.parse(MyRecordConfig, args=[]).dataset)

    def test_partial_cli_overrides_keep_other_defaults(self):
        for args, overrides in [
            (['--dataset.repo_id=mia/test'], {'repo_id': 'mia/test'}),
            (['--dataset.root=/tmp/myarm-test'], {'root': '/tmp/myarm-test'}),
            (['--dataset.fps=15'], {'fps': 15}),
            (['--dataset.streaming_encoding=true'], {}),
            (['--dataset.rgb_encoder.vcodec=h264'], {}),
        ]:
            with self.subTest(args=args):
                cfg = draccus.parse(MyRecordConfig, args=args)
                self.assert_dataset_defaults(cfg.dataset, **overrides)

    def test_explicit_episode_time_wins(self):
        for value in [5, 60, 240, 12.5]:
            with self.subTest(value=value):
                cfg = draccus.parse(MyRecordConfig, args=[
                    '--dataset.repo_id=mia/test',
                    f'--dataset.episode_time_s={value}',
                ])
                self.assert_dataset_defaults(cfg.dataset, repo_id='mia/test', episode_time_s=value)

    def test_partial_file_config_keeps_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.yaml'
            path.write_text('dataset:\n  repo_id: mia/from-file\n', encoding='utf-8')
            cfg = draccus.parse(MyRecordConfig, config_path=path, args=[])
        self.assert_dataset_defaults(cfg.dataset, repo_id='mia/from-file')

    def test_official_wrapper_conversion_preserves_dataset(self):
        cfg = draccus.parse(MyRecordConfig, args=['--dataset.episode_time_s=240'])
        converted = RecordConfig(**{
            item.name: getattr(cfg, item.name) for item in fields(RecordConfig) if item.init
        })
        self.assertIs(type(converted), RecordConfig)
        self.assertIs(converted.dataset, cfg.dataset)
        self.assertEqual(converted.dataset.episode_time_s, 240)


class RecordingSummaryTests(unittest.TestCase):
    def test_uses_latest_episode_length_not_cumulative_frames(self):
        dataset = SimpleNamespace(
            fps=30,
            num_frames=100000,
            meta=SimpleNamespace(latest_episode={'episode_index': [12], 'length': [540]}),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_recording_summary(dataset)
        self.assertIn('episode 12', output.getvalue())
        self.assertIn('540 帧', output.getvalue())
        self.assertIn('18.00s', output.getvalue())

    def test_no_saved_episode(self):
        dataset = SimpleNamespace(meta=SimpleNamespace(latest_episode=None))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_recording_summary(dataset)
        self.assertIn('未保存新的 episode', output.getvalue())


if __name__ == '__main__':
    unittest.main()
