"""配置层单元测试：不导入训练依赖、不加载权重、不启动训练。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_smolvla import TrainingSettings, build_arguments, lerobot_arguments, validate_paths


class TrainingConfigTests(unittest.TestCase):
    def test_smoke_uses_short_schedule_and_dataset_features(self):
        args = build_arguments(TrainingSettings())
        for flag in ('--steps=20', '--policy.scheduler_warmup_steps=5',
                     '--policy.scheduler_decay_steps=20', '--policy.input_features=null',
                     '--save_freq=20', '--policy.push_to_hub=false'):
            self.assertIn(flag, args)

    def test_train_uses_editable_settings(self):
        args = build_arguments(TrainingSettings(mode='train', steps=8000,
                               learning_rate=5e-5, batch_size=2, save_freq=400))
        for flag in ('--steps=8000', '--policy.optimizer_lr=5e-05',
                     '--batch_size=2', '--save_freq=400'):
            self.assertIn(flag, args)

    def test_resume_does_not_override_original_training_config(self):
        path = Path('/tmp/example-checkpoint/pretrained_model/train_config.json')
        self.assertEqual(build_arguments(TrainingSettings(mode='resume', resume_config=path)),
                         ['--resume=true', f'--config_path={path}'])

    def test_invalid_settings_fail_early(self):
        for fields in ({'mode': 'invalid'}, {'mode': 'resume'}, {'batch_size': 0},
                       {'run_name': '../outside'}, {'mixed_precision': 'invalid'},
                       {'smoke_warmup_steps': 20}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                build_arguments(TrainingSettings(**fields))

    def test_argument_context_restores_sys_argv_after_failure(self):
        original = sys.argv
        with self.assertRaises(RuntimeError):
            with lerobot_arguments(['--steps=20']):
                self.assertEqual(sys.argv[1:], ['--steps=20'])
                raise RuntimeError('expected')
        self.assertIs(sys.argv, original)

    def test_fresh_path_check_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, model = root / 'dataset', root / 'model'
            (dataset / 'meta').mkdir(parents=True)
            model.mkdir()
            for path in (dataset / 'meta/info.json', model / 'config.json', model / 'model.safetensors'):
                path.write_text('{}')
            settings = TrainingSettings(dataset_root=dataset, model_path=model,
                                        output_parent=root / 'outputs', run_name='test')
            args = build_arguments(settings)
            validate_paths(settings, args)
            self.assertFalse((root / 'outputs').exists())
            (root / 'outputs/test').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                validate_paths(settings, args)

    def test_resume_checks_training_state_and_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'pretrained_model/train_config.json'
            config.parent.mkdir()
            config.write_text(json.dumps({'steps': 100}))
            settings = TrainingSettings(mode='resume', resume_config=config)
            args = build_arguments(settings)
            with self.assertRaises(FileNotFoundError):
                validate_paths(settings, args)
            state = root / 'training_state/training_step.json'
            state.parent.mkdir()
            state.write_text(json.dumps({'step': 50}))
            validate_paths(settings, args)
            state.write_text(json.dumps({'step': 100}))
            with self.assertRaises(ValueError):
                validate_paths(settings, args)


if __name__ == '__main__':
    unittest.main()
