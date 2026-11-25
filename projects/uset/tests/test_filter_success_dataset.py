"""标注解析、安全检查及LeRobot v3 导出回归测试。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filter_success_dataset import (
    export_dataset,
    main,
    read_results,
    select_episodes,
    validate_paths,
    validate_source_episodes,
)


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def read(self, text):
        path = self.root / "episode_result.json"
        path.write_text(text, encoding="utf-8")
        return read_results(path)

    def test_jsonl_with_bom_and_blank_lines(self):
        self.assertEqual(self.read('\ufeff{"episode_index":0,"success":false}\n\n'
                                   '{"episode_index":1,"success":true}\n'), {0: False, 1: True})

    def test_array_and_single_object(self):
        for text in ['[{"episode_index":2,"success":true}]', '{"episode_index":2,"success":true}']:
            with self.subTest(text=text):
                self.assertEqual(self.read(text), {2: True})

    def test_invalid_records(self):
        for text in ['', 'null', '[1]', '{"episode_index":true,"success":true}',
                     '{"episode_index":-1,"success":true}', '{"episode_index":0,"success":"false"}',
                     '{"episode_index":0,"success":1}', '{"episode_index":0}',
                     '{"episode_index":0,"success":true}\nnot json',
                     '[{"episode_index":0,"success":true},{"episode_index":0,"success":false}]']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)

    def test_select_sorted_and_exclude_missing(self):
        self.assertEqual(select_episodes({3: True, 0: False, 1: True}, 5), ([1, 3], [0], [2, 4]))
        with self.assertRaises(ValueError):
            select_episodes({5: True}, 5)

    def test_reject_existing_or_nested_output(self):
        source = self.root / "source"
        source.mkdir()
        for output in [source, source / "filtered", self.root]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate_paths(source, output)
        output = self.root / "output"
        output.mkdir()
        with self.assertRaises(ValueError):
            validate_paths(source, output)

    def test_dry_run_does_not_create_output(self):
        source = self.root / "source"
        (source / "meta").mkdir(parents=True)
        (source / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "total_episodes": 2}))
        (source / "episode_result.json").write_text('[{"episode_index":1,"success":true}]')
        output = self.root / "output"
        self.assertEqual(main(["--source-root", str(source), "--output-root", str(output), "--dry-run"]), 0)
        self.assertFalse(output.exists())


class SourceValidationTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.files = {0: Path("first.parquet"), 1: Path("shared.parquet"), 2: Path("shared.parquet")}
        self.episodes = [{"episode_index": i, "length": 2, "data/chunk_index": 0,
                          "data/file_index": 0 if i == 0 else 1} for i in range(3)]
        self.dataset = SimpleNamespace(root=self.root, meta=SimpleNamespace(
            episodes=self.episodes, get_data_file_path=lambda i: self.files[i]))
        pq.write_table(pa.table({"episode_index": [0, 0]}), self.root / "first.parquet")
        pq.write_table(pa.table({"episode_index": [1, 1, 2, 2]}), self.root / "shared.parquet")

    def test_separate_and_shared_files(self):
        self.assertEqual(validate_source_episodes(self.dataset, [0, 1, 2]), 6)
        self.assertEqual(validate_source_episodes(self.dataset, [2]), 2)

    def test_missing_episode_has_file_and_id_not_nan_error(self):
        # 元数据错误地把 episode 1 指向只包含 episode 0 的文件。
        self.files[1] = self.files[0]
        with self.assertRaisesRegex(ValueError, r"episode 1.*first.parquet.*实际找到 0 帧"):
            validate_source_episodes(self.dataset, [1])

    def test_length_mismatch(self):
        self.episodes[1]["length"] = 3
        with self.assertRaisesRegex(ValueError, r"episode 1.*length=3.*实际找到 2 帧"):
            validate_source_episodes(self.dataset, [1])

    def test_nonfinite_metadata_and_wrong_episode_order(self):
        for field in ("episode_index", "length", "data/chunk_index", "data/file_index"):
            with self.subTest(field=field):
                saved = self.episodes[1][field]
                self.episodes[1][field] = float("nan")
                with self.assertRaisesRegex(ValueError, rf"episode 1.*{field}.*有限整数"):
                    validate_source_episodes(self.dataset, [1])
                self.episodes[1][field] = saved
        self.episodes[1]["episode_index"] = 2
        with self.assertRaisesRegex(ValueError, "编号或排序不一致"):
            validate_source_episodes(self.dataset, [1])

    def test_null_episode_id_rejected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"episode_index": pa.array([1, None], type=pa.int64())}),
                       self.root / "shared.parquet")
        with self.assertRaisesRegex(ValueError, "shared.parquet.*episode_index.*空值"):
            validate_source_episodes(self.dataset, [1])

    def test_invalid_source_fails_before_video_copy(self):
        self.files[1] = self.files[0]
        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset", return_value=self.dataset), \
             patch("lerobot.datasets.dataset_tools.split_dataset") as split:
            with self.assertRaisesRegex(ValueError, "episode 1"):
                export_dataset(self.root, self.root.parent / (self.root.name + "-output"),
                               self.root / "episode_result.json", "local/test", [1], [], [])
        split.assert_not_called()

    def test_debug_preserves_traceback(self):
        source = self.root / "source"
        (source / "meta").mkdir(parents=True)
        (source / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "total_episodes": 2}))
        (source / "episode_result.json").write_text('[{"episode_index":1,"success":true}]')
        error = io.StringIO()
        with patch("filter_success_dataset.export_dataset", side_effect=ValueError("cannot convert float NaN to integer")), \
             contextlib.redirect_stderr(error), contextlib.redirect_stdout(io.StringIO()):
            result = main(["--source-root", str(source), "--output-root", str(self.root / "output"), "--debug"])
        self.assertEqual(result, 1)
        self.assertIn("Traceback (most recent call last)", error.getvalue())
        self.assertIn("cannot convert float NaN to integer", error.getvalue())


class ExportTests(unittest.TestCase):
    def test_real_v3_export_reindexes_and_all_success_is_supported(self):
        import numpy as np
        import pyarrow.parquet as pq
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            dataset = LeRobotDataset.create(
                repo_id="local/filter_test", fps=10, root=source,
                features={
                    "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
                    "action": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
                }, use_videos=False,
            )
            for episode in range(3):
                for frame in range(2):
                    dataset.add_frame({
                        "observation.state": np.array([episode * 10 + frame], dtype=np.float32),
                        "action": np.array([episode], dtype=np.float32), "task": "test task",
                    })
                dataset.save_episode()
            dataset.finalize()
            annotation = source / "episode_result.json"
            annotation.write_text('[{"episode_index":0,"success":false},'
                                  '{"episode_index":1,"success":true},{"episode_index":2,"success":true}]')
            original_files = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            output = root / "filtered"
            self.assertEqual(export_dataset(source, output, annotation, "local/filter_test", [1, 2], [0], []), (2, 4))
            filtered = LeRobotDataset(repo_id="local/filter_test_success", root=output)
            self.assertEqual(filtered.num_episodes, 2)
            self.assertEqual(filtered.num_frames, 4)
            table = pq.read_table(sorted((output / "data").rglob("*.parquet"))).to_pandas()
            self.assertEqual(table["episode_index"].tolist(), [0, 0, 1, 1])
            self.assertEqual(table["index"].tolist(), [0, 1, 2, 3])
            self.assertEqual([float(np.asarray(a).reshape(-1)[0]) for a in table["action"]], [1, 1, 2, 2])
            self.assertEqual(read_results(output / "episode_result.json"), {0: True, 1: True})
            manifest = json.loads((output / "success_filter_manifest.json").read_text())
            self.assertEqual(manifest["episode_mapping"], [
                {"source_episode_index": 1, "episode_index": 0},
                {"source_episode_index": 2, "episode_index": 1},
            ])
            self.assertEqual(float(filtered.meta.stats["action"]["min"][0]), 1)
            self.assertEqual(float(filtered.meta.stats["action"]["max"][0]), 2)
            for relative, content in original_files.items():
                self.assertEqual((source / relative).read_bytes(), content, str(relative))
            second = root / "all_success"
            self.assertEqual(export_dataset(output, second, output / "episode_result.json",
                                            "local/filter_test_success", [0, 1], [], []), (2, 4))

    def test_video_export_keeps_references_readable(self):
        import av
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            key = "observation.images.front"
            dataset = LeRobotDataset.create(
                repo_id="local/video_test", fps=10, root=source,
                features={
                    "action": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
                    key: {"dtype": "video", "shape": (64, 64, 3),
                          "names": ["height", "width", "channels"]},
                }, video_backend="pyav", encoder_threads=1,
            )
            for episode in range(3):
                for _ in range(4):
                    dataset.add_frame({
                        "action": np.array([episode], dtype=np.float32),
                        key: np.full((64, 64, 3), 30 + episode * 80, dtype=np.uint8),
                        "task": "video test",
                    })
                dataset.save_episode()
            dataset.finalize()
            annotation = source / "episode_result.json"
            annotation.write_text('[{"episode_index":0,"success":false},'
                                  '{"episode_index":1,"success":true},'
                                  '{"episode_index":2,"success":true}]')
            output = root / "filtered"
            self.assertEqual(export_dataset(source, output, annotation,
                                            "local/video_test", [1, 2], [0], []), (2, 8))
            filtered = LeRobotDataset(repo_id="local/video_test_success", root=output,
                                      video_backend="pyav")
            for episode in range(2):
                metadata = filtered.meta.episodes[episode]
                path = output / filtered.meta.get_video_file_path(episode, key)
                self.assertTrue(path.is_file())
                with av.open(str(path)) as container:
                    frames = list(container.decode(video=0))
                start = round(metadata[f"videos/{key}/from_timestamp"] * 10)
                end = round(metadata[f"videos/{key}/to_timestamp"] * 10)
                self.assertEqual(end - start, 4)
                self.assertGreaterEqual(len(frames), end)
                expected_color = 30 + (episode + 1) * 80
                self.assertAlmostEqual(float(frames[start].to_ndarray(format="rgb24").mean()),
                                       expected_color, delta=8)


if __name__ == "__main__":
    unittest.main()
