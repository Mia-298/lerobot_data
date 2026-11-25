#!/usr/bin/env python3
"""根据 episode_result.json / JSONL 导出只包含成功 episode 的 LeRobot v3 数据集。

不会修改原始数据，不会上传 Hugging Face。路径中包含 * 时务必加引号。
只预览筛选结果时加 --dry-run；默认读取源目录下的 episode_result.json。
"""

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from collections import Counter
from numbers import Real
from pathlib import Path


def read_results(path: Path) -> dict[int, bool]:
    """兼容 JSON 数组、单个对象，以及每行一个对象的 JSONL（含 .json 后缀）。"""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"标注文件为空：{path}")
    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"标注第 {line_number} 行不是有效 JSON：{exc.msg}") from exc
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        raise ValueError("标注应为 JSON 数组，或每行一个对象的 JSONL")

    results: dict[int, bool] = {}
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {position} 条标注必须是对象")
        index = record.get("episode_index")
        success = record.get("success")
        if type(index) is not int or index < 0:
            raise ValueError(f"第 {position} 条 episode_index 必须是非负整数")
        if type(success) is not bool:
            raise ValueError(f"episode {index} 的 success 必须是 JSON 布尔值 true/false")
        if index in results:
            raise ValueError(f"episode {index} 重复标注，请先消除重复或冲突")
        results[index] = success
    return results


def select_episodes(results: dict[int, bool], total: int) -> tuple[list[int], list[int], list[int]]:
    """返回成功、失败、未标注的原 episode 编号；未标注者不导出。"""
    if type(total) is not int or total < 1:
        raise ValueError("源数据集 total_episodes 必须是正整数")
    invalid = sorted(set(results) - set(range(total)))
    if invalid:
        raise ValueError(f"标注含超出数据集范围的 episode：{invalid}；有效范围为 0..{total - 1}")
    kept = sorted(index for index, success in results.items() if success)
    failed = sorted(index for index, success in results.items() if not success)
    missing = sorted(set(range(total)) - set(results))
    return kept, failed, missing


def validate_paths(source: Path, output: Path) -> None:
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出目录不能与源目录相同，也不能互相嵌套")
    if output.exists() or output.is_symlink():
        raise ValueError(f"输出目录已存在，拒绝覆盖：{output}")


def write_provenance(
    output: Path, source: Path, annotation: Path, kept: list[int], failed: list[int], missing: list[int]
) -> None:
    # 输出标注使用新编号，同时保留原编号，防止二次筛选时误用源编号。
    records = [
        {"episode_index": new, "success": True, "source_episode_index": old}
        for new, old in enumerate(kept)
    ]
    (output / "episode_result.json").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    manifest = {
        "source_root": str(source),
        "annotation_path": str(annotation),
        "kept_source_episode_indices": kept,
        "failed_source_episode_indices": failed,
        "unlabeled_source_episode_indices": missing,
        "episode_mapping": [
            {"source_episode_index": old, "episode_index": new} for new, old in enumerate(kept)
        ],
    }
    (output / "success_filter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_source_episodes(dataset, kept: list[int]) -> int:
    """在耗时的视频复制前核对元数据引用，避免空 episode 的 NaN 被转换为整数。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    def integer(value, context: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{context} 必须是有限整数，实际为 {value!r}")
        result = int(value)
        if result != value or result < 0:
            raise ValueError(f"{context} 必须是非负整数，实际为 {value!r}")
        return result

    files: dict[Path, dict[int, int]] = {}
    expected_frames = 0
    for index in kept:
        episode = dataset.meta.episodes[index]
        actual_index = integer(episode["episode_index"], f"episode {index} 元数据 episode_index")
        if actual_index != index:
            raise ValueError(f"元数据第 {index} 行实际为 episode {actual_index}，编号或排序不一致")
        length = integer(episode["length"], f"episode {index} 元数据 length")
        if length == 0:
            raise ValueError(f"episode {index} 的元数据 length 为 0，不能导出空 episode")
        for field in ("data/chunk_index", "data/file_index"):
            integer(episode[field], f"episode {index} 元数据 {field}")
        path = dataset.root / dataset.meta.get_data_file_path(index)
        files.setdefault(path, {})[index] = length
        expected_frames += length

    for path, episodes in files.items():
        try:
            table = pq.read_table(path, columns=["episode_index"])
        except (OSError, pa.ArrowException) as exc:
            raise ValueError(f"无法读取 episode {sorted(episodes)} 引用的数据文件：{path}：{exc}") from exc
        column = table.column("episode_index")
        if column.null_count or not pa.types.is_integer(column.type):
            raise ValueError(f"{path} 的 episode_index 含空值或不是整数列（{column.type}）")
        counts = Counter(column.to_pylist())
        for index, length in episodes.items():
            actual = counts[index]
            if actual != length:
                raise ValueError(
                    f"episode {index} 的元数据与 parquet 不一致：{path}；"
                    f"元数据 length={length}，实际找到 {actual} 帧；"
                    f"文件内 episode_index={sorted(counts)}。"
                    "请检查是否混用了不同数据集的元数据、parquet 或标注文件；"
                    "脚本不会将缺失数据填成 0 或静默跳过。"
                )
    return expected_frames


def export_dataset(
    source: Path,
    output: Path,
    annotation: Path,
    repo_id: str,
    kept: list[int],
    failed: list[int],
    missing: list[int],
) -> tuple[int, int]:
    """用本地 LeRobot 官方 split 工具维护 parquet、视频引用及统计信息。"""
    if not kept:
        raise ValueError("没有成功 episode，不能导出空数据集")
    validate_paths(source, output)
    # 数据必须已完整保存在本地，不因文件缺失而自动联网下载。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=source)
    print("检查成功 episode 的元数据与 parquet 引用……", flush=True)
    expected_frames = validate_source_episodes(dataset, kept)
    print(f"检查通过：{len(kept)} 个 episode，{expected_frames} 帧。", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 临时目录与目标目录在同一文件系统；失败时清理，不留下看似完整的目标数据集。
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        exported = split_dataset(dataset, splits={"success": kept}, output_dir=temporary)["success"]
        if exported.num_episodes != len(kept) or exported.num_frames != expected_frames:
            raise ValueError("导出后的 episode / 帧数与预期不一致，取消保存")
        staging = Path(temporary) / "success"
        write_provenance(staging, source, annotation, kept, failed, missing)
        # 导出可能耗时较长，再次检查，避免覆盖期间新建的目标目录。
        validate_paths(source, output)
        staging.rename(output)
    return len(kept), expected_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", required=True, type=Path, help="源数据集目录，包含 meta/info.json")
    parser.add_argument("--output-root", required=True, type=Path, help="新数据集目录，必须尚不存在")
    parser.add_argument("--results-json", type=Path, help="标注文件；默认 <source-root>/episode_result.json")
    parser.add_argument("--repo-id", default="local/dataset", help="源数据集逻辑名称；输出名称追加 _success，不上传")
    parser.add_argument("--dry-run", action="store_true", help="仅检查标注和打印筛选结果，不复制数据")
    parser.add_argument("--debug", action="store_true", help="打印 Python / LeRobot 路径及完整错误堆栈")
    args = parser.parse_args(argv)
    try:
        source = args.source_root.expanduser().resolve()
        output = args.output_root.expanduser().resolve()
        annotation = (args.results_json or source / "episode_result.json").expanduser().resolve()
        validate_paths(source, output)
        info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
        if info.get("codebase_version") != "v3.0":
            raise ValueError("本脚本使用本地 LeRobot v3 工具，仅支持 codebase_version=v3.0 的数据集")
        kept, failed, missing = select_episodes(read_results(annotation), info["total_episodes"])
        print(f"源数据集：{source}")
        print(f"标注文件：{annotation}")
        print(f"总 episode：{info['total_episodes']}；成功：{len(kept)}；失败：{len(failed)}；未标注：{len(missing)}")
        print(f"保留原编号：{kept}")
        print(f"排除失败编号：{failed}")
        if missing:
            print(f"未标注编号（不导出）：{missing}")
        print(f"输出目录：{output}", flush=True)
        if not kept:
            raise ValueError("没有成功 episode，不能导出空数据集")
        if args.dry_run:
            print("预览完成，未写入任何数据。")
            return 0
        if args.debug:
            import lerobot.datasets.dataset_tools as dataset_tools
            import pandas

            print(f"Python：{sys.executable}；pandas：{pandas.__version__}", flush=True)
            print(f"LeRobot dataset_tools：{dataset_tools.__file__}", flush=True)
        episodes, frames = export_dataset(source, output, annotation, args.repo_id, kept, failed, missing)
        print(f"导出完成：{episodes} 个 episode，{frames} 帧；新 episode 编号为 0..{episodes - 1}。")
        print("原数据集保持不变；原编号映射见输出目录 success_filter_manifest.json。")
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        else:
            print("需要完整调用堆栈时，请在原命令后加 --debug。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
