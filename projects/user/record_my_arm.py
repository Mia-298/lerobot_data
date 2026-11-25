import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

_IS_WSL = "microsoft" in os.uname().release.lower()
if _IS_WSL:
    # WSLg 的 OpenGL 后端缺少 Rerun 需要的 R32Float 渲染能力；Vulkan 可用。
    os.environ.setdefault("WGPU_BACKEND", "vulkan")

from lerobot.configs import parser
from lerobot.scripts import lerobot_record as _lerobot_record
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.configs.dataset import DatasetRecordConfig
# 导入这个包时，也会执行其中的配置类注册
from lerobot_robot_myarm import MyArmConfig, make_myarm_processor

_RERUN_PID_FILE = Path.home() / ".cache" / "lerobot" / "rerun_web_viewer.pid"
_RERUN_PREVIEW_FPS = 5.0
_RERUN_PREVIEW_MAX_WIDTH = 320
_RERUN_PREVIEW_MAX_HEIGHT = 240


def _port_is_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rerun_web_viewer_url(web_port: int, grpc_port: int) -> str:
    query = urlencode(
        {
            "url": f"rerun+http://127.0.0.1:{grpc_port}/proxy",
            "hide_welcome_screen": "true",
            "renderer": "webgpu",
        }
    )
    return f"http://127.0.0.1:{web_port}/?{query}"


def _start_rerun_web_viewer_on_wsl(cfg: RecordConfig) -> subprocess.Popen | None:
    if not (
        cfg.display_data
        and cfg.display_mode == "rerun"
        and _IS_WSL
        and cfg.display_ip is None
        and cfg.display_port is None
    ):
        return None

    _stop_previous_rerun_web_viewer()

    # 每次录制使用独立服务，避免浏览器选中上一次录制的 recording 或时间点。
    grpc_port = _find_free_port()
    web_port = _find_free_port()
    rerun_executable = Path(sys.executable).with_name("rerun")
    if not rerun_executable.is_file():
        raise RuntimeError(f"当前 Python 环境中找不到 Rerun：{rerun_executable}")
    process = subprocess.Popen(
        [
            str(rerun_executable),
            "--serve-web",
            "--hide-welcome-screen",
            "--port",
            str(grpc_port),
            "--web-viewer-port",
            str(web_port),
        ],
        start_new_session=True,
    )
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(f"Rerun Web Viewer 提前退出，退出码：{process.returncode}")
        if _port_is_open(grpc_port) and _port_is_open(web_port):
            break
        time.sleep(0.1)
    else:
        _stop_rerun_web_viewer(process)
        raise RuntimeError("Rerun Web Viewer 启动超时")

    _RERUN_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RERUN_PID_FILE.write_text(f"{process.pid}\n", encoding="ascii")

    viewer_url = _rerun_web_viewer_url(web_port, grpc_port)
    webbrowser.open(viewer_url)

    cfg.display_ip = "127.0.0.1"
    cfg.display_port = grpc_port
    print(f"WSL 下使用 Rerun Web Viewer：{viewer_url}")
    return process


def _stop_previous_rerun_web_viewer() -> None:
    try:
        pid = int(_RERUN_PID_FILE.read_text(encoding="ascii").strip())
    except (FileNotFoundError, ValueError):
        _RERUN_PID_FILE.unlink(missing_ok=True)
        return

    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        _RERUN_PID_FILE.unlink(missing_ok=True)
        return

    if "rerun" not in cmdline or "--serve-web" not in cmdline:
        _RERUN_PID_FILE.unlink(missing_ok=True)
        return

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    else:
        for _ in range(50):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.1)
        else:
            os.killpg(pid, signal.SIGKILL)
    _RERUN_PID_FILE.unlink(missing_ok=True)


def _stop_rerun_web_viewer(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def _resize_preview_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        return image

    channels_first = image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4)
    height, width = image.shape[1:3] if channels_first else image.shape[:2]
    row_step = max(1, (height + _RERUN_PREVIEW_MAX_HEIGHT - 1) // _RERUN_PREVIEW_MAX_HEIGHT)
    col_step = max(1, (width + _RERUN_PREVIEW_MAX_WIDTH - 1) // _RERUN_PREVIEW_MAX_WIDTH)
    if row_step == 1 and col_step == 1:
        return image
    if channels_first:
        return image[:, ::row_step, ::col_step]
    return image[::row_step, ::col_step]


def _limit_rerun_preview_load_on_wsl(cfg: RecordConfig) -> None:
    if not (cfg.display_data and cfg.display_mode == "rerun" and _IS_WSL):
        return

    from lerobot.utils import rerun_visualization

    # Rerun's automatic layout is more reliable in the Web Viewer than the custom grid blueprint.
    rerun_visualization._ensure_blueprint = lambda *_args, **_kwargs: None

    original_log = _lerobot_record.log_visualization_data
    min_interval_s = 1.0 / _RERUN_PREVIEW_FPS
    last_log_time = float("-inf")

    def log_preview(
        display_mode: str,
        observation=None,
        action=None,
        compress_images: bool = False,
    ) -> None:
        nonlocal last_log_time
        now = time.monotonic()
        if display_mode == "rerun" and now - last_log_time < min_interval_s:
            return
        last_log_time = now

        preview_observation = observation
        if observation:
            preview_observation = {
                key: _resize_preview_image(value) if isinstance(value, np.ndarray) else value
                for key, value in observation.items()
            }
        original_log(
            display_mode,
            observation=preview_observation,
            action=action,
            compress_images=True if display_mode == "rerun" else compress_images,
        )

    _lerobot_record.log_visualization_data = log_preview
    print(
        "WSL 下 Rerun 网页预览限制为 "
        f"{_RERUN_PREVIEW_FPS:g} FPS、最大 {_RERUN_PREVIEW_MAX_WIDTH}x{_RERUN_PREVIEW_MAX_HEIGHT}；"
        "数据集仍保存原始分辨率和帧率。"
    )
def select_task(
    tasks: list[str],
    previous_id: int | None = None,
) -> int | None:
    if not tasks:
        raise ValueError("任务列表不能为空")

    for number, task in enumerate(tasks, start=1):
        print(f"{number:2d}. {task}")

    if previous_id is not None:
        print(f"当前任务：{previous_id}. {tasks[previous_id - 1]}")

    while True:
        value = input("输入编号，回车沿用上次任务，q 退出：").strip()

        if value.lower() == "q":
            return None

        if not value and previous_id is not None:
            return previous_id

        try:
            number = int(value)
        except ValueError:
            print("请输入整数编号")
            continue

        if 1 <= number <= len(tasks):
            return number

        print(f"编号必须在 1～{len(tasks)} 之间")

from lerobot_robot_myarm import MyArmConfig, MyArmKeyboardConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot_robot_myarm.task_tags import resolve_task_tags


@dataclass
class MyDatasetRecordConfig(DatasetRecordConfig):
    # Draccus 在传入任意 --dataset.* 参数时会重新构造嵌套配置；
    # 默认值必须声明在字段上，不能只写在外层 default_factory 的 lambda 里。
    repo_id: str = "mia/pick_place_16"
    root: str | Path | None = "/home/mia/data/smolvla_work/datasets/pick_place_16"
    no_stamp: bool = True
    fps: int = 30
    episode_time_s: int | float = 90
    num_episodes: int = 1
    push_to_hub: bool = False


@dataclass
class MyRecordConfig(RecordConfig):
    robot: MyArmConfig = field(
        default_factory=lambda: MyArmConfig(
            xml_path="/home/mia/data/smolvla_work/projects/models/world.xml",
            randomize_objects=True,
            randomize_lighting=True,
        )
    )

    teleop: TeleoperatorConfig = field(
        default_factory=MyArmKeyboardConfig
    )

    dataset: MyDatasetRecordConfig = field(default_factory=MyDatasetRecordConfig)

    display_data: bool = True
    play_sounds: bool = False


def _print_recording_summary(dataset) -> None:
    # 官方 record() 返回前已 finalize；只读本次 writer 保存的最新 episode，
    # 不用累计 num_frames，避免 resume 时把历史帧数误算成本段时长。
    latest_episode = dataset.meta.latest_episode
    if latest_episode is None:
        print("本次未保存新的 episode。", flush=True)
        return
    episode_index = int(latest_episode["episode_index"][0])
    frames = int(latest_episode["length"][0])
    fps = dataset.fps
    print(
        f"已保存 episode {episode_index}：{frames} 帧，数据集 FPS={fps}，"
        f"按帧数计算的视频时长约 {frames / fps:.2f}s。"
        "episode_time_s 限制的是现实录制时间，不保证采满目标帧数。",
        flush=True,
    )


@parser.wrap()
def main(cfg: MyRecordConfig):
    # record() 的 CLI 包装器按精确类型判断；转换后直接使用已解析的配置。
    cfg = RecordConfig(**{item.name: getattr(cfg, item.name) for item in fields(RecordConfig) if item.init})
    if not isinstance(cfg.robot, MyArmConfig):
        raise ValueError("这个启动脚本需要使用 --robot.type=my_arm")
    _start_rerun_web_viewer_on_wsl(cfg)
    _limit_rerun_preview_load_on_wsl(cfg)
    num_episodes = 16
 
    previous_id = None
    for _ in range(num_episodes*20):
        # task_id = select_task(cfg.robot.tasklist, previous_id)
        task_id = (_+1)%num_episodes
        if task_id is None:
            break
        cfg.robot.task_id = task_id
        cfg.dataset.single_task = cfg.robot.tasklist[task_id - 1]
        cfg.dataset.num_episodes = 1
        cfg.dataset.no_stamp = True
        cfg.dataset.push_to_hub = False
        if cfg.teleop.type == "my_arm_auto":
            try:
                targets = resolve_task_tags(task_id, cfg.robot.tasklist)
            except ValueError as exc:
                print(exc)
                continue
            print(
                f"任务 {task_id}：抓取 Tag {targets.object_tag_id}，"
                f"放到盘子 Tag {targets.plate_tag_id}"
            )

        processor = make_myarm_processor(
            cfg.robot,
            auto_config=cfg.teleop if cfg.teleop.type == "my_arm_auto" else None,
        )

        print(
            f"本次录制有效配置：episode_time_s={cfg.dataset.episode_time_s}s（现实时间上限），"
            f"fps={cfg.dataset.fps}，repo_id={cfg.dataset.repo_id}，"
            f"root={cfg.dataset.root}，resume={cfg.resume}",
            flush=True,
        )
        dataset = _lerobot_record.record(
            cfg,
            teleop_action_processor=processor,
        )
        if dataset is not None:
            _print_recording_summary(dataset)
        # 官方 record 每次新建机器人；固定 seed 时递增索引，避免重复同一次采样。
        cfg.robot.scene_randomization_index += 1
        previous_id = task_id
        # 后续调用追加到同一个数据集
        cfg.resume = True

    


if __name__ == "__main__":
    main()
