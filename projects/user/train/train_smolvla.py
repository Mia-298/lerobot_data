#!/usr/bin/env python3
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import myarm_eval_config 

@dataclass
class TrainingSettings:
    # ========== 1. 最常修改：模式和自己的数据集 ==========
    mode: str = "train"  # smoke：20 步试跑；train：正式微调；resume：恢复中断训练
    check_only: bool = False  # True：只检查路径和配置，不加载权重、不训练、不保存模型
    dataset_repo_id: str = "mia/pick_place_auto_done"  # 本地数据集对应的标识，不会因此上传
    # mia配置
    # dataset_root: Path = Path("/home/mia/data/smolvla_work/datasets/pick_place_auto_done")
    # model_path: Path = Path("/home/mia/data/smolvla_work/models/smolvla_base")
    # output_parent: Path = Path("/home/mia/data/smolvla_work/outputs/train")

    # autodl配置
    dataset_root: Path = Path("/root/autodl-tmp/vla_work/datasets/publish/pick_place_auto_done")
    model_path: Path = Path("/root/autodl-tmp/vla_work/models/smolvla_base")
    output_parent: Path = Path("/root/autodl-tmp/vla_work/outputs/train")
    run_name: str | None = None  # None 自动生成时间戳；自定义名称不能和已有目录重名
    # 验证集分割
    dataset_eval_split: float = 0.2
    # 验证集step
    eval_steps: int = 1000
    # ========== 网络与底层 VLM 依赖（本地 SmolVLA 权重不等于依赖已齐全） ==========
    prepare_assets_only: bool = False  # True：只下载/检查底层 VLM 和 tokenizer，绝不训练
    offline_mode: bool = True  # True：训练时禁止 Hub 联网；必须事先准备完整缓存
    # None：自动使用 model_path 上两级目录下 cache/huggingface/hub。
    # 例如 AutoDL model_path=/root/autodl-tmp/vla_work/models/smolvla_base 时，
    # 缓存为 /root/autodl-tmp/vla_work/cache/huggingface/hub；也可以直接填 Path(...)
    hf_cache_dir: Path | None = None
    hf_endpoint: str | None = None  # None 保留默认/外部设置；可填你已确认能访问且可信的 Hub 兼容地址

    # ========== 2. 正式训练超参数（smoke 模式使用下面独立的试跑参数） ==========
    steps: int = 20_000  # 训练循环迭代次数，不是遍历数据集的 epoch 数
    learning_rate: float = 1e-4  # 参数更新幅度；过大易不稳定，过小学习慢
    warmup_steps: int = 500  # 开始阶段逐渐增大学习率，减少训练初期震荡
    decay_steps: int = 10_000  # 学习率衰减到最小值所用的调度总步数（含 warmup）
    min_learning_rate: float = 2.5e-6  # 衰减末端学习率
    log_freq: int = 500  # 每多少步打印一次 loss 等日志
    save_freq: int = 1_000  # 每多少步保存；最后一步也会保存

    # ========== 3. 显存和速度（试跑和正式训练共用） ==========
    batch_size: int = 4  # 每次读入的样本数；增大通常会增加显存占用
    num_workers: int = 4  # 数据加载子进程数；先用 0 排错，再按 CPU/内存情况增大
    mixed_precision: str = "bf16"  # 1660 SUPER 先用 fp16；不支持/不稳定时可试 no
    gradient_accumulation_steps: int = 1  # 累积若干微批次再更新权重；先保持 1
    image_size: int = 512  # 保持基础模型设置；显存不足可尝试 256，需重新验证效果
    seed: int = 1000  # 随机种子，便于对比实验；不保证所有硬件上完全确定性

    # ========== 4. 微调范围：小显存先不要解冻整个视觉语言模型 ==========
    freeze_vision_encoder: bool = True  # 冻结视觉编码器，减少训练开销
    train_expert_only: bool = True  # 主要训练动作专家，不全量微调语言骨干
    train_state_proj: bool = True  # 同时训练机器人状态输入投影层
    weight_decay: float = 1e-10  # 权重衰减，正则化强度；先保留基础配置
    grad_clip_norm: float = 10.0  # 梯度裁剪上限，限制异常大的梯度

    # ========== 5. 试跑参数：仅验证流程/显存，不代表模型已经学会任务 ==========
    smoke_steps: int = 20
    smoke_warmup_steps: int = 5

    # ========== 6. 恢复中断训练 ==========
    # 指向原训练目录下 checkpoints/last/pretrained_model/train_config.json。
    # resume 会恢复原配置、优化器、调度器和步数，上面新训练参数不生效。
    # 用于继续未完成的训练；不是给已完成实验自动追加 steps。
    resume_config: Path = Path(
    "/root/autodl-tmp/vla_work/outputs/train/"
    "smolvla_train_20261127_160958_293634/checkpoints/001000/"
    "pretrained_model/train_config.json"
)
    # ========== 7. 仿真评测配置 ==========
    env_type: str | None = "myarm_sim"
    env_task_id: int = 1
    # env_xml_path: Path = Path(
    #     "/home/mia/data/smolvla_work/projects/models/world.xml"
    # )
    env_xml_path = Path("/root/autodl-tmp/vla_work/projects/models/world.xml")

    # 0 表示关闭训练期间仿真评测
    env_eval_freq: int = 1000
    eval_batch_size: int = 1
    eval_n_episodes: int = 5
    eval_use_async_envs: bool = False
    save_checkpoint: bool  = True
    use_policy_training_preset: bool  = True
    wandb_enable: bool  = False


CONFIG = TrainingSettings()



# 仅在“下载依赖”机器上没有本地模型配置时使用；正常训练优先读取模型实际配置。
DEFAULT_VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def configure_huggingface(settings: TrainingSettings) -> Path:
    """必须在导入 huggingface_hub / transformers / LeRobot 前调用；修改后需重启脚本。"""
    cache = (Path(settings.hf_cache_dir) if settings.hf_cache_dir is not None else
             Path(settings.model_path).expanduser().resolve().parents[1] / "cache/huggingface/hub")
    cache = cache.expanduser().resolve()
    os.environ["HF_HUB_CACHE"] = str(cache)
    # 避免旧环境变量让 transformers 意外使用另一份缓存；不主动新增废弃变量。
    for key in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if key in os.environ:
            os.environ[key] = str(cache)
    offline = settings.offline_mode and not settings.prepare_assets_only
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    if settings.hf_endpoint is not None:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint.rstrip("/")
    return cache


def required_assets(settings: TrainingSettings) -> dict[str, bool]:
    """返回依赖来源和是否需要 VLM 权重；兼顾 checkpoint 原配置与保存的 tokenizer。"""
    if settings.mode == "resume" and settings.resume_config is not None:
        config_path = Path(settings.resume_config).expanduser()
        model_dir = config_path.parent
        policy = json.loads(config_path.read_text())["policy"]
    else:
        model_dir = Path(settings.model_path).expanduser()
        config_path = model_dir / "config.json"
        policy = json.loads(config_path.read_text()) if config_path.is_file() else {}
    vlm = policy.get("vlm_model_name", DEFAULT_VLM_REPO)
    # 下载完整底层权重，保持当前基础模型行为；不靠 load_vlm_weights=False 绕过联网。
    assets = {vlm: True}
    preprocessor = model_dir / "policy_preprocessor.json"
    if preprocessor.is_file():
        for step in json.loads(preprocessor.read_text()).get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                tokenizer = step.get("config", {}).get("tokenizer_name")
                if tokenizer:
                    assets.setdefault(tokenizer, False)
    return assets


def validate_asset_snapshot(snapshot: Path, require_weights: bool) -> None:
    """读取小配置/处理器，不加载模型权重；不能把半下载的 snapshot 当完整模型。"""
    if require_weights:
        if not (snapshot / "config.json").is_file():
            raise FileNotFoundError(f"缺少底层模型配置：{snapshot / 'config.json'}")
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards:
                raise ValueError(f"模型分片索引为空：{index}")
        else:
            shards = {"model.safetensors"}
        for shard in shards:
            file = snapshot / shard
            if not file.is_file() or file.stat().st_size == 0:
                raise FileNotFoundError(f"底层模型权重缺失或为空：{file}")
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    if require_weights:
        AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
        AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)


def prepare_or_check_assets(settings: TrainingSettings, cache: Path) -> None:
    """下载模式允许联网；离线预检只读缓存。两者都不创建训练输出或开始训练。"""
    from huggingface_hub import snapshot_download

    downloading = settings.prepare_assets_only
    for source, require_weights in required_assets(settings).items():
        try:
            if Path(source).expanduser().is_dir():
                snapshot = Path(source).expanduser().resolve()
            else:
                print(f"{'准备下载' if downloading else '离线检查'}依赖：{source}", flush=True)
                snapshot = Path(snapshot_download(
                    repo_id=source,
                    cache_dir=str(cache),
                    local_files_only=not downloading,
                    # 保留 transformers 所需配置、处理器、tokenizer、safetensors；不下载 ONNX 等副本。
                    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                    ignore_patterns=["onnx/*"],
                    max_workers=2,
                ))
            validate_asset_snapshot(snapshot, require_weights)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"{'准备' if downloading else '离线检查'}依赖失败：{source}\n"
                f"缓存目录：{cache}\n"
                "在能访问 Hub 的机器上设置 prepare_assets_only=True，运行本文件准备依赖；"
                "如果 AutoDL 无法联网，请复制完整缓存仓库（blobs、refs、snapshots），保留有效链接。"
                "然后设置 prepare_assets_only=False、offline_mode=True，重新运行。\n"
                f"原始原因：{exc}"
            ) from exc
        print(f"依赖检查通过：{snapshot}", flush=True)


def _flag(name, value):
    if isinstance(value, bool):
        value = str(value).lower()
    return f"--{name}={value}"


def build_arguments(settings: TrainingSettings) -> list[str]:
    """只构建配置，不导入训练依赖，不创建任何输出目录。"""
    if settings.mode not in {"smoke", "train", "resume"}:
        raise ValueError('mode 必须是 "smoke"、"train" 或 "resume"。')
    if settings.mode == "resume":
        if settings.resume_config is None:
            raise ValueError("resume 模式必须填写 resume_config。")
        return ["--resume=true", _flag("config_path", Path(settings.resume_config).expanduser().resolve())]

    steps = settings.smoke_steps if settings.mode == "smoke" else settings.steps
    warmup = settings.smoke_warmup_steps if settings.mode == "smoke" else settings.warmup_steps
    decay = steps if settings.mode == "smoke" else settings.decay_steps
    if min(steps, settings.batch_size, settings.gradient_accumulation_steps, settings.image_size) <= 0:
        raise ValueError("steps、batch_size、梯度累积步数和 image_size 必须大于 0。")
    if settings.num_workers < 0 or not 0 <= warmup < decay:
        raise ValueError("num_workers 必须 >= 0，且 0 <= warmup_steps < decay_steps。")
    if not 0 < settings.min_learning_rate <= settings.learning_rate:
        raise ValueError("学习率需要满足：0 < min_learning_rate <= learning_rate。")
    if settings.mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError('mixed_precision 必须为 "no"、"fp16" 或 "bf16"。')
    name = settings.run_name or f"smolvla_{settings.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if name in {".", ".."} or Path(name).name != name:
        raise ValueError("run_name 只能是单个目录名称，不能含路径。")
    output = Path(settings.output_parent).expanduser().resolve() / name
    values = {
        "dataset.repo_id": settings.dataset_repo_id,
        "dataset.root": Path(settings.dataset_root).expanduser().resolve(),
        "policy.path": Path(settings.model_path).expanduser().resolve(),
        # 关键：清除基础模型原相机/6维状态输入，由当前数据集推导 front/wrist 和7维状态。
        # 不能替换成 {}（解析器会合并字典）；动作输出由 LeRobot 工厂从数据集推导。
        # max_state_dim/max_action_dim 保持预训练值32，不改为7。
        "policy.input_features": "null",
        "policy.device": "cuda",
        "policy.push_to_hub": False,
        "policy.freeze_vision_encoder": settings.freeze_vision_encoder,
        "policy.train_expert_only": settings.train_expert_only,
        "policy.train_state_proj": settings.train_state_proj,
        "policy.resize_imgs_with_padding": json.dumps([settings.image_size, settings.image_size]),
        "policy.optimizer_lr": settings.learning_rate,
        "policy.optimizer_weight_decay": settings.weight_decay,
        "policy.optimizer_grad_clip_norm": settings.grad_clip_norm,
        "policy.scheduler_warmup_steps": warmup,
        "policy.scheduler_decay_steps": decay,
        "policy.scheduler_decay_lr": settings.min_learning_rate,
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "steps": steps,
        "seed": settings.seed,
        "accelerator.mixed_precision": settings.mixed_precision,
        "accelerator.gradient_accumulation.steps": settings.gradient_accumulation_steps,
        "output_dir": output,
        "job_name": name,
        "log_freq": 1 if settings.mode == "smoke" else settings.log_freq,
        "save_freq": steps if settings.mode == "smoke" else settings.save_freq,
        "save_checkpoint": settings.save_checkpoint,
        "use_policy_training_preset": settings.use_policy_training_preset,
        "env.type": settings.env_type,
        "env.task_id": settings.env_task_id,
        "env.xml_path": settings.env_xml_path,
        "env_eval_freq": settings.env_eval_freq,
        "eval.batch_size":settings.eval_batch_size,
        "eval.n_episodes": settings.eval_n_episodes,
        "eval.use_async_envs": settings.eval_use_async_envs,
        "eval_steps": settings.eval_steps,
        "dataset.eval_split": settings.dataset_eval_split,
        "wandb.enable": settings.wandb_enable,
    }
    return [_flag(key, value) for key, value in values.items()]


def validate_paths(settings: TrainingSettings, arguments: list[str]) -> None:
    """本地路径预检；不创建目录、不读取大权重。"""
    if settings.mode == "resume":
        config = Path(settings.resume_config).expanduser().resolve()
        if config.name != "train_config.json" or not config.is_file():
            raise FileNotFoundError(f"需要 checkpoint 中的 train_config.json：{config}")
        state = config.parent.parent / "training_state" / "training_step.json"
        if not state.is_file():
            raise FileNotFoundError(f"缺少恢复训练所需状态：{state}，仅模型权重不能恢复训练。")
        saved = json.loads(config.read_text())
        step = json.loads(state.read_text()).get("step")
        if step is not None and step >= saved["steps"]:
            raise ValueError("该 checkpoint 已达到原计划总步数；请新建微调实验，而不是恢复中断训练。")
        return
    for file in (
        Path(settings.dataset_root).expanduser() / "meta" / "info.json",
        Path(settings.model_path).expanduser() / "config.json",
        Path(settings.model_path).expanduser() / "model.safetensors",
    ):
        if not file.is_file():
            raise FileNotFoundError(f"缺少文件：{file}")
    output = Path(next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--output_dir=")))
    if output.exists():
        raise FileExistsError(f"新训练输出路径已存在，请修改 run_name：{output}")


@contextmanager
def lerobot_arguments(arguments):
    """LeRobot validate() 还会读取 sys.argv；训练期间保留完整列表。"""
    original = sys.argv
    sys.argv = [str(Path(__file__).resolve()), *arguments]
    try:
        yield
    finally:
        sys.argv = original


def check_configuration():
    # 仅解析配置，不导入训练入口，不创建数据加载器或模型。
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401

    @parser.wrap()
    def check(cfg: TrainPipelineConfig):
        cfg.validate()
        print(f"\n配置解析通过：steps={cfg.steps}, batch_size={cfg.batch_size}")
        print(f"学习率={cfg.optimizer.lr}, 精度={cfg.accelerator.mixed_precision}")
        print(f"输出目录：{cfg.output_dir}")
        print(f"input_features={cfg.policy.input_features}（新训练为 None，稍后按数据集推导）")
        print("未检查底层 VLM 缓存、未加载模型权重、未读取训练批次、未启动训练。实际兼容性仍需试跑验证。")

    check()


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--check", action="store_true", help="只检查配置，不训练")
    cli = options.parse_args()
    cache = configure_huggingface(CONFIG)
    print(f"Hugging Face 缓存：{cache}", flush=True)
    if CONFIG.prepare_assets_only:
        if cli.check or CONFIG.check_only:
            raise ValueError("只下载依赖和只检查配置不能同时启用；请关闭其中一个。")
        prepare_or_check_assets(CONFIG, cache)
        print("依赖已准备完成，没有训练。下一次设 prepare_assets_only=False、offline_mode=True。")
        return
    arguments = build_arguments(CONFIG)
    validate_paths(CONFIG, arguments)
    print(f"Python：{sys.executable}\n模式：{CONFIG.mode}", flush=True)
    print("本次交给 LeRobot 的配置：\n" + "\n".join(arguments), flush=True)
    if CONFIG.mode == "resume":
        print("恢复模式：读取 checkpoint 原配置，不应用顶部的新训练超参数。", flush=True)
    else:
        effective_batch = CONFIG.batch_size * CONFIG.gradient_accumulation_steps
        print(f"单卡有效 batch={effective_batch}；显存是否足够，以当前 GPU 的真实试跑为准。", flush=True)
        if CONFIG.gradient_accumulation_steps > 1:
            print("注意：当前训练循环 steps 按微批次计数，不等于优化器更新次数。", flush=True)
    try:
        with lerobot_arguments(arguments):
            if cli.check or CONFIG.check_only:
                check_configuration()
            else:
                if CONFIG.offline_mode:
                    prepare_or_check_assets(CONFIG, cache)
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA 不可用。请确认 IDE 使用 lerobot 环境、PyTorch CUDA 和驱动正常。")
                from lerobot.scripts.lerobot_train import main as train_main

                train_main()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"训练依赖未找到，当前解释器：{sys.executable}。请在 IDE 选择已安装 LeRobot 的环境。"
            f" 缺少模块：{exc.name}"
        ) from exc


if __name__ == "__main__":
    main()
