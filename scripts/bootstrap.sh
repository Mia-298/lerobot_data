#!/usr/bin/env bash

# 任意一步失败就停止，避免继续执行后面的安装。
set -euo pipefail

# 根据脚本所在位置找到仓库根目录，不写死 /home/mia。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="lerobot"
LEROBOT_DIR="$ROOT_DIR/projects/lerobot"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu128}"

# pip 默认走国内镜像；调用脚本前设置 PIP_INDEX_URL 可临时覆盖。
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_DISABLE_PIP_VERSION_CHECK="1"

echo "项目根目录：$ROOT_DIR"
echo "pip 软件源：$PIP_INDEX_URL"
echo "PyTorch 软件源：$PYTORCH_INDEX_URL"

# ---------- 1. 检查基础工具 ----------

for tool in conda git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "缺少命令：$tool，请先安装并确保终端能使用它。"
        exit 1
    fi
done

# ---------- 2. 检查恢复所需文件 ----------

for file in environment.yml lerobot-revision.txt; do
    if [[ ! -f "$ROOT_DIR/$file" ]]; then
        echo "缺少文件：$ROOT_DIR/$file"
        exit 1
    fi
done

REVISION="$(tr -d '[:space:]' < "$ROOT_DIR/lerobot-revision.txt")"

if [[ ! "$REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "lerobot-revision.txt 应包含一个完整的 Git 提交 SHA。"
    exit 1
fi

# ---------- 3. 恢复 Conda 环境 ----------

CONDA_BASE="$(conda info --base)"

ENV_EXISTS="$(
    conda env list --json |
        "$CONDA_BASE/bin/python" -c '
import json
import sys
from pathlib import Path

envs = json.load(sys.stdin)["envs"]
print("yes" if any(Path(p).name == "lerobot" for p in envs) else "no")
'
)"

if [[ "$ENV_EXISTS" == "yes" ]]; then
    echo "已存在 $ENV_NAME 环境，保留现有环境，不重新创建。"
else
    echo "正在根据 environment.yml 创建 $ENV_NAME 环境……"
    conda env create \
        --name "$ENV_NAME" \
        --file "$ROOT_DIR/environment.yml"
fi

# 后续命令统一在指定 Conda 环境中运行。
run_in_env() {
    conda run --no-capture-output --name "$ENV_NAME" "$@"
}

run_in_env python --version

# ---------- 4. 安装与当前显卡驱动兼容的 PyTorch ----------

run_in_env python -m pip install \
    --upgrade \
    --extra-index-url "$PYTORCH_INDEX_URL" \
    "torch==2.11.0+cu128" \
    "torchvision==0.26.0+cu128"

# ---------- 5. 恢复指定版本的 LeRobot ----------

if [[ ! -e "$LEROBOT_DIR" ]]; then
    mkdir -p "$LEROBOT_DIR"
    git -C "$LEROBOT_DIR" init
    git -C "$LEROBOT_DIR" remote add origin \
        https://github.com/huggingface/lerobot.git
    git -C "$LEROBOT_DIR" fetch --depth 1 --no-tags origin "$REVISION"
    git -C "$LEROBOT_DIR" checkout --detach FETCH_HEAD
else
    # 避免把普通目录或外层仓库误认为 LeRobot 仓库。
    if [[ ! -e "$LEROBOT_DIR/.git" ]]; then
        echo "$LEROBOT_DIR 已存在，但不是独立 Git 仓库。"
        exit 1
    fi

    CURRENT_REVISION="$(git -C "$LEROBOT_DIR" rev-parse HEAD)"

    if [[ "$CURRENT_REVISION" != "$REVISION" ]]; then
        echo "现有 LeRobot 版本与记录不一致，请手动检查："
        echo "当前：$CURRENT_REVISION"
        echo "需要：$REVISION"
        exit 1
    fi

    if [[ -n "$(git -C "$LEROBOT_DIR" status --porcelain)" ]]; then
        echo "LeRobot 中存在本地修改，脚本停止，不覆盖你的代码。"
        exit 1
    fi

    echo "LeRobot 已存在，版本正确且工作区干净。"
fi

# ---------- 6. 安装本地项目 ----------

for project in \
    lerobot \
    mia_kinematics \
    lerobot_robot_myarm \
    lerobot_robot_aubo
do
    PROJECT_DIR="$ROOT_DIR/projects/$project"

    if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
        echo "找不到项目配置：$PROJECT_DIR/pyproject.toml"
        exit 1
    fi

    echo "正在安装本地项目：$project"

    run_in_env python -m pip install \
        --no-deps \
        --no-build-isolation \
        -e "$PROJECT_DIR"
done

# ---------- 7. 检查依赖 ----------

run_in_env python -m pip check

echo "代码和环境恢复步骤完成。"
echo "注意：模型权重尚未下载，也没有连接或控制机器人。"
