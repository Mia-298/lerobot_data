#!/usr/bin/env bash
set -euo pipefail

# 根据脚本所在位置找到项目根目录，不写死 /home/mia。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="lerobot"
MODEL_REPO="lerobot/smolvla_base"
MODEL_DIR="$ROOT_DIR/models/smolvla_base"
REVISION_FILE="$ROOT_DIR/smolvla-revision.txt"

if ! command -v conda >/dev/null 2>&1; then
    echo "错误：找不到 conda，请先安装并初始化 Conda。"
    exit 1
fi

if [[ ! -f "$REVISION_FILE" ]]; then
    echo "错误：找不到模型版本文件：$REVISION_FILE"
    exit 1
fi

REVISION="$(tr -d '[:space:]' < "$REVISION_FILE")"

if [[ ! "$REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "错误：模型版本应当是 40 位 Git commit SHA。"
    exit 1
fi

mkdir -p "$MODEL_DIR"

echo "模型仓库：$MODEL_REPO"
echo "模型版本：$REVISION"
echo "保存目录：$MODEL_DIR"

# 默认禁用 Xet；如果你事先设置了这个变量，则保留你的设置。
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

conda run --no-capture-output --name "$ENV_NAME" \
    hf download "$MODEL_REPO" \
    --revision "$REVISION" \
    --local-dir "$MODEL_DIR"

echo "模型下载完成。"