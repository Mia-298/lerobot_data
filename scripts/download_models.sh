#!/usr/bin/env bash
set -euo pipefail

# 根据脚本所在位置找到项目根目录，不写死 /home/mia。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="lerobot"
MODEL_REPO="lerobot/smolvla_base"
MODEL_DIR="$ROOT_DIR/models/smolvla_base"
REVISION_FILE="$ROOT_DIR/smolvla-revision.txt"
EXPECTED_MODEL_SHA256="7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"

# 国内网络默认使用已验证可访问的镜像；这些变量均可在调用脚本前覆盖。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"

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
echo "下载端点：$HF_ENDPOINT"
echo "并发任务：$HF_MAX_WORKERS"

conda run --no-capture-output --name "$ENV_NAME" \
    hf download "$MODEL_REPO" \
    --revision "$REVISION" \
    --local-dir "$MODEL_DIR" \
    --max-workers "$HF_MAX_WORKERS"

for required_file in \
    config.json \
    model.safetensors \
    policy_preprocessor.json \
    policy_postprocessor.json
do
    if [[ ! -s "$MODEL_DIR/$required_file" ]]; then
        echo "错误：下载结果缺少文件或文件为空：$MODEL_DIR/$required_file"
        exit 1
    fi
done

ACTUAL_MODEL_SHA256="$(sha256sum "$MODEL_DIR/model.safetensors" | cut -d ' ' -f 1)"
if [[ "$ACTUAL_MODEL_SHA256" != "$EXPECTED_MODEL_SHA256" ]]; then
    echo "错误：model.safetensors 的 SHA256 校验失败。"
    echo "预期：$EXPECTED_MODEL_SHA256"
    echo "实际：$ACTUAL_MODEL_SHA256"
    exit 1
fi

echo "模型下载并校验完成。"
echo "model.safetensors SHA256：$ACTUAL_MODEL_SHA256"
