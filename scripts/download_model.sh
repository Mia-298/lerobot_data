
#!/usr/bin/env bash
set -euo pipefail

# AutoDL SSH 别名
REMOTE="CQ681"

# 云端发布目录
REMOTE_DIR="/root/autodl-tmp/vla_work/outputs/publish"

# 本地项目路径
PROJECT_DIR="/home/mia/data/smolvla_work"

# 本地下载目录
LOCAL_DIR="$PROJECT_DIR/outputs/publish"

echo "======================================"
echo "    AutoDL Publish Synchronization"
echo "======================================"

echo "Checking SSH connection..."

if ! ssh -o ConnectTimeout=10 "$REMOTE" "true"; then
    echo "ERROR: SSH connection failed."
    exit 1
fi

echo "Checking remote publish directory..."

if ! ssh "$REMOTE" "test -d '$REMOTE_DIR'"; then
    echo "ERROR: Remote publish directory does not exist."
    exit 1
fi

mkdir -p "$LOCAL_DIR"

echo "Downloading all published files..."

rsync -avh \
    --partial \
    --info=progress2 \
    "$REMOTE:$REMOTE_DIR/" \
    "$LOCAL_DIR/"

echo "======================================"
echo "    Synchronization completed"
echo "======================================"
echo "Local directory: $LOCAL_DIR"
