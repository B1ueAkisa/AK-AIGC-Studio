#!/bin/bash
# ==============================================================================
# AK-AIGC-Studio: Colab 一键环境配置与 Drive 挂载映射脚本
# ==============================================================================
set -e

echo "🚀 [AK-AIGC-Studio] 开始初始化运行环境..."

# 1. 挂载 Google Drive
DRIVE_MOUNT_POINT="/content/drive"
WORKSPACE_DIR="$DRIVE_MOUNT_POINT/MyDrive/AIGC_Workspace"

if [ ! -d "$DRIVE_MOUNT_POINT/MyDrive" ]; then
    echo "📦 正在挂载 Google Drive..."
    python3 -c "from google.colab import drive; drive.mount('$DRIVE_MOUNT_POINT')"
else
    echo "✅ Google Drive 已挂载。"
fi

# 2. 验证并链接 Workspace 结构
echo "📁 检查云端工作区: $WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR/models/audio"
mkdir -p "$WORKSPACE_DIR/models/image"
mkdir -p "$WORKSPACE_DIR/models/video"
mkdir -p "$WORKSPACE_DIR/datasets"
mkdir -p "$WORKSPACE_DIR/outputs"

# 创建软链接到本地项目根目录方便直接读写
PROJECT_ROOT=$(pwd)
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/outputs"

ln -sfn "$WORKSPACE_DIR/models/audio" "$PROJECT_ROOT/models/audio"
ln -sfn "$WORKSPACE_DIR/models/image" "$PROJECT_ROOT/models/image"
ln -sfn "$WORKSPACE_DIR/models/video" "$PROJECT_ROOT/models/video"
ln -sfn "$WORKSPACE_DIR/outputs" "$PROJECT_ROOT/outputs/drive_outputs"

echo "✅ Google Drive 软链接映射就绪！"
echo "  - 模型存储: $PROJECT_ROOT/models/ -> $WORKSPACE_DIR/models/"
echo "  - 成果产出: $PROJECT_ROOT/outputs/drive_outputs -> $WORKSPACE_DIR/outputs/"

# 3. 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 显卡状态:"
    nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader
fi

echo "🎉 [AK-AIGC-Studio] 环境就绪，随时可以开始生成！"
