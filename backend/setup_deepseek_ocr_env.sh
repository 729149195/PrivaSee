#!/bin/bash
# DeepSeek-OCR 环境安装脚本
# 创建 conda 环境并安装所有依赖

set -e

echo "========================================"
echo "DeepSeek-OCR 环境安装"
echo "========================================"

# 配置
ENV_NAME="deepseek-ocr"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ 未找到 conda，请先安装 Anaconda 或 Miniconda${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 找到 conda${NC}"

# 检查环境是否存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}环境 ${ENV_NAME} 已存在${NC}"
    read -p "是否删除并重新创建？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "取消安装"
        exit 0
    fi
fi

# 创建 conda 环境
echo ""
echo "========================================"
echo "步骤 1/5: 创建 Conda 环境"
echo "========================================"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
echo -e "${GREEN}✓ 环境创建完成${NC}"

# 激活环境
echo ""
echo "========================================"
echo "步骤 2/5: 激活环境"
echo "========================================"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
echo -e "${GREEN}✓ 环境已激活: $(which python)${NC}"

# 安装 PyTorch (CUDA 版本)
echo ""
echo "========================================"
echo "步骤 3/5: 安装 PyTorch"
echo "========================================"
echo "CUDA 版本: ${CUDA_VERSION}"

# 检查是否有 NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ 检测到 NVIDIA GPU${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    # 安装 CUDA 版本的 PyTorch
    pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}⚠ 未检测到 NVIDIA GPU，将安装 CPU 版本${NC}"
    pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✓ PyTorch 安装完成${NC}"

# 安装其他依赖
echo ""
echo "========================================"
echo "步骤 4/5: 安装其他依赖"
echo "========================================"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/deepseek_ocr_requirements.txt"

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo -e "${RED}✗ 未找到 requirements 文件: ${REQUIREMENTS_FILE}${NC}"
    exit 1
fi

pip install -r "${REQUIREMENTS_FILE}"
echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 安装系统依赖（PDF 支持）
echo ""
echo "========================================"
echo "步骤 5/5: 安装系统依赖（PDF 支持）"
echo "========================================"

if command -v apt-get &> /dev/null; then
    echo "检测到 apt-get，尝试安装 poppler-utils..."
    sudo apt-get update && sudo apt-get install -y poppler-utils
    echo -e "${GREEN}✓ poppler-utils 安装完成${NC}"
elif command -v brew &> /dev/null; then
    echo "检测到 Homebrew，尝试安装 poppler..."
    brew install poppler
    echo -e "${GREEN}✓ poppler 安装完成${NC}"
else
    echo -e "${YELLOW}⚠ 未找到包管理器，请手动安装 poppler-utils${NC}"
    echo "  Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "  macOS: brew install poppler"
fi

# Flash Attention (可选)
echo ""
echo "========================================"
echo "可选: Flash Attention 2"
echo "========================================"
echo "Flash Attention 2 可以加速推理，但需要编译"
echo "如果安装失败，模型会自动降级到标准 attention"
echo ""
read -p "是否安装 Flash Attention 2？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装 Flash Attention 2..."
    pip install flash-attn==2.7.3 --no-build-isolation || {
        echo -e "${YELLOW}⚠ Flash Attention 2 安装失败（这是正常的）${NC}"
        echo "模型将使用标准 attention"
    }
fi

# 安装完成
echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo -e "${GREEN}✓ 环境名称: ${ENV_NAME}${NC}"
echo -e "${GREEN}✓ Python 版本: $(python --version)${NC}"
echo -e "${GREEN}✓ PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')${NC}"
echo ""
echo "激活环境命令:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "启动服务命令:"
echo "  cd ${SCRIPT_DIR}"
echo "  bash start_deepseek_ocr_service.sh"
echo ""
echo "========================================"

