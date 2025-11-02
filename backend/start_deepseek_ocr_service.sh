#!/bin/bash
# DeepSeek-OCR 服务启动脚本

set -e

# 配置
ENV_NAME="deepseek-ocr"
SERVICE_PORT=5001
MODEL_PATH="/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "DeepSeek-OCR 服务启动"
echo "========================================"

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ 未找到 conda${NC}"
    exit 1
fi

# 检查环境是否存在
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${RED}✗ 环境 ${ENV_NAME} 不存在${NC}"
    echo "请先运行安装脚本:"
    echo "  bash setup_deepseek_ocr_env.sh"
    exit 1
fi

# 检查模型是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo -e "${RED}✗ 模型文件不存在: ${MODEL_PATH}${NC}"
    echo ""
    echo "请先下载模型:"
    echo "  方法1: 从 Hugging Face 下载"
    echo "    git lfs install"
    echo "    git clone https://huggingface.co/deepseek-ai/deepseek-ocr ${MODEL_PATH}"
    echo ""
    echo "  方法2: 使用 huggingface-cli"
    echo "    pip install huggingface-hub"
    echo "    huggingface-cli download deepseek-ai/deepseek-ocr --local-dir ${MODEL_PATH}"
    exit 1
fi

# 检查端口是否被占用
if lsof -Pi :${SERVICE_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ 端口 ${SERVICE_PORT} 已被占用${NC}"
    echo "尝试停止旧进程..."
    
    PID=$(lsof -Pi :${SERVICE_PORT} -sTCP:LISTEN -t)
    kill -9 ${PID} 2>/dev/null || true
    sleep 2
    
    if lsof -Pi :${SERVICE_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}✗ 无法停止旧进程，请手动处理${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 旧进程已停止${NC}"
fi

# 激活环境
echo ""
echo "激活 conda 环境: ${ENV_NAME}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 检查 Python 和依赖
echo ""
echo "环境信息:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="${SCRIPT_DIR}/deepseek_ocr_service.py"

if [ ! -f "${SERVICE_FILE}" ]; then
    echo -e "${RED}✗ 服务文件不存在: ${SERVICE_FILE}${NC}"
    exit 1
fi

# 启动服务
echo ""
echo "========================================"
echo "启动服务..."
echo "========================================"
echo -e "${GREEN}监听地址: http://0.0.0.0:${SERVICE_PORT}${NC}"
echo -e "${GREEN}本地访问: http://localhost:${SERVICE_PORT}/api/health${NC}"
echo ""
echo "按 Ctrl+C 停止服务"
echo "========================================"
echo ""

cd "${SCRIPT_DIR}"
python deepseek_ocr_service.py

