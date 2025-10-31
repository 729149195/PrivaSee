#!/bin/bash
# DeepSeek-OCR 服务启动脚本

echo "======================================"
echo "  DeepSeek-OCR 服务启动"
echo "======================================"
echo ""

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 初始化 conda
eval "$(conda shell.bash hook)"

# 激活环境
echo "激活 privasee 环境..."
conda activate privasee

if [ $? -ne 0 ]; then
    echo "错误: 无法激活 privasee 环境"
    echo "请先运行: bash setup_deepseek_ocr.sh"
    exit 1
fi

# 检查必要的包
echo "检查依赖包..."
python -c "import transformers; import torch; import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 部分依赖包未安装"
    echo "正在安装依赖..."
    pip install -r requirements_ocr.txt
fi

# 检查模型是否存在
MODEL_PATH="/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr"
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 未找到 DeepSeek-OCR 模型"
    echo "模型路径: $MODEL_PATH"
    echo "请先下载模型"
    exit 1
fi

# 检查 GPU
echo ""
echo "检查 GPU 状态..."
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU 数量: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('未检测到 GPU，将使用 CPU（速度较慢）')"

echo ""
echo "======================================"
echo "  启动 DeepSeek-OCR 服务"
echo "  端口: 5001"
echo "  按 Ctrl+C 停止服务"
echo "======================================"
echo ""

# 启动服务
python deepseek_ocr_server.py

