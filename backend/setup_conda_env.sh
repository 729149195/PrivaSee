#!/bin/bash
# PrivaSee 统一后端 Conda 环境配置脚本

echo "======================================"
echo "  PrivaSee 统一后端环境设置"
echo "======================================"
echo ""

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 conda，请先安装 Anaconda 或 Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 初始化 conda
eval "$(conda shell.bash hook)"

# 检查环境是否已存在
if conda env list | grep -q "^privasee "; then
    echo "privasee 环境已存在，跳过创建..."
else
    echo "创建 PrivaSee 虚拟环境 (Python 3.10)..."
    conda create -n privasee python=3.10 -y
    
    if [ $? -ne 0 ]; then
        echo "环境创建失败"
        exit 1
    fi
fi

echo ""
echo "激活环境并安装依赖..."

# 激活环境
conda activate privasee

# 安装 ffmpeg（Whisper 必需）
echo ""
echo ">>> 安装 ffmpeg（语音处理必需）..."
conda install -c conda-forge ffmpeg libiconv -y

# 安装 poppler（PDF 处理必需）
echo ""
echo ">>> 安装 poppler（PDF 处理必需）..."
conda install -c conda-forge poppler -y

# 安装 PyTorch（如果有 GPU）
echo ""
echo ">>> 检查 GPU 并安装 PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU，安装 CUDA 版本的 PyTorch..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "未检测到 GPU，安装 CPU 版本的 PyTorch..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# 升级 pip
echo ""
echo ">>> 升级 pip..."
python -m pip install --upgrade pip

# 安装 Python 依赖包
echo ""
echo ">>> 安装 Python 依赖包..."
python -m pip install -r requirements.txt

echo ""
echo "======================================"
echo "  ✅ 安装完成！"
echo "======================================"
echo ""
echo "使用方式："
echo "  1. 激活环境:"
echo "     conda activate privasee"
echo ""
echo "  2. 启动统一后端服务:"
echo "     bash start.sh"
echo ""
echo "  3. 或者使用 Python 直接启动:"
echo "     python app.py"
echo ""
echo "可选参数:"
echo "  bash start.sh --preload       # 预加载模型"
echo "  bash start.sh --whisper-only  # 只启动 Whisper"
echo "  bash start.sh --ocr-only      # 只启动 OCR"
echo "  bash start.sh --port 8000     # 指定端口"
echo ""
echo "API 端点:"
echo "  http://localhost:5000/api/health      # 健康检查"
echo "  http://localhost:5000/api/services    # 服务列表"
echo "  http://localhost:5000/api/ocr/*       # OCR 服务"
echo "  http://localhost:5000/api/whisper/*   # Whisper 服务"
echo ""
echo "查看已安装的包："
echo "  conda activate privasee && pip list"
echo ""

