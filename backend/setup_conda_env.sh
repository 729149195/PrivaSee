#!/bin/bash
# PrivaSee Conda环境配置脚本

echo "======================================"
echo "  PrivaSee 环境设置"
echo "======================================"
echo ""

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到conda，请先安装Anaconda或Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "创建 PrivaSee 虚拟环境..."
conda create -n privasee python=3.10 -y

if [ $? -ne 0 ]; then
    echo "环境创建失败"
    exit 1
fi

echo ""
echo "激活环境并安装依赖..."

# 初始化conda以便在脚本中使用activate
eval "$(conda shell.bash hook)"

# 激活环境
conda activate privasee

# 首先安装ffmpeg和依赖（Whisper必需）
echo "安装 ffmpeg 和依赖（语音处理必需）..."
conda install -c conda-forge ffmpeg libiconv -y

# 使用环境中的pip安装依赖
echo "安装 Python 依赖包..."
python -m pip install --upgrade pip
python -m pip install openai-whisper==20231117
python -m pip install flask==3.0.0
python -m pip install flask-cors==4.0.0
python -m pip install faiss-cpu==1.12.0
python -m pip install pymongo==4.15.1
python -m pip install sentence-transformers==5.1.1
python -m pip install scikit-learn==1.7.2
python -m pip install numpy==1.26.3

echo ""
echo "======================================"
echo "  ✅ 安装完成！"
echo "======================================"
echo ""
echo "使用方式："
echo "  1. 激活环境: conda activate privasee"
echo "  2. 启动Whisper服务: python whisper_server.py"
echo "  3. 或直接运行: bash start_whisper.sh"
echo ""
echo "查看已安装的包："
echo "  conda activate privasee && conda list"
echo ""

