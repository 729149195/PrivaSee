#!/bin/bash
# DeepSeek-OCR 全新安装脚本
# 基于官方教程: https://huggingface.co/deepseek-ai/DeepSeek-OCR

set -e  # 遇到错误立即退出

echo "======================================"
echo "  DeepSeek-OCR 全新安装"
echo "======================================"
echo ""

# 初始化 conda
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 conda"
    exit 1
fi

eval "$(conda shell.bash hook)"

# 删除旧环境（如果存在）
echo "检查并删除旧环境..."
if conda env list | grep -q "deepseek-ocr"; then
    echo "删除旧的 deepseek-ocr 环境..."
    conda remove -n deepseek-ocr --all -y
fi

# 创建新环境
echo ""
echo "创建新的 conda 环境: deepseek-ocr (Python 3.12)..."
conda create -n deepseek-ocr python=3.12 -y

# 激活环境
echo ""
echo "激活 deepseek-ocr 环境..."
conda activate deepseek-ocr

# 检查 CUDA 版本
echo ""
echo "检查 CUDA 版本..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader
fi

# 安装 PyTorch (CUDA 11.8)
echo ""
echo "安装 PyTorch 2.6.0 (CUDA 11.8)..."
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 transformers 和其他依赖
echo ""
echo "安装 transformers 和相关依赖..."
pip install transformers==4.46.3
pip install tokenizers==0.20.3
pip install einops
pip install addict
pip install easydict

# 安装 Flash Attention 2
echo ""
echo "安装 Flash Attention 2..."
pip install flash-attn==2.7.3 --no-build-isolation || echo "警告: Flash Attention 2 安装失败，将使用标准 attention"

# 安装其他必需的包
echo ""
echo "安装其他依赖包..."
pip install flask flask-cors
pip install pillow
pip install pdf2image
pip install numpy

# 创建模型目录
echo ""
echo "创建模型目录..."
mkdir -p /home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr

# 下载模型
echo ""
echo "======================================"
echo "  下载 DeepSeek-OCR 模型"
echo "======================================"
echo ""
echo "正在从 Hugging Face 下载模型..."
echo "这可能需要较长时间，请耐心等待..."
echo ""

cd /home/zhangxiangxuan/桌面/Projects/PrivaSee/models

python -c "
from transformers import AutoModel, AutoTokenizer
import os

model_name = 'deepseek-ai/DeepSeek-OCR'
save_path = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr'

print('正在下载 tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print('正在下载模型...')
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True
)
model.save_pretrained(save_path)

print('✓ 模型下载完成！')
"

echo ""
echo "======================================"
echo "  安装完成！"
echo "======================================"
echo ""
echo "环境名称: deepseek-ocr"
echo "Python 版本: 3.12"
echo "模型路径: /home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr"
echo ""
echo "使用方法:"
echo "  conda activate deepseek-ocr"
echo "  python deepseek_ocr_server.py"
echo ""

