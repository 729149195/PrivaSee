#!/bin/bash
# Whisper服务启动脚本

echo "======================================"
echo "  启动 Whisper 语音转文本服务"
echo "======================================"
echo "服务地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

# 检查conda是否可用
if command -v conda &> /dev/null; then
    echo "检测到conda，尝试激活privasee环境..."
    # 初始化conda
    eval "$(conda shell.bash hook)"
    
    # 检查privasee环境是否存在
    if conda env list | grep -q "^privasee "; then
        echo "✓ 激活privasee环境..."
        conda activate privasee
    else
        echo ""
        echo "❌ 错误: privasee环境不存在"
        echo ""
        echo "请先运行以下命令创建环境："
        echo "  bash setup_conda_env.sh"
        echo ""
        exit 1
    fi
# 否则尝试venv
elif [ -d "venv" ]; then
    echo "使用venv环境..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "使用venv环境..."
    source ../venv/bin/activate
else
    echo "警告: 未检测到虚拟环境"
fi

echo ""
echo "======================================"
# 启动服务
python whisper_server.py

