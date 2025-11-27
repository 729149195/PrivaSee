#!/bin/bash
# PrivaSee 统一后端启动脚本

echo "======================================"
echo "  PrivaSee 后端服务"
echo "======================================"
echo ""

# 默认参数
HOST="0.0.0.0"
PORT="5000"
PRELOAD=""
DEBUG=""
NO_OCR=""
NO_WHISPER=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --preload)
            PRELOAD="--preload"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --no-ocr)
            NO_OCR="--no-ocr"
            shift
            ;;
        --no-whisper)
            NO_WHISPER="--no-whisper"
            shift
            ;;
        --whisper-only)
            NO_OCR="--no-ocr"
            shift
            ;;
        --ocr-only)
            NO_WHISPER="--no-whisper"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --host HOST       监听地址 (默认: 0.0.0.0)"
            echo "  --port PORT       监听端口 (默认: 5000)"
            echo "  --preload         预加载模型"
            echo "  --debug           调试模式"
            echo "  --no-ocr          禁用 OCR 服务"
            echo "  --no-whisper      禁用 Whisper 服务"
            echo "  --whisper-only    只启动 Whisper 服务"
            echo "  --ocr-only        只启动 OCR 服务"
            echo "  -h, --help        显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查并激活 conda 环境
if command -v conda &> /dev/null; then
    echo "检测到 conda，尝试激活 privasee 环境..."
    eval "$(conda shell.bash hook)"
    
    if conda env list | grep -q "^privasee "; then
        echo "✓ 激活 privasee 环境..."
        conda activate privasee
    else
        echo ""
        echo "❌ 错误: privasee 环境不存在"
        echo ""
        echo "请先运行以下命令创建环境："
        echo "  bash setup_conda_env.sh"
        echo ""
        exit 1
    fi
else
    echo "⚠ 未检测到 conda，使用系统 Python"
fi

echo ""
echo "服务地址: http://$HOST:$PORT"
echo "API 文档: http://$HOST:$PORT/api/services"
echo ""
echo "启动选项:"
[ -n "$PRELOAD" ] && echo "  - 预加载模型"
[ -n "$DEBUG" ] && echo "  - 调试模式"
[ -n "$NO_OCR" ] && echo "  - OCR 服务已禁用"
[ -n "$NO_WHISPER" ] && echo "  - Whisper 服务已禁用"
echo ""
echo "按 Ctrl+C 停止服务"
echo "======================================"
echo ""

# 启动服务
python app.py --host "$HOST" --port "$PORT" $PRELOAD $DEBUG $NO_OCR $NO_WHISPER
