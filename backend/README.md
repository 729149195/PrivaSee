# PrivaSee 统一后端服务

模块化的后端服务，整合 OCR 和 Whisper 功能。

## 目录结构

```
backend/
├── app.py                  # 主入口文件
├── config.py               # 统一配置
├── requirements.txt        # Python 依赖
├── start.sh               # 统一启动脚本
├── setup_conda_env.sh     # Conda 环境设置脚本
├── services/              # 服务模块
│   ├── __init__.py
│   ├── ocr_service.py     # OCR 服务 (Flask Blueprint)
│   └── whisper_service.py # Whisper 服务 (Flask Blueprint)
└── README.md
```

## 快速开始

### 1. 设置环境

```bash
# 运行设置脚本（自动创建 conda 环境并安装依赖）
bash setup_conda_env.sh

# 激活环境
conda activate privasee
```

### 2. 启动服务

```bash
# 启动完整服务（OCR + Whisper）
bash start.sh

# 或使用 Python 直接启动
python app.py
```

### 3. 启动选项

```bash
# 预加载模型（启动较慢，但首次请求快）
bash start.sh --preload

# 只启动 Whisper 服务
bash start.sh --whisper-only

# 只启动 OCR 服务
bash start.sh --ocr-only

# 指定端口
bash start.sh --port 8000

# 调试模式
bash start.sh --debug
```

## API 端点

### 通用接口

| 端点 | 描述 |
|------|------|
| `GET /api/health` | 健康检查 |
| `GET /api/services` | 服务列表 |

### OCR 服务 (`/api/ocr/*`)

| 端点 | 描述 |
|------|------|
| `GET /api/ocr/health` | OCR 健康检查 |
| `GET /api/ocr/functions` | 列出 OCR 功能 |
| `GET /api/ocr/resolutions` | 列出分辨率模式 |
| `POST /api/ocr/upload` | 上传文件 |
| `POST /api/ocr/process` | 处理文件（同步） |
| `POST /api/ocr/process/stream` | 处理文件（流式 SSE） |

### Whisper 服务 (`/api/whisper/*`)

| 端点 | 描述 |
|------|------|
| `GET /api/whisper/health` | Whisper 健康检查 |
| `POST /api/whisper/transcribe` | 语音转文本 |
| `GET /api/whisper/models` | 列出可用模型 |
| `GET /api/whisper/languages` | 列出支持的语言 |

## 配置

编辑 `config.py` 修改配置：

```python
# 服务端口
SERVER_PORT = 5000

# 模型配置
DEEPSEEK_OCR_MODEL_PATH = '/path/to/deepseek-ocr'
WHISPER_MODEL_SIZE = 'base'  # tiny, base, small, medium, large

# 文件上传限制
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
```

## 依赖说明

- **PyTorch**: 深度学习框架（建议通过 conda 安装）
- **transformers**: Hugging Face 模型库
- **openai-whisper**: 语音识别
- **flask**: Web 框架
- **pdf2image**: PDF 转图片（需要 poppler）

## 系统依赖

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg poppler-utils libreoffice

# macOS
brew install ffmpeg poppler libreoffice
```
