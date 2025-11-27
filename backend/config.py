"""
PrivaSee 后端配置文件
统一管理所有服务的配置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path('/home/zhangxiangxuan/桌面/Projects/PrivaSee')
BACKEND_ROOT = PROJECT_ROOT / 'backend'

# =============================================================================
# 服务端口配置
# =============================================================================
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000  # 统一端口

# =============================================================================
# 文件存储路径
# =============================================================================
DATA_DIR = PROJECT_ROOT / 'data'
UPLOAD_FOLDER = DATA_DIR / 'uploads'
OUTPUT_FOLDER = DATA_DIR / 'outputs'

# OCR 专用目录
OCR_UPLOAD_FOLDER = UPLOAD_FOLDER / 'ocr'
OCR_OUTPUT_FOLDER = OUTPUT_FOLDER / 'ocr'

# Whisper 专用目录
WHISPER_UPLOAD_FOLDER = UPLOAD_FOLDER / 'audio'

# 确保目录存在
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, OCR_UPLOAD_FOLDER, OCR_OUTPUT_FOLDER, WHISPER_UPLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# 模型配置
# =============================================================================
MODELS_DIR = PROJECT_ROOT / 'models'

# DeepSeek-OCR 模型路径
DEEPSEEK_OCR_MODEL_PATH = MODELS_DIR / 'deepseek-ocr'

# Whisper 模型配置
WHISPER_MODEL_SIZE = 'base'  # 可选: tiny, base, small, medium, large

# =============================================================================
# 文件上传限制
# =============================================================================
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# =============================================================================
# 日志配置
# =============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BACKEND_ROOT / 'privasee_backend.log'
