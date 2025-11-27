# PrivaSee Backend Services
# 模块化服务包

from .ocr_service import ocr_bp
from .whisper_service import whisper_bp

__all__ = ['ocr_bp', 'whisper_bp']
