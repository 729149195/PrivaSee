#!/usr/bin/env python3
"""
Whisper语音转文本服务
支持中英文语音识别
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import tempfile
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置CORS：允许所有来源（开发环境）
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# 全局变量：Whisper模型
whisper_model = None
MODEL_SIZE = 'base'  # 可选: tiny, base, small, medium, large

# 添加CORS响应头
@app.after_request
def after_request(response):
    """为所有响应添加CORS头"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

def load_whisper_model():
    """加载Whisper模型"""
    global whisper_model
    if whisper_model is None:
        logger.info(f"加载Whisper模型: {MODEL_SIZE}")
        try:
            whisper_model = whisper.load_model(MODEL_SIZE)
            logger.info("Whisper模型加载成功")
        except Exception as e:
            logger.error(f"Whisper模型加载失败: {e}")
            raise
    return whisper_model

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        model = load_whisper_model()
        return jsonify({
            'status': 'ok',
            'model': MODEL_SIZE,
            'model_loaded': model is not None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """
    语音转文本接口
    
    请求参数:
    - audio: 音频文件 (multipart/form-data)
    - language: 语言代码 (可选，默认auto自动检测)
    
    返回:
    {
        "text": "识别的文本",
        "language": "检测到的语言",
        "segments": [...],  # 详细分段信息（可选）
    }
    """
    try:
        # 检查是否有文件
        if 'audio' not in request.files:
            return jsonify({'error': '未找到音频文件'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 获取语言参数
        language = request.form.get('language', 'auto')
        if language == 'auto':
            language = None  # Whisper自动检测
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # 加载模型
            model = load_whisper_model()
            
            logger.info(f"开始转录: {audio_file.filename}, 语言: {language or 'auto'}")
            
            # 执行转录
            result = model.transcribe(
                tmp_path,
                language=language,
                task='transcribe',  # 'transcribe' (识别) or 'translate' (翻译成英文)
                fp16=False,  # 在CPU上运行时设为False
                verbose=False
            )
            
            # 提取结果
            text = result['text'].strip()
            detected_language = result.get('language', 'unknown')
            
            logger.info(f"转录完成: {len(text)} 字符, 语言: {detected_language}")
            
            # 返回结果
            response = {
                'text': text,
                'language': detected_language,
            }
            
            # 可选：返回详细分段信息
            if 'segments' in result:
                response['segments'] = [
                    {
                        'start': seg.get('start'),
                        'end': seg.get('end'),
                        'text': seg.get('text', '').strip()
                    }
                    for seg in result['segments']
                ]
            
            return jsonify(response)
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")
    
    except Exception as e:
        logger.error(f"转录失败: {e}", exc_info=True)
        return jsonify({'error': f'转录失败: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用的Whisper模型"""
    models = ['tiny', 'base', 'small', 'medium', 'large']
    return jsonify({
        'available_models': models,
        'current_model': MODEL_SIZE
    })

@app.route('/api/languages', methods=['GET'])
def list_languages():
    """列出支持的语言"""
    # Whisper支持的主要语言
    languages = {
        'auto': '自动检测',
        'zh': '中文',
        'en': '英文',
        # 'ja': '日语',
        # 'ko': '韩语',
        # 'es': '西班牙语',
        # 'fr': '法语',
        # 'de': '德语',
        # 'ru': '俄语',
        # 'ar': '阿拉伯语',
    }
    return jsonify(languages)

if __name__ == '__main__':
    # 预加载模型
    logger.info("启动Whisper服务...")
    try:
        load_whisper_model()
        logger.info("模型预加载完成")
    except Exception as e:
        logger.error(f"模型预加载失败: {e}")
    
    # 启动Flask服务
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

