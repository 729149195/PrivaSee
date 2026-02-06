#!/usr/bin/env python3
"""
PrivaSee 统一后端服务
整合 OCR 和 Whisper 功能模块
"""

import os
import sys
import logging
import argparse

from flask import Flask, jsonify
from flask_cors import CORS

# 配置日志
from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, SERVER_HOST, SERVER_PORT, MAX_CONTENT_LENGTH

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Flask 应用创建
# =============================================================================

def create_app(enable_ocr: bool = True, enable_whisper: bool = True):
    """
    创建 Flask 应用
    
    Args:
        enable_ocr: 是否启用 OCR 服务
        enable_whisper: 是否启用 Whisper 服务
    """
    app = Flask(__name__)
    
    # 配置
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    
    # CORS 配置
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # 添加通用 CORS 响应头
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Max-Age', '3600')
        return response
    
    # 注册服务模块
    enabled_services = []
    
    if enable_ocr:
        try:
            from services.ocr_service import ocr_bp, init_ocr_service
            app.register_blueprint(ocr_bp)
            enabled_services.append('OCR')
        except ImportError as e:
            logger.warning(f"OCR import failed: {e}")
    
    if enable_whisper:
        try:
            from services.whisper_service import whisper_bp, init_whisper_service
            app.register_blueprint(whisper_bp)
            enabled_services.append('Whisper')
        except ImportError as e:
            logger.warning(f"Whisper import failed: {e}")
    
    # 主记忆流服务 (Memory Stream) - 始终启用
    try:
        from services.memory_stream_service import memory_bp, init_memory_stream_service
        app.register_blueprint(memory_bp)
        enabled_services.append('MemoryStream')
    except ImportError as e:
        logger.warning(f"MemoryStream import failed: {e}")
    
    # =============================================================================
    # 通用 API 路由
    # =============================================================================
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """统一健康检查"""
        import torch
        
        status = {
            'status': 'ok',
            'service': 'PrivaSee Backend',
            'enabled_services': enabled_services,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        if torch.cuda.is_available():
            status['gpu'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            }
        
        return jsonify(status)
    
    @app.route('/api/services', methods=['GET'])
    def list_services():
        """列出所有可用服务"""
        services = []
        
        if 'OCR' in enabled_services:
            services.append({
                'name': 'OCR',
                'description': 'DeepSeek-OCR 文档识别服务',
                'endpoints': {
                    'health': '/api/ocr/health',
                    'functions': '/api/ocr/functions',
                    'resolutions': '/api/ocr/resolutions',
                    'upload': '/api/ocr/upload',
                    'process': '/api/ocr/process',
                    'process_stream': '/api/ocr/process/stream',
                    'unload': '/api/ocr/unload (POST - 卸载模型释放显存)'
                }
            })
        
        if 'Whisper' in enabled_services:
            services.append({
                'name': 'Whisper',
                'description': 'Whisper 语音转文本服务',
                'endpoints': {
                    'health': '/api/whisper/health',
                    'transcribe': '/api/whisper/transcribe',
                    'models': '/api/whisper/models',
                    'languages': '/api/whisper/languages'
                }
            })
        
        if 'MemoryStream' in enabled_services:
            services.append({
                'name': 'MemoryStream',
                'description': '主记忆流与关联回溯服务',
                'endpoints': {
                    'health': '/api/memory/health',
                    'ingest': '/api/memory/ingest (POST - 批量写入信息元)',
                    'search': '/api/memory/search (POST - 向量相似度检索)',
                    'trigger_check': '/api/memory/trigger-check (POST - 风险触发检索)',
                    'backtrace': '/api/memory/backtrace/<iid> (GET - 关联回溯查询)',
                    'clear': '/api/memory/clear (POST - 一键清空)',
                    'stats': '/api/memory/stats'
                }
            })
        
        return jsonify({
            'services': services,
            'total': len(services)
        })
    
    @app.route('/', methods=['GET'])
    def index():
        """首页"""
        return jsonify({
            'name': 'PrivaSee Backend API',
            'version': '1.0.0',
            'services': enabled_services,
            'docs': '/api/services'
        })
    
    return app

# =============================================================================
# 主程序
# =============================================================================

def main():
    """启动服务"""
    parser = argparse.ArgumentParser(description='PrivaSee 后端服务')
    parser.add_argument('--host', default=SERVER_HOST, help='监听地址')
    parser.add_argument('--port', type=int, default=SERVER_PORT, help='监听端口')
    parser.add_argument('--no-ocr', action='store_true', help='禁用 OCR 服务')
    parser.add_argument('--no-whisper', action='store_true', help='禁用 Whisper 服务')
    parser.add_argument('--preload', action='store_true', help='预加载模型')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 创建应用
    app = create_app(
        enable_ocr=not args.no_ocr,
        enable_whisper=not args.no_whisper
    )
    
    # 预加载模型
    if args.preload:
        if not args.no_ocr:
            try:
                from services.ocr_service import init_ocr_service
                init_ocr_service(preload_model=True)
            except Exception as e:
                logger.warning(f"OCR preload failed: {e}")
        
        if not args.no_whisper:
            try:
                from services.whisper_service import init_whisper_service
                init_whisper_service(preload_model=True)
            except Exception as e:
                logger.warning(f"Whisper preload failed: {e}")
    
    # 启动服务
    logger.info(f"PrivaSee Backend running at http://{args.host}:{args.port}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == '__main__':
    main()
