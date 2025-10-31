#!/usr/bin/env python3
"""
DeepSeek-OCR 服务
支持多种 OCR 功能：视觉问答、文档识别、Markdown 转换等
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from transformers import AutoModel, AutoTokenizer
import os
import tempfile
import logging
from pathlib import Path
import json
from datetime import datetime
import base64
from PIL import Image
from io import BytesIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PDF 支持
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
    logger.info("PDF 支持已启用")
except ImportError:
    PDF_SUPPORT = False
    logger.warning("pdf2image 未安装，将不支持 PDF 文件处理")

app = Flask(__name__)

def convert_pdf_to_images(pdf_path, output_folder):
    """
    将 PDF 文件转换为图像
    
    参数:
        pdf_path: PDF 文件路径
        output_folder: 输出文件夹
    
    返回:
        图像文件路径列表
    """
    if not PDF_SUPPORT:
        raise RuntimeError("PDF 支持未启用，请安装 pdf2image 和 poppler-utils")
    
    try:
        # 转换 PDF 为图像（每页一张）
        images = convert_from_path(pdf_path, dpi=300)
        
        image_paths = []
        pdf_name = Path(pdf_path).stem
        
        for i, image in enumerate(images):
            # 保存为 PNG
            image_path = os.path.join(output_folder, f"{pdf_name}_page_{i+1}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
            logger.info(f"转换 PDF 第 {i+1} 页 -> {image_path}")
        
        return image_paths
    except Exception as e:
        logger.error(f"PDF 转换失败: {e}")
        raise RuntimeError(f"PDF 转换失败: {str(e)}")

# 配置 CORS - 只允许特定源，避免多个 Origin 值冲突
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # 开发环境允许所有源
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 全局变量
ocr_model = None
ocr_tokenizer = None
MODEL_PATH = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr'

# 上传文件存储路径
UPLOAD_FOLDER = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/data/uploads'
OUTPUT_FOLDER = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/data/ocr_outputs'

# 确保文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# OCR 功能模板
OCR_TEMPLATES = {
    'free_ocr': {
        'name': '自由OCR识别',
        'prompt': '<image>\nFree OCR.',
        'description': '提取图像中的所有文本内容'
    },
    'markdown': {
        'name': '转换为Markdown',
        'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
        'description': '将文档转换为结构化的 Markdown 格式'
    },
    'table_extract': {
        'name': '表格提取',
        'prompt': '<image>\n<|grounding|>Extract all tables from this document and convert to markdown table format.',
        'description': '识别并提取文档中的表格'
    },
    'formula_extract': {
        'name': '公式识别',
        'prompt': '<image>\nExtract all mathematical formulas in LaTeX format.',
        'description': '识别数学公式并转换为 LaTeX 格式'
    },
    'visual_qa': {
        'name': '视觉问答',
        'prompt': '<image>\n{question}',
        'description': '回答关于图像内容的问题'
    },
    'layout_analysis': {
        'name': '布局分析',
        'prompt': '<image>\n<|grounding|>Analyze the layout of this document.',
        'description': '分析文档的布局结构'
    },
    'key_value_extract': {
        'name': '键值对提取',
        'prompt': '<image>\n<|grounding|>Extract all key-value pairs from this document.',
        'description': '提取文档中的结构化键值对信息'
    }
}

# 分辨率模式配置
RESOLUTION_MODES = {
    'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False},
    'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False},
    'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False},
    'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False},
    'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True}  # 推荐
}

# CORS 已通过 flask-cors 配置，无需手动添加响应头

def load_ocr_model():
    """加载 DeepSeek-OCR 模型"""
    global ocr_model, ocr_tokenizer
    
    if ocr_model is None:
        logger.info(f"加载 DeepSeek-OCR 模型: {MODEL_PATH}")
        try:
            # 检查是否有 CUDA 可用
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"使用设备: {device}")
            
            if device == 'cpu':
                logger.warning("未检测到 GPU，将使用 CPU 运行（速度较慢）")
            
            # 加载 tokenizer
            ocr_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, 
                trust_remote_code=True
            )
            
            # 加载模型
            if device == 'cuda':
                try:
                    # 尝试使用 Flash Attention 2
                    logger.info("尝试使用 Flash Attention 2...")
                    ocr_model = AutoModel.from_pretrained(
                        MODEL_PATH,
                        _attn_implementation='flash_attention_2',
                        trust_remote_code=True,
                        use_safetensors=True
                    )
                    logger.info("成功加载模型（Flash Attention 2）")
                except Exception as e:
                    # 降级到标准 attention
                    logger.warning(f"Flash Attention 2 加载失败: {e}")
                    logger.info("降级到标准 attention...")
                    ocr_model = AutoModel.from_pretrained(
                        MODEL_PATH,
                        trust_remote_code=True,
                        use_safetensors=True,
                        torch_dtype=torch.bfloat16
                    )
                    logger.info("成功加载模型（标准 attention）")
                
                ocr_model = ocr_model.eval().cuda().to(torch.bfloat16)
            else:
                ocr_model = AutoModel.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    use_safetensors=True
                )
                ocr_model = ocr_model.eval()
            
            logger.info("DeepSeek-OCR 模型加载成功")
        except Exception as e:
            logger.error(f"DeepSeek-OCR 模型加载失败: {e}", exc_info=True)
            raise
    
    return ocr_model, ocr_tokenizer

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        model, tokenizer = load_ocr_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return jsonify({
            'status': 'ok',
            'model': 'DeepSeek-OCR',
            'model_loaded': model is not None and tokenizer is not None,
            'device': device,
            'model_path': MODEL_PATH
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/ocr/functions', methods=['GET'])
def list_functions():
    """列出所有可用的 OCR 功能"""
    return jsonify({
        'functions': [
            {
                'id': key,
                'name': value['name'],
                'description': value['description']
            }
            for key, value in OCR_TEMPLATES.items()
        ]
    })

@app.route('/api/ocr/resolutions', methods=['GET'])
def list_resolutions():
    """列出所有可用的分辨率模式"""
    return jsonify({
        'modes': [
            {
                'id': key,
                'config': value
            }
            for key, value in RESOLUTION_MODES.items()
        ],
        'recommended': 'gundam'
    })

@app.route('/api/ocr/process', methods=['POST'])
def process_ocr():
    """
    OCR 处理接口
    
    请求参数:
    - file: 图像/文档文件 (multipart/form-data)
    - function: OCR 功能类型 (free_ocr, markdown, table_extract等)
    - question: 自定义问题（仅用于 visual_qa）
    - resolution: 分辨率模式 (tiny, small, base, large, gundam)
    - save_result: 是否保存结果到文件 (true/false)
    
    返回:
    {
        "text": "识别的文本/处理结果",
        "function": "使用的功能",
        "metadata": {...}  # 额外信息
    }
    """
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '未找到文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 获取参数
        function_type = request.form.get('function', 'free_ocr')
        resolution_mode = request.form.get('resolution', 'gundam')
        custom_question = request.form.get('question', '')
        save_result = request.form.get('save_result', 'false').lower() == 'true'
        
        # 验证参数
        if function_type not in OCR_TEMPLATES:
            return jsonify({'error': f'不支持的功能类型: {function_type}'}), 400
        
        if resolution_mode not in RESOLUTION_MODES:
            return jsonify({'error': f'不支持的分辨率模式: {resolution_mode}'}), 400
        
        # 保存上传的文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        logger.info(f"处理文件: {filename}, 功能: {function_type}, 分辨率: {resolution_mode}")
        
        try:
            # 检查文件类型，如果是 PDF 需要转换
            file_ext = Path(filename).suffix.lower()
            files_to_process = []
            
            if file_ext == '.pdf':
                logger.info(f"检测到 PDF 文件，正在转换为图像...")
                if not PDF_SUPPORT:
                    return jsonify({
                        'error': 'PDF 支持未启用',
                        'message': '请安装 pdf2image 和 poppler-utils: sudo apt-get install poppler-utils'
                    }), 400
                
                # 转换 PDF 为图像
                output_path = os.path.join(OUTPUT_FOLDER, timestamp)
                os.makedirs(output_path, exist_ok=True)
                
                try:
                    files_to_process = convert_pdf_to_images(file_path, output_path)
                except RuntimeError as e:
                    return jsonify({
                        'error': 'PDF 转换失败',
                        'message': str(e)
                    }), 500
            else:
                # 图像文件，直接处理
                files_to_process = [file_path]
            
            # 加载模型
            model, tokenizer = load_ocr_model()
            
            # 构建 prompt
            template = OCR_TEMPLATES[function_type]
            if function_type == 'visual_qa' and custom_question:
                prompt = template['prompt'].format(question=custom_question)
            else:
                prompt = template['prompt']
            
            # 获取分辨率配置
            res_config = RESOLUTION_MODES[resolution_mode]
            
            # 执行 OCR 处理
            if file_ext != '.pdf':
                output_path = os.path.join(OUTPUT_FOLDER, timestamp)
                os.makedirs(output_path, exist_ok=True)
            
            # 处理所有文件（PDF 的多页或单个图像）
            all_results = []
            for idx, image_file in enumerate(files_to_process):
                logger.info(f"处理图像 {idx+1}/{len(files_to_process)}: {image_file}")
                
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=image_file,
                    output_path=output_path,
                    base_size=res_config['base_size'],
                    image_size=res_config['image_size'],
                    crop_mode=res_config['crop_mode'],
                    save_results=save_result,
                    test_compress=False
                )
                
                if len(files_to_process) > 1:
                    # 多页 PDF，添加页码标记
                    all_results.append(f"=== 第 {idx+1} 页 ===\n{result}")
                else:
                    all_results.append(result)
            
            # 合并所有结果
            final_result = "\n\n".join(all_results)
            
            logger.info(f"OCR 处理完成，结果长度: {len(final_result) if final_result else 0}")
            
            # 准备响应
            response_data = {
                'text': final_result,
                'function': function_type,
                'function_name': template['name'],
                'metadata': {
                    'filename': file.filename,
                    'resolution_mode': resolution_mode,
                    'timestamp': timestamp,
                    'file_path': file_path if save_result else None,
                    'output_path': output_path if save_result else None,
                    'is_pdf': file_ext == '.pdf',
                    'pages': len(files_to_process) if file_ext == '.pdf' else 1
                }
            }
            
            return jsonify(response_data)
            
        finally:
            # 如果不保存结果，清理上传的文件
            if not save_result:
                try:
                    os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {e}")
    
    except Exception as e:
        logger.error(f"OCR 处理失败: {e}", exc_info=True)
        return jsonify({'error': f'OCR 处理失败: {str(e)}'}), 500

@app.route('/api/ocr/batch', methods=['POST'])
def batch_process():
    """
    批量 OCR 处理接口
    支持一次处理多个文件
    """
    try:
        if 'files' not in request.files:
            return jsonify({'error': '未找到文件'}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': '文件列表为空'}), 400
        
        function_type = request.form.get('function', 'free_ocr')
        resolution_mode = request.form.get('resolution', 'gundam')
        
        # 加载模型
        model, tokenizer = load_ocr_model()
        
        results = []
        for file in files:
            if file.filename == '':
                continue
            
            try:
                # 保存文件
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"{timestamp}_{file.filename}"
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                
                # 处理
                template = OCR_TEMPLATES[function_type]
                res_config = RESOLUTION_MODES[resolution_mode]
                output_path = os.path.join(OUTPUT_FOLDER, timestamp)
                os.makedirs(output_path, exist_ok=True)
                
                result = model.infer(
                    tokenizer,
                    prompt=template['prompt'],
                    image_file=file_path,
                    output_path=output_path,
                    base_size=res_config['base_size'],
                    image_size=res_config['image_size'],
                    crop_mode=res_config['crop_mode'],
                    save_results=True
                )
                
                results.append({
                    'filename': file.filename,
                    'text': result,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"处理文件 {file.filename} 失败: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return jsonify({
            'results': results,
            'total': len(files),
            'success': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed'])
        })
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}", exc_info=True)
        return jsonify({'error': f'批量处理失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 预加载模型
    logger.info("启动 DeepSeek-OCR 服务...")
    try:
        load_ocr_model()
        logger.info("模型预加载完成")
    except Exception as e:
        logger.error(f"模型预加载失败: {e}")
        logger.warning("服务将启动，但模型加载失败。请检查模型路径和 GPU 配置。")
    
    # 启动 Flask 服务
    app.run(
        host='0.0.0.0',
        port=5001,  # 使用 5001 端口（5000 已被 Whisper 占用）
        debug=False,
        threaded=True
    )

