#!/usr/bin/env python3
"""
OCR 服务模块 - 使用 DeepSeek-OCR 模型
支持图片和 PDF 文档处理，提供多种 OCR 功能
"""

import os
import sys
import logging
import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from flask import Blueprint, request, jsonify, Response, stream_with_context
import torch

# 创建 Blueprint
ocr_bp = Blueprint('ocr', __name__, url_prefix='/api/ocr')

logger = logging.getLogger(__name__)

# =============================================================================
# 配置
# =============================================================================

# 延迟导入配置，避免循环导入
def get_config():
    from config import (
        DEEPSEEK_OCR_MODEL_PATH, OCR_UPLOAD_FOLDER, OCR_OUTPUT_FOLDER
    )
    return {
        'model_path': str(DEEPSEEK_OCR_MODEL_PATH),
        'upload_folder': str(OCR_UPLOAD_FOLDER),
        'output_folder': str(OCR_OUTPUT_FOLDER)
    }

# 全局模型变量
ocr_model = None
ocr_tokenizer = None

# PDF 支持
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
    logger.info("✓ PDF 支持已启用")
except ImportError:
    PDF_SUPPORT = False
    logger.warning("✗ PDF 支持未启用 (需要安装 pdf2image 和 poppler-utils)")

# =============================================================================
# OCR 功能模板定义
# =============================================================================

OCR_FUNCTIONS = {
    'free_ocr': {
        'name': '自由OCR识别',
        'description': '提取图像中的文本内容',
        'prompt_template': '<image>\nFree OCR.',
        'supports_custom_question': False
    },
    'markdown': {
        'name': '转换为Markdown',
        'description': '转换为结构化 Markdown',
        'prompt_template': '<image>\n<|grounding|>Convert the document to markdown format.',
        'supports_custom_question': False
    },
    'table_extract': {
        'name': '表格提取',
        'description': '识别并提取表格数据',
        'prompt_template': '<image>\n<|grounding|>Extract all tables from this document and convert to markdown table format.',
        'supports_custom_question': False
    },
    'formula_extract': {
        'name': '公式识别',
        'description': '转换为 LaTeX 格式',
        'prompt_template': '<image>\nExtract all mathematical formulas and convert them to LaTeX format.',
        'supports_custom_question': False
    },
    'visual_qa': {
        'name': '视觉问答',
        'description': '回答图像相关问题',
        'prompt_template': '<image>\n{question}',
        'supports_custom_question': True
    },
    'layout_analysis': {
        'name': '布局分析',
        'description': '分析文档结构信息',
        'prompt_template': '<image>\n<|grounding|>Analyze the document layout and structure.',
        'supports_custom_question': False
    },
    'key_value_extract': {
        'name': '键值对提取',
        'description': '提取结构化信息',
        'prompt_template': '<image>\n<|grounding|>Extract all key-value pairs from this document.',
        'supports_custom_question': False
    }
}

# 分辨率模式配置
RESOLUTION_MODES = {
    'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False, 'description': '快速预览（512px）'},
    'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False, 'description': '标准处理（640px）'},
    'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False, 'description': '高质量（1024px）'},
    'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False, 'description': '超高质量（1280px）'},
    'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True, 'description': '推荐模式（智能裁剪）'}
}

# =============================================================================
# 模型加载
# =============================================================================

def load_model():
    """加载 DeepSeek-OCR 模型"""
    global ocr_model, ocr_tokenizer
    
    if ocr_model is not None and ocr_tokenizer is not None:
        return ocr_model, ocr_tokenizer
    
    config = get_config()
    model_path = config['model_path']
    
    logger.info("=" * 70)
    logger.info("开始加载 DeepSeek-OCR 模型...")
    logger.info(f"模型路径: {model_path}")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device.upper()}")
        
        if device == 'cuda':
            logger.info(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        
        # 加载 tokenizer
        logger.info("加载 Tokenizer...")
        ocr_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if ocr_tokenizer.pad_token is None:
            ocr_tokenizer.pad_token = ocr_tokenizer.eos_token
        
        logger.info("✓ Tokenizer 加载完成")
        
        # 加载模型
        logger.info("加载模型权重...")
        if device == 'cuda':
            try:
                ocr_model = AutoModel.from_pretrained(
                    model_path,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True,
                    torch_dtype=torch.bfloat16
                )
                logger.info("✓ 已启用 Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 2 不可用: {e}")
                ocr_model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_safetensors=True,
                    torch_dtype=torch.bfloat16
                )
            ocr_model = ocr_model.eval().cuda()
        else:
            ocr_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_safetensors=True
            )
            ocr_model = ocr_model.eval()
        
        logger.info("✓ DeepSeek-OCR 模型加载完成")
        logger.info("=" * 70)
        
        return ocr_model, ocr_tokenizer
        
    except Exception as e:
        logger.error(f"✗ 模型加载失败: {e}", exc_info=True)
        raise

# =============================================================================
# 工具函数
# =============================================================================

def convert_office_to_pdf(input_path: str, output_dir: str) -> str:
    """将 Office 文档转换为 PDF"""
    import subprocess
    
    logger.info(f"转换 Office 文档为 PDF: {input_path}")
    
    try:
        subprocess.run(['libreoffice', '--version'], capture_output=True, check=True, timeout=5)
    except:
        raise RuntimeError("LibreOffice 未安装，请安装: sudo apt-get install libreoffice")
    
    cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, input_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice 转换失败: {result.stderr}")
    
    pdf_path = os.path.join(output_dir, f"{Path(input_path).stem}.pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"PDF 文件未生成: {pdf_path}")
    
    return pdf_path

def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> List[str]:
    """将 PDF 转换为图片列表"""
    if not PDF_SUPPORT:
        raise RuntimeError("PDF 支持未启用，请安装: sudo apt-get install poppler-utils && pip install pdf2image")
    
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    pdf_name = Path(pdf_path).stem
    
    for i, image in enumerate(images, 1):
        image_path = os.path.join(output_dir, f"{pdf_name}_page_{i:03d}.png")
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    
    return image_paths

def cleanup_files(files: List[str], dirs: List[str] = None):
    """清理文件和目录"""
    import shutil
    for f in files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass
    if dirs:
        for d in dirs:
            try:
                if os.path.exists(d):
                    shutil.rmtree(d)
            except:
                pass

# =============================================================================
# 自定义流式 Streamer
# =============================================================================

from transformers import TextStreamer

class CallbackTextStreamer(TextStreamer):
    """自定义 TextStreamer，通过回调函数实时发送生成的文本"""
    def __init__(self, tokenizer, callback=None, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.callback = callback
        self.text_queue = []
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """当文本块生成完成时调用"""
        # 过滤掉 EOS token
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        if eos_text in text:
            text = text.replace(eos_text, '')
        if text and self.callback:
            self.callback(text)

# =============================================================================
# OCR 处理函数
# =============================================================================

def process_image(
    image_path: str,
    function_type: str,
    resolution_mode: str = 'gundam',
    custom_question: str = '',
    save_results: bool = False,
    output_dir: Optional[str] = None,
    stream_callback=None,
    history_messages: Optional[list] = None,
    stop_flag: Optional[dict] = None
) -> str:
    """处理单张图片"""
    config = get_config()
    model, tokenizer = load_model()
    
    if function_type not in OCR_FUNCTIONS:
        raise ValueError(f"不支持的功能类型: {function_type}")
    
    function_config = OCR_FUNCTIONS[function_type]
    prompt_template = function_config['prompt_template']
    
    # 构建 prompt（包含历史消息）
    if history_messages and len(history_messages) > 0:
        context_parts = ["以下是之前的对话记录："]
        for msg in history_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '').strip()
            if content:
                if role == 'user':
                    context_parts.append(f"Q: {content}")
                elif role == 'assistant':
                    if len(content) > 500:
                        content = content[:500] + "..."
                    context_parts.append(f"A: {content}")
        
        context_text = '\n'.join(context_parts)
        current_question = custom_question if custom_question else "请继续回答"
        
        if function_config['supports_custom_question']:
            prompt = f"{context_text}\n\n当前问题：\nQ: {current_question}\nA: "
        else:
            prompt = prompt_template
    else:
        if function_config['supports_custom_question'] and custom_question:
            prompt = prompt_template.format(question=custom_question)
        else:
            prompt = prompt_template
    
    if resolution_mode not in RESOLUTION_MODES:
        raise ValueError(f"不支持的分辨率模式: {resolution_mode}")
    
    res_config = RESOLUTION_MODES[resolution_mode]
    
    if output_dir is None:
        output_dir = os.path.join(config['output_folder'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"处理图片: {Path(image_path).name}")
    logger.info(f"  功能: {function_config['name']}, 分辨率: {resolution_mode}")
    
    custom_streamer = None
    if stream_callback:
        custom_streamer = CallbackTextStreamer(tokenizer, callback=stream_callback, skip_prompt=True)
    
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=res_config['base_size'],
        image_size=res_config['image_size'],
        crop_mode=res_config['crop_mode'],
        save_results=save_results,
        test_compress=False,
        eval_mode=True,
        custom_streamer=custom_streamer
    )
    
    if result is None:
        result = ""
    
    # 检测重复内容
    if result and len(result) > 1000:
        words = result.split()
        unique_words = set(words)
        if len(unique_words) < len(words) * 0.15:
            result = result[:500] + "\n\n[检测到重复内容，已自动截断]"
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result

# =============================================================================
# API 路由
# =============================================================================

@ocr_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        config = get_config()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpu_info = None
        
        if device == 'cuda':
            gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
                'memory_allocated': f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            }
        
        return jsonify({
            'status': 'ok',
            'service': 'DeepSeek-OCR',
            'model_loaded': ocr_model is not None,
            'device': device,
            'gpu_info': gpu_info,
            'pdf_support': PDF_SUPPORT,
            'model_path': config['model_path']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@ocr_bp.route('/functions', methods=['GET'])
def list_functions():
    """列出所有可用功能"""
    return jsonify({
        'functions': [
            {
                'id': key,
                'name': value['name'],
                'description': value['description'],
                'supports_custom_question': value['supports_custom_question']
            }
            for key, value in OCR_FUNCTIONS.items()
        ]
    })

@ocr_bp.route('/resolutions', methods=['GET'])
def list_resolutions():
    """列出所有分辨率模式"""
    return jsonify({
        'modes': [
            {
                'id': key,
                'description': value['description'],
                'config': {'base_size': value['base_size'], 'image_size': value['image_size'], 'crop_mode': value['crop_mode']}
            }
            for key, value in RESOLUTION_MODES.items()
        ],
        'recommended': 'gundam'
    })

@ocr_bp.route('/upload', methods=['POST'])
def upload_file():
    """仅上传文件"""
    try:
        config = get_config()
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'}), 400
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        upload_path = os.path.join(config['upload_folder'], filename)
        file.save(upload_path)
        
        logger.info(f"✓ 文件上传成功: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': upload_path,
            'originalName': file.filename,
            'size': os.path.getsize(upload_path)
        })
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@ocr_bp.route('/process', methods=['POST'])
def process_file():
    """处理文件（图片或 PDF）"""
    config = get_config()
    dirs_to_cleanup = []
    
    try:
        function_type = request.form.get('function', 'free_ocr')
        resolution_mode = request.form.get('resolution', 'gundam')
        custom_question = request.form.get('question', '')
        save_results = request.form.get('save_results', 'false').lower() == 'true'
        uploaded_filename = request.form.get('uploaded_filename', '')
        
        # 历史消息
        history_messages = []
        messages_json = request.form.get('messages', '')
        if messages_json:
            try:
                history_messages = json.loads(messages_json)
            except:
                pass
        
        # 处理文件
        if uploaded_filename:
            upload_path = os.path.join(config['upload_folder'], uploaded_filename)
            if not os.path.exists(upload_path):
                return jsonify({'success': False, 'error': '文件不存在'}), 400
            filename = uploaded_filename
            parts = filename.split('_', 2)
            original_filename = parts[2] if len(parts) >= 3 else filename
        else:
            if 'file' not in request.files:
                return jsonify({'error': '未上传文件'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '文件名为空'}), 400
            original_filename = file.filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            upload_path = os.path.join(config['upload_folder'], filename)
            file.save(upload_path)
        
        if function_type not in OCR_FUNCTIONS:
            return jsonify({'error': f'不支持的功能: {function_type}'}), 400
        if resolution_mode not in RESOLUTION_MODES:
            return jsonify({'error': f'不支持的分辨率: {resolution_mode}'}), 400
        
        logger.info(f"处理请求: {filename}, 功能: {OCR_FUNCTIONS[function_type]['name']}")
        
        try:
            file_ext = Path(filename).suffix.lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(config['output_folder'], timestamp)
            os.makedirs(output_dir, exist_ok=True)
            dirs_to_cleanup.append(output_dir)
            
            office_extensions = ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.odt', '.ods', '.odp']
            files_to_process = []
            
            if file_ext in office_extensions:
                pdf_path = convert_office_to_pdf(upload_path, output_dir)
                files_to_process = convert_pdf_to_images(pdf_path, output_dir)
            elif file_ext == '.pdf':
                if not PDF_SUPPORT:
                    return jsonify({'error': 'PDF 支持未启用'}), 400
                files_to_process = convert_pdf_to_images(upload_path, output_dir)
            else:
                files_to_process = [upload_path]
            
            results = []
            for idx, image_path in enumerate(files_to_process, 1):
                result = process_image(
                    image_path=image_path,
                    function_type=function_type,
                    resolution_mode=resolution_mode,
                    custom_question=custom_question,
                    save_results=save_results,
                    output_dir=output_dir,
                    history_messages=history_messages
                )
                results.append(result)
            
            final_text = "\n\n".join(results)
            
            return jsonify({
                'success': True,
                'text': final_text,
                'metadata': {
                    'filename': original_filename,
                    'function': function_type,
                    'function_name': OCR_FUNCTIONS[function_type]['name'],
                    'resolution': resolution_mode,
                    'is_pdf': file_ext == '.pdf',
                    'pages': len(files_to_process)
                }
            })
        finally:
            if dirs_to_cleanup:
                cleanup_files([], dirs_to_cleanup)
    
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@ocr_bp.route('/process/stream', methods=['POST'])
def process_file_stream():
    """流式处理文件（SSE）"""
    config = get_config()
    
    def generate():
        dirs_to_cleanup = []
        
        try:
            function_type = request.form.get('function', 'free_ocr')
            resolution_mode = request.form.get('resolution', 'gundam')
            custom_question = request.form.get('question', '')
            uploaded_filename = request.form.get('uploaded_filename', '')
            
            history_messages = []
            messages_json = request.form.get('messages', '')
            if messages_json:
                try:
                    history_messages = json.loads(messages_json)
                except:
                    pass
            
            yield f"data: {json.dumps({'type': 'start', 'message': '开始处理...'})}\n\n"
            
            # 处理文件
            if uploaded_filename:
                upload_path = os.path.join(config['upload_folder'], uploaded_filename)
                if not os.path.exists(upload_path):
                    yield f"data: {json.dumps({'error': '文件不存在'})}\n\n"
                    return
                filename = uploaded_filename
                parts = filename.split('_', 2)
                original_filename = parts[2] if len(parts) >= 3 else filename
            else:
                if 'file' not in request.files:
                    yield f"data: {json.dumps({'error': '未上传文件'})}\n\n"
                    return
                file = request.files['file']
                if file.filename == '':
                    yield f"data: {json.dumps({'error': '文件名为空'})}\n\n"
                    return
                original_filename = file.filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{file.filename}"
                upload_path = os.path.join(config['upload_folder'], filename)
                file.save(upload_path)
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': '文件已准备', 'progress': 10})}\n\n"
            
            file_ext = Path(filename).suffix.lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(config['output_folder'], timestamp)
            os.makedirs(output_dir, exist_ok=True)
            dirs_to_cleanup.append(output_dir)
            
            office_extensions = ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt']
            files_to_process = []
            
            if file_ext in office_extensions:
                yield f"data: {json.dumps({'type': 'progress', 'stage': '转换文档...', 'progress': 15})}\n\n"
                pdf_path = convert_office_to_pdf(upload_path, output_dir)
                files_to_process = convert_pdf_to_images(pdf_path, output_dir)
            elif file_ext == '.pdf':
                yield f"data: {json.dumps({'type': 'progress', 'stage': '转换PDF...', 'progress': 20})}\n\n"
                files_to_process = convert_pdf_to_images(upload_path, output_dir)
            else:
                files_to_process = [upload_path]
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': '加载模型...', 'progress': 40})}\n\n"
            load_model()
            
            all_results = []
            stop_flag = {'stop': False}
            
            for idx, image_path in enumerate(files_to_process, 1):
                if stop_flag['stop']:
                    break
                
                progress = 40 + int((idx / len(files_to_process)) * 50)
                yield f"data: {json.dumps({'type': 'progress', 'stage': f'处理第{idx}/{len(files_to_process)}页...', 'progress': progress})}\n\n"
                
                text_queue = queue.Queue()
                result_container = {'result': None, 'error': None}
                
                def stream_callback(text_chunk):
                    if not stop_flag['stop']:
                        text_queue.put(('content', text_chunk))
                
                def run_inference():
                    try:
                        result = process_image(
                            image_path=image_path,
                            function_type=function_type,
                            resolution_mode=resolution_mode,
                            custom_question=custom_question,
                            save_results=False,
                            output_dir=output_dir,
                            stream_callback=stream_callback,
                            history_messages=history_messages,
                            stop_flag=stop_flag
                        )
                        if not stop_flag['stop']:
                            result_container['result'] = result
                            text_queue.put(('done', None))
                    except Exception as e:
                        if not stop_flag['stop']:
                            result_container['error'] = str(e)
                            text_queue.put(('error', str(e)))
                
                inference_thread = threading.Thread(target=run_inference, daemon=True)
                inference_thread.start()
                
                while True:
                    try:
                        msg_type, data = text_queue.get(timeout=0.1)
                        if msg_type == 'content':
                            yield f"data: {json.dumps({'type': 'content', 'text': data})}\n\n"
                        elif msg_type == 'done':
                            break
                        elif msg_type == 'error':
                            yield f"data: {json.dumps({'type': 'error', 'error': data})}\n\n"
                            break
                    except queue.Empty:
                        if not inference_thread.is_alive() and text_queue.empty():
                            break
                
                inference_thread.join(timeout=1.0)
                if result_container['result']:
                    all_results.append(result_container['result'])
                
                if len(files_to_process) > 1 and idx < len(files_to_process):
                    separator = '\n\n'
                    yield f"data: {json.dumps({'type': 'content', 'text': separator})}\n\n"
            
            if not stop_flag['stop']:
                metadata = {
                    'filename': original_filename,
                    'function': function_type,
                    'function_name': OCR_FUNCTIONS[function_type]['name'],
                    'resolution': resolution_mode,
                    'pages': len(files_to_process)
                }
                yield f"data: {json.dumps({'type': 'done', 'metadata': metadata})}\n\n"
        
        except Exception as e:
            logger.error(f"流式处理失败: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            if dirs_to_cleanup:
                cleanup_files([], dirs_to_cleanup)
    
    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

# =============================================================================
# 模块初始化
# =============================================================================

def init_ocr_service(preload_model: bool = False):
    """初始化 OCR 服务"""
    logger.info("初始化 OCR 服务模块...")
    
    # 确保目录存在
    config = get_config()
    os.makedirs(config['upload_folder'], exist_ok=True)
    os.makedirs(config['output_folder'], exist_ok=True)
    
    if preload_model:
        try:
            load_model()
            logger.info("✓ OCR 模型预加载完成")
        except Exception as e:
            logger.warning(f"OCR 模型预加载失败: {e}")
