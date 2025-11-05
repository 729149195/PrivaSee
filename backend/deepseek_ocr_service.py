#!/usr/bin/env python3
"""
DeepSeek-OCR 本地部署服务
支持图片和 PDF 文档处理，提供多种 OCR 功能
"""

import os
import sys
import logging
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, TextStreamer
import queue
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deepseek_ocr_service.log')
    ]
)
logger = logging.getLogger(__name__)

# PDF 支持
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
    logger.info("✓ PDF 支持已启用")
except ImportError:
    PDF_SUPPORT = False
    logger.warning("✗ PDF 支持未启用 (需要安装 pdf2image 和 poppler-utils)")

# Flask 应用
app = Flask(__name__)
# 设置最大文件上传大小为 50MB（40MB + 一些余量）
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# =============================================================================
# 配置区域
# =============================================================================

# 模型路径配置
MODEL_PATH = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr'

# 文件存储路径
UPLOAD_FOLDER = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/data/uploads/ocr'
OUTPUT_FOLDER = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/data/outputs/ocr'

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 全局模型变量
ocr_model = None
ocr_tokenizer = None

# =============================================================================
# OCR 功能模板定义（对应图片中的功能列表）
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

# =============================================================================
# 分辨率模式配置
# =============================================================================

RESOLUTION_MODES = {
    'tiny': {
        'base_size': 512,
        'image_size': 512,
        'crop_mode': False,
        'description': '快速预览（512px）'
    },
    'small': {
        'base_size': 640,
        'image_size': 640,
        'crop_mode': False,
        'description': '标准处理（640px）'
    },
    'base': {
        'base_size': 1024,
        'image_size': 1024,
        'crop_mode': False,
        'description': '高质量（1024px）'
    },
    'large': {
        'base_size': 1280,
        'image_size': 1280,
        'crop_mode': False,
        'description': '超高质量（1280px）'
    },
    'gundam': {
        'base_size': 1024,
        'image_size': 640,
        'crop_mode': True,
        'description': '推荐模式（智能裁剪）'
    }
}

# =============================================================================
# 模型加载与管理
# =============================================================================

def load_model():
    """加载 DeepSeek-OCR 模型"""
    global ocr_model, ocr_tokenizer
    
    if ocr_model is not None and ocr_tokenizer is not None:
        return ocr_model, ocr_tokenizer
    
    logger.info("=" * 70)
    logger.info("开始加载 DeepSeek-OCR 模型...")
    logger.info(f"模型路径: {MODEL_PATH}")
    
    try:
        # 检查设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device.upper()}")
        
        if device == 'cpu':
            logger.warning("⚠️  未检测到 GPU，将使用 CPU 运行（速度较慢）")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU 设备: {gpu_name}")
        
        # 加载 tokenizer
        logger.info("加载 Tokenizer...")
        ocr_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # 设置 pad_token 以消除警告
        if ocr_tokenizer.pad_token is None:
            ocr_tokenizer.pad_token = ocr_tokenizer.eos_token
            logger.info("✓ 已设置 pad_token = eos_token")
        
        logger.info("✓ Tokenizer 加载完成")
        
        # 加载模型
        logger.info("加载模型权重...")
        if device == 'cuda':
            try:
                # 尝试使用 Flash Attention 2
                logger.info("尝试启用 Flash Attention 2...")
                ocr_model = AutoModel.from_pretrained(
                    MODEL_PATH,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True,
                    torch_dtype=torch.bfloat16
                )
                logger.info("✓ 已启用 Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention 2 不可用: {e}")
                logger.info("降级到标准 Attention...")
                ocr_model = AutoModel.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    use_safetensors=True,
                    torch_dtype=torch.bfloat16
                )
                logger.info("✓ 使用标准 Attention")
            
            ocr_model = ocr_model.eval().cuda()
        else:
            ocr_model = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                use_safetensors=True
            )
            ocr_model = ocr_model.eval()
        
        logger.info("✓ 模型加载完成")
        logger.info("=" * 70)
        
        return ocr_model, ocr_tokenizer
        
    except Exception as e:
        logger.error(f"✗ 模型加载失败: {e}", exc_info=True)
        raise

# =============================================================================
# 文档转换工具
# =============================================================================

def convert_office_to_pdf(input_path: str, output_dir: str) -> str:
    """
    将 Office 文档（Word, Excel, PowerPoint）转换为 PDF
    使用 LibreOffice 命令行工具
    
    Args:
        input_path: 输入文件路径 (.docx, .xlsx, .pptx 等)
        output_dir: 输出目录
        
    Returns:
        转换后的 PDF 文件路径
    """
    import subprocess
    
    logger.info(f"转换 Office 文档为 PDF: {input_path}")
    
    try:
        # 检查 LibreOffice 是否安装
        try:
            subprocess.run(['libreoffice', '--version'], 
                         capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError(
                "LibreOffice 未安装或不可用。\n"
                "安装方法: sudo apt-get install libreoffice\n"
                "或 macOS: brew install libreoffice"
            )
        
        # 使用 LibreOffice 转换为 PDF
        cmd = [
            'libreoffice',
            '--headless',  # 无界面模式
            '--convert-to', 'pdf',
            '--outdir', output_dir,
            input_path
        ]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60秒超时
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice 转换失败: {result.stderr}")
        
        # 查找生成的 PDF 文件
        input_stem = Path(input_path).stem
        pdf_path = os.path.join(output_dir, f"{input_stem}.pdf")
        
        if not os.path.exists(pdf_path):
            raise RuntimeError(f"PDF 文件未生成: {pdf_path}")
        
        logger.info(f"✓ Office 文档转换完成: {pdf_path}")
        return pdf_path
        
    except subprocess.TimeoutExpired:
        logger.error("LibreOffice 转换超时")
        raise RuntimeError("文档转换超时（60秒），文件可能太大或格式复杂")
    except Exception as e:
        logger.error(f"Office 文档转换失败: {e}")
        raise

def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> List[str]:
    """
    将 PDF 转换为图片列表
    
    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        dpi: 分辨率
        
    Returns:
        图片路径列表
    """
    if not PDF_SUPPORT:
        raise RuntimeError("PDF 支持未启用，请安装: sudo apt-get install poppler-utils && pip install pdf2image")
    
    logger.info(f"转换 PDF: {pdf_path}")
    logger.info(f"DPI: {dpi}")
    
    try:
        # 转换 PDF 为图片
        images = convert_from_path(pdf_path, dpi=dpi)
        
        image_paths = []
        pdf_name = Path(pdf_path).stem
        
        for i, image in enumerate(images, 1):
            image_path = os.path.join(output_dir, f"{pdf_name}_page_{i:03d}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
            logger.info(f"  → 第 {i} 页: {image_path}")
        
        logger.info(f"✓ PDF 转换完成，共 {len(image_paths)} 页")
        return image_paths
        
    except Exception as e:
        logger.error(f"PDF 转换失败: {e}")
        raise

# =============================================================================
# 自定义流式 Streamer
# =============================================================================

class CallbackTextStreamer(TextStreamer):
    """
    自定义 TextStreamer，通过回调函数实时发送生成的文本
    """
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
# 文件清理工具
# =============================================================================

def cleanup_files(files_to_delete: List[str], dirs_to_delete: List[str] = None):
    """
    清理文件和目录
    
    Args:
        files_to_delete: 要删除的文件列表
        dirs_to_delete: 要删除的目录列表
    """
    import shutil
    
    # 清理文件
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"  ✓ 已删除文件: {Path(file_path).name}")
        except Exception as e:
            logger.warning(f"  ✗ 删除文件失败 {file_path}: {e}")
    
    # 清理目录
    if dirs_to_delete:
        for dir_path in dirs_to_delete:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.info(f"  ✓ 已删除目录: {Path(dir_path).name}")
            except Exception as e:
                logger.warning(f"  ✗ 删除目录失败 {dir_path}: {e}")

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
    stream_callback=None,  # 新增：流式生成回调函数
    history_messages: Optional[list] = None,  # 新增：历史消息列表
    stop_flag: Optional[dict] = None  # 新增：停止标志
) -> str:
    """
    处理单张图片
    
    Args:
        image_path: 图片路径
        function_type: 功能类型
        resolution_mode: 分辨率模式
        custom_question: 自定义问题（用于视觉问答）
        save_results: 是否保存结果
        output_dir: 输出目录
        stream_callback: 流式生成回调函数
        history_messages: 历史消息列表 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
    Returns:
        OCR 结果文本
    """
    # 加载模型
    model, tokenizer = load_model()
    
    # 获取功能配置
    if function_type not in OCR_FUNCTIONS:
        raise ValueError(f"不支持的功能类型: {function_type}")
    
    function_config = OCR_FUNCTIONS[function_type]
    
    # 构建 prompt
    prompt_template = function_config['prompt_template']
    
    # 如果有历史消息，将其作为上下文添加到 prompt 中
    if history_messages and len(history_messages) > 0:
        logger.info(f"  包含历史上下文，共 {len(history_messages)} 条消息")
        
        # 构建对话历史
        context_parts = ["以下是之前的对话记录："]
        for msg in history_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '').strip()
            if content:
                if role == 'user':
                    context_parts.append(f"Q: {content}")
                elif role == 'assistant':
                    # 限制助手回复长度，避免上下文过长
                    if len(content) > 500:
                        content = content[:500] + "..."
                    context_parts.append(f"A: {content}")
        
        context_text = '\n'.join(context_parts)
        
        # 构建当前问题
        if custom_question:
            current_question = custom_question
        else:
            current_question = "请继续回答"
        
        # 对于视觉问答，使用对话格式
        if function_config['supports_custom_question']:
            prompt = f"{context_text}\n\n当前问题：\nQ: {current_question}\nA: "
        else:
            # 对于其他 OCR 功能，保持原有格式
            prompt = prompt_template
    else:
        # 没有历史消息，使用原始 prompt
        if function_config['supports_custom_question'] and custom_question:
            prompt = prompt_template.format(question=custom_question)
        else:
            prompt = prompt_template
    
    # 获取分辨率配置
    if resolution_mode not in RESOLUTION_MODES:
        raise ValueError(f"不支持的分辨率模式: {resolution_mode}")
    
    res_config = RESOLUTION_MODES[resolution_mode]
    
    # 准备输出目录
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_FOLDER, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行推理
    logger.info(f"处理图片: {Path(image_path).name}")
    logger.info(f"  功能: {function_config['name']}")
    logger.info(f"  分辨率: {resolution_mode} ({res_config['description']})")
    
    # 如果提供了流式回调，使用自定义 streamer
    custom_streamer = None
    if stream_callback:
        logger.info(f"  使用流式生成模式")
        custom_streamer = CallbackTextStreamer(
            tokenizer, 
            callback=stream_callback,
            skip_prompt=True,
            skip_special_tokens=False
        )
    
    # 使用 eval_mode=True 确保返回结果
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
        eval_mode=True,  # 重要：确保返回文本结果
        custom_streamer=custom_streamer  # 传递自定义 streamer
    )
    
    # 确保结果不为空
    if result is None:
        result = ""
        logger.warning("  ⚠ 模型返回 None")
    
    # 检测并处理重复内容（特别是视觉问答可能出现的问题）
    if result and len(result) > 1000:
        words = result.split()
        unique_words = set(words)
        if len(unique_words) < len(words) * 0.15:  # 如果唯一词少于15%
            logger.warning(f"  ⚠ 检测到大量重复内容 - 唯一词: {len(unique_words)}, 总词数: {len(words)}")
            # 截取前500字符作为结果
            result = result[:500] + "\n\n[检测到模型生成重复内容，已自动截断。建议：1) 换一张图片测试 2) 换一个提问方式 3) 使用其他OCR功能]"
    
    # 如果结果为空字符串，记录警告
    if not result or len(result.strip()) == 0:
        logger.warning(f"  ⚠ 模型返回空结果 - 功能: {function_config['name']}, 图片: {Path(image_path).name}")
        if function_type == 'visual_qa' and not custom_question:
            logger.warning("  ⚠ 视觉问答可能缺少问题参数")
            result = "视觉问答需要提供问题。请在发送时输入您的问题。"
    
    # 检查是否已被中断
    if stop_flag and stop_flag.get('stop'):
        logger.info("  处理被中断，不发送完成事件")
    else:
        logger.info(f"  ✓ 处理完成，结果长度: {len(result)} 字符")
    
    # 清理 GPU 缓存，释放显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("  已清理 GPU 缓存")
    
    return result

# =============================================================================
# API 路由
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
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
            'model_path': MODEL_PATH
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/functions', methods=['GET'])
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

@app.route('/api/resolutions', methods=['GET'])
def list_resolutions():
    """列出所有分辨率模式"""
    return jsonify({
        'modes': [
            {
                'id': key,
                'description': value['description'],
                'config': {
                    'base_size': value['base_size'],
                    'image_size': value['image_size'],
                    'crop_mode': value['crop_mode']
                }
            }
            for key, value in RESOLUTION_MODES.items()
        ],
        'recommended': 'gundam'
    })

@app.route('/api/process/stream', methods=['POST'])
def process_file_stream():
    """
    流式处理文件（SSE）
    
    返回 Server-Sent Events 格式的流式数据
    """
    import time
    import json
    
    def generate():
        # 用于追踪需要清理的输出目录（只清理outputs，不清理uploads）
        dirs_to_cleanup = []
        
        try:
            # 获取参数
            function_type = request.form.get('function', 'free_ocr')
            resolution_mode = request.form.get('resolution', 'gundam')
            custom_question = request.form.get('question', '')
            uploaded_filename = request.form.get('uploaded_filename', '')
            
            # 获取历史消息（JSON格式）
            history_messages = []
            messages_json = request.form.get('messages', '')
            if messages_json:
                try:
                    history_messages = json.loads(messages_json)
                    logger.info(f"  接收到历史消息: {len(history_messages)} 条")
                except json.JSONDecodeError as e:
                    logger.warning(f"  历史消息JSON解析失败: {e}")
                    history_messages = []
            
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'message': '开始处理...'})}\n\n"
            
            # 判断是使用已上传的文件还是新上传的文件
            if uploaded_filename:
                # 使用已上传的文件
                upload_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)
                if not os.path.exists(upload_path):
                    yield f"data: {json.dumps({'error': '已上传的文件不存在'})}\n\n"
                    return
                filename = uploaded_filename
                # 从文件名中提取原始文件名（去掉时间戳前缀）
                # 格式：YYYYMMDD_HHMMSS_原始文件名
                parts = filename.split('_', 2)  # 最多分割成3部分
                if len(parts) >= 3:
                    original_filename = parts[2]  # 原始文件名
                else:
                    original_filename = filename
                yield f"data: {json.dumps({'type': 'progress', 'stage': '使用已上传文件', 'progress': 10})}\n\n"
            else:
                # 新上传文件
                if 'file' not in request.files:
                    yield f"data: {json.dumps({'error': '未上传文件'})}\n\n"
                    return
                
                file = request.files['file']
                if file.filename == '':
                    yield f"data: {json.dumps({'error': '文件名为空'})}\n\n"
                    return
                
                original_filename = file.filename
                # 保存文件
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{file.filename}"
                upload_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(upload_path)
                
                yield f"data: {json.dumps({'type': 'progress', 'stage': '文件已上传', 'progress': 10})}\n\n"
            
            # 准备处理
            file_ext = Path(filename).suffix.lower()
            # 如果使用已上传文件，从文件名中提取 timestamp；否则使用新的 timestamp
            if uploaded_filename:
                # 尝试从文件名提取时间戳，格式：YYYYMMDD_HHMMSS_...
                parts = filename.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
                    timestamp = parts[0]
                else:
                    # 如果格式不匹配，使用当前时间戳
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(OUTPUT_FOLDER, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            # 标记输出目录需要清理
            dirs_to_cleanup.append(output_dir)
            
            files_to_process = []
            
            # Office 文档格式列表
            office_extensions = ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.odt', '.ods', '.odp']
            
            # 处理 Office 文档：先转换为 PDF
            if file_ext in office_extensions:
                yield f"data: {json.dumps({'type': 'progress', 'stage': f'转换 {file_ext.upper()} 为 PDF...', 'progress': 15})}\n\n"
                try:
                    pdf_path = convert_office_to_pdf(upload_path, output_dir)
                    yield f"data: {json.dumps({'type': 'progress', 'stage': 'Office 文档已转换为 PDF', 'progress': 20})}\n\n"
                    
                    # 转换 PDF 为图片
                    if not PDF_SUPPORT:
                        yield f"data: {json.dumps({'error': 'PDF支持未启用，无法继续处理'})}\n\n"
                        return
                    files_to_process = convert_pdf_to_images(pdf_path, output_dir)
                    yield f"data: {json.dumps({'type': 'progress', 'stage': f'PDF已转换({len(files_to_process)}页)', 'progress': 30})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': f'Office文档转换失败: {str(e)}'})}\n\n"
                    return
            
            # PDF转换
            elif file_ext == '.pdf':
                yield f"data: {json.dumps({'type': 'progress', 'stage': '转换PDF...', 'progress': 20})}\n\n"
                if not PDF_SUPPORT:
                    yield f"data: {json.dumps({'error': 'PDF支持未启用'})}\n\n"
                    return
                files_to_process = convert_pdf_to_images(upload_path, output_dir)
                yield f"data: {json.dumps({'type': 'progress', 'stage': f'PDF已转换({len(files_to_process)}页)', 'progress': 30})}\n\n"
            else:
                # 图片文件直接处理
                files_to_process = [upload_path]
            
            # 加载模型
            yield f"data: {json.dumps({'type': 'progress', 'stage': '加载模型...', 'progress': 40})}\n\n"
            model, tokenizer = load_model()
            
            # 处理文件
            all_results = []
            stop_flag = {'stop': False}  # 共享停止标志
            
            for idx, image_path in enumerate(files_to_process, 1):
                # 检查是否需要停止
                if stop_flag['stop']:
                    logger.info("检测到客户端断开，停止处理")
                    break
                    
                progress = 40 + int((idx / len(files_to_process)) * 50)
                try:
                    yield f"data: {json.dumps({'type': 'progress', 'stage': f'处理第{idx}/{len(files_to_process)}页...', 'progress': progress})}\n\n"
                except (GeneratorExit, StopIteration, BrokenPipeError, ConnectionError):
                    logger.info("客户端连接已断开，停止处理")
                    stop_flag['stop'] = True
                    break
                
                # 使用队列实现真正的实时流式传输
                text_queue = queue.Queue()
                result_container = {'result': None, 'error': None}
                
                def stream_callback(text_chunk):
                    """实时发送文本块到队列"""
                    if not stop_flag['stop']:
                        text_queue.put(('content', text_chunk))
                
                def run_inference():
                    """在独立线程中运行推理"""
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
                            stop_flag=stop_flag  # 传递停止标志
                        )
                        # 只有在未被中断时才发送完成信号
                        if not stop_flag['stop']:
                            result_container['result'] = result
                            text_queue.put(('done', None))
                    except Exception as e:
                        if not stop_flag['stop']:
                            result_container['error'] = str(e)
                            text_queue.put(('error', str(e)))
                
                # 启动推理线程
                inference_thread = threading.Thread(target=run_inference, daemon=True)
                inference_thread.start()
                
                # 主线程持续从队列读取并发送数据
                client_disconnected = False
                while True:
                    try:
                        msg_type, data = text_queue.get(timeout=0.1)
                        
                        if msg_type == 'content':
                            # 实时发送文本块
                            try:
                                yield f"data: {json.dumps({'type': 'content', 'text': data})}\n\n"
                            except (GeneratorExit, StopIteration, BrokenPipeError, ConnectionError) as e:
                                # 客户端断开连接
                                logger.info(f"检测到客户端断开连接: {type(e).__name__}")
                                stop_flag['stop'] = True
                                client_disconnected = True
                                break
                        elif msg_type == 'done':
                            # 推理完成
                            break
                        elif msg_type == 'error':
                            # 发生错误
                            yield f"data: {json.dumps({'type': 'error', 'error': data})}\n\n"
                            break
                    except queue.Empty:
                        # 队列为空，继续等待
                        # 检查线程是否还活着
                        if not inference_thread.is_alive() and text_queue.empty():
                            break
                
                if client_disconnected:
                    logger.info("客户端已断开，停止所有处理")
                    break
                
                # 等待线程结束（设置超时避免无限等待）
                inference_thread.join(timeout=1.0)
                
                # 记录完整结果
                if result_container['result']:
                    all_results.append(result_container['result'])
                
                # 在多页文档中，每页之间添加分隔符
                if len(files_to_process) > 1 and idx < len(files_to_process):
                    separator = '\n\n'
                    yield f"data: {json.dumps({'type': 'content', 'text': separator})}\n\n"
            
            # 发送完成事件（仅当没有被中断时）
            if not stop_flag['stop']:
                final_text = "\n\n".join(all_results)
                metadata = {
                    'filename': original_filename,
                    'function': function_type,
                    'function_name': OCR_FUNCTIONS[function_type]['name'],
                    'resolution': resolution_mode,
                    'is_pdf': file_ext == '.pdf',
                    'is_office': file_ext in office_extensions,
                    'pages': len(files_to_process),
                    'timestamp': timestamp
                }
                
                yield f"data: {json.dumps({'type': 'done', 'metadata': metadata})}\n\n"
            else:
                logger.info("处理被中断，不发送完成事件")
            
        except Exception as e:
            logger.error(f"流式处理失败: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # 清理输出目录（保留上传的文件）
            if dirs_to_cleanup:
                logger.info("开始清理输出文件...")
                cleanup_files([], dirs_to_cleanup)
                logger.info("✓ 输出文件清理完成")
    
    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

@app.route('/api/process', methods=['POST'])
def process_file():
    """
    处理文件（图片或 PDF）
    
    请求参数:
        - file: 文件（multipart/form-data）
        - function: 功能类型 (free_ocr, markdown, table_extract 等)
        - resolution: 分辨率模式 (tiny, small, base, large, gundam)
        - question: 自定义问题（用于 visual_qa）
        - save_results: 是否保存结果文件 (true/false)
    
    返回:
        {
            "success": true,
            "text": "识别结果",
            "metadata": {...}
        }
    """
    # 用于追踪需要清理的输出目录（只清理outputs，不清理uploads）
    dirs_to_cleanup = []
    
    try:
        # 获取参数
        function_type = request.form.get('function', 'free_ocr')
        resolution_mode = request.form.get('resolution', 'gundam')
        custom_question = request.form.get('question', '')
        save_results = request.form.get('save_results', 'false').lower() == 'true'
        uploaded_filename = request.form.get('uploaded_filename', '')
        
        # 获取历史消息（JSON格式）
        history_messages = []
        messages_json = request.form.get('messages', '')
        if messages_json:
            try:
                import json
                history_messages = json.loads(messages_json)
                logger.info(f"  接收到历史消息: {len(history_messages)} 条")
            except json.JSONDecodeError as e:
                logger.warning(f"  历史消息JSON解析失败: {e}")
                history_messages = []
        
        # 判断是使用已上传的文件还是新上传的文件
        if uploaded_filename:
            # 使用已上传的文件
            upload_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)
            if not os.path.exists(upload_path):
                return jsonify({
                    'success': False,
                    'error': '已上传的文件不存在'
                }), 400
            filename = uploaded_filename
            # 从文件名中提取原始文件名（去掉时间戳前缀）
            # 格式：YYYYMMDD_HHMMSS_原始文件名
            parts = filename.split('_', 2)  # 最多分割成3部分
            if len(parts) >= 3:
                original_filename = parts[2]  # 原始文件名
            else:
                original_filename = filename
            logger.info(f"使用已上传文件: {filename}")
        else:
            # 新上传文件
            if 'file' not in request.files:
                return jsonify({'error': '未上传文件'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '文件名为空'}), 400
            
            original_filename = file.filename
            # 保存上传的文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)
            
            logger.info(f"新上传文件: {filename}")
        
        # 验证参数
        if function_type not in OCR_FUNCTIONS:
            return jsonify({'error': f'不支持的功能: {function_type}'}), 400
        
        if resolution_mode not in RESOLUTION_MODES:
            return jsonify({'error': f'不支持的分辨率模式: {resolution_mode}'}), 400
        
        logger.info("=" * 70)
        logger.info(f"收到处理请求: {filename}")
        logger.info(f"功能: {OCR_FUNCTIONS[function_type]['name']}")
        logger.info(f"分辨率: {resolution_mode}")
        
        try:
            # 判断文件类型
            file_ext = Path(filename).suffix.lower()
            # 如果使用已上传文件，从文件名中提取 timestamp；否则使用新的 timestamp
            if uploaded_filename:
                # 尝试从文件名提取时间戳，格式：YYYYMMDD_HHMMSS_...
                parts = filename.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
                    timestamp = parts[0]
                else:
                    # 如果格式不匹配，使用当前时间戳
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(OUTPUT_FOLDER, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            # 标记输出目录需要清理
            dirs_to_cleanup.append(output_dir)
            
            files_to_process = []
            
            # Office 文档格式列表
            office_extensions = ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.odt', '.ods', '.odp']
            
            # 处理 Office 文档：先转换为 PDF
            if file_ext in office_extensions:
                logger.info(f"检测到 {file_ext.upper()} 文件，开始转换为 PDF...")
                try:
                    pdf_path = convert_office_to_pdf(upload_path, output_dir)
                    logger.info(f"✓ Office 文档已转换为 PDF: {pdf_path}")
                    
                    # 转换 PDF 为图片
                    if not PDF_SUPPORT:
                        return jsonify({
                            'error': 'PDF 支持未启用',
                            'message': '请安装: sudo apt-get install poppler-utils && pip install pdf2image'
                        }), 400
                    
                    logger.info("转换 PDF 为图片...")
                    files_to_process = convert_pdf_to_images(pdf_path, output_dir)
                    logger.info(f"✓ 已转换为 {len(files_to_process)} 张图片")
                except Exception as e:
                    logger.error(f"Office 文档转换失败: {e}")
                    return jsonify({
                        'success': False,
                        'error': f'Office 文档转换失败: {str(e)}'
                    }), 500
            
            # 处理 PDF
            elif file_ext == '.pdf':
                if not PDF_SUPPORT:
                    return jsonify({
                        'error': 'PDF 支持未启用',
                        'message': '请安装: sudo apt-get install poppler-utils && pip install pdf2image'
                    }), 400
                
                logger.info("检测到 PDF 文件，开始转换...")
                files_to_process = convert_pdf_to_images(upload_path, output_dir)
            else:
                # 图片文件直接处理
                files_to_process = [upload_path]
            
            # 处理所有文件
            results = []
            for idx, image_path in enumerate(files_to_process, 1):
                logger.info(f"处理 {idx}/{len(files_to_process)}...")
                
                result = process_image(
                    image_path=image_path,
                    function_type=function_type,
                    resolution_mode=resolution_mode,
                    custom_question=custom_question,
                    save_results=save_results,
                    output_dir=output_dir,
                    history_messages=history_messages
                )
                
                # 直接添加结果，不添加页面标题
                results.append(result)
            
            # 合并结果
            final_text = "\n\n".join(results)
            
            logger.info("=" * 70)
            logger.info(f"✓ 处理完成")
            
            # 返回结果
            response = {
                'success': True,
                'text': final_text,
                'metadata': {
                    'filename': original_filename,
                    'function': function_type,
                    'function_name': OCR_FUNCTIONS[function_type]['name'],
                    'resolution': resolution_mode,
                    'is_pdf': file_ext == '.pdf',
                    'is_office': file_ext in office_extensions,
                    'pages': len(files_to_process),
                    'timestamp': timestamp,
                    'output_dir': output_dir if save_results else None
                }
            }
            
            return jsonify(response)
            
        finally:
            # 清理输出目录（保留上传的文件）
            if dirs_to_cleanup:
                logger.info("开始清理输出文件...")
                cleanup_files([], dirs_to_cleanup)
                logger.info("✓ 输出文件清理完成")
    
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    仅上传文件到服务器，不进行OCR处理
    返回上传后的文件路径
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '未上传文件'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '文件名为空'
            }), 400
        
        # 生成唯一的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # 保存文件
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
        logger.error(f"文件上传失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch', methods=['POST'])
def batch_process():
    """
    批量处理多个文件
    """
    # 用于追踪需要清理的输出目录（只清理outputs，不清理uploads）
    dirs_to_cleanup = []
    
    try:
        if 'files' not in request.files:
            return jsonify({'error': '未上传文件'}), 400
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': '文件列表为空'}), 400
        
        function_type = request.form.get('function', 'free_ocr')
        resolution_mode = request.form.get('resolution', 'gundam')
        
        logger.info("=" * 70)
        logger.info(f"批量处理: {len(files)} 个文件")
        
        results = []
        for file in files:
            if file.filename == '':
                continue
            
            try:
                # 保存文件
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"{timestamp}_{file.filename}"
                upload_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(upload_path)
                
                # 处理
                output_dir = os.path.join(OUTPUT_FOLDER, timestamp)
                # 标记输出目录需要清理
                dirs_to_cleanup.append(output_dir)
                
                result_text = process_image(
                    image_path=upload_path,
                    function_type=function_type,
                    resolution_mode=resolution_mode,
                    save_results=True,
                    output_dir=output_dir
                )
                
                results.append({
                    'filename': file.filename,
                    'text': result_text,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"处理 {file.filename} 失败: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'status': 'failed'
                })
        
        success_count = len([r for r in results if r['status'] == 'success'])
        
        logger.info(f"批量处理完成: {success_count}/{len(files)} 成功")
        logger.info("=" * 70)
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(files),
            'success_count': success_count,
            'failed_count': len(files) - success_count
        })
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # 清理输出目录（保留上传的文件）
        if dirs_to_cleanup:
            logger.info("开始清理输出文件...")
            cleanup_files([], dirs_to_cleanup)
            logger.info("✓ 输出文件清理完成")

# =============================================================================
# 主程序
# =============================================================================

def main():
    """启动服务"""
    logger.info("=" * 70)
    logger.info("DeepSeek-OCR 本地部署服务")
    logger.info("=" * 70)
    
    # 预加载模型
    try:
        logger.info("预加载模型...")
        load_model()
        logger.info("✓ 模型预加载完成")
    except Exception as e:
        logger.error(f"✗ 模型预加载失败: {e}")
        logger.warning("服务将启动，但首次请求可能较慢")
    
    # 启动服务
    logger.info("=" * 70)
    logger.info("启动 Flask 服务...")
    logger.info("监听地址: http://0.0.0.0:5001")
    logger.info("API 文档: http://localhost:5001/api/health")
    logger.info("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()

