#!/usr/bin/env python3
"""
快速测试 DeepSeek-OCR 模型加载
"""

import torch
from transformers import AutoModel, AutoTokenizer
import sys

MODEL_PATH = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr'

print("=" * 60)
print("测试 DeepSeek-OCR 模型加载")
print("=" * 60)

# 检查 CUDA
print(f"\n1. CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU 数量: {torch.cuda.device_count()}")
    print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   使用设备: {device}")

# 加载 tokenizer
print("\n2. 加载 tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True
    )
    print("   ✅ Tokenizer 加载成功")
except Exception as e:
    print(f"   ❌ Tokenizer 加载失败: {e}")
    sys.exit(1)

# 加载模型
print("\n3. 加载模型...")
if device == 'cuda':
    # 尝试方法1: Flash Attention 2
    print("   尝试方法1: Flash Attention 2...")
    try:
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )
        print("   ✅ 模型加载成功 (Flash Attention 2)")
        model = model.eval().cuda().to(torch.bfloat16)
    except Exception as e:
        print(f"   ⚠️  Flash Attention 2 失败: {str(e)[:100]}...")
        
        # 尝试方法2: 标准 attention
        print("   尝试方法2: 标准 attention...")
        try:
            model = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16
            )
            print("   ✅ 模型加载成功 (标准 attention)")
            model = model.eval().cuda()
        except Exception as e2:
            print(f"   ❌ 标准 attention 也失败: {e2}")
            sys.exit(1)
else:
    try:
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_safetensors=True
        )
        print("   ✅ 模型加载成功 (CPU)")
        model = model.eval()
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        sys.exit(1)

print("\n4. 模型信息:")
print(f"   模型类型: {type(model).__name__}")
print(f"   设备: {next(model.parameters()).device}")
print(f"   数据类型: {next(model.parameters()).dtype}")

print("\n" + "=" * 60)
print("✅ 所有测试通过！模型可以正常使用。")
print("=" * 60)

