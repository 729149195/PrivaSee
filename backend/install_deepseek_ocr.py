#!/usr/bin/env python3
"""
DeepSeek-OCR 自动安装脚本
基于官方教程: https://huggingface.co/deepseek-ai/DeepSeek-OCR
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, check=True, shell=True):
    """运行 shell 命令"""
    print(f"\n运行: {cmd}")
    result = subprocess.run(cmd, shell=shell, check=False)
    if check and result.returncode != 0:
        print(f"错误: 命令执行失败 (退出码: {result.returncode})")
        return False
    return True

def main():
    print("=" * 60)
    print("  DeepSeek-OCR 全新安装")
    print("  基于官方教程: https://huggingface.co/deepseek-ai/DeepSeek-OCR")
    print("=" * 60)
    print()

    # 获取项目路径
    project_root = Path("/home/zhangxiangxuan/桌面/Projects/PrivaSee")
    model_path = project_root / "models" / "deepseek-ocr"
    
    # 步骤 1: 删除旧模型
    if model_path.exists():
        print("\n步骤 1: 删除旧的模型文件...")
        try:
            shutil.rmtree(model_path)
            print("✓ 已删除旧模型")
        except Exception as e:
            print(f"警告: 删除旧模型失败: {e}")
    else:
        print("\n步骤 1: 未发现旧模型文件")

    # 步骤 2: 清理缓存
    print("\n步骤 2: 清理 Hugging Face 缓存...")
    cache_path = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    deepseek_caches = list(cache_path.glob("deepseek*"))
    for cache in deepseek_caches:
        try:
            shutil.rmtree(cache)
            print(f"✓ 已删除缓存: {cache.name}")
        except Exception as e:
            print(f"警告: 删除缓存失败: {e}")

    # 步骤 3: 检查并创建 conda 环境
    print("\n步骤 3: 设置 conda 环境...")
    print("请在另一个终端手动执行以下命令:")
    print()
    print("  # 删除旧环境（如果存在）")
    print("  conda remove -n deepseek-ocr --all -y")
    print()
    print("  # 创建新环境")
    print("  conda create -n deepseek-ocr python=3.12 -y")
    print()
    print("  # 激活环境")
    print("  conda activate deepseek-ocr")
    print()
    
    input("完成后按 Enter 继续...")

    # 步骤 4: 安装依赖包
    print("\n步骤 4: 安装依赖包...")
    print("请在激活的 deepseek-ocr 环境中执行以下命令:")
    print()
    print("  # 安装 PyTorch (CUDA 11.8)")
    print("  pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("  # 安装 transformers 和相关包")
    print("  pip install transformers==4.46.3 tokenizers==0.20.3 einops addict easydict")
    print()
    print("  # 安装 Flash Attention 2（可选）")
    print("  pip install flash-attn==2.7.3 --no-build-isolation")
    print()
    print("  # 安装服务器依赖")
    print("  pip install flask flask-cors pillow pdf2image numpy")
    print()
    
    input("完成后按 Enter 继续...")

    # 步骤 5: 下载模型
    print("\n步骤 5: 下载模型...")
    print("正在创建模型下载脚本...")
    
    download_script = project_root / "backend" / "download_model.py"
    download_script.write_text("""
from transformers import AutoModel, AutoTokenizer
import os

model_name = 'deepseek-ai/DeepSeek-OCR'
save_path = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr'

print('正在下载 tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

print('正在下载模型...')
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True
)
model.save_pretrained(save_path)

print('✓ 模型下载完成！')
""")
    
    print(f"✓ 已创建下载脚本: {download_script}")
    print()
    print("请在激活的 deepseek-ocr 环境中运行:")
    print(f"  python {download_script}")
    print()
    print("注意: 模型下载可能需要较长时间，请耐心等待。")
    print()
    
    input("完成后按 Enter 继续...")

    # 步骤 6: 验证安装
    print("\n步骤 6: 验证安装...")
    verify_script = project_root / "backend" / "verify_installation.py"
    verify_script.write_text("""
from transformers import AutoModel, AutoTokenizer
import torch

print('测试环境...')
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

print('\\n加载模型...')
model_path = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr'

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print('✓ Tokenizer 加载成功')
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    print('✓ 模型加载成功')
    
    print('\\n✓✓✓ 环境测试通过！')
    print('\\n可以运行以下命令启动服务:')
    print('  bash start_deepseek_ocr_new.sh')
except Exception as e:
    print(f'\\n✗ 测试失败: {e}')
""")
    
    print(f"✓ 已创建验证脚本: {verify_script}")
    print()
    print("请在激活的 deepseek-ocr 环境中运行:")
    print(f"  python {verify_script}")
    print()

    print("\n" + "=" * 60)
    print("  安装向导完成！")
    print("=" * 60)
    print()
    print("后续步骤:")
    print("1. 运行验证脚本确认安装成功")
    print("2. 使用 start_deepseek_ocr_new.sh 启动服务")
    print()
    print("详细说明请查看: INSTALL_DEEPSEEK_OCR.md")
    print()

if __name__ == "__main__":
    main()

