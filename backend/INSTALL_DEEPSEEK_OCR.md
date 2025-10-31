# DeepSeek-OCR 全新安装指南

基于官方教程: https://huggingface.co/deepseek-ai/DeepSeek-OCR

## 步骤 1: 设置脚本权限

```bash
cd /home/zhangxiangxuan/桌面/Projects/PrivaSee/backend
chmod +x setup_deepseek_ocr_fresh.sh
chmod +x start_deepseek_ocr_new.sh
```

## 步骤 2: 运行安装脚本

这将自动完成以下操作：
- 删除旧的 deepseek-ocr 环境（如果存在）
- 创建新的 conda 环境（Python 3.12）
- 安装所有依赖包
- 从 Hugging Face 下载模型

```bash
bash setup_deepseek_ocr_fresh.sh
```

**注意**: 模型下载可能需要较长时间（取决于网络速度），请耐心等待。

## 步骤 3: 启动服务

安装完成后，使用新的启动脚本：

```bash
bash start_deepseek_ocr_new.sh
```

## 手动安装步骤（如果脚本失败）

### 1. 创建 conda 环境

```bash
# 删除旧环境（如果存在）
conda remove -n deepseek-ocr --all -y

# 创建新环境
conda create -n deepseek-ocr python=3.12 -y

# 激活环境
conda activate deepseek-ocr
```

### 2. 安装依赖包

```bash
# 安装 PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 transformers 和相关包
pip install transformers==4.46.3
pip install tokenizers==0.20.3
pip install einops
pip install addict
pip install easydict

# 安装 Flash Attention 2（可选，如果失败会自动使用标准 attention）
pip install flash-attn==2.7.3 --no-build-isolation

# 安装服务器依赖
pip install flask flask-cors
pip install pillow
pip install pdf2image
pip install numpy
```

### 3. 下载模型

```bash
cd /home/zhangxiangxuan/桌面/Projects/PrivaSee/models

python << EOF
from transformers import AutoModel, AutoTokenizer

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
EOF
```

### 4. 测试安装

```bash
conda activate deepseek-ocr

python -c "
from transformers import AutoModel, AutoTokenizer
import torch

print('测试环境...')
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')

print('加载模型...')
model_path = '/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

print('✓ 环境测试通过！')
"
```

### 5. 启动服务

```bash
cd /home/zhangxiangxuan/桌面/Projects/PrivaSee/backend
conda activate deepseek-ocr
python deepseek_ocr_server.py
```

## 官方推荐配置

根据官方文档，不同的图像分辨率设置：

- **Tiny**: `base_size = 512, image_size = 512, crop_mode = False`
- **Small**: `base_size = 640, image_size = 640, crop_mode = False`
- **Base**: `base_size = 1024, image_size = 1024, crop_mode = False`
- **Large**: `base_size = 1280, image_size = 1280, crop_mode = False`
- **Gundam** (推荐): `base_size = 1024, image_size = 640, crop_mode = True`

## 验证安装

服务启动后，你应该看到：

```
====================================
  启动 DeepSeek-OCR 服务
  端口: 5001
  按 Ctrl+C 停止服务
====================================

* Running on http://127.0.0.1:5001
```

## 常见问题

### Q: Flash Attention 2 安装失败？
A: 这是正常的，服务会自动降级到标准 attention，功能不受影响，只是速度稍慢。

### Q: 模型下载很慢或失败？
A: 可以考虑：
1. 使用 Hugging Face 镜像站
2. 手动从 https://huggingface.co/deepseek-ai/DeepSeek-OCR 下载模型文件

### Q: CUDA 不可用？
A: 检查：
1. NVIDIA 驱动是否正确安装
2. PyTorch 版本是否匹配 CUDA 版本
3. 可以使用 CPU 模式，但速度会较慢

## 参考链接

- 官方模型页面: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- GitHub: https://github.com/deepseek-ai/DeepSeek-OCR
- 论文: https://arxiv.org/abs/2510.18234

