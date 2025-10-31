# DeepSeek-OCR 快速开始

## 方法 1: 使用自动化脚本（推荐）

### 第一步：运行安装脚本

```bash
cd /home/zhangxiangxuan/桌面/Projects/PrivaSee/backend
chmod +x setup_deepseek_ocr_fresh.sh
bash setup_deepseek_ocr_fresh.sh
```

### 第二步：启动服务

```bash
chmod +x start_deepseek_ocr_new.sh
bash start_deepseek_ocr_new.sh
```

## 方法 2: 使用 Python 安装向导

```bash
cd /home/zhangxiangxuan/桌面/Projects/PrivaSee/backend
python install_deepseek_ocr.py
```

然后按照屏幕提示逐步完成安装。

## 方法 3: 手动安装（完全控制）

### 1. 创建环境

```bash
# 删除旧环境
conda remove -n deepseek-ocr --all -y

# 创建新环境
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
```

### 2. 安装依赖

```bash
# PyTorch
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers 相关
pip install transformers==4.46.3 tokenizers==0.20.3 einops addict easydict

# Flash Attention（可选）
pip install flash-attn==2.7.3 --no-build-isolation

# 服务器依赖
pip install flask flask-cors pillow pdf2image numpy
```

### 3. 下载模型

```bash
cd /home/zhangxiangxuan/桌面/Projects/PrivaSee/backend
python download_model.py
```

### 4. 验证安装

```bash
python verify_installation.py
```

### 5. 启动服务

```bash
python deepseek_ocr_server.py
```

## 验证服务运行

服务启动后访问：http://localhost:5001

应该看到：
```
* Running on http://127.0.0.1:5001
```

## 测试 OCR

```python
import requests

url = "http://localhost:5001/api/ocr/process"
files = {'file': open('your_image.jpg', 'rb')}
data = {
    'function': 'markdown',
    'resolution': 'gundam'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## 已删除的旧文件

以下文件已被清理：
- ✓ `/home/zhangxiangxuan/桌面/Projects/PrivaSee/models/deepseek-ocr` (旧模型)
- ✓ `~/.cache/huggingface/modules/transformers_modules/deepseek*` (缓存)

## 新环境信息

- **环境名称**: deepseek-ocr
- **Python 版本**: 3.12
- **PyTorch 版本**: 2.6.0
- **Transformers 版本**: 4.46.3
- **CUDA 版本**: 11.8

## 常见问题

**Q: 脚本执行权限错误？**
```bash
chmod +x setup_deepseek_ocr_fresh.sh start_deepseek_ocr_new.sh
```

**Q: conda 环境激活失败？**
```bash
eval "$(conda shell.bash hook)"
conda activate deepseek-ocr
```

**Q: Flash Attention 安装失败？**  
A: 这是正常的，服务会自动使用标准 attention。

**Q: 模型下载慢？**  
A: 考虑使用 Hugging Face 镜像或手动下载。

## 参考文档

- 详细安装说明: `INSTALL_DEEPSEEK_OCR.md`
- 官方文档: https://huggingface.co/deepseek-ai/DeepSeek-OCR

