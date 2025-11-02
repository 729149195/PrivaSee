#!/usr/bin/env python3
"""
DeepSeek-OCR 服务测试脚本
测试所有 API 端点和功能
"""

import requests
import sys
import os
from pathlib import Path
from PIL import Image
import io

# 配置
SERVICE_URL = "http://localhost:5001"
TEST_IMAGE_PATH = None  # 可以指定测试图片路径

# 颜色输出
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.NC}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.NC}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.NC}")

def create_test_image():
    """创建一个测试图片"""
    from PIL import Image, ImageDraw, ImageFont
    
    # 创建白色背景
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # 绘制文本
    text = "DeepSeek OCR Test\n这是一个测试图片\n包含中英文文本\n\nFormula: E = mc²\n\nTable:\n┌─────┬─────┐\n│  A  │  B  │\n├─────┼─────┤\n│  1  │  2  │\n└─────┴─────┘"
    
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 50), text, fill='black', font=font)
    
    # 保存到内存
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

def test_health_check():
    """测试健康检查"""
    print("\n" + "=" * 70)
    print("测试 1: 健康检查")
    print("=" * 70)
    
    try:
        response = requests.get(f"{SERVICE_URL}/api/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("服务正常运行")
            print_info(f"  状态: {data.get('status')}")
            print_info(f"  服务: {data.get('service')}")
            print_info(f"  模型已加载: {data.get('model_loaded')}")
            print_info(f"  设备: {data.get('device')}")
            
            if data.get('gpu_info'):
                print_info(f"  GPU: {data['gpu_info'].get('name')}")
                print_info(f"  显存: {data['gpu_info'].get('memory_total')}")
            
            print_info(f"  PDF 支持: {data.get('pdf_support')}")
            return True
        else:
            print_error(f"服务返回错误: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("无法连接到服务，请确保服务已启动")
        print_info(f"启动命令: bash start_deepseek_ocr_service.sh")
        return False
    except Exception as e:
        print_error(f"测试失败: {e}")
        return False

def test_list_functions():
    """测试功能列表"""
    print("\n" + "=" * 70)
    print("测试 2: 获取功能列表")
    print("=" * 70)
    
    try:
        response = requests.get(f"{SERVICE_URL}/api/functions")
        
        if response.status_code == 200:
            data = response.json()
            functions = data.get('functions', [])
            
            print_success(f"获取到 {len(functions)} 个功能:")
            for func in functions:
                print_info(f"  [{func['id']}] {func['name']}")
                print(f"      {func['description']}")
            return True
        else:
            print_error(f"获取失败: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"测试失败: {e}")
        return False

def test_list_resolutions():
    """测试分辨率列表"""
    print("\n" + "=" * 70)
    print("测试 3: 获取分辨率模式")
    print("=" * 70)
    
    try:
        response = requests.get(f"{SERVICE_URL}/api/resolutions")
        
        if response.status_code == 200:
            data = response.json()
            modes = data.get('modes', [])
            recommended = data.get('recommended')
            
            print_success(f"获取到 {len(modes)} 种分辨率模式:")
            for mode in modes:
                marker = " [推荐]" if mode['id'] == recommended else ""
                print_info(f"  [{mode['id']}] {mode['description']}{marker}")
                print(f"      配置: {mode['config']}")
            return True
        else:
            print_error(f"获取失败: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"测试失败: {e}")
        return False

def test_process_image(image_path=None):
    """测试图片处理"""
    print("\n" + "=" * 70)
    print("测试 4: 图片处理")
    print("=" * 70)
    
    try:
        # 准备测试图片
        if image_path and os.path.exists(image_path):
            print_info(f"使用测试图片: {image_path}")
            files = {'file': open(image_path, 'rb')}
        else:
            print_info("生成测试图片...")
            test_img = create_test_image()
            files = {'file': ('test.png', test_img, 'image/png')}
        
        # 测试自由 OCR
        print("\n测试功能: 自由OCR识别")
        data = {
            'function': 'free_ocr',
            'resolution': 'small',
            'save_results': 'false'
        }
        
        response = requests.post(
            f"{SERVICE_URL}/api/process",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success("处理成功")
            print_info(f"  功能: {result['metadata']['function_name']}")
            print_info(f"  分辨率: {result['metadata']['resolution']}")
            print_info(f"  结果长度: {len(result['text'])} 字符")
            print("\n识别结果:")
            print("-" * 70)
            print(result['text'][:500] + ("..." if len(result['text']) > 500 else ""))
            print("-" * 70)
            return True
        else:
            print_error(f"处理失败: {response.status_code}")
            print_error(response.text)
            return False
            
    except Exception as e:
        print_error(f"测试失败: {e}")
        return False

def test_visual_qa(image_path=None):
    """测试视觉问答"""
    print("\n" + "=" * 70)
    print("测试 5: 视觉问答")
    print("=" * 70)
    
    try:
        # 准备测试图片
        if image_path and os.path.exists(image_path):
            files = {'file': open(image_path, 'rb')}
        else:
            test_img = create_test_image()
            files = {'file': ('test.png', test_img, 'image/png')}
        
        data = {
            'function': 'visual_qa',
            'question': 'What text do you see in this image?',
            'resolution': 'small',
            'save_results': 'false'
        }
        
        response = requests.post(
            f"{SERVICE_URL}/api/process",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success("视觉问答成功")
            print_info(f"  问题: {data['question']}")
            print("\n回答:")
            print("-" * 70)
            print(result['text'])
            print("-" * 70)
            return True
        else:
            print_error(f"处理失败: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("=" * 70)
    print("DeepSeek-OCR 服务测试")
    print("=" * 70)
    print_info(f"服务地址: {SERVICE_URL}")
    
    # 检查是否指定了测试图片
    test_image = None
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        if not os.path.exists(test_image):
            print_warning(f"测试图片不存在: {test_image}")
            test_image = None
    
    # 运行测试
    results = []
    
    results.append(("健康检查", test_health_check()))
    results.append(("功能列表", test_list_functions()))
    results.append(("分辨率模式", test_list_resolutions()))
    results.append(("图片处理", test_process_image(test_image)))
    results.append(("视觉问答", test_visual_qa(test_image)))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        if result:
            print_success(f"{name}: 通过")
            passed += 1
        else:
            print_error(f"{name}: 失败")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"总计: {passed + failed} 个测试")
    print_success(f"通过: {passed}")
    if failed > 0:
        print_error(f"失败: {failed}")
    print("=" * 70)
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

