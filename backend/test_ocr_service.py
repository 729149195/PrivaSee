#!/usr/bin/env python3
"""
DeepSeek-OCR 服务测试脚本
用于快速测试 OCR 服务是否正常工作
"""

import requests
import sys
import os
from pathlib import Path

OCR_SERVER_URL = "http://localhost:5001"

def test_health_check():
    """测试健康检查接口"""
    print("🔍 测试健康检查接口...")
    try:
        response = requests.get(f"{OCR_SERVER_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ 服务健康检查通过")
            print(f"   - 模型已加载: {data.get('model_loaded')}")
            print(f"   - 设备: {data.get('device')}")
            print(f"   - 模型路径: {data.get('model_path')}")
            return True
        else:
            print(f"❌ 服务健康检查失败: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务，请确认服务已启动")
        print("   启动命令: bash start_deepseek_ocr.sh")
        return False
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_list_functions():
    """测试功能列表接口"""
    print("\n🔍 测试功能列表接口...")
    try:
        response = requests.get(f"{OCR_SERVER_URL}/api/ocr/functions")
        if response.status_code == 200:
            data = response.json()
            functions = data.get('functions', [])
            print(f"✅ 获取到 {len(functions)} 个功能:")
            for func in functions:
                print(f"   - {func['name']} ({func['id']})")
            return True
        else:
            print(f"❌ 获取功能列表失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取功能列表失败: {e}")
        return False

def test_list_resolutions():
    """测试分辨率模式列表"""
    print("\n🔍 测试分辨率模式列表...")
    try:
        response = requests.get(f"{OCR_SERVER_URL}/api/ocr/resolutions")
        if response.status_code == 200:
            data = response.json()
            modes = data.get('modes', [])
            recommended = data.get('recommended')
            print(f"✅ 获取到 {len(modes)} 个分辨率模式:")
            for mode in modes:
                marker = " (推荐)" if mode['id'] == recommended else ""
                print(f"   - {mode['id']}{marker}: {mode['config']}")
            return True
        else:
            print(f"❌ 获取分辨率模式失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取分辨率模式失败: {e}")
        return False

def test_ocr_process(image_path=None):
    """测试 OCR 处理接口"""
    print("\n🔍 测试 OCR 处理接口...")
    
    if not image_path:
        print("⚠️  未提供测试图片，跳过 OCR 处理测试")
        print("   使用方法: python test_ocr_service.py <image_path>")
        return True
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'function': 'free_ocr',
                'resolution': 'gundam',
                'save_result': 'false'
            }
            
            print(f"   正在处理: {image_path}")
            print("   这可能需要一些时间...")
            
            response = requests.post(
                f"{OCR_SERVER_URL}/api/ocr/process",
                files=files,
                data=data,
                timeout=300  # 5 分钟超时
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ OCR 处理成功")
                print(f"   - 功能: {result.get('function_name')}")
                print(f"   - 文本长度: {len(result.get('text', ''))} 字符")
                print(f"   - 结果预览:")
                text = result.get('text', '')
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"     {preview}")
                return True
            else:
                error = response.json()
                print(f"❌ OCR 处理失败: {error.get('error')}")
                return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时，处理时间过长")
        return False
    except Exception as e:
        print(f"❌ OCR 处理失败: {e}")
        return False

def main():
    """主测试流程"""
    print("=" * 60)
    print("  DeepSeek-OCR 服务测试")
    print("=" * 60)
    
    # 测试1: 健康检查
    if not test_health_check():
        print("\n⚠️  服务未启动或不可用，停止测试")
        sys.exit(1)
    
    # 测试2: 功能列表
    test_list_functions()
    
    # 测试3: 分辨率模式
    test_list_resolutions()
    
    # 测试4: OCR 处理（如果提供了图片）
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_ocr_process(image_path)
    
    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)
    
    if image_path:
        print("\n💡 提示: 如需测试其他功能，请修改 test_ocr_process 中的 'function' 参数")
        print("   可用功能: free_ocr, markdown, table_extract, formula_extract, visual_qa 等")

if __name__ == '__main__':
    main()

