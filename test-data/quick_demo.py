#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速演示脚本 - 生成5个测试用例
"""

import asyncio
from generate_test_cases_with_api import generate_batch, save_to_csv

async def main():
    print("🚀 快速演示：生成5个测试用例")
    print("=" * 80)
    
    test_cases = await generate_batch(batch_size=5)
    
    print(f"\n✅ 成功生成 {len(test_cases)} 个测试用例")
    
    # 显示生成的内容
    for case in test_cases:
        print(f"\n{'='*80}")
        print(f"测试用例 #{case['id']}")
        print(f"类别: {case['metadata']['category']}")
        print(f"难度: {case['metadata']['difficulty']}")
        print(f"文本: {case['text'][:100]}...")
    
    # 保存到文件
    filename = "test_cases.csv"
    save_to_csv(test_cases, filename)
    
    print(f"\n{'='*80}")
    print("🎉 演示完成！")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())

