#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试中文字体显示"""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# 直接使用字体文件
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

if os.path.exists(font_path):
    print(f"✓ 字体文件存在: {font_path}")
    
    # 创建字体属性
    font_prop = FontProperties(fname=font_path)
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 测试中文文本
    test_texts = [
        '程序员', '教师', '医生', '学生', '上班族',
        '日常生活场景', '工作场景', '社交场景',
        '简单推理', '复杂关联', '直接暴露'
    ]
    
    for i, text in enumerate(test_texts):
        ax.text(0.5, 0.9 - i*0.08, text, 
                fontproperties=font_prop,
                fontsize=16,
                ha='center',
                transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('中文字体测试 Chinese Font Test', 
                 fontproperties=font_prop, 
                 fontsize=20, 
                 pad=20)
    
    plt.tight_layout()
    plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 测试图表已生成: font_test.png")
    print("  请打开查看中文是否正常显示")
else:
    print(f"✗ 字体文件不存在: {font_path}")

