#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基准测试配置文件
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ACE 2005 数据集路径
ACE_DATA_PATH = PROJECT_ROOT / "test-data" / "ace_2005_td_v7"

# Gold数据输出路径
GOLD_OUTPUT_PATH = PROJECT_ROOT / "benchmark" / "gold_data"

# 默认配置
DEFAULT_CONFIG = {
    # 解析配置
    'languages': ['Arabic'],  # 可选: Arabic, Chinese, English
    'sources': ['bn', 'nw'],  # 可选: bn, nw, wl, bc, cts, un
    'annotation_level': 'adj',  # 可选: adj, 1p, dual
    
    # 转换配置
    'use_subtype': True,      # 使用子类型作为entity类别
    'use_head_only': False,   # 使用extent作为attribute（完整短语，如"国际艺术团体"而非"团体"）
    
    # 评估配置
    'exact_match_threshold': 0.95,    # 精确匹配相似度阈值
    'partial_match_threshold': 0.5,   # 部分匹配相似度阈值
    'case_sensitive': False,          # 是否区分大小写
    
    # API配置
    'api_url': 'http://localhost:3001',
    'api_timeout': 60,
    
    # 测试配置
    'test_limit': 100,        # 默认测试样本数
    'max_text_length': 2000,  # 文本截断长度
}

# ACE实体类型中文名称
ENTITY_TYPE_NAMES_ZH = {
    'PER': '人物',
    'ORG': '组织',
    'GPE': '地缘政治实体',
    'LOC': '位置',
    'FAC': '设施',
    'VEH': '载具',
    'WEA': '武器',
}

# ACE关系类型中文名称
RELATION_TYPE_NAMES_ZH = {
    'PHYS': '物理关系',
    'PART-WHOLE': '部分-整体关系',
    'PER-SOC': '人际关系',
    'ORG-AFF': '组织关系',
    'ART': '人工制品关系',
    'GEN-AFF': '一般关系',
}

# ACE事件类型中文名称
EVENT_TYPE_NAMES_ZH = {
    'Life': '生命事件',
    'Movement': '移动事件',
    'Transaction': '交易事件',
    'Business': '商业事件',
    'Conflict': '冲突事件',
    'Contact': '联系事件',
    'Personnel': '人事事件',
    'Justice': '司法事件',
}
