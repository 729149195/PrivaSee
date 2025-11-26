#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 测试你的信息元提取系统
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.ace_parser import ACEParser
from benchmark.ace_to_infons import ACEToInfonsConverter, infon_to_dict, infon_to_compact_format
from benchmark.evaluator import InfonEvaluator, parse_compact_format, print_evaluation_report


def test_with_sample_data():
    """使用样例数据测试"""
    print("=" * 60)
    print("       ACE 2005 信息元提取 Benchmark 快速测试")
    print("=" * 60)
    
    # 1. 加载ACE数据
    print("\n[1] 加载ACE 2005测试数据...")
    ace_path = Path(__file__).parent.parent / 'test-data' / 'ace_2005_td_v7'
    
    if not ace_path.exists():
        print(f"错误: 找不到ACE数据集: {ace_path}")
        return
    
    parser = ACEParser(str(ace_path))
    documents = parser.parse_all(annotation_level='adj', limit=5)
    print(f"   加载了 {len(documents)} 个文档")
    
    # 2. 转换为Gold标准
    print("\n[2] 转换为Infons格式...")
    converter = ACEToInfonsConverter()
    gold_samples = converter.convert_all(documents)
    
    total_infons = sum(len(s.infons) for s in gold_samples)
    print(f"   生成了 {total_infons} 个Gold信息元")
    
    # 3. 展示样例
    print("\n[3] Gold标准样例:")
    print("-" * 60)
    
    for sample in gold_samples[:2]:
        print(f"\n文档: {sample.doc_id}")
        print(f"文本前200字: {sample.text[:200]}...")
        print(f"\n信息元 ({len(sample.infons)}个):")
        
        desc_count = sum(1 for i in sample.infons if i.infon_type == 'DESC')
        scen_count = sum(1 for i in sample.infons if i.infon_type == 'SCEN')
        rel_count = sum(1 for i in sample.infons if i.infon_type == 'REL')
        print(f"  DESC: {desc_count}, SCEN: {scen_count}, REL: {rel_count}")
        
        print("\n  前5个DESC:")
        for inf in [i for i in sample.infons if i.infon_type == 'DESC'][:5]:
            print(f"    {infon_to_compact_format(inf)}")
        
        print("\n  前3个REL:")
        for inf in [i for i in sample.infons if i.infon_type == 'REL'][:3]:
            print(f"    {infon_to_compact_format(inf)}")
        
        if scen_count > 0:
            print("\n  SCEN:")
            for inf in [i for i in sample.infons if i.infon_type == 'SCEN'][:2]:
                print(f"    {infon_to_compact_format(inf)}")
    
    # 4. 模拟评估
    print("\n" + "=" * 60)
    print("[4] 评估示例（使用Gold数据模拟完美预测）")
    print("=" * 60)
    
    predictions_list = []
    for sample in gold_samples:
        preds = [infon_to_dict(inf) for inf in sample.infons]
        predictions_list.append(preds)
    
    evaluator = InfonEvaluator()
    result = evaluator.evaluate_batch(gold_samples, predictions_list)
    print_evaluation_report(result, detailed=False)
    
    # 5. 使用指南
    print("\n" + "=" * 60)
    print("[5] 如何评估你的系统")
    print("=" * 60)
    print("""
方法1: 使用API自动测试
  python -m benchmark.run_benchmark test \\
      --ace-path ./test-data/ace_2005_td_v7 \\
      --api-url http://localhost:3001 \\
      --limit 50

方法2: 手动评估
  1. 先生成Gold数据:
     python -m benchmark.run_benchmark convert \\
         --ace-path ./test-data/ace_2005_td_v7 \\
         --output ./benchmark/gold_data
  
  2. 用你的系统提取每个样本的infons，保存为JSON:
     {
       "DOC_ID_1": [{"iid":"desc:1","infon_type":"DESC",...}],
       "DOC_ID_2": [...]
     }
  
  3. 运行评估:
     python -m benchmark.run_benchmark evaluate \\
         --gold ./benchmark/gold_data/gold.json \\
         --predictions ./your_predictions.json

方法3: 在Python代码中评估
  from benchmark.evaluator import InfonEvaluator, parse_compact_format
  
  evaluator = InfonEvaluator()
  
  # 你的预测结果（compact格式字符串）
  pred_text = '''
  desc:r1_1,DESC,Person,王小明,string,0.95
  desc:r1_2,DESC,Location,北京,string,0.90
  rel:r1_3,REL,located_at,desc:r1_1|desc:r1_2,0.85
  '''
  predictions = parse_compact_format(pred_text)
  
  # 评估
  result = evaluator.evaluate_single(gold_sample, predictions)
""")


if __name__ == '__main__':
    test_with_sample_data()
