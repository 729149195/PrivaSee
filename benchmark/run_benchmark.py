#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACE 2005 信息元提取基准测试主程序

功能：
1. 解析ACE 2005数据集
2. 转换为PrivaSee信息元格式
3. 运行系统提取
4. 评估提取结果
5. 生成详细报告

使用方式：
    # 转换ACE数据并导出Gold标准
    python -m benchmark.run_benchmark convert --ace-path ./test-data/ace_2005_td_v7 --output ./benchmark/gold_data
    
    # 运行评估（需要提供预测结果）
    python -m benchmark.run_benchmark evaluate --gold ./benchmark/gold_data/gold.json --predictions ./predictions.json
    
    # 完整测试流程（使用PrivaSee API）
    python -m benchmark.run_benchmark test --ace-path ./test-data/ace_2005_td_v7 --api-url http://localhost:3001
"""

import argparse
import json
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# 添加父目录到path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.ace_parser import ACEParser, get_statistics
from benchmark.ace_to_infons import (
    ACEToInfonsConverter, GoldSample,
    export_to_json, export_to_csv,
    infon_to_dict, sample_to_dict
)
from benchmark.evaluator import (
    InfonEvaluator, parse_compact_format,
    print_evaluation_report
)


class Colors:
    """ANSI颜色代码"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    """打印横幅"""
    print(f"""
{Colors.HEADER}{'='*70}
{Colors.BOLD}     ACE 2005 → PrivaSee 信息元提取基准测试工具
{'='*70}{Colors.ENDC}
""")


def cmd_convert(args):
    """转换ACE数据集为Gold标准"""
    print(f"\n{Colors.OKBLUE}[1/3] 解析ACE 2005数据集...{Colors.ENDC}")
    
    parser = ACEParser(args.ace_path)
    
    # 查找文档
    apf_files = parser.find_all_documents(
        languages=args.languages.split(',') if args.languages else None,
        sources=args.sources.split(',') if args.sources else None,
        annotation_level=args.annotation_level
    )
    
    print(f"  找到 {len(apf_files)} 个APF文件")
    
    if args.limit:
        apf_files = apf_files[:args.limit]
        print(f"  限制处理前 {args.limit} 个文件")
    
    # 解析文档
    documents = []
    for i, apf_path in enumerate(apf_files):
        if (i + 1) % 100 == 0:
            print(f"  已解析 {i+1}/{len(apf_files)} ...")
        doc = parser.parse_document(apf_path)
        if doc:
            documents.append(doc)
    
    print(f"  成功解析 {len(documents)} 个文档")
    
    # 显示统计
    stats = get_statistics(documents)
    print(f"\n{Colors.OKBLUE}[2/3] ACE数据集统计:{Colors.ENDC}")
    print(f"  实体总数: {stats['total_entities']}")
    print(f"  实体提及: {stats['total_entity_mentions']}")
    print(f"  关系总数: {stats['total_relations']}")
    print(f"  事件总数: {stats['total_events']}")
    
    # 转换为Infons
    print(f"\n{Colors.OKBLUE}[3/3] 转换为Infons格式...{Colors.ENDC}")
    
    converter = ACEToInfonsConverter(
        use_subtype=not args.no_subtype,
        use_head_only=args.use_head_only
    )
    
    samples = converter.convert_all(documents)
    
    # 统计转换结果
    total_desc = sum(sum(1 for inf in s.infons if inf.infon_type == 'DESC') for s in samples)
    total_scen = sum(sum(1 for inf in s.infons if inf.infon_type == 'SCEN') for s in samples)
    total_rel = sum(sum(1 for inf in s.infons if inf.infon_type == 'REL') for s in samples)
    
    print(f"  转换完成:")
    print(f"    DESC信息元: {total_desc}")
    print(f"    SCEN信息元: {total_scen}")
    print(f"    REL信息元:  {total_rel}")
    print(f"    总计:       {total_desc + total_scen + total_rel}")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导出
    json_path = output_dir / 'gold.json'
    csv_path = output_dir / 'gold.csv'
    
    export_to_json(samples, str(json_path))
    export_to_csv(samples, str(csv_path))
    
    # 保存统计信息
    stats_path = output_dir / 'statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'ace_statistics': stats,
            'infon_statistics': {
                'total_samples': len(samples),
                'total_desc': total_desc,
                'total_scen': total_scen,
                'total_rel': total_rel,
            },
            'conversion_config': {
                'use_subtype': not args.no_subtype,
                'use_head_only': args.use_head_only,
                'annotation_level': args.annotation_level,
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{Colors.OKGREEN}导出完成:{Colors.ENDC}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  统计: {stats_path}")


def cmd_evaluate(args):
    """评估预测结果"""
    print(f"\n{Colors.OKBLUE}加载数据...{Colors.ENDC}")
    
    # 加载Gold标准
    with open(args.gold, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    # 重建GoldSample对象
    gold_samples = []
    for item in gold_data:
        infons = []
        for inf_dict in item.get('infons', []):
            infon_type = inf_dict.get('infon_type', '').upper()
            if infon_type == 'DESC':
                from benchmark.ace_to_infons import DescInfon
                infons.append(DescInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='DESC',
                    entity=inf_dict.get('entity', ''),
                    attribute=inf_dict.get('attribute', ''),
                    data_type=inf_dict.get('data_type', 'string'),
                    confidence=inf_dict.get('confidence', 1.0),
                    source_ace_id=inf_dict.get('source_ace_id', ''),
                    char_start=inf_dict.get('char_start', -1),
                    char_end=inf_dict.get('char_end', -1)
                ))
            elif infon_type == 'SCEN':
                from benchmark.ace_to_infons import ScenInfon
                infons.append(ScenInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='SCEN',
                    temporal=inf_dict.get('temporal', ''),
                    spatial=inf_dict.get('spatial', ''),
                    confidence=inf_dict.get('confidence', 1.0),
                    source_ace_id=inf_dict.get('source_ace_id', ''),
                    char_start=inf_dict.get('char_start', -1),
                    char_end=inf_dict.get('char_end', -1)
                ))
            elif infon_type == 'REL':
                from benchmark.ace_to_infons import RelInfon
                infons.append(RelInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='REL',
                    relation_name=inf_dict.get('relation_name', ''),
                    arg_refs=inf_dict.get('arg_refs', []),
                    confidence=inf_dict.get('confidence', 1.0),
                    source_ace_id=inf_dict.get('source_ace_id', ''),
                    char_start=inf_dict.get('char_start', -1),
                    char_end=inf_dict.get('char_end', -1)
                ))
        
        sample = GoldSample(
            doc_id=item.get('doc_id', ''),
            text=item.get('text', ''),
            infons=infons,
            source_file=item.get('source_file', ''),
            language=item.get('language', '')
        )
        gold_samples.append(sample)
    
    print(f"  加载了 {len(gold_samples)} 个Gold样本")
    
    # 加载预测结果
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    # 预测结果应该是与gold_samples对应的列表
    if isinstance(predictions_data, dict):
        # 如果是按doc_id索引的字典
        predictions_list = []
        for sample in gold_samples:
            preds = predictions_data.get(sample.doc_id, [])
            if isinstance(preds, str):
                preds = parse_compact_format(preds)
            predictions_list.append(preds)
    else:
        predictions_list = predictions_data
    
    print(f"  加载了 {len(predictions_list)} 个预测结果")
    
    # 评估
    print(f"\n{Colors.OKBLUE}运行评估...{Colors.ENDC}")
    
    evaluator = InfonEvaluator(
        exact_match_threshold=args.exact_threshold,
        partial_match_threshold=args.partial_threshold,
        case_sensitive=args.case_sensitive
    )
    
    result = evaluator.evaluate_batch(gold_samples, predictions_list)
    
    # 打印报告
    print_evaluation_report(result, detailed=args.detailed)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n{Colors.OKGREEN}评估结果已保存到: {output_path}{Colors.ENDC}")


def cmd_test(args):
    """完整测试流程"""
    print_banner()
    
    # 1. 加载Gold数据
    if args.gold:
        print(f"\n{Colors.OKBLUE}[1/4] 加载预生成的Gold数据...{Colors.ENDC}")
        with open(args.gold, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
    else:
        print(f"\n{Colors.OKBLUE}[1/4] 解析并转换ACE数据...{Colors.ENDC}")
        parser = ACEParser(args.ace_path)
        documents = parser.parse_all(
            annotation_level=args.annotation_level,
            limit=args.limit
        )
        converter = ACEToInfonsConverter()
        samples = converter.convert_all(documents)
        gold_data = [sample_to_dict(s) for s in samples]
    
    print(f"  Gold样本数: {len(gold_data)}")
    
    # 2. 调用API提取信息元
    print(f"\n{Colors.OKBLUE}[2/4] 调用PrivaSee API提取信息元...{Colors.ENDC}")
    
    predictions = {}
    success_count = 0
    error_count = 0
    
    for i, item in enumerate(gold_data):
        doc_id = item.get('doc_id', f'doc_{i}')
        text = item.get('text', '')
        
        if not text.strip():
            predictions[doc_id] = []
            continue
        
        # 截断过长文本
        if len(text) > 2000:
            text = text[:2000]
        
        try:
            # 调用API
            response = requests.post(
                f"{args.api_url}/api/extract",
                json={'text': text, 'modality': 'text'},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # 解析返回的infons
                if 'infons' in result:
                    predictions[doc_id] = result['infons']
                elif 'raw_output' in result:
                    predictions[doc_id] = parse_compact_format(result['raw_output'])
                else:
                    predictions[doc_id] = []
                success_count += 1
            else:
                print(f"  {Colors.WARNING}API错误 ({doc_id}): {response.status_code}{Colors.ENDC}")
                predictions[doc_id] = []
                error_count += 1
                
        except requests.exceptions.Timeout:
            print(f"  {Colors.WARNING}超时 ({doc_id}){Colors.ENDC}")
            predictions[doc_id] = []
            error_count += 1
        except Exception as e:
            print(f"  {Colors.FAIL}异常 ({doc_id}): {e}{Colors.ENDC}")
            predictions[doc_id] = []
            error_count += 1
        
        # 进度显示
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(gold_data)} (成功: {success_count}, 错误: {error_count})")
        
        # 避免请求过快
        time.sleep(0.1)
    
    print(f"\n  提取完成: 成功 {success_count}, 错误 {error_count}")
    
    # 3. 重建GoldSample并评估
    print(f"\n{Colors.OKBLUE}[3/4] 评估提取结果...{Colors.ENDC}")
    
    # 重建samples（简化版本，用于评估）
    gold_samples = []
    predictions_list = []
    
    for item in gold_data:
        doc_id = item.get('doc_id', '')
        infons = []
        for inf_dict in item.get('infons', []):
            infon_type = inf_dict.get('infon_type', '').upper()
            if infon_type == 'DESC':
                from benchmark.ace_to_infons import DescInfon
                infons.append(DescInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='DESC',
                    entity=inf_dict.get('entity', ''),
                    attribute=inf_dict.get('attribute', ''),
                    data_type=inf_dict.get('data_type', 'string'),
                    confidence=inf_dict.get('confidence', 1.0)
                ))
            elif infon_type == 'SCEN':
                from benchmark.ace_to_infons import ScenInfon
                infons.append(ScenInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='SCEN',
                    temporal=inf_dict.get('temporal', ''),
                    spatial=inf_dict.get('spatial', '')
                ))
            elif infon_type == 'REL':
                from benchmark.ace_to_infons import RelInfon
                infons.append(RelInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='REL',
                    relation_name=inf_dict.get('relation_name', ''),
                    arg_refs=inf_dict.get('arg_refs', [])
                ))
        
        sample = GoldSample(
            doc_id=doc_id,
            text=item.get('text', ''),
            infons=infons
        )
        gold_samples.append(sample)
        predictions_list.append(predictions.get(doc_id, []))
    
    evaluator = InfonEvaluator()
    result = evaluator.evaluate_batch(gold_samples, predictions_list)
    
    # 4. 生成报告
    print(f"\n{Colors.OKBLUE}[4/4] 生成报告...{Colors.ENDC}")
    print_evaluation_report(result, detailed=args.detailed)
    
    # 保存结果
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存评估结果
        eval_path = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存预测结果
        pred_path = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(pred_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        print(f"\n{Colors.OKGREEN}结果已保存:{Colors.ENDC}")
        print(f"  评估结果: {eval_path}")
        print(f"  预测结果: {pred_path}")


def cmd_stats(args):
    """显示数据集统计信息"""
    print_banner()
    
    print(f"\n{Colors.OKBLUE}解析ACE 2005数据集...{Colors.ENDC}")
    
    parser = ACEParser(args.ace_path)
    documents = parser.parse_all(
        annotation_level=args.annotation_level,
        limit=args.limit
    )
    
    stats = get_statistics(documents)
    
    print(f"\n{Colors.OKGREEN}=== ACE 2005 数据集统计 ==={Colors.ENDC}")
    print(f"\n文档数量: {stats['total_documents']}")
    
    print(f"\n【实体统计】")
    print(f"  总数: {stats['total_entities']}")
    print(f"  提及: {stats['total_entity_mentions']}")
    print(f"  类型分布:")
    for etype, count in sorted(stats['entity_types'].items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count}")
    
    print(f"\n【关系统计】")
    print(f"  总数: {stats['total_relations']}")
    print(f"  提及: {stats['total_relation_mentions']}")
    print(f"  类型分布:")
    for rtype, count in sorted(stats['relation_types'].items(), key=lambda x: -x[1])[:15]:
        print(f"    {rtype}: {count}")
    
    print(f"\n【事件统计】")
    print(f"  总数: {stats['total_events']}")
    print(f"  提及: {stats['total_event_mentions']}")
    print(f"  类型分布:")
    for etype, count in sorted(stats['event_types'].items(), key=lambda x: -x[1])[:15]:
        print(f"    {etype}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='ACE 2005 信息元提取基准测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换ACE数据并导出Gold标准
  python -m benchmark.run_benchmark convert --ace-path ./test-data/ace_2005_td_v7 --output ./benchmark/gold_data
  
  # 显示数据集统计
  python -m benchmark.run_benchmark stats --ace-path ./test-data/ace_2005_td_v7
  
  # 评估预测结果
  python -m benchmark.run_benchmark evaluate --gold ./benchmark/gold_data/gold.json --predictions ./predictions.json
  
  # 完整测试（需要运行PrivaSee后端）
  python -m benchmark.run_benchmark test --ace-path ./test-data/ace_2005_td_v7 --api-url http://localhost:3001
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # convert子命令
    convert_parser = subparsers.add_parser('convert', help='转换ACE数据集为Gold标准')
    convert_parser.add_argument('--ace-path', required=True, help='ACE 2005数据集路径')
    convert_parser.add_argument('--output', '-o', required=True, help='输出目录')
    convert_parser.add_argument('--languages', help='语言列表，逗号分隔 (Arabic,Chinese,English)')
    convert_parser.add_argument('--sources', help='来源列表，逗号分隔 (bn,nw,wl,bc,cts,un)')
    convert_parser.add_argument('--annotation-level', default='adj', help='标注级别 (adj, 1p, dual)')
    convert_parser.add_argument('--limit', type=int, help='限制处理的文档数量')
    convert_parser.add_argument('--no-subtype', action='store_true', help='不使用子类型')
    convert_parser.add_argument('--use-head-only', action='store_true', help='只使用head（核心词）而非extent（完整短语）')
    
    # evaluate子命令
    eval_parser = subparsers.add_parser('evaluate', help='评估预测结果')
    eval_parser.add_argument('--gold', required=True, help='Gold标准JSON文件')
    eval_parser.add_argument('--predictions', required=True, help='预测结果JSON文件')
    eval_parser.add_argument('--output', '-o', help='保存评估结果的路径')
    eval_parser.add_argument('--exact-threshold', type=float, default=0.95, help='精确匹配阈值')
    eval_parser.add_argument('--partial-threshold', type=float, default=0.5, help='部分匹配阈值')
    eval_parser.add_argument('--case-sensitive', action='store_true', help='区分大小写')
    eval_parser.add_argument('--detailed', action='store_true', help='显示详细结果')
    
    # test子命令
    test_parser = subparsers.add_parser('test', help='完整测试流程')
    test_parser.add_argument('--ace-path', help='ACE 2005数据集路径')
    test_parser.add_argument('--gold', help='预生成的Gold数据JSON')
    test_parser.add_argument('--api-url', default='http://localhost:3001', help='PrivaSee API地址')
    test_parser.add_argument('--output', '-o', help='输出目录')
    test_parser.add_argument('--annotation-level', default='adj', help='标注级别')
    test_parser.add_argument('--limit', type=int, default=50, help='限制测试的样本数')
    test_parser.add_argument('--detailed', action='store_true', help='显示详细结果')
    
    # stats子命令
    stats_parser = subparsers.add_parser('stats', help='显示数据集统计')
    stats_parser.add_argument('--ace-path', required=True, help='ACE 2005数据集路径')
    stats_parser.add_argument('--annotation-level', default='adj', help='标注级别')
    stats_parser.add_argument('--limit', type=int, help='限制处理的文档数量')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        cmd_convert(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'test':
        if not args.ace_path and not args.gold:
            print(f"{Colors.FAIL}错误: 必须指定 --ace-path 或 --gold{Colors.ENDC}")
            sys.exit(1)
        cmd_test(args)
    elif args.command == 'stats':
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
