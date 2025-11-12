#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试结果分析脚本
用于分析性能测试的CSV结果文件，生成统计报告和对比表格
"""

import csv
import glob
from pathlib import Path
from collections import defaultdict
import sys

# ANSI颜色代码
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_result_files(pattern="results_*.csv"):
    """加载所有结果文件"""
    script_dir = Path(__file__).parent
    files = list(script_dir.glob(pattern))
    
    if not files:
        print(f"{Colors.FAIL}未找到任何结果文件（pattern: {pattern}）{Colors.ENDC}")
        return []
    
    print(f"{Colors.OKGREEN}找到 {len(files)} 个结果文件{Colors.ENDC}\n")
    return files


def analyze_single_file(filepath):
    """分析单个结果文件"""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    if not results:
        return None
    
    # 提取基本信息
    model = results[0].get('model', 'Unknown')
    mode = results[0].get('mode', 'Unknown')
    
    # 统计成功的测试
    successful = [r for r in results if r.get('status') == 'success']
    
    if not successful:
        return {
            'model': model,
            'mode': mode,
            'total_cases': len(results),
            'successful_cases': 0,
            'success_rate': 0.0
        }
    
    # 计算统计数据
    stats = {
        'model': model,
        'mode': mode,
        'total_cases': len(results),
        'successful_cases': len(successful),
        'success_rate': len(successful) / len(results) * 100
    }
    
    # 根据模式计算不同的指标
    if mode == 'extraction':
        # 信息元提取模式
        try:
            stats['avg_infons_ttft'] = sum(float(r.get('infons_ttft', 0)) for r in successful) / len(successful)
            stats['avg_infons_total'] = sum(float(r.get('infons_total_time', 0)) for r in successful) / len(successful)
            stats['avg_risk_ttft'] = sum(float(r.get('risk_ttft', 0)) for r in successful) / len(successful)
            stats['avg_risk_total'] = sum(float(r.get('risk_total_time', 0)) for r in successful) / len(successful)
            stats['avg_total_time'] = sum(float(r.get('total_time', 0)) for r in successful) / len(successful)
            
            # 计算最小值和最大值
            stats['min_total_time'] = min(float(r.get('total_time', 0)) for r in successful)
            stats['max_total_time'] = max(float(r.get('total_time', 0)) for r in successful)
        except (ValueError, ZeroDivisionError) as e:
            print(f"{Colors.WARNING}解析 {filepath.name} 时出错: {e}{Colors.ENDC}")
            return None
    else:
        # 直接推断模式
        try:
            stats['avg_risk_ttft'] = sum(float(r.get('risk_ttft', 0)) for r in successful) / len(successful)
            stats['avg_risk_total'] = sum(float(r.get('risk_total_time', 0)) for r in successful) / len(successful)
            stats['avg_total_time'] = stats['avg_risk_total']
            
            # 计算最小值和最大值
            stats['min_total_time'] = min(float(r.get('total_time', 0)) for r in successful)
            stats['max_total_time'] = max(float(r.get('total_time', 0)) for r in successful)
        except (ValueError, ZeroDivisionError) as e:
            print(f"{Colors.WARNING}解析 {filepath.name} 时出错: {e}{Colors.ENDC}")
            return None
    
    return stats


def print_detailed_report(stats_list):
    """打印详细报告"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*100}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'详细性能报告':^100}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*100}{Colors.ENDC}\n")
    
    # 按模式分组
    extraction_stats = [s for s in stats_list if s['mode'] == 'extraction']
    direct_stats = [s for s in stats_list if s['mode'] == 'direct']
    
    # 信息元提取模式报告
    if extraction_stats:
        print(f"{Colors.BOLD}{Colors.OKBLUE}信息元提取模式 (Information Extraction Mode){Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'-'*100}{Colors.ENDC}\n")
        
        # 按总耗时排序
        extraction_stats.sort(key=lambda x: x['avg_total_time'])
        
        print(f"{'排名':<6}{'模型':<40}{'成功率':<12}{'信息元TTFT':<15}{'风险分析TTFT':<15}{'总耗时':<12}")
        print(f"{'-'*100}")
        
        for i, stats in enumerate(extraction_stats, 1):
            model_short = stats['model'][:38]
            success_rate = f"{stats['success_rate']:.1f}%"
            infons_ttft = f"{stats.get('avg_infons_ttft', 0):.2f}s"
            risk_ttft = f"{stats.get('avg_risk_ttft', 0):.2f}s"
            total_time = f"{stats['avg_total_time']:.2f}s"
            
            # 对最快的3个模型高亮
            if i <= 3:
                color = Colors.OKGREEN
            else:
                color = ""
            
            print(f"{color}{i:<6}{model_short:<40}{success_rate:<12}{infons_ttft:<15}{risk_ttft:<15}{total_time:<12}{Colors.ENDC}")
        
        print()
    
    # 直接推断模式报告
    if direct_stats:
        print(f"{Colors.BOLD}{Colors.OKCYAN}直接推断模式 (Direct Inference Mode){Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'-'*100}{Colors.ENDC}\n")
        
        # 按总耗时排序
        direct_stats.sort(key=lambda x: x['avg_total_time'])
        
        print(f"{'排名':<6}{'模型':<40}{'成功率':<12}{'风险分析TTFT':<15}{'总耗时':<12}")
        print(f"{'-'*100}")
        
        for i, stats in enumerate(direct_stats, 1):
            model_short = stats['model'][:38]
            success_rate = f"{stats['success_rate']:.1f}%"
            risk_ttft = f"{stats.get('avg_risk_ttft', 0):.2f}s"
            total_time = f"{stats['avg_total_time']:.2f}s"
            
            # 对最快的3个模型高亮
            if i <= 3:
                color = Colors.OKGREEN
            else:
                color = ""
            
            print(f"{color}{i:<6}{model_short:<40}{success_rate:<12}{risk_ttft:<15}{total_time:<12}{Colors.ENDC}")
        
        print()


def print_comparison_table(stats_list):
    """打印模型对比表"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*100}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'模型性能对比表':^100}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*100}{Colors.ENDC}\n")
    
    # 按模型分组
    model_groups = defaultdict(dict)
    for stats in stats_list:
        model = stats['model']
        mode = stats['mode']
        model_groups[model][mode] = stats
    
    print(f"{'模型':<40}{'信息元模式总耗时':<20}{'直接推断模式总耗时':<20}{'差异':<15}")
    print(f"{'-'*100}")
    
    for model, modes in sorted(model_groups.items()):
        model_short = model[:38]
        
        extraction_time = modes.get('extraction', {}).get('avg_total_time', None)
        direct_time = modes.get('direct', {}).get('avg_total_time', None)
        
        extraction_str = f"{extraction_time:.2f}s" if extraction_time else "N/A"
        direct_str = f"{direct_time:.2f}s" if direct_time else "N/A"
        
        if extraction_time and direct_time:
            diff = extraction_time - direct_time
            diff_percent = (diff / direct_time) * 100
            diff_str = f"{diff:+.2f}s ({diff_percent:+.1f}%)"
            
            # 如果直接推断更快，标记为绿色
            if diff > 0:
                color = Colors.OKGREEN
            else:
                color = Colors.WARNING
        else:
            diff_str = "N/A"
            color = ""
        
        print(f"{model_short:<40}{extraction_str:<20}{direct_str:<20}{color}{diff_str:<15}{Colors.ENDC}")
    
    print()


def print_summary(stats_list):
    """打印总结"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*100}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'性能测试总结':^100}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*100}{Colors.ENDC}\n")
    
    # 找出最快的模型
    extraction_stats = [s for s in stats_list if s['mode'] == 'extraction']
    direct_stats = [s for s in stats_list if s['mode'] == 'direct']
    
    if extraction_stats:
        fastest_extraction = min(extraction_stats, key=lambda x: x['avg_total_time'])
        print(f"{Colors.OKGREEN}信息元提取模式最快模型:{Colors.ENDC}")
        print(f"  模型: {fastest_extraction['model']}")
        print(f"  平均总耗时: {fastest_extraction['avg_total_time']:.2f}秒")
        print(f"  信息元提取TTFT: {fastest_extraction.get('avg_infons_ttft', 0):.2f}秒")
        print(f"  风险分析TTFT: {fastest_extraction.get('avg_risk_ttft', 0):.2f}秒")
        print()
    
    if direct_stats:
        fastest_direct = min(direct_stats, key=lambda x: x['avg_total_time'])
        print(f"{Colors.OKGREEN}直接推断模式最快模型:{Colors.ENDC}")
        print(f"  模型: {fastest_direct['model']}")
        print(f"  平均总耗时: {fastest_direct['avg_total_time']:.2f}秒")
        print(f"  风险分析TTFT: {fastest_direct.get('avg_risk_ttft', 0):.2f}秒")
        print()
    
    # 响应速度（TTFT）最快的模型
    if extraction_stats:
        fastest_ttft_extraction = min(extraction_stats, key=lambda x: x.get('avg_infons_ttft', float('inf')))
        print(f"{Colors.OKCYAN}响应速度最快（信息元提取）:{Colors.ENDC}")
        print(f"  模型: {fastest_ttft_extraction['model']}")
        print(f"  首次响应时间: {fastest_ttft_extraction.get('avg_infons_ttft', 0):.2f}秒")
        print()
    
    if direct_stats:
        fastest_ttft_direct = min(direct_stats, key=lambda x: x.get('avg_risk_ttft', float('inf')))
        print(f"{Colors.OKCYAN}响应速度最快（直接推断）:{Colors.ENDC}")
        print(f"  模型: {fastest_ttft_direct['model']}")
        print(f"  首次响应时间: {fastest_ttft_direct.get('avg_risk_ttft', 0):.2f}秒")
        print()
    
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*100}{Colors.ENDC}\n")


def export_summary_csv(stats_list, output_path):
    """导出汇总CSV"""
    if not stats_list:
        return
    
    # 确定所有可能的字段
    all_fields = set()
    for stats in stats_list:
        all_fields.update(stats.keys())
    
    fieldnames = sorted(all_fields)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_list)
    
    print(f"{Colors.OKGREEN}汇总数据已导出到: {output_path}{Colors.ENDC}\n")


def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("="*100)
    print("                         测试结果分析工具")
    print("="*100)
    print(f"{Colors.ENDC}")
    
    # 加载结果文件
    result_files = load_result_files()
    
    if not result_files:
        return
    
    # 分析每个文件
    print(f"{Colors.OKCYAN}分析结果文件...{Colors.ENDC}\n")
    stats_list = []
    
    for filepath in result_files:
        print(f"  分析: {filepath.name}")
        stats = analyze_single_file(filepath)
        if stats:
            stats_list.append(stats)
    
    if not stats_list:
        print(f"{Colors.FAIL}没有可分析的数据{Colors.ENDC}")
        return
    
    # 打印报告
    print_detailed_report(stats_list)
    print_comparison_table(stats_list)
    print_summary(stats_list)
    
    # 导出汇总CSV
    script_dir = Path(__file__).parent
    summary_path = script_dir / "summary_statistics.csv"
    export_summary_csv(stats_list, summary_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}分析被用户中断{Colors.ENDC}")
        sys.exit(0)

