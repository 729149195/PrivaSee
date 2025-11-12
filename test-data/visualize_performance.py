#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能可视化脚本
将测试结果生成可视化图表（需要安装 matplotlib 和 pandas）
"""

import csv
from pathlib import Path
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


def check_dependencies():
    """检查是否安装了必要的依赖"""
    try:
        import matplotlib
        import pandas
        return True
    except ImportError:
        return False


def load_summary_data(csv_path):
    """加载汇总数据"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def create_bar_chart(data, output_dir):
    """创建性能对比柱状图"""
    import matplotlib.pyplot as plt
    import matplotlib
    
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 按模式分组
    extraction_data = [d for d in data if d['mode'] == 'extraction']
    direct_data = [d for d in data if d['mode'] == 'direct']
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图表1: 信息元提取模式
    if extraction_data:
        models = [d['model'].split(':')[0][-20:] for d in extraction_data]  # 简化模型名
        times = [float(d['avg_total_time']) for d in extraction_data]
        
        # 排序
        sorted_pairs = sorted(zip(models, times), key=lambda x: x[1])
        models, times = zip(*sorted_pairs)
        
        bars = axes[0].barh(models, times, color='skyblue')
        axes[0].set_xlabel('平均总耗时 (秒)', fontsize=12)
        axes[0].set_title('信息元提取模式 - 性能对比', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            axes[0].text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}s', ha='left', va='center', fontsize=9)
    
    # 图表2: 直接推断模式
    if direct_data:
        models = [d['model'].split(':')[0][-20:] for d in direct_data]
        times = [float(d['avg_total_time']) for d in direct_data]
        
        # 排序
        sorted_pairs = sorted(zip(models, times), key=lambda x: x[1])
        models, times = zip(*sorted_pairs)
        
        bars = axes[1].barh(models, times, color='lightcoral')
        axes[1].set_xlabel('平均总耗时 (秒)', fontsize=12)
        axes[1].set_title('直接推断模式 - 性能对比', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            axes[1].text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}s', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"{Colors.OKGREEN}✓ 性能对比图已保存: {output_path}{Colors.ENDC}")
    
    plt.close()


def create_ttft_chart(data, output_dir):
    """创建首次响应时间对比图"""
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    extraction_data = [d for d in data if d['mode'] == 'extraction' and 'avg_risk_ttft' in d]
    
    if not extraction_data:
        print(f"{Colors.WARNING}没有足够的数据生成 TTFT 对比图{Colors.ENDC}")
        return
    
    # 排序
    extraction_data.sort(key=lambda x: float(x.get('avg_infons_ttft', 0)))
    
    models = [d['model'].split(':')[0][-20:] for d in extraction_data]
    infons_ttft = [float(d.get('avg_infons_ttft', 0)) for d in extraction_data]
    risk_ttft = [float(d.get('avg_risk_ttft', 0)) for d in extraction_data]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, infons_ttft, width, label='信息元提取 TTFT', color='lightblue')
    bars2 = ax.bar(x + width/2, risk_ttft, width, label='风险分析 TTFT', color='lightgreen')
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('首次响应时间 (秒)', fontsize=12)
    ax.set_title('首次响应时间 (TTFT) 对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / 'ttft_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"{Colors.OKGREEN}✓ TTFT对比图已保存: {output_path}{Colors.ENDC}")
    
    plt.close()


def create_mode_comparison(data, output_dir):
    """创建两种模式的对比图"""
    import matplotlib.pyplot as plt
    import matplotlib
    
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 按模型分组
    model_dict = {}
    for d in data:
        model = d['model']
        mode = d['mode']
        if model not in model_dict:
            model_dict[model] = {}
        model_dict[model][mode] = float(d.get('avg_total_time', 0))
    
    # 只保留两种模式都有数据的模型
    filtered_models = {k: v for k, v in model_dict.items() 
                       if 'extraction' in v and 'direct' in v}
    
    if not filtered_models:
        print(f"{Colors.WARNING}没有足够的数据生成模式对比图{Colors.ENDC}")
        return
    
    # 准备数据
    models = [m.split(':')[0][-20:] for m in filtered_models.keys()]
    extraction_times = [v['extraction'] for v in filtered_models.values()]
    direct_times = [v['direct'] for v in filtered_models.values()]
    
    # 按提取模式时间排序
    sorted_data = sorted(zip(models, extraction_times, direct_times), 
                        key=lambda x: x[1])
    models, extraction_times, direct_times = zip(*sorted_data)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = range(len(models))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], extraction_times, width, 
                   label='信息元提取模式', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], direct_times, width, 
                   label='直接推断模式', color='coral')
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('平均总耗时 (秒)', fontsize=12)
    ax.set_title('两种推理模式性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / 'mode_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"{Colors.OKGREEN}✓ 模式对比图已保存: {output_path}{Colors.ENDC}")
    
    plt.close()


def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("="*70)
    print("                  性能可视化工具")
    print("="*70)
    print(f"{Colors.ENDC}\n")
    
    # 检查依赖
    print(f"{Colors.OKCYAN}检查依赖...{Colors.ENDC}", end=" ")
    if not check_dependencies():
        print(f"{Colors.FAIL}✗{Colors.ENDC}")
        print(f"\n{Colors.WARNING}需要安装以下依赖:{Colors.ENDC}")
        print("  pip install matplotlib pandas")
        print("\n或者使用 conda:")
        print("  conda install matplotlib pandas")
        return
    print(f"{Colors.OKGREEN}✓{Colors.ENDC}")
    
    # 查找汇总文件
    script_dir = Path(__file__).parent
    summary_path = script_dir / 'summary_statistics.csv'
    
    print(f"{Colors.OKCYAN}查找汇总数据...{Colors.ENDC}", end=" ")
    if not summary_path.exists():
        print(f"{Colors.FAIL}✗{Colors.ENDC}")
        print(f"\n{Colors.WARNING}未找到汇总文件: {summary_path}{Colors.ENDC}")
        print("请先运行分析脚本:")
        print("  python3 analyze_results.py")
        return
    print(f"{Colors.OKGREEN}✓{Colors.ENDC}")
    
    # 加载数据
    print(f"{Colors.OKCYAN}加载数据...{Colors.ENDC}", end=" ")
    try:
        data = load_summary_data(summary_path)
        print(f"{Colors.OKGREEN}✓ ({len(data)} 条记录){Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}✗ {e}{Colors.ENDC}")
        return
    
    if not data:
        print(f"{Colors.FAIL}没有可用的数据{Colors.ENDC}")
        return
    
    # 创建输出目录
    output_dir = script_dir / 'charts'
    output_dir.mkdir(exist_ok=True)
    
    # 生成图表
    print(f"\n{Colors.BOLD}生成图表...{Colors.ENDC}\n")
    
    try:
        create_bar_chart(data, output_dir)
        create_ttft_chart(data, output_dir)
        create_mode_comparison(data, output_dir)
    except Exception as e:
        print(f"{Colors.FAIL}生成图表时出错: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return
    
    # 完成
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}所有图表已生成！{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}{'='*70}{Colors.ENDC}\n")
    
    print(f"图表保存在: {Colors.BOLD}{output_dir}{Colors.ENDC}")
    print(f"\n生成的图表:")
    print(f"  1. performance_comparison.png - 性能对比图")
    print(f"  2. ttft_comparison.png - TTFT对比图")
    print(f"  3. mode_comparison.png - 模式对比图")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}操作被用户中断{Colors.ENDC}")
        sys.exit(0)

