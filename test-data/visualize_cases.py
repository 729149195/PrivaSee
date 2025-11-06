#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试用例分布可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager, FontManager
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 直接指定中文字体文件路径
def setup_chinese_font():
    """设置中文字体，使用字体文件路径"""
    # 常见中文字体路径
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]
    
    # 查找第一个存在的字体文件
    for font_path in font_paths:
        if os.path.exists(font_path):
            print(f"✓ 使用字体文件: {font_path}")
            # 添加字体到matplotlib
            from matplotlib.font_manager import fontManager
            fontManager.addfont(font_path)
            # 获取字体名称
            font_prop = FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            return font_name, font_path
    
    print("⚠ 警告: 未找到中文字体文件")
    print("  建议安装: sudo apt-get install fonts-noto-cjk")
    return 'DejaVu Sans', None

# 配置中文字体
import matplotlib
font_name, font_path = setup_chinese_font()

# 设置matplotlib使用该字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [font_name]
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置专业风格
sns.set_style("whitegrid")
sns.set_palette("husl")
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 10

# 创建一个可复用的字体属性对象
if font_path:
    chinese_font_prop = FontProperties(fname=font_path)
else:
    chinese_font_prop = None

def load_data(filename='cases.csv'):
    """加载CSV数据"""
    df = pd.read_csv(filename)
    return df

def extract_category_prefix(category):
    """提取类别前缀（如 Category A）"""
    if ':' in category:
        return category.split(':')[0].strip()
    return category

def plot_category_distribution(df):
    """隐私类别分布"""
    df['category_prefix'] = df['category'].apply(extract_category_prefix)
    category_counts = df['category_prefix'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(category_counts)), category_counts.values, color='#2E86AB')
    ax.set_yticks(range(len(category_counts)))
    ax.set_yticklabels(category_counts.index, fontsize=9, fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Privacy Category Distribution', fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, (bar, value) in enumerate(zip(bars, category_counts.values)):
        ax.text(value + max(category_counts.values) * 0.01, i, str(value), 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('distribution_category.png', bbox_inches='tight')
    plt.close()
    print("✓ 生成: distribution_category.png")

def plot_difficulty_distribution(df):
    """难度级别分布"""
    difficulty_counts = df['difficulty'].value_counts()
    difficulty_order = ['direct', 'simple', 'complex']
    difficulty_counts = difficulty_counts.reindex([d for d in difficulty_order if d in difficulty_counts.index])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#06A77D', '#F77F00', '#D62828']
    wedges, texts, autotexts = ax.pie(difficulty_counts.values, 
                                        labels=difficulty_counts.index,
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        startangle=90,
                                        textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Difficulty Level Distribution', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('distribution_difficulty.png', bbox_inches='tight')
    plt.close()
    print("✓ 生成: distribution_difficulty.png")

def plot_language_distribution(df):
    """语言分布"""
    language_counts = df['language'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(language_counts)), language_counts.values, 
                   color=['#A23B72', '#F18F01', '#C73E1D'])
    ax.set_xticks(range(len(language_counts)))
    ax.set_xticklabels(language_counts.index, fontsize=10, fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Language Distribution', fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, value in zip(bars, language_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('distribution_language.png', bbox_inches='tight')
    plt.close()
    print("✓ 生成: distribution_language.png")

def plot_occupation_distribution(df, top_n=15):
    """职业分布（Top N）"""
    occupation_counts = df['occupation'].value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(occupation_counts)), occupation_counts.values, color='#6A4C93')
    ax.set_yticks(range(len(occupation_counts)))
    ax.set_yticklabels(occupation_counts.index, fontsize=9, fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Occupation Distribution', fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    
    for i, (bar, value) in enumerate(zip(bars, occupation_counts.values)):
        ax.text(value + max(occupation_counts.values) * 0.01, i, str(value), 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('distribution_occupation.png', bbox_inches='tight')
    plt.close()
    print("✓ 生成: distribution_occupation.png")

def plot_scenario_distribution(df):
    """场景分布"""
    scenario_counts = df['scenario'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(scenario_counts)), scenario_counts.values, color='#FF6B35')
    ax.set_yticks(range(len(scenario_counts)))
    ax.set_yticklabels(scenario_counts.index, fontsize=8, fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Scenario Distribution', fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, (bar, value) in enumerate(zip(bars, scenario_counts.values)):
        ax.text(value + max(scenario_counts.values) * 0.01, i, str(value), 
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('distribution_scenario.png', bbox_inches='tight')
    plt.close()
    print("✓ 生成: distribution_scenario.png")

def plot_inference_style_distribution(df):
    """推理风格分布"""
    style_counts = df['inference_style'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(style_counts)), style_counts.values, color='#4ECDC4')
    ax.set_xticks(range(len(style_counts)))
    ax.set_xticklabels(style_counts.index, fontsize=9, rotation=45, ha='right', fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Inference Style Distribution', fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, value in zip(bars, style_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('distribution_inference_style.png', bbox_inches='tight')
    plt.close()
    print("✓ 生成: distribution_inference_style.png")

def plot_combined_overview(df):
    """综合概览（2x2网格）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 难度分布
    difficulty_counts = df['difficulty'].value_counts()
    axes[0, 0].bar(range(len(difficulty_counts)), difficulty_counts.values, 
                   color=['#06A77D', '#F77F00', '#D62828'])
    axes[0, 0].set_xticks(range(len(difficulty_counts)))
    axes[0, 0].set_xticklabels(difficulty_counts.index, fontsize=9)
    axes[0, 0].set_title('Difficulty Distribution', fontsize=11, fontweight='bold')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    
    # 语言分布
    language_counts = df['language'].value_counts()
    axes[0, 1].bar(range(len(language_counts)), language_counts.values,
                   color=['#A23B72', '#F18F01', '#C73E1D'])
    axes[0, 1].set_xticks(range(len(language_counts)))
    labels_01 = axes[0, 1].set_xticklabels(language_counts.index, fontsize=9)
    if chinese_font_prop:
        for label in labels_01:
            label.set_fontproperties(chinese_font_prop)
    axes[0, 1].set_title('Language Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    
    # 类别分布（Top 8）
    df['category_prefix'] = df['category'].apply(extract_category_prefix)
    category_counts = df['category_prefix'].value_counts().head(8)
    axes[1, 0].barh(range(len(category_counts)), category_counts.values, color='#2E86AB')
    axes[1, 0].set_yticks(range(len(category_counts)))
    labels_10 = axes[1, 0].set_yticklabels(category_counts.index, fontsize=8)
    if chinese_font_prop:
        for label in labels_10:
            label.set_fontproperties(chinese_font_prop)
    axes[1, 0].set_title('Top 8 Privacy Categories', fontsize=11, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)
    
    # 职业分布（Top 8）
    occupation_counts = df['occupation'].value_counts().head(8)
    axes[1, 1].barh(range(len(occupation_counts)), occupation_counts.values, color='#6A4C93')
    axes[1, 1].set_yticks(range(len(occupation_counts)))
    labels_11 = axes[1, 1].set_yticklabels(occupation_counts.index, fontsize=8)
    if chinese_font_prop:
        for label in labels_11:
            label.set_fontproperties(chinese_font_prop)
    axes[1, 1].set_title('Top 8 Occupations', fontsize=11, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    
    plt.suptitle('Test Cases Distribution Overview', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('distribution_overview.png', bbox_inches='tight')
    plt.close()
    print("✓ 生成: distribution_overview.png")

def plot_text_length_distribution(df):
    """文本长度分布"""
    df['text_length'] = df['text'].str.len()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['text_length'], bins=30, color='#5B8C5A', edgecolor='black', alpha=0.7)
    ax.axvline(df['text_length'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["text_length"].mean():.0f}')
    ax.axvline(df['text_length'].median(), color='orange', linestyle='--', 
               linewidth=2, label=f'Median: {df["text_length"].median():.0f}')
    ax.set_xlabel('Text Length (characters)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Text Length Distribution', fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('distribution_text_length.png', bbox_inches='tight')
    plt.close()
    print("✓ 生成: distribution_text_length.png")

def generate_summary_stats(df):
    """生成统计摘要"""
    stats = {
        'Total Cases': len(df),
        'Avg Text Length': f"{df['text'].str.len().mean():.1f}",
        'Categories': df['category'].nunique(),
        'Occupations': df['occupation'].nunique(),
        'Languages': df['language'].nunique(),
        'Difficulties': df['difficulty'].nunique(),
    }
    
    print("\n" + "="*50)
    print("📊 统计摘要")
    print("="*50)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    print("="*50 + "\n")

def plot_all_in_one(df):
    """将所有图表组合到一张大图中"""
    # 创建大图，使用3x3网格布局
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 隐私类别分布 (左上，大)
    ax1 = fig.add_subplot(gs[0, :2])
    df['category_prefix'] = df['category'].apply(extract_category_prefix)
    category_counts = df['category_prefix'].value_counts().sort_index()
    bars1 = ax1.barh(range(len(category_counts)), category_counts.values, color='#2E86AB')
    ax1.set_yticks(range(len(category_counts)))
    ax1.set_yticklabels(category_counts.index, fontsize=9, fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax1.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Privacy Category Distribution', fontsize=13, fontweight='bold', pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for i, (bar, value) in enumerate(zip(bars1, category_counts.values)):
        ax1.text(value + max(category_counts.values) * 0.01, i, str(value), va='center', fontsize=8)
    
    # 2. 难度级别分布 (右上)
    ax2 = fig.add_subplot(gs[0, 2])
    difficulty_counts = df['difficulty'].value_counts()
    difficulty_order = ['direct', 'simple', 'complex']
    difficulty_counts = difficulty_counts.reindex([d for d in difficulty_order if d in difficulty_counts.index])
    colors2 = ['#06A77D', '#F77F00', '#D62828']
    wedges, texts, autotexts = ax2.pie(difficulty_counts.values, labels=difficulty_counts.index,
                                        autopct='%1.1f%%', colors=colors2, startangle=90,
                                        textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('Difficulty Distribution', fontsize=13, fontweight='bold', pad=15)
    
    # 3. 职业分布 Top 10 (中左)
    ax3 = fig.add_subplot(gs[1, :2])
    occupation_counts = df['occupation'].value_counts().head(10)
    bars3 = ax3.barh(range(len(occupation_counts)), occupation_counts.values, color='#6A4C93')
    ax3.set_yticks(range(len(occupation_counts)))
    ax3.set_yticklabels(occupation_counts.index, fontsize=9, fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax3.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Top 10 Occupation Distribution', fontsize=13, fontweight='bold', pad=15)
    ax3.invert_yaxis()
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    for i, (bar, value) in enumerate(zip(bars3, occupation_counts.values)):
        ax3.text(value + max(occupation_counts.values) * 0.01, i, str(value), va='center', fontsize=8)
    
    # 4. 语言分布 (中右)
    ax4 = fig.add_subplot(gs[1, 2])
    language_counts = df['language'].value_counts()
    bars4 = ax4.bar(range(len(language_counts)), language_counts.values,
                    color=['#A23B72', '#F18F01', '#C73E1D'])
    ax4.set_xticks(range(len(language_counts)))
    ax4.set_xticklabels(language_counts.index, fontsize=10, fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Language Distribution', fontsize=13, fontweight='bold', pad=15)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    for bar, value in zip(bars4, language_counts.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{int(value)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. 场景分布 (下左)
    ax5 = fig.add_subplot(gs[2, 0])
    scenario_counts = df['scenario'].value_counts().head(8)
    bars5 = ax5.barh(range(len(scenario_counts)), scenario_counts.values, color='#FF6B35')
    ax5.set_yticks(range(len(scenario_counts)))
    ax5.set_yticklabels(scenario_counts.index, fontsize=8, fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax5.set_xlabel('Count', fontsize=10, fontweight='bold')
    ax5.set_title('Top 8 Scenario Distribution', fontsize=12, fontweight='bold', pad=15)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    for i, (bar, value) in enumerate(zip(bars5, scenario_counts.values)):
        ax5.text(value + max(scenario_counts.values) * 0.01, i, str(value), va='center', fontsize=7)
    
    # 6. 推理风格分布 (下中)
    ax6 = fig.add_subplot(gs[2, 1])
    style_counts = df['inference_style'].value_counts()
    bars6 = ax6.bar(range(len(style_counts)), style_counts.values, color='#4ECDC4')
    ax6.set_xticks(range(len(style_counts)))
    ax6.set_xticklabels(style_counts.index, fontsize=8, rotation=45, ha='right',
                        fontproperties=chinese_font_prop if chinese_font_prop else None)
    ax6.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax6.set_title('Inference Style Distribution', fontsize=12, fontweight='bold', pad=15)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    for bar, value in zip(bars6, style_counts.values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height, f'{int(value)}',
                ha='center', va='bottom', fontsize=8)
    
    # 7. 文本长度分布 (下右)
    ax7 = fig.add_subplot(gs[2, 2])
    df['text_length'] = df['text'].str.len()
    ax7.hist(df['text_length'], bins=25, color='#5B8C5A', edgecolor='black', alpha=0.7)
    ax7.axvline(df['text_length'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["text_length"].mean():.0f}')
    ax7.axvline(df['text_length'].median(), color='orange', linestyle='--',
                linewidth=2, label=f'Median: {df["text_length"].median():.0f}')
    ax7.set_xlabel('Text Length (chars)', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax7.set_title('Text Length Distribution', fontsize=12, fontweight='bold', pad=15)
    ax7.legend(fontsize=9)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    
    # 添加总标题
    fig.suptitle('Test Cases Distribution Overview', fontsize=20, fontweight='bold', y=0.995)
    
    plt.savefig('distribution_all_in_one.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ 生成综合图表: distribution_all_in_one.png")

def main():
    print("\n" + "="*50)
    print("📊 测试用例分布可视化")
    print("="*50 + "\n")
    
    df = load_data('cases.csv')
    print(f"✓ 加载数据: {len(df)} 条记录\n")
    
    generate_summary_stats(df)
    
    print("生成图表中...\n")
    plot_all_in_one(df)
    
    print("\n" + "="*50)
    print("✅ 图表生成完成")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

