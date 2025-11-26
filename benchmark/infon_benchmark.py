#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息元提取 Benchmark 测试工具
测试不同 Ollama 模型使用 infons.js 提示词提取信息元的准确性
"""

import csv
import json
import time
import requests
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加父目录到path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.ace_to_infons import GoldSample, DescInfon, ScenInfon, RelInfon
from benchmark.evaluator import InfonEvaluator, parse_compact_format, print_evaluation_report


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# Ollama API配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# 可用的模型列表
AVAILABLE_MODELS = [
    "niels32167/qwen3-4b-instruct:latest",
    "qwen2.5:7b-instruct",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen3:14b",
    "gemma3:12b",
    "deepseek-r1:14b",
    "phi4-mini:latest",
    "llama3.2-vision:11b",
]

# JS模板目录
JS_TEMPLATES_DIR = None


def init_js_templates():
    """初始化JS模板路径"""
    global JS_TEMPLATES_DIR
    script_dir = Path(__file__).parent
    JS_TEMPLATES_DIR = script_dir.parent / 'frontend' / 'src' / 'templates'
    return JS_TEMPLATES_DIR.exists()


def load_js_template_string(js_file_path, export_name):
    """从JS文件中提取导出的模板字符串"""
    try:
        with open(js_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern_prefix = rf'export\s+const\s+{export_name}\s*=\s*(?:String\.raw)?`'
        match = re.search(pattern_prefix, content, re.DOTALL)
        
        if not match:
            return None
        
        start_pos = match.end() - 1
        i = start_pos + 1
        template_content = []
        
        while i < len(content):
            char = content[i]
            
            if char == '`':
                consecutive_backticks = 0
                j = i
                while j < len(content) and content[j] == '`':
                    consecutive_backticks += 1
                    j += 1
                
                if consecutive_backticks == 3:
                    template_content.append('```')
                    i = j
                    continue
                
                if consecutive_backticks == 1:
                    if i + 1 >= len(content) or content[i + 1] in ['\n', ';', ' ', '\r', ')']:
                        return ''.join(template_content)
                    else:
                        template_content.append(char)
                        i += 1
                        continue
                else:
                    template_content.append('`' * consecutive_backticks)
                    i = j
                    continue
            else:
                template_content.append(char)
                i += 1
        
        return None
    except Exception as e:
        print(f"{Colors.WARNING}加载模板失败: {e}{Colors.ENDC}")
        return None


def build_infons_extraction_prompt(text: str, round_num: int = 1) -> str:
    """构建信息元提取prompt"""
    global JS_TEMPLATES_DIR
    if not JS_TEMPLATES_DIR:
        init_js_templates()
    
    infons_js = JS_TEMPLATES_DIR / 'infons.js'
    if not infons_js.exists():
        print(f"{Colors.FAIL}找不到 {infons_js}{Colors.ENDC}")
        return f"Extract information from: {text}"
    
    # 读取模板
    core_def = load_js_template_string(infons_js, 'CORE_DEFINITION')
    ontology = load_js_template_string(infons_js, 'ONTOLOGY')
    output_constraints = load_js_template_string(infons_js, 'OUTPUT_CONSTRAINTS')
    output_format = load_js_template_string(infons_js, 'OUTPUT_FORMAT')
    text_extraction = load_js_template_string(infons_js, 'TEXT_EXTRACTION')
    self_checklist = load_js_template_string(infons_js, 'SELF_CHECKLIST')
    
    parts = [p for p in [core_def, ontology, output_constraints, output_format, text_extraction, self_checklist] if p]
    
    # 添加输入上下文
    context_info = f"""
【Current Extraction Context】
- Current conversation round: {round_num}
- Generate iid using format: "{{type_prefix}}:r{round_num}_{{index}}"

**CRITICAL REMINDER**: 
Extract information FROM THE USER INPUT BELOW.
DO NOT output example data. Extract ONLY what the user actually wrote.

USER INPUT:
{text}

NOW OUTPUT THE COMPACT FORMAT:"""
    
    parts.append(context_info)
    return '\n\n'.join(parts)


def call_ollama_api(model: str, prompt: str, timeout: int = 120) -> Tuple[str, float, float]:
    """
    调用Ollama API
    
    Returns:
        (response_text, ttft, total_time)
    """
    try:
        start_time = time.time()
        ttft = 0
        response_text = ""
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2048,
                }
            },
            stream=True,
            timeout=timeout
        )
        
        if response.status_code != 200:
            return f"ERROR: {response.status_code}", 0, 0
        
        first_token = True
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        if first_token:
                            ttft = time.time() - start_time
                            first_token = False
                        response_text += data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        total_time = time.time() - start_time
        return response_text, ttft, total_time
        
    except requests.exceptions.Timeout:
        return "ERROR: Timeout", 0, timeout
    except Exception as e:
        return f"ERROR: {str(e)}", 0, 0


def extract_infons_from_response(response: str) -> List[Dict]:
    """从模型响应中解析信息元"""
    # 清理响应：移除markdown标记等
    response = response.strip()
    
    # 移除可能的markdown代码块标记
    if response.startswith('```'):
        lines = response.split('\n')
        # 找到第一个和最后一个```
        start_idx = 0
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith('```') and i == 0:
                start_idx = 1
            elif line.strip() == '```':
                end_idx = i
                break
        response = '\n'.join(lines[start_idx:end_idx])
    
    return parse_compact_format(response)


def load_gold_samples(gold_path: str, limit: int = None) -> List[GoldSample]:
    """加载Gold标准数据"""
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    if limit:
        gold_data = gold_data[:limit]
    
    samples = []
    for item in gold_data:
        infons = []
        for inf_dict in item.get('infons', []):
            infon_type = inf_dict.get('infon_type', '').upper()
            if infon_type == 'DESC':
                infons.append(DescInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='DESC',
                    entity=inf_dict.get('entity', ''),
                    attribute=inf_dict.get('attribute', ''),
                    data_type=inf_dict.get('data_type', 'string'),
                    confidence=inf_dict.get('confidence', 1.0)
                ))
            elif infon_type == 'SCEN':
                infons.append(ScenInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='SCEN',
                    temporal=inf_dict.get('temporal', ''),
                    spatial=inf_dict.get('spatial', '')
                ))
            elif infon_type == 'REL':
                infons.append(RelInfon(
                    iid=inf_dict.get('iid', ''),
                    infon_type='REL',
                    relation_name=inf_dict.get('relation_name', ''),
                    arg_refs=inf_dict.get('arg_refs', [])
                ))
        
        sample = GoldSample(
            doc_id=item.get('doc_id', ''),
            text=item.get('text', ''),
            infons=infons,
            language=item.get('language', '')
        )
        samples.append(sample)
    
    return samples


def test_single_sample(model: str, sample: GoldSample, sample_idx: int, total: int) -> Dict:
    """测试单个样本"""
    # 截断过长文本
    text = sample.text
    if len(text) > 3000:
        text = text[:3000] + "..."
    
    # 构建prompt
    prompt = build_infons_extraction_prompt(text)
    
    # 调用API
    response, ttft, total_time = call_ollama_api(model, prompt)
    
    # 解析结果
    if response.startswith("ERROR"):
        return {
            'doc_id': sample.doc_id,
            'status': 'error',
            'error': response,
            'ttft': ttft,
            'total_time': total_time,
            'gold_count': len(sample.infons),
            'pred_count': 0,
            'predictions': []
        }
    
    predictions = extract_infons_from_response(response)
    
    return {
        'doc_id': sample.doc_id,
        'language': sample.language,
        'status': 'success',
        'ttft': ttft,
        'total_time': total_time,
        'gold_count': len(sample.infons),
        'pred_count': len(predictions),
        'predictions': predictions,
        'raw_response': response[:500]  # 保存部分原始响应用于调试
    }


def run_benchmark(model: str, gold_samples: List[GoldSample], output_dir: Path) -> Dict:
    """运行benchmark测试"""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}测试模型: {model}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    
    results = []
    predictions_list = []
    
    for i, sample in enumerate(gold_samples):
        print(f"{Colors.OKCYAN}[{i+1}/{len(gold_samples)}]{Colors.ENDC} {sample.doc_id} ({sample.language})...", end=" ", flush=True)
        
        result = test_single_sample(model, sample, i, len(gold_samples))
        results.append(result)
        predictions_list.append(result.get('predictions', []))
        
        if result['status'] == 'success':
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} TTFT={result['ttft']:.2f}s, Total={result['total_time']:.2f}s, Pred={result['pred_count']}/{result['gold_count']}")
        else:
            print(f"{Colors.FAIL}✗{Colors.ENDC} {result.get('error', 'Unknown error')}")
        
        # 每5个样本休息一下
        if (i + 1) % 5 == 0 and i + 1 < len(gold_samples):
            time.sleep(1)
    
    # 评估
    print(f"\n{Colors.OKBLUE}评估中...{Colors.ENDC}")
    evaluator = InfonEvaluator()
    eval_result = evaluator.evaluate_batch(gold_samples, predictions_list)
    
    # 打印评估报告
    print_evaluation_report(eval_result, detailed=False)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace('/', '_').replace(':', '_')
    
    # 保存详细结果
    result_path = output_dir / f"benchmark_{model_safe}_{timestamp}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': model,
            'timestamp': timestamp,
            'num_samples': len(gold_samples),
            'evaluation': eval_result,
            'details': [{
                'doc_id': r['doc_id'],
                'status': r['status'],
                'ttft': r.get('ttft', 0),
                'total_time': r.get('total_time', 0),
                'gold_count': r.get('gold_count', 0),
                'pred_count': r.get('pred_count', 0),
            } for r in results]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{Colors.OKGREEN}结果已保存: {result_path}{Colors.ENDC}")
    
    # 计算性能统计
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        avg_ttft = sum(r['ttft'] for r in successful) / len(successful)
        avg_total = sum(r['total_time'] for r in successful) / len(successful)
        print(f"\n{Colors.BOLD}性能统计:{Colors.ENDC}")
        print(f"  成功率: {len(successful)}/{len(results)}")
        print(f"  平均TTFT: {avg_ttft:.2f}s")
        print(f"  平均总耗时: {avg_total:.2f}s")
    
    return eval_result


def select_models() -> List[str]:
    """让用户选择模型"""
    print(f"\n{Colors.BOLD}可用的Ollama模型:{Colors.ENDC}")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {model}")
    
    print(f"\n请选择要测试的模型（输入数字，逗号分隔，或 'all'）:")
    choice = input("> ").strip()
    
    if choice.lower() == 'all':
        return AVAILABLE_MODELS
    
    try:
        indices = [int(x.strip()) for x in choice.split(',')]
        return [AVAILABLE_MODELS[i-1] for i in indices if 1 <= i <= len(AVAILABLE_MODELS)]
    except:
        print(f"{Colors.WARNING}无效选择，使用第一个模型{Colors.ENDC}")
        return [AVAILABLE_MODELS[0]]


def main():
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("="*70)
    print("        信息元提取 Benchmark 测试工具")
    print("        (ACE 2005 → PrivaSee Infons)")
    print("="*70)
    print(f"{Colors.ENDC}")
    
    # 初始化
    if not init_js_templates():
        print(f"{Colors.FAIL}错误: 找不到JS模板目录{Colors.ENDC}")
        return
    print(f"{Colors.OKGREEN}✓ 已加载JS模板{Colors.ENDC}")
    
    # 加载Gold数据
    script_dir = Path(__file__).parent
    gold_path = script_dir / 'gold_data' / 'gold.json'
    
    if not gold_path.exists():
        print(f"{Colors.FAIL}错误: 找不到Gold数据 {gold_path}{Colors.ENDC}")
        print(f"请先运行: python -m benchmark.run_benchmark convert ...")
        return
    
    print(f"\n请输入要测试的样本数量（默认10，输入0表示全部）:")
    limit_input = input("> ").strip()
    limit = int(limit_input) if limit_input.isdigit() and int(limit_input) > 0 else 10
    if limit_input == '0':
        limit = None
    
    gold_samples = load_gold_samples(str(gold_path), limit)
    print(f"{Colors.OKGREEN}✓ 加载了 {len(gold_samples)} 个Gold样本{Colors.ENDC}")
    
    # 显示语言分布
    langs = {}
    for s in gold_samples:
        langs[s.language] = langs.get(s.language, 0) + 1
    print(f"  语言分布: {langs}")
    
    # 选择模型
    selected_models = select_models()
    if not selected_models:
        print(f"{Colors.FAIL}没有选择任何模型{Colors.ENDC}")
        return
    
    # 创建输出目录
    output_dir = script_dir / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # 运行测试
    all_results = {}
    for model in selected_models:
        try:
            result = run_benchmark(model, gold_samples, output_dir)
            all_results[model] = result
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}测试被中断{Colors.ENDC}")
            break
        except Exception as e:
            print(f"{Colors.FAIL}模型 {model} 测试失败: {e}{Colors.ENDC}")
            continue
    
    # 汇总对比
    if len(all_results) > 1:
        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}模型对比汇总{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"\n{'模型':<35} {'P':>8} {'R':>8} {'F1':>8}")
        print("-" * 60)
        for model, result in all_results.items():
            o = result['overall']
            print(f"{model:<35} {o['precision']:>8.4f} {o['recall']:>8.4f} {o['f1']:>8.4f}")
    
    print(f"\n{Colors.OKGREEN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}测试完成！{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{'='*70}{Colors.ENDC}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}测试被用户中断{Colors.ENDC}")
