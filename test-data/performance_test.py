#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本 - 测试不同Ollama模型在隐私推理中的性能
测试两种模式：
1. 信息元提取模式（Information Extraction Mode）
2. 直接推断模式（Direct Inference Mode）

从前端 JS 文件直接导入提示词，保持同步。
"""

import csv
import json
import time
import requests
from datetime import datetime
from pathlib import Path
import os
import sys
import re

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

# Ollama API配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# 可用的模型列表（从ollama list获取）
AVAILABLE_MODELS = [
    "niels32167/qwen3-4b-instruct:latest",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:7b-instruct",
    "phi4-mini:latest",
    "gemma3:12b",
    "deepseek-r1:14b",
    "qwen2.5vl:7b",
    "qwen3:14b",
    "llama3.2-vision:11b",
    "gpt-oss:20b"
]

# JS文件路径
JS_TEMPLATES_DIR = None


def load_js_template_string(js_file_path, export_name):
    """
    从JS文件中提取导出的模板字符串
    支持两种格式：
    1. export const NAME = `...`
    2. export const NAME = String.raw`...`
    
    使用手动解析处理嵌套的反引号（在 ```...``` 代码块中）
    """
    try:
        with open(js_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找 export const EXPORT_NAME = 
        pattern_prefix = rf'export\s+const\s+{export_name}\s*=\s*(?:String\.raw)?`'
        match = re.search(pattern_prefix, content, re.DOTALL)
        
        if not match:
            return None
        
        # 从第一个反引号开始解析
        start_pos = match.end() - 1  # 反引号位置
        
        # 手动解析，处理嵌套的反引号
        i = start_pos + 1  # 跳过开始的反引号
        template_content = []
        backtick_depth = 0  # 跟踪连续反引号数量（用于 ```代码块```）
        
        while i < len(content):
            char = content[i]
            
            if char == '`':
                # 检查连续的反引号数量
                consecutive_backticks = 0
                j = i
                while j < len(content) and content[j] == '`':
                    consecutive_backticks += 1
                    j += 1
                
                # 如果是3个反引号，这是代码块标记
                if consecutive_backticks == 3:
                    template_content.append('```')
                    i = j
                    continue
                
                # 单个反引号 - 可能是模板字符串结束
                if consecutive_backticks == 1:
                    # 检查下一个字符，确保这是结束标记
                    if i + 1 >= len(content) or content[i + 1] in ['\n', ';', ' ', '\r', ')']:
                        # 模板字符串结束
                        return ''.join(template_content)
                    else:
                        # 模板表达式中的反引号，继续
                        template_content.append(char)
                        i += 1
                        continue
                else:
                    # 多个反引号但不是3个，添加所有
                    template_content.append('`' * consecutive_backticks)
                    i = j
                    continue
            else:
                template_content.append(char)
                i += 1
        
        # 如果到达文件末尾还没找到结束，返回None
        return None
    except Exception as e:
        print(f"{Colors.WARNING}加载 {js_file_path} 中的 {export_name} 失败: {e}{Colors.ENDC}")
        return None


def extract_js_function_body(js_file_path, function_name):
    """
    从JS文件中提取函数体（用于理解逻辑）
    这是一个简化版本，仅用于提取关键部分
    """
    try:
        with open(js_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 匹配函数定义
        pattern = rf'export\s+function\s+{function_name}\s*\([^)]*\)\s*\{{(.*?)\n\}}\s*(?:export|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return match.group(1)
        
        return None
    except Exception as e:
        print(f"{Colors.WARNING}提取 {js_file_path} 中的函数 {function_name} 失败: {e}{Colors.ENDC}")
        return None


def init_js_templates():
    """初始化JS模板路径"""
    global JS_TEMPLATES_DIR
    script_dir = Path(__file__).parent
    JS_TEMPLATES_DIR = script_dir.parent / 'frontend' / 'src' / 'templates'
    
    if not JS_TEMPLATES_DIR.exists():
        print(f"{Colors.WARNING}警告：找不到JS模板目录 {JS_TEMPLATES_DIR}{Colors.ENDC}")
        return False
    
    return True

# 法律树结构（从PIPL.json加载）
def load_pipl_law_tree():
    """加载PIPL.json法律树结构"""
    # 尝试从多个可能的路径加载
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent / 'frontend' / 'law' / 'PIPL.json',  # 相对于test-data的路径
        Path('/home/zhangxiangxuan/桌面/Projects/PrivaSee/frontend/law/PIPL.json'),  # 绝对路径
    ]
    
    for pipl_path in possible_paths:
        if pipl_path.exists():
            try:
                with open(pipl_path, 'r', encoding='utf-8') as f:
                    law_data = json.load(f)
                return extract_law_tree_text(law_data)
            except Exception as e:
                print(f"{Colors.WARNING}加载 {pipl_path} 失败: {e}{Colors.ENDC}")
                continue
    
    # 如果加载失败，返回简化版本
    print(f"{Colors.WARNING}未能加载 PIPL.json，使用简化版本{Colors.ENDC}")
    return """Privacy Categories (13 items):
姓名, 年龄, 性别, 身份证号, 联系方式, 住址, 位置信息, 医疗健康, 财务信息, 职业信息, 教育信息, 社交关系, 行为偏好"""


def extract_law_tree_text(law_data, indent=0):
    """
    递归提取法律树的文本表示，只提取叶子节点（用于LLM推理）
    这与前端的 extractLawTreeSummary 函数保持一致
    """
    if not law_data:
        return 'No legal structure available'
    
    leaf_nodes = []
    
    def collect_leaf_nodes(node, path=""):
        """递归收集所有叶子节点"""
        node_name = node.get('name', '')
        current_path = f"{path} > {node_name}" if path else node_name
        
        children = node.get('children', [])
        is_leaf = not children or len(children) == 0
        
        if is_leaf and node_name:
            # 叶子节点，添加到列表
            leaf_nodes.append(node_name)
        elif children:
            # 中间节点，继续递归
            for child in children:
                collect_leaf_nodes(child, current_path)
    
    collect_leaf_nodes(law_data)
    
    # 返回简洁的叶子节点列表（与inference.js中的格式一致）
    if leaf_nodes:
        return f"Privacy Categories ({len(leaf_nodes)} items):\n" + ", ".join(leaf_nodes)
    else:
        return "No privacy categories available"


# 加载法律树（在脚本启动时加载一次）
LAW_TREE_TEXT = None


def build_infons_extraction_prompt(text, round_num=1):
    """
    构建信息元提取prompt（从infons.js导入）
    """
    global JS_TEMPLATES_DIR
    if not JS_TEMPLATES_DIR:
        init_js_templates()
    
    infons_js = JS_TEMPLATES_DIR / 'infons.js'
    if not infons_js.exists():
        print(f"{Colors.FAIL}错误：找不到 {infons_js}{Colors.ENDC}")
        return f"Extract information from: {text}"
    
    # 读取infons.js中的各个部分
    core_def = load_js_template_string(infons_js, 'CORE_DEFINITION')
    ontology = load_js_template_string(infons_js, 'ONTOLOGY')
    output_constraints = load_js_template_string(infons_js, 'OUTPUT_CONSTRAINTS')
    output_format = load_js_template_string(infons_js, 'OUTPUT_FORMAT')
    text_extraction = load_js_template_string(infons_js, 'TEXT_EXTRACTION')
    self_checklist = load_js_template_string(infons_js, 'SELF_CHECKLIST')
    
    # 组装提示词（模拟buildSystemPrompt函数）
    parts = []
    if core_def:
        parts.append(core_def)
    if ontology:
        parts.append(ontology)
    if output_constraints:
        parts.append(output_constraints)
    if output_format:
        parts.append(output_format)
    if text_extraction:
        parts.append(text_extraction)
    if self_checklist:
        parts.append(self_checklist)
    
    # 添加轮次上下文
    context_info = f"""
【Current Extraction Context】
- Current conversation round: {round_num}
- Generate iid using format: "{{type_prefix}}:r{round_num}_{{index}}" (index starts from 1 for this extraction)

**CRITICAL REMINDER**: 
You are about to receive the USER'S ACTUAL INPUT below. 
Your task is to READ IT CAREFULLY and extract information FROM IT.
DO NOT output example data. DO NOT make up information.
Extract ONLY what the user actually wrote/showed.

USER INPUT:
{text}

NOW OUTPUT THE COMPACT FORMAT:"""
    
    parts.append(context_info)
    
    return '\n\n'.join(parts)


def build_privacy_analysis_prompt(infons_summary, law_tree, is_direct=False, direct_text=None):
    """
    构建隐私风险分析prompt（从inference.js的fillPromptTemplate逻辑导入）
    模拟 inference.js 中的 fillPromptTemplate 函数
    """
    global JS_TEMPLATES_DIR
    if not JS_TEMPLATES_DIR:
        init_js_templates()
    
    # 法律树摘要
    law_tree_summary = law_tree or "No legal framework available"
    
    if is_direct and direct_text:
        # 直接推断模式 (对应 fillPromptTemplate 的 directInput 分支)
        prompt = f"""You are a privacy risk analyzer. Analyze the COMPLETE user conversation below and identify ALL privacy risks by considering cross-message correlations.

USER INPUT (single message):
{direct_text}

LEGAL FRAMEWORK:
{law_tree_summary}

CRITICAL TASK REQUIREMENTS:
1. ANALYZE ALL MESSAGES TOGETHER: Information from different messages may combine to create privacy risks
   - Example: If message 1 mentions "name" and message 2 mentions "address", this is HIGH risk (full identity)
   - Example: If message 1 mentions "hospital visit" and message 2 mentions "medication", infer health condition
2. EXTRACT KEYWORDS FROM ALL MESSAGES: Don't just focus on the last message
3. IDENTIFY CROSS-MESSAGE PATTERNS: Look for information that connects across messages
4. MAP EACH RISK to the most specific legal clause name
5. Output ONLY compact format (no JSON, no markdown, no extra text, NO header line)
6. **ONE RISK PER LINE**: Each risk MUST be on a separate line (press Enter after each risk)

OUTPUT FORMAT (COMPACT - NO HEADER, direct data output, ONE RISK PER LINE):
value1,value2,value3,value4,value5
value1,value2,value3,value4,value5

FIELD DEFINITIONS:
- law_node_name: exact leaf node name from legal framework (NO translation, NO abbreviation)
- risk_level: HIGH | MEDIUM | LOW (based on INFERENCE CERTAINTY)
- privacy_exposure: what privacy info is exposed (consider information from ALL messages)
- inference_chain: reasoning - what data appears, how they connect, what can be inferred, confidence level
- used_infons: information elements in format "TYPE:VALUE" separated by |
  * **DESC format**: DESC:attribute_value (ONLY attribute, NOT "entity:attribute")
  * **SCEN format**: SCEN:temporal@spatial
  * **REL format**: REL:relation_name
  * Example: DESC:Klook|DESC:台北|SCEN:下周@东京
  * Extract concrete keywords from input, NOT inferences

CRITICAL RULES:
- Output ONLY the compact format, no other text
- **ONE RISK PER LINE**: Each complete risk entry MUST be on its own line (use real line breaks between risks)
- Escape commas in text with \\,, newlines with \\n, backslashes with \\\\
- law_node_name MUST be exact copy from legal framework above (NO translation, NO abbreviation)
- used_infons format: "TYPE:VALUE" separated by | (e.g., DESC:Klook|DESC:台北|SCEN:下周@东京)
- Deep inference: infer health conditions, beliefs from behaviors across messages
- RISK LEVEL ASSIGNMENT: Evaluate inference CERTAINTY based on data clarity and context
- CROSS-MESSAGE ANALYSIS: A single privacy risk may be supported by keywords from multiple messages
- Sort by risk_level: HIGH first
- LANGUAGE CONSISTENCY: Write privacy_exposure and inference_chain in the SAME language as the input

**FINAL FORMAT CHECK BEFORE OUTPUT**:
Each line MUST have EXACTLY 5 fields in this order:
1. law_node_name (copy from legal framework)
2. risk_level (HIGH or MEDIUM or LOW)
3. privacy_exposure (what is exposed)
4. inference_chain (reasoning)
5. used_infons (format: TYPE:VALUE separated by |)

NOW OUTPUT THE COMPACT FORMAT:"""
    else:
        # 信息元提取模式 (对应 fillPromptTemplate 的默认分支)
        prompt = f"""You are a privacy risk analyzer. Analyze the information elements below and identify privacy risks.

INPUT DATA TO ANALYZE:
{infons_summary}

LEGAL FRAMEWORK:
{law_tree_summary}

TASK:
1. Identify what privacy information can be inferred from the input data
2. Map each privacy risk to the most specific legal clause name
3. Output ONLY compact format (no JSON, no markdown, no extra text, NO header line)
4. **ONE RISK PER LINE**: Each risk MUST be on a separate line (press Enter after each risk)

OUTPUT FORMAT (COMPACT - NO HEADER, direct data output, ONE RISK PER LINE):
value1,value2,value3,value4,value5
value1,value2,value3,value4,value5

FIELD DEFINITIONS:
- law_node_name: exact leaf node name from legal framework (NO translation, NO abbreviation)
- risk_level: HIGH | MEDIUM | LOW (based on INFERENCE CERTAINTY)
- privacy_exposure: what privacy info is exposed
- inference_chain: reasoning - what data shows, what it implies, why it matters, confidence level
- used_infons: information elements in format "TYPE:VALUE" separated by |
  * **DESC format**: DESC:attribute_value (ONLY attribute, NOT "entity:attribute")
  * **SCEN format**: SCEN:temporal@spatial
  * **REL format**: REL:relation_name
  * Example: DESC:Klook|DESC:台北|SCEN:下周@东京
  * Extract concrete keywords from input, NOT inferences

CRITICAL RULES:
- Output ONLY the compact format, no other text
- **ONE RISK PER LINE**: Each complete risk entry MUST be on its own line (use real line breaks between risks)
- Escape commas in text with \\,, newlines with \\n, backslashes with \\\\
- law_node_name MUST be exact copy from legal framework above (NO translation, NO abbreviation)
- used_infons format: "TYPE:VALUE" separated by |
  * DESC: Extract ONLY attribute value (e.g., "DESC: 平台: Klook" → "DESC:Klook")
  * SCEN: Combine temporal@spatial (e.g., "SCEN: 下周 @ 东京" → "SCEN:下周@东京")
  * REL: Use relation name (e.g., "REL: 住宿预订" → "REL:住宿预订")
- Deep inference: infer health conditions, beliefs from behaviors (e.g., gluten-free → celiac disease)
- RISK LEVEL ASSIGNMENT: Evaluate inference CERTAINTY based on data clarity and context
- Sort by risk_level: HIGH first
- LANGUAGE CONSISTENCY: Write privacy_exposure and inference_chain in the SAME language as the input data

**FINAL FORMAT CHECK BEFORE OUTPUT**:
Each line MUST have EXACTLY 5 fields in this order:
1. law_node_name (copy from legal framework, NO translation, NO abbreviation)
2. risk_level (HIGH or MEDIUM or LOW)
3. privacy_exposure (what is exposed)
4. inference_chain (reasoning)
5. used_infons (format: TYPE:VALUE separated by |)

NOW OUTPUT THE COMPACT FORMAT:"""
    
    return prompt


def call_ollama_stream(model, prompt):
    """
    调用Ollama API进行流式推理
    返回：(首次输出时间, 结束输出时间, 生成的文本, 是否成功)
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    try:
        start_time = time.time()
        first_token_time = None
        response_text = ""
        
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=300)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    chunk = data['response']
                    if chunk and first_token_time is None:
                        first_token_time = time.time()
                    response_text += chunk
                
                if data.get('done', False):
                    break
        
        end_time = time.time()
        
        if first_token_time is None:
            first_token_time = end_time
        
        return first_token_time - start_time, end_time - start_time, response_text, True
        
    except Exception as e:
        print(f"{Colors.FAIL}错误: {str(e)}{Colors.ENDC}")
        return None, None, None, False


def parse_infons_from_output(output_text):
    """从输出中解析信息元并生成摘要"""
    # 简化的解析逻辑，提取关键信息
    lines = output_text.strip().split('\n')
    infons_list = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('infons[') and not line.startswith('#'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                # 格式：iid, infon_type, entity, attribute, ...
                infons_list.append(f"- [{parts[0]}] {parts[1]}: {parts[2]}: {parts[3]}")
    
    if infons_list:
        return '\n'.join(infons_list[:20])  # 最多取20条
    return "No information elements extracted"


def test_single_case(model, case_data, mode='extraction', law_tree_text=None):
    """
    测试单个案例
    mode: 'extraction' (信息元提取模式) 或 'direct' (直接推断模式)
    law_tree_text: 法律树文本（从PIPL.json加载）
    """
    case_id = case_data['id']
    text = case_data['text']
    
    result = {
        'case_id': case_id,
        'model': model,
        'mode': mode,
        'text_length': len(text),
        'category': case_data.get('category', ''),
        'language': case_data.get('language', ''),
    }
    
    # 使用传入的法律树文本
    law_tree = law_tree_text or "No legal framework available"
    
    try:
        if mode == 'extraction':
            # 步骤1: 提取信息元
            print(f"  {Colors.OKCYAN}→ 提取信息元...{Colors.ENDC}", end=' ')
            infons_prompt = build_infons_extraction_prompt(text)
            ttft_infons, total_time_infons, infons_output, success = call_ollama_stream(model, infons_prompt)
            
            if not success:
                print(f"{Colors.FAIL}✗{Colors.ENDC}")
                result['status'] = 'failed_infons'
                return result
            
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} (TTFT: {ttft_infons:.2f}s, 总计: {total_time_infons:.2f}s)")
            
            result['infons_ttft'] = ttft_infons
            result['infons_total_time'] = total_time_infons
            result['infons_length'] = len(infons_output)
            
            # 步骤2: 使用提取的信息元进行隐私风险分析
            print(f"  {Colors.OKCYAN}→ 隐私风险分析...{Colors.ENDC}", end=' ')
            infons_summary = parse_infons_from_output(infons_output)
            risk_prompt = build_privacy_analysis_prompt(infons_summary, law_tree, is_direct=False)
            ttft_risk, total_time_risk, risk_output, success = call_ollama_stream(model, risk_prompt)
            
            if not success:
                print(f"{Colors.FAIL}✗{Colors.ENDC}")
                result['status'] = 'failed_risk_analysis'
                return result
            
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} (TTFT: {ttft_risk:.2f}s, 总计: {total_time_risk:.2f}s)")
            
            result['risk_ttft'] = ttft_risk
            result['risk_total_time'] = total_time_risk
            result['risk_length'] = len(risk_output)
            result['total_time'] = total_time_infons + total_time_risk
            result['status'] = 'success'
            
        else:  # direct mode
            # 直接隐私风险分析
            print(f"  {Colors.OKCYAN}→ 直接隐私风险分析...{Colors.ENDC}", end=' ')
            risk_prompt = build_privacy_analysis_prompt(None, law_tree, is_direct=True, direct_text=text)
            ttft_risk, total_time_risk, risk_output, success = call_ollama_stream(model, risk_prompt)
            
            if not success:
                print(f"{Colors.FAIL}✗{Colors.ENDC}")
                result['status'] = 'failed'
                return result
            
            print(f"{Colors.OKGREEN}✓{Colors.ENDC} (TTFT: {ttft_risk:.2f}s, 总计: {total_time_risk:.2f}s)")
            
            result['risk_ttft'] = ttft_risk
            result['risk_total_time'] = total_time_risk
            result['risk_length'] = len(risk_output)
            result['total_time'] = total_time_risk
            result['status'] = 'success'
    
    except Exception as e:
        print(f"{Colors.FAIL}✗ 异常: {str(e)}{Colors.ENDC}")
        result['status'] = 'exception'
        result['error'] = str(e)
    
    return result


def load_test_cases(csv_path, limit=None):
    """加载测试数据"""
    cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            cases.append(row)
    return cases


def save_results(results, output_path):
    """保存测试结果到CSV"""
    if not results:
        return
    
    fieldnames = list(results[0].keys())
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{Colors.OKGREEN}结果已保存到: {output_path}{Colors.ENDC}")


def print_summary_statistics(results, model, mode):
    """打印统计摘要"""
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print(f"{Colors.WARNING}没有成功的测试结果{Colors.ENDC}")
        return
    
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}统计摘要 - {model} ({mode}模式){Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    if mode == 'extraction':
        # 信息元提取模式统计
        avg_infons_ttft = sum(r['infons_ttft'] for r in successful) / len(successful)
        avg_infons_total = sum(r['infons_total_time'] for r in successful) / len(successful)
        avg_risk_ttft = sum(r['risk_ttft'] for r in successful) / len(successful)
        avg_risk_total = sum(r['risk_total_time'] for r in successful) / len(successful)
        avg_total = sum(r['total_time'] for r in successful) / len(successful)
        
        print(f"成功测试数: {len(successful)}/{len(results)}")
        print(f"\n信息元提取阶段:")
        print(f"  平均首次响应时间 (TTFT): {avg_infons_ttft:.2f}秒")
        print(f"  平均总耗时: {avg_infons_total:.2f}秒")
        print(f"\n隐私风险分析阶段:")
        print(f"  平均首次响应时间 (TTFT): {avg_risk_ttft:.2f}秒")
        print(f"  平均总耗时: {avg_risk_total:.2f}秒")
        print(f"\n总体:")
        print(f"  平均总耗时: {avg_total:.2f}秒")
    else:
        # 直接推断模式统计
        avg_risk_ttft = sum(r['risk_ttft'] for r in successful) / len(successful)
        avg_risk_total = sum(r['risk_total_time'] for r in successful) / len(successful)
        
        print(f"成功测试数: {len(successful)}/{len(results)}")
        print(f"\n隐私风险分析:")
        print(f"  平均首次响应时间 (TTFT): {avg_risk_ttft:.2f}秒")
        print(f"  平均总耗时: {avg_risk_total:.2f}秒")
    
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def select_models():
    """让用户选择要测试的模型"""
    print(f"\n{Colors.BOLD}可用的Ollama模型:{Colors.ENDC}")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {model}")
    
    print(f"\n请选择要测试的模型（输入数字，用逗号分隔，或输入 'all' 测试所有模型）:")
    choice = input("> ").strip()
    
    if choice.lower() == 'all':
        return AVAILABLE_MODELS
    
    try:
        indices = [int(x.strip()) for x in choice.split(',')]
        selected = [AVAILABLE_MODELS[i-1] for i in indices if 1 <= i <= len(AVAILABLE_MODELS)]
        return selected
    except:
        print(f"{Colors.FAIL}无效的选择，将使用第一个模型{Colors.ENDC}")
        return [AVAILABLE_MODELS[0]]


def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("="*70)
    print("           Ollama 模型隐私推理性能测试工具")
    print("           (从前端JS模板同步提示词)")
    print("="*70)
    print(f"{Colors.ENDC}")
    
    # 初始化JS模板路径
    print(f"\n{Colors.OKCYAN}初始化JS模板...{Colors.ENDC}")
    if not init_js_templates():
        print(f"{Colors.WARNING}警告: 无法找到JS模板目录，将使用内置提示词{Colors.ENDC}")
    else:
        print(f"{Colors.OKGREEN}成功加载JS模板: {JS_TEMPLATES_DIR}{Colors.ENDC}")
    
    # 设置路径
    script_dir = Path(__file__).parent
    csv_path = script_dir / "cases.csv"
    
    if not csv_path.exists():
        print(f"{Colors.FAIL}错误: 找不到测试数据文件 {csv_path}{Colors.ENDC}")
        return
    
    # 选择模型
    selected_models = select_models()
    if not selected_models:
        print(f"{Colors.FAIL}没有选择任何模型{Colors.ENDC}")
        return
    
    print(f"\n{Colors.OKGREEN}已选择 {len(selected_models)} 个模型{Colors.ENDC}")
    
    # 询问测试数据量
    print(f"\n请输入要测试的案例数量（默认全部 {Colors.WARNING}注意：全部测试可能需要很长时间{Colors.ENDC}）:")
    limit_input = input("> ").strip()
    limit = int(limit_input) if limit_input.isdigit() else None
    
    # 加载测试数据
    print(f"\n{Colors.OKCYAN}加载测试数据...{Colors.ENDC}")
    test_cases = load_test_cases(csv_path, limit)
    print(f"{Colors.OKGREEN}加载了 {len(test_cases)} 个测试案例{Colors.ENDC}")
    
    # 加载法律树
    print(f"\n{Colors.OKCYAN}加载法律框架 (PIPL.json)...{Colors.ENDC}")
    law_tree_text = load_pipl_law_tree()
    if law_tree_text:
        # 显示加载的法律类别数量
        if "Privacy Categories" in law_tree_text:
            import re
            match = re.search(r'Privacy Categories \((\d+) items\)', law_tree_text)
            if match:
                num_categories = match.group(1)
                print(f"{Colors.OKGREEN}成功加载 {num_categories} 个隐私类别{Colors.ENDC}")
            else:
                print(f"{Colors.OKGREEN}成功加载法律框架{Colors.ENDC}")
        else:
            print(f"{Colors.OKGREEN}成功加载法律框架{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}使用默认法律框架{Colors.ENDC}")
    
    # 询问测试模式
    print(f"\n{Colors.BOLD}选择测试模式:{Colors.ENDC}")
    print("  1. 信息元提取模式（Information Extraction Mode）")
    print("  2. 直接推断模式（Direct Inference Mode）")
    print("  3. 两种模式都测试")
    mode_choice = input("> ").strip()
    
    modes_to_test = []
    if mode_choice == '1':
        modes_to_test = ['extraction']
    elif mode_choice == '2':
        modes_to_test = ['direct']
    else:
        modes_to_test = ['extraction', 'direct']
    
    # 开始测试
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model in selected_models:
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}测试模型: {model}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        for mode in modes_to_test:
            mode_name = "信息元提取模式" if mode == 'extraction' else "直接推断模式"
            print(f"\n{Colors.BOLD}{Colors.OKBLUE}模式: {mode_name}{Colors.ENDC}")
            print(f"{Colors.OKBLUE}{'-'*70}{Colors.ENDC}\n")
            
            results = []
            
            for i, case in enumerate(test_cases, 1):
                print(f"{Colors.BOLD}案例 {i}/{len(test_cases)}{Colors.ENDC} (ID: {case['id']})")
                result = test_single_case(model, case, mode, law_tree_text)
                results.append(result)
                
                # 每10个案例休息一下
                if i % 10 == 0:
                    print(f"{Colors.WARNING}休息3秒...{Colors.ENDC}")
                    time.sleep(3)
            
            # 保存结果
            model_safe_name = model.replace('/', '_').replace(':', '_')
            output_file = script_dir / f"results_{model_safe_name}_{mode}_{timestamp}.csv"
            save_results(results, output_file)
            
            # 打印统计信息
            print_summary_statistics(results, model, mode_name)
    
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}所有测试完成！{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}{'='*70}{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}测试被用户中断{Colors.ENDC}")
        sys.exit(0)

