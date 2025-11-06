#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrivaSee 测试用例自动生成脚本
使用 DeepSeek API 生成多样化的隐私推理测试用例
"""

import asyncio
import aiohttp
import random
import json
from datetime import datetime
from typing import List, Dict

# DeepSeek API 配置
API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "sk-8c2ee9474f2f44f5969dcd5de280e634"
MODEL = "deepseek-chat"

# 隐私类别（CCPA/CPRA 11大类）
PRIVACY_CATEGORIES = [
    "Category A: Identifiers (标识符)",
    "Category B: Customer Records PI (客户记录)",
    "Category C: Protected Classification (受保护分类)",
    "Category D: Commercial Information (商业信息)",
    "Category E: Biometric Information (生物特征)",
    "Category F: Internet/Network Activity (网络活动)",
    "Category G: Geolocation Data (地理位置)",
    "Category H: Sensory Data (感官数据)",
    "Category I: Professional/Employment Info (职业信息)",
    "Category J: Education Info (教育信息)",
    "Category K: Inferences (推断信息)"
]

# 难度级别
DIFFICULTY_LEVELS = [
    "direct (直接暴露)",
    "simple (简单推理)",
    "complex (复杂关联)"
]

# 场景类型
SCENARIO_TYPES = [
    "日常生活场景（就医、购药、餐饮、购物、出行）",
    "工作场景（求职、面试、工作交流、职业发展）",
    "社交场景（家庭关系、朋友聚会、恋爱关系）",
    "学习场景（学校教育、在线学习、考试成绩）",
    "健康场景（体检、疾病治疗、心理咨询、健康管理）",
    "金融场景（银行业务、投资理财、贷款、消费）",
    "旅游场景（出行计划、住宿预订、景点游览）",
    "法律场景（诉讼、合同、移民、签证）",
    "宗教文化场景（宗教活动、文化习俗、节日庆祝）",
    "在线活动场景（社交媒体、网购、搜索、游戏）"
]

# 语言设置
LANGUAGES = ["中文", "英文", "中英文混合"]

# 用户职业
OCCUPATIONS = [
    "学生",
    "上班族",
    "医生",
    "教师",
    "程序员",
    "自由职业者",
    "企业家",
    "艺术家",
    "运动员",
    "退休人士",
    "公务员",
    "科研人员",
    "金融分析师",
    "产品经理",
    "设计师",
    "销售代表",
    "法律顾问",
    "心理咨询师",
    "护士",
    "工程师",
    "作家",
    "记者",
    "服务员",
    "厨师",
    "司机",
    "警察",
    "军人",
    "农民",
    "工厂工人",
    "家庭主妇",
    "留学生",
    "项目经理",
    "数据分析师",
    "营销专员",
    "短视频博主",
    "宠物美容师",
    "摄影师",
    "翻译",
    "采购专员",
    "理发师",
    "电商运营",
    "医生助理",
    "健身教练",
    "新媒体运营",
    "人力资源专员",
    "保险代理人",
]

# 提示词模板
PROMPT_TEMPLATES = [
    """你是一个真实用户，正在和AI助手聊天。你的身份是{occupation}。

【场景背景】
你现在处于{scenario}中，遇到了一个需要AI帮助的情况。这个情况自然涉及到{category}相关的信息。

【你的任务】
像平时和ChatGPT/Claude聊天一样，随意地发一条消息。可以是提问、分享、求助、吐槽、闲聊等任何形式。

【重要提示】
- 就像平时发消息一样自然，想怎么说就怎么说
- 长度随意，可以一句话也可以几句话
- 可以用口语、网络用语，但不要用表情符号
- 在自然描述中带入隐私细节（{difficulty}级别）
- **必须使用真实具体的名称，绝不用XX、某某等占位符**

【语言】{language}

直接输出你要发给AI的那条消息，不要加任何解释。""",

    """模拟一个{occupation}和AI助手的真实对话输入。

情境：{scenario}
语言：{language}
表达风格：{inference_style}

这个用户在日常使用AI助手时，自然地描述了一个情况，无意中透露了{category}相关的隐私信息（{difficulty}推理难度）。

要求：
- 完全自然，像真人在用ChatGPT聊天
- 长度随意，任何形式都可以（提问/陈述/求助/闲聊/吐槽）
- 隐私信息自然融入，不刻意显露
- 不要使用表情符号
- **使用真实具体名称，禁止XX、某某等占位符**

只输出用户的消息内容。""",

    """想象你是一个{occupation}，正在使用AI助手。

情境：{scenario}
语言：{language}
风格：{inference_style}

你想和AI聊点什么或问点什么。在自然交流中，你的话会涉及{category}相关的个人信息（{difficulty}方式透露）。

要求：真实、自然，就像平时用ChatGPT一样。使用真实具体名称，不用占位符。不要用表情符号。

直接输出你要说的话。""",

    """创建一条{occupation}发给AI助手的真实消息

场景：{scenario}
语言：{language}
风格：{inference_style}
隐私：{category}（{difficulty}级别）

要求：
- 一条普通的用户消息，自然表达
- 长短随意，自然包含隐私信息
- 使用真实具体名称，严禁占位符
- 不要使用表情符号

直接输出消息内容。""",

    """你是{occupation}，正在和AI聊天。

情境：{scenario}
语言：{language}
风格：{inference_style}

你要发一条消息（提问/求助/分享/闲聊等）。消息会自然涉及{category}相关信息（{difficulty}级别）。

要求：真实自然，像用ChatGPT一样。使用真实具体名称，不用占位符。不要用表情符号。

只输出消息本身。""",

    """场景模拟：{occupation}在{scenario}时打开AI助手输入一段话。

语言：{language}
风格：{inference_style}
隐私：{category}（{difficulty}级别）

要求：真实自然的用户输入，使用真实具体名称，不用占位符。不要用表情符号。

直接输出这段话。""",

    """模拟{occupation}的AI使用场景。

情境：{scenario}
语言：{language}
风格：{inference_style}
隐私：{category}（{difficulty}级别）

生成用户输入。核心原则：真实、自然、随意。使用真实具体名称。不要用表情符号。

只输出用户消息。""",

    """想象场景：{occupation}在{scenario}时打开AI助手输入一段话。

语言：{language}
风格：{inference_style}
隐私：{category}（{difficulty}级别）

要求：真实自然的用户输入。使用真实具体名称，不用占位符。不要用表情符号。

只输出这段话。"""
]

# 推理风格/表达方式
INFERENCE_STYLES = [
    "直白表达",
    "委婉暗示",
    "情景描述",
    "问题咨询",
    "经历分享",
    "计划安排",
    "寻求帮助",
    "抱怨倾诉",
    "炫耀展示",
    "日常闲聊"
]


def generate_prompt() -> tuple:
    """随机生成一个提示词"""
    template = random.choice(PROMPT_TEMPLATES)
    category = random.choice(PRIVACY_CATEGORIES)
    difficulty = random.choice(DIFFICULTY_LEVELS)
    scenario = random.choice(SCENARIO_TYPES)
    language = random.choice(LANGUAGES)
    inference_style = random.choice(INFERENCE_STYLES)
    occupation = random.choice(OCCUPATIONS)
    
    prompt = template.format(
        category=category,
        difficulty=difficulty,
        scenario=scenario,
        language=language,
        inference_style=inference_style,
        occupation=occupation
    )
    
    return prompt, {
        "category": category,
        "difficulty": difficulty.split()[0],  # 只取英文部分
        "scenario": scenario,
        "language": language,
        "inference_style": inference_style,
        "occupation": occupation
    }


def generate_api_params() -> dict:
    """随机生成API调用参数，增加多样性"""
    return {
        "temperature": random.uniform(0.7, 1.3),  # 温度：控制随机性
        "top_p": random.uniform(0.85, 0.95),      # Top-p：核采样
        "max_tokens": random.randint(100, 300),   # 最大token数
        "frequency_penalty": random.uniform(0.0, 0.3),  # 频率惩罚：减少重复
        "presence_penalty": random.uniform(0.0, 0.3)    # 存在惩罚：鼓励新话题
    }


async def call_deepseek_api(session: aiohttp.ClientSession, prompt: str, params: dict) -> str:
    """调用 DeepSeek API 生成文本"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **params
    }
    
    try:
        async with session.post(API_URL, headers=headers, json=payload, timeout=30) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                error_text = await response.text()
                print(f"❌ API错误 (status {response.status}): {error_text}")
                return None
    except asyncio.TimeoutError:
        print(f"❌ API请求超时")
        return None
    except Exception as e:
        print(f"❌ API调用异常: {str(e)}")
        return None


async def generate_batch(batch_size: int = 10) -> List[Dict]:
    """并行生成一批测试用例"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        prompts_meta = []
        
        for _ in range(batch_size):
            prompt, meta = generate_prompt()
            params = generate_api_params()
            prompts_meta.append((meta, params))
            tasks.append(call_deepseek_api(session, prompt, params))
        
        results = await asyncio.gather(*tasks)
        
        test_cases = []
        for i, (result, (meta, params)) in enumerate(zip(results, prompts_meta)):
            if result:
                test_cases.append({
                    "id": len(test_cases) + 1,
                    "text": result,
                    "metadata": meta,
                    "api_params": params
                })
        
        return test_cases


def save_to_csv(test_cases: List[Dict], filename: str):
    """保存测试用例到CSV文件（追加模式，不覆盖）"""
    import csv
    import os
    
    # 定义CSV列
    fieldnames = [
        'id', 'text', 'category', 'difficulty', 'scenario', 
        'language', 'inference_style', 'occupation'
    ]
    
    # 检查文件是否存在
    file_exists = os.path.isfile(filename)
    
    # 如果文件存在，读取最大ID
    max_id = 0
    if file_exists:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    current_id = int(row['id'])
                    if current_id > max_id:
                        max_id = current_id
                except (ValueError, KeyError):
                    pass
    
    with open(filename, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        if not file_exists:
            writer.writeheader()
        
        current_id = max_id + 1
        for case in test_cases:
            row = {
                'id': current_id,
                'text': case['text'],
                'category': case['metadata']['category'],
                'difficulty': case['metadata']['difficulty'],
                'scenario': case['metadata']['scenario'],
                'language': case['metadata']['language'],
                'inference_style': case['metadata']['inference_style'],
                'occupation': case['metadata']['occupation']
            }
            writer.writerow(row)
            current_id += 1
    
    print(f"\n✅ 测试用例已追加到: {filename} (从ID {max_id + 1} 开始)")


async def main():
    """主函数"""
    print("=" * 80)
    print("🚀 PrivaSee 测试用例自动生成器")
    print("=" * 80)
    print(f"📡 API: {API_URL}")
    print(f"🤖 模型: {MODEL}")
    print(f"⚙️  并行批次大小: 10")
    print("=" * 80)
    
    # 获取用户输入
    try:
        total_count = int(input("\n请输入要生成的测试用例总数（建议100-500）: "))
    except ValueError:
        print("❌ 无效输入，使用默认值 100")
        total_count = 100
    
    batch_size = 10
    num_batches = (total_count + batch_size - 1) // batch_size
    
    print(f"\n📊 生成计划：")
    print(f"   - 总数量: {total_count} 个")
    print(f"   - 批次数: {num_batches} 批")
    print(f"   - 每批: {batch_size} 个（并行）")
    print(f"\n⏳ 开始生成...\n")
    
    all_test_cases = []
    
    for batch_num in range(num_batches):
        current_batch_size = min(batch_size, total_count - len(all_test_cases))
        
        print(f"🔄 批次 {batch_num + 1}/{num_batches} - 生成 {current_batch_size} 个用例...")
        
        batch_cases = await generate_batch(current_batch_size)
        
        # 更新ID
        for case in batch_cases:
            case['id'] = len(all_test_cases) + 1
            all_test_cases.append(case)
        
        print(f"   ✅ 完成 {len(batch_cases)}/{current_batch_size} 个")
        print(f"   📈 累计: {len(all_test_cases)}/{total_count}")
        
        # 添加短暂延迟，避免API限流
        if batch_num < num_batches - 1:
            await asyncio.sleep(1)
    
    # 统计信息
    print(f"\n{'='*80}")
    print(f"📊 生成完成统计")
    print(f"{'='*80}")
    print(f"✅ 成功生成: {len(all_test_cases)} 个")
    
    # 按类别统计
    category_count = {}
    difficulty_count = {}
    language_count = {}
    occupation_count = {}
    
    for case in all_test_cases:
        cat = case['metadata']['category'].split(':')[0]
        category_count[cat] = category_count.get(cat, 0) + 1
        
        diff = case['metadata']['difficulty']
        difficulty_count[diff] = difficulty_count.get(diff, 0) + 1
        
        lang = case['metadata']['language']
        language_count[lang] = language_count.get(lang, 0) + 1
        
        occ = case['metadata']['occupation']
        occupation_count[occ] = occupation_count.get(occ, 0) + 1
    
    print(f"\n📋 类别分布:")
    for cat, count in sorted(category_count.items()):
        print(f"   {cat}: {count} 个")
    
    print(f"\n📊 难度分布:")
    for diff, count in sorted(difficulty_count.items()):
        print(f"   {diff}: {count} 个")
    
    print(f"\n🌍 语言分布:")
    for lang, count in sorted(language_count.items()):
        print(f"   {lang}: {count} 个")
    
    print(f"\n👥 职业分布:")
    for occ, count in sorted(occupation_count.items()):
        print(f"   {occ}: {count} 个")
    
    # 保存结果
    filename = "cases.csv"
    save_to_csv(all_test_cases, filename)
    
    print(f"\n{'='*80}")
    print(f"🎉 全部完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())

