#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义匹配器 V2 - 并行LLM调用 + 进度显示
"""

import json
import requests
import hashlib
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Qwen API配置
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_API_KEY = "sk-050b8f5117124731a5c962e5890500aa"
QWEN_MODEL = "qwen-turbo-latest"

_similarity_cache: Dict[str, bool] = {}
_cache_lock = Lock()
_progress = {'total': 0, 'done': 0, 'hits': 0}


def get_cache_key(text1: str, text2: str) -> str:
    t1, t2 = text1.lower().strip(), text2.lower().strip()
    combined = f"{min(t1,t2)}|||{max(t1,t2)}"
    return hashlib.md5(combined.encode()).hexdigest()


def call_qwen_batch(prompts: List[str], timeout: int = 10) -> List[Optional[str]]:
    """并行调用多个prompt"""
    results = [None] * len(prompts)
    
    def call_single(idx, prompt):
        try:
            resp = requests.post(
                QWEN_API_URL,
                headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                json={"model": QWEN_MODEL, "messages": [{"role": "user", "content": prompt}], 
                      "temperature": 0.1, "max_tokens": 50},
                timeout=timeout
            )
            if resp.status_code == 200:
                return idx, resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            return idx, None
        except:
            return idx, None
    
    with ThreadPoolExecutor(max_workers=min(10, len(prompts))) as executor:
        futures = [executor.submit(call_single, i, p) for i, p in enumerate(prompts)]
        for f in as_completed(futures):
            idx, result = f.result()
            results[idx] = result
    
    return results


def parallel_similarity_check(pairs: List[Tuple[str, str]], 
                             max_workers: int = 8,
                             batch_size: int = 10,
                             show_progress: bool = True) -> Dict[str, bool]:
    """并行检查语义相似性"""
    global _progress
    
    tasks = []
    results_map = {}
    
    for t1, t2 in pairs:
        t1_n, t2_n = t1.lower().strip(), t2.lower().strip()
        
        # 快速规则判断
        if t1_n == t2_n:
            results_map[get_cache_key(t1, t2)] = True
            continue
        if t1_n in t2_n or t2_n in t1_n:
            results_map[get_cache_key(t1, t2)] = True
            continue
        
        cache_key = get_cache_key(t1, t2)
        with _cache_lock:
            if cache_key in _similarity_cache:
                results_map[cache_key] = _similarity_cache[cache_key]
                _progress['hits'] += 1
                continue
        
        tasks.append((cache_key, t1, t2))
    
    if not tasks:
        return results_map
    
    _progress['total'] = len(tasks)
    _progress['done'] = 0
    
    if show_progress:
        print(f"\r  LLM语义匹配: 0/{len(tasks)} (规则匹配: {len(results_map)}, 缓存: {_progress['hits']})", end='', flush=True)
    
    # 构建prompt
    prompts = []
    for _, t1, t2 in tasks:
        prompt = f"""判断这两个词/短语是否指代相同或相似的事物（包含关系也算相似）:
A: {t1}
B: {t2}
只回答YES或NO:"""
        prompts.append(prompt)
    
    # 分批并行调用
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_tasks = tasks[i:i+batch_size]
        
        results = call_qwen_batch(batch_prompts, timeout=10)
        
        for (cache_key, _, _), result in zip(batch_tasks, results):
            is_similar = result and 'YES' in result.upper()
            results_map[cache_key] = is_similar
            with _cache_lock:
                _similarity_cache[cache_key] = is_similar
            _progress['done'] += 1
        
        if show_progress:
            print(f"\r  LLM语义匹配: {_progress['done']}/{len(tasks)} (规则: {len(pairs)-len(tasks)}, 缓存: {_progress['hits']})", end='', flush=True)
    
    if show_progress:
        print(f"\r  LLM语义匹配: {_progress['done']}/{len(tasks)} ✓                              ")
    
    return results_map


def is_relation_similar(rel1: str, rel2: str) -> bool:
    """关系名相似性判断"""
    if not rel1 or not rel2:
        return False
    
    r1 = rel1.lower().replace('_', ' ').replace('-', ' ').strip()
    r2 = rel2.lower().replace('_', ' ').replace('-', ' ').strip()
    
    if r1 == r2:
        return True
    
    synonyms = {
        'located at': {'located in', 'in', 'at', 'place', 'based in', 'location', 'is in', 'is at'},
        'located in': {'located at', 'in', 'at', 'place', 'based in', 'location'},
        'employed by': {'works for', 'employee of', 'work at', 'works at', 'work for', 'employed at'},
        'part of': {'belongs to', 'member of', 'in', 'subset of', 'division of', 'included in'},
        'part of geo': {'located in', 'in', 'within', 'inside', 'geo part', 'part of'},
        'citizen of': {'nationality', 'from', 'citizen', 'national of', 'native of', 'born in'},
        'near': {'close to', 'nearby', 'adjacent', 'next to', 'beside', 'around'},
        'owns': {'owner of', 'has', 'possesses', 'controls', 'owned by'},
        'founder of': {'founded', 'created', 'established', 'started', 'set up'},
        'member of': {'belongs to', 'in', 'affiliated with', 'part of', 'joined'},
        'subsidiary of': {'owned by', 'part of', 'division of', 'belongs to', 'under'},
        'has': {'owns', 'possesses', 'contains', 'includes', 'with'},
        'invited by': {'invited to', 'invited', 'called by', 'asked by'},
        'has feature': {'has', 'contains', 'includes', 'with', 'feature'},
        'works for': {'employed by', 'employee of', 'work at', 'works at'},
        'affiliated with': {'member of', 'part of', 'belongs to', 'associated with'},
        'resident of': {'lives in', 'resides in', 'citizen of', 'from'},
        'org based in': {'located at', 'based in', 'headquarters in', 'located in'},
    }
    
    for base, syns in synonyms.items():
        all_v = {base} | syns
        if r1 in all_v and r2 in all_v:
            return True
    
    return False


def clear_cache():
    global _similarity_cache, _progress
    with _cache_lock:
        _similarity_cache = {}
    _progress = {'total': 0, 'done': 0, 'hits': 0}


def get_cache_stats() -> Dict:
    with _cache_lock:
        return {'size': len(_similarity_cache), 'hits': _progress['hits']}
