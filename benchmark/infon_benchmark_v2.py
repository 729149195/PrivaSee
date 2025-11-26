#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息元提取 Benchmark V2 
- 从 infons.js 导入提示词模板
- 惰性LLM匹配: 只在评估时按需调用
- 语义匹配: DESC/SCEN/REL 都用语义相似判断
"""

import json, time, requests, re, sys, threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.ace_to_infons import GoldSample, DescInfon, ScenInfon, RelInfon
from benchmark.semantic_matcher import get_cache_key, clear_cache

# ============================================================================
# 配置
# ============================================================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_API_KEY = "sk-050b8f5117124731a5c962e5890500aa"

AVAILABLE_MODELS = [
    "niels32167/qwen3-4b-instruct:latest",
    "qwen2.5:7b-instruct", 
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen3:14b",
    "gemma3:12b",
    "deepseek-r1:14b",
]

# ============================================================================
# LLM语义匹配 - 带缓存
# ============================================================================
_llm_cache = {}
_cache_lock = threading.Lock()

def qwen_check_batch(pairs: List[Tuple[str, str, str]]) -> Dict[str, bool]:
    """批量调用Qwen判断语义相似"""
    results = {}
    
    def check_one(ck, t1, t2):
        try:
            resp = requests.post(QWEN_API_URL,
                headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
                json={"model": "qwen-turbo-latest",
                      "messages": [{"role": "user", "content": f'"{t1}"和"{t2}"意思相同或相似吗？只答YES或NO'}],
                      "temperature": 0.1, "max_tokens": 10},
                timeout=5)
            is_sim = resp.status_code == 200 and 'YES' in resp.json().get('choices', [{}])[0].get('message', {}).get('content', '').upper()
            return ck, is_sim
        except:
            return ck, False
    
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(check_one, ck, t1, t2) for ck, t1, t2 in pairs]
        for f in as_completed(futures):
            ck, is_sim = f.result()
            results[ck] = is_sim
    
    return results

def semantic_match(t1: str, t2: str) -> bool:
    """判断两个文本是否语义相似"""
    t1n, t2n = t1.lower().strip(), t2.lower().strip()
    if not t1n or not t2n or len(t1n) < 2 or len(t2n) < 2:
        return False
    if t1n == t2n or t1n in t2n or t2n in t1n:
        return True
    
    ck = get_cache_key(t1, t2)
    with _cache_lock:
        if ck in _llm_cache:
            return _llm_cache[ck]
    return False  # 缓存未命中时返回False，由批量预热填充

def preheat_cache(pairs: List[Tuple[str, str]]):
    """预热缓存 - 批量调用LLM"""
    to_check = []
    for t1, t2 in pairs:
        t1n, t2n = t1.lower().strip(), t2.lower().strip()
        if not t1n or not t2n or len(t1n) < 2 or len(t2n) < 2:
            continue
        if t1n == t2n or t1n in t2n or t2n in t1n:
            continue
        ck = get_cache_key(t1, t2)
        with _cache_lock:
            if ck not in _llm_cache:
                to_check.append((ck, t1, t2))
    
    if not to_check:
        return
    
    # 批量处理
    batch_size = 50
    for i in range(0, len(to_check), batch_size):
        batch = to_check[i:i+batch_size]
        results = qwen_check_batch(batch)
        with _cache_lock:
            _llm_cache.update(results)
        print(f"\r  LLM匹配: {min(i+batch_size, len(to_check))}/{len(to_check)}", end="", flush=True)
    print(" ✓")

# ============================================================================
# 从 infons.js 加载提示词模板
# ============================================================================
def load_infons_template() -> str:
    js_path = Path(__file__).parent.parent / 'frontend' / 'src' / 'templates' / 'infons.js'
    if not js_path.exists():
        return None
    
    with open(js_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.search(r'export const BENCHMARK_EXTRACTION = String\.raw`(.*?)`\s*;', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

_LOADED_TEMPLATE = load_infons_template()

def get_prompt(text: str) -> str:
    if _LOADED_TEMPLATE:
        return f"{_LOADED_TEMPLATE}\n\n文本：\n{text}\n\n输出："
    return f"提取文本中的所有信息元，每行一个CSV格式。\n\n文本：{text}\n\n输出："

# ============================================================================
# Ollama API
# ============================================================================
def call_ollama(model: str, prompt: str) -> Tuple[str, float, float]:
    try:
        start, ttft, text = time.time(), 0, ""
        resp = requests.post(OLLAMA_API_URL,
            json={"model": model, "prompt": prompt, "stream": True,
                  "options": {"temperature": 0.2, "num_predict": 4096}},
            stream=True, timeout=180)
        if resp.status_code != 200:
            return f"ERROR:{resp.status_code}", 0, 0
        first = True
        for line in resp.iter_lines():
            if line:
                try:
                    d = json.loads(line)
                    if 'response' in d:
                        if first: ttft = time.time() - start; first = False
                        text += d['response']
                    if d.get('done'): break
                except: pass
        return text, ttft, time.time() - start
    except Exception as e:
        return f"ERROR:{e}", 0, 0

# ============================================================================
# 解析
# ============================================================================
def parse_line(line: str) -> Optional[Dict]:
    line = re.sub(r'^\d+\.\s*', '', line.strip())
    line = re.sub(r'^(DESC|SCEN|REL):\s*', '', line, flags=re.I)
    parts = [p.strip() for p in line.replace('\\,', '\x00').split(',')]
    parts = [p.replace('\x00', ',') for p in parts]
    if len(parts) < 4: return None
    iid, itype = parts[0], parts[1].upper()
    if ':' not in iid: iid = f"{itype.lower()[:4]}:{iid}"
    try:
        conf = lambda i: float(parts[i]) if len(parts) > i and parts[i].replace('.','',1).isdigit() else 0.9
        if itype == 'DESC':
            return {'iid': iid, 'infon_type': 'DESC', 'entity': parts[2], 'attribute': parts[3], 'confidence': conf(5)}
        if itype == 'SCEN':
            return {'iid': iid, 'infon_type': 'SCEN', 'temporal': parts[2], 'spatial': parts[3], 'confidence': conf(4)}
        if itype == 'REL':
            return {'iid': iid, 'infon_type': 'REL', 'relation_name': parts[2], 'arg_refs': parts[3].split('|'), 'confidence': conf(4)}
    except: pass
    return None

def extract_infons(resp: str) -> List[Dict]:
    if '```' in resp:
        resp = '\n'.join(l for l in resp.split('\n') if not l.strip().startswith('```'))
    return [r for r in (parse_line(l) for l in resp.split('\n')) if r]

def load_samples(path: str, limit: int = None) -> List[GoldSample]:
    with open(path, 'r') as f: data = json.load(f)
    if limit: data = data[:limit]
    samples = []
    for it in data:
        infons = []
        for d in it.get('infons', []):
            t = d.get('infon_type', '').upper()
            if t == 'DESC':
                infons.append(DescInfon(iid=d.get('iid',''), infon_type='DESC', entity=d.get('entity',''),
                    attribute=d.get('attribute',''), data_type='string', confidence=1.0))
            elif t == 'SCEN':
                infons.append(ScenInfon(iid=d.get('iid',''), infon_type='SCEN',
                    temporal=d.get('temporal',''), spatial=d.get('spatial','')))
            elif t == 'REL':
                infons.append(RelInfon(iid=d.get('iid',''), infon_type='REL',
                    relation_name=d.get('relation_name',''), arg_refs=d.get('arg_refs',[])))
        samples.append(GoldSample(doc_id=it.get('doc_id',''), text=it.get('text',''),
            infons=infons, language=it.get('language','')))
    return samples

# ============================================================================
# 匹配逻辑 - 语义匹配
# ============================================================================
def match_desc(g, p) -> Tuple[bool, float]:
    ga, pa = g.get('attribute','').strip(), p.get('attribute','').strip()
    if not ga or not pa: return False, 0
    gan, pan = ga.lower(), pa.lower()
    if gan == pan: return True, 1.0
    if gan in pan or pan in gan: return True, 0.85
    if semantic_match(ga, pa): return True, 0.7
    return False, 0

def match_scen(g, p) -> Tuple[bool, float]:
    gt, gs = g.get('temporal','').strip(), g.get('spatial','').strip()
    pt, ps = p.get('temporal','').strip(), p.get('spatial','').strip()
    
    temp_ok = (not gt or not pt or gt.lower() == pt.lower() or 
               gt.lower() in pt.lower() or pt.lower() in gt.lower() or
               semantic_match(gt, pt))
    spat_ok = (not gs or not ps or gs.lower() == ps.lower() or
               gs.lower() in ps.lower() or ps.lower() in gs.lower() or
               semantic_match(gs, ps))
    
    if temp_ok and spat_ok and (gt or gs): return True, 0.8
    if temp_ok or spat_ok: return True, 0.5
    return False, 0

def match_rel(g, p, gm, pm) -> Tuple[bool, float]:
    gr, pr = g.get('relation_name','').strip(), p.get('relation_name','').strip()
    if not gr or not pr: return False, 0
    grn, prn = gr.lower().replace('_',' '), pr.lower().replace('_',' ')
    if grn == prn: return True, 1.0
    if grn in prn or prn in grn: return True, 0.9
    if semantic_match(gr, pr): return True, 0.8
    return False, 0

def collect_pairs(all_gold, all_pred) -> List[Tuple[str, str]]:
    """收集需要LLM判断的pairs - 排除能通过规则判断的"""
    pairs = set()
    
    def needs_llm(t1, t2):
        t1n, t2n = t1.lower().strip(), t2.lower().strip()
        if not t1n or not t2n or len(t1n) < 2 or len(t2n) < 2:
            return False
        if t1n == t2n or t1n in t2n or t2n in t1n:
            return False  # 规则可判断
        return True
    
    for golds, preds in zip(all_gold, all_pred):
        # 只取topN避免太多匹配
        desc_g = [g for g in golds if g.get('infon_type') == 'DESC'][:50]
        desc_p = [p for p in preds if p.get('infon_type') == 'DESC'][:50]
        rel_g = [g for g in golds if g.get('infon_type') == 'REL'][:30]
        rel_p = [p for p in preds if p.get('infon_type') == 'REL'][:30]
        
        for p in desc_p:
            pa = p.get('attribute','')
            for g in desc_g:
                ga = g.get('attribute','')
                if needs_llm(ga, pa):
                    pairs.add((ga, pa))
        
        for p in rel_p:
            pr = p.get('relation_name','')
            for g in rel_g:
                gr = g.get('relation_name','')
                if needs_llm(gr, pr):
                    pairs.add((gr, pr))
    
    # 限制最大数量，采样
    pairs_list = list(pairs)
    if len(pairs_list) > 500:
        import random
        random.shuffle(pairs_list)
        pairs_list = pairs_list[:500]
    return pairs_list

def evaluate(gold_list, pred_list) -> Dict:
    total = {t: {'tp':0,'fp':0,'fn':0,'partial':0} for t in ['DESC','SCEN','REL']}
    for golds, preds in zip(gold_list, pred_list):
        if not preds: continue
        gm = {i.get('iid',''): i for i in golds if i.get('infon_type')=='DESC'}
        pm = {i.get('iid',''): i for i in preds if i.get('infon_type')=='DESC'}
        for t in ['DESC','SCEN','REL']:
            gs = [i for i in golds if i.get('infon_type')==t]
            ps = [i for i in preds if i.get('infon_type')==t]
            matched = set()
            for p in ps:
                best_i, best_s = None, 0
                for gi, g in enumerate(gs):
                    if gi in matched: continue
                    m, s = (match_desc(g,p) if t=='DESC' else match_scen(g,p) if t=='SCEN' else match_rel(g,p,gm,pm))
                    if m and s > best_s: best_i, best_s = gi, s
                if best_i is not None:
                    matched.add(best_i)
                    total[t]['tp' if best_s >= 0.6 else 'partial'] += 1
                else:
                    total[t]['fp'] += 1
            total[t]['fn'] += len(gs) - len(matched)
    return total

def calc(d):
    tp, fp, fn, pt = d['tp'], d['fp'], d['fn'], d['partial']
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    eff = tp + pt*0.5
    pp = eff/(eff+fp) if eff+fp else 0
    pr = eff/(eff+fn) if eff+fn else 0
    pf1 = 2*pp*pr/(pp+pr) if pp+pr else 0
    return {**d, 'precision':p, 'recall':r, 'f1':f1, 'partial_f1':pf1}

# ============================================================================
# 主流程
# ============================================================================
def run_benchmark(model: str, samples: List[GoldSample], out_dir: Path) -> Dict:
    global _llm_cache
    
    print(f"\n{'='*60}\n模型: {model}\n{'='*60}")
    if _LOADED_TEMPLATE:
        print(f"✓ 已加载 infons.js BENCHMARK_EXTRACTION 模板")
    
    # 1. 提取阶段
    all_gold, all_pred = [], []
    for i, s in enumerate(samples):
        print(f"\r[{i+1}/{len(samples)}] {s.doc_id[:35]}...", end=" ")
        resp, ttft, total = call_ollama(model, get_prompt(s.text[:3000]))
        if resp.startswith("ERROR"):
            print(f"✗ {resp}")
            all_gold.append([])
            all_pred.append([])
            continue
        preds = extract_infons(resp)
        golds = [vars(inf) if hasattr(inf,'__dict__') else inf for inf in s.infons]
        all_gold.append(golds)
        all_pred.append(preds)
        print(f"✓ {total:.1f}s P={len(preds)}/{len(golds)}")
    
    # 2. 收集需要LLM判断的pairs
    print(f"\n收集匹配对...", end=" ")
    pairs = collect_pairs(all_gold, all_pred)
    print(f"{len(pairs)} 对")
    
    # 3. 批量预热LLM缓存
    if pairs:
        print(f"预热LLM缓存...")
        preheat_cache(pairs)
    
    # 4. 评估
    print(f"评估中...")
    by_type = evaluate(all_gold, all_pred)
    overall = {'tp':0,'fp':0,'fn':0,'partial':0}
    for t in by_type:
        for k in overall: overall[k] += by_type[t][k]
    result = {'overall': calc(overall), 'by_type': {t: calc(by_type[t]) for t in by_type}, 'num_samples': len(samples)}
    
    # 打印
    print(f"\n{'='*60}")
    o = result['overall']
    print(f"整体: P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} Partial-F1={o['partial_f1']:.3f}")
    for t in ['DESC','SCEN','REL']:
        m = result['by_type'][t]
        print(f"  {t}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} TP={m['tp']} Partial={m['partial']}")
    print(f"{'='*60}")
    
    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"bench_{model.replace('/','_').replace(':','_')}_{ts}.json"
    with open(path, 'w') as f:
        json.dump({'model':model, 'result':result}, f, ensure_ascii=False, indent=2)
    print(f"保存: {path}")
    return result

def main():
    global _llm_cache
    _llm_cache = {}
    
    print("\n" + "="*60 + "\n     信息元提取 Benchmark V2\n     (infons.js模板 + 语义匹配)\n" + "="*60)
    
    gold_path = Path(__file__).parent / 'gold_data' / 'gold.json'
    if not gold_path.exists():
        print("找不到Gold数据")
        return
    
    limit = input("样本数量（默认20，0=全部）: ").strip()
    limit = None if limit == '0' else (int(limit) if limit.isdigit() and int(limit) > 0 else 20)
    samples = load_samples(str(gold_path), limit)
    print(f"✓ 加载 {len(samples)} 个样本")
    
    print("\n可用模型:")
    for i, m in enumerate(AVAILABLE_MODELS, 1): print(f"  {i}. {m}")
    choice = input("选择（数字/all）: ").strip()
    selected = AVAILABLE_MODELS if choice.lower() == 'all' else \
        [AVAILABLE_MODELS[int(x)-1] for x in choice.split(',') if x.strip().isdigit() and 1 <= int(x) <= len(AVAILABLE_MODELS)] or [AVAILABLE_MODELS[0]]
    
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    clear_cache()
    
    for model in selected:
        try: run_benchmark(model, samples, out_dir)
        except KeyboardInterrupt: print("\n中断"); break
        except Exception as e: print(f"错误: {e}"); import traceback; traceback.print_exc()
    print("\n完成！")

if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: print("\n中断")
