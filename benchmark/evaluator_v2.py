#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息元评估器 V2 - 优化REL匹配逻辑
"""

import json
import re
import sys
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from difflib import SequenceMatcher

from benchmark.semantic_matcher import (
    parallel_similarity_check, 
    is_relation_similar,
    get_cache_key
)


@dataclass
class EvalConfig:
    """评估配置"""
    use_llm_matching: bool = True
    text_similarity_threshold: float = 0.4
    partial_match_weight: float = 0.5
    llm_workers: int = 8
    llm_batch_size: int = 10


class SemanticEvaluator:
    """语义评估器 - 优化REL匹配"""
    
    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        if text1 in text2 or text2 in text1:
            shorter = min(len(text1), len(text2))
            longer = max(len(text1), len(text2))
            return 0.5 + 0.5 * (shorter / longer)
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _expand_rel_to_text(self, rel: Dict, desc_map: Dict) -> Tuple[str, List[str]]:
        """
        将REL展开为文本表示
        返回: (关系名, [参数1文本, 参数2文本, ...])
        """
        rel_name = rel.get('relation_name', '').strip()
        arg_refs = rel.get('arg_refs', [])
        
        expanded_args = []
        for ref in arg_refs:
            if ref in desc_map:
                desc = desc_map[ref]
                # 优先用attribute，其次用entity
                text = desc.get('attribute', '') or desc.get('entity', '')
                expanded_args.append(text.strip())
            else:
                # 如果找不到引用，尝试从ref本身提取（可能是直接文本）
                # 有些模型可能直接写 "张三|李四" 而非 iid
                expanded_args.append(ref.strip())
        
        return rel_name, expanded_args
    
    def _collect_llm_pairs(self, gold_infons: List[Dict], pred_infons: List[Dict],
                          gold_desc_map: Dict, pred_desc_map: Dict) -> List[Tuple[str, str]]:
        """收集需要LLM判断的文本对（包括REL参数）"""
        pairs = []
        pairs_set = set()
        
        # DESC属性对
        gold_descs = [g for g in gold_infons if g.get('infon_type', '').upper() == 'DESC']
        pred_descs = [p for p in pred_infons if p.get('infon_type', '').upper() == 'DESC']
        
        for pred in pred_descs:
            pred_attr = pred.get('attribute', '').strip().lower()
            if len(pred_attr) < 2:
                continue
            
            for gold in gold_descs:
                gold_attr = gold.get('attribute', '').strip().lower()
                if len(gold_attr) < 2:
                    continue
                
                sim = self._text_similarity(gold_attr, pred_attr)
                if 0.2 < sim < self.config.text_similarity_threshold:
                    pair = (gold_attr, pred_attr) if gold_attr < pred_attr else (pred_attr, gold_attr)
                    if pair not in pairs_set:
                        pairs_set.add(pair)
                        pairs.append(pair)
        
        # REL展开后的参数对
        gold_rels = [g for g in gold_infons if g.get('infon_type', '').upper() == 'REL']
        pred_rels = [p for p in pred_infons if p.get('infon_type', '').upper() == 'REL']
        
        for pred in pred_rels:
            _, pred_args = self._expand_rel_to_text(pred, pred_desc_map)
            
            for gold in gold_rels:
                _, gold_args = self._expand_rel_to_text(gold, gold_desc_map)
                
                for p_arg in pred_args:
                    if len(p_arg) < 2:
                        continue
                    for g_arg in gold_args:
                        if len(g_arg) < 2:
                            continue
                        
                        p_norm = p_arg.lower()
                        g_norm = g_arg.lower()
                        
                        sim = self._text_similarity(g_norm, p_norm)
                        if 0.2 < sim < self.config.text_similarity_threshold:
                            pair = (g_norm, p_norm) if g_norm < p_norm else (p_norm, g_norm)
                            if pair not in pairs_set:
                                pairs_set.add(pair)
                                pairs.append(pair)
        
        return pairs
    
    def evaluate_single(self, gold_infons: List[Dict], pred_infons: List[Dict], 
                       llm_results: Dict[str, bool] = None) -> Dict:
        """评估单个样本"""
        # 构建DESC映射
        gold_desc_map = {inf.get('iid', ''): inf for inf in gold_infons if inf.get('infon_type', '').upper() == 'DESC'}
        pred_desc_map = {inf.get('iid', ''): inf for inf in pred_infons if inf.get('infon_type', '').upper() == 'DESC'}
        
        # 分类
        gold_by_type = {'DESC': [], 'SCEN': [], 'REL': []}
        pred_by_type = {'DESC': [], 'SCEN': [], 'REL': []}
        
        for inf in gold_infons:
            t = inf.get('infon_type', '').upper()
            if t in gold_by_type:
                gold_by_type[t].append(inf)
        
        for inf in pred_infons:
            t = inf.get('infon_type', '').upper()
            if t in pred_by_type:
                pred_by_type[t].append(inf)
        
        results = {'DESC': {}, 'SCEN': {}, 'REL': {}}
        
        for infon_type in ['DESC', 'SCEN', 'REL']:
            golds = gold_by_type[infon_type]
            preds = pred_by_type[infon_type]
            
            tp, fp, fn, partial = 0, 0, 0, 0
            matched_gold = set()
            
            for pred in preds:
                best_match = None
                best_score = 0.0
                
                for gi, gold in enumerate(golds):
                    if gi in matched_gold:
                        continue
                    
                    if infon_type == 'DESC':
                        is_match, score = self._match_desc(gold, pred, llm_results)
                    elif infon_type == 'SCEN':
                        is_match, score = self._match_scen(gold, pred, llm_results)
                    else:
                        is_match, score = self._match_rel(gold, pred, gold_desc_map, pred_desc_map, llm_results)
                    
                    if is_match and score > best_score:
                        best_match = gi
                        best_score = score
                
                if best_match is not None:
                    matched_gold.add(best_match)
                    if best_score >= 0.6:
                        tp += 1
                    else:
                        partial += 1
                else:
                    fp += 1
            
            fn = len(golds) - len(matched_gold)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            effective_tp = tp + partial * self.config.partial_match_weight
            pp = effective_tp / (effective_tp + fp) if (effective_tp + fp) > 0 else 0.0
            pr = effective_tp / (effective_tp + fn) if (effective_tp + fn) > 0 else 0.0
            pf1 = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0.0
            
            results[infon_type] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'partial': partial,
                'precision': precision, 'recall': recall, 'f1': f1, 'partial_f1': pf1
            }
        
        # 整体
        total_tp = sum(r['tp'] for r in results.values())
        total_fp = sum(r['fp'] for r in results.values())
        total_fn = sum(r['fn'] for r in results.values())
        total_partial = sum(r['partial'] for r in results.values())
        
        op = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        orc = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        of1 = 2 * op * orc / (op + orc) if (op + orc) > 0 else 0.0
        
        eff_tp = total_tp + total_partial * self.config.partial_match_weight
        pp = eff_tp / (eff_tp + total_fp) if (eff_tp + total_fp) > 0 else 0.0
        pr = eff_tp / (eff_tp + total_fn) if (eff_tp + total_fn) > 0 else 0.0
        pf1 = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0.0
        
        return {
            'overall': {
                'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'partial': total_partial,
                'precision': op, 'recall': orc, 'f1': of1, 'partial_f1': pf1
            },
            'by_type': results
        }
    
    def _match_desc(self, gold: Dict, pred: Dict, llm_results: Dict = None) -> Tuple[bool, float]:
        """匹配DESC"""
        gold_attr = gold.get('attribute', '').strip()
        pred_attr = pred.get('attribute', '').strip()
        
        if not gold_attr or not pred_attr:
            return False, 0.0
        
        g_lower = gold_attr.lower()
        p_lower = pred_attr.lower()
        
        if g_lower == p_lower:
            return True, 1.0
        
        if g_lower in p_lower or p_lower in g_lower:
            return True, 0.85
        
        sim = self._text_similarity(gold_attr, pred_attr)
        if sim >= self.config.text_similarity_threshold:
            return True, sim
        
        if llm_results and self.config.use_llm_matching:
            cache_key = get_cache_key(g_lower, p_lower)
            if cache_key in llm_results and llm_results[cache_key]:
                return True, 0.7
        
        return False, sim
    
    def _match_scen(self, gold: Dict, pred: Dict, llm_results: Dict = None) -> Tuple[bool, float]:
        """匹配SCEN"""
        g_temp = gold.get('temporal', '').strip()
        g_spat = gold.get('spatial', '').strip()
        p_temp = pred.get('temporal', '').strip()
        p_spat = pred.get('spatial', '').strip()
        
        temp_match = False
        spat_match = False
        
        if g_temp and p_temp:
            temp_match = self._text_similarity(g_temp, p_temp) >= 0.4
        elif not g_temp and not p_temp:
            temp_match = True
        
        if g_spat and p_spat:
            spat_match = self._text_similarity(g_spat, p_spat) >= 0.4
        elif not g_spat and not p_spat:
            spat_match = True
        
        if temp_match and spat_match:
            return True, 1.0
        elif temp_match or spat_match:
            return True, 0.5
        
        return False, 0.0
    
    def _match_rel(self, gold: Dict, pred: Dict, gold_map: Dict, pred_map: Dict, 
                   llm_results: Dict = None) -> Tuple[bool, float]:
        """
        匹配REL - 核心优化：展开iid为实际文本后比较
        """
        g_rel = gold.get('relation_name', '').strip()
        p_rel = pred.get('relation_name', '').strip()
        
        # 1. 关系名匹配（放宽条件）
        rel_match = False
        g_rel_norm = g_rel.lower().replace('_', ' ')
        p_rel_norm = p_rel.lower().replace('_', ' ')
        
        if g_rel_norm == p_rel_norm:
            rel_match = True
        elif g_rel_norm in p_rel_norm or p_rel_norm in g_rel_norm:
            rel_match = True
        elif is_relation_similar(g_rel, p_rel):
            rel_match = True
        elif self._text_similarity(g_rel, p_rel) >= 0.4:
            rel_match = True
        
        if not rel_match:
            return False, 0.0
        
        # 2. 展开参数为实际文本
        _, g_args = self._expand_rel_to_text(gold, gold_map)
        _, p_args = self._expand_rel_to_text(pred, pred_map)
        
        # 如果没有参数，关系名匹配就算成功
        if not g_args and not p_args:
            return True, 0.7
        
        if not g_args or not p_args:
            return True, 0.4  # 一方有参数一方没有，部分匹配
        
        # 3. 参数匹配 - 找最佳匹配
        # 计算每个gold参数能匹配到的最大分数
        arg_scores = []
        
        for g_arg in g_args:
            best_score = 0.0
            g_lower = g_arg.lower()
            
            for p_arg in p_args:
                p_lower = p_arg.lower()
                
                # 完全匹配
                if g_lower == p_lower:
                    best_score = 1.0
                    break
                
                # 包含关系
                if g_lower in p_lower or p_lower in g_lower:
                    best_score = max(best_score, 0.85)
                    continue
                
                # 文本相似度
                sim = self._text_similarity(g_arg, p_arg)
                if sim >= self.config.text_similarity_threshold:
                    best_score = max(best_score, sim)
                    continue
                
                # LLM结果
                if llm_results and self.config.use_llm_matching:
                    cache_key = get_cache_key(g_lower, p_lower)
                    if cache_key in llm_results and llm_results[cache_key]:
                        best_score = max(best_score, 0.7)
            
            arg_scores.append(best_score)
        
        # 计算平均分数
        if arg_scores:
            avg_score = sum(arg_scores) / len(arg_scores)
            matched_count = sum(1 for s in arg_scores if s >= 0.3)
            
            # 只要有一个参数匹配上就算部分成功
            if matched_count >= 1:
                # 根据匹配程度给分
                match_ratio = matched_count / len(arg_scores)
                final_score = avg_score * 0.5 + match_ratio * 0.5
                return True, max(0.4, final_score)
        
        return False, 0.0
    
    def evaluate_batch(self, gold_samples: List, pred_list: List[List[Dict]], 
                      show_progress: bool = True) -> Dict:
        """批量评估"""
        
        # 第1步：收集所有需要LLM判断的文本对（包括REL参数）
        all_pairs = []
        if self.config.use_llm_matching:
            if show_progress:
                print("  收集待匹配文本对...", end=" ", flush=True)
            
            for sample, preds in zip(gold_samples, pred_list):
                gold_infons = [vars(inf) if hasattr(inf, '__dict__') else inf for inf in sample.infons]
                
                # 构建映射
                gold_desc_map = {inf.get('iid', ''): inf for inf in gold_infons if inf.get('infon_type', '').upper() == 'DESC'}
                pred_desc_map = {inf.get('iid', ''): inf for inf in preds if inf.get('infon_type', '').upper() == 'DESC'}
                
                pairs = self._collect_llm_pairs(gold_infons, preds, gold_desc_map, pred_desc_map)
                all_pairs.extend(pairs)
            
            # 去重
            unique_pairs = list(set(all_pairs))
            if show_progress:
                print(f"共 {len(unique_pairs)} 对")
        else:
            unique_pairs = []
        
        # 第2步：并行LLM匹配
        llm_results = {}
        if self.config.use_llm_matching and unique_pairs:
            llm_results = parallel_similarity_check(
                unique_pairs,
                max_workers=self.config.llm_workers,
                batch_size=self.config.llm_batch_size,
                show_progress=show_progress
            )
        
        # 第3步：评估
        if show_progress:
            print("  评估样本...", end=" ", flush=True)
        
        all_results = []
        for i, (sample, preds) in enumerate(zip(gold_samples, pred_list)):
            gold_infons = [vars(inf) if hasattr(inf, '__dict__') else inf for inf in sample.infons]
            result = self.evaluate_single(gold_infons, preds, llm_results)
            all_results.append(result)
            
            if show_progress and (i + 1) % 50 == 0:
                print(f"{i+1}/{len(gold_samples)}", end=" ", flush=True)
        
        if show_progress:
            print("✓")
        
        # 汇总
        total = {'tp': 0, 'fp': 0, 'fn': 0, 'partial': 0}
        by_type = {t: {'tp': 0, 'fp': 0, 'fn': 0, 'partial': 0} for t in ['DESC', 'SCEN', 'REL']}
        
        for r in all_results:
            for k in total:
                total[k] += r['overall'][k]
            for t in by_type:
                for k in by_type[t]:
                    by_type[t][k] += r['by_type'][t][k]
        
        def calc(d):
            tp, fp, fn, partial = d['tp'], d['fp'], d['fn'], d['partial']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            eff = tp + partial * self.config.partial_match_weight
            pp = eff / (eff + fp) if (eff + fp) > 0 else 0.0
            pr = eff / (eff + fn) if (eff + fn) > 0 else 0.0
            pf1 = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0.0
            return {**d, 'precision': p, 'recall': r, 'f1': f1, 'partial_f1': pf1}
        
        return {
            'overall': calc(total),
            'by_type': {t: calc(by_type[t]) for t in by_type},
            'num_samples': len(gold_samples)
        }


def print_evaluation_report_v2(result: Dict, detailed: bool = False):
    """打印评估报告"""
    print("\n" + "="*70)
    print("                    信息元提取评估报告 (V2)")
    print("="*70)
    
    print(f"\n样本数量: {result.get('num_samples', 'N/A')}")
    
    o = result['overall']
    print(f"\n【整体指标】")
    print("-"*50)
    print(f"  Precision: {o['precision']:.4f}")
    print(f"  Recall:    {o['recall']:.4f}")
    print(f"  F1:        {o['f1']:.4f}")
    print(f"  Partial-F1:{o['partial_f1']:.4f}")
    print(f"  (TP={o['tp']}, FP={o['fp']}, FN={o['fn']}, Partial={o['partial']})")
    
    print(f"\n【分类型指标】")
    print("-"*50)
    
    for infon_type in ['DESC', 'SCEN', 'REL']:
        t = result['by_type'][infon_type]
        print(f"\n  {infon_type}:")
        print(f"    Precision: {t['precision']:.4f}")
        print(f"    Recall:    {t['recall']:.4f}")
        print(f"    F1:        {t['f1']:.4f}")
        print(f"    Partial-F1:{t['partial_f1']:.4f}")
        print(f"    (TP={t['tp']}, FP={t['fp']}, FN={t['fn']}, Partial={t['partial']})")
    
    print("\n" + "="*70)
