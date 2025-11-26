#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息元提取评估器
用于评估系统提取的信息元与Gold标准的匹配度
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import json
import csv
import re
from collections import defaultdict
from difflib import SequenceMatcher

from .ace_to_infons import (
    Infon, DescInfon, ScenInfon, RelInfon, GoldSample,
    infon_to_dict, infon_to_compact_format
)


@dataclass
class MatchResult:
    """单个infon的匹配结果"""
    gold_infon: Optional[Infon]
    pred_infon: Optional[Infon]
    match_type: str  # 'exact', 'partial', 'type_only', 'miss', 'spurious'
    similarity_score: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class TypeMetrics:
    """单个类型的评估指标"""
    tp: int = 0          # True Positives
    fp: int = 0          # False Positives  
    fn: int = 0          # False Negatives
    partial: int = 0     # 部分匹配
    
    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)
    
    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    @property
    def partial_precision(self) -> float:
        """包含部分匹配的精确率"""
        if self.tp + self.partial + self.fp == 0:
            return 0.0
        return (self.tp + 0.5 * self.partial) / (self.tp + self.partial + self.fp)
    
    @property
    def partial_recall(self) -> float:
        """包含部分匹配的召回率"""
        if self.tp + self.partial + self.fn == 0:
            return 0.0
        return (self.tp + 0.5 * self.partial) / (self.tp + self.partial + self.fn)
    
    @property
    def partial_f1(self) -> float:
        p, r = self.partial_precision, self.partial_recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class EvaluationResult:
    """完整评估结果"""
    doc_id: str
    metrics_by_type: Dict[str, TypeMetrics]
    overall_metrics: TypeMetrics
    match_results: List[MatchResult]
    
    # 统计
    gold_count: int = 0
    pred_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'gold_count': self.gold_count,
            'pred_count': self.pred_count,
            'overall': {
                'precision': self.overall_metrics.precision,
                'recall': self.overall_metrics.recall,
                'f1': self.overall_metrics.f1,
                'partial_f1': self.overall_metrics.partial_f1,
            },
            'by_type': {
                t: {
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1': m.f1,
                    'tp': m.tp, 'fp': m.fp, 'fn': m.fn, 'partial': m.partial
                }
                for t, m in self.metrics_by_type.items()
            }
        }


class InfonEvaluator:
    """信息元评估器"""
    
    def __init__(self, 
                 exact_match_threshold: float = 0.95,
                 partial_match_threshold: float = 0.5,
                 case_sensitive: bool = False):
        """
        初始化评估器
        
        Args:
            exact_match_threshold: 精确匹配的相似度阈值
            partial_match_threshold: 部分匹配的相似度阈值
            case_sensitive: 是否区分大小写
        """
        self.exact_threshold = exact_match_threshold
        self.partial_threshold = partial_match_threshold
        self.case_sensitive = case_sensitive
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本用于比较"""
        if not text:
            return ''
        
        text = str(text).strip()
        if not self.case_sensitive:
            text = text.lower()
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        t1 = self._normalize_text(text1)
        t2 = self._normalize_text(text2)
        
        if not t1 or not t2:
            return 0.0
        
        if t1 == t2:
            return 1.0
        
        # 使用SequenceMatcher计算相似度
        return SequenceMatcher(None, t1, t2).ratio()
    
    def _match_desc(self, gold: DescInfon, pred: Dict) -> Tuple[str, float, Dict]:
        """
        匹配DESC信息元
        
        Returns:
            (match_type, similarity_score, details)
        """
        pred_type = str(pred.get('infon_type', '')).upper()
        if pred_type != 'DESC':
            return ('type_mismatch', 0.0, {})
        
        # 比较entity（类别）
        entity_sim = self._text_similarity(gold.entity, pred.get('entity', ''))
        
        # 比较attribute（值）- 这是主要的匹配依据
        attr_sim = self._text_similarity(gold.attribute, pred.get('attribute', ''))
        
        # 综合相似度（attribute权重更高）
        overall_sim = 0.3 * entity_sim + 0.7 * attr_sim
        
        details = {
            'entity_similarity': entity_sim,
            'attribute_similarity': attr_sim,
            'gold_entity': gold.entity,
            'gold_attribute': gold.attribute,
            'pred_entity': pred.get('entity', ''),
            'pred_attribute': pred.get('attribute', ''),
        }
        
        if overall_sim >= self.exact_threshold:
            return ('exact', overall_sim, details)
        elif attr_sim >= self.exact_threshold:
            # attribute精确匹配，entity不同
            return ('exact', attr_sim, details)
        elif overall_sim >= self.partial_threshold:
            return ('partial', overall_sim, details)
        else:
            return ('miss', overall_sim, details)
    
    def _match_scen(self, gold: ScenInfon, pred: Dict) -> Tuple[str, float, Dict]:
        """匹配SCEN信息元"""
        pred_type = str(pred.get('infon_type', '')).upper()
        if pred_type != 'SCEN':
            return ('type_mismatch', 0.0, {})
        
        # 比较temporal
        temporal_sim = self._text_similarity(gold.temporal, pred.get('temporal', ''))
        
        # 比较spatial
        spatial_sim = self._text_similarity(gold.spatial, pred.get('spatial', ''))
        
        # 综合相似度
        overall_sim = 0.5 * temporal_sim + 0.5 * spatial_sim
        
        details = {
            'temporal_similarity': temporal_sim,
            'spatial_similarity': spatial_sim,
            'gold_temporal': gold.temporal,
            'gold_spatial': gold.spatial,
            'pred_temporal': pred.get('temporal', ''),
            'pred_spatial': pred.get('spatial', ''),
        }
        
        if overall_sim >= self.exact_threshold:
            return ('exact', overall_sim, details)
        elif overall_sim >= self.partial_threshold:
            return ('partial', overall_sim, details)
        else:
            return ('miss', overall_sim, details)
    
    def _match_rel(self, gold: RelInfon, pred: Dict, 
                   gold_iid_to_attr: Dict, pred_iid_to_attr: Dict) -> Tuple[str, float, Dict]:
        """
        匹配REL信息元
        
        由于REL的arg_refs是对其他infon的引用，需要解析引用来比较
        """
        pred_type = str(pred.get('infon_type', '')).upper()
        if pred_type != 'REL':
            return ('type_mismatch', 0.0, {})
        
        # 比较relation_name
        rel_sim = self._text_similarity(gold.relation_name, pred.get('relation_name', ''))
        
        # 比较arg_refs（通过解析引用的实际内容）
        gold_args = [gold_iid_to_attr.get(ref, ref) for ref in gold.arg_refs]
        pred_args = [pred_iid_to_attr.get(ref, ref) for ref in pred.get('arg_refs', [])]
        
        # 计算参数匹配度（考虑顺序和内容）
        args_sim = 0.0
        if gold_args and pred_args:
            # 尝试找到最佳匹配
            matched = 0
            for ga in gold_args:
                best_match = max((self._text_similarity(ga, pa) for pa in pred_args), default=0)
                if best_match >= self.partial_threshold:
                    matched += 1
            args_sim = matched / max(len(gold_args), len(pred_args))
        
        overall_sim = 0.4 * rel_sim + 0.6 * args_sim
        
        details = {
            'relation_similarity': rel_sim,
            'args_similarity': args_sim,
            'gold_relation': gold.relation_name,
            'gold_args': gold_args,
            'pred_relation': pred.get('relation_name', ''),
            'pred_args': pred_args,
        }
        
        if overall_sim >= self.exact_threshold:
            return ('exact', overall_sim, details)
        elif overall_sim >= self.partial_threshold:
            return ('partial', overall_sim, details)
        else:
            return ('miss', overall_sim, details)
    
    def _build_iid_to_attr_map(self, infons: List) -> Dict[str, str]:
        """构建iid到attribute的映射，用于REL评估"""
        mapping = {}
        for inf in infons:
            if isinstance(inf, Infon):
                iid = inf.iid
                if isinstance(inf, DescInfon):
                    mapping[iid] = inf.attribute
                elif isinstance(inf, ScenInfon):
                    mapping[iid] = f"{inf.temporal}@{inf.spatial}"
            elif isinstance(inf, dict):
                iid = inf.get('iid', '')
                infon_type = str(inf.get('infon_type', '')).upper()
                if infon_type == 'DESC':
                    mapping[iid] = inf.get('attribute', '')
                elif infon_type == 'SCEN':
                    mapping[iid] = f"{inf.get('temporal', '')}@{inf.get('spatial', '')}"
        return mapping
    
    def evaluate_single(self, gold_sample: GoldSample, predictions: List[Dict]) -> EvaluationResult:
        """
        评估单个样本
        
        Args:
            gold_sample: Gold标准样本
            predictions: 系统预测的infon列表（字典格式）
            
        Returns:
            EvaluationResult
        """
        # 按类型分组
        gold_by_type = defaultdict(list)
        for inf in gold_sample.infons:
            gold_by_type[inf.infon_type].append(inf)
        
        pred_by_type = defaultdict(list)
        for inf in predictions:
            infon_type = str(inf.get('infon_type', '')).upper()
            pred_by_type[infon_type].append(inf)
        
        # 构建iid到attribute的映射
        gold_iid_map = self._build_iid_to_attr_map(gold_sample.infons)
        pred_iid_map = self._build_iid_to_attr_map(predictions)
        
        metrics_by_type = {}
        all_match_results = []
        overall_metrics = TypeMetrics()
        
        for infon_type in ['DESC', 'SCEN', 'REL']:
            golds = gold_by_type.get(infon_type, [])
            preds = pred_by_type.get(infon_type, [])
            
            metrics = TypeMetrics()
            
            # 用于追踪已匹配的预测
            matched_preds = set()
            
            for gold in golds:
                best_match_type = 'miss'
                best_sim = 0.0
                best_pred_idx = -1
                best_details = {}
                
                for idx, pred in enumerate(preds):
                    if idx in matched_preds:
                        continue
                    
                    # 根据类型选择匹配方法
                    if infon_type == 'DESC':
                        match_type, sim, details = self._match_desc(gold, pred)
                    elif infon_type == 'SCEN':
                        match_type, sim, details = self._match_scen(gold, pred)
                    elif infon_type == 'REL':
                        match_type, sim, details = self._match_rel(
                            gold, pred, gold_iid_map, pred_iid_map
                        )
                    else:
                        continue
                    
                    if sim > best_sim:
                        best_match_type = match_type
                        best_sim = sim
                        best_pred_idx = idx
                        best_details = details
                
                # 记录匹配结果
                matched_pred = preds[best_pred_idx] if best_pred_idx >= 0 else None
                
                if best_match_type == 'exact':
                    metrics.tp += 1
                    overall_metrics.tp += 1
                    matched_preds.add(best_pred_idx)
                elif best_match_type == 'partial':
                    metrics.partial += 1
                    overall_metrics.partial += 1
                    matched_preds.add(best_pred_idx)
                else:
                    metrics.fn += 1
                    overall_metrics.fn += 1
                
                all_match_results.append(MatchResult(
                    gold_infon=gold,
                    pred_infon=matched_pred,
                    match_type=best_match_type,
                    similarity_score=best_sim,
                    details=best_details
                ))
            
            # 未匹配的预测为FP
            for idx, pred in enumerate(preds):
                if idx not in matched_preds:
                    metrics.fp += 1
                    overall_metrics.fp += 1
                    all_match_results.append(MatchResult(
                        gold_infon=None,
                        pred_infon=pred,
                        match_type='spurious',
                        similarity_score=0.0
                    ))
            
            metrics_by_type[infon_type] = metrics
        
        return EvaluationResult(
            doc_id=gold_sample.doc_id,
            metrics_by_type=metrics_by_type,
            overall_metrics=overall_metrics,
            match_results=all_match_results,
            gold_count=len(gold_sample.infons),
            pred_count=len(predictions)
        )
    
    def evaluate_batch(self, gold_samples: List[GoldSample], 
                       predictions_list: List[List[Dict]]) -> Dict:
        """
        批量评估
        
        Args:
            gold_samples: Gold样本列表
            predictions_list: 每个样本对应的预测列表
            
        Returns:
            汇总的评估结果
        """
        assert len(gold_samples) == len(predictions_list)
        
        results = []
        for gold, preds in zip(gold_samples, predictions_list):
            result = self.evaluate_single(gold, preds)
            results.append(result)
        
        # 汇总指标
        total_by_type = {t: TypeMetrics() for t in ['DESC', 'SCEN', 'REL']}
        total_overall = TypeMetrics()
        
        for result in results:
            for t, m in result.metrics_by_type.items():
                total_by_type[t].tp += m.tp
                total_by_type[t].fp += m.fp
                total_by_type[t].fn += m.fn
                total_by_type[t].partial += m.partial
            
            total_overall.tp += result.overall_metrics.tp
            total_overall.fp += result.overall_metrics.fp
            total_overall.fn += result.overall_metrics.fn
            total_overall.partial += result.overall_metrics.partial
        
        return {
            'num_samples': len(results),
            'overall': {
                'precision': total_overall.precision,
                'recall': total_overall.recall,
                'f1': total_overall.f1,
                'partial_f1': total_overall.partial_f1,
                'tp': total_overall.tp,
                'fp': total_overall.fp,
                'fn': total_overall.fn,
                'partial': total_overall.partial,
            },
            'by_type': {
                t: {
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1': m.f1,
                    'partial_f1': m.partial_f1,
                    'tp': m.tp, 'fp': m.fp, 'fn': m.fn, 'partial': m.partial
                }
                for t, m in total_by_type.items()
            },
            'per_sample': [r.to_dict() for r in results]
        }


def parse_compact_format(text: str) -> List[Dict]:
    """
    解析compact格式的infon输出
    
    格式示例：
    desc:r1_1,DESC,姓名,王小明,string,0.95
    scen:r1_2,SCEN,今年,北京,0.90
    rel:r1_3,REL,个人信息,desc:r1_1|desc:r1_2,0.90
    """
    infons = []
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 跳过非infon行
        if not line.startswith(('desc:', 'scen:', 'rel:')):
            continue
        
        # 解析逗号分隔的字段（处理转义）
        parts = []
        current = ''
        escaped = False
        
        for ch in line:
            if escaped:
                current += ch
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == ',':
                parts.append(current)
                current = ''
            else:
                current += ch
        parts.append(current)
        
        if len(parts) < 3:
            continue
        
        iid = parts[0]
        infon_type = parts[1].upper()
        
        if infon_type == 'DESC' and len(parts) >= 6:
            infons.append({
                'iid': iid,
                'infon_type': 'DESC',
                'entity': parts[2],
                'attribute': parts[3],
                'data_type': parts[4],
                'confidence': float(parts[5]) if parts[5] and parts[5].replace('.','',1).isdigit() else 0.9
            })
        elif infon_type == 'SCEN' and len(parts) >= 5:
            infons.append({
                'iid': iid,
                'infon_type': 'SCEN',
                'temporal': parts[2],
                'spatial': parts[3],
                'confidence': float(parts[4]) if parts[4] and parts[4].replace('.','',1).isdigit() else 0.9
            })
        elif infon_type == 'REL' and len(parts) >= 5:
            arg_refs = parts[3].split('|') if parts[3] else []
            infons.append({
                'iid': iid,
                'infon_type': 'REL',
                'relation_name': parts[2],
                'arg_refs': arg_refs,
                'confidence': float(parts[4]) if parts[4] and parts[4].replace('.','',1).isdigit() else 0.9
            })
    
    return infons


def print_evaluation_report(result: Dict, detailed: bool = False):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("                    信息元提取评估报告")
    print("=" * 70)
    
    print(f"\n样本数量: {result['num_samples']}")
    
    print("\n【整体指标】")
    print("-" * 50)
    o = result['overall']
    print(f"  Precision: {o['precision']:.4f}")
    print(f"  Recall:    {o['recall']:.4f}")
    print(f"  F1:        {o['f1']:.4f}")
    print(f"  Partial-F1:{o['partial_f1']:.4f}")
    print(f"  (TP={o['tp']}, FP={o['fp']}, FN={o['fn']}, Partial={o['partial']})")
    
    print("\n【分类型指标】")
    print("-" * 50)
    for t, m in result['by_type'].items():
        print(f"\n  {t}:")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    F1:        {m['f1']:.4f}")
        print(f"    (TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, Partial={m['partial']})")
    
    if detailed and 'per_sample' in result:
        print("\n【每个样本详情】")
        print("-" * 50)
        for sample in result['per_sample'][:10]:  # 只显示前10个
            print(f"\n  {sample['doc_id']}:")
            print(f"    Gold: {sample['gold_count']}, Pred: {sample['pred_count']}")
            print(f"    F1: {sample['overall']['f1']:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    # 测试评估器
    from .ace_parser import ACEParser
    from .ace_to_infons import ACEToInfonsConverter
    
    # 加载一些样本
    parser = ACEParser('../test-data/ace_2005_td_v7')
    converter = ACEToInfonsConverter()
    
    docs = parser.parse_all(annotation_level='adj', limit=5)
    samples = converter.convert_all(docs)
    
    # 创建评估器
    evaluator = InfonEvaluator()
    
    # 模拟预测（使用gold作为预测来测试）
    predictions_list = []
    for sample in samples:
        preds = [infon_to_dict(inf) for inf in sample.infons]
        predictions_list.append(preds)
    
    # 评估
    result = evaluator.evaluate_batch(samples, predictions_list)
    print_evaluation_report(result, detailed=True)
