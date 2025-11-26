#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACE 2005 到 Infons 格式转换器
将 ACE 2005 标注转换为 PrivaSee 的 DESC/SCEN/REL 信息元格式
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import json
import csv
import hashlib

from .ace_parser import (
    ACEParser, ACEDocument, Entity, EntityMention,
    Relation, Event, Timex, Value, CharSpan
)


@dataclass
class Infon:
    """信息元基类"""
    iid: str
    infon_type: str
    confidence: float = 1.0
    source_ace_id: str = ''  # 原始ACE ID，用于追溯
    char_start: int = -1     # 字符起始位置
    char_end: int = -1       # 字符结束位置


@dataclass
class DescInfon(Infon):
    """DESC信息元 - 实体与属性"""
    entity: str = ''      # 类别（如 Person, Organization）
    attribute: str = ''   # 具体值（原文提取）
    data_type: str = 'string'
    
    def __post_init__(self):
        self.infon_type = 'DESC'


@dataclass
class ScenInfon(Infon):
    """SCEN信息元 - 时空场景"""
    temporal: str = ''    # 时间
    spatial: str = ''     # 空间
    temporal_start: int = -1
    temporal_end: int = -1
    spatial_start: int = -1
    spatial_end: int = -1
    
    def __post_init__(self):
        self.infon_type = 'SCEN'


@dataclass
class RelInfon(Infon):
    """REL信息元 - 关系"""
    relation_name: str = ''
    arg_refs: List[str] = field(default_factory=list)
    arity: int = 0
    
    def __post_init__(self):
        self.infon_type = 'REL'
        self.arity = len(self.arg_refs)


@dataclass
class GoldSample:
    """Gold标准样本"""
    doc_id: str
    text: str
    infons: List[Infon]
    source_file: str = ''
    language: str = ''
    
    # ACE原始统计
    ace_entities: int = 0
    ace_relations: int = 0
    ace_events: int = 0


class ACEToInfonsConverter:
    """ACE 2005到Infons格式转换器"""
    
    # ACE实体类型到DESC entity映射
    ENTITY_TYPE_MAP = {
        'PER': 'Person',
        'ORG': 'Organization', 
        'GPE': 'GeoPoliticalEntity',
        'LOC': 'Location',
        'FAC': 'Facility',
        'VEH': 'Vehicle',
        'WEA': 'Weapon',
    }
    
    # ACE实体子类型细化映射（更详细的类别）
    ENTITY_SUBTYPE_MAP = {
        ('PER', 'Individual'): 'Individual',
        ('PER', 'Group'): 'Group',
        ('ORG', 'Government'): 'Government',
        ('ORG', 'Commercial'): 'Commercial',
        ('ORG', 'Educational'): 'Educational',
        ('ORG', 'Non-Governmental'): 'NGO',
        ('ORG', 'Media'): 'Media',
        ('ORG', 'Religious'): 'Religious',
        ('ORG', 'Medical-Science'): 'Medical',
        ('GPE', 'Nation'): 'Nation',
        ('GPE', 'State-or-Province'): 'Province',
        ('GPE', 'County-or-District'): 'District',
        ('GPE', 'Population-Center'): 'City',
        ('LOC', 'Address'): 'Address',
        ('LOC', 'Boundary'): 'Boundary',
        ('LOC', 'Celestial'): 'Celestial',
        ('LOC', 'Water-Body'): 'WaterBody',
        ('LOC', 'Land-Region-Natural'): 'LandRegion',
        ('LOC', 'Region-International'): 'InternationalRegion',
        ('LOC', 'Region-General'): 'GeneralRegion',
        ('FAC', 'Airport'): 'Airport',
        ('FAC', 'Building-Grounds'): 'Building',
        ('FAC', 'Path'): 'Path',
        ('FAC', 'Plant'): 'Plant',
        ('FAC', 'Subarea-Facility'): 'SubareaFacility',
    }
    
    # ACE关系类型到REL relation_name映射
    RELATION_TYPE_MAP = {
        # Physical relations
        ('PHYS', 'Located'): 'located_at',
        ('PHYS', 'Near'): 'near',
        # Part-Whole relations
        ('PART-WHOLE', 'Geographical'): 'part_of_geo',
        ('PART-WHOLE', 'Subsidiary'): 'subsidiary_of',
        ('PART-WHOLE', 'Artifact'): 'part_of_artifact',
        # Personal-Social relations
        ('PER-SOC', 'Business'): 'business_relation',
        ('PER-SOC', 'Family'): 'family_relation',
        ('PER-SOC', 'Lasting-Personal'): 'personal_relation',
        # Organization-Affiliation relations
        ('ORG-AFF', 'Employment'): 'employed_by',
        ('ORG-AFF', 'Ownership'): 'owns',
        ('ORG-AFF', 'Founder'): 'founded_by',
        ('ORG-AFF', 'Student-Alum'): 'student_of',
        ('ORG-AFF', 'Sports-Affiliation'): 'affiliated_with_sports',
        ('ORG-AFF', 'Investor-Shareholder'): 'investor_in',
        ('ORG-AFF', 'Membership'): 'member_of',
        # Agent-Artifact relations
        ('ART', 'User-Owner-Inventor-Manufacturer'): 'uses_or_owns',
        # General-Affiliation relations
        ('GEN-AFF', 'Citizen-Resident-Religion-Ethnicity'): 'citizen_of',
        ('GEN-AFF', 'Org-Location'): 'org_located_at',
    }
    
    # ACE事件类型（用于识别时空共现）
    EVENT_TYPES_WITH_LOCATION = {
        'Movement:Transport',
        'Transaction:Transfer-Money',
        'Transaction:Transfer-Ownership',
        'Business:Start-Org',
        'Business:End-Org',
        'Business:Declare-Bankruptcy',
        'Business:Merge-Org',
        'Conflict:Attack',
        'Conflict:Demonstrate',
        'Contact:Meet',
        'Contact:Phone-Write',
        'Justice:Arrest-Jail',
        'Justice:Trial-Hearing',
        'Justice:Charge-Indict',
        'Justice:Sue',
        'Justice:Convict',
        'Justice:Sentence',
        'Justice:Fine',
        'Justice:Execute',
        'Justice:Extradite',
        'Justice:Acquit',
        'Justice:Pardon',
        'Justice:Appeal',
        'Justice:Release-Parole',
        'Life:Be-Born',
        'Life:Die',
        'Life:Marry',
        'Life:Divorce',
        'Life:Injure',
        'Personnel:Start-Position',
        'Personnel:End-Position',
        'Personnel:Nominate',
        'Personnel:Elect',
    }
    
    def __init__(self, use_subtype: bool = True, use_head_only: bool = False):
        """
        初始化转换器
        
        Args:
            use_subtype: 是否使用子类型作为entity类别
            use_head_only: 是否只使用head作为attribute（否则使用extent）
                           注意：对于中文等语言，建议使用 extent (False)，
                           因为 head 只包含核心词，会丢失修饰语信息，
                           如 "国际艺术团体" 的 head 是 "团体"，丢失了 "国际艺术"
        """
        self.use_subtype = use_subtype
        self.use_head_only = use_head_only
        self._iid_counter = 0
        self._entity_to_iid: Dict[str, str] = {}  # ACE entity ID -> infon iid
    
    def _generate_iid(self, prefix: str) -> str:
        """生成唯一的infon ID"""
        self._iid_counter += 1
        return f"{prefix}:gold_{self._iid_counter}"
    
    def _reset_counter(self):
        """重置计数器（每个文档重置）"""
        self._iid_counter = 0
        self._entity_to_iid.clear()
    
    def _get_entity_category(self, entity: Entity) -> str:
        """获取实体的类别名称"""
        if self.use_subtype:
            key = (entity.entity_type, entity.subtype)
            if key in self.ENTITY_SUBTYPE_MAP:
                return self.ENTITY_SUBTYPE_MAP[key]
        
        return self.ENTITY_TYPE_MAP.get(entity.entity_type, entity.entity_type)
    
    def _get_relation_name(self, relation: Relation) -> str:
        """获取关系名称"""
        key = (relation.relation_type, relation.subtype)
        return self.RELATION_TYPE_MAP.get(key, f"{relation.relation_type}_{relation.subtype}")
    
    def _clean_text(self, text: str) -> str:
        """清理文本：去除多余空格和换行符"""
        if not text:
            return ''
        import re as regex
        # 将换行符替换为空格
        text = text.replace('\n', '').replace('\r', '')
        # 去除中文字符之间的空格 (保留英文单词间的空格)
        text = regex.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        # 压缩连续空格
        text = regex.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _convert_entity_to_desc(self, entity: Entity, doc: ACEDocument) -> List[DescInfon]:
        """
        将ACE实体转换为DESC信息元
        每个entity_mention生成一个DESC（但共享同一个iid用于去重）
        """
        infons = []
        category = self._get_entity_category(entity)
        
        # 对于每个mention，生成一个DESC
        # 但我们只为每个unique的cleaned text生成一个
        seen_texts = set()
        
        for mention in entity.mentions:
            # 默认使用extent（完整短语），可选head（核心词）
            if self.use_head_only:
                span = mention.head
            else:
                span = mention.extent
            
            # 清理文本：去除换行符和多余空格
            attribute = self._clean_text(span.text)
            
            # 跳过太短或重复的
            if len(attribute) < 1:
                continue
            if attribute in seen_texts:
                continue
            seen_texts.add(attribute)
            
            # 生成IID
            iid = self._generate_iid('desc')
            
            # 记录映射（使用mention ID）
            self._entity_to_iid[mention.id] = iid
            
            infon = DescInfon(
                iid=iid,
                infon_type='DESC',
                entity=category,
                attribute=attribute,
                data_type='string',
                confidence=1.0,
                source_ace_id=mention.id,
                char_start=span.start,
                char_end=span.end
            )
            infons.append(infon)
        
        # 同时记录entity ID的映射（指向第一个mention的iid）
        if infons:
            self._entity_to_iid[entity.id] = infons[0].iid
        
        return infons
    
    def _convert_timex_to_desc(self, timex: Timex) -> List[DescInfon]:
        """将时间表达式转换为DESC信息元"""
        infons = []
        
        for mention in timex.mentions:
            attribute = self._clean_text(mention.extent.text)
            if len(attribute) < 1:
                continue
            
            iid = self._generate_iid('desc')
            self._entity_to_iid[mention.id] = iid
            self._entity_to_iid[timex.id] = iid
            
            infon = DescInfon(
                iid=iid,
                infon_type='DESC',
                entity='Time',
                attribute=attribute,
                data_type='datetime',
                confidence=1.0,
                source_ace_id=mention.id,
                char_start=mention.extent.start,
                char_end=mention.extent.end
            )
            infons.append(infon)
        
        return infons
    
    def _convert_value_to_desc(self, value: Value) -> List[DescInfon]:
        """将值转换为DESC信息元"""
        infons = []
        
        for mention in value.mentions:
            attribute = self._clean_text(mention.extent.text)
            if len(attribute) < 1:
                continue
            
            iid = self._generate_iid('desc')
            self._entity_to_iid[mention.id] = iid
            self._entity_to_iid[value.id] = iid
            
            # 确定数据类型
            data_type = 'string'
            if value.value_type == 'Numeric':
                data_type = 'number'
            elif value.value_type == 'Money':
                data_type = 'currency'
            elif value.value_type == 'Percent':
                data_type = 'percent'
            
            entity_name = value.subtype or value.value_type
            
            infon = DescInfon(
                iid=iid,
                infon_type='DESC',
                entity=entity_name,
                attribute=attribute,
                data_type=data_type,
                confidence=1.0,
                source_ace_id=mention.id,
                char_start=mention.extent.start,
                char_end=mention.extent.end
            )
            infons.append(infon)
        
        return infons
    
    def _convert_relation_to_rel(self, relation: Relation) -> List[RelInfon]:
        """将ACE关系转换为REL信息元"""
        infons = []
        
        relation_name = self._get_relation_name(relation)
        
        # 获取论元的iid引用
        arg_refs = []
        if relation.arg1_id and relation.arg1_id in self._entity_to_iid:
            arg_refs.append(self._entity_to_iid[relation.arg1_id])
        if relation.arg2_id and relation.arg2_id in self._entity_to_iid:
            arg_refs.append(self._entity_to_iid[relation.arg2_id])
        
        # 只有当两个论元都有对应的iid时才生成REL
        if len(arg_refs) >= 2:
            iid = self._generate_iid('rel')
            
            infon = RelInfon(
                iid=iid,
                infon_type='REL',
                relation_name=relation_name,
                arg_refs=arg_refs,
                arity=len(arg_refs),
                confidence=1.0,
                source_ace_id=relation.id
            )
            infons.append(infon)
        
        return infons
    
    def _convert_event_to_scen_and_rel(self, event: Event, doc: ACEDocument) -> Tuple[List[ScenInfon], List[RelInfon]]:
        """
        将ACE事件转换为SCEN和REL信息元
        
        当事件同时有Time和Place论元时，生成SCEN
        事件参与者之间的关系生成REL
        """
        scen_infons = []
        rel_infons = []
        
        # 检查事件是否有Time和Place论元
        time_id = None
        place_id = None
        
        for role, ref_id in event.args.items():
            if 'Time' in role:
                time_id = ref_id
            elif role in ('Place', 'Destination', 'Origin'):
                place_id = ref_id
        
        # 如果同时有时间和地点，生成SCEN
        if time_id and place_id:
            # 获取时间文本
            temporal = ''
            temporal_start = -1
            temporal_end = -1
            if time_id in doc.timexes:
                timex = doc.timexes[time_id]
                if timex.mentions:
                    temporal = timex.mentions[0].extent.text
                    temporal_start = timex.mentions[0].extent.start
                    temporal_end = timex.mentions[0].extent.end
            
            # 获取地点文本
            spatial = ''
            spatial_start = -1
            spatial_end = -1
            # place_id可能指向entity或GPE
            if place_id in doc.entities:
                entity = doc.entities[place_id]
                if entity.mentions:
                    span = entity.mentions[0].head if self.use_head_only else entity.mentions[0].extent
                    spatial = span.text
                    spatial_start = span.start
                    spatial_end = span.end
            
            if temporal and spatial:
                iid = self._generate_iid('scen')
                scen = ScenInfon(
                    iid=iid,
                    infon_type='SCEN',
                    temporal=temporal,
                    spatial=spatial,
                    temporal_start=temporal_start,
                    temporal_end=temporal_end,
                    spatial_start=spatial_start,
                    spatial_end=spatial_end,
                    confidence=1.0,
                    source_ace_id=event.id
                )
                scen_infons.append(scen)
        
        # 生成事件参与者之间的关系
        event_type = f"{event.event_type}:{event.subtype}"
        participant_iids = []
        
        for role, ref_id in event.args.items():
            if 'Time' not in role:  # 排除时间论元
                if ref_id in self._entity_to_iid:
                    participant_iids.append(self._entity_to_iid[ref_id])
        
        # 如果有多个参与者，生成事件关系
        if len(participant_iids) >= 2:
            iid = self._generate_iid('rel')
            rel = RelInfon(
                iid=iid,
                infon_type='REL',
                relation_name=event_type.replace(':', '_').lower(),
                arg_refs=participant_iids,
                arity=len(participant_iids),
                confidence=1.0,
                source_ace_id=event.id
            )
            rel_infons.append(rel)
        
        return scen_infons, rel_infons
    
    def convert_document(self, doc: ACEDocument) -> GoldSample:
        """
        将单个ACE文档转换为Gold样本
        
        Args:
            doc: ACEDocument对象
            
        Returns:
            GoldSample对象
        """
        self._reset_counter()
        
        all_infons: List[Infon] = []
        
        # 1. 转换实体为DESC
        for entity in doc.entities.values():
            desc_infons = self._convert_entity_to_desc(entity, doc)
            all_infons.extend(desc_infons)
        
        # 2. 转换时间表达式为DESC
        for timex in doc.timexes.values():
            desc_infons = self._convert_timex_to_desc(timex)
            all_infons.extend(desc_infons)
        
        # 3. 转换值为DESC
        for value in doc.values.values():
            desc_infons = self._convert_value_to_desc(value)
            all_infons.extend(desc_infons)
        
        # 4. 转换关系为REL
        for relation in doc.relations.values():
            rel_infons = self._convert_relation_to_rel(relation)
            all_infons.extend(rel_infons)
        
        # 5. 转换事件为SCEN和REL
        for event in doc.events.values():
            scen_infons, rel_infons = self._convert_event_to_scen_and_rel(event, doc)
            all_infons.extend(scen_infons)
            all_infons.extend(rel_infons)
        
        # 创建Gold样本
        sample = GoldSample(
            doc_id=doc.doc_id,
            text=doc.raw_text,
            infons=all_infons,
            source_file=doc.source_file,
            language=self._detect_language(doc),
            ace_entities=len(doc.entities),
            ace_relations=len(doc.relations),
            ace_events=len(doc.events)
        )
        
        return sample
    
    def _detect_language(self, doc: ACEDocument) -> str:
        """检测文档语言"""
        # 优先使用解析时记录的语言
        if hasattr(doc, 'language') and doc.language:
            return doc.language
        return 'English'
    
    def convert_all(self, documents: List[ACEDocument]) -> List[GoldSample]:
        """转换所有文档"""
        return [self.convert_document(doc) for doc in documents]


def infon_to_compact_format(infon: Infon) -> str:
    """
    将Infon转换为compact格式字符串（与PrivaSee前端格式一致）
    """
    if isinstance(infon, DescInfon):
        # DESC: iid,DESC,entity,attribute,data_type,confidence
        return f"{infon.iid},DESC,{infon.entity},{infon.attribute},{infon.data_type},{infon.confidence:.2f}"
    
    elif isinstance(infon, ScenInfon):
        # SCEN: iid,SCEN,temporal,spatial,confidence
        return f"{infon.iid},SCEN,{infon.temporal},{infon.spatial},{infon.confidence:.2f}"
    
    elif isinstance(infon, RelInfon):
        # REL: iid,REL,relation_name,arg_refs,confidence
        arg_refs_str = '|'.join(infon.arg_refs)
        return f"{infon.iid},REL,{infon.relation_name},{arg_refs_str},{infon.confidence:.2f}"
    
    return ''


def infon_to_dict(infon: Infon) -> Dict:
    """将Infon转换为字典格式"""
    base = {
        'iid': infon.iid,
        'infon_type': infon.infon_type,
        'confidence': infon.confidence,
        'source_ace_id': infon.source_ace_id,
        'char_start': infon.char_start,
        'char_end': infon.char_end,
    }
    
    if isinstance(infon, DescInfon):
        base.update({
            'entity': infon.entity,
            'attribute': infon.attribute,
            'data_type': infon.data_type,
        })
    elif isinstance(infon, ScenInfon):
        base.update({
            'temporal': infon.temporal,
            'spatial': infon.spatial,
        })
    elif isinstance(infon, RelInfon):
        base.update({
            'relation_name': infon.relation_name,
            'arg_refs': infon.arg_refs,
            'arity': infon.arity,
        })
    
    return base


def sample_to_dict(sample: GoldSample) -> Dict:
    """将GoldSample转换为字典格式"""
    return {
        'doc_id': sample.doc_id,
        'text': sample.text,
        'source_file': sample.source_file,
        'language': sample.language,
        'ace_entities': sample.ace_entities,
        'ace_relations': sample.ace_relations,
        'ace_events': sample.ace_events,
        'infons': [infon_to_dict(inf) for inf in sample.infons],
        'statistics': {
            'total_infons': len(sample.infons),
            'desc_count': sum(1 for inf in sample.infons if inf.infon_type == 'DESC'),
            'scen_count': sum(1 for inf in sample.infons if inf.infon_type == 'SCEN'),
            'rel_count': sum(1 for inf in sample.infons if inf.infon_type == 'REL'),
        }
    }


def export_to_json(samples: List[GoldSample], output_path: str):
    """导出为JSON格式"""
    data = [sample_to_dict(sample) for sample in samples]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_to_csv(samples: List[GoldSample], output_path: str):
    """导出为CSV格式（用于与cases.csv兼容的评估）"""
    rows = []
    
    for sample in samples:
        # 生成compact格式的infons
        compact_infons = '\n'.join(infon_to_compact_format(inf) for inf in sample.infons)
        
        rows.append({
            'id': sample.doc_id,
            'text': sample.text,  # 完整文本
            'language': sample.language,
            'source': sample.source_file,
            'gold_infons': compact_infons,
            'gold_infons_json': json.dumps([infon_to_dict(inf) for inf in sample.infons], ensure_ascii=False),
            'desc_count': sum(1 for inf in sample.infons if inf.infon_type == 'DESC'),
            'scen_count': sum(1 for inf in sample.infons if inf.infon_type == 'SCEN'),
            'rel_count': sum(1 for inf in sample.infons if inf.infon_type == 'REL'),
        })
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    import sys
    
    # 测试转换
    ace_root = sys.argv[1] if len(sys.argv) > 1 else '../test-data/ace_2005_td_v7'
    
    parser = ACEParser(ace_root)
    converter = ACEToInfonsConverter(use_subtype=True, use_head_only=False)
    
    # 解析并转换
    documents = parser.parse_all(annotation_level='adj', limit=5)
    samples = converter.convert_all(documents)
    
    print(f"转换了 {len(samples)} 个文档")
    
    for sample in samples[:2]:
        print(f"\n=== {sample.doc_id} ===")
        print(f"文本长度: {len(sample.text)}")
        print(f"DESC: {sum(1 for inf in sample.infons if inf.infon_type == 'DESC')}")
        print(f"SCEN: {sum(1 for inf in sample.infons if inf.infon_type == 'SCEN')}")
        print(f"REL: {sum(1 for inf in sample.infons if inf.infon_type == 'REL')}")
        print("\n前5个infons:")
        for inf in sample.infons[:5]:
            print(f"  {infon_to_compact_format(inf)}")
