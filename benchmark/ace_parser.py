#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACE 2005 XML 解析器
解析 .apf.xml 和 .sgm 文件，提取实体、关系、事件、时间表达式等标注
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from html import unescape


@dataclass
class CharSpan:
    """字符位置跨度"""
    start: int
    end: int
    text: str


@dataclass
class EntityMention:
    """实体提及"""
    id: str
    mention_type: str  # NAM, NOM, PRO
    extent: CharSpan
    head: CharSpan


@dataclass
class Entity:
    """实体"""
    id: str
    entity_type: str      # PER, ORG, GPE, LOC, FAC, VEH, WEA
    subtype: str          # Individual, Group, etc.
    entity_class: str     # SPC (specific), GEN (generic), etc.
    mentions: List[EntityMention] = field(default_factory=list)
    names: List[str] = field(default_factory=list)  # 规范化名称


@dataclass
class TimexMention:
    """时间表达式提及"""
    id: str
    extent: CharSpan
    val: Optional[str] = None  # 规范化值 (如 2000-10-14)


@dataclass
class Timex:
    """时间表达式"""
    id: str
    mentions: List[TimexMention] = field(default_factory=list)


@dataclass
class ValueMention:
    """值提及（数字、金额等）"""
    id: str
    extent: CharSpan


@dataclass
class Value:
    """值"""
    id: str
    value_type: str  # Numeric, Money, etc.
    subtype: Optional[str] = None
    mentions: List[ValueMention] = field(default_factory=list)


@dataclass
class RelationMentionArg:
    """关系提及论元"""
    refid: str
    role: str
    extent: Optional[CharSpan] = None


@dataclass
class RelationMention:
    """关系提及"""
    id: str
    extent: CharSpan
    args: List[RelationMentionArg] = field(default_factory=list)


@dataclass
class Relation:
    """关系"""
    id: str
    relation_type: str    # PHYS, PART-WHOLE, PER-SOC, ORG-AFF, ART, GEN-AFF
    subtype: str
    tense: Optional[str] = None
    modality: Optional[str] = None
    arg1_id: Optional[str] = None  # 实体ID
    arg2_id: Optional[str] = None
    mentions: List[RelationMention] = field(default_factory=list)


@dataclass
class EventMentionArg:
    """事件提及论元"""
    refid: str
    role: str  # 如 Victim, Place, Time-Within, Agent, etc.
    extent: Optional[CharSpan] = None


@dataclass
class EventMention:
    """事件提及"""
    id: str
    extent: CharSpan
    anchor: CharSpan  # 触发词
    args: List[EventMentionArg] = field(default_factory=list)


@dataclass
class Event:
    """事件"""
    id: str
    event_type: str      # Life, Movement, Transaction, Business, Conflict, etc.
    subtype: str         # Die, Transport, Transfer-Money, etc.
    modality: str        # Asserted, Other
    polarity: str        # Positive, Negative
    genericity: str      # Specific, Generic
    tense: str           # Past, Present, Future, Unspecified
    args: Dict[str, str] = field(default_factory=dict)  # role -> entity/timex/value ID
    mentions: List[EventMention] = field(default_factory=list)


@dataclass
class ACEDocument:
    """ACE文档"""
    doc_id: str
    source: str           # newswire, broadcast_news, etc.
    source_file: str
    language: str = 'English'  # Arabic, Chinese, English
    raw_text: str = ''    # 原始文本
    entities: Dict[str, Entity] = field(default_factory=dict)
    timexes: Dict[str, Timex] = field(default_factory=dict)
    values: Dict[str, Value] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    events: Dict[str, Event] = field(default_factory=dict)


class ACEParser:
    """ACE 2005 XML解析器"""
    
    def __init__(self, ace_root: str):
        """
        初始化解析器
        
        Args:
            ace_root: ACE 2005数据集根目录路径
        """
        self.ace_root = Path(ace_root)
        self.data_dir = self.ace_root / "data"
    
    def _parse_charseq(self, elem: ET.Element) -> Optional[CharSpan]:
        """解析charseq元素"""
        charseq = elem.find('charseq')
        if charseq is None:
            return None
        
        start = int(charseq.get('START', 0))
        end = int(charseq.get('END', 0))
        text = charseq.text or ''
        
        return CharSpan(start=start, end=end, text=text)
    
    def _parse_entity(self, elem: ET.Element) -> Entity:
        """解析entity元素"""
        entity = Entity(
            id=elem.get('ID', ''),
            entity_type=elem.get('TYPE', ''),
            subtype=elem.get('SUBTYPE', ''),
            entity_class=elem.get('CLASS', '')
        )
        
        # 解析 entity_mention
        for mention_elem in elem.findall('entity_mention'):
            extent = self._parse_charseq(mention_elem.find('extent'))
            head = self._parse_charseq(mention_elem.find('head'))
            
            if extent and head:
                mention = EntityMention(
                    id=mention_elem.get('ID', ''),
                    mention_type=mention_elem.get('TYPE', ''),
                    extent=extent,
                    head=head
                )
                entity.mentions.append(mention)
        
        # 解析 entity_attributes (names)
        attrs = elem.find('entity_attributes')
        if attrs is not None:
            for name_elem in attrs.findall('name'):
                name = name_elem.get('NAME', '')
                if name:
                    entity.names.append(name)
        
        return entity
    
    def _parse_timex(self, elem: ET.Element) -> Timex:
        """解析timex2元素"""
        timex = Timex(id=elem.get('ID', ''))
        
        for mention_elem in elem.findall('timex2_mention'):
            extent = self._parse_charseq(mention_elem.find('extent'))
            if extent:
                mention = TimexMention(
                    id=mention_elem.get('ID', ''),
                    extent=extent
                )
                timex.mentions.append(mention)
        
        return timex
    
    def _parse_value(self, elem: ET.Element) -> Value:
        """解析value元素"""
        value = Value(
            id=elem.get('ID', ''),
            value_type=elem.get('TYPE', ''),
            subtype=elem.get('SUBTYPE')
        )
        
        for mention_elem in elem.findall('value_mention'):
            extent = self._parse_charseq(mention_elem.find('extent'))
            if extent:
                mention = ValueMention(
                    id=mention_elem.get('ID', ''),
                    extent=extent
                )
                value.mentions.append(mention)
        
        return value
    
    def _parse_relation(self, elem: ET.Element) -> Relation:
        """解析relation元素"""
        relation = Relation(
            id=elem.get('ID', ''),
            relation_type=elem.get('TYPE', ''),
            subtype=elem.get('SUBTYPE', ''),
            tense=elem.get('TENSE'),
            modality=elem.get('MODALITY')
        )
        
        # 解析关系论元
        for arg_elem in elem.findall('relation_argument'):
            role = arg_elem.get('ROLE', '')
            refid = arg_elem.get('REFID', '')
            if role == 'Arg-1':
                relation.arg1_id = refid
            elif role == 'Arg-2':
                relation.arg2_id = refid
        
        # 解析关系提及
        for mention_elem in elem.findall('relation_mention'):
            extent = self._parse_charseq(mention_elem.find('extent'))
            if extent:
                mention = RelationMention(
                    id=mention_elem.get('ID', ''),
                    extent=extent
                )
                
                for arg_elem in mention_elem.findall('relation_mention_argument'):
                    arg_extent = self._parse_charseq(arg_elem.find('extent'))
                    arg = RelationMentionArg(
                        refid=arg_elem.get('REFID', ''),
                        role=arg_elem.get('ROLE', ''),
                        extent=arg_extent
                    )
                    mention.args.append(arg)
                
                relation.mentions.append(mention)
        
        return relation
    
    def _parse_event(self, elem: ET.Element) -> Event:
        """解析event元素"""
        event = Event(
            id=elem.get('ID', ''),
            event_type=elem.get('TYPE', ''),
            subtype=elem.get('SUBTYPE', ''),
            modality=elem.get('MODALITY', ''),
            polarity=elem.get('POLARITY', ''),
            genericity=elem.get('GENERICITY', ''),
            tense=elem.get('TENSE', '')
        )
        
        # 解析事件论元
        for arg_elem in elem.findall('event_argument'):
            role = arg_elem.get('ROLE', '')
            refid = arg_elem.get('REFID', '')
            event.args[role] = refid
        
        # 解析事件提及
        for mention_elem in elem.findall('event_mention'):
            extent = self._parse_charseq(mention_elem.find('extent'))
            anchor = self._parse_charseq(mention_elem.find('anchor'))
            
            if extent and anchor:
                mention = EventMention(
                    id=mention_elem.get('ID', ''),
                    extent=extent,
                    anchor=anchor
                )
                
                for arg_elem in mention_elem.findall('event_mention_argument'):
                    arg_extent = self._parse_charseq(arg_elem.find('extent'))
                    arg = EventMentionArg(
                        refid=arg_elem.get('REFID', ''),
                        role=arg_elem.get('ROLE', ''),
                        extent=arg_extent
                    )
                    mention.args.append(arg)
                
                event.mentions.append(mention)
        
        return event
    
    def _parse_sgm(self, sgm_path: Path) -> str:
        """解析SGM文件，提取原始文本"""
        try:
            with open(sgm_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(sgm_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # 提取TEXT标签内的内容
        text_match = re.search(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL)
        if text_match:
            text = text_match.group(1)
        else:
            # 如果没有TEXT标签，提取BODY内容
            body_match = re.search(r'<BODY>(.*?)</BODY>', content, re.DOTALL)
            if body_match:
                text = body_match.group(1)
            else:
                text = content
        
        # 移除SGML标签但保留文本
        text = re.sub(r'<[^>]+>', '', text)
        text = unescape(text)
        
        return text
    
    def parse_document(self, apf_path: Path) -> Optional[ACEDocument]:
        """
        解析单个ACE文档
        
        Args:
            apf_path: .apf.xml文件路径
            
        Returns:
            ACEDocument对象，如果解析失败返回None
        """
        try:
            # 解析APF XML
            tree = ET.parse(apf_path)
            root = tree.getroot()
            
            # 获取文档元信息
            source_file = root.get('URI', '')
            source = root.get('SOURCE', '')
            
            # 查找document元素
            doc_elem = root.find('document')
            if doc_elem is None:
                return None
            
            doc_id = doc_elem.get('DOCID', '')
            
            # 解析对应的SGM文件
            sgm_path = apf_path.with_suffix('').with_suffix('.sgm')
            raw_text = ''
            if sgm_path.exists():
                raw_text = self._parse_sgm(sgm_path)
            
            # 从路径提取语言
            language = "English"
            path_str = str(apf_path)
            if "/Chinese/" in path_str:
                language = "Chinese"
            elif "/Arabic/" in path_str:
                language = "Arabic"
            
            # 创建文档对象
            doc = ACEDocument(
                doc_id=doc_id,
                source=source,
                source_file=source_file,
                language=language,
                raw_text=raw_text
            )
            
            # 解析实体
            for entity_elem in doc_elem.findall('entity'):
                entity = self._parse_entity(entity_elem)
                doc.entities[entity.id] = entity
            
            # 解析时间表达式
            for timex_elem in doc_elem.findall('timex2'):
                timex = self._parse_timex(timex_elem)
                doc.timexes[timex.id] = timex
            
            # 解析值
            for value_elem in doc_elem.findall('value'):
                value = self._parse_value(value_elem)
                doc.values[value.id] = value
            
            # 解析关系
            for relation_elem in doc_elem.findall('relation'):
                relation = self._parse_relation(relation_elem)
                doc.relations[relation.id] = relation
            
            # 解析事件
            for event_elem in doc_elem.findall('event'):
                event = self._parse_event(event_elem)
                doc.events[event.id] = event
            
            return doc
            
        except ET.ParseError as e:
            print(f"XML解析错误 {apf_path}: {e}")
            return None
        except Exception as e:
            print(f"解析错误 {apf_path}: {e}")
            return None
    
    def find_all_documents(self, 
                          languages: List[str] = None,
                          sources: List[str] = None,
                          annotation_level: str = 'adj') -> List[Path]:
        """
        查找所有APF文件
        
        Args:
            languages: 语言列表 ['Arabic', 'Chinese', 'English']
            sources: 来源列表 ['bn', 'nw', 'wl', 'bc', 'cts', 'un']
            annotation_level: 标注级别 'adj', '1p', 'dual'
            
        Returns:
            APF文件路径列表
        """
        if languages is None:
            languages = ['Arabic', 'Chinese', 'English']
        if sources is None:
            sources = ['bn', 'nw', 'wl', 'bc', 'cts', 'un']
        
        apf_files = []
        
        for lang in languages:
            lang_dir = self.data_dir / lang
            if not lang_dir.exists():
                continue
            
            for source in sources:
                source_dir = lang_dir / source
                if not source_dir.exists():
                    continue
                
                # 查找指定标注级别的目录
                level_dir = source_dir / annotation_level
                if level_dir.exists():
                    apf_files.extend(level_dir.glob('*.apf.xml'))
                else:
                    # 如果没有子目录，直接在source目录下查找
                    apf_files.extend(source_dir.glob('*.apf.xml'))
        
        return sorted(apf_files)
    
    def parse_all(self, 
                 languages: List[str] = None,
                 sources: List[str] = None,
                 annotation_level: str = 'adj',
                 limit: int = None) -> List[ACEDocument]:
        """
        解析所有文档
        
        Args:
            languages: 语言列表
            sources: 来源列表
            annotation_level: 标注级别
            limit: 限制解析的文档数量
            
        Returns:
            ACEDocument列表
        """
        apf_files = self.find_all_documents(languages, sources, annotation_level)
        
        if limit:
            apf_files = apf_files[:limit]
        
        documents = []
        for apf_path in apf_files:
            doc = self.parse_document(apf_path)
            if doc:
                documents.append(doc)
        
        return documents


def get_statistics(documents: List[ACEDocument]) -> Dict:
    """
    获取文档集合的统计信息
    
    Args:
        documents: ACEDocument列表
        
    Returns:
        统计信息字典
    """
    stats = {
        'total_documents': len(documents),
        'total_entities': 0,
        'total_entity_mentions': 0,
        'total_timexes': 0,
        'total_values': 0,
        'total_relations': 0,
        'total_relation_mentions': 0,
        'total_events': 0,
        'total_event_mentions': 0,
        'entity_types': {},
        'relation_types': {},
        'event_types': {},
    }
    
    for doc in documents:
        stats['total_entities'] += len(doc.entities)
        stats['total_timexes'] += len(doc.timexes)
        stats['total_values'] += len(doc.values)
        stats['total_relations'] += len(doc.relations)
        stats['total_events'] += len(doc.events)
        
        for entity in doc.entities.values():
            stats['total_entity_mentions'] += len(entity.mentions)
            etype = entity.entity_type
            stats['entity_types'][etype] = stats['entity_types'].get(etype, 0) + 1
        
        for relation in doc.relations.values():
            stats['total_relation_mentions'] += len(relation.mentions)
            rtype = f"{relation.relation_type}:{relation.subtype}"
            stats['relation_types'][rtype] = stats['relation_types'].get(rtype, 0) + 1
        
        for event in doc.events.values():
            stats['total_event_mentions'] += len(event.mentions)
            etype = f"{event.event_type}:{event.subtype}"
            stats['event_types'][etype] = stats['event_types'].get(etype, 0) + 1
    
    return stats


if __name__ == '__main__':
    import sys
    
    # 测试解析器
    ace_root = sys.argv[1] if len(sys.argv) > 1 else '../test-data/ace_2005_td_v7'
    
    parser = ACEParser(ace_root)
    
    # 查找所有文档
    apf_files = parser.find_all_documents(annotation_level='adj')
    print(f"找到 {len(apf_files)} 个APF文件")
    
    # 解析前10个文档进行测试
    documents = parser.parse_all(annotation_level='adj', limit=10)
    print(f"成功解析 {len(documents)} 个文档")
    
    # 显示统计信息
    stats = get_statistics(documents)
    print("\n统计信息:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in sorted(value.items(), key=lambda x: -x[1])[:10]:
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
