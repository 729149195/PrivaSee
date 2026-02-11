#!/usr/bin/env python3
"""
Memory Stream Service - 主记忆流与关联回溯服务模块

功能概述：
1. 主记忆流 (MemoryStream): 跨会话的向量结构化信息元库，配合风险触发式可控检索
2. 关联回溯 (AssociationBacktracking): 基于主记忆流索引库的 Top-K 关联嵌入机制

核心特性：
- 以信息元为最小存储单元，每个信息元对应一个语义向量
- 采用 HNSW 算法实现毫秒级向量相似度检索
- 仅追加不更新策略，保留完整历史轨迹
- 风险触发式检索：准标识符组合检测、细化线索检测、敏感域命中
- 写入时同步计算 Top-K 关联绑定，无需后台进程
- 统一的跨模态证据标识格式
"""

import os
import json
import time
import hashlib
import logging
import threading
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from flask import Blueprint, request, jsonify

# 创建 Blueprint
memory_bp = Blueprint('memory', __name__, url_prefix='/api/memory')

logger = logging.getLogger(__name__)

# =============================================================================
# 配置
# =============================================================================

def get_config():
    """获取配置"""
    from config import MEMORY_STREAM_DATA_DIR, MEMORY_STREAM_EMBEDDING_MODEL
    return {
        'data_dir': str(MEMORY_STREAM_DATA_DIR),
        'embedding_model': MEMORY_STREAM_EMBEDDING_MODEL,
    }

# =============================================================================
# 常量定义
# =============================================================================

# HNSW 参数
HNSW_SPACE = 'cosine'
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16
HNSW_EF_SEARCH = 50

# 检索参数
DEFAULT_TOP_K = 5
MAX_RETRIEVAL_INFONS = 5
MAX_RETRIEVAL_TOKENS = 500
REFINEMENT_SIMILARITY_THRESHOLD = 0.85
ASSOCIATION_TOP_K = 3

# 准标识符类别关键词
QUASI_IDENTIFIER_CATEGORIES = {
    'geo_location': [
        '地址', '位置', '城市', '省份', '区', '街道', '路', '号', '小区', '社区',
        '地点', '坐标', '经纬度', '邮编',
        'address', 'location', 'city', 'province', 'street', 'district',
        'zip', 'postal', 'coordinates', 'latitude', 'longitude',
    ],
    'temporal': [
        '日期', '时间', '年份', '月份', '出生日期', '生日',
        'date', 'time', 'birthday', 'birth_date', 'year', 'month',
        'schedule', 'appointment',
    ],
    'org_role': [
        '公司', '单位', '组织', '学校', '大学', '部门', '职位', '职务', '工号',
        '学号', '员工', '同事', '上司', '下属',
        'company', 'organization', 'school', 'university', 'department',
        'position', 'title', 'employee', 'colleague', 'student_id',
    ],
    'rare_interest': [
        '病症', '过敏', '药物', '手术', '基因', '血型',
        '收藏', '嗜好', '癖好', '特殊习惯', '罕见',
        'rare', 'hobby', 'collection', 'allergy', 'medication',
        'surgery', 'genetic', 'blood_type',
    ],
    'biometric': [
        '指纹', '虹膜', '面部', '人脸', '声纹', '体重', '身高', '步态',
        '基因', 'DNA',
        'fingerprint', 'iris', 'face', 'facial', 'voiceprint',
        'weight', 'height', 'gait', 'dna', 'biometric',
    ],
}

# 敏感领域关键词
SENSITIVE_DOMAINS = {
    'health_medical': [
        '病', '症', '诊断', '治疗', '手术', '药物', '处方', '病历', '体检',
        '医院', '医生', '护士', '科室', '住院', '门诊', '化验', '检查',
        '癌', '肿瘤', '糖尿病', '高血压', '心脏', '精神', '抑郁', '焦虑',
        'disease', 'diagnosis', 'treatment', 'surgery', 'prescription',
        'hospital', 'doctor', 'medical', 'health', 'cancer', 'diabetes',
        'depression', 'anxiety', 'mental',
    ],
    'financial': [
        '银行', '账户', '账号', '卡号', '信用卡', '贷款', '存款', '工资',
        '收入', '税', '投资', '股票', '基金', '保险', '理财', '债务',
        'bank', 'account', 'credit_card', 'loan', 'salary', 'income',
        'tax', 'investment', 'stock', 'fund', 'insurance', 'debt',
    ],
    'legal_dispute': [
        '案件', '诉讼', '法院', '律师', '判决', '犯罪', '逮捕', '拘留',
        '罚款', '违法', '刑事', '民事', '仲裁',
        'case', 'lawsuit', 'court', 'lawyer', 'verdict', 'crime',
        'arrest', 'penalty', 'criminal', 'civil', 'arbitration',
    ],
    'intimate_relationship': [
        '恋人', '配偶', '伴侣', '婚姻', '离婚', '约会', '性', '怀孕',
        '情感', '亲密',
        'spouse', 'partner', 'marriage', 'divorce', 'dating',
        'pregnancy', 'intimate', 'relationship',
    ],
    'explicit_pii': [
        '身份证', '护照', '社保', '驾照', '学历证', '毕业证', '户口',
        '手机号', '电话', '邮箱', 'email', '微信', 'QQ',
        'ID_card', 'passport', 'social_security', 'driver_license',
        'phone', 'mobile', 'wechat',
    ],
    'document_image': [
        '证件', '证书', '合同', '协议', '发票', '收据', '成绩单', '简历',
        '委托书', '授权书',
        'certificate', 'contract', 'agreement', 'invoice', 'receipt',
        'transcript', 'resume', 'authorization',
    ],
}

# =============================================================================
# 全局变量
# =============================================================================

_embedding_model = None
_embedding_lock = threading.Lock()
_db_lock = threading.Lock()

# 用户隔离：每个用户独立的管理器实例
_managers: Dict[str, 'MemoryStreamManager'] = {}
_managers_lock = threading.Lock()


# =============================================================================
# Embedding 模型管理
# =============================================================================

def load_embedding_model():
    """加载 sentence-transformers 嵌入模型"""
    global _embedding_model

    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer
                config = get_config()
                model_name = config['embedding_model']
                logger.info(f"加载嵌入模型: {model_name}")
                _embedding_model = SentenceTransformer(model_name)
                logger.info(f"✓ 嵌入模型加载完成, 维度: {_embedding_model.get_sentence_embedding_dimension()}")

    return _embedding_model


def compute_embedding(text: str) -> np.ndarray:
    """计算文本的语义向量"""
    model = load_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.astype(np.float32)


def compute_embeddings_batch(texts: List[str]) -> np.ndarray:
    """批量计算语义向量"""
    model = load_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return embeddings.astype(np.float32)


def get_embedding_dim() -> int:
    """获取嵌入维度"""
    model = load_embedding_model()
    return model.get_sentence_embedding_dimension()


# =============================================================================
# SQLite 存储层
# =============================================================================

class MemoryStore:
    """信息元持久化存储 (SQLite)"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS infon_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iid TEXT NOT NULL UNIQUE,
                    infon_type TEXT NOT NULL,
                    modality TEXT NOT NULL DEFAULT 'text',
                    session_id TEXT NOT NULL,
                    round_num INTEGER NOT NULL DEFAULT 1,
                    entity TEXT DEFAULT '',
                    attribute TEXT DEFAULT '',
                    text_for_embedding TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    evidence_pointer TEXT DEFAULT '',
                    associations TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    extra_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_infon_iid ON infon_memory(iid)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_infon_session ON infon_memory(session_id)
            """)
            conn.commit()

    def insert_infon(self, record: Dict) -> bool:
        """插入一条信息元记录 (append-only)"""
        vector_blob = np.array(record['vector'], dtype=np.float32).tobytes()
        try:
            with _db_lock:
                with self._get_conn() as conn:
                    conn.execute("""
                        INSERT OR IGNORE INTO infon_memory
                        (iid, infon_type, modality, session_id, round_num,
                         entity, attribute, text_for_embedding, vector,
                         evidence_pointer, associations, created_at, extra_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record['iid'],
                        record.get('infon_type', 'DESC'),
                        record.get('modality', 'text'),
                        record.get('session_id', ''),
                        record.get('round_num', 1),
                        record.get('entity', ''),
                        record.get('attribute', ''),
                        record['text_for_embedding'],
                        vector_blob,
                        record.get('evidence_pointer', ''),
                        json.dumps(record.get('associations', []), ensure_ascii=False),
                        record.get('created_at', datetime.now().isoformat()),
                        json.dumps(record.get('extra', {}), ensure_ascii=False),
                    ))
                    conn.commit()
            return True
        except sqlite3.IntegrityError:
            logger.debug(f"信息元已存在 (append-only skip): {record['iid']}")
            return False
        except Exception as e:
            logger.error(f"插入信息元失败: {e}")
            return False

    def exists_iid(self, iid: str) -> bool:
        """检查 iid 是否已存在"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM infon_memory WHERE iid = ? LIMIT 1", (iid,)
            ).fetchone()
        return row is not None

    def get_all_vectors(self) -> Tuple[List[str], np.ndarray]:
        """获取所有信息元的 iid 和向量"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT iid, vector FROM infon_memory ORDER BY id"
            ).fetchall()

        if not rows:
            return [], np.array([], dtype=np.float32)

        iids = [r['iid'] for r in rows]
        dim = len(np.frombuffer(rows[0]['vector'], dtype=np.float32))
        vectors = np.zeros((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            vectors[i] = np.frombuffer(r['vector'], dtype=np.float32)

        return iids, vectors

    def get_infon_by_iid(self, iid: str) -> Optional[Dict]:
        """根据 iid 查询信息元"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM infon_memory WHERE iid = ?", (iid,)
            ).fetchone()

        if not row:
            return None
        return self._row_to_dict(row)

    def get_infons_by_iids(self, iids: List[str]) -> List[Dict]:
        """批量根据 iid 查询信息元"""
        if not iids:
            return []
        placeholders = ','.join('?' * len(iids))
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM infon_memory WHERE iid IN ({placeholders})", iids
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_all_infons_for_viz(self) -> List[Dict]:
        """获取所有信息元的元数据 + 向量（用于可视化降维）"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT iid, infon_type, modality, session_id, round_num, "
                "entity, attribute, text_for_embedding, vector, "
                "evidence_pointer, associations, created_at "
                "FROM infon_memory ORDER BY id"
            ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            # 解析 vector
            if isinstance(d['vector'], bytes):
                d['vector'] = np.frombuffer(d['vector'], dtype=np.float32).tolist()
            # 解析 associations
            if isinstance(d.get('associations'), str):
                try:
                    d['associations'] = json.loads(d['associations'])
                except json.JSONDecodeError:
                    d['associations'] = []
            results.append(d)
        return results

    def count(self) -> int:
        """获取信息元总数"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM infon_memory").fetchone()
        return row['cnt'] if row else 0

    def clear_all(self):
        """清空所有数据"""
        with _db_lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM infon_memory")
                conn.commit()
        logger.info("✓ 所有信息元记录已清空")

    def get_meta_by_session(self, session_id: str) -> List[Dict]:
        """获取指定会话下的信息元元数据（用于按生命周期管理）"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT iid, extra_json FROM infon_memory WHERE session_id = ?",
                (session_id,)
            ).fetchall()
        result = []
        for r in rows:
            extra = {}
            if isinstance(r['extra_json'], str):
                try:
                    extra = json.loads(r['extra_json'])
                except json.JSONDecodeError:
                    extra = {}
            result.append({'iid': r['iid'], 'extra_json': extra})
        return result

    def delete_infons_by_iids(self, iids: List[str]) -> int:
        """按 iid 批量删除信息元"""
        if not iids:
            return 0
        placeholders = ','.join('?' * len(iids))
        with _db_lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    f"DELETE FROM infon_memory WHERE iid IN ({placeholders})", iids
                )
                conn.commit()
                return int(cur.rowcount or 0)

    def patch_extra_by_iids(self, iids: List[str], patch: Dict[str, Any]) -> int:
        """按 iid 批量更新 extra_json 字段"""
        if not iids or not patch:
            return 0
        updated = 0
        with _db_lock:
            with self._get_conn() as conn:
                placeholders = ','.join('?' * len(iids))
                rows = conn.execute(
                    f"SELECT iid, extra_json FROM infon_memory WHERE iid IN ({placeholders})",
                    iids
                ).fetchall()
                for row in rows:
                    extra = {}
                    if isinstance(row['extra_json'], str):
                        try:
                            extra = json.loads(row['extra_json'])
                        except json.JSONDecodeError:
                            extra = {}
                    extra.update(patch)
                    conn.execute(
                        "UPDATE infon_memory SET extra_json = ? WHERE iid = ?",
                        (json.dumps(extra, ensure_ascii=False), row['iid'])
                    )
                    updated += 1
                conn.commit()
        return updated

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """将数据库行转为字典"""
        d = dict(row)
        # 解析 vector
        if 'vector' in d and isinstance(d['vector'], bytes):
            d['vector'] = np.frombuffer(d['vector'], dtype=np.float32).tolist()
        # 解析 associations
        if 'associations' in d and isinstance(d['associations'], str):
            try:
                d['associations'] = json.loads(d['associations'])
            except json.JSONDecodeError:
                d['associations'] = []
        # 解析 extra_json
        if 'extra_json' in d and isinstance(d['extra_json'], str):
            try:
                d['extra_json'] = json.loads(d['extra_json'])
            except json.JSONDecodeError:
                d['extra_json'] = {}
        return d


# =============================================================================
# HNSW 向量索引
# =============================================================================

class HNSWIndex:
    """基于 hnswlib 的 HNSW 向量索引"""

    def __init__(self, dim: int, max_elements: int = 100000):
        self.dim = dim
        self.max_elements = max_elements
        self.index = None
        self.iid_to_label = {}   # iid -> internal label
        self.label_to_iid = {}   # internal label -> iid
        self.current_count = 0
        self._lock = threading.Lock()
        self._init_index()

    def _init_index(self):
        """初始化 HNSW 索引"""
        try:
            import hnswlib
            self.index = hnswlib.Index(space=HNSW_SPACE, dim=self.dim)
            self.index.init_index(
                max_elements=self.max_elements,
                ef_construction=HNSW_EF_CONSTRUCTION,
                M=HNSW_M
            )
            self.index.set_ef(HNSW_EF_SEARCH)
            logger.info(f"✓ HNSW 索引初始化完成 (dim={self.dim})")
        except ImportError:
            logger.warning("hnswlib 未安装，回退到 FAISS")
            self._init_faiss_index()

    def _init_faiss_index(self):
        """使用 FAISS 作为后备索引"""
        import faiss
        self.index = faiss.IndexFlatIP(self.dim)  # 内积 (已归一化 = 余弦相似度)
        self._is_faiss = True
        logger.info(f"✓ FAISS 索引初始化完成 (dim={self.dim})")

    @property
    def _is_hnswlib(self):
        return not getattr(self, '_is_faiss', False)

    def add(self, iid: str, vector: np.ndarray):
        """添加向量到索引"""
        with self._lock:
            if iid in self.iid_to_label:
                return  # 已存在，跳过

            vec = vector.reshape(1, -1).astype(np.float32)
            label = self.current_count

            if self._is_hnswlib:
                # 检查是否需要扩容
                if self.current_count >= self.max_elements:
                    new_max = self.max_elements * 2
                    self.index.resize_index(new_max)
                    self.max_elements = new_max
                self.index.add_items(vec, np.array([label]))
            else:
                self.index.add(vec)

            self.iid_to_label[iid] = label
            self.label_to_iid[label] = iid
            self.current_count += 1

    def search(self, query_vector: np.ndarray, k: int = DEFAULT_TOP_K,
               exclude_iids: Optional[set] = None) -> List[Tuple[str, float]]:
        """检索最相似的 k 个向量，返回 [(iid, similarity), ...]"""
        if self.current_count == 0:
            return []

        with self._lock:
            vec = query_vector.reshape(1, -1).astype(np.float32)
            # 多取一些以便排除
            actual_k = min(k + len(exclude_iids or set()) + 1, self.current_count)

            if self._is_hnswlib:
                labels, distances = self.index.knn_query(vec, k=actual_k)
                # hnswlib cosine space 返回的距离 = 1 - cos_sim
                results = []
                for lbl, dist in zip(labels[0], distances[0]):
                    iid = self.label_to_iid.get(int(lbl))
                    if iid and (not exclude_iids or iid not in exclude_iids):
                        similarity = 1.0 - float(dist)
                        results.append((iid, similarity))
                    if len(results) >= k:
                        break
            else:
                scores, indices = self.index.search(vec, actual_k)
                results = []
                for idx, score in zip(indices[0], scores[0]):
                    if idx < 0:
                        continue
                    iid = self.label_to_iid.get(int(idx))
                    if iid and (not exclude_iids or iid not in exclude_iids):
                        results.append((iid, float(score)))
                    if len(results) >= k:
                        break

        return results

    def clear(self):
        """清空索引"""
        with self._lock:
            self.iid_to_label.clear()
            self.label_to_iid.clear()
            self.current_count = 0
            self._init_index()
        logger.info("✓ HNSW 索引已清空")

    def rebuild_from_store(self, store: MemoryStore):
        """从存储中重建索引"""
        iids, vectors = store.get_all_vectors()
        if len(iids) == 0:
            logger.info("存储为空，无需重建索引")
            return

        with self._lock:
            self.iid_to_label.clear()
            self.label_to_iid.clear()
            self.current_count = 0

            # 重新初始化
            new_max = max(self.max_elements, len(iids) * 2)
            if self._is_hnswlib:
                import hnswlib
                self.index = hnswlib.Index(space=HNSW_SPACE, dim=self.dim)
                self.index.init_index(
                    max_elements=new_max,
                    ef_construction=HNSW_EF_CONSTRUCTION,
                    M=HNSW_M
                )
                self.index.set_ef(HNSW_EF_SEARCH)
                self.max_elements = new_max
            else:
                import faiss
                self.index = faiss.IndexFlatIP(self.dim)

        # 逐条添加
        for iid, vec in zip(iids, vectors):
            self.add(iid, vec)

        logger.info(f"✓ 索引重建完成，共 {len(iids)} 条记录")


# =============================================================================
# 风险触发检测器
# =============================================================================

class RiskTriggerDetector:
    """风险触发条件检测"""

    @staticmethod
    def classify_quasi_identifiers(infons: List[Dict]) -> Dict[str, List[Dict]]:
        """将信息元按准标识符类别分类"""
        categories_found = {}

        for infon in infons:
            entity = str(infon.get('entity', '')).lower()
            attribute = str(infon.get('attribute', '')).lower()
            combined = f"{entity} {attribute}"

            for category, keywords in QUASI_IDENTIFIER_CATEGORIES.items():
                for kw in keywords:
                    if kw.lower() in combined:
                        if category not in categories_found:
                            categories_found[category] = []
                        categories_found[category].append(infon)
                        break

        return categories_found

    @staticmethod
    def check_quasi_identifier_combination(infons: List[Dict]) -> Tuple[bool, Dict]:
        """检测准标识符组合 (>= 2 类则触发)"""
        categories = RiskTriggerDetector.classify_quasi_identifiers(infons)
        triggered = len(categories) >= 2
        return triggered, {
            'trigger_type': 'quasi_identifier_combination',
            'categories_count': len(categories),
            'categories': list(categories.keys()),
        }

    @staticmethod
    def check_refinement(infons: List[Dict], hnsw_index: 'HNSWIndex',
                         threshold: float = REFINEMENT_SIMILARITY_THRESHOLD) -> Tuple[bool, Dict]:
        """检测细化线索 (与历史信息元的语义相似度超过阈值)"""
        if hnsw_index.current_count == 0:
            return False, {'trigger_type': 'refinement_detection', 'max_similarity': 0.0}

        max_sim = 0.0
        triggered_infon = None
        current_iids = {inf.get('iid', '') for inf in infons}

        for infon in infons:
            text = _build_embedding_text(infon)
            if not text.strip():
                continue

            vec = compute_embedding(text)
            results = hnsw_index.search(vec, k=1, exclude_iids=current_iids)

            if results:
                _, sim = results[0]
                if sim > max_sim:
                    max_sim = sim
                    triggered_infon = infon

        triggered = max_sim >= threshold
        return triggered, {
            'trigger_type': 'refinement_detection',
            'max_similarity': round(max_sim, 4),
            'threshold': threshold,
            'triggered_infon_iid': triggered_infon.get('iid') if triggered_infon else None,
        }

    @staticmethod
    def check_sensitive_domain(infons: List[Dict]) -> Tuple[bool, Dict]:
        """检测敏感领域命中"""
        domains_hit = set()

        for infon in infons:
            entity = str(infon.get('entity', '')).lower()
            attribute = str(infon.get('attribute', '')).lower()
            combined = f"{entity} {attribute}"

            for domain, keywords in SENSITIVE_DOMAINS.items():
                for kw in keywords:
                    if kw.lower() in combined:
                        domains_hit.add(domain)
                        break

        triggered = len(domains_hit) > 0
        return triggered, {
            'trigger_type': 'sensitive_domain_hit',
            'domains_hit': list(domains_hit),
        }


# =============================================================================
# 辅助函数
# =============================================================================

def _build_embedding_text(infon: Dict) -> str:
    """构建用于嵌入的文本 (entity + attribute 拼接)"""
    infon_type = str(infon.get('infon_type', '')).upper()

    if infon_type == 'DESC':
        entity = infon.get('entity', '')
        attribute = infon.get('attribute', '')
        return f"{entity} {attribute}".strip()
    elif infon_type == 'SCEN':
        temporal = infon.get('temporal', '')
        spatial = infon.get('spatial', '')
        return f"{temporal} {spatial}".strip()
    elif infon_type == 'REL':
        rel_name = infon.get('relation_name', '')
        arg_refs = ' '.join(infon.get('arg_refs', []))
        return f"{rel_name} {arg_refs}".strip()
    else:
        # 兜底：尝试拼接所有文本字段
        parts = []
        for key in ['entity', 'attribute', 'temporal', 'spatial', 'relation_name', 'description']:
            val = infon.get(key, '')
            if val:
                parts.append(str(val))
        return ' '.join(parts).strip()


def _build_evidence_pointer(infon: Dict) -> str:
    """构建证据唯一标识"""
    modality = infon.get('modality', infon.get('run_metadata', {}).get('modality', 'text'))
    session_id = infon.get('session_id', '')
    round_num = infon.get('round_num', 1)

    # 根据模态构建 span_locator
    span = infon.get('span', None)
    if modality == 'image':
        ocr_box_id = infon.get('ocr_box_id', 0)
        span_locator = f"ocr_box_{ocr_box_id}"
    elif modality == 'audio':
        seg_id = infon.get('segment_id', 0)
        span_locator = f"seg_{seg_id}"
    else:
        # text
        if span and isinstance(span, (list, tuple)) and len(span) == 2:
            span_locator = f"{span[0]}-{span[1]}"
        else:
            span_locator = "0-0"

    return f"{modality}:{session_id}:{round_num}:{span_locator}"


def _estimate_token_count(text: str) -> int:
    """粗略估算 token 数量"""
    # 中文约1字=1.5token，英文约1词=1.3token
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.3)


def _sanitize_iid_component(value: str, fallback: str = 'x') -> str:
    """将任意字符串规范化为可拼接到 iid 的安全片段"""
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    out = []
    for ch in text:
        if ch.isalnum() or ch in ['_', '-']:
            out.append(ch)
        else:
            out.append('_')
    normalized = ''.join(out).strip('_')
    return normalized or fallback


# =============================================================================
# 主记忆流管理器
# =============================================================================

class MemoryStreamManager:
    """主记忆流管理器 - 整合存储、索引和检索 (按用户隔离)"""

    def __init__(self, user_id: str = 'default'):
        self.user_id = user_id
        self.store: Optional[MemoryStore] = None
        self.index: Optional[HNSWIndex] = None
        self.detector = RiskTriggerDetector()
        self._initialized = False
        # 可视化结果缓存 (实例级，避免不同用户/管理器串缓存)
        self._viz_cache_hash: Optional[str] = None
        self._viz_cache_result: Optional[Dict] = None

    def initialize(self):
        """初始化存储和索引 (每个用户独立的数据库和索引)"""
        if self._initialized:
            return

        config = get_config()
        data_dir = config['data_dir']
        os.makedirs(data_dir, exist_ok=True)

        # 按用户隔离：每个用户一个独立的 SQLite 数据库
        safe_uid = self.user_id.replace('/', '_').replace('\\', '_').replace('..', '_')
        db_filename = f'memory_stream_{safe_uid}.db' if safe_uid != 'default' else 'memory_stream.db'
        db_path = os.path.join(data_dir, db_filename)
        self.store = MemoryStore(db_path)

        # 加载嵌入模型获取维度
        dim = get_embedding_dim()
        self.index = HNSWIndex(dim=dim)

        # 从存储中重建索引
        self.index.rebuild_from_store(self.store)

        self._initialized = True
        logger.info(f"✓ 主记忆流初始化完成 (用户: {self.user_id}, 存储: {self.store.count()} 条)")

    def _ensure_initialized(self):
        if not self._initialized:
            self.initialize()

    def _resolve_unique_iid(self, source_iid: str, modality: str,
                            session_id: str, round_num: int,
                            reserved_iids: set) -> str:
        """
        为写入库生成唯一 iid。
        强制带作用域后缀：user/session/round，避免跨窗口与跨轮次混淆。
        """
        base = str(source_iid or '').strip() or f"desc:r{round_num}_auto"
        safe_user = _sanitize_iid_component(self.user_id, 'user')
        safe_modality = _sanitize_iid_component(modality or 'text', 'text')
        safe_session = _sanitize_iid_component(session_id, 'session')
        suffix_seed = f"u{safe_user}_s{safe_session}_r{int(round_num)}_{safe_modality}"
        scoped_base = f"{base}__{suffix_seed}"

        # 优先使用稳定作用域 IID，便于排查与回溯
        if scoped_base not in reserved_iids and not self.store.exists_iid(scoped_base):
            reserved_iids.add(scoped_base)
            return scoped_base

        # 同一作用域下再次冲突时再追加序号
        idx = 1
        while True:
            candidate = f"{scoped_base}_{idx}"
            if candidate not in reserved_iids and not self.store.exists_iid(candidate):
                reserved_iids.add(candidate)
                return candidate
            idx += 1

    def ingest(self, infons: List[Dict], session_id: str, round_num: int) -> Dict:
        """
        批量写入信息元到记忆流
        同时完成关联绑定 (Module 2: Association Backtracking)

        流程：
        1. 对每个信息元计算语义向量
        2. 在存入前先检索 Top-K 关联 (关联回溯)
        3. 记录关联关系
        4. 存入数据库并更新索引
        """
        self._ensure_initialized()

        ingested = []
        skipped = []

        reserved_iids = set()

        for infon in infons:
            source_iid = infon.get('_source_iid', infon.get('iid', ''))
            if not source_iid:
                skipped.append({'reason': 'missing_iid'})
                continue

            # 构建嵌入文本
            embed_text = _build_embedding_text(infon)
            if not embed_text.strip():
                skipped.append({'iid': source_iid, 'reason': 'empty_embedding_text'})
                continue

            # 计算语义向量
            vector = compute_embedding(embed_text)

            # === 关联回溯：写入前检索 Top-K 关联 ===
            associations = []
            if self.index.current_count > 0:
                search_results = self.index.search(
                    vector, k=ASSOCIATION_TOP_K,
                    exclude_iids={source_iid}
                )
                for assoc_iid, similarity in search_results:
                    associations.append({
                        'iid': assoc_iid,
                        'similarity': round(float(similarity), 4)
                    })

            # 构建证据指针
            evidence_pointer = _build_evidence_pointer({
                **infon,
                'session_id': session_id,
                'round_num': round_num,
            })

            # 获取模态标签
            modality = infon.get('modality',
                       infon.get('run_metadata', {}).get('modality', 'text'))
            resolved_iid = self._resolve_unique_iid(
                source_iid=source_iid,
                modality=modality,
                session_id=session_id,
                round_num=round_num,
                reserved_iids=reserved_iids
            )

            # 构建存储记录
            record = {
                'iid': resolved_iid,
                'infon_type': infon.get('infon_type', 'DESC'),
                'modality': modality,
                'session_id': session_id,
                'round_num': round_num,
                'entity': infon.get('entity', ''),
                'attribute': infon.get('attribute', ''),
                'text_for_embedding': embed_text,
                'vector': vector.tolist(),
                'evidence_pointer': evidence_pointer,
                'associations': associations,
                'created_at': datetime.now().isoformat(),
                'extra': {
                    'confidence': infon.get('confidence'),
                    'temporal': infon.get('temporal'),
                    'spatial': infon.get('spatial'),
                    'relation_name': infon.get('relation_name'),
                    'arg_refs': infon.get('arg_refs'),
                    # 生命周期元数据：支持 pending 先入库、未发送撤销、发送后转正式记录
                    'memory_target_type': infon.get('_memory_target_type', 'message'),
                    'memory_target_key': infon.get('_memory_target_key', ''),
                    'memory_run_id': infon.get('_memory_run_id', ''),
                    'memory_modality': infon.get('_memory_modality', modality),
                },
            }

            # 存入数据库
            inserted = self.store.insert_infon(record)

            if inserted:
                # 更新 HNSW 索引 (在数据库写入之后)
                self.index.add(resolved_iid, vector)
                ingested.append({
                    'source_iid': source_iid,
                    'iid': resolved_iid,
                    'evidence_pointer': evidence_pointer,
                    'associations': associations,
                })
            else:
                skipped.append({'iid': source_iid, 'reason': 'duplicate'})

        return {
            'ingested_count': len(ingested),
            'skipped_count': len(skipped),
            'ingested': ingested,
            'skipped': skipped,
            'total_in_store': self.store.count(),
        }

    def remove_infons_by_lifecycle(self, session_id: str,
                                   target_type: Optional[str] = None,
                                   run_ids: Optional[List[str]] = None) -> Dict:
        """按生命周期元数据移除信息元（用于 pending 更新/撤销）"""
        self._ensure_initialized()
        run_id_set = set(run_ids or [])

        candidates = self.store.get_meta_by_session(session_id)
        remove_iids = []
        for row in candidates:
            extra = row.get('extra_json') or {}
            if target_type and extra.get('memory_target_type') != target_type:
                continue
            if run_id_set and extra.get('memory_run_id') not in run_id_set:
                continue
            remove_iids.append(row['iid'])

        removed = self.store.delete_infons_by_iids(remove_iids)
        if removed > 0:
            self.index.rebuild_from_store(self.store)

        return {
            'removed_count': removed,
            'requested_count': len(remove_iids),
            'total_in_store': self.store.count(),
        }

    def promote_pending_infons(self, session_id: str, run_ids: List[str], message_id: str) -> Dict:
        """将 pending 信息元标记为 message（发送成功后调用）"""
        self._ensure_initialized()
        run_id_set = set(run_ids or [])
        if not run_id_set:
            return {'updated_count': 0, 'total_in_store': self.store.count()}

        candidates = self.store.get_meta_by_session(session_id)
        target_iids = []
        for row in candidates:
            extra = row.get('extra_json') or {}
            if extra.get('memory_target_type') != 'pending':
                continue
            if extra.get('memory_run_id') not in run_id_set:
                continue
            target_iids.append(row['iid'])

        updated = self.store.patch_extra_by_iids(target_iids, {
            'memory_target_type': 'message',
            'memory_target_key': message_id or '',
        })
        return {
            'updated_count': updated,
            'requested_count': len(target_iids),
            'total_in_store': self.store.count(),
        }

    def trigger_check_and_retrieve(self, infons: List[Dict]) -> Dict:
        """
        风险触发检测 + 可控检索

        三种触发条件：
        1. 准标识符组合检测
        2. 细化线索检测
        3. 敏感域命中

        默认不检索，仅在触发条件满足时执行
        """
        self._ensure_initialized()

        # 逐项检查触发条件
        triggers = []

        # 1. 准标识符组合检测
        qi_triggered, qi_info = self.detector.check_quasi_identifier_combination(infons)
        if qi_triggered:
            triggers.append(qi_info)

        # 2. 细化线索检测
        ref_triggered, ref_info = self.detector.check_refinement(infons, self.index)
        if ref_triggered:
            triggers.append(ref_info)

        # 3. 敏感域命中
        sd_triggered, sd_info = self.detector.check_sensitive_domain(infons)
        if sd_triggered:
            triggers.append(sd_info)

        # 如果没有触发，不执行检索
        if not triggers:
            return {
                'triggered': False,
                'triggers': [],
                'retrieved_infons': [],
            }

        # === 执行检索 ===
        retrieved = self._execute_retrieval(infons)

        return {
            'triggered': True,
            'triggers': triggers,
            'retrieved_infons': retrieved,
        }

    def _execute_retrieval(self, query_infons: List[Dict]) -> List[Dict]:
        """执行向量检索，返回最相关的历史信息元"""
        if self.index.current_count == 0:
            return []

        # 收集所有当前信息元的 iid
        current_iids = {inf.get('iid', '') for inf in query_infons}

        # 对高危信息元计算向量进行检索
        all_results = {}  # iid -> (infon_dict, max_similarity)

        for infon in query_infons:
            text = _build_embedding_text(infon)
            if not text.strip():
                continue

            vec = compute_embedding(text)
            results = self.index.search(vec, k=DEFAULT_TOP_K, exclude_iids=current_iids)

            for result_iid, similarity in results:
                if result_iid not in all_results or similarity > all_results[result_iid][1]:
                    all_results[result_iid] = (result_iid, similarity)

        if not all_results:
            return []

        # 按相似度排序
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)

        # 硬性截断：最多 MAX_RETRIEVAL_INFONS 条
        top_results = sorted_results[:MAX_RETRIEVAL_INFONS]

        # 从存储中获取完整信息元
        result_iids = [r[0] for r in top_results]
        stored_infons = self.store.get_infons_by_iids(result_iids)

        # 构建返回结果
        retrieved = []
        total_tokens = 0
        iid_to_sim = {r[0]: r[1] for r in top_results}

        for infon in stored_infons:
            iid = infon['iid']
            # token 截断
            text = infon.get('text_for_embedding', '')
            token_count = _estimate_token_count(text)
            if total_tokens + token_count > MAX_RETRIEVAL_TOKENS:
                break
            total_tokens += token_count

            # 移除 vector 字段减少传输量
            result = {k: v for k, v in infon.items() if k != 'vector'}
            result['retrieval_similarity'] = round(iid_to_sim.get(iid, 0.0), 4)
            retrieved.append(result)

        return retrieved

    def search(self, query_text: str, k: int = DEFAULT_TOP_K) -> List[Dict]:
        """直接向量搜索"""
        self._ensure_initialized()

        if self.index.current_count == 0:
            return []

        vec = compute_embedding(query_text)
        results = self.index.search(vec, k=k)

        result_iids = [r[0] for r in results]
        stored_infons = self.store.get_infons_by_iids(result_iids)

        iid_to_sim = {r[0]: r[1] for r in results}
        output = []
        for infon in stored_infons:
            result = {k: v for k, v in infon.items() if k != 'vector'}
            result['similarity'] = round(iid_to_sim.get(infon['iid'], 0.0), 4)
            output.append(result)

        output.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return output

    def backtrace(self, iid: str) -> Optional[Dict]:
        """
        关联回溯查询
        给定 iid，返回证据指针 + 关联信息元列表
        """
        self._ensure_initialized()

        infon = self.store.get_infon_by_iid(iid)
        if not infon:
            return None

        # 解析证据指针
        evidence_pointer = infon.get('evidence_pointer', '')
        parsed_pointer = self._parse_evidence_pointer(evidence_pointer)

        # 获取关联信息元的详细信息
        associations = infon.get('associations', [])
        assoc_iids = [a['iid'] for a in associations if 'iid' in a]
        assoc_infons = self.store.get_infons_by_iids(assoc_iids)
        assoc_map = {inf['iid']: inf for inf in assoc_infons}

        enriched_associations = []
        for assoc in associations:
            assoc_iid = assoc.get('iid', '')
            full_infon = assoc_map.get(assoc_iid, {})
            enriched_associations.append({
                'iid': assoc_iid,
                'similarity': assoc.get('similarity', 0.0),
                'infon_type': full_infon.get('infon_type', ''),
                'entity': full_infon.get('entity', ''),
                'attribute': full_infon.get('attribute', ''),
                'modality': full_infon.get('modality', ''),
                'session_id': full_infon.get('session_id', ''),
                'round_num': full_infon.get('round_num', 0),
                'evidence_pointer': full_infon.get('evidence_pointer', ''),
            })

        return {
            'iid': iid,
            'infon_type': infon.get('infon_type', ''),
            'entity': infon.get('entity', ''),
            'attribute': infon.get('attribute', ''),
            'modality': infon.get('modality', ''),
            'evidence_pointer': evidence_pointer,
            'parsed_pointer': parsed_pointer,
            'associations': enriched_associations,
            'created_at': infon.get('created_at', ''),
        }

    def _parse_evidence_pointer(self, pointer: str) -> Dict:
        """解析证据指针字符串"""
        if not pointer:
            return {}

        parts = pointer.split(':')
        if len(parts) < 4:
            return {'raw': pointer}

        result = {
            'modality': parts[0],
            'session_id': parts[1],
            'round_num': int(parts[2]) if parts[2].isdigit() else 0,
            'span_locator': parts[3],
        }

        # 进一步解析 span_locator
        span = parts[3]
        if span.startswith('ocr_box_'):
            result['locator_type'] = 'ocr_box'
            result['box_index'] = int(span.replace('ocr_box_', ''))
        elif span.startswith('seg_'):
            result['locator_type'] = 'segment'
            result['segment_index'] = int(span.replace('seg_', ''))
        elif '-' in span:
            result['locator_type'] = 'char_range'
            range_parts = span.split('-')
            if len(range_parts) == 2:
                result['char_start'] = int(range_parts[0])
                result['char_end'] = int(range_parts[1])

        return result

    @staticmethod
    def _auto_perplexity(n: int) -> int:
        """
        根据数据规模自动计算最优 perplexity

        经验公式：perplexity ≈ min(30, sqrt(n))
        - n < 10   → 使用 PCA (t-SNE 无意义)
        - n 10-100 → perplexity = max(5, n//3)
        - n 100-1000 → perplexity = 30
        - n > 1000 → perplexity = 50
        """
        if n <= 10:
            return max(2, n - 1)
        elif n <= 100:
            return max(5, min(n // 3, 30))
        elif n <= 1000:
            return 30
        else:
            return 50

    # 大数据采样阈值
    VIZ_SAMPLE_THRESHOLD = 2000   # 超过此数量进行采样
    VIZ_SAMPLE_SIZE = 1500        # 采样数量
    VIZ_TSNE_LIMIT = 5000         # 超过此数量强制 PCA (t-SNE 太慢)

    def get_visualization_data(self, method: str = 'auto') -> Dict:
        """
        获取可视化数据：对所有信息元向量进行降维，返回 2D 坐标 + 元数据 + 关联边

        策略：
        - method='auto' (默认): 根据数据量自动选择最优降维方法和参数
          · n ≤ 3       → PCA
          · 4 ≤ n ≤ 5000 → t-SNE (perplexity 全自动)
          · n > 5000    → PCA (t-SNE O(n²) 太慢)
        - method='tsne' : 强制 t-SNE
        - method='pca'  : 强制 PCA

        大数据优化：
        - n > 2000 时进行随机采样，保证交互响应速度
        - 缓存降维结果，数据不变时直接返回

        Args:
            method: 'auto' (默认), 'tsne' 或 'pca'

        Returns:
            {
                points: [{iid, x, y, infon_type, entity, attribute, ...}],
                edges: [{source, target, similarity}],
                total: int,            # 总信息元数
                displayed: int,        # 实际显示的点数 (采样后)
                sampled: bool,         # 是否进行了采样
                method: str,           # 实际使用的降维方法
                perplexity: int|null,  # 实际 perplexity (仅 t-SNE)
                stats: {...},          # 统计摘要
            }
        """
        self._ensure_initialized()

        all_infons = self.store.get_all_infons_for_viz()
        if not all_infons:
            return {
                'points': [], 'edges': [], 'total': 0, 'displayed': 0,
                'sampled': False, 'method': method, 'perplexity': None,
                'stats': {},
            }

        n_total = len(all_infons)

        # ---- 缓存检查 ----
        # 基于完整数据指纹缓存，避免“数量相同但内容不同”时误命中
        fp_src = '|'.join(f"{x.get('iid', '')}@{x.get('created_at', '')}" for x in all_infons)
        fp_hash = hashlib.sha1(fp_src.encode('utf-8')).hexdigest()
        cache_key = f"{method}_{fp_hash}"
        if self._viz_cache_hash == cache_key and self._viz_cache_result:
            logger.info(f"可视化缓存命中 (n={n_total}, method={method})")
            return self._viz_cache_result

        import time as _time
        t0 = _time.time()

        # ---- 大数据采样 ----
        sampled = False
        if n_total > self.VIZ_SAMPLE_THRESHOLD:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n_total, size=min(self.VIZ_SAMPLE_SIZE, n_total), replace=False)
            sample_idx.sort()
            all_infons = [all_infons[i] for i in sample_idx]
            sampled = True
            logger.info(f"可视化采样: {n_total} → {len(all_infons)}")

        n = len(all_infons)
        vectors = np.array([inf['vector'] for inf in all_infons], dtype=np.float32)

        # ---- 自动选择降维方法 ----
        actual_method = method
        actual_perplexity = None

        if method == 'auto':
            if n <= 3:
                actual_method = 'pca'
            elif n > self.VIZ_TSNE_LIMIT:
                actual_method = 'pca'
                logger.info(f"数据量过大 (n={n}), 自动切换到 PCA")
            else:
                actual_method = 'tsne'

        # ---- 降维 ----
        if n == 1:
            coords_2d = np.array([[0.5, 0.5]])
        elif actual_method == 'pca' or n <= 3:
            actual_method = 'pca'
            from sklearn.decomposition import PCA
            n_comp = min(2, n, vectors.shape[1])
            pca = PCA(n_components=n_comp)
            coords_2d = pca.fit_transform(vectors)
            if n_comp == 1:
                coords_2d = np.column_stack([coords_2d, np.zeros(n)])
        else:
            actual_method = 'tsne'
            from sklearn.manifold import TSNE
            actual_perplexity = self._auto_perplexity(n)
            # 确保 perplexity < n
            actual_perplexity = min(actual_perplexity, max(2, n - 1))
            tsne = TSNE(
                n_components=2,
                perplexity=actual_perplexity,
                random_state=42,
                init='pca',
                learning_rate='auto',
                max_iter=1000,
            )
            coords_2d = tsne.fit_transform(vectors)

        # ---- 归一化到 [0, 1] (加 5% padding) ----
        for dim in range(2):
            col = coords_2d[:, dim]
            vmin, vmax = col.min(), col.max()
            spread = vmax - vmin if vmax > vmin else 1.0
            coords_2d[:, dim] = (col - vmin) / spread

        # ---- 构建点数据 + 统计 ----
        points = []
        edges = []
        seen_edges = set()
        type_counts = {}
        modality_counts = {}
        session_set = set()

        iid_set_in_points = set()
        for inf in all_infons:
            iid_set_in_points.add(inf['iid'])

        for i, inf in enumerate(all_infons):
            associations = inf.get('associations', [])
            infon_type = inf.get('infon_type', '')
            modality = inf.get('modality', 'text')
            session_id = inf.get('session_id', '')

            type_counts[infon_type] = type_counts.get(infon_type, 0) + 1
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            session_set.add(session_id)

            points.append({
                'iid': inf['iid'],
                'x': round(float(coords_2d[i, 0]), 4),
                'y': round(float(coords_2d[i, 1]), 4),
                'infon_type': infon_type,
                'entity': inf.get('entity', ''),
                'attribute': inf.get('attribute', ''),
                'modality': modality,
                'session_id': session_id,
                'round_num': inf.get('round_num', 1),
                'created_at': inf.get('created_at', ''),
                'text_for_embedding': inf.get('text_for_embedding', ''),
                'associations': associations,
            })

            # 关联边 (只保留两端都在当前点集中的边)
            for assoc in associations:
                assoc_iid = assoc.get('iid', '')
                if assoc_iid and assoc_iid in iid_set_in_points:
                    edge_key = tuple(sorted([inf['iid'], assoc_iid]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({
                            'source': inf['iid'],
                            'target': assoc_iid,
                            'similarity': assoc.get('similarity', 0),
                        })

        elapsed = round(_time.time() - t0, 2)

        result = {
            'points': points,
            'edges': edges,
            'total': n_total,
            'displayed': n,
            'sampled': sampled,
            'method': actual_method,
            'perplexity': actual_perplexity,
            'stats': {
                'type_counts': type_counts,
                'modality_counts': modality_counts,
                'sessions': len(session_set),
                'edges': len(edges),
                'compute_time': elapsed,
            },
        }

        # 缓存
        self._viz_cache_hash = cache_key
        self._viz_cache_result = result
        logger.info(f"可视化计算完成: method={actual_method}, n={n}, perp={actual_perplexity}, "
                     f"edges={len(edges)}, time={elapsed}s")

        return result

    def get_stats(self) -> Dict:
        """获取统计信息"""
        self._ensure_initialized()

        return {
            'total_infons': self.store.count(),
            'index_size': self.index.current_count,
            'embedding_dim': self.index.dim,
        }

    def clear(self):
        """一键清空"""
        self._ensure_initialized()
        self.store.clear_all()
        self.index.clear()
        self._viz_cache_hash = None
        self._viz_cache_result = None
        logger.info("✓ 主记忆流已完全清空 (存储 + 索引)")


# =============================================================================
# 全局管理器实例 (按用户隔离)
# =============================================================================

def get_manager(user_id: str = 'default') -> MemoryStreamManager:
    """获取指定用户的管理器实例 (懒加载, 线程安全)"""
    uid = user_id or 'default'
    if uid not in _managers:
        with _managers_lock:
            if uid not in _managers:
                _managers[uid] = MemoryStreamManager(user_id=uid)
                logger.info(f"创建用户 [{uid}] 的 MemoryStreamManager 实例")
    return _managers[uid]


def _extract_user_id_from_request() -> str:
    """从请求中提取 user_id (优先级: JSON body > query param > header)"""
    # 1. 从 JSON body 中提取
    if request.is_json:
        try:
            data = request.get_json(silent=True)
            if data and data.get('user_id'):
                return str(data['user_id'])
        except Exception:
            pass

    # 2. 从 query param 中提取
    uid = request.args.get('user_id', '')
    if uid:
        return str(uid)

    # 3. 从 header 中提取
    uid = request.headers.get('X-User-Id', '')
    if uid:
        return str(uid)

    return 'default'


# =============================================================================
# API 路由
# =============================================================================

@memory_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查 (按用户隔离)"""
    try:
        user_id = _extract_user_id_from_request()
        manager = get_manager(user_id)
        manager._ensure_initialized()
        stats = manager.get_stats()
        return jsonify({
            'status': 'ok',
            'service': 'MemoryStream',
            'user_id': user_id,
            **stats,
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@memory_bp.route('/ingest', methods=['POST'])
def ingest():
    """
    批量写入信息元 (含关联绑定, 按用户隔离)

    请求体:
    {
        "user_id": "user123",     // 用户标识 (必填)
        "infons": [...],          // 信息元列表
        "session_id": "abc123",   // 会话标识
        "round_num": 1            // 轮次编号
    }
    """
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id', '') or _extract_user_id_from_request()
        infons = data.get('infons', [])
        session_id = data.get('session_id', '')
        round_num = data.get('round_num', 1)

        if not infons:
            return jsonify({'error': '信息元列表为空'}), 400

        manager = get_manager(user_id)
        result = manager.ingest(infons, session_id, round_num)

        return jsonify(result)

    except Exception as e:
        logger.error(f"信息元写入失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/search', methods=['POST'])
def search():
    """
    向量相似度检索 (按用户隔离)

    请求体:
    {
        "user_id": "user123",     // 用户标识
        "query": "查询文本",
        "k": 5                    // 返回数量
    }
    """
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id', '') or _extract_user_id_from_request()
        query_text = data.get('query', '')
        k = data.get('k', DEFAULT_TOP_K)

        if not query_text:
            return jsonify({'error': '查询文本为空'}), 400

        manager = get_manager(user_id)
        results = manager.search(query_text, k=k)

        return jsonify({
            'results': results,
            'count': len(results),
        })

    except Exception as e:
        logger.error(f"向量检索失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/trigger-check', methods=['POST'])
def trigger_check():
    """
    风险触发式可控检索 (按用户隔离)

    请求体:
    {
        "user_id": "user123",     // 用户标识
        "infons": [...]           // 当前消息的信息元列表
    }
    """
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id', '') or _extract_user_id_from_request()
        infons = data.get('infons', [])

        if not infons:
            return jsonify({
                'triggered': False,
                'triggers': [],
                'retrieved_infons': [],
            })

        manager = get_manager(user_id)
        result = manager.trigger_check_and_retrieve(infons)

        return jsonify(result)

    except Exception as e:
        logger.error(f"触发检测失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/backtrace/<iid>', methods=['GET'])
def backtrace(iid: str):
    """
    关联回溯查询 (按用户隔离)

    返回指定信息元的证据指针和关联信息元列表
    Query param: user_id
    """
    try:
        user_id = _extract_user_id_from_request()
        manager = get_manager(user_id)
        result = manager.backtrace(iid)

        if result is None:
            return jsonify({'error': f'信息元不存在: {iid}'}), 404

        return jsonify(result)

    except Exception as e:
        logger.error(f"关联回溯查询失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/clear', methods=['POST'])
def clear():
    """
    清空指定用户的所有信息元记录和向量索引 (按用户隔离)
    用于测试、调参和保证实验可复现性

    请求体:
    {
        "user_id": "user123"      // 用户标识 (仅清空该用户的数据)
    }
    """
    try:
        user_id = 'default'
        if request.is_json:
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id', '') or _extract_user_id_from_request()
        else:
            user_id = _extract_user_id_from_request()

        manager = get_manager(user_id)
        manager.clear()

        return jsonify({
            'status': 'ok',
            'user_id': user_id,
            'message': f'用户 [{user_id}] 的所有信息元记录和向量索引已清空',
        })

    except Exception as e:
        logger.error(f"清空失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/stats', methods=['GET'])
def stats():
    """获取记忆流统计信息 (按用户隔离)"""
    try:
        user_id = _extract_user_id_from_request()
        manager = get_manager(user_id)
        result = manager.get_stats()
        result['user_id'] = user_id
        return jsonify(result)

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/remove', methods=['POST'])
def remove_infons():
    """
    按生命周期元数据删除信息元（例如删除已失效 pending）

    请求体:
    {
        "user_id": "user123",
        "session_id": "session_xxx",
        "target_type": "pending",   // 可选
        "run_ids": ["run_a", ...]   // 可选
    }
    """
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id', '') or _extract_user_id_from_request()
        session_id = data.get('session_id', '')
        target_type = data.get('target_type', None)
        run_ids = data.get('run_ids', None)

        if not session_id:
            return jsonify({'error': 'session_id 不能为空'}), 400

        manager = get_manager(user_id)
        result = manager.remove_infons_by_lifecycle(
            session_id=session_id,
            target_type=target_type,
            run_ids=run_ids if isinstance(run_ids, list) else None
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"删除信息元失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/promote-pending', methods=['POST'])
def promote_pending():
    """
    将 pending 信息元转为 message 信息元

    请求体:
    {
        "user_id": "user123",
        "session_id": "session_xxx",
        "run_ids": ["run_a", ...],
        "message_id": "msg_xxx"
    }
    """
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id', '') or _extract_user_id_from_request()
        session_id = data.get('session_id', '')
        run_ids = data.get('run_ids', [])
        message_id = data.get('message_id', '')

        if not session_id:
            return jsonify({'error': 'session_id 不能为空'}), 400
        if not isinstance(run_ids, list) or len(run_ids) == 0:
            return jsonify({'error': 'run_ids 不能为空'}), 400

        manager = get_manager(user_id)
        result = manager.promote_pending_infons(
            session_id=session_id,
            run_ids=run_ids,
            message_id=message_id
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"升级 pending 信息元失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/visualization', methods=['GET'])
def visualization():
    """
    获取信息元可视化数据 (自动降维到 2D, 按用户隔离)

    Query params:
        user_id: 用户标识
        method: 'auto' (默认, 自动选择), 'tsne' 或 'pca'

    返回:
        {
            points: [{iid, x, y, infon_type, entity, attribute, ...}],
            edges: [{source, target, similarity}],
            total: int,       // 总信息元数
            displayed: int,   // 实际显示的点数
            sampled: bool,    // 是否进行了采样
            method: str,      // 实际使用的方法
            perplexity: int,  // 自动 perplexity (仅 t-SNE)
            stats: {...},     // 统计摘要
        }
    """
    try:
        user_id = _extract_user_id_from_request()
        method = request.args.get('method', 'auto')

        manager = get_manager(user_id)
        result = manager.get_visualization_data(method=method)

        return jsonify(result)

    except Exception as e:
        logger.error(f"获取可视化数据失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# =============================================================================
# 模块初始化
# =============================================================================

def _cleanup_anonymous_dbs():
    """清理匿名用户的过期临时数据库文件 (以 memory_stream__anon_ 开头的文件)"""
    try:
        config = get_config()
        data_dir = config['data_dir']
        count = 0
        for f in os.listdir(data_dir):
            if f.startswith('memory_stream__anon_') and f.endswith('.db'):
                fpath = os.path.join(data_dir, f)
                try:
                    os.remove(fpath)
                    count += 1
                except OSError:
                    pass
                # 同时清理 WAL 和 SHM 文件
                for suffix in ['-wal', '-shm']:
                    try:
                        os.remove(fpath + suffix)
                    except OSError:
                        pass
        if count > 0:
            logger.info(f"✓ 清理了 {count} 个匿名用户临时数据库")
    except Exception as e:
        logger.warning(f"清理匿名数据库失败: {e}")


def init_memory_stream_service(preload_model: bool = False):
    """初始化主记忆流服务 (按用户隔离模式)"""
    logger.info("初始化主记忆流服务模块 (用户隔离模式)...")

    config = get_config()
    os.makedirs(config['data_dir'], exist_ok=True)

    # 启动时清理上次残留的匿名临时数据库
    _cleanup_anonymous_dbs()

    if preload_model:
        try:
            # 预加载嵌入模型 (所有用户共享同一个嵌入模型)
            load_embedding_model()
            logger.info("✓ 主记忆流嵌入模型预加载完成")
        except Exception as e:
            logger.warning(f"主记忆流服务预加载失败: {e}")

