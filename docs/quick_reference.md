# PrivaSee 快速参考卡片

## 📋 核心术语速查表

### 信息元类型
| 类型 | 前缀 | 结构 | 中文示例 | 英文示例 |
|-----|------|-----|---------|---------|
| DESC | `desc:` | entity, attribute | 姓名, 张伟 | name, Alice |
| SCEN | `scen:` | temporal, spatial | 2024年3月, 北京 | March 2024, Beijing |
| REL | `rel:` | relation_name, refs | 雇佣关系, desc:r1_1\|desc:r1_2 | employment, desc:r1_1\|desc:r1_2 |

### 触发器类型
| 触发器 | 阈值 | 检测对象 |
|--------|------|---------|
| 准标识符组合 | ≥2类 | 地理、时间、组织、兴趣、生物特征 |
| 细化线索 | 相似度≥0.85 | 与历史信息元的语义相似度 |
| 敏感域命中 | >0个 | 健康、财务、法律、亲密、PII、证件 |

### 核心参数
| 参数 | 值 | 说明 |
|-----|---|------|
| Top-K | 5 | 检索返回数量 |
| Token上限 | 500 | 检索结果token数 |
| 关联数 | 3 | 每个Infon的关联数 |
| 相似度阈值 | 0.85 | 细化触发阈值 |
| 向量维度 | 384 | Embedding维度 |

---

## 🔍 API 端点速查

### 主记忆流 API
```
POST   /api/memory/ingest          # 写入信息元
POST   /api/memory/trigger-check   # 触发检测+检索
POST   /api/memory/search          # 向量检索
GET    /api/memory/backtrace/:iid  # 关联回溯
GET    /api/memory/visualization   # 可视化数据
GET    /api/memory/stats           # 统计信息
POST   /api/memory/clear           # 清空数据
```

### 请求示例
```bash
# 写入信息元
curl -X POST http://localhost:5000/api/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "session_id": "session_abc",
    "round_num": 1,
    "infons": [
      {
        "iid": "desc:r1_1",
        "infon_type": "DESC",
        "entity": "姓名",
        "attribute": "张伟"
      }
    ]
  }'

# 触发检测
curl -X POST http://localhost:5000/api/memory/trigger-check \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "infons": [...]
  }'

# 关联回溯
curl http://localhost:5000/api/memory/backtrace/desc:r1_1?user_id=user123
```

---

## 📊 置信度评分指南

| 置信度区间 | 级别 | 指示词 | 示例 |
|-----------|------|--------|------|
| 0.95-1.0 | 极高 | is, 叫, 确切数字 | "我叫张伟" → 0.98 |
| 0.85-0.94 | 高 | works at, 住在 | "在Google工作" → 0.88 |
| 0.70-0.84 | 中 | probably, 大概 | "大概30多岁" → 0.75 |
| 0.50-0.69 | 低 | maybe, 可能 | "可能是他同事" → 0.60 |
| <0.50 | 极低 | 不要提取 | - |

---

## 🎯 准标识符类别

### 5大类别 + 关键词
| 类别 | 英文 | 中文关键词 | 英文关键词 |
|-----|------|----------|----------|
| 地理位置 | geo_location | 地址, 位置, 城市, 省份 | address, location, city |
| 时间信息 | temporal | 日期, 时间, 生日 | date, birthday, schedule |
| 组织角色 | org_role | 公司, 单位, 学校, 职位 | company, school, position |
| 罕见兴趣 | rare_interest | 病症, 过敏, 癖好 | allergy, medication, hobby |
| 生物特征 | biometric | 指纹, 人脸, 虹膜 | fingerprint, face, iris |

### 触发示例
- ✅ **触发**: "我叫张伟，住在北京" (隐含身份 + 地理)
- ✅ **触发**: "在Google工作，1994年出生" (组织 + 时间)
- ✗ **不触发**: "我今天很开心" (无准标识符)

---

## 🔐 敏感域分类

### 6大敏感域
| 敏感域 | 英文 | 风险级 | 中文关键词 | 英文关键词 |
|-------|------|-------|----------|----------|
| 健康医疗 | health_medical | ⚠️⚠️⚠️ | 病, 诊断, 药物 | disease, medical |
| 金融财务 | financial | ⚠️⚠️⚠️ | 银行, 账户, 工资 | bank, salary |
| 法律纠纷 | legal_dispute | ⚠️⚠️ | 案件, 诉讼 | lawsuit, court |
| 亲密关系 | intimate_relationship | ⚠️⚠️ | 恋人, 配偶 | spouse, dating |
| 显式PII | explicit_pii | ⚠️⚠️⚠️ | 身份证, 护照 | passport, ID_card |
| 证件图像 | document_image | ⚠️⚠️ | 证件, 合同 | certificate, contract |

---

## 🔗 证据指针格式

### 统一格式
```
{modality}:{session_id}:{round_num}:{span_locator}
```

### 按模态分类
| 模态 | 定位器格式 | 示例 |
|-----|-----------|------|
| 文本 | `start-end` | `text:abc123:1:0-10` |
| 图像 | `ocr_box_{id}` | `image:abc123:2:ocr_box_3` |
| 音频 | `seg_{id}` | `audio:abc123:3:seg_5` |

### 解析示例
```javascript
// 输入: "text:session_abc:1:0-10"
{
  modality: "text",
  session_id: "session_abc",
  round_num: 1,
  locator_type: "char_range",
  char_start: 0,
  char_end: 10
}
```

---

## 📈 HNSW 参数配置

| 参数 | 值 | 说明 | 影响 |
|-----|---|------|------|
| M | 16 | 最大连接数 | 影响准确率和内存 |
| ef_construction | 200 | 构建候选数 | 构建速度 vs 质量 |
| ef_search | 50 | 搜索候选数 | 查询速度 vs 准确率 |
| space | cosine | 距离度量 | 相似度计算方式 |

### 性能指标
| 数据规模 | 检索延迟 | Recall@5 | 内存占用 |
|---------|---------|----------|---------|
| 1K | <1ms | >0.98 | ~15MB |
| 10K | <5ms | >0.96 | ~150MB |
| 100K | <10ms | >0.95 | ~1.5GB |
| 1M | <50ms | >0.90 | ~15GB |

---

## 💡 提示词工程原则

### 4层结构
1. **任务定义** (What to do)
   - 角色: "You are an information extractor"
   - 目标: "Extract facts as CSV lines"

2. **格式约束** (How to output)
   - 输出格式: CSV/JSON
   - 字段规范: desc:r{round}_{n},DESC,entity,attribute,type,confidence

3. **内容规则** (What to extract/skip)
   - 提取: 名称、数字、行为
   - 跳过: 常见词、填充词

4. **质量控制** (Confidence)
   - 置信度评分: 0.0-1.0
   - 指示词识别: "is" → 高, "maybe" → 低

### 优化技巧
- ✅ 精简上下文 (针对4B小模型)
- ✅ 原子化提取 (一行一个事实)
- ✅ 示例驱动 (Few-shot)
- ✅ 负面约束 (明确不要提取什么)
- ✅ 自查清单 (Self-Checklist)

---

## 🐛 已知问题与解决方案

### 问题1: Infon Cloud英文长度问题
**现象**: 长英文文本提取时出现换行符导致CSV解析错误

**解决方案**:
1. 转义换行符: `\n` → `\\n`
2. 长度截断: attribute字段限制200字符
3. 逗号替换: `,` → `;` (避免CSV冲突)

**代码位置**: `frontend/src/templates/infons.js` - OUTPUT_FORMAT

### 问题2: 提取过度
**现象**: 提取了过多无用信息

**解决方案**:
- 添加数量限制: DESC<12, SCEN<2, REL<5
- 强化"What to Skip"规则
- 提高置信度阈值 (>0.6才提取)

---

## 📚 论文撰写速查

### 章节建议
```
3.3 Memory Stream Module (主记忆流) ← 贡献1
  3.3.1 Risk-Triggered Retrieval
  3.3.2 HNSW Vector Index
  3.3.3 User-Level Isolation
  
3.4 Association Backtracking Module (关联回溯) ← 贡献2
  3.4.1 Write-Time Binding
  3.4.2 Evidence Pointer
  3.4.3 Association Network
```

### 图表建议
- **Figure 3**: Memory Stream Trigger Flow (触发决策流程)
- **Figure 4**: Association Network Example (关联网络示例)
- **Table 2**: Module Features Comparison (模块功能对比)
- **Table 3**: Performance Benchmarks (性能基准)

### 术语一致性
| 概念 | 推荐 | 避免 |
|-----|------|------|
| 信息元 | **Infon** | Info, Entity |
| 主记忆流 | **Memory Stream** | Memory Bank |
| 关联回溯 | **Association Backtracking** | Link Tracing |
| 触发检索 | **Triggered Retrieval** | Selective Search |

---

## 🚀 快速测试命令

### 本地开发
```bash
# 启动后端
cd backend && bash start.sh

# 启动前端
cd frontend && npm run dev

# 检查健康状态
curl http://localhost:5000/api/health
```

### 测试记忆流
```bash
# 1. 清空数据
curl -X POST http://localhost:5000/api/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user"}'

# 2. 写入测试数据
curl -X POST http://localhost:5000/api/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"test_user",
    "session_id":"test_session",
    "round_num":1,
    "infons":[
      {"iid":"desc:r1_1","infon_type":"DESC","entity":"姓名","attribute":"张伟"}
    ]
  }'

# 3. 检查统计
curl "http://localhost:5000/api/memory/stats?user_id=test_user"

# 4. 可视化
curl "http://localhost:5000/api/memory/visualization?user_id=test_user"
```

---

## 📖 延伸阅读

### 代码文件
- `backend/services/memory_stream_service.py` - 主记忆流实现
- `frontend/src/templates/infons.js` - 提示词模板
- `frontend/src/store/slices/infonSlice.js` - 前端状态管理

### 文档
- `docs/memory_stream_documentation.md` - 详细实现文档
- `docs/terminology_and_flowcharts.md` - 术语和流程图
- `README.md` - 系统架构总览

---

## ✨ 常见使用场景

### 场景1: 隐私拼图检测
```
用户输入序列:
R1: "我叫张伟"
R2: "在Google工作" (触发: 组织角色)
R3: "住在北京海淀区" (触发: 组织+地理 → 准标识符组合)
→ 系统检索R1,R2 → 风险提示: "多条信息可能被组合重识别"
```

### 场景2: 细化线索追踪
```
历史: "我的公司是Google" (相似度: 0.85)
新输入: "我在谷歌工作" (触发: 细化线索)
→ 系统识别为同一实体的细化 → 关联两条信息
```

### 场景3: 敏感域预警
```
用户输入: "我有糖尿病，需要胰岛素"
→ 触发: 敏感域命中 (health_medical)
→ 系统提示: "涉及健康医疗敏感信息"
```

---

**最后更新**: 2025-02-09  
**版本**: v1.0  
**作者**: PrivaSee Team

