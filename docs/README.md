# PrivaSee 文档中心

本目录包含 PrivaSee 项目的详细技术文档，特别是**主记忆流**和**关联回溯**两个核心模块的实现说明。

---

## 📚 文档列表

### 1. 核心实现文档

#### 📖 [memory_stream_documentation.md](./memory_stream_documentation.md)
**主记忆流与关联回溯完整实现文档**

**内容概览**:
- ✅ 模块概述与功能定位
- ✅ 提示词优化方法论 (针对4B小参数模型)
- ✅ 主记忆流模块详细设计
  - 风险触发式可控检索
  - HNSW向量索引
  - 三种触发器实现
- ✅ 关联回溯模块详细设计
  - 写入时同步关联绑定
  - 跨模态证据指针
  - 关联网络可视化
- ✅ 触发条件框图
- ✅ 术语与英文定义
- ✅ 技术贡献总结
- ✅ API使用示例

**适用场景**: 深入理解系统实现、代码开发、技术答辩

---

#### 📖 [terminology_and_flowcharts.md](./terminology_and_flowcharts.md)
**术语对照与流程框图详解**

**内容概览**:
- ✅ 核心概念术语 (中英文对照)
  - 信息元相关 (Infon, IID, DESC/SCEN/REL)
  - 记忆流相关 (Memory Stream, Vector Index)
  - 关联回溯相关 (Association Backtracking, Evidence Pointer)
  - 触发检测相关 (Trigger, QI, Sensitive Domain)
- ✅ 触发条件详细框图 (Mermaid)
  - 准标识符组合检测
  - 细化线索检测
  - 敏感域命中检测
  - 关联回溯查询流程
- ✅ 模块功能包装说明 (论文用)
- ✅ 论文撰写建议
  - 章节结构
  - 图表建议
  - 实验设计
  - 术语一致性
- ✅ Infon Cloud英文提取问题分析与解决方案

**适用场景**: 论文撰写、术语统一、流程图制作

---

#### 📖 [quick_reference.md](./quick_reference.md)
**快速参考卡片**

**内容概览**:
- ✅ 核心术语速查表
- ✅ API端点速查
- ✅ 置信度评分指南
- ✅ 准标识符类别与关键词
- ✅ 敏感域分类
- ✅ 证据指针格式
- ✅ HNSW参数配置
- ✅ 提示词工程原则
- ✅ 已知问题与解决方案
- ✅ 论文撰写速查
- ✅ 快速测试命令
- ✅ 常见使用场景

**适用场景**: 日常开发、快速查询、新人入门

---

#### 📖 [flowcharts_for_paper.md](./flowcharts_for_paper.md)
**论文流程图集**

**内容概览**:
- ✅ 10个高质量Mermaid流程图
  - Figure 1: 系统整体架构
  - Figure 2: 主记忆流触发决策流程
  - Figure 3: 准标识符组合检测详细流程
  - Figure 4: 关联回溯机制
  - Figure 5: 证据指针格式与解析
  - Figure 6: 信息元关联网络示例
  - Figure 7: 提示词工程四层结构
  - Figure 8: HNSW向量索引结构
  - Figure 9: 隐私拼图攻击场景
  - Figure 10: 系统性能对比
- ✅ 图表使用指南
- ✅ 导出为图片的方法
- ✅ 自定义样式技巧

**适用场景**: 论文插图、演示文稿、技术展示

---

### 2. 问题修复文档

#### 📖 [infon_cloud_english_fix.md](./infon_cloud_english_fix.md)
**Infon Cloud 英文提取长度问题修复方案**

**内容概览**:
- ✅ 问题分析
  - 换行符导致CSV解析错误
  - 长度限制缺失
  - 逗号冲突
- ✅ 解决方案对比 (4种方案)
- ✅ 推荐方案: 混合方案 (长度截断 + 转义换行符)
- ✅ 实施步骤
  - 修改提示词模板
  - 修改解析器
  - 更新前端调用代码
- ✅ 测试用例
- ✅ 回归测试
- ✅ 部署检查清单
- ✅ 后续优化方向

**适用场景**: Bug修复、代码改进、质量保证

---

## 🎯 快速导航

### 按使用场景

| 场景 | 推荐文档 |
|-----|---------|
| 📝 **论文撰写** | [terminology_and_flowcharts.md](./terminology_and_flowcharts.md) + [flowcharts_for_paper.md](./flowcharts_for_paper.md) |
| 💻 **代码开发** | [memory_stream_documentation.md](./memory_stream_documentation.md) + [quick_reference.md](./quick_reference.md) |
| 🐛 **Bug修复** | [infon_cloud_english_fix.md](./infon_cloud_english_fix.md) |
| 🎓 **新人入门** | [quick_reference.md](./quick_reference.md) → [memory_stream_documentation.md](./memory_stream_documentation.md) |
| 🎤 **技术答辩** | [memory_stream_documentation.md](./memory_stream_documentation.md) + [flowcharts_for_paper.md](./flowcharts_for_paper.md) |
| 🔍 **术语查询** | [quick_reference.md](./quick_reference.md) (速查) 或 [terminology_and_flowcharts.md](./terminology_and_flowcharts.md) (详细) |

### 按模块

| 模块 | 相关文档 |
|-----|---------|
| **主记忆流 (Memory Stream)** | [memory_stream_documentation.md](./memory_stream_documentation.md) § 主记忆流模块 |
| **关联回溯 (Association Backtracking)** | [memory_stream_documentation.md](./memory_stream_documentation.md) § 关联回溯模块 |
| **风险触发检测 (Risk Trigger Detection)** | [terminology_and_flowcharts.md](./terminology_and_flowcharts.md) § 触发条件详细框图 |
| **提示词工程 (Prompt Engineering)** | [memory_stream_documentation.md](./memory_stream_documentation.md) § 提示词优化方法 |
| **信息元提取 (Infon Extraction)** | [infon_cloud_english_fix.md](./infon_cloud_english_fix.md) |

---

## 📊 核心概念速览

### 主记忆流 (Memory Stream)

**定义**: 跨会话的持久化信息元向量库，配合风险触发式可控检索

**核心特性**:
- ✅ 仅追加不更新 (Append-Only)
- ✅ HNSW向量索引 (毫秒级检索)
- ✅ 三种触发器 (准标识符组合、细化线索、敏感域命中)
- ✅ 用户级隔离 (Multi-Tenancy)
- ✅ 可控检索 (最多5条 / 500 tokens)

**技术栈**: SQLite + HNSW + sentence-transformers

---

### 关联回溯 (Association Backtracking)

**定义**: 基于向量相似度的 Top-K 关联绑定机制，支持跨模态证据追踪

**核心特性**:
- ✅ 写入时同步绑定 (无需后台进程)
- ✅ Top-3 语义关联
- ✅ 统一证据指针 (`{modality}:{session}:{round}:{locator}`)
- ✅ 跨模态追踪 (text/image/audio)
- ✅ 关联网络可视化

**技术栈**: Vector Search + Graph Structure

---

### 三种触发器

| 触发器 | 阈值 | 检测对象 | 示例 |
|--------|------|---------|------|
| **准标识符组合** | ≥2类 | 地理、时间、组织、兴趣、生物特征 | "在Google工作" + "住北京" |
| **细化线索** | 相似度≥0.85 | 与历史信息元的语义相似度 | "公司是Google" → "在谷歌工作" |
| **敏感域命中** | >0个 | 健康、财务、法律、亲密、PII、证件 | "我有糖尿病" |

---

## 🔧 常用命令

### API测试

```bash
# 健康检查
curl http://localhost:5000/api/memory/health

# 写入信息元
curl -X POST http://localhost:5000/api/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","session_id":"s1","round_num":1,"infons":[...]}'

# 触发检测
curl -X POST http://localhost:5000/api/memory/trigger-check \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","infons":[...]}'

# 关联回溯
curl "http://localhost:5000/api/memory/backtrace/desc:r1_1?user_id=test"

# 可视化
curl "http://localhost:5000/api/memory/visualization?user_id=test&method=auto"

# 统计信息
curl "http://localhost:5000/api/memory/stats?user_id=test"

# 清空数据
curl -X POST http://localhost:5000/api/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test"}'
```

### 开发环境

```bash
# 启动后端
cd backend && bash start.sh

# 启动前端
cd frontend && npm run dev

# 运行基准测试
cd benchmark && python run_benchmark.py evaluate
```

---

## 📈 性能指标

### HNSW索引性能

| 数据规模 | 检索延迟 | Recall@5 | 内存占用 |
|---------|---------|----------|---------|
| 1K | <1ms | >0.98 | ~15MB |
| 10K | <5ms | >0.96 | ~150MB |
| 100K | <10ms | >0.95 | ~1.5GB |
| 1M | <50ms | >0.90 | ~15GB |

### 触发器准确率 (预期)

| 触发器 | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| 准标识符组合 | 0.85+ | 0.75+ | 0.80+ |
| 细化线索 | 0.90+ | 0.70+ | 0.79+ |
| 敏感域命中 | 0.95+ | 0.80+ | 0.87+ |

---

## 🎓 论文撰写建议

### 章节结构

```
3. System Design
  3.1 Overview
  3.2 Infon Extraction
    3.2.1 Prompt Engineering for Small LLMs
    3.2.2 Multi-Modal Extraction
  3.3 Memory Stream Module ← 贡献1
    3.3.1 Architecture
    3.3.2 Risk-Triggered Retrieval
    3.3.3 HNSW Vector Index
  3.4 Association Backtracking Module ← 贡献2
    3.4.1 Write-Time Association Binding
    3.4.2 Cross-Modal Evidence Tracing
    3.4.3 Association Network Visualization
  3.5 Privacy Risk Inference
```

### 推荐图表

- **Figure 2**: Memory Stream Trigger Decision Flow ([flowcharts_for_paper.md](./flowcharts_for_paper.md) § Figure 2)
- **Figure 4**: Association Backtracking Mechanism ([flowcharts_for_paper.md](./flowcharts_for_paper.md) § Figure 4)
- **Figure 6**: Information Element Association Network ([flowcharts_for_paper.md](./flowcharts_for_paper.md) § Figure 6)
- **Table 2**: Memory Stream Module Features ([terminology_and_flowcharts.md](./terminology_and_flowcharts.md) § 模块功能包装)
- **Table 3**: Performance Benchmarks (上方性能指标表)

### 术语一致性

| 概念 | 推荐 | 避免 |
|-----|------|------|
| 信息元 | **Infon** | Info, Entity |
| 主记忆流 | **Memory Stream** | Memory Bank |
| 关联回溯 | **Association Backtracking** | Link Tracing |
| 触发检索 | **Triggered Retrieval** | Selective Search |
| 准标识符 | **Quasi-Identifier** | Semi-Identifier |

---

## 🐛 已知问题

### 1. Infon Cloud 英文提取长度问题

**状态**: 📝 待修复

**描述**: 长英文文本提取时出现换行符导致CSV解析错误

**解决方案**: 见 [infon_cloud_english_fix.md](./infon_cloud_english_fix.md)

**优先级**: 🔴 高

---

### 2. 触发器误报问题

**状态**: 🔍 调研中

**描述**: 准标识符组合触发器在某些场景下误报率较高

**临时方案**: 调整关键词列表，增加上下文判断

**优先级**: 🟡 中

---

## 📞 联系方式

**项目负责人**: 张翔轩

**技术支持**: PrivaSee Team

**问题反馈**: 提交 Issue 到项目仓库

---

## 📝 更新日志

### v1.0 (2025-02-09)

**新增**:
- ✅ 主记忆流与关联回溯完整实现文档
- ✅ 术语对照与流程框图详解
- ✅ 快速参考卡片
- ✅ 论文流程图集 (10个高质量图表)
- ✅ Infon Cloud英文提取问题修复方案

**改进**:
- ✅ 统一术语定义 (中英文对照)
- ✅ 完善API使用示例
- ✅ 添加性能基准数据

**待办**:
- ⏳ 实施Infon Cloud英文提取修复
- ⏳ 完善单元测试
- ⏳ 添加更多使用示例

---

## 📚 延伸阅读

### 代码文件

- `backend/services/memory_stream_service.py` - 主记忆流实现
- `frontend/src/templates/infons.js` - 提示词模板
- `frontend/src/store/slices/infonSlice.js` - 前端状态管理
- `benchmark/infon_benchmark_v2.py` - 基准测试

### 外部资源

- HNSW论文: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
- Sentence-Transformers: https://www.sbert.net/
- GDPR官方文档: https://gdpr.eu/
- PIPL官方文档: http://www.npc.gov.cn/

---

## 📄 许可证

MIT License

---

**最后更新**: 2025-02-09  
**文档版本**: v1.0  
**维护者**: PrivaSee Team

