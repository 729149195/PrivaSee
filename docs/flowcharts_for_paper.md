# PrivaSee 论文流程图集

本文档包含适合用于论文的高质量流程图，使用Mermaid语法。

---

## Figure 1: 系统整体架构

```mermaid
graph TB
    subgraph "Input Layer"
        Text[Text Input]
        Image[Image Upload]
        Audio[Audio Recording]
    end
    
    subgraph "Extraction Layer"
        OCR[OCR Service<br/>DeepSeek-OCR]
        Whisper[Whisper Service<br/>Speech-to-Text]
        LLM1[LLM<br/>Infon Extraction]
    end
    
    subgraph "Memory Layer"
        MS[Memory Stream<br/>Vector Index + SQLite]
        RTD[Risk Trigger<br/>Detection]
        AB[Association<br/>Backtracking]
    end
    
    subgraph "Inference Layer"
        LLM2[LLM<br/>Privacy Risk Inference]
        LawKB[Law Knowledge Base<br/>GDPR/PIPL/CCPA]
    end
    
    subgraph "Output Layer"
        Risk[Risk Analysis]
        Suggest[Protection Suggestions]
        Viz[Visualization]
    end
    
    Text --> LLM1
    Image --> OCR --> LLM1
    Audio --> Whisper --> LLM1
    
    LLM1 --> MS
    MS --> RTD
    RTD -->|Triggered| AB
    AB --> LLM2
    
    LawKB --> LLM2
    LLM2 --> Risk
    LLM2 --> Suggest
    MS --> Viz
    
    style MS fill:#90EE90
    style AB fill:#87CEEB
    style RTD fill:#FFD700
```

**说明**: 展示PrivaSee的五层架构，突出Memory Stream和Association Backtracking模块。

---

## Figure 2: 主记忆流触发决策流程

```mermaid
flowchart TD
    Start([New Message]) --> Extract[Extract Infons]
    Extract --> Ingest[Ingest to Memory Stream<br/>+ Compute Associations]
    
    Ingest --> Trigger{Risk Trigger<br/>Detection}
    
    Trigger --> T1[Trigger 1<br/>QI Combination]
    Trigger --> T2[Trigger 2<br/>Refinement Clue]
    Trigger --> T3[Trigger 3<br/>Sensitive Domain]
    
    T1 --> D1{Categories ≥ 2?}
    T2 --> D2{Similarity ≥ 0.85?}
    T3 --> D3{Domain Hit?}
    
    D1 -->|Yes| Retrieve[Vector Retrieval<br/>HNSW Index]
    D1 -->|No| Skip[Skip Retrieval]
    D2 -->|Yes| Retrieve
    D2 -->|No| Skip
    D3 -->|Yes| Retrieve
    D3 -->|No| Skip
    
    Retrieve --> Rank[Rank by Similarity]
    Rank --> Limit[Apply Limits<br/>Max 5 infons / 500 tokens]
    Limit --> Return([Return Historical Infons])
    
    Skip --> Empty([Return Empty])
    
    style Retrieve fill:#90EE90
    style Skip fill:#FFB6C1
    style T1 fill:#FFE4B5
    style T2 fill:#FFE4B5
    style T3 fill:#FFE4B5
```

**说明**: 展示风险触发式可控检索的决策流程，三种触发器的OR逻辑。

---

## Figure 3: 准标识符组合检测详细流程

```mermaid
flowchart LR
    Start([Input Infons]) --> Parse[Parse Entity<br/>+ Attribute]
    
    Parse --> Match[Keyword Matching]
    
    Match --> Cat1{Geo<br/>Location?}
    Match --> Cat2{Temporal<br/>Info?}
    Match --> Cat3{Org<br/>Role?}
    Match --> Cat4{Rare<br/>Interest?}
    Match --> Cat5{Biometric?}
    
    Cat1 -->|Match| Set[Category Set]
    Cat2 -->|Match| Set
    Cat3 -->|Match| Set
    Cat4 -->|Match| Set
    Cat5 -->|Match| Set
    
    Set --> Count[Count Categories]
    Count --> Judge{Count ≥ 2?}
    
    Judge -->|Yes| Trigger([✓ Trigger])
    Judge -->|No| NoTrigger([✗ No Trigger])
    
    style Trigger fill:#90EE90
    style NoTrigger fill:#FFB6C1
    style Set fill:#FFD700
```

**说明**: 准标识符组合检测的详细流程，展示5大类别的分类逻辑。

---

## Figure 4: 关联回溯机制

```mermaid
flowchart TD
    subgraph "Write-Time Binding"
        New[New Infon] --> Embed[Compute<br/>Embedding]
        Embed --> Search[Vector Search<br/>Top-K=3]
        Search --> Bind[Bind Associations<br/>with Similarity]
        Bind --> Store[Store to DB<br/>+ Update Index]
    end
    
    subgraph "Backtracing Query"
        Query([Query by IID]) --> Load[Load from DB]
        Load --> ParseEP[Parse Evidence<br/>Pointer]
        ParseEP --> GetAssoc[Get Association<br/>List]
        GetAssoc --> Enrich[Enrich with<br/>Full Details]
        Enrich --> Return([Return Chain])
    end
    
    Store -.->|Historical Data| Search
    Store -.->|Stored Associations| GetAssoc
    
    style Embed fill:#87CEEB
    style Search fill:#90EE90
    style Bind fill:#FFD700
    style ParseEP fill:#FFB6C1
```

**说明**: 展示关联回溯的两个核心流程：写入时绑定和回溯查询。

---

## Figure 5: 证据指针格式与解析

```mermaid
flowchart LR
    subgraph "Evidence Pointer Format"
        Format["{modality}:{session}:{round}:{locator}"]
    end
    
    subgraph "Modality-Specific Locators"
        Text["text → char_range<br/>Example: 0-10"]
        Image["image → ocr_box<br/>Example: ocr_box_3"]
        Audio["audio → segment<br/>Example: seg_5"]
    end
    
    subgraph "Parsing Result"
        Parsed["Parsed Object:<br/>- modality<br/>- session_id<br/>- round_num<br/>- locator_type<br/>- position"]
    end
    
    Format --> Text
    Format --> Image
    Format --> Audio
    
    Text --> Parsed
    Image --> Parsed
    Audio --> Parsed
    
    style Format fill:#FFD700
    style Parsed fill:#90EE90
```

**说明**: 统一跨模态证据指针的格式和解析逻辑。

---

## Figure 6: 信息元关联网络示例

```mermaid
graph TB
    subgraph "Session 1"
        N1["desc:r1_1<br/>姓名=张伟<br/>text:s1:1:0-10"]
        N2["desc:r1_2<br/>年龄=28<br/>text:s1:1:15-20"]
        N3["scen:r1_1<br/>2024年3月@北京<br/>text:s1:1:25-40"]
    end
    
    subgraph "Session 2"
        N4["desc:r2_1<br/>公司=Google<br/>text:s2:1:0-15"]
        N5["desc:r2_2<br/>职位=工程师<br/>text:s2:1:20-30"]
    end
    
    subgraph "Session 3"
        N6["desc:r3_1<br/>地址=海淀区<br/>image:s3:1:ocr_box_2"]
    end
    
    N1 -.->|0.85| N2
    N1 -.->|0.78| N4
    N2 -.->|0.72| N3
    N4 -.->|0.88| N5
    N1 -.->|0.65| N6
    N3 -.->|0.70| N6
    
    style N1 fill:#FFB6C1
    style N4 fill:#87CEEB
    style N6 fill:#90EE90
```

**说明**: 展示跨会话的信息元关联网络，边权重表示语义相似度。

---

## Figure 7: 提示词工程四层结构

```mermaid
flowchart TB
    subgraph "Layer 1: Task Definition"
        L1["Role: Information Extractor<br/>Goal: Extract facts as CSV"]
    end
    
    subgraph "Layer 2: Format Constraints"
        L2["Output Format: CSV<br/>Schema: iid,type,field1,field2,value_type,confidence"]
    end
    
    subgraph "Layer 3: Content Rules"
        L3A["What to Extract:<br/>- Names, numbers, actions"]
        L3B["What to Skip:<br/>- Common words, fillers"]
    end
    
    subgraph "Layer 4: Quality Control"
        L4["Confidence Scoring:<br/>0.95+ Explicit<br/>0.85-0.94 Clear<br/>0.70-0.84 Inferred<br/>0.50-0.69 Uncertain"]
    end
    
    L1 --> L2
    L2 --> L3A
    L2 --> L3B
    L3A --> L4
    L3B --> L4
    
    style L1 fill:#FFE4B5
    style L2 fill:#87CEEB
    style L3A fill:#90EE90
    style L3B fill:#FFB6C1
    style L4 fill:#FFD700
```

**说明**: 展示针对小参数LLM的提示词工程四层结构。

---

## Figure 8: HNSW向量索引结构

```mermaid
graph TB
    subgraph "Layer 0 (Base)"
        N01["Node 1"] --- N02["Node 2"]
        N02 --- N03["Node 3"]
        N03 --- N04["Node 4"]
        N04 --- N05["Node 5"]
    end
    
    subgraph "Layer 1"
        N11["Node 1"] --- N13["Node 3"]
        N13 --- N15["Node 5"]
    end
    
    subgraph "Layer 2"
        N21["Node 1"] --- N25["Node 5"]
    end
    
    Query([Query Vector]) -.->|1. Search L2| N21
    N21 -.->|2. Descend to L1| N11
    N11 -.->|3. Descend to L0| N01
    N01 -.->|4. Find Neighbors| Result([Top-K Results])
    
    style Query fill:#FFD700
    style Result fill:#90EE90
    style N21 fill:#FFB6C1
    style N11 fill:#87CEEB
    style N01 fill:#90EE90
```

**说明**: HNSW分层图结构，展示从顶层到底层的搜索路径。

---

## Figure 9: 隐私拼图攻击场景

```mermaid
flowchart TD
    subgraph "Attacker's View"
        A1[Round 1: "我叫张伟"]
        A2[Round 2: "在Google工作"]
        A3[Round 3: "住在北京海淀区"]
        A4[Round 4: "1994年出生"]
    end
    
    subgraph "Puzzle Assembly"
        P1["姓名: 张伟"]
        P2["公司: Google"]
        P3["地址: 北京海淀区"]
        P4["年龄: ~30岁"]
    end
    
    subgraph "Re-identification"
        R1["Search: Google员工 + 北京 + 30岁 + 张伟"]
        R2["Result: Unique Individual Identified"]
    end
    
    A1 --> P1
    A2 --> P2
    A3 --> P3
    A4 --> P4
    
    P1 --> R1
    P2 --> R1
    P3 --> R1
    P4 --> R1
    
    R1 --> R2
    
    style R2 fill:#FF6B6B
    style R1 fill:#FFD700
```

**说明**: 展示隐私拼图攻击的场景，多条看似无害的信息组合后可重识别个人。

---

## Figure 10: 系统性能对比

```mermaid
graph LR
    subgraph "Traditional RAG"
        T1["Always Retrieve<br/>High Privacy Risk"]
        T2["No Trigger Detection<br/>Over-Exposure"]
        T3["Static Retrieval<br/>Fixed K"]
    end
    
    subgraph "PrivaSee Memory Stream"
        P1["Triggered Retrieval<br/>Controlled Exposure"]
        P2["Triple Trigger Detection<br/>Risk-Aware"]
        P3["Dynamic Limits<br/>5 infons / 500 tokens"]
    end
    
    T1 -.->|vs| P1
    T2 -.->|vs| P2
    T3 -.->|vs| P3
    
    P1 --> Adv1["✓ Reduced Exposure"]
    P2 --> Adv2["✓ Higher Precision"]
    P3 --> Adv3["✓ Better Control"]
    
    style T1 fill:#FFB6C1
    style T2 fill:#FFB6C1
    style T3 fill:#FFB6C1
    style P1 fill:#90EE90
    style P2 fill:#90EE90
    style P3 fill:#90EE90
```

**说明**: 对比传统RAG和PrivaSee的风险触发式检索机制。

---

## 使用指南

### 在Markdown中渲染

这些Mermaid图可以直接在支持Mermaid的Markdown渲染器中显示，如：
- GitHub
- GitLab
- Obsidian
- Typora
- VS Code (with Mermaid extension)

### 导出为图片

**方法1: 使用Mermaid Live Editor**
1. 访问 https://mermaid.live/
2. 粘贴代码
3. 导出为PNG/SVG

**方法2: 使用命令行工具**
```bash
# 安装 mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# 转换为PNG
mmdc -i flowchart.mmd -o flowchart.png

# 转换为SVG (矢量图，推荐用于论文)
mmdc -i flowchart.mmd -o flowchart.svg
```

**方法3: 使用Python**
```python
from mermaid import Mermaid

mermaid_code = """
graph TB
    A --> B
"""

Mermaid(mermaid_code).to_png('output.png')
```

### 论文中的使用建议

**图表编号与标题**:
- Figure 1: System Architecture of PrivaSee
- Figure 2: Risk-Triggered Retrieval Decision Flow
- Figure 3: Quasi-Identifier Combination Detection
- Figure 4: Association Backtracking Mechanism
- Figure 5: Cross-Modal Evidence Pointer Format
- Figure 6: Information Element Association Network
- Figure 7: Four-Layer Prompt Engineering Structure
- Figure 8: HNSW Vector Index Structure
- Figure 9: Privacy Puzzle Attack Scenario
- Figure 10: Performance Comparison with Traditional RAG

**图表说明示例**:
> **Figure 2**: Risk-Triggered Retrieval Decision Flow. The system employs three triggers (QI combination, refinement clue, and sensitive domain hit) to determine whether to retrieve historical infons. Only when at least one trigger is activated, the system performs vector retrieval with hard limits (max 5 infons or 500 tokens) to balance privacy protection and information utility.

---

## 自定义样式

### 修改颜色

```mermaid
graph TB
    A[Node A]
    B[Node B]
    
    A --> B
    
    style A fill:#FF6B6B,stroke:#333,stroke-width:2px
    style B fill:#4ECDC4,stroke:#333,stroke-width:2px
```

### 常用配色方案

| 用途 | 颜色代码 | 示例 |
|-----|---------|------|
| 成功/通过 | `#90EE90` | 浅绿色 |
| 警告/注意 | `#FFD700` | 金黄色 |
| 错误/拒绝 | `#FFB6C1` | 浅粉色 |
| 信息/中性 | `#87CEEB` | 浅蓝色 |
| 强调/重要 | `#FFE4B5` | 浅橙色 |
| 危险/高风险 | `#FF6B6B` | 红色 |

---

## 高级技巧

### 子图嵌套

```mermaid
graph TB
    subgraph "Outer"
        subgraph "Inner 1"
            A[Node A]
        end
        subgraph "Inner 2"
            B[Node B]
        end
    end
    
    A --> B
```

### 条件分支

```mermaid
flowchart TD
    Start --> Condition{Condition?}
    Condition -->|Yes| Action1[Action 1]
    Condition -->|No| Action2[Action 2]
    Action1 --> End
    Action2 --> End
```

### 时序图 (可选)

```mermaid
sequenceDiagram
    participant User
    participant System
    participant MemoryStream
    
    User->>System: Send Message
    System->>MemoryStream: Extract Infons
    MemoryStream->>MemoryStream: Trigger Detection
    alt Triggered
        MemoryStream->>System: Return Historical Infons
    else Not Triggered
        MemoryStream->>System: Return Empty
    end
    System->>User: Display Response
```

---

## 版本记录

- **v1.0** (2025-02-09): 初始版本，包含10个核心流程图
- **待更新**: 根据审稿意见调整图表样式

---

**作者**: PrivaSee Team  
**用途**: 论文插图、演示文稿、技术文档  
**许可**: MIT

