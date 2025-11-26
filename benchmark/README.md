# ACE 2005 信息元提取基准测试工具

本工具用于将 ACE 2005 数据集转换为 PrivaSee 的信息元格式（DESC/SCEN/REL），并进行信息元提取的准确性评估。

## 目录结构

```
benchmark/
├── __init__.py           # 包初始化
├── ace_parser.py         # ACE 2005 XML解析器
├── ace_to_infons.py      # ACE到Infons格式转换器
├── evaluator.py          # 评估器（计算P/R/F1）
├── run_benchmark.py      # 主运行脚本（CLI）
├── config.py             # 配置文件
├── README.md             # 本文档
└── gold_data/            # 生成的Gold标准数据（运行后生成）
    ├── gold.json
    ├── gold.csv
    └── statistics.json
```

## 映射关系

### ACE 2005 → PrivaSee Infons

| ACE 2005 | PrivaSee | 说明 |
|----------|----------|------|
| Entity + Entity Mention | **DESC** | `entity`=类型, `attribute`=文本 |
| Timex2 | **DESC** | `entity`="Time", `attribute`=时间表达式 |
| Value | **DESC** | `entity`=值类型, `attribute`=数值 |
| Event (Time + Place args) | **SCEN** | `temporal`=时间, `spatial`=地点 |
| Relation | **REL** | `relation_name`=关系类型 |
| Event (participants) | **REL** | `relation_name`=事件类型 |

### 实体类型映射

| ACE Type | PrivaSee Entity |
|----------|-----------------|
| PER | Person/Individual/Group |
| ORG | Organization/Government/Commercial/... |
| GPE | GeoPoliticalEntity/Nation/City/... |
| LOC | Location/Address/Region/... |
| FAC | Facility/Building/Airport/... |
| VEH | Vehicle |
| WEA | Weapon |

### 关系类型映射

| ACE Relation | PrivaSee REL Name |
|--------------|-------------------|
| PHYS:Located | located_at |
| PHYS:Near | near |
| PART-WHOLE:Geographical | part_of_geo |
| PER-SOC:Family | family_relation |
| PER-SOC:Business | business_relation |
| ORG-AFF:Employment | employed_by |
| ORG-AFF:Membership | member_of |
| ART:User-Owner-... | uses_or_owns |
| GEN-AFF:Citizen-... | citizen_of |

## 快速开始

### 1. 转换ACE数据集

```bash
# 转换全部数据（Arabic语言，adj标注级别）
python -m benchmark.run_benchmark convert \
    --ace-path ./test-data/ace_2005_td_v7 \
    --output ./benchmark/gold_data

# 限制数量进行测试
python -m benchmark.run_benchmark convert \
    --ace-path ./test-data/ace_2005_td_v7 \
    --output ./benchmark/gold_data \
    --limit 100
```

### 2. 查看数据集统计

```bash
python -m benchmark.run_benchmark stats \
    --ace-path ./test-data/ace_2005_td_v7
```

### 3. 评估提取结果

准备你的预测结果文件 `predictions.json`，格式如下：

```json
{
  "DOC_ID_1": [
    {"iid": "desc:1", "infon_type": "DESC", "entity": "Person", "attribute": "John"},
    {"iid": "rel:1", "infon_type": "REL", "relation_name": "located_at", "arg_refs": ["desc:1", "desc:2"]}
  ],
  "DOC_ID_2": [...]
}
```

或使用compact格式字符串：

```json
{
  "DOC_ID_1": "desc:1,DESC,Person,John,string,0.95\nrel:1,REL,located_at,desc:1|desc:2,0.90"
}
```

然后运行评估：

```bash
python -m benchmark.run_benchmark evaluate \
    --gold ./benchmark/gold_data/gold.json \
    --predictions ./predictions.json \
    --output ./evaluation_result.json \
    --detailed
```

### 4. 完整测试流程

需要先启动 PrivaSee 后端服务：

```bash
# 启动后端（在另一个终端）
cd backend && python app.py

# 运行测试
python -m benchmark.run_benchmark test \
    --ace-path ./test-data/ace_2005_td_v7 \
    --api-url http://localhost:3001 \
    --output ./benchmark/results \
    --limit 50 \
    --detailed
```

## 评估指标

### 精确率 (Precision)
```
P = TP / (TP + FP)
```
系统提取的信息元中，正确的比例。

### 召回率 (Recall)
```
R = TP / (TP + FN)
```
Gold标准中的信息元，被系统正确提取的比例。

### F1分数
```
F1 = 2 * P * R / (P + R)
```

### 部分匹配F1 (Partial-F1)
对于部分匹配的情况（相似度 ≥ 0.5 但 < 0.95），按0.5计算：
```
P_partial = (TP + 0.5 * Partial) / (TP + Partial + FP)
R_partial = (TP + 0.5 * Partial) / (TP + Partial + FN)
```

### 匹配规则

**DESC 匹配**:
- `attribute` 完全匹配或相似度 ≥ 0.95 → 精确匹配
- `attribute` 相似度 ≥ 0.5 → 部分匹配

**SCEN 匹配**:
- `temporal` 和 `spatial` 综合相似度 ≥ 0.95 → 精确匹配
- 综合相似度 ≥ 0.5 → 部分匹配

**REL 匹配**:
- `relation_name` 和 `arg_refs` 引用的实体内容综合相似度评估

## API接口

如果你想用代码调用，可以这样：

```python
from benchmark.ace_parser import ACEParser
from benchmark.ace_to_infons import ACEToInfonsConverter, export_to_json
from benchmark.evaluator import InfonEvaluator, print_evaluation_report

# 1. 解析ACE数据
parser = ACEParser('./test-data/ace_2005_td_v7')
documents = parser.parse_all(annotation_level='adj', limit=100)

# 2. 转换为Infons
converter = ACEToInfonsConverter(use_subtype=True)
gold_samples = converter.convert_all(documents)

# 3. 获取你的预测结果
predictions_list = [your_model.extract(s.text) for s in gold_samples]

# 4. 评估
evaluator = InfonEvaluator()
result = evaluator.evaluate_batch(gold_samples, predictions_list)
print_evaluation_report(result, detailed=True)
```

## 注意事项

1. **数据集路径**: ACE 2005数据集应解压到 `test-data/ace_2005_td_v7/`

2. **语言支持**: 本工具支持ACE的三种语言（Arabic/Chinese/English），但中文和英文目录可能为空，需要检查

3. **SCEN生成规则**: 只有当ACE事件同时有Time和Place论元时，才会生成SCEN信息元

4. **文本编码**: ACE的SGM文件可能使用不同编码，解析器会自动尝试UTF-8和Latin-1

## 常见问题

**Q: 为什么SCEN数量很少？**

A: ACE数据集中，只有事件同时标注了时间和地点论元时才能转换为SCEN。大部分事件只有其中之一或都没有。

**Q: 评估结果F1很低怎么办？**

A: 可能原因：
- 模型提取的实体类别与ACE不匹配（如用"人名"而ACE用"Person"）
- 模型使用的语言与Gold不一致
- 可以调低 `partial_match_threshold` 来放宽匹配条件

**Q: 如何只评估DESC？**

A: 查看评估结果中的 `by_type.DESC` 部分，或筛选Gold数据只保留DESC类型。


tmux new -s infon-bench

tmux ls
tmux attach -t infon-bench / tmux a -t infon-bench

