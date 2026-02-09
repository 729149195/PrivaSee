# Infon Cloud 英文提取长度问题修复方案

## 问题分析

### 当前问题

在 `frontend/src/templates/infons.js` 中，当前的 CSV 输出格式对于**长英文文本**存在以下问题：

1. **换行符导致CSV解析错误**
   - 问题: `attribute` 字段包含真实换行符时，CSV解析器会将其识别为新行
   - 示例: `"This is a\nlong text"` → CSV解析器认为这是两行

2. **长度限制缺失**
   - 问题: 未对提取的英文文本长度做限制
   - 后果: 可能超过模型输出窗口 (4K tokens)，导致截断或生成失败

3. **逗号冲突**
   - 问题: `attribute` 中的逗号会被CSV解析器识别为字段分隔符
   - 示例: `"Apple, Google"` → 被解析为3个字段

### 影响范围

- **文本提取**: 长段落、多行文本
- **图像OCR**: 多行文档、合同、证件
- **音频转录**: 长对话、演讲

---

## 解决方案

### 方案对比

| 方案 | 优点 | 缺点 | 实施难度 | 推荐度 |
|------|------|------|---------|--------|
| **方案1: 转义特殊字符** | 兼容现有CSV格式 | 需前后端同步更新 | ⭐⭐ | ⭐⭐⭐ |
| **方案2: 长度截断** | 简单直接 | 可能丢失信息 | ⭐ | ⭐⭐⭐⭐ |
| **方案3: JSON Lines** | 格式灵活 | 改动大，需重构 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **方案4: 分块提取** | 保留完整信息 | 增加信息元数量 | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 推荐方案: 混合方案 (方案1 + 方案2)

**核心策略**:
1. **长度截断**: attribute字段限制200字符
2. **转义换行符**: `\n` → `\\n`
3. **替换逗号**: `,` → `;` (或使用引号包裹)
4. **提取摘要**: 对于长文本，提取关键短语而非全文

---

## 实施步骤

### Step 1: 修改提示词模板

**文件**: `frontend/src/templates/infons.js`

#### 修改 OUTPUT_FORMAT

```javascript
export const OUTPUT_FORMAT = String.raw`**Output Format**: One CSV line per fact. Start output immediately with data lines.

Format by type:
- DESC: desc:r{round}_{n},DESC,entity,attribute,string,{confidence}
- SCEN: scen:r{round}_{n},SCEN,time,place,{confidence}
- REL: rel:r{round}_{n},REL,relation_name,iid1|iid2,{confidence}

**CRITICAL RULES for Attribute Field**:
1. **Length Limit**: Max 200 characters
   - For longer text, extract key phrase or summary ONLY
   - Example: "A 500-word contract..." → "contract terms agreed on 2024-03-15"
   
2. **No Real Newlines**: Replace newlines with \\n (escaped)
   - Correct: "Line 1\\nLine 2" (backslash + n)
   - Wrong: "Line 1
             Line 2" (actual newline)
   
3. **No Commas**: Replace commas with semicolons OR use quotes
   - Option A: "Apple, Google" → "Apple; Google"
   - Option B: "Apple, Google" → "\"Apple, Google\""
   
4. **Special Characters**: Escape quotes and backslashes
   - Quote: " → \\"
   - Backslash: \\ → \\\\

**Confidence Scoring Guide** (0.0-1.0):
- 0.95-1.0: Explicit, exact values (names, IDs, quoted text)
- 0.85-0.94: Clear but could have variants (company names, titles)
- 0.70-0.84: Inferred/approximate (age ranges, rough times, implied info)
- 0.50-0.69: Uncertain/ambiguous (guessed context, unclear references)
- <0.50: Very uncertain (avoid extracting)

**Example output for "我叫张伟，今年28岁"**:
desc:r1_1,DESC,姓名,张伟,string,0.98
desc:r1_2,DESC,年龄,28,number,0.95
rel:r1_3,REL,个人信息,desc:r1_1|desc:r1_2,0.90

**Example output for "Alice probably works at Google, Microsoft"**:
desc:r1_1,DESC,name,Alice,string,0.98
desc:r1_2,DESC,companies,Google; Microsoft,string,0.75
rel:r1_3,REL,employment,desc:r1_1|desc:r1_2,0.70

**Example output for long text**:
desc:r1_1,DESC,document_type,employment contract,string,0.95
desc:r1_2,DESC,contract_date,2024-03-15,string,0.98
desc:r1_3,DESC,key_terms,30-day payment; confidentiality clause,string,0.85

Output ONLY the CSV lines, nothing else.
`;
```

#### 修改 TEXT_EXTRACTION

```javascript
export const TEXT_EXTRACTION = String.raw`**What to Extract**:
- Names of people, places, companies, products
- Numbers: age, money, phone, ID
- Specific actions or events

**What to Skip**:
- Common words: is, have, go, the, a
- Filler words: um, well, so
- Already extracted items

**Attribute Length Rules**:
- Keep attribute < 200 chars
- For long English text:
  * Extract key phrases, NOT full paragraphs
  * Use summaries for contracts/documents
  * Example: "This is a very long contract with many clauses..." 
    → Extract as: "contract with payment and confidentiality terms"
  * Split into multiple DESC if needed:
    - desc:r1_1: document_type, contract
    - desc:r1_2: key_terms, payment within 30 days
    - desc:r1_3: key_terms, confidentiality clause

**Confidence Indicators**:
- High (0.90+): "is", "叫", exact quotes, specific numbers
- Medium (0.75-0.89): "works at", "住在", clear context
- Lower (0.60-0.74): "probably", "maybe", "大概", "可能", implied info
- Uncertain (<0.60): Don't extract

**Limits**:
- SCEN: 0-2 only (need both time AND place)
- REL: 2-5 relationships
- Total: 5-12 facts

**Remember**: Output ONLY CSV lines starting with desc:/scen:/rel:
`;
```

#### 修改 IMAGE_EXTRACTION

```javascript
export const IMAGE_EXTRACTION = String.raw`**Image Extraction**:

Extract for each person/object:
- Physical traits (gender, age, clothing)
- Actions (standing, holding, etc.)
- Visible text (OCR) - **IMPORTANT**: Apply 200-char limit, escape newlines
- Location/brand indicators

**For OCR Text**:
- If text is long (>200 chars), extract document type + key info
- Example: Long contract → "contract dated 2024-03-15; parties: Company A; Company B"
- Escape newlines: "Line 1\\nLine 2" (NOT real newline)

SCEN: Use bbox for positions.
REL: Spatial relations (near, holding, wearing).

**Critical**: Do NOT create duplicate DESC lines with same entity+attribute.
- If multiple objects share a trait (e.g. 10 plates all from same region), extract it ONCE.
- Each unique entity-attribute pair appears only once; use count/note if needed.
- For multiple similar objects (e.g. license plates), extract each plate number as separate DESC, but shared attributes (region, color, type) only once.
- Target: 8-20 DESC, 2-8 REL. Quality over quantity.
`;
```

### Step 2: 修改解析器

**文件**: `frontend/src/utils/infonParser.js` (如果不存在则创建)

```javascript
/**
 * 解析信息元CSV行，处理转义字符和长度截断
 */
export function parseInfonCSVLine(line) {
  // 移除首尾空白
  line = line.trim();
  
  // 跳过空行和注释
  if (!line || line.startsWith('#')) {
    return null;
  }
  
  // 简单CSV解析 (不支持引号内的逗号)
  const parts = line.split(',').map(p => p.trim());
  
  if (parts.length < 6) {
    console.warn('[InfonParser] Invalid CSV line (too few fields):', line);
    return null;
  }
  
  const [iid, type, field1, field2, valueType, confidence] = parts;
  
  // 反转义 attribute 字段
  const unescapeAttribute = (attr) => {
    if (!attr) return '';
    
    // 1. 反转义换行符: \\n → \n
    attr = attr.replace(/\\n/g, '\n');
    
    // 2. 反转义引号: \\" → "
    attr = attr.replace(/\\"/g, '"');
    
    // 3. 反转义反斜杠: \\\\ → \\
    attr = attr.replace(/\\\\/g, '\\');
    
    // 4. 恢复逗号 (如果使用分号替换方案)
    // attr = attr.replace(/;/g, ',');
    
    // 5. 截断过长文本 (保护性措施)
    if (attr.length > 250) {
      attr = attr.substring(0, 247) + '...';
    }
    
    return attr;
  };
  
  // 构建信息元对象
  const infon = {
    iid,
    infon_type: type.toUpperCase(),
  };
  
  if (type.toUpperCase() === 'DESC') {
    infon.entity = field1;
    infon.attribute = unescapeAttribute(field2);
    infon.value_type = valueType;
  } else if (type.toUpperCase() === 'SCEN') {
    infon.temporal = field1;
    infon.spatial = unescapeAttribute(field2);
  } else if (type.toUpperCase() === 'REL') {
    infon.relation_name = field1;
    infon.arg_refs = field2.split('|').map(ref => ref.trim());
  }
  
  infon.confidence = parseFloat(confidence) || 0.5;
  
  return infon;
}

/**
 * 批量解析多行CSV
 */
export function parseInfonCSV(csvText) {
  const lines = csvText.split('\n');
  const infons = [];
  
  for (const line of lines) {
    const infon = parseInfonCSVLine(line);
    if (infon) {
      infons.push(infon);
    }
  }
  
  return infons;
}

/**
 * 验证信息元格式
 */
export function validateInfon(infon) {
  const errors = [];
  
  // 检查必需字段
  if (!infon.iid) {
    errors.push('Missing iid');
  }
  if (!infon.infon_type) {
    errors.push('Missing infon_type');
  }
  
  // 检查类型特定字段
  if (infon.infon_type === 'DESC') {
    if (!infon.entity) errors.push('DESC missing entity');
    if (!infon.attribute) errors.push('DESC missing attribute');
    
    // 检查长度
    if (infon.attribute && infon.attribute.length > 250) {
      errors.push(`DESC attribute too long: ${infon.attribute.length} chars`);
    }
  } else if (infon.infon_type === 'SCEN') {
    if (!infon.temporal) errors.push('SCEN missing temporal');
    if (!infon.spatial) errors.push('SCEN missing spatial');
  } else if (infon.infon_type === 'REL') {
    if (!infon.relation_name) errors.push('REL missing relation_name');
    if (!infon.arg_refs || infon.arg_refs.length < 2) {
      errors.push('REL needs at least 2 arg_refs');
    }
  }
  
  // 检查置信度
  if (infon.confidence < 0 || infon.confidence > 1) {
    errors.push(`Invalid confidence: ${infon.confidence}`);
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}
```

### Step 3: 更新前端调用代码

**文件**: `frontend/src/store/slices/infonSlice.js`

在解析LLM返回的CSV时，使用新的解析器：

```javascript
import { parseInfonCSV, validateInfon } from '../../utils/infonParser';

// 在 _parseInfonChunk 函数中
_parseInfonChunk(rawText) {
  try {
    // 使用新的解析器
    const infons = parseInfonCSV(rawText);
    
    // 验证每个信息元
    const validInfons = [];
    for (const infon of infons) {
      const validation = validateInfon(infon);
      if (validation.valid) {
        validInfons.push(infon);
      } else {
        console.warn('[InfonSlice] Invalid infon:', infon, validation.errors);
      }
    }
    
    return validInfons;
  } catch (error) {
    console.error('[InfonSlice] Parse error:', error);
    return [];
  }
}
```

---

## 测试用例

### 测试1: 长英文文本

**输入**:
```
This is a very long employment contract between Company A and Employee B. 
The contract states that the employee will work for 40 hours per week, 
with a salary of $80,000 per year. The contract includes confidentiality 
clauses and non-compete agreements for 2 years after termination.
```

**期望输出**:
```csv
desc:r1_1,DESC,document_type,employment contract,string,0.95
desc:r1_2,DESC,parties,Company A and Employee B,string,0.98
desc:r1_3,DESC,work_hours,40 hours per week,string,0.95
desc:r1_4,DESC,salary,$80000 per year,string,0.98
desc:r1_5,DESC,key_terms,confidentiality; non-compete for 2 years,string,0.90
rel:r1_6,REL,employment_contract,desc:r1_1|desc:r1_2,0.95
```

### 测试2: 多行文本 (换行符)

**输入**:
```
Address:
123 Main Street
Apt 4B
New York, NY 10001
```

**期望输出**:
```csv
desc:r1_1,DESC,address,123 Main Street\\nApt 4B\\nNew York NY 10001,string,0.98
desc:r1_2,DESC,city,New York,string,0.98
desc:r1_3,DESC,state,NY,string,0.98
desc:r1_4,DESC,zip_code,10001,string,0.98
```

**解析后**:
```javascript
{
  iid: 'desc:r1_1',
  infon_type: 'DESC',
  entity: 'address',
  attribute: '123 Main Street\nApt 4B\nNew York NY 10001',  // 真实换行符
  confidence: 0.98
}
```

### 测试3: 包含逗号的文本

**输入**:
```
Alice works at Google, Microsoft, and Apple.
```

**期望输出**:
```csv
desc:r1_1,DESC,name,Alice,string,0.98
desc:r1_2,DESC,companies,Google; Microsoft; Apple,string,0.85
rel:r1_3,REL,employment,desc:r1_1|desc:r1_2,0.80
```

---

## 回归测试

### 确保不影响现有功能

**测试用例集**:

1. **中文提取** (无变化)
   ```
   输入: "我叫张伟，今年28岁"
   期望: desc:r1_1,DESC,姓名,张伟,string,0.98
   ```

2. **短英文提取** (无变化)
   ```
   输入: "Alice is 25 years old"
   期望: desc:r1_1,DESC,name,Alice,string,0.98
   ```

3. **图像OCR** (应用新规则)
   ```
   输入: 身份证图片 (多行文本)
   期望: 转义换行符，限制长度
   ```

4. **音频转录** (应用新规则)
   ```
   输入: 长对话转录
   期望: 提取关键短语，不提取全文
   ```

---

## 部署检查清单

- [ ] 修改 `frontend/src/templates/infons.js` 的 OUTPUT_FORMAT
- [ ] 修改 `frontend/src/templates/infons.js` 的 TEXT_EXTRACTION
- [ ] 修改 `frontend/src/templates/infons.js` 的 IMAGE_EXTRACTION
- [ ] 创建 `frontend/src/utils/infonParser.js`
- [ ] 更新 `frontend/src/store/slices/infonSlice.js` 的解析逻辑
- [ ] 运行单元测试 (如果有)
- [ ] 手动测试: 长英文文本提取
- [ ] 手动测试: 多行文本提取
- [ ] 手动测试: 包含逗号的文本
- [ ] 回归测试: 中文提取
- [ ] 回归测试: 图像OCR
- [ ] 回归测试: 音频转录
- [ ] 更新文档

---

## 预期效果

### 修复前
```
❌ 问题: 长英文段落被完整提取，导致CSV解析错误
desc:r1_1,DESC,contract_text,This is a very long contract with many clauses.
It includes payment terms, confidentiality agreements, and more...,string,0.90
```

### 修复后
```
✅ 正确: 提取关键短语，转义换行符
desc:r1_1,DESC,document_type,employment contract,string,0.95
desc:r1_2,DESC,key_terms,payment terms; confidentiality; non-compete,string,0.90
```

---

## 后续优化方向

### 方向1: 智能分块
对于超长文本，自动分块提取：
```javascript
// 伪代码
if (text.length > 500) {
  const chunks = smartChunk(text, maxLength=200);
  for (const chunk of chunks) {
    extractInfon(chunk);
  }
}
```

### 方向2: 文档摘要
对于文档类输入，先生成摘要再提取：
```javascript
// 伪代码
if (isDocument(text)) {
  const summary = await generateSummary(text);
  extractInfon(summary);
}
```

### 方向3: JSON Lines格式
长期方案：迁移到JSON Lines格式，彻底避免CSV的转义问题：
```jsonl
{"iid":"desc:r1_1","type":"DESC","entity":"name","attribute":"Alice","confidence":0.98}
{"iid":"desc:r1_2","type":"DESC","entity":"company","attribute":"Google, Microsoft","confidence":0.85}
```

---

## 相关Issue

- **Issue #1**: Infon Cloud英文提取长度问题
- **Issue #2**: CSV解析错误 (换行符)
- **Issue #3**: 逗号导致字段错位

---

## 参考资料

- CSV RFC 4180: https://tools.ietf.org/html/rfc4180
- 转义字符规范: https://en.wikipedia.org/wiki/Escape_character
- JSON Lines格式: https://jsonlines.org/

---

**文档版本**: v1.0  
**创建日期**: 2025-02-09  
**作者**: PrivaSee Team  
**状态**: 待实施

