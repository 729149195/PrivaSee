/**
 * infons.js - 信息元提取提示词模板（针对4B小参数模型优化）
 * 特点：精简上下文、原子化提取、格式清晰
 */

// ============================================================================
// 核心提示词模块（精简版）
// ============================================================================

export const CORE_DEFINITION = String.raw`You are an information extractor. Extract facts from user input as CSV lines.

**3 Types**:
- DESC: entity-attribute pairs (name, age, location, etc.)
- SCEN: time+place combinations (only when both appear together)
- REL: relationships between entities

**Critical Rules**:
- One fact per line
- Use EXACT words from input as attribute
- Output in same language as input
`;

export const OUTPUT_FORMAT = String.raw`**Output Format**: One CSV line per fact. Start output immediately with data lines.

Format by type:
- DESC: desc:r{round}_{n},DESC,entity,attribute,string,{confidence}
- SCEN: scen:r{round}_{n},SCEN,time,place,{confidence}
- REL: rel:r{round}_{n},REL,relation_name,iid1|iid2,{confidence}

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

**Example output for "Alice probably works at Google"**:
desc:r1_1,DESC,name,Alice,string,0.98
desc:r1_2,DESC,company,Google,string,0.75
rel:r1_3,REL,employment,desc:r1_1|desc:r1_2,0.70

**Example output for "他大概三十多岁"**:
desc:r1_1,DESC,年龄,三十多岁,string,0.65

Output ONLY the CSV lines, nothing else.
`;

export const TEXT_EXTRACTION = String.raw`**What to Extract**:
- Names of people, places, companies, products
- Numbers: age, money, phone, ID
- Specific actions or events

**What to Skip**:
- Common words: is, have, go, the, a
- Filler words: um, well, so
- Already extracted items

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

export const IMAGE_EXTRACTION = String.raw`**Image Extraction**:

Extract for each person/object:
- Physical traits (gender, age, clothing)
- Actions (standing, holding, etc.)
- Visible text (OCR)
- Location/brand indicators

SCEN: Use bbox for positions.
REL: Spatial relations (near, holding, wearing).

**Critical**: Do NOT create duplicate DESC lines with same entity+attribute.
- If multiple objects share a trait (e.g. 10 plates all from same region), extract it ONCE.
- Each unique entity-attribute pair appears only once; use count/note if needed.
- For multiple similar objects (e.g. license plates), extract each plate number as separate DESC, but shared attributes (region, color, type) only once.
- Target: 8-20 DESC, 2-8 REL. Quality over quantity.
`;

export const AUDIO_EXTRACTION = String.raw`**Audio Extraction**:

Transcribed speech → extract same as text.
Mark speakers if multiple.
`;

export const SELF_CHECKLIST = String.raw`**Checklist**:
- Extract from actual user input only
- One attribute per DESC line
- Use exact text from input
- SCEN rare (0-2 max)
- NO duplicate entity+attribute pairs
- Text: 5-12 lines total; Image: 8-20 lines total
`;

// ============================================================================
// Benchmark模式（精准NER提取）
// ============================================================================
export const BENCHMARK_EXTRACTION = String.raw`**Benchmark Mode - Named Entity Extraction**

Extract ONLY named entities:
✓ Person names (full names)
✓ Organizations
✓ Locations/Places
✓ Dates/Times
✓ Money/Percentages

Skip:
✗ Pronouns (he, she, it)
✗ Generic nouns (man, company)
✗ Verbs, adjectives

**Confidence**: 0.95+ exact match, 0.85-0.94 clear but variant possible, 0.70-0.84 inferred, 0.50-0.69 uncertain

Output: 5-15 DESC, 0-2 SCEN, 2-5 REL

Example:
desc:r1_1,DESC,Person,John Smith,string,0.98
desc:r1_2,DESC,Organization,Apple Inc,string,0.92
rel:r1_3,REL,employed_by,desc:r1_1|desc:r1_2,0.85
`;

// ============================================================================
// 兼容性导出（保持向后兼容）
// ============================================================================
export const ONTOLOGY = ''
export const OUTPUT_CONSTRAINTS = ''
export const EXAMPLES_SNIPPET = ''

// ============================================================================
// 构建系统提示词
// ============================================================================
export function buildSystemPrompt(options = {}) {
  const {
    modalities = ["text"],
    includeExamples = false,
    extraInstructions = "",
    currentRound = 1,
    existingInfons = [],
    benchmark = false
  } = options;

  // Benchmark模式使用精简提示词
  if (benchmark) {
    return `${BENCHMARK_EXTRACTION}\n\nRound: ${currentRound}\nIID format: {type}:r${currentRound}_{index}`;
  }

  const parts = [CORE_DEFINITION, OUTPUT_FORMAT];

  if (modalities.includes("text")) parts.push(TEXT_EXTRACTION);
  if (modalities.includes("image")) parts.push(IMAGE_EXTRACTION);
  if (modalities.includes("audio")) parts.push(AUDIO_EXTRACTION);

  parts.push(SELF_CHECKLIST);
  
  // 轮次上下文（精简）
  parts.push(`\n**Round ${currentRound}** - Use IID: {type}:r${currentRound}_{index}`);
  
  // 已有信息元引用（精简）
  if (Array.isArray(existingInfons) && existingInfons.length > 0) {
    const refs = existingInfons.slice(-10).map(inf => {
      const t = String(inf.infon_type || '').toUpperCase();
      if (t === 'DESC') return `${inf.iid}: ${inf.entity}=${inf.attribute}`;
      if (t === 'SCEN') return `${inf.iid}: ${inf.temporal}@${inf.spatial}`;
      return `${inf.iid}: ${inf.relation_name}`;
    }).join('\n');
    parts.push(`\nExisting infons (for REL refs):\n${refs}`);
  }
  
  if (extraInstructions) parts.push(extraInstructions);

  return parts.join("\n\n");
}

// ============================================================================
// JSON Schema（可选验证）
// ============================================================================
export const INFON_OUTPUT_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "infons": {"type": "array"}
  }
};

export const INFON_OUTPUT_SCHEMA_STR = JSON.stringify(INFON_OUTPUT_SCHEMA);

// ============================================================================
// 解析器（保持不变）
// ============================================================================

function unescapeValue(value) {
  if (typeof value !== 'string') return value
  return value.replace(/\\,/g, ',').replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
}

function splitArrayField(value) {
  if (!value || typeof value !== 'string') return []
  return value.split(/(?<!\\)\|/).map(p => p.trim()).filter(Boolean)
}

function parseCompactInfonLine(line, options = {}) {
  if (!line || !line.trim()) return null
  
  const values = []
  let currentValue = ''
  let escaped = false
  
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (escaped) { currentValue += ch; escaped = false; continue }
    if (ch === '\\') { currentValue += ch; escaped = true; continue }
    if (ch === ',') { values.push(currentValue); currentValue = ''; continue }
    currentValue += ch
  }
  if (currentValue || values.length > 0) values.push(currentValue)
  
  const iid = values[0] || ''
  const infon_type = values[1] || ''
  const recordTime = options.recordTime || new Date().toISOString()
  
  const infon = { iid, infon_type, record_time: recordTime }
  
  if (infon_type === 'DESC') {
    infon.entity = unescapeValue(values[2] || '')
    infon.attribute = unescapeValue(values[3] || '')
    infon.data_type = values[4] || 'string'
    const conf = parseFloat(values[5])
    // 默认置信度降低到0.80，鼓励模型主动给出准确的置信度
    infon.confidence = !isNaN(conf) ? Math.min(1.0, Math.max(0.0, conf)) : 0.80
  } else if (infon_type === 'SCEN') {
    infon.temporal = unescapeValue(values[2] || '')
    infon.spatial = unescapeValue(values[3] || '')
    const conf = parseFloat(values[4])
    // SCEN通常需要推断，默认置信度0.75
    infon.confidence = !isNaN(conf) ? Math.min(1.0, Math.max(0.0, conf)) : 0.75
  } else if (infon_type === 'REL') {
    infon.relation_name = unescapeValue(values[2] || '')
    infon.arg_refs = splitArrayField(values[3] || '')
    infon.arity = infon.arg_refs.length
    const conf = parseFloat(values[4])
    // REL依赖引用的准确性，默认置信度0.75
    infon.confidence = !isNaN(conf) ? Math.min(1.0, Math.max(0.0, conf)) : 0.75
  }
  
  return Object.keys(infon).length > 2 ? infon : null
}

export function parseCompactInfonsFormat(text) {
  if (!text || typeof text !== 'string') return null
  
  const headerMatch = text.match(/infons\[(\d+)\]:/)
  let dataText = text
  if (headerMatch) {
    dataText = text.slice(headerMatch.index + headerMatch[0].length)
  }
  
  const lines = dataText.split('\n')
  const infons = []
  const recordTime = new Date().toISOString()
  
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || !trimmed.match(/^(desc|scen|rel):/)) continue
    const infon = parseCompactInfonLine(trimmed, { recordTime })
    if (infon) infons.push(infon)
  }
  
  return { infons }
}

export function incrementalExtractInfonsCompact(streamText, parser) {
  const state = {
    foundHeader: parser?.foundHeader ?? false,
    scanPos: parser?.scanPos ?? 0,
    parsedLines: parser?.parsedLines ?? 0,
    buffer: parser?.buffer ?? '',
    infonCount: parser?.infonCount ?? 0,
    recordTime: parser?.recordTime ?? null,
    processedText: parser?.processedText ?? '',
    ...(parser || {})
  }
  
  const yielded = []
  const text = String(streamText || '')
  
  if (!state.recordTime) state.recordTime = new Date().toISOString()
  
  // 跳过可能的 header 或 markdown 代码块标记
  if (!state.foundHeader) {
    const headerMatch = text.match(/infons\[(\d+)\]:/)
    const codeBlockMatch = text.match(/```\w*\n?/)
    state.foundHeader = true
    if (headerMatch) {
      state.scanPos = headerMatch.index + headerMatch[0].length
    } else if (codeBlockMatch) {
      state.scanPos = codeBlockMatch.index + codeBlockMatch[0].length
    } else {
      state.scanPos = 0
    }
  }
  
  const dataText = text.slice(state.scanPos)
  // 移除末尾的 ``` 如果有的话
  const cleanedText = dataText.replace(/```\s*$/, '')
  const endsWithNewline = cleanedText.endsWith('\n')
  const lines = cleanedText.split('\n')
  
  // 关键修复：移除尾部换行符产生的空字符串元素
  // 否则 parsedLines 会被空元素递增，导致后续新行永远不会被解析到
  if (endsWithNewline && lines.length > 0 && lines[lines.length - 1] === '') {
    lines.pop()
  }
  
  for (let i = state.parsedLines; i < lines.length; i++) {
    // 对于最后一行，如果原文不以换行符结尾，说明该行可能不完整
    const isLastLine = (i === lines.length - 1)
    if (isLastLine && !endsWithNewline) {
      // 行太短则跳过等待更多数据；不递增 parsedLines 以便下次重新解析
      if (lines[i].length < 20) break
    }
    
    let trimmed = lines[i].trim()
    
    // 跳过空行、注释、markdown标记
    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('```') || trimmed.startsWith('**')) {
      state.parsedLines++
      continue
    }
    
    // 宽松匹配：支持各种前缀格式
    // 匹配 desc:, DESC:, - desc:, * desc:, 1. desc: 等
    const match = trimmed.match(/^[-*\d.)\s]*(desc|scen|rel):/i)
    if (!match) {
      // 对于未完成的最后一行，不递增 parsedLines，等待更多数据
      if (isLastLine && !endsWithNewline) break
      state.parsedLines++
      continue
    }
    
    // 提取实际的数据行，去掉前缀
    const prefixEnd = trimmed.toLowerCase().indexOf(match[1].toLowerCase() + ':')
    if (prefixEnd >= 0) {
      trimmed = trimmed.slice(prefixEnd)
    }
    
    const infon = parseCompactInfonLine(trimmed, { recordTime: state.recordTime })
    if (infon && infon.iid) {
      infon._objIndex = state.infonCount
      infon._isComplete = !isLastLine || endsWithNewline
      yielded.push(infon)
      state.infonCount++
    }
    state.parsedLines++
  }
  
  return { state, yielded }
}
