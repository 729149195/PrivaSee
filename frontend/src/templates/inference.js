// Privacy Inference Prompt Template
export const PRIVACY_INFERENCE_PROMPT = `
You are a privacy risk analysis expert specializing in deep inference and contextual analysis. Based on exposed Information Elements, you must infer ALL possible privacy exposures, including explicit, implicit, and contextually derived information.

## Task
Conduct comprehensive privacy risk analysis by:
1. Identifying DIRECT privacy exposures from explicit information
2. Inferring IMPLICIT privacy through deep reasoning (e.g., "gluten-free menu" → potential celiac disease/health condition)
3. Discovering CONTEXTUAL privacy leaks through cross-referencing multiple information elements
4. Analyzing behavior patterns that reveal sensitive attributes
5. Mapping ALL findings to the MOST SPECIFIC (leaf-level) legal clauses

**IMPORTANT**: If the legal structure below indicates "CUSTOM PRIVACY ANALYSIS MODE", you MUST ONLY analyze and report privacy risks that match the explicitly listed selected items. DO NOT report risks for privacy categories not in the user's selection, even if you can infer them.

## Input
1. **Exposed Information Elements List**:
{{INFONS}}

2. **Legal Clause Hierarchical Structure** (Currently Selected Law):
{{LAW_TREE}}

## Deep Inference Rules

### 1. Direct Exposure
- Explicitly stated personal information (name, ID, contact, etc.)
- Directly mentioned attributes (age, gender, occupation, etc.)

### 2. Implicit Inference (CRITICAL)
- **Health Conditions**: Dietary restrictions → allergies/diseases (e.g., gluten-free → celiac disease, halal → religious belief)
- **Beliefs & Values**: Food preferences, lifestyle choices → religious/philosophical beliefs
- **Socioeconomic Status**: Purchase behaviors, location patterns → income level, social class
- **Psychological State**: Language patterns, content preferences → mental state, personality traits
- **Identity Attributes**: Name patterns, language use → ethnicity, national origin

### 3. Cross-Context Correlation
- Temporal + Spatial patterns → daily routines, home/work locations
- Relationship networks (REL) → social circles, family structure
- Multiple behavioral data points → comprehensive profiling

### 4. Sensitive Category Recognition
Always check if inferred information falls into sensitive categories:
- Health/Medical conditions
- Racial/Ethnic origin
- Religious/Philosophical beliefs
- Sexual orientation
- Political opinions
- Biometric data
- Genetic data
- Trade union membership
- Criminal history

## Risk Level Criteria
**Risk level is determined by the CERTAINTY/CONFIDENCE of privacy inference from input data:**
- **HIGH**: Input data (combined with context) can DEFINITIVELY or with VERY HIGH CONFIDENCE reveal specific privacy information. The inference chain is direct and unambiguous. Examples: explicit personal identifiers, precise location combined with time patterns that clearly reveal home/work addresses, dietary restrictions with medical records that definitively indicate health conditions.
- **MEDIUM**: Input data allows privacy inference with MODERATE CONFIDENCE. Some reasoning is required, but the inference is reasonably supported. Examples: behavioral patterns suggesting lifestyle preferences, partial identifiers that narrow down to a small group, indirect health indicators that suggest but don't confirm conditions.
- **LOW**: Input data provides WEAK or VAGUE clues about privacy. High uncertainty in inference, multiple interpretations possible, or only very general/public information can be derived. Examples: generic demographic trends, broad location areas, ambiguous preferences without clear implications.

## Output Format
**CRITICAL - Streaming Optimization & Compact Format**: Output in compact format for faster token generation:

**Format Syntax** (NO header line, direct data output, EXACTLY 5 FIELDS):
\`\`\`
field1,field2,field3,field4,field5
field1,field2,field3,field4,field5
\`\`\`

**CRITICAL RULES**:
- NO header line
- EXACTLY 5 comma-separated fields per line (law_node_name, risk_level, privacy_exposure, inference_chain, used_infons)
- Each risk on a SEPARATE LINE (use line breaks between risks)
- NO extra fields, NO missing fields
- Field 5 (used_infons) MUST use format "TYPE:VALUE" (e.g., DESC:Klook, SCEN:下周@东京)

**Field Order (MUST follow exactly, 5 fields per line)**:
1. law_node_name - Most specific (leaf-level) node name (EXACT COPY from law tree, NO translation, NO abbreviation)
2. risk_level - HIGH | MEDIUM | LOW
3. privacy_exposure - Specific privacy information exposed
4. inference_chain - Step-by-step reasoning
5. used_infons - Information elements supporting this risk in format "TYPE:VALUE" (separated by |)
   - **DESC format**: DESC:attribute_value (ONLY the attribute value, NOT "entity:attribute")
     * Example: If infon shows "DESC: 平台: Klook" → use "DESC:Klook" (NOT "DESC:平台:Klook")
     * Example: If infon shows "DESC: 地点: 台北" → use "DESC:台北" (NOT "DESC:地点:台北")
   - **SCEN format**: SCEN:temporal@spatial (combine temporal and spatial)
     * Example: If infon shows "SCEN: 下周 @ 东京" → use "SCEN:下周@东京"
   - **REL format**: REL:relation_name
     * Example: If infon shows "REL: 住宿预订" → use "REL:住宿预订"
   - Extract from INPUT information elements, NOT from your inference

**Escaping Rules (CRITICAL)**:
- Commas in text → \\, (backslash-comma)
- Newlines WITHIN fields → \\n (backslash-n) - DO NOT use actual line breaks within a field
- Backslashes → \\\\ (double backslash)
- Array elements → Use | separator (e.g., keyword1|keyword2|keyword3)
- BETWEEN risk entries → Use ACTUAL LINE BREAK (press Enter after each risk)

**Example Output** (NO header, EXACTLY 5 fields per line, one risk per line):
\`\`\`
医疗健康,HIGH,用户可能患有乳糜泻,用户搜索无麸质餐厅菜单\\,表明有健康饮食限制\\,推断为麸质不耐受或乳糜泻,DESC:gluten-free|DESC:restaurant menu
姓名,HIGH,暴露真实姓名,文本中明确提到姓名信息\\,直接识别个人身份,DESC:王小明
位置信息,MEDIUM,暴露大致位置,搜索餐厅行为暴露地理位置偏好,DESC:restaurant|DESC:location
\`\`\`

**Field count verification**:
- Line 1: 医疗健康 (1), HIGH (2), 用户可能患有乳糜泻 (3), 用户搜索... (4), DESC:gluten-free|DESC:restaurant menu (5) ✓
- Line 2: 姓名 (1), HIGH (2), 暴露真实姓名 (3), 文本中明确... (4), DESC:王小明 (5) ✓
- Line 3: 位置信息 (1), MEDIUM (2), 暴露大致位置 (3), 搜索餐厅... (4), DESC:restaurant|DESC:location (5) ✓

**used_infons field format (Field 5) - CRITICAL**:
- MUST follow "TYPE:VALUE" pattern
- **DESC type**: Extract ONLY the attribute value (NOT "entity:attribute")
  * If infon shows "DESC: 平台: Klook" → write "DESC:Klook"
  * If infon shows "DESC: 地点: 台北" → write "DESC:台北"
  * If infon shows "DESC: 姓名: 王小明" → write "DESC:王小明"
- **SCEN type**: Combine temporal@spatial
  * If infon shows "SCEN: 下周 @ 东京" → write "SCEN:下周@东京"
- **REL type**: Use relation name
  * If infon shows "REL: 住宿预订" → write "REL:住宿预订"
- Multiple items: Separate with | (e.g., DESC:Klook|DESC:台北|SCEN:下周@东京)
- Extract from INPUT information elements in the list above, NOT from your inference

## Critical Requirements
1. **Output ONLY Compact Format** - NO header line, NO JSON, NO markdown code blocks, NO explanatory text, NO statements like "cannot infer" or "insufficient data". Start directly with data lines. DO NOT output any text other than the risk data lines.
2. **SKIP Uncertain Risks** - CRITICAL: If you CANNOT confidently infer a privacy risk from the input data, DO NOT output anything for that risk. DO NOT output statements like "无法推断", "不确定", "insufficient information", etc. Simply skip that risk and move to the next one.
3. **ONE RISK PER LINE** - CRITICAL: Each risk entry MUST be on its own separate line. Press Enter/newline after completing each risk. DO NOT output all risks in one continuous line.
4. **Field Order Matters** - MUST output fields in the exact order: law_node_name, risk_level, privacy_exposure, inference_chain, used_infons
5. **Map to LEAF NODES ONLY** - CRITICAL: Always identify the MOST SPECIFIC legal clause at the DEEPEST level in hierarchy (leaf nodes). NEVER map to intermediate/parent nodes. The law_node_name MUST be the final level clause name that appears in the law tree structure.
6. **EXACT Name Matching** - The law_node_name MUST be copied EXACTLY from the law tree structure, character by character. DO NOT paraphrase, summarize, or modify the node name. Look for the [LEAF NODE - USE THIS] markers in the law tree and copy the name EXACTLY.
7. **CUSTOM MODE RESTRICTION** - If you see "CUSTOM PRIVACY ANALYSIS MODE" in the law tree section, the law_node_name MUST be one of the explicitly listed selected items (marked with ✓). You CANNOT use any other privacy category names, even if you can infer them. If a privacy risk doesn't match any selected item, skip it entirely.
8. **Deep Inference** - Don't just report explicit data; infer health conditions, beliefs, status from behavior/preferences
9. **Comprehensive Coverage** - Analyze ALL possible privacy angles: direct exposure + implicit inference + contextual correlation (BUT respect Custom Mode restrictions)
10. **Clear Attribution** - Every risk must trace back to specific information elements with logical reasoning
11. **used_infons Extraction** - CRITICAL FORMAT: "TYPE:VALUE" separated by |
   - **For DESC infons**: Extract ONLY the attribute value (NOT "entity:attribute")
     * Infon shows "DESC: 平台: Klook" → Output "DESC:Klook" (NOT "DESC:平台:Klook")
     * Infon shows "DESC: 地点: 台北" → Output "DESC:台北" (NOT "DESC:地点:台北")
     * Infon shows "DESC: 姓名: 王小明" → Output "DESC:王小明"
   - **For SCEN infons**: Combine temporal@spatial
     * Infon shows "SCEN: 下周 @ 东京" → Output "SCEN:下周@东京"
     * Infon shows "SCEN: 今年 @ 北京" → Output "SCEN:今年@北京"
   - **For REL infons**: Use relation name directly
     * Infon shows "REL: 住宿预订" → Output "REL:住宿预订"
     * Infon shows "REL: 旅行计划" → Output "REL:旅行计划"
   - Multiple items example: DESC:Klook|DESC:台北|DESC:订住宿|SCEN:下周@东京|REL:行程决策
   - DO NOT extract abstract concepts or your own inferences - ONLY concrete information elements from the input
12. **Evaluate Inference Certainty** - Risk level depends on inference confidence: definitive/unambiguous data → HIGH risk; moderate confidence inference → MEDIUM risk; vague/uncertain inference → LOW risk. Consider both data specificity and contextual strength.
13. **Sort Properly** - HIGH risks first
14. **Verify Before Output** - Before outputting each risk, find the [LEAF NODE - USE THIS] entry in the provided law tree and copy its exact name. If you cannot find an exact leaf node match, choose the closest leaf node from the tree.
15. **LANGUAGE CONSISTENCY** - Write privacy_exposure and inference_chain in the SAME language as the input information elements. If the input data is in Chinese, write your analysis in Chinese. If in English, write in English. Match the language of the user's data.
16. **Escape Special Characters** - Remember to escape commas (\\,), newlines (\\n), and backslashes (\\\\\\) in text fields.
17. **NO UNCERTAIN OUTPUT** - CRITICAL: Do NOT output any risk if you are uncertain or cannot confidently infer it. Do NOT explain why you cannot infer something. Simply output nothing for uncertain cases.

## Example Output Format
When you find a privacy risk, you MUST:
1. Find the DEEPEST leaf node in the law tree that matches this privacy exposure
2. Set law_node_name to ONLY the leaf node name (copied EXACTLY from the law tree)
3. DO NOT use intermediate node names - always use the most specific (leaf) node
4. Output in compact format with proper escaping (NO header line)

Example 1:
If input contains information elements:
- [desc:r1_1] DESC: 饮食限制: gluten-free
- [desc:r1_2] DESC: 搜索内容: restaurant menu
Output (EXACTLY 5 fields per line):
\`\`\`
医疗健康,MEDIUM,User likely has celiac disease or gluten intolerance,Dietary restriction (gluten-free) provides moderate evidence of health condition\\, but not definitive without medical context,DESC:gluten-free|DESC:restaurant menu
饮食偏好,LOW,User's dietary preference,General preference\\, multiple possible interpretations,DESC:gluten-free
位置信息,LOW,Location search reveals geographic pattern,Single search\\, insufficient data for confident inference about regular patterns,DESC:restaurant|DESC:menu
\`\`\`
(3 lines, each with 5 fields: law_node_name, risk_level, privacy_exposure, inference_chain, used_infons)

Example 2:
If input contains information elements:
- [desc:r2_1] DESC: 姓名: 王小明  ← Extract attribute "王小明"
- [desc:r2_2] DESC: 年龄: 27      ← Extract attribute "27"
- [scen:r2_3] SCEN: 今年 @ (empty) ← Extract "今年"
- [desc:r2_4] DESC: 地点: 北京市   ← Extract attribute "北京市"
- [desc:r2_5] DESC: 地点: 海淀区   ← Extract attribute "海淀区"
Output (EXACTLY 5 fields per line):
\`\`\`
姓名,HIGH,暴露真实姓名,文本中明确提到姓名\\,可直接识别个人身份,DESC:王小明
年龄,HIGH,暴露精确年龄,明确的年龄信息\\,属于人口统计学属性,DESC:27|SCEN:今年
住址,HIGH,暴露居住地址,精确的住址信息\\,结合姓名可唯一识别个人,DESC:北京市|DESC:海淀区
\`\`\`
(3 lines, each with 5 fields: law_node_name, risk_level, privacy_exposure, inference_chain, used_infons)
Note: DESC extracts ONLY the attribute value (王小明, 27, 北京市, 海淀区), NOT "姓名:王小明" or "地点:北京市"

**Custom Mode Example**:
If law tree shows "CUSTOM PRIVACY ANALYSIS MODE" with selected items: ["Home Address", "Location/GPS", "Health Data"]
And input contains information elements:
- [desc:r3_1] DESC: 活动: Friday prayers
- [desc:r3_2] DESC: 地点: mosque
- [desc:r3_3] DESC: 搜索: halal restaurants
Output only selected categories (EXACTLY 5 fields per line):
\`\`\`
Health Data,MEDIUM,Dietary preference suggests health/dietary need,Halal food search suggests dietary restriction\\, moderate confidence in health-related inference,DESC:halal|DESC:restaurants
Location/GPS,MEDIUM,Location search pattern,Temporal pattern with location\\, insufficient data for high-confidence inference about home/work,DESC:Friday prayers|DESC:mosque
\`\`\`
(2 lines, each with 5 fields: law_node_name, risk_level, privacy_exposure, inference_chain, used_infons)

- ✅ REPORT: Categories in selected list (Health Data, Location/GPS)
- ❌ DO NOT REPORT: Religious belief inference (Islam) → NOT in selected list, must skip even though it's inferable
- ❌ DO NOT REPORT: Any privacy category not explicitly listed in the selected items

REMEMBER: Output ONLY valid risk data lines in the compact format. DO NOT output any statements like "无法推断", "不确定", "cannot determine", "insufficient information", etc. If you cannot confidently infer a privacy risk, simply don't output anything for that risk. Only output actual risk data lines following the 5-field format.

NOW analyze the input and output the complete compact format.
`

// Extract information elements summary (for filling template)
export function extractInfonsSummary(infons) {
  if (!Array.isArray(infons) || infons.length === 0) {
    return 'No information elements available'
  }
  
  return infons.map((infon, idx) => {
    const type = String(infon.infon_type || '').toUpperCase()
    const iid = infon.iid || `infon_${idx}`
    
    let keyword = ''
    if (type === 'DESC') {
      const entity = infon.entity ?? ''
      const attribute = infon.attribute ?? ''
      keyword = entity && attribute ? `${entity}: ${attribute}` : (entity || attribute || 'Description')
    } else if (type === 'SCEN') {
      const temporal = infon.temporal ?? ''
      const spatial = infon.spatial ?? ''
      keyword = temporal && spatial ? `${temporal} @ ${spatial}` : (temporal || spatial || 'Scenario')
    } else if (type === 'REL') {
      keyword = infon.relation_name ?? 'Relation'
    } else if (type === 'SIT') {
      keyword = infon.description ?? 'Situation'
    }
    
    return `- [${iid}] ${type}: ${keyword} (confidence: ${infon.confidence ?? 0.7})`
  }).join('\n')
}

// Extract law tree summary (for filling template) - 优化版：只提取叶子节点
export function extractLawTreeSummary(lawData) {
  if (!lawData) {
    return 'No legal structure available'
  }
  
  // Custom模式：只使用用户选中的隐私项
  if (lawData.isCustom) {
    const customItems = lawData.customItems || []
    if (customItems.length === 0) {
      return 'No items selected'
    }
    
    // 极简模式：只列出选中的项
    const itemNames = customItems.map(item => item.label || item.id).filter(Boolean)
    return `CUSTOM MODE - Selected items: ${itemNames.join(', ')}`
  }
  
  // 标准法律树模式：只提取叶子节点名称
  if (!lawData.name) {
    return 'No legal structure available'
  }
  
  const leafNodes = []
  
  // 递归收集所有叶子节点
  function collectLeafNodes(node) {
    const isLeaf = !Array.isArray(node.children) || node.children.length === 0
    
    if (isLeaf) {
      leafNodes.push(node.name)
    } else if (Array.isArray(node.children)) {
      node.children.forEach(child => collectLeafNodes(child))
    }
  }
  
  collectLeafNodes(lawData)
  
  // 返回简洁的叶子节点列表
  return `Privacy Categories (${leafNodes.length} items):\n${leafNodes.join(', ')}`
}

// Fill prompt template
export function fillPromptTemplate(infons, lawData, directInput = null, historicalKeywords = []) {
  const lawTreeSummary = extractLawTreeSummary(lawData)
  
  // 直接推断模式：使用用户原始输入
  if (directInput !== null) {
    // 分析输入结构：区分历史消息和当前输入
    const inputLines = directInput.split('\n\n').filter(Boolean)
    const hasMultipleMessages = inputLines.length > 1
    
    // 构建历史关键词上下文
    let keywordsContext = ''
    if (historicalKeywords && historicalKeywords.length > 0) {
      keywordsContext = `\n\nKNOWN PRIVACY KEYWORDS:
${historicalKeywords.join(', ')}

IMPORTANT: The above keywords are from previous messages. When analyzing the complete conversation below, you MUST:
1. Identify privacy risks by considering ALL messages together (cross-message correlation)
2. Extract NEW keywords from ALL messages (not just the last one)
3. Recognize that privacy information may be split across multiple messages
4. Example: Message 1 has "name", Message 2 has "address" → Together they reveal identity
`
    }
    
    // 构建用户输入说明
    let inputDescription = hasMultipleMessages 
      ? `COMPLETE USER CONVERSATION (${inputLines.length} messages, analyze ALL together for comprehensive privacy assessment):`
      : 'USER INPUT (single message):'
    
    const simplePrompt = `You are a privacy risk analyzer. Analyze the COMPLETE user conversation below and identify ALL privacy risks by considering cross-message correlations.

${inputDescription}
${directInput}${keywordsContext}

LEGAL FRAMEWORK:
${lawTreeSummary}

CRITICAL TASK REQUIREMENTS:
1. ANALYZE ALL MESSAGES TOGETHER: Information from different messages may combine to create privacy risks
   - Example: If message 1 mentions "name" and message 2 mentions "address", this is HIGH risk (full identity)
   - Example: If message 1 mentions "hospital visit" and message 2 mentions "medication", infer health condition
2. EXTRACT KEYWORDS FROM ALL MESSAGES: Don't just focus on the last message
3. IDENTIFY CROSS-MESSAGE PATTERNS: Look for information that connects across messages
4. MAP EACH RISK to the most specific legal clause name
5. Output ONLY compact format (no JSON, no markdown, no extra text, NO header line)
6. **ONE RISK PER LINE**: Each risk MUST be on a separate line (press Enter after each risk)

OUTPUT FORMAT (COMPACT - NO HEADER, direct data output, ONE RISK PER LINE):
value1,value2,value3,value4,value5
value1,value2,value3,value4,value5

FIELD DEFINITIONS:
- law_node_name: exact leaf node name from legal framework (NO translation, NO abbreviation)
- risk_level: HIGH | MEDIUM | LOW (based on INFERENCE CERTAINTY)
- privacy_exposure: what privacy info is exposed (consider information from ALL messages)
- inference_chain: reasoning - what data appears, how they connect, what can be inferred, confidence level
- used_infons: information elements in format "TYPE:VALUE" separated by |
  * **DESC format**: DESC:attribute_value (ONLY attribute, NOT "entity:attribute")
  * **SCEN format**: SCEN:temporal@spatial
  * **REL format**: REL:relation_name
  * Example: DESC:Klook|DESC:台北|SCEN:下周@东京
  * Extract concrete keywords from input, NOT inferences

CRITICAL RULES:
- Output ONLY the compact format, no other text
- **NO UNCERTAIN OUTPUT**: DO NOT output "无法推断", "不确定", "insufficient data", etc. If you cannot confidently infer a privacy risk, skip it entirely. Do NOT output any explanatory text.
- **ONE RISK PER LINE**: Each complete risk entry MUST be on its own line (use real line breaks between risks)
- Escape commas in text with \\,, newlines with \\n, backslashes with \\\\
- law_node_name MUST be exact copy from legal framework above (NO translation, NO abbreviation)
- used_infons format: "TYPE:VALUE" separated by | (e.g., DESC:Klook|DESC:台北|SCEN:下周@东京)
- If you see "CUSTOM PRIVACY ANALYSIS MODE", ONLY analyze the selected items marked with ✓
- Deep inference: infer health conditions, beliefs from behaviors across messages
- RISK LEVEL ASSIGNMENT: Evaluate inference CERTAINTY based on data clarity and context
- CROSS-MESSAGE ANALYSIS: A single privacy risk may be supported by keywords from multiple messages
- Sort by risk_level: HIGH first
- LANGUAGE CONSISTENCY: Write privacy_exposure and inference_chain in the SAME language as the input

**FINAL FORMAT CHECK BEFORE OUTPUT**:
Each line MUST have EXACTLY 5 fields in this order:
1. law_node_name (copy from legal framework)
2. risk_level (HIGH or MEDIUM or LOW)
3. privacy_exposure (what is exposed)
4. inference_chain (reasoning)
5. used_infons (format: TYPE:VALUE separated by |)

Example format to follow:
姓名,HIGH,暴露真实姓名,明确提到姓名信息,DESC:王小明
年龄,HIGH,暴露精确年龄,明确的年龄数值,DESC:27岁

REMEMBER: Output ONLY valid risk data lines. DO NOT output any statements like "无法推断", "不确定", "cannot determine", etc. If uncertain about a risk, simply don't output it.

NOW OUTPUT THE COMPACT FORMAT:`
    
    return simplePrompt
  }
  
  // 提取信息元模式：使用信息元列表
  const infonsSummary = extractInfonsSummary(infons)
  
  // 构建简洁版提示词，更适合本地模型
  const simplePrompt = `You are a privacy risk analyzer. Analyze the information elements below and identify privacy risks.

INPUT DATA TO ANALYZE:
${infonsSummary}

LEGAL FRAMEWORK:
${lawTreeSummary}

TASK:
1. Identify what privacy information can be inferred from the input data
2. Map each privacy risk to the most specific legal clause name
3. Output ONLY compact format (no JSON, no markdown, no extra text, NO header line)
4. **ONE RISK PER LINE**: Each risk MUST be on a separate line (press Enter after each risk)

OUTPUT FORMAT (COMPACT - NO HEADER, direct data output, ONE RISK PER LINE):
value1,value2,value3,value4,value5
value1,value2,value3,value4,value5

FIELD DEFINITIONS:
- law_node_name: exact leaf node name from legal framework (NO translation, NO abbreviation)
- risk_level: HIGH | MEDIUM | LOW (based on INFERENCE CERTAINTY)
- privacy_exposure: what privacy info is exposed
- inference_chain: reasoning - what data shows, what it implies, why it matters, confidence level
- used_infons: information elements in format "TYPE:VALUE" separated by |
  * **DESC format**: DESC:attribute_value (ONLY attribute, NOT "entity:attribute")
  * **SCEN format**: SCEN:temporal@spatial
  * **REL format**: REL:relation_name
  * Example: DESC:Klook|DESC:台北|SCEN:下周@东京
  * Extract concrete keywords from input, NOT inferences

CRITICAL RULES:
- Output ONLY the compact format, no other text
- **NO UNCERTAIN OUTPUT**: DO NOT output "无法推断", "不确定", "insufficient data", etc. If you cannot confidently infer a privacy risk, skip it entirely. Do NOT output any explanatory text.
- **ONE RISK PER LINE**: Each complete risk entry MUST be on its own line (use real line breaks between risks)
- Escape commas in text with \\,, newlines with \\n, backslashes with \\\\
- law_node_name MUST be exact copy from legal framework above (NO translation, NO abbreviation)
- used_infons format: "TYPE:VALUE" separated by |
  * DESC: Extract ONLY attribute value (e.g., "DESC: 平台: Klook" → "DESC:Klook")
  * SCEN: Combine temporal@spatial (e.g., "SCEN: 下周 @ 东京" → "SCEN:下周@东京")
  * REL: Use relation name (e.g., "REL: 住宿预订" → "REL:住宿预订")
- If you see "CUSTOM PRIVACY ANALYSIS MODE", ONLY analyze the selected items marked with ✓
- Deep inference: infer health conditions, beliefs from behaviors (e.g., gluten-free → celiac disease)
- RISK LEVEL ASSIGNMENT: Evaluate inference CERTAINTY based on data clarity and context
- Sort by risk_level: HIGH first
- LANGUAGE CONSISTENCY: Write privacy_exposure and inference_chain in the SAME language as the input data

**FINAL FORMAT CHECK BEFORE OUTPUT**:
Each line MUST have EXACTLY 5 fields in this order:
1. law_node_name (copy from legal framework, NO translation, NO abbreviation)
2. risk_level (HIGH or MEDIUM or LOW)
3. privacy_exposure (what is exposed)
4. inference_chain (reasoning)
5. used_infons (format: TYPE:VALUE separated by |)

Example format to follow:
姓名,HIGH,暴露真实姓名,明确提到姓名信息,DESC:王小明
年龄,HIGH,暴露精确年龄,明确的年龄数值,DESC:27岁

REMEMBER: Output ONLY valid risk data lines. DO NOT output any statements like "无法推断", "不确定", "cannot determine", etc. If uncertain about a risk, simply don't output it.

NOW OUTPUT THE COMPACT FORMAT:`
  
  return simplePrompt
}

// Incremental parser for streaming risk items with partial object support
// Auto-detects format (compact vs JSON) and uses appropriate parser
export function incrementalExtractRisks(streamText, parser) {
  const text = String(streamText || '')
  
  // Auto-detect format on first call
  if (!parser || !parser.formatDetected) {
    // Look for compact format header
    const compactMatch = text.match(/risks\[(\d+)\]\{([^}]+)\}:/)
    // Look for JSON format
    const jsonMatch = text.match(/"risks"\s*:\s*\[/)
    
    // Determine format based on which appears first
    let useCompact = false
    if (compactMatch && jsonMatch) {
      useCompact = compactMatch.index < jsonMatch.index
    } else if (compactMatch) {
      useCompact = true
    } else if (jsonMatch) {
      useCompact = false
    } else {
      // No format detected yet, keep old state if exists
      if (parser) return { state: parser, yielded: [] }
      
      // Initialize with default (try compact first)
      useCompact = true
    }
    
    // Initialize parser state with format info
    if (!parser) {
      parser = {
        formatDetected: true,
        isCompact: useCompact
      }
    } else {
      parser.formatDetected = true
      parser.isCompact = useCompact
    }
  }
  
  // Route to appropriate parser
  if (parser.isCompact) {
    return incrementalExtractRisksCompact(text, parser)
  } else {
    return incrementalExtractRisksJSON(text, parser)
  }
}

// Original JSON parser (renamed)
function incrementalExtractRisksJSON(streamText, parser) {
  const state = parser || { 
    foundArray: false, 
    arrayStart: -1, 
    scanPos: 0, 
    inString: false, 
    escape: false, 
    objStart: -1, 
    braceDepth: 0, 
    closed: false, 
    objectStates: new Map(), // Map<objIndex, {lastParsedHash, data}>
    currentObjIndex: 0,
    formatDetected: true,
    isCompact: false
  }
  const yielded = []
  const text = String(streamText || '')

  // 查找 "risks" 数组
  if (!state.foundArray) {
    const m = /"risks"\s*:\s*\[/.exec(text)
    if (!m) {
      state.scanPos = text.length
      return { state, yielded }
    }
    state.foundArray = true
    state.arrayStart = m.index + m[0].lastIndexOf('[')
    state.scanPos = state.arrayStart + 1
  }

  let i = state.scanPos
  let inString = state.inString
  let escape = state.escape
  let objStart = state.objStart
  let braceDepth = state.braceDepth

  // 数组已关闭则不再扫描
  if (state.closed) return { state, yielded }

  for (; i < text.length; i++) {
    const ch = text[i]
    if (inString) {
      if (escape) { escape = false; continue }
      if (ch === '\\') { escape = true; continue }
      if (ch === '"') { inString = false; continue }
      continue
    }
    if (ch === '"') { inString = true; continue }
    if (ch === '{') {
      if (objStart < 0) { 
        objStart = i
        braceDepth = 1
        // 新对象开始
        if (!state.objectStates.has(state.currentObjIndex)) {
          state.objectStates.set(state.currentObjIndex, { lastParsedHash: null, data: {} })
        }
      } else { 
        braceDepth++ 
      }
      continue
    }
    if (ch === '}') {
      if (objStart >= 0) {
        braceDepth--
        if (braceDepth === 0) {
          // 对象完整闭合，进行完整解析
          let objText = text.slice(objStart, i + 1)
          
          // 尝试清理可能的额外内容（LLM输出可能包含注释或额外文本）
          objText = objText.trim()
          
          // 确保对象文本以 } 结尾
          if (!objText.endsWith('}')) {
            const lastBrace = objText.lastIndexOf('}')
            if (lastBrace >= 0) {
              objText = objText.slice(0, lastBrace + 1)
            }
          }
          
          try {
            const value = JSON.parse(objText)
            const objState = state.objectStates.get(state.currentObjIndex)
            if (objState) {
              yielded.push({ ...value, _objIndex: state.currentObjIndex, _isComplete: true })
              objState.data = value
              objState.lastParsedHash = computeHashId(objText)
            }
          } catch (err) {
            // JSON解析失败时，尝试使用部分解析逻辑作为兜底
            const objState = state.objectStates.get(state.currentObjIndex)
            if (objState && Object.keys(objState.data).length > 0) {
              // 如果部分数据已经存在，标记为完成
              yielded.push({ ...objState.data, _objIndex: state.currentObjIndex, _isComplete: true })
              objState.lastParsedHash = computeHashId(objText)
            }
            // 只在开发环境输出警告
            if (process.env.NODE_ENV === 'development') {
              console.debug('[incrementalExtractRisks] Failed to parse complete object, using partial data:', err.message)
            }
          }
          objStart = -1
          state.currentObjIndex++
        }
      }
      continue
    }
    if (ch === ']') {
      // 数组关闭
      if (objStart < 0) { state.closed = true; i++; break }
    }
    
    // 尝试部分解析（每隔一定字符数或遇到特定标记时）
    if (objStart >= 0 && braceDepth > 0) {
      const objText = text.slice(objStart, i + 1)
      // 当累积了足够多的内容时，尝试部分解析（检测到逗号或已有足够内容）
      if ((ch === ',' || ch === '\n') && (i - objStart) > 20) {
        const objState = state.objectStates.get(state.currentObjIndex)
        const currentHash = computeHashId(objText)
        
        // 只有当内容有变化时才解析
        if (objState && objState.lastParsedHash !== currentHash) {
          const partialData = parsePartialObject(objText)
          if (partialData && Object.keys(partialData).length > 0) {
            // 检查是否有新字段
            const hasNewData = Object.keys(partialData).some(
              key => partialData[key] !== objState.data[key]
            )
            
            if (hasNewData) {
              objState.data = { ...objState.data, ...partialData }
              objState.lastParsedHash = currentHash
              yielded.push({ ...objState.data, _objIndex: state.currentObjIndex, _isComplete: false })
            }
          }
        }
      }
    }
  }

  state.inString = inString
  state.escape = escape
  state.objStart = objStart
  state.braceDepth = braceDepth
  state.scanPos = i
  return { state, yielded }
}

// Parse partial JSON object by extracting completed fields
function parsePartialObject(objText) {
  const result = {}
  
  // 尝试提取已完成的字段（针对关键字段优先提取）
  const criticalFields = ['law_node_name', 'risk_level', 'confidence']
  const otherFields = ['privacy_exposure', 'inference_chain', 'used_infons']
  
  for (const field of [...criticalFields, ...otherFields]) {
    const value = extractFieldValue(objText, field)
    if (value !== null) {
      result[field] = value
    }
  }
  
  return result
}

// Extract a specific field value from partial JSON text
function extractFieldValue(text, fieldName) {
  // 匹配 "fieldName": value 模式
  const patterns = [
    // 字符串值
    new RegExp(`"${fieldName}"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"`, 's'),
    // 数字值
    new RegExp(`"${fieldName}"\\s*:\\s*(\\d+\\.?\\d*)`, 's'),
    // 数组值（简单处理，找到完整的数组）
    new RegExp(`"${fieldName}"\\s*:\\s*(\\[[^\\]]*\\])`, 's'),
    // 对象值（简单处理，需要平衡的大括号）
    new RegExp(`"${fieldName}"\\s*:\\s*(\\{[^}]*\\})`, 's')
  ]
  
  for (const pattern of patterns) {
    const match = text.match(pattern)
    if (match) {
      try {
        // 对于字符串，直接返回捕获的内容
        if (pattern.source.includes('"([^"]*')) {
          return match[1].replace(/\\"/g, '"').replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
        }
        // 对于数字、数组、对象，尝试JSON解析
        return JSON.parse(match[1])
      } catch (err) {
        // 解析失败，继续尝试下一个模式
        continue
      }
    }
  }
  
  return null
}

// Compute simple hash ID for deduplication
function computeHashId(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash // Convert to 32bit integer
  }
  return hash.toString(36)
}

// Parse streaming response (legacy, for backward compatibility)
export function parseStreamingResponse(text) {
  try {
    return JSON.parse(text)
  } catch {
    const match = text.match(/\{[\s\S]*\}/)
    if (match) {
      try {
        return JSON.parse(match[0])
      } catch {
        return null
      }
    }
    return null
  }
}

// ============================================================================
// COMPACT FORMAT PARSERS (New)
// ============================================================================

/**
 * Unescape special characters in compact format field values
 * @param {string} value - Escaped value
 * @returns {string} Unescaped value
 */
function unescapeValue(value) {
  if (typeof value !== 'string') return value
  return value
    .replace(/\\,/g, ',')
    .replace(/\\n/g, '\n')
    .replace(/\\\\/g, '\\')
}

/**
 * Split array field by | separator (for used_infons and similar fields)
 * @param {string} value - Value containing | separators
 * @returns {Array<string>} Array of split values
 */
function splitArrayField(value) {
  if (!value || typeof value !== 'string') return []
  // Split by | but not by escaped \|
  const parts = value.split(/(?<!\\)\|/)
  return parts.map(p => p.trim()).filter(Boolean)
}

/**
 * Parse a single compact format line into a risk object
 * @param {string} line - Data line (e.g., "医疗健康,HIGH,User likely has...")
 * @param {Array<string>} fields - Field names from header
 * @returns {Object|null} Parsed risk object or null if parsing fails
 */
function parseCompactLine(line, fields) {
  if (!line || !line.trim()) return null
  
  // Split by comma, but respect escaped commas
  const values = []
  let currentValue = ''
  let escaped = false
  
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    
    if (escaped) {
      currentValue += ch
      escaped = false
      continue
    }
    
    if (ch === '\\') {
      currentValue += ch
      escaped = true
      continue
    }
    
    if (ch === ',') {
      values.push(currentValue)
      currentValue = ''
      continue
    }
    
    currentValue += ch
  }
  
  // Push the last value
  if (currentValue || values.length > 0) {
    values.push(currentValue)
  }
  
  // Build the risk object
  const risk = {}
  
  // Handle field mapping with flexibility for missing fields
  for (let i = 0; i < fields.length; i++) {
    const field = fields[i].trim()
    let value = ''
    
    if (i < values.length) {
      value = values[i].trim()
    } else {
      // Field is missing, use empty string
      value = ''
    }
    
    // Handle array fields (used_infons, etc.)
    if (field === 'used_infons' || field === 'arg_refs' || field === 'arg_types') {
      risk[field] = value ? splitArrayField(value) : []
    } else {
      risk[field] = unescapeValue(value)
    }
  }
  
  return Object.keys(risk).length > 0 ? risk : null
}

/**
 * Parse complete compact format text into an array of risk objects
 * Supports both headerless format (new) and header format (old, for compatibility)
 * @param {string} text - Complete compact format text
 * @returns {Object|null} Parsed object with risks array, or null if parsing fails
 */
export function parseCompactFormat(text) {
  if (!text || typeof text !== 'string') return null
  
  // Fixed field order: law_node_name, risk_level, privacy_exposure, inference_chain, used_infons
  const defaultFields = ['law_node_name', 'risk_level', 'privacy_exposure', 'inference_chain', 'used_infons']
  
  // Try to match optional header: risks[N]{field1,field2,...}:
  const headerMatch = text.match(/risks\[(\d+)\]\{([^}]+)\}:/)
  
  let fields = defaultFields
  let dataText = text
  
  if (headerMatch) {
    // Header found (old format), extract fields from header
    const fieldsStr = headerMatch[2]
    fields = fieldsStr.split(',').map(f => f.trim())
    const headerEnd = headerMatch.index + headerMatch[0].length
    dataText = text.slice(headerEnd)
  }
  // If no header, treat entire text as data (new format)
  
  const lines = dataText.split('\n')
  const risks = []
  
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue
    
    // Parse data line
    const risk = parseCompactLine(trimmed, fields)
    if (risk) {
      risks.push(risk)
    }
  }
  
  return { risks }
}

/**
 * Incremental compact format parser for streaming
 * Parses compact format line by line as data arrives
 * Supports both headerless format (new) and header format (old)
 * @param {string} streamText - Accumulated stream text
 * @param {Object} parser - Parser state object
 * @returns {Object} { state, yielded } - Updated state and newly parsed risks
 */
export function incrementalExtractRisksCompact(streamText, parser) {
  // Fixed field order: law_node_name, risk_level, privacy_exposure, inference_chain, used_infons
  const defaultFields = ['law_node_name', 'risk_level', 'privacy_exposure', 'inference_chain', 'used_infons']
  
  // Initialize state with defaults, then merge with parser (ensuring all fields have values)
  const state = {
    foundHeader: false,
    fields: defaultFields,
    count: 0,
    scanPos: 0,
    parsedLines: 0,
    ...(parser || {})
  }
  
  // Ensure critical fields are never undefined
  if (state.parsedLines === undefined) state.parsedLines = 0
  if (state.scanPos === undefined) state.scanPos = 0
  if (!state.fields || state.fields.length === 0) state.fields = defaultFields
  
  const yielded = []
  const text = String(streamText || '')
  
  // Step 1: Check for optional header
  if (!state.foundHeader) {
    const headerMatch = text.match(/risks\[(\d+)\]\{([^}]+)\}:/)
    if (headerMatch) {
      // Header found (old format)
      state.foundHeader = true
      state.count = parseInt(headerMatch[1], 10)
      state.fields = headerMatch[2].split(',').map(f => f.trim())
      state.scanPos = headerMatch.index + headerMatch[0].length
    } else {
      // No header (new format), treat as headerless
      state.foundHeader = true
      state.fields = defaultFields
      state.scanPos = 0
    }
  }
  
  // Step 2: Parse data lines incrementally
  const dataText = text.slice(state.scanPos)
  const lines = dataText.split('\n')
  
  // Process each line after the already parsed ones
  for (let i = state.parsedLines; i < lines.length; i++) {
    const line = lines[i]
    
    const trimmed = line.trim()
    
    // Check if this is the last line and it doesn't end with newline (still streaming)
    const isLastLine = (i === lines.length - 1)
    const hasNewlineEnding = dataText.endsWith('\n')
    const isStreamingLine = isLastLine && !hasNewlineEnding
    
    // Skip empty lines
    if (!trimmed) {
      // Only increment parsedLines for completed lines (with newline)
      if (!isStreamingLine) {
        state.parsedLines++
      }
      continue
    }
    
    // For streaming lines (no newline yet), check if it looks complete enough to parse
    if (isStreamingLine) {
      // Count unescaped commas to estimate field count
      const commaCount = (trimmed.match(/(?<!\\),/g) || []).length
      
      // Need at least 4 commas for 5 fields
      // BUT: Allow 3 commas (4 fields) in case inference_chain is omitted by LLM
      if (commaCount < 3) {
        // Not enough fields yet, skip this line (don't increment parsedLines)
        break
      }
    }
    
    // Try to parse the line
    const risk = parseCompactLine(trimmed, state.fields)
    if (risk) {
      risk._objIndex = i  // Use line index as object index for tracking
      risk._isComplete = !isStreamingLine  // Mark as incomplete if still streaming
      yielded.push(risk)
    }
    
    // Only increment parsedLines for completed lines (with newline)
    // For streaming lines, keep parsedLines unchanged so we re-parse next time
    if (!isStreamingLine) {
      state.parsedLines++
    }
  }
  
  return { state, yielded }
}

