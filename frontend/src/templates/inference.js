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
- **HIGH**: Sensitive personal data (health, beliefs, biometrics, children's data) can be directly inferred or strongly suggested
- **MEDIUM**: Behavioral patterns, preferences, or non-sensitive attributes can be derived with reasonable confidence
- **LOW**: General trends or public information can be weakly inferred

## Output Format
**CRITICAL - Streaming Optimization**: To enable fast progressive visualization, output fields in THIS EXACT ORDER for each risk:
1. FIRST: law_node_name (for quick highlighting)
2. SECOND: risk_level (for color coding)
3. THEN: all other fields (inference_chain, privacy_exposure, used_infons, etc.)

Output ONLY valid JSON (no markdown, no explanation):
{
  "risks": [
    {
      "law_node_name": "Most specific (leaf-level) node name, e.g., 'Article 13, Paragraph 2'",
      "risk_level": "HIGH | MEDIUM | LOW",
      "privacy_exposure": "Specific privacy information exposed or inferable (be explicit about implicit inferences)",
      "inference_chain": "Step-by-step reasoning: [1] What explicit info? [2] What can be implicitly inferred? [3] Why does this violate the specific clause?",
      "used_infons": ["exact_entity_name", "attribute_value", "temporal_expression", "spatial_location"]
    }
  ]
}

## Critical Requirements
1. **Output ONLY JSON** - no markdown code blocks, no explanatory text
2. **Field Order Matters** - MUST output fields in the exact order shown above (law_node_name → risk_level → other fields) for optimal streaming performance
3. **Map to LEAF NODES ONLY** - CRITICAL: Always identify the MOST SPECIFIC legal clause at the DEEPEST level in hierarchy (leaf nodes). NEVER map to intermediate/parent nodes. The law_node_name MUST be the final level clause name that appears in the law tree structure.
4. **EXACT Name Matching** - The law_node_name MUST be copied EXACTLY from the law tree structure, character by character. DO NOT paraphrase, summarize, or modify the node name. Look for the [LEAF NODE - USE THIS] markers in the law tree and copy the name EXACTLY.
5. **CUSTOM MODE RESTRICTION** - If you see "CUSTOM PRIVACY ANALYSIS MODE" in the law tree section, the law_node_name MUST be one of the explicitly listed selected items (marked with ✓). You CANNOT use any other privacy category names, even if you can infer them. If a privacy risk doesn't match any selected item, skip it entirely.
6. **Deep Inference** - Don't just report explicit data; infer health conditions, beliefs, status from behavior/preferences
7. **Comprehensive Coverage** - Analyze ALL possible privacy angles: direct exposure + implicit inference + contextual correlation (BUT respect Custom Mode restrictions)
8. **Clear Attribution** - Every risk must trace back to specific information elements with logical reasoning
9. **used_infons Extraction** - CRITICAL: Extract EXACT KEYWORDS from input that support each risk. Like infons.js DESC/SCEN extraction: concrete entity names, attribute values, temporal expressions, spatial locations. Extract the ORIGINAL TEXT (highlightable keywords), NOT infon IDs, NOT paraphrases. Examples: entity names like "王小明", attribute values like "27", temporal like "今年", spatial like "北京市". DO NOT extract abstract relations or inferred concepts - only concrete observable keywords from the source.
10. **Prioritize Sensitivity** - Treat health, beliefs, children, biometrics as HIGH risk
11. **Sort Properly** - HIGH risks first
12. **Verify Before Output** - Before outputting each risk, find the [LEAF NODE - USE THIS] entry in the provided law tree and copy its exact name. If you cannot find an exact leaf node match, choose the closest leaf node from the tree.
13. **LANGUAGE CONSISTENCY** - Write privacy_exposure and inference_chain in the SAME language as the input information elements. If the input data is in Chinese, write your analysis in Chinese. If in English, write in English. Match the language of the user's data.

## Example Output Format
When you find a privacy risk, you MUST:
1. Find the DEEPEST leaf node in the law tree that matches this privacy exposure
2. Set law_node_name to ONLY the leaf node name (copied EXACTLY from the law tree)
3. DO NOT use intermediate node names - always use the most specific (leaf) node
4. Output fields in order: law_node_name, risk_level, then other fields

Example:
If input contains "looking for gluten-free restaurant menu", infer:
- IMPLICIT: User likely has celiac disease or gluten intolerance (health condition = HIGH risk)
  → Find the LEAF node for health data in the law tree (e.g., "医疗健康")
  → law_node_name = "医疗健康" (copied exactly from law tree)
  → used_infons: ["gluten-free", "restaurant menu"] (exact keywords from input)
- DIRECT: User's dietary preference (MEDIUM risk)
  → used_infons: ["gluten-free"] (exact text)
- CONTEXTUAL: Location search reveals geographic pattern (LOW-MEDIUM risk)
  → used_infons: ["restaurant", "location search"] (observable keywords)

If input is "我叫王小明，今年27岁，住在北京市海淀区", extract keywords:
- Name exposure: used_infons: ["王小明"] (entity name from input)
- Age exposure: used_infons: ["27", "今年"] (attribute value + temporal expression)
- Location exposure: used_infons: ["北京市", "海淀区"] (spatial location keywords)

**Custom Mode Example**:
If law tree shows "CUSTOM PRIVACY ANALYSIS MODE" with selected items: ["Home Address", "Location/GPS", "Health Data"]
And input contains: "User attends Friday prayers at mosque and searches for halal restaurants"
- ✅ REPORT: Dietary preference (halal) → inferred health/dietary need → law_node_name = "Health Data" (selected)
  → used_infons: ["halal", "restaurants"] (exact keywords)
- ✅ REPORT: Location search pattern → law_node_name = "Location/GPS" (selected)
  → used_infons: ["Friday prayers", "mosque"] (location/time keywords)
- ❌ DO NOT REPORT: Religious belief inference (Islam) → NOT in selected list, must skip even though it's inferable
- ❌ DO NOT REPORT: Any privacy category not explicitly listed in the selected items

NOW analyze the input and output the complete JSON.
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
      keywordsContext = `\n\nKNOWN PRIVACY KEYWORDS (already identified in previous analyses):
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
5. Output ONLY valid JSON (no markdown, no extra text)

OUTPUT FORMAT (EXACT JSON):
{
  "risks": [
    {
      "law_node_name": "exact leaf node name from legal framework",
      "risk_level": "HIGH or MEDIUM or LOW",
      "privacy_exposure": "what privacy info is exposed (consider information from ALL messages)",
      "inference_chain": "reasoning: 1) what data appears in which message(s) 2) how they connect 3) what can be inferred 4) why it matters",
      "used_infons": ["exact keyword/entity from ANY message", "attribute value", "time expression", "location name"]
    }
  ]
}

CRITICAL RULES:
- Output ONLY the JSON object, no other text
- law_node_name MUST be exact copy from legal framework above
- used_infons should contain EXACT KEYWORDS from ANY/ALL messages (entities, attribute values, time expressions, locations) that support this risk. Extract concrete nouns, names, values, dates, places from the ENTIRE conversation. DO NOT use infon IDs.
- If you see "CUSTOM PRIVACY ANALYSIS MODE", ONLY analyze the selected items marked with ✓
- Deep inference: infer health conditions, beliefs from behaviors across messages (e.g., "gluten-free" in msg1 + "stomach pain" in msg2 → celiac disease)
- CROSS-MESSAGE ANALYSIS: A single privacy risk may be supported by keywords from multiple messages
- Sort by risk_level: HIGH first
- LANGUAGE CONSISTENCY: Write privacy_exposure and inference_chain in the SAME language as the input (if input is in Chinese, respond in Chinese; if in English, respond in English)

NOW OUTPUT THE JSON:`
    
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
3. Output ONLY valid JSON (no markdown, no extra text)

OUTPUT FORMAT (EXACT JSON):
{
  "risks": [
    {
      "law_node_name": "exact leaf node name from legal framework",
      "risk_level": "HIGH or MEDIUM or LOW",
      "privacy_exposure": "what privacy info is exposed",
      "inference_chain": "reasoning: 1) what data shows 2) what it implies 3) why it matters",
      "used_infons": ["exact entity/attribute keyword", "time expression", "location name", "value from input"]
    }
  ]
}

CRITICAL RULES:
- Output ONLY the JSON object, no other text
- law_node_name MUST be exact copy from legal framework above
- used_infons should contain EXACT KEYWORDS from the input data that support this risk. Extract concrete information elements: entity names, attribute values, temporal expressions, spatial locations (like DESC.attribute and SCEN.temporal/spatial from infons.js). DO NOT use infon IDs like "desc:r1_1".
- If you see "CUSTOM PRIVACY ANALYSIS MODE", ONLY analyze the selected items marked with ✓
- Deep inference: infer health conditions, beliefs from behaviors (e.g., gluten-free → celiac disease)
- Sort by risk_level: HIGH first
- LANGUAGE CONSISTENCY: Write privacy_exposure and inference_chain in the SAME language as the input data (if input is in Chinese, respond in Chinese; if in English, respond in English)

NOW OUTPUT THE JSON:`
  
  return simplePrompt
}

// Incremental parser for streaming risk items with partial object support
export function incrementalExtractRisks(streamText, parser) {
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
    currentObjIndex: 0
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

