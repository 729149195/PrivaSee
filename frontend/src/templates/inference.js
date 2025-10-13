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
      "used_infons": ["infon_id_1", "infon_id_2"]
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
9. **Prioritize Sensitivity** - Treat health, beliefs, children, biometrics as HIGH risk
10. **Sort Properly** - HIGH risks first
11. **Verify Before Output** - Before outputting each risk, find the [LEAF NODE - USE THIS] entry in the provided law tree and copy its exact name. If you cannot find an exact leaf node match, choose the closest leaf node from the tree.

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
- DIRECT: User's dietary preference (MEDIUM risk)
- CONTEXTUAL: Location search reveals geographic pattern (LOW-MEDIUM risk)

**Custom Mode Example**:
If law tree shows "CUSTOM PRIVACY ANALYSIS MODE" with selected items: ["Home Address", "Location/GPS", "Health Data"]
And input contains: "User attends Friday prayers at mosque and searches for halal restaurants"
- ✅ REPORT: Dietary preference (halal) → inferred health/dietary need → law_node_name = "Health Data" (selected)
- ✅ REPORT: Location search pattern → law_node_name = "Location/GPS" (selected)
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

// Extract law tree summary (for filling template)
export function extractLawTreeSummary(lawData) {
  if (!lawData) {
    return 'No legal structure available'
  }
  
  // Custom模式：只使用用户选中的隐私项
  if (lawData.isCustom) {
    const customItems = lawData.customItems || []
    if (customItems.length === 0) {
      return 'Custom Privacy Items:\n- (No items selected - cannot perform analysis)'
    }
    
    const lines = [
      '=== CUSTOM PRIVACY ANALYSIS MODE ===',
      '',
      '⚠️ CRITICAL CONSTRAINT: The user has ONLY selected the following specific privacy items for analysis.',
      'You MUST analyze ONLY these items. Do NOT analyze or report any privacy risks outside this list.',
      '',
      '✅ SELECTED PRIVACY ITEMS (analyze these):',
      ''
    ]
    
    customItems.forEach(item => {
      // item 现在包含 { id, label, category }
      const itemLabel = item.label || item.id || 'Unknown'
      const itemCategory = item.category || 'General'
      lines.push(`  ✓ "${itemLabel}" (Category: ${itemCategory}) [LEAF NODE - USE THIS]`)
    })
    
    lines.push('')
    lines.push('📋 STRICT MATCHING RULES:')
    lines.push('1. The law_node_name field MUST be EXACTLY one of the selected item labels listed above.')
    lines.push('2. Copy the exact label text (including capitalization and spacing).')
    lines.push('3. If you infer a privacy exposure that does NOT match any selected item above, SKIP IT entirely.')
    lines.push('4. Example: If you infer "Religious Belief" but it is NOT in the selected list above, do NOT report it.')
    lines.push('5. Example: If "Health Data" IS selected and you infer health conditions from diet preferences, report it with law_node_name="Health Data".')
    lines.push('')
    lines.push('❌ DO NOT report risks for privacy categories not in the selected list above, even if you can infer them from the data.')
    
    return lines.join('\n')
  }
  
  // 标准法律树模式
  if (!lawData.name) {
    return 'No legal structure available'
  }
  
  const lines = []
  
  function traverse(node, depth = 0, path = []) {
    const indent = '  '.repeat(depth)
    const currentPath = [...path, node.name]
    const isLeaf = !Array.isArray(node.children) || node.children.length === 0
    
    // 标记叶子节点，并显示完整路径
    if (isLeaf) {
      const fullPath = currentPath.join(' > ')
      lines.push(`${indent}- ${node.name} [LEAF NODE - USE THIS] (Path: ${fullPath})`)
    } else {
      lines.push(`${indent}- ${node.name} [PARENT NODE - DO NOT USE]`)
    }
    
    if (Array.isArray(node.children) && node.children.length > 0) {
      node.children.forEach(child => traverse(child, depth + 1, currentPath))
    }
  }
  
  traverse(lawData)
  return lines.join('\n')
}

// Fill prompt template
export function fillPromptTemplate(infons, lawData) {
  const infonsSummary = extractInfonsSummary(infons)
  const lawTreeSummary = extractLawTreeSummary(lawData)
  
  return PRIVACY_INFERENCE_PROMPT
    .replace('{{INFONS}}', infonsSummary)
    .replace('{{LAW_TREE}}', lawTreeSummary)
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

