// Privacy Inference Prompt Template — 针对4B小参数模型优化

// Extract information elements summary with ID mapping
export function extractInfonsSummary(infons) {
  if (!Array.isArray(infons) || infons.length === 0) {
    return { summary: 'No information elements', idMap: new Map() }
  }
  
  const idMap = new Map() // I1 -> infon object
  
  const lines = infons.map((infon, idx) => {
    const type = String(infon.infon_type || '').toUpperCase()
    const shortId = `I${idx + 1}` // I1, I2, I3...
    
    // Store mapping: shortId -> original infon (with iid)
    idMap.set(shortId, { ...infon, _shortId: shortId })
    
    let keyword = ''
    if (type === 'DESC') {
      const entity = infon.entity ?? ''
      const attribute = infon.attribute ?? ''
      keyword = entity && attribute ? `${entity}:${attribute}` : (entity || attribute || '-')
    } else if (type === 'SCEN') {
      const temporal = infon.temporal ?? ''
      const spatial = infon.spatial ?? ''
      keyword = temporal || spatial ? `${temporal}@${spatial}` : '-'
    } else if (type === 'REL') {
      keyword = infon.relation_name ?? '-'
    } else if (type === 'SIT') {
      keyword = infon.description ?? '-'
    }
    
    return `${shortId}=${type}:${keyword}`
  })
  
  return { summary: lines.join(', '), idMap }
}

// Extract law tree summary with ID mapping (for filling template)
// Returns { summary: string, idMap: Map<string, nodeInfo> }
export function extractLawTreeSummary(lawData) {
  if (!lawData) {
    return { summary: 'No legal structure', idMap: new Map() }
  }
  
  const idMap = new Map() // L1 -> { name, id, path }
  
  // Custom模式：只使用用户选中的隐私项
  if (lawData.isCustom) {
    const customItems = lawData.customItems || []
    if (customItems.length === 0) {
      return { summary: 'No items selected', idMap }
    }
    
    const lines = customItems.map((item, idx) => {
      const shortId = `L${idx + 1}`
      const name = item.label || item.id || ''
      idMap.set(shortId, { name, id: item.id, _shortId: shortId })
      return `${shortId}=${name}`
    })
    
    return { summary: `CUSTOM: ${lines.join(', ')}`, idMap }
  }
  
  // 标准法律树模式：只提取叶子节点
  if (!lawData.name) {
    return { summary: 'No legal structure', idMap }
  }
  
  const leafNodes = []
  let counter = 1
  
  // 递归收集所有叶子节点
  function collectLeafNodes(node, path = []) {
    const currentPath = [...path, node.name]
    const isLeaf = !Array.isArray(node.children) || node.children.length === 0
    
    if (isLeaf) {
      const shortId = `L${counter++}`
      leafNodes.push({ shortId, name: node.name, id: node.id })
      idMap.set(shortId, { name: node.name, id: node.id, path: currentPath, _shortId: shortId })
    } else if (Array.isArray(node.children)) {
      node.children.forEach(child => collectLeafNodes(child, currentPath))
    }
  }
  
  collectLeafNodes(lawData)
  
  // 返回编号映射列表
  const lines = leafNodes.map(n => `${n.shortId}=${n.name}`)
  return { summary: lines.join(', '), idMap }
}

// Fill prompt template - 针对4B小参数模型优化，使用编号映射
// Returns { prompt: string, lawIdMap: Map, infonIdMap: Map, isEmpty: boolean }
export function fillPromptTemplate(infons, lawData) {
  const { summary: lawSummary, idMap: lawIdMap } = extractLawTreeSummary(lawData)
  const { summary: infonSummary, idMap: infonIdMap } = extractInfonsSummary(infons)
  
  // 检查是否有可用的隐私类别
  const hasCategories = lawIdMap.size > 0
  const hasInfons = infonIdMap.size > 0
  
  // 如果没有类别或没有信息元，返回空标记
  if (!hasCategories || !hasInfons) {
    return { 
      prompt: '', 
      lawIdMap, 
      infonIdMap, 
      isEmpty: true,
      emptyReason: !hasCategories ? 'no_categories' : 'no_infons'
    }
  }
  
  // 构建精简版提示词，使用编号映射，严格控制误报
  const simplePrompt = `Check if information elements reveal the USER's own personal privacy.

## Information Elements
${infonSummary}

## Privacy Categories
${lawSummary}

## Task
Check each infon: does it CONCRETELY reveal the user's own personal data matching a category?

## Output Format (one risk per line)
law_id,level,reason,infon_ids

## Strict Rules
- ONLY report when an infon contains CONCRETE personal data about the user (e.g., real name, real ID number, real address)
- Do NOT speculate or infer risks from greetings, common words, or vague context
- Do NOT force output — if no infon matches any category, output exactly: NONE
- Output format per line: law_id (L1/L2...), level (HIGH/MEDIUM/LOW), reason (Chinese, ≤15 chars), infon_ids (I1|I2...)
- HIGH: infon contains user's exact personal identifier (full name, ID number, phone number, bank account)
- MEDIUM: infon contains user's partial but identifiable data (city + age, partial address)
- LOW: infon contains weak personal indicator that alone cannot identify user
- Maximum 6 lines, HIGH first. Most inputs should yield 0-2 risks.

## Examples
If infons are just greetings like "你好", output:
NONE

If I1=姓名:张三, I2=城市:北京:
L1,HIGH,明确提及用户真实姓名,I1
L2,MEDIUM,提及用户所在城市,I2

Output:`
  
  return { prompt: simplePrompt, lawIdMap, infonIdMap, isEmpty: false }
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

// ============================================================================
// COMPACT FORMAT PARSERS
// ============================================================================

function unescapeValue(value) {
  if (typeof value !== 'string') return value
  return value
    .replace(/\\,/g, ',')
    .replace(/\\n/g, '\n')
    .replace(/\\\\/g, '\\')
}

// Split array field by | separator
function splitArrayField(value) {
  if (!value || typeof value !== 'string') return []
  // Split by | but not by escaped \|
  const parts = value.split(/(?<!\\)\|/)
  return parts.map(p => p.trim()).filter(Boolean)
}

// Parse a single compact format line into a risk object
function parseCompactLine(line, fields) {
  if (!line || !line.trim()) return null
  
  // Skip "NONE" or non-risk lines (model outputs NONE when no risk found)
  const upper = line.trim().toUpperCase()
  if (upper === 'NONE' || upper === 'N/A' || upper === 'NO RISKS' || upper === 'NO RISK') return null
  
  // Must start with a valid law_id pattern (L followed by number) to be a risk line
  if (!/^L\d+\s*,/.test(line.trim())) return null
  
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
    
    // Handle array fields (infon_ids, used_infons, etc.)
    if (field === 'infon_ids' || field === 'used_infons' || field === 'arg_refs' || field === 'arg_types') {
      risk[field] = value ? splitArrayField(value) : []
    } else {
      risk[field] = unescapeValue(value)
    }
  }
  
  return Object.keys(risk).length > 0 ? risk : null
}

// Resolve L1/I1 IDs to actual names using the ID maps
export function resolveRiskIds(risk, lawIdMap, infonIdMap) {
  const resolved = { ...risk }
  
  // Resolve law_id -> law_node_name
  if (risk.law_id && lawIdMap) {
    const lawInfo = lawIdMap.get(risk.law_id)
    if (lawInfo) {
      resolved.law_node_name = lawInfo.name
      resolved.law_node_id = lawInfo.id
      resolved.law_node_path = lawInfo.path
    } else {
      // Fallback: use the ID as name if not found
      resolved.law_node_name = risk.law_id
    }
  }
  
  // Resolve infon_ids -> used_infons (with original iid references)
  if (risk.infon_ids && Array.isArray(risk.infon_ids) && infonIdMap) {
    resolved.used_infons = risk.infon_ids.map(shortId => {
      const infonInfo = infonIdMap.get(shortId)
      if (infonInfo) {
        // Return original iid for matching
        return infonInfo.iid || shortId
      }
      return shortId
    })
    // Also keep reference to full infon objects for detailed info
    resolved._resolved_infons = risk.infon_ids.map(shortId => infonIdMap.get(shortId)).filter(Boolean)
  }
  
  // Map reason to inference_chain for compatibility
  if (risk.reason) {
    resolved.inference_chain = risk.reason
  }
  
  // Map risk_level for consistency
  if (risk.risk_level) {
    resolved.risk_level = risk.risk_level.toUpperCase()
  }
  
  return resolved
}

// Parse complete compact format text into risk objects array
export function parseCompactFormat(text) {
  if (!text || typeof text !== 'string') return null
  
  // New 4-field format: law_id, risk_level, reason, infon_ids
  const defaultFields = ['law_id', 'risk_level', 'reason', 'infon_ids']
  
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

// Incremental compact format parser for streaming
export function incrementalExtractRisksCompact(streamText, parser) {
  // New 4-field format: law_id, risk_level, reason, infon_ids
  const defaultFields = ['law_id', 'risk_level', 'reason', 'infon_ids']
  
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
  
  // Guard: if buffer shrank (e.g. <think> block removed), reset parsedLines
  if (state.parsedLines > lines.length) {
    state.parsedLines = 0
  }
  
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
