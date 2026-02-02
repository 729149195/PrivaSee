/**
 * 信息元增量解析器模块
 * 支持 JSON 和 compact 两种格式的流式增量解析
 */

import { tryParseJSON, computeHashId } from './utils.js'

/**
 * 在流中增量解析 infons 数组，逐个对象产出
 * Auto-detects format (compact vs JSON) and uses appropriate parser
 * @param {string} streamText - 流式文本
 * @param {object} parser - 解析器状态
 * @returns {{state: object, yielded: Array}} 更新后的状态和产出的 infon 列表
 */
export async function incrementalExtractInfons(streamText, parser) {
  const text = String(streamText || '')
  
  // Auto-detect format on first call
  if (!parser || !parser.formatDetected) {
    // Look for compact format patterns (宽松匹配):
    // 1. Header: infons[N]:
    // 2. CSV lines: desc:xxx,DESC,... or similar
    // 3. Any line containing desc:/scen:/rel: followed by comma
    const compactHeaderMatch = text.match(/infons\[\d+\]:/)
    const compactDataMatch = text.match(/(^|\n)\s*[-*\d.)\s]*(desc|scen|rel):\w+,/im)
    const compactMatch = compactHeaderMatch || compactDataMatch
    
    // Look for JSON format
    const jsonMatch = text.match(/"infons"\s*:\s*\[/)
    
    // Determine format based on which appears first
    let useCompact = false
    if (compactMatch && jsonMatch) {
      const compactIdx = compactMatch.index || 0
      const jsonIdx = jsonMatch.index || 0
      useCompact = compactIdx < jsonIdx
    } else if (compactMatch) {
      useCompact = true
    } else if (jsonMatch) {
      useCompact = false
    } else {
      // 更宽松的检测：任何包含 desc: 或 scen: 或 rel: 的内容
      const looseMatch = text.match(/(desc|scen|rel):/i)
      if (looseMatch) {
        useCompact = true
      } else if (text.length > 100) {
        // 内容足够多但没检测到格式，默认使用 compact
        useCompact = true
      } else {
        // 内容太少，等待更多
        return { state: parser || { formatDetected: false }, yielded: [] }
      }
    }
    
    // Initialize parser state with format info
    parser = {
      ...(parser || {}),
      formatDetected: true,
      isCompact: useCompact
    }
  }
  
  // Route to appropriate parser
  if (parser.isCompact) {
    // Import and use compact parser
    const { incrementalExtractInfonsCompact } = await import('../templates/infons.js')
    return incrementalExtractInfonsCompact(text, parser)
  } else {
    return incrementalExtractInfonsJSON(text, parser)
  }
}

/**
 * JSON 格式的增量解析器
 * @param {string} streamText - 流式文本
 * @param {object} parser - 解析器状态
 * @returns {{state: object, yielded: Array}} 更新后的状态和产出的 infon 列表
 */
export function incrementalExtractInfonsJSON(streamText, parser) {
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

  // 若尚未定位到 infons 数组，先查找
  if (!state.foundArray) {
    const m = /"infons"\s*:\s*\[/.exec(text)
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

  // 当数组已经关闭则不再扫描
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
          // 对象完整闭合
          let objText = text.slice(objStart, i + 1)
          objText = objText.trim()
          
          if (!objText.endsWith('}')) {
            const lastBrace = objText.lastIndexOf('}')
            if (lastBrace >= 0) {
              objText = objText.slice(0, lastBrace + 1)
            }
          }
          
          const hash = computeHashId(objText)
          if (!state.objectStates.get(state.currentObjIndex)?.lastParsedHash || 
              state.objectStates.get(state.currentObjIndex).lastParsedHash !== hash) {
            const { ok, value } = tryParseJSON(objText)
            if (ok) {
              yielded.push({ ...value, _objIndex: state.currentObjIndex, _isComplete: true })
              const objState = state.objectStates.get(state.currentObjIndex)
              if (objState) {
                objState.data = value
                objState.lastParsedHash = hash
              }
            } else {
              // 解析失败时使用部分数据
              const objState = state.objectStates.get(state.currentObjIndex)
              if (objState && Object.keys(objState.data).length > 0) {
                yielded.push({ ...objState.data, _objIndex: state.currentObjIndex, _isComplete: true })
                objState.lastParsedHash = hash
              }
            }
          }
          objStart = -1
          state.currentObjIndex++
        }
      }
      continue
    }
    if (ch === ']') {
      // 数组关闭（仅当当前不在对象中）
      if (objStart < 0) { state.closed = true; i++ ; break }
    }
    
    // 尝试部分解析（每隔一定字符数或遇到特定标记时）
    if (objStart >= 0 && braceDepth > 0) {
      const objText = text.slice(objStart, i + 1)
      // 当累积了足够多的内容时，尝试部分解析
      if ((ch === ',' || ch === '\n') && (i - objStart) > 20) {
        const objState = state.objectStates.get(state.currentObjIndex)
        const currentHash = computeHashId(objText)
        
        // 只有当内容有变化时才解析
        if (objState && objState.lastParsedHash !== currentHash) {
          const partialData = parsePartialInfon(objText)
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

/**
 * 解析部分 infon 对象
 * @param {string} objText - 部分 JSON 文本
 * @returns {object} 解析出的字段
 */
export function parsePartialInfon(objText) {
  const result = {}
  
  // 关键字段优先提取
  const criticalFields = ['iid', 'infon_type', 'entity', 'attribute', 'temporal', 'spatial']
  const otherFields = ['data_type', 'relation_name', 'arity', 'arg_refs', 'description', 'confidence', 'bbox']
  
  for (const field of [...criticalFields, ...otherFields]) {
    const value = extractInfonFieldValue(objText, field)
    if (value !== null) {
      result[field] = value
    }
  }
  
  return result
}

/**
 * 从部分 JSON 文本中提取字段值
 * @param {string} text - JSON 文本
 * @param {string} fieldName - 字段名
 * @returns {any} 字段值或 null
 */
export function extractInfonFieldValue(text, fieldName) {
  const patterns = [
    // 字符串值
    new RegExp(`"${fieldName}"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"`, 's'),
    // 数字值
    new RegExp(`"${fieldName}"\\s*:\\s*(\\d+\\.?\\d*)`, 's'),
    // 布尔值
    new RegExp(`"${fieldName}"\\s*:\\s*(true|false)`, 's'),
    // 数组值（简单处理）
    new RegExp(`"${fieldName}"\\s*:\\s*(\\[[^\\]]*\\])`, 's'),
  ]
  
  for (const pattern of patterns) {
    const match = text.match(pattern)
    if (match) {
      try {
        // 字符串
        if (pattern.source.includes('"([^"]*')) {
          return match[1].replace(/\\"/g, '"').replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
        }
        // 数字、布尔、数组
        return JSON.parse(match[1])
      } catch (err) {
        continue
      }
    }
  }
  
  return null
}
