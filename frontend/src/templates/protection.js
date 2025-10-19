// Privacy Protection Suggestions Prompt Template
// 隐私保护修改建议提示词模板

export const PROTECTION_SUGGESTIONS_PROMPT = `
您是一位隐私保护专家，专门为用户提供文本修改建议，帮助在隐私保护和模型效用之间找到平衡。

## 任务
基于以下输入，提供3种不同级别的隐私保护修改建议：
1. **高隐私保护（低效用）**：最大程度保护隐私，可能显著降低模型理解和响应质量
2. **平衡方案（中等效用）**：在隐私保护和模型效用之间取得平衡
3. **低隐私保护（高效用）**：最小程度修改，保持模型效用，仅移除最敏感信息

## 输入数据

### 1. 原始文本
{{ORIGINAL_TEXT}}

### 2. 检测到的隐私风险
{{PRIVACY_RISKS}}

### 3. 检测到的信息元
{{INFONS}}

## 修改原则

### 高隐私保护策略
- 移除或替换所有可识别个人的信息
- 泛化所有具体细节（时间、地点、数字等）
- 使用抽象描述替代具体内容
- 优先考虑匿名性，即使损失信息完整性

### 平衡策略
- 保留必要的上下文信息
- 对敏感信息进行适度泛化
- 保持查询的核心意图
- 在隐私和可用性间权衡

### 低隐私保护策略
- 仅移除最敏感的个人信息（如身份证号、真实姓名等）
- 保留大部分上下文细节
- 最小化修改，保持原意
- 优先保证模型理解准确性

## 输出格式

**流式渲染优化**: 为了实现流畅的流式显示效果，请按以下顺序输出每个建议的字段：
1. 先输出 level 和 label（用于快速识别）
2. 再输出 modified_text（主要内容，将会逐字显示）
3. 最后输出 changes_summary 和 removed_risks

输出**仅包含有效JSON**（无markdown代码块，无额外说明）：

{
  "suggestions": [
    {
      "level": "high_privacy",
      "label": "高隐私保护",
      "modified_text": "修改后的文本内容",
      "changes_summary": "简要说明做了哪些修改，为什么这样修改",
      "removed_risks": ["列出移除了哪些隐私风险"]
    },
    {
      "level": "balanced",
      "label": "平衡方案",
      "modified_text": "修改后的文本内容",
      "changes_summary": "简要说明做了哪些修改",
      "removed_risks": ["列出移除了哪些隐私风险"]
    },
    {
      "level": "low_privacy",
      "label": "低隐私保护",
      "modified_text": "修改后的文本内容",
      "changes_summary": "简要说明做了哪些修改",
      "removed_risks": ["列出移除了哪些隐私风险"]
    }
  ]
}

## 关键要求
1. **仅输出JSON** - 不要使用markdown代码块，不要添加任何解释文字
2. **保持语言一致** - 如果输入是中文，修改后的文本也必须是中文；英文同理
3. **修改要具体** - modified_text必须是完整的、可直接使用的文本
4. **说明要清晰** - changes_summary要简洁说明修改了什么，为什么
5. **按顺序输出** - 必须按high_privacy → balanced → low_privacy顺序，且字段顺序为level → label → modified_text → changes_summary → removed_risks
6. **保持完整性** - 即使高隐私保护方案也要确保文本可读、有意义

## 示例

输入：
"我叫张三，身份证号123456789012345678，住在北京市朝阳区某小区，想咨询一下我的糖尿病饮食建议"

输出应该包含三个级别的建议，例如：
- 高隐私：移除所有个人信息，泛化健康状况
- 平衡：保留健康状况但泛化位置和身份
- 低隐私：仅移除身份证号和详细地址

现在请分析输入并输出完整的JSON建议。
`

/**
 * 填充保护建议提示词模板
 * @param {string} originalText - 原始文本
 * @param {Array} privacyRisks - 隐私风险列表
 * @param {Array} infons - 信息元列表
 * @returns {string} 填充后的提示词
 */
export function fillProtectionPrompt(originalText, privacyRisks, infons) {
  // 格式化隐私风险
  let risksText = '未检测到隐私风险'
  if (Array.isArray(privacyRisks) && privacyRisks.length > 0) {
    risksText = privacyRisks.map((risk, idx) => {
      const level = risk.risk_level || 'UNKNOWN'
      const lawNode = risk.law_node_name || '未知'
      const exposure = risk.privacy_exposure || '未知'
      return `${idx + 1}. [${level}] ${lawNode}: ${exposure}`
    }).join('\n')
  }

  // 格式化信息元
  let infonsText = '未检测到信息元'
  if (Array.isArray(infons) && infons.length > 0) {
    infonsText = infons.map((infon, idx) => {
      const type = String(infon.infon_type || '').toUpperCase()
      const iid = infon.iid || `infon_${idx}`
      
      let detail = ''
      if (type === 'DESC') {
        const entity = infon.entity || ''
        const attribute = infon.attribute || ''
        detail = `${entity}: ${attribute}`
      } else if (type === 'SCEN') {
        const temporal = infon.temporal || ''
        const spatial = infon.spatial || ''
        detail = `${temporal} @ ${spatial}`
      } else if (type === 'REL') {
        detail = infon.relation_name || '关系'
      } else if (type === 'SIT') {
        detail = infon.description || '情境'
      }
      
      return `- [${iid}] ${type}: ${detail}`
    }).join('\n')
  }

  // 填充模板
  const prompt = PROTECTION_SUGGESTIONS_PROMPT
    .replace('{{ORIGINAL_TEXT}}', originalText || '')
    .replace('{{PRIVACY_RISKS}}', risksText)
    .replace('{{INFONS}}', infonsText)

  return prompt
}

/**
 * 解析保护建议响应
 * @param {string} responseText - API响应文本
 * @returns {Object|null} 解析后的建议对象
 */
export function parseProtectionResponse(responseText) {
  try {
    // 尝试直接解析
    const parsed = JSON.parse(responseText)
    if (parsed && parsed.suggestions && Array.isArray(parsed.suggestions)) {
      return parsed
    }
  } catch (e) {
    // 如果直接解析失败，尝试提取JSON对象
    const jsonMatch = responseText.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[0])
        if (parsed && parsed.suggestions && Array.isArray(parsed.suggestions)) {
          return parsed
        }
      } catch (e2) {
        console.warn('[Protection] 无法解析JSON响应', e2)
      }
    }
  }
  
  return null
}

/**
 * 验证建议数据的完整性
 * @param {Object} suggestion - 单个建议对象
 * @returns {boolean} 是否有效
 */
export function validateSuggestion(suggestion) {
  if (!suggestion || typeof suggestion !== 'object') return false
  
  const required = ['level', 'label', 'modified_text']
  return required.every(field => suggestion[field] !== undefined && suggestion[field] !== null)
}

/**
 * 流式增量解析保护建议
 * @param {string} streamText - 流式接收的文本
 * @param {object} parser - 解析器状态
 * @returns {object} { state, yielded }
 */
export function incrementalExtractSuggestions(streamText, parser) {
  const state = parser || {
    foundArray: false,
    arrayStart: -1,
    scanPos: 0,
    inString: false,
    escape: false,
    objStart: -1,
    braceDepth: 0,
    closed: false,
    objectStates: new Map(),
    currentObjIndex: 0
  }
  const yielded = []
  const text = String(streamText || '')

  // 查找 "suggestions" 数组
  if (!state.foundArray) {
    const m = /"suggestions"\s*:\s*\[/.exec(text)
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
          let objText = text.slice(objStart, i + 1).trim()
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
            const objState = state.objectStates.get(state.currentObjIndex)
            if (objState && Object.keys(objState.data).length > 0) {
              yielded.push({ ...objState.data, _objIndex: state.currentObjIndex, _isComplete: true })
              objState.lastParsedHash = computeHashId(objText)
            }
          }
          objStart = -1
          state.currentObjIndex++
        }
      }
      continue
    }
    if (ch === ']') {
      if (objStart < 0) { state.closed = true; i++; break }
    }
    
    // 部分解析
    if (objStart >= 0 && braceDepth > 0) {
      const objText = text.slice(objStart, i + 1)
      if ((ch === ',' || ch === '\n') && (i - objStart) > 20) {
        const objState = state.objectStates.get(state.currentObjIndex)
        const currentHash = computeHashId(objText)
        
        if (objState && objState.lastParsedHash !== currentHash) {
          const partialData = parsePartialSuggestion(objText)
          if (partialData && Object.keys(partialData).length > 0) {
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
 * 解析部分建议对象
 */
function parsePartialSuggestion(objText) {
  const result = {}
  const criticalFields = ['level', 'label']
  const otherFields = ['modified_text', 'changes_summary', 'removed_risks']
  
  for (const field of [...criticalFields, ...otherFields]) {
    const value = extractFieldValue(objText, field)
    if (value !== null) {
      result[field] = value
    }
  }
  
  return result
}

/**
 * 提取字段值
 */
function extractFieldValue(text, fieldName) {
  const patterns = [
    new RegExp(`"${fieldName}"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"`, 's'),
    new RegExp(`"${fieldName}"\\s*:\\s*(\\d+\\.?\\d*)`, 's'),
    new RegExp(`"${fieldName}"\\s*:\\s*(\\[[^\\]]*\\])`, 's'),
    new RegExp(`"${fieldName}"\\s*:\\s*(\\{[^}]*\\})`, 's')
  ]
  
  for (const pattern of patterns) {
    const match = text.match(pattern)
    if (match) {
      try {
        if (pattern.source.includes('"([^"]*')) {
          return match[1].replace(/\\"/g, '"').replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
        }
        return JSON.parse(match[1])
      } catch (err) {
        continue
      }
    }
  }
  
  return null
}

/**
 * 计算简单哈希ID
 */
function computeHashId(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash
  }
  return hash.toString(36)
}

