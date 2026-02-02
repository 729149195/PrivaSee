/**
 * 隐私推理辅助函数
 * 提供隐私风险分析的流式解析逻辑
 */

import { tryParseJSON, extractFirstJSONObject } from '../utils.js'

/**
 * 收集提取信息元模式的信息元
 */
export function collectInfonsForInference(infonSessions, sessionId) {
  const runs = infonSessions?.[sessionId]?.runs || []
  const allRawInfons = []
  const supersededIids = new Set()
  
  // 第一遍：收集所有信息元和被取代的iid
  runs.forEach(run => {
    if (run.status === 'done' || run.status === 'running') {
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      allRawInfons.push(...infons)
      infons.forEach(infon => {
        if (Array.isArray(infon._supersedes)) {
          infon._supersedes.forEach(oldIid => supersededIids.add(oldIid))
        }
      })
    }
  })
  
  // 第二遍：过滤掉被取代的信息元
  return allRawInfons.filter(infon => infon.iid && !supersededIids.has(infon.iid))
}

// ============== 隐私风险解析器 ==============

/**
 * 创建隐私风险增量解析器
 */
export function createPrivacyRiskParser() {
  return {
    foundArray: false,
    arrayStart: -1,
    scanPos: 0,
    inString: false,
    escape: false,
    objStart: -1,
    braceDepth: 0,
    closed: false,
    objectStates: new Map(),
    currentObjIndex: 0,
  }
}

/**
 * 增量解析隐私风险数组
 */
export function incrementalExtractRisks(streamText, parser) {
  const state = parser || createPrivacyRiskParser()
  const yielded = []
  const text = String(streamText || '')

  // 查找 risks 数组
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
            if (lastBrace >= 0) objText = objText.slice(0, lastBrace + 1)
          }
          
          const { ok, value } = tryParseJSON(objText)
          if (ok) {
            yielded.push({ ...value, _objIndex: state.currentObjIndex, _isComplete: true })
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
  }

  state.inString = inString
  state.escape = escape
  state.objStart = objStart
  state.braceDepth = braceDepth
  state.scanPos = i
  return { state, yielded }
}

/**
 * 从风险项中提取关键词
 */
export function extractKeywordsFromRisks(risks) {
  const keywords = new Set()
  
  risks.forEach(risk => {
    // 提取 used_infons 作为关键词
    const usedInfons = risk.used_infons || []
    usedInfons.forEach(keyword => {
      if (typeof keyword === 'string' && keyword.trim()) {
        keywords.add(keyword.trim())
      }
    })
  })
  
  return keywords
}

/**
 * 清理 buffer 并解析完整 JSON
 */
export function parsePrivacyBuffer(buffer) {
  let cleanBuffer = buffer
  
  // 移除 <think>...</think> 标签
  cleanBuffer = cleanBuffer.replace(/<think>[\s\S]*?<\/think>/gi, '')
  
  // 移除 markdown 代码块标记
  cleanBuffer = cleanBuffer.replace(/^```json\s*/i, '').replace(/```\s*$/i, '')
  cleanBuffer = cleanBuffer.replace(/^```\s*/i, '').replace(/```\s*$/i, '')
  
  // 提取 JSON 部分
  const firstBrace = cleanBuffer.indexOf('{')
  const lastBrace = cleanBuffer.lastIndexOf('}')
  
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    cleanBuffer = cleanBuffer.slice(firstBrace, lastBrace + 1)
    
    const { ok, value } = tryParseJSON(cleanBuffer)
    if (ok && value.risks && Array.isArray(value.risks)) {
      return { success: true, risks: value.risks, cleanBuffer }
    }
  }
  
  return { success: false, risks: [], cleanBuffer }
}

/**
 * 构建隐私推理的系统提示词
 */
export function buildPrivacyInferencePrompt({ infons, lawData, selectedPrivacyItems, customPrivacyItems }) {
  const privacyCategories = selectedPrivacyItems.length > 0 
    ? [...selectedPrivacyItems, ...customPrivacyItems.map(item => item.name)]
    : ['姓名', '身份证件号码', '电话号码', '地址', '生物识别信息']
  
  const categoryList = privacyCategories.map((c, i) => `${i + 1}. ${c}`).join('\n')
  
  const infonsList = infons.map((inf, i) => {
    const type = inf.infon_type || 'unknown'
    if (type === 'DESC') {
      return `[${i + 1}] ${inf.entity}: ${inf.attribute}`
    } else if (type === 'REL') {
      return `[${i + 1}] ${inf.relation_name}: ${(inf.arg_refs || []).join(' → ')}`
    } else if (type === 'SCEN') {
      return `[${i + 1}] 场景: ${inf.description || inf.spatial || inf.temporal}`
    }
    return `[${i + 1}] ${JSON.stringify(inf)}`
  }).join('\n')
  const inputDescription = `提取的信息元:\n${infonsList}`
  
  return `你是一个隐私风险分析专家。请分析以下内容中可能存在的隐私风险。

${inputDescription}

请根据以下隐私类别进行分析:
${categoryList}

${lawData ? `参考法律法规: ${lawData.name || '个人信息保护法'}` : ''}

请以 JSON 格式输出分析结果:
{
  "risks": [
    {
      "category": "隐私类别名称",
      "risk_level": "high/medium/low",
      "description": "风险描述",
      "used_infons": ["相关的关键信息词"]
    }
  ]
}`
}

// ============== 隐私推理状态更新辅助函数 ==============

/**
 * 合并新解析的 risks 到现有列表（基于 _objIndex 去重）
 */
export function mergeRisks(currentRisks, newRisks) {
  const updated = [...currentRisks]
  newRisks.forEach(newRisk => {
    const objIndex = newRisk._objIndex
    if (objIndex !== undefined) {
      const existingIndex = updated.findIndex(r => r._objIndex === objIndex)
      if (existingIndex >= 0) {
        updated[existingIndex] = { ...updated[existingIndex], ...newRisk }
      } else {
        updated.push(newRisk)
      }
    } else {
      updated.push(newRisk)
    }
  })
  return updated
}

/**
 * 清理 buffer 并尝试解析完整 JSON（处理 think 标签和 markdown）
 */
export function cleanAndParseBuffer(buffer) {
  let cleanBuffer = buffer
  
  // 移除 <think>...</think> 标签
  cleanBuffer = cleanBuffer.replace(/<think>[\s\S]*?<\/think>/gi, '')
  
  // 移除 markdown 代码块标记
  cleanBuffer = cleanBuffer.replace(/^```json\s*/i, '').replace(/```\s*$/i, '')
  cleanBuffer = cleanBuffer.replace(/^```\s*/i, '').replace(/```\s*$/i, '')
  
  // 提取 JSON 部分
  const firstBrace = cleanBuffer.indexOf('{')
  const lastBrace = cleanBuffer.lastIndexOf('}')
  
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    cleanBuffer = cleanBuffer.slice(firstBrace, lastBrace + 1)
    const { ok, value } = tryParseJSON(cleanBuffer)
    if (ok && value.risks && Array.isArray(value.risks)) {
      return { success: true, risks: value.risks, cleanBuffer, isCompact: false }
    }
  }
  
  // 检查是否为紧凑格式
  const isCompact = buffer.indexOf('{') === -1
  return { success: false, risks: [], cleanBuffer, isCompact }
}

/**
 * 处理流式 SSE 数据行，提取内容增量
 */
export function parseSSELine(line) {
  if (!line.trim() || !line.startsWith('data: ')) return null
  const data = line.slice(6).trim()
  if (data === '[DONE]') return null
  
  try {
    const parsed = JSON.parse(data)
    const delta = parsed?.choices?.[0]?.delta || {}
    return delta?.content || delta?.reasoning_content || delta?.reasoning || delta?.thoughts || delta?.inner_thoughts || ''
  } catch {
    return null
  }
}
