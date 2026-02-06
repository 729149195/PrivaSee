/**
 * Store 基础工具函数模块
 * 包含ID生成、JSON解析、哈希计算等通用工具
 */

// 生成短随机 ID
export function generateId() {
  return Math.random().toString(36).slice(2, 10)
}

// 尝试解析 JSON，返回 {ok, value, error}
export function tryParseJSON(text) {
  try {
    return { ok: true, value: JSON.parse(text), error: null }
  } catch (e) {
    return { ok: false, value: null, error: e }
  }
}

// 从原始文本中提取第一个完整 JSON 对象
export function extractFirstJSONObject(text) {
  const start = text.indexOf('{')
  if (start < 0) return null
  let depth = 0, inStr = false, escape = false
  for (let i = start; i < text.length; i++) {
    const ch = text[i]
    if (inStr) {
      if (escape) { escape = false; continue }
      if (ch === '\\') { escape = true; continue }
      if (ch === '"') { inStr = false }
      continue
    }
    if (ch === '"') { inStr = true; continue }
    if (ch === '{') { depth++ }
    else if (ch === '}') {
      depth--
      if (depth === 0) return text.slice(start, i + 1)
    }
  }
  return null
}

// 简易哈希（用于去重比较）
export function computeHashId(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const chr = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + chr
    hash |= 0
  }
  return hash.toString(36)
}

// 规范化 infon 输出结构
export function normalizeInfonOutput(raw, { recordTimeISO, defaultModality, sessionId, messageRound, infonIndex, infonType }) {
  // 确保 raw 是对象
  if (!raw || typeof raw !== 'object') {
    return { infons: [] }
  }
  
  // 获取 infons 数组
  let infons = []
  if (Array.isArray(raw.infons)) {
    infons = raw.infons
  } else if (Array.isArray(raw)) {
    infons = raw
  }
  
  // 规范化每个 infon
  infons = infons.map((infon, idx) => {
    const normalized = { ...infon }
    
    // 确保 iid 存在
    if (!normalized.iid) {
      normalized.iid = `${sessionId || 'S'}_R${messageRound || 1}_${infonType || 'desc'}_${infonIndex + idx}`
    }
    
    // 确保 run_metadata 存在
    if (!normalized.run_metadata) {
      normalized.run_metadata = {}
    }
    
    // 设置 record_time
    if (!normalized.run_metadata.record_time) {
      normalized.run_metadata.record_time = recordTimeISO
    }
    
    // 设置 modality
    if (!normalized.run_metadata.modality) {
      normalized.run_metadata.modality = defaultModality || 'text'
    }
    
    // === 主记忆流扩展字段 ===
    
    // 模态标签 (区分来源: text | image | audio)
    if (!normalized.modality_tag) {
      normalized.modality_tag = defaultModality || 'text'
    }
    
    // 证据指针 (用于回溯到原始输入位置, 由后端 memory stream 填充)
    if (!normalized.evidence_pointer) {
      normalized.evidence_pointer = null
    }
    
    // 语义向量 (由后端计算, 前端不存储)
    // normalized.semantic_vector = null
    
    // 关联信息元列表 (由后端 memory stream ingest 时的 Top-K 绑定填充)
    if (!normalized.associations) {
      normalized.associations = []
    }
    
    // 会话标识和轮次编号 (用于证据指针的构建)
    if (!normalized.session_id) {
      normalized.session_id = sessionId || ''
    }
    if (normalized.round_num === undefined) {
      normalized.round_num = messageRound || 1
    }
    
    return normalized
  })
  
  return { infons }
}

// 新建会话：创建一个空消息会话，标题默认 "New chat"
export const createEmptySession = () => ({
  id: generateId(),
  title: 'New chat',
  createdAt: Date.now(),
  updatedAt: Date.now(),
  messages: [], // {id, role: 'user'|'assistant'|'system', content, createdAt, streaming?, error?}
})
