import { create } from 'zustand'
import { buildSystemPrompt } from './templates/infons.js'

// 说明（中文注释）：
// 1) 本 store 管理 ChatGPT 风格的多会话、消息流与流式生成状态；
// 2) 采用 OpenAI Chat Completions 协议与本地 Ollama(OpenAI 兼容)接口对接；
// 3) 所有 UI 文本在代码中使用英文，注释使用中文；
// 4) 每次对话上下文从当前会话的全部消息构建，以实现连续对话记忆。

// 生成唯一 ID（中文注释）：优先使用浏览器 crypto.randomUUID，降级到时间戳
const generateId = () => {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  } catch (_) { }
  return 'id_' + Date.now() + '_' + Math.random().toString(16).slice(2)
}

// 安全 JSON 解析（中文注释）
function tryParseJSON(text) {
  try {
    return { ok: true, value: JSON.parse(text) }
  } catch (_) {
    return { ok: false, value: null }
  }
}

// 从文本中提取首个完整 JSON 对象（中文注释）：忽略 JSON 内部字符串中的大括号
function extractFirstJSONObject(text) {
  if (typeof text !== 'string' || !text) return null
  const start = text.indexOf('{')
  if (start < 0) return null
  let depth = 0
  let inString = false
  let escape = false
  for (let i = start; i < text.length; i++) {
    const ch = text[i]
    if (inString) {
      if (escape) { escape = false; continue }
      if (ch === '\\') { escape = true; continue }
      if (ch === '"') { inString = false; continue }
    } else {
      if (ch === '"') { inString = true; continue }
      if (ch === '{') depth++
      if (ch === '}') {
        depth--
        if (depth === 0) {
          return text.slice(start, i + 1)
        }
      }
    }
  }
  return null
}

// 简单稳定哈希（中文注释）：用于文本/图片去重
function computeHashId(input) {
  const s = String(input || '')
  let h = 5381
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) + h) + s.charCodeAt(i)
    h = h >>> 0
  }
  return 'h' + h.toString(16)
}

// 规范化模型输出：确保符合 OUTPUT_FORMAT，并填充 record_time（中文注释）
function normalizeInfonOutput(obj, { recordTimeISO, defaultModality, sessionId, messageRound, infonIndex, infonType }) {
  const out = (obj && typeof obj === 'object') ? obj : {}
  const now = recordTimeISO || new Date().toISOString()
  out.run_metadata = out.run_metadata && typeof out.run_metadata === 'object' ? out.run_metadata : {}
  if (!out.run_metadata.record_time) out.run_metadata.record_time = now
  if (!out.run_metadata.source_id) out.run_metadata.source_id = 'unknown'
  if (!out.run_metadata.generator) out.run_metadata.generator = 'infon_extractor'
  out.situations = Array.isArray(out.situations) ? out.situations : []
  out.entities = Array.isArray(out.entities) ? out.entities : []
  out.infons = Array.isArray(out.infons) ? out.infons : []
  out.quality_report = out.quality_report && typeof out.quality_report === 'object' ? out.quality_report : { stats: {} }
  // 填充每个 situation/infons 的 record_time 与 modality 缺省（中文注释）
  out.situations = out.situations.map((s) => {
    const t = { ...(typeof s === 'object' ? s : {}) }
    if (!t.record_time) t.record_time = now
    if (!t.modality && defaultModality) t.modality = defaultModality
    return t
  })
  out.infons = out.infons.map((i, index) => {
    const t = { ...(typeof i === 'object' ? i : {}) }
    if (!t.record_time) t.record_time = now
    // 生成基于对话轮次和信息元次序的iid（中文注释）
    if (!t.iid && infonType) {
      const typePrefix = infonType.toLowerCase().slice(0, 3) // 取前三个字母作为前缀
      const round = messageRound || 1
      const idx = (infonIndex || 0) + index + 1
      t.iid = `${typePrefix}:r${round}_${idx}`
    }
    return t
  })
  return out
}

function buildInfonSystemPrompt(modalities, nowISO) {
  return buildSystemPrompt({
    modalities,
    includeExamples: false,
    extraInstructions: `System time (ISO8601) = ${nowISO}. Set run_metadata.record_time to this value. For each situation and infon, if record_time is missing, set it to this value. Only set occur_time when it is explicitly expressed; otherwise omit.`
  })
}

// 在流中增量解析 infons 数组，逐个对象产出（中文注释）
function incrementalExtractInfons(streamText, parser) {
  const state = parser || { foundArray: false, arrayStart: -1, scanPos: 0, inString: false, escape: false, objStart: -1, braceDepth: 0, closed: false, yieldedHashes: [] }
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
      if (objStart < 0) { objStart = i; braceDepth = 1 } else { braceDepth++ }
      continue
    }
    if (ch === '}') {
      if (objStart >= 0) {
        braceDepth--
        if (braceDepth === 0) {
          const objText = text.slice(objStart, i + 1)
          const hash = computeHashId(objText)
          if (!state.yieldedHashes.includes(hash)) {
            const { ok, value } = tryParseJSON(objText)
            if (ok) {
              yielded.push(value)
              state.yieldedHashes = [...state.yieldedHashes, hash]
            }
          }
          objStart = -1
        }
      }
      continue
    }
    if (ch === ']') {
      // 数组关闭（仅当当前不在对象中）
      if (objStart < 0) { state.closed = true; i++ ; break }
    }
  }

  state.inString = inString
  state.escape = escape
  state.objStart = objStart
  state.braceDepth = braceDepth
  state.scanPos = i
  return { state, yielded }
}

// 新建会话（中文注释）：创建一个空消息会话，标题默认 "New chat"
const createEmptySession = () => ({
  id: generateId(),
  title: 'New chat',
  createdAt: Date.now(),
  updatedAt: Date.now(),
  messages: [], // {id, role: 'user'|'assistant'|'system', content, createdAt, streaming?, error?}
})

// 解析 OpenAI SSE 流（中文注释）：将 response.body 按行解析 data: 片段
async function streamOpenAIResponse(reader, onDelta) {
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop() || ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      if (trimmed.startsWith('data:')) {
        const payload = trimmed.slice('data:'.length).trim()
        if (payload === '[DONE]') return
        try {
          const json = JSON.parse(payload)
          // OpenAI Chat 模式：choices[0].delta.content 为增量
          const choice = json?.choices?.[0]
          const contentDelta = choice?.delta?.content ?? ''
          // 兼容多种字段名：reasoning_content / reasoning / thoughts / inner_thoughts
          const reasoningDelta = (
            choice?.delta?.reasoning_content ??
            choice?.delta?.reasoning ??
            choice?.delta?.thoughts ??
            choice?.delta?.inner_thoughts ??
            ''
          )
          const finish = choice?.finish_reason || null
          if (contentDelta || reasoningDelta) onDelta({ content: contentDelta, reasoning: reasoningDelta, finish: null })
          if (finish) onDelta({ content: '', reasoning: '', finish })
        } catch (_) {
          // 忽略不可解析的行
        }
      }
    }
  }
}

// 解析 Ollama /api/chat 流（中文注释）：逐行解析 JSON，并处理“全量快照”或“增量 token”两种格式
async function streamOllamaChatResponse(reader, onDelta) {
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let accumulated = '' // 用于处理返回全量快照时的去重
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop() || ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        const json = JSON.parse(trimmed)
        // chat 流常见字段：message.content；fallback 到 generate 的 response
        const nextFull = (
          (typeof json?.message?.content === 'string' ? json.message.content : '') ||
          (typeof json?.response === 'string' ? json.response : '')
        )
        const finish = json?.done ? 'stop' : null
        if (nextFull) {
          let delta = nextFull
          if (nextFull.startsWith(accumulated)) {
            delta = nextFull.slice(accumulated.length)
          }
          accumulated = nextFull
          if (delta) onDelta({ content: delta, reasoning: '', finish: null })
        }
        if (finish) onDelta({ content: '', reasoning: '', finish })
      } catch (_) {
        // 忽略不可解析的行
      }
    }
  }
}

export const useStore = create((set, get) => ({
  // 基础配置（中文注释）：指向本地 Ollama OpenAI 兼容接口
  baseUrl: '/v1',
  model: 'gemma3:12b',
  models: [], // 可选模型列表（中文注释）
  customModels: [], // 通过 API key 添加的自定义模型（中文注释）
  customProviders: {}, // { [modelId]: { baseUrl, apiKey } }（中文注释）

  // 多会话与状态（中文注释）：初始化一个空会话
  sessions: (() => {
    const s = createEmptySession()
    return [s]
  })(),
  currentSessionId: null,
  isGenerating: false,
  abortController: null,

  // 信息元提取：按会话维护运行列表（中文注释）
  // infonSessions: { [sessionId]: { runs: Array<Run> } }
  // Run: { id, targetType: 'pending'|'message', targetKey: string, modality: 'text'|'image', imageIndex?, status: 'running'|'done'|'aborted'|'error', progress: number, buffer: string, resultJson: any|null, error?: string, createdAt }
  infonSessions: {},
  // 缓存：上次提取的文本与每张图片的哈希（中文注释）
  lastPendingTextHash: null,
  lastPendingImageHashes: [],
  // 流式增量解析器状态：按 runId 维护（中文注释）
  infonParsers: {},

  // 初始化当前会话（中文注释）：第一次使用时指向首个会话
  _ensureCurrentSession() {
    const { sessions, currentSessionId } = get()
    if (!currentSessionId && sessions.length > 0) {
      set({ currentSessionId: sessions[0].id })
    }
  },

  // 读取当前会话（中文注释）：找不到则返回 null
  getCurrentSession() {
    const { sessions, currentSessionId } = get()
    return sessions.find(s => s.id === currentSessionId) || null
  },

  // 内部：获取或创建当前会话的信息元会话容器（中文注释）
  _getOrCreateInfonSession(sessionId) {
    let box = get().infonSessions?.[sessionId]
    if (!box) {
      box = { runs: [] }
      set((state) => ({ infonSessions: { ...(state.infonSessions || {}), [sessionId]: box } }))
    }
    return box
  },

  // 内部：追加信息元运行（中文注释）
  _appendInfonRun(sessionId, run) {
    set((state) => {
      const current = state.infonSessions?.[sessionId] || { runs: [] }
      const next = { runs: [...current.runs, run] }
      return { infonSessions: { ...(state.infonSessions || {}), [sessionId]: next } }
    })
  },

  // 内部：更新信息元运行（中文注释）
  _updateInfonRun(sessionId, runId, updater) {
    set((state) => {
      const box = state.infonSessions?.[sessionId]
      if (!box) return {}
      const runs = box.runs.map(r => r.id === runId ? updater(r) : r)
      return { infonSessions: { ...state.infonSessions, [sessionId]: { runs } } }
    })
  },

  // 读取当前会话的所有信息元运行（中文注释）
  getCurrentInfonRuns() {
    const session = get().getCurrentSession()
    if (!session) return []
    return (get().infonSessions?.[session.id]?.runs) || []
  },

  // 清空所有 pending 信息元（中文注释）：供组件发送前调用
  clearAllPendingInfons() {
    const session = get().getCurrentSession()
    if (!session) return
    try {
      const runs = (get().infonSessions?.[session.id]?.runs) || []
      runs.forEach((r) => {
        if (r.targetType === 'pending' && r.status === 'running') {
          try { r.controller?.abort?.() } catch (_) {}
        }
      })
    } catch (_) {}
    set((state) => {
      const box = state.infonSessions?.[session.id] || { runs: [] }
      const nextRuns = box.runs.filter(r => r.targetType !== 'pending')
      return { infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
  },

  // 采纳当前会话的所有 pending 结果为指定 message 的结果（中文注释）
  // 将 targetType: 'pending' 的 run 改写为 targetType: 'message'，targetKey=messageId
  // 返回被采纳的数量
  adoptPendingInfonsToMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session || !messageId) return 0
    let adopted = 0
    set((state) => {
      const box = state.infonSessions?.[session.id]
      if (!box) return {}
      const runs = box.runs.map((r) => {
        if (r.targetType === 'pending') {
          adopted++
          return { ...r, targetType: 'message', targetKey: messageId }
        }
        return r
      })
      return { infonSessions: { ...state.infonSessions, [session.id]: { runs } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
    return adopted
  },

  // 创建会话（中文注释）：并切换为当前
  createSession() {
    const newSession = createEmptySession()
    set((state) => ({
      sessions: [newSession, ...state.sessions],
      currentSessionId: newSession.id,
    }))
  },

  // 设置模型（中文注释）
  setModel(modelId) {
    set({ model: modelId })
  },

  // 拉取模型列表（中文注释）：兼容 OpenAI / Ollama 响应结构
  async fetchModels() {
    try {
      const res = await fetch(`${get().baseUrl}/models`, { method: 'GET' })
      const json = await res.json().catch(() => ({}))
      let list = []
      // OpenAI: { data: [{id}...] }
      if (Array.isArray(json?.data)) {
        list = json.data.map((m) => m?.id).filter(Boolean)
      }
      // 某些实现: { models: [...] }
      if (!list.length && Array.isArray(json?.models)) {
        list = json.models.map((m) => m?.id || m?.name || m).filter(Boolean)
      }
      // 兜底：如果直接是数组
      if (!list.length && Array.isArray(json)) {
        list = json.map((m) => m?.id || m?.name || m).filter(Boolean)
      }
      if (list.length) set((state) => ({ models: Array.from(new Set([...(state.models || []), ...list])) }))
    } catch (_) {
      // 忽略错误
    }
  },

  // 添加自定义 API 模型（中文注释）：提供 modelId/baseUrl/apiKey，合并到选择列表
  addApiModel({ id, baseUrl, apiKey }) {
    if (!id || !baseUrl || !apiKey) return
    set((state) => ({
      customProviders: { ...(state.customProviders || {}), [id]: { baseUrl, apiKey } },
      customModels: Array.from(new Set([...(state.customModels || []), id])),
      models: Array.from(new Set([...(state.models || []), id]))
    }))
  },

  // 切换会话（中文注释）
  switchSession(id) {
    set({ currentSessionId: id })
  },

  // 删除会话（中文注释）：如果删除当前会话，则自动切换到剩余第一个
  deleteSession(id) {
    set((state) => {
      const nextSessions = state.sessions.filter(s => s.id !== id)
      let nextCurrent = state.currentSessionId
      if (state.currentSessionId === id) {
        nextCurrent = nextSessions[0]?.id || null
      }
      return { sessions: nextSessions, currentSessionId: nextCurrent }
    })
  },

  // 重命名会话（中文注释）
  renameSession(id, title) {
    set((state) => ({
      sessions: state.sessions.map(s => s.id === id ? { ...s, title, updatedAt: Date.now() } : s),
    }))
  },

  // 追加消息（中文注释）：用于用户或助手消息写入
  _appendMessage(sessionId, message) {
    set((state) => ({
      sessions: state.sessions.map(s => {
        if (s.id !== sessionId) return s
        return {
          ...s,
          updatedAt: Date.now(),
          messages: [...s.messages, message],
        }
      })
    }))
  },

  // 更新某条消息（中文注释）：按消息 id 定位并更新（用于流式增量）
  _updateMessage(sessionId, messageId, updater) {
    set((state) => ({
      sessions: state.sessions.map(s => {
        if (s.id !== sessionId) return s
        const messages = s.messages.map(m => m.id === messageId ? updater(m) : m)
        return { ...s, messages, updatedAt: Date.now() }
      })
    }))
  },

  // ---------- 信息元提取：启动/中止（中文注释） ----------
  // 停止所有 pending 目标的提取；clear=true 时同时清除结果
  abortPendingInfons(clear = false) {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = (get().infonSessions?.[session.id]?.runs) || []
    for (const r of runs) {
      if (r.targetType === 'pending' && r.status === 'running') {
        try { r.controller?.abort?.() } catch (_) {}
      }
    }
    if (clear) {
      set((state) => {
        const box = state.infonSessions?.[session.id]
        if (!box) return {}
        const nextRuns = box.runs.filter(r => r.targetType !== 'pending')
        return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } } }
      })
    } else {
      // 直接移除被中止的 pending 运行（中文注释）
      const toAbort = new Set(runs.filter(r => r.targetType === 'pending' && r.status === 'running').map(r => r.id))
      set((state) => {
        const box = state.infonSessions?.[session.id]
        if (!box) return {}
        const nextRuns = box.runs.filter(r => !(r.targetType === 'pending' && toAbort.has(r.id)))
        return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } } }
      })
    }
  },

  // 单独中止某个 run（中文注释）
  abortInfonRun(runId) {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = (get().infonSessions?.[session.id]?.runs) || []
    const r = runs.find(x => x.id === runId)
    if (!r) return
    try { r.controller?.abort?.() } catch (_) {}
    // 移除该 run（中文注释）
    set((state) => {
      const box = state.infonSessions?.[session.id]
      if (!box) return {}
      const nextRuns = box.runs.filter(x => x.id !== runId)
      return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } } }
    })
  },

  // 发送消息时处理 pending 信息元：清除所有 pending 任务，因为 message 任务将替代它们（中文注释）
  clearAllPendingInfons() {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = (get().infonSessions?.[session.id]?.runs) || []
    // 先中止所有 pending 运行
    runs.forEach((r) => {
      if (r.targetType === 'pending' && r.status === 'running') {
        try { r.controller?.abort?.() } catch (_) {}
      }
    })
    // 再移除
    const toRemove = new Set(runs.filter(r => r.targetType === 'pending').map(r => r.id))
    if (toRemove.size > 0) {
      set((state) => {
        const box = state.infonSessions?.[session.id]
        if (!box) return {}
        const nextRuns = box.runs.filter(r => !toRemove.has(r.id))
        return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } } }
      })
    }
  },

  // 启动基于 pending 输入的信息元提取（中文注释）
  startPendingInfons(text, imageDataUrls) {
    const session = get().getCurrentSession()
    if (!session) return
    // 输入为空不启动（中文注释）
    const t = (text || '').trim()
    const imgs = Array.isArray(imageDataUrls) ? imageDataUrls.filter(Boolean) : []
    if (!t && imgs.length === 0) return

    // 计算哈希（中文注释）
    const textHash = t ? computeHashId(t) : null
    const imageHashes = imgs.map((u) => computeHashId(u))

    // 文本：只有 hash 改变才重提；若改变则移除旧文本 pending 结果（中文注释）
    if (t) {
      if (textHash !== get().lastPendingTextHash) {
        // 先中止旧的 pending 文本 run，再移除
        try {
          const currentRuns = (get().infonSessions?.[session.id]?.runs) || []
          currentRuns.forEach((r) => {
            if (r.targetType === 'pending' && r.modality === 'text' && r.status === 'running') {
              try { r.controller?.abort?.() } catch (_) {}
            }
          })
        } catch (_) {}
        set((state) => {
          const box = state.infonSessions?.[session.id] || { runs: [] }
          const nextRuns = box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'text'))
          return { infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } }, lastPendingTextHash: textHash }
        })
        get()._startTextInfonRun({ targetType: 'pending', targetKey: 'pending', text: t })
      }
    } else {
      // 没有文本则中止并清理所有 pending 文本 run
      try {
        const currentRuns = (get().infonSessions?.[session.id]?.runs) || []
        currentRuns.forEach((r) => {
          if (r.targetType === 'pending' && r.modality === 'text' && r.status === 'running') {
            try { r.controller?.abort?.() } catch (_) {}
          }
        })
      } catch (_) {}
      set((state) => {
        const box = state.infonSessions?.[session.id]
        if (!box) return {}
        const nextRuns = box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'text'))
        return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } }, lastPendingTextHash: null }
      })
    }

    // 图片：新增只为新 hash 启动；移除消失的 hash 的 run（中文注释）
    // 中止并移除不再存在的图片 pending runs
    try {
      const box = get().infonSessions?.[session.id] || { runs: [] }
      const currentHashes = new Set(imageHashes)
      box.runs.forEach((r) => {
        if (r.targetType === 'pending' && r.modality === 'image' && !currentHashes.has(r._hash) && r.status === 'running') {
          try { r.controller?.abort?.() } catch (_) {}
        }
      })
    } catch (_) {}
    set((state) => {
      const box = state.infonSessions?.[session.id] || { runs: [] }
      const currentHashes = new Set(imageHashes)
      let nextRuns = box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'image' && !currentHashes.has(r._hash)))
      // 再为新增的 hash 启动 run
      const existing = new Set(nextRuns.filter(r => r.targetType === 'pending' && r.modality === 'image').map(r => r._hash))
      const toStart = []
      imageHashes.forEach((h, idx) => { if (!existing.has(h)) toStart.push(idx) })
      // 写回 runs（暂不加入新 run，这里只清理；启动在 set 之后执行）
      return { infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } }, lastPendingImageHashes: imageHashes }
    })

    // 启动新增图片的 run（中文注释）
    const existingHashes = new Set(((get().infonSessions?.[session.id]?.runs) || []).filter(r => r.targetType === 'pending' && r.modality === 'image').map(r => r._hash))
    imageHashes.forEach((h, idx) => {
      if (!existingHashes.has(h)) {
        get()._startImageInfonRun({ targetType: 'pending', targetKey: 'pending', dataUrl: imgs[idx], imageIndex: idx, _hash: h })
      }
    })
  },

  // 发送后基于消息 ID 启动信息元提取（中文注释）
  startMessageInfons(messageId) {
    const session = get().getCurrentSession()
    if (!session) return
    const m = (session.messages || []).find(x => x.id === messageId)
    if (!m) return
    const t = (m.content || '').trim()
    const imgs = Array.isArray(m.images) ? m.images.filter(Boolean) : []
    // 先中止旧的该 message 的运行，再清理（中文注释）
    try {
      const runs = (get().infonSessions?.[session.id]?.runs) || []
      runs.forEach((r) => {
        if (r.targetType === 'message' && r.targetKey === messageId && r.status === 'running') {
          try { r.controller?.abort?.() } catch (_) {}
        }
      })
    } catch (_) {}
    // 清理旧的该 message 的 runs（中文注释）
    set((state) => {
      const box = state.infonSessions?.[session.id] || { runs: [] }
      const nextRuns = box.runs.filter(r => r.targetType !== 'message' || r.targetKey !== messageId)
      return { infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } } }
    })
    if (t) get()._startTextInfonRun({ targetType: 'message', targetKey: messageId, text: t })
    if (imgs.length) imgs.forEach((dataUrl, idx) => get()._startImageInfonRun({ targetType: 'message', targetKey: messageId, dataUrl, imageIndex: idx }))
  },

  // 内部：文本信息元提取（/v1/chat/completions）（中文注释）
  async _startTextInfonRun({ targetType, targetKey, text }) {
    const session = get().getCurrentSession()
    if (!session) return

    const runId = generateId()
    const run = {
      id: runId,
      targetType,
      targetKey,
      modality: 'text',
      status: 'running',
      progress: 0,
      buffer: '',
      resultJson: null,
      createdAt: Date.now(),
      controller: null,
    }
    get()._appendInfonRun(session.id, run)

    const provider = get().customProviders?.[get().model]
    const baseUrl = provider ? provider.baseUrl : get().baseUrl
    const headers = { 'Content-Type': 'application/json' }
    if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`

    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['text'], nowISO)
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons as a strict single JSON object. Input text:\n\n${text}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, (r) => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ model: get().model, messages, temperature: 0, stream: true }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: errText || 'Request failed' }))
        return
      }
      const reader = res.body?.getReader()
      if (!reader) {
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'No stream' }))
        return
      }

      await streamOpenAIResponse(reader, ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          // 1) 追加流文本
          get()._updateInfonRun(session.id, runId, (r) => ({ ...r, buffer: r.buffer + content }))
          // 2) 尝试从流文本中增量解析 infons 并即时推送
          const currentBuffer = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const parserState = get().infonParsers?.[runId] || null
          const { state, yielded } = incrementalExtractInfons(currentBuffer, parserState)
          set((st) => ({ infonParsers: { ...(st.infonParsers || {}), [runId]: state } }))
          if (yielded && yielded.length) {
            const nowISO2 = nowISO
            const normalizedYield = yielded.map((o) => ({ record_time: nowISO2, ...o }))
            get()._updateInfonRun(session.id, runId, (r) => {
              const base = r.resultJson && typeof r.resultJson === 'object' ? r.resultJson : { run_metadata: {}, infons: [], quality_report: { stats: {} } }
              return { ...r, resultJson: { ...base, infons: [...(base.infons || []), ...normalizedYield] } }
            })
          }
        }
        if (finish) {
          const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const sliced = extractFirstJSONObject(raw) || raw
          const { ok, value } = tryParseJSON(sliced)
          if (ok) {
            // 计算当前对话轮次和信息元次序（中文注释）
            const sessionObj = get().getCurrentSession()
            const messageCount = (sessionObj?.messages || []).length
            const messageRound = Math.floor(messageCount / 2) + 1 // 每轮对话包含用户和助手消息
            const currentRuns = get().getCurrentInfonRuns()
            const completedRuns = currentRuns.filter(r => r.status === 'done')
            const infonIndex = completedRuns.reduce((sum, r) => sum + (r.resultJson?.infons?.length || 0), 0)

            const normalized = normalizeInfonOutput(value, {
              recordTimeISO: nowISO,
              defaultModality: 'text',
              sessionId: session.id,
              messageRound,
              infonIndex,
              infonType: 'text'
            })
            get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'done', progress: 100, resultJson: normalized }))
          } else {
            get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
          }
        }
      })
    } catch (err) {
      const aborted = err && err.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 内部：图像信息元提取（/api/chat）（中文注释）
  async _startImageInfonRun({ targetType, targetKey, dataUrl, imageIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const runId = generateId()
    const run = {
      id: runId,
      targetType,
      targetKey,
      modality: 'image',
      imageIndex,
      _hash,
      status: 'running',
      progress: 0,
      buffer: '',
      resultJson: null,
      createdAt: Date.now(),
      controller: null,
    }
    get()._appendInfonRun(session.id, run)

    // 自定义提供商通常不支持图片（中文注释）
    const provider = get().customProviders?.[get().model]
    if (provider) {
      get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'Image messages are not supported for this model' }))
      return
    }

    const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')
    const stripDataUrl = (s) => {
      if (typeof s !== 'string') return s
      const i = s.indexOf(',')
      if (i >= 0 && s.slice(0, i).includes('base64')) return s.slice(i + 1)
      return s
    }

    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['image'], nowISO)
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: 'Extract Situation Theory infons as a strict single JSON object.', images: [stripDataUrl(dataUrl)] },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, (r) => ({ ...r, controller }))

    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: get().model, messages, stream: true, options: { temperature: 0 } }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: errText || 'Request failed' }))
        return
      }
      const reader = res.body?.getReader()
      if (!reader) {
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'No stream' }))
        return
      }

      await streamOllamaChatResponse(reader, ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          // 1) 追加流文本
          get()._updateInfonRun(session.id, runId, (r) => ({ ...r, buffer: r.buffer + content }))
          // 2) 尝试从流文本中增量解析 infons 并即时推送
          const currentBuffer = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const parserState = get().infonParsers?.[runId] || null
          const { state, yielded } = incrementalExtractInfons(currentBuffer, parserState)
          set((st) => ({ infonParsers: { ...(st.infonParsers || {}), [runId]: state } }))
          if (yielded && yielded.length) {
            const nowISO2 = nowISO
            const normalizedYield = yielded.map((o) => ({ record_time: nowISO2, ...o }))
            get()._updateInfonRun(session.id, runId, (r) => {
              const base = r.resultJson && typeof r.resultJson === 'object' ? r.resultJson : { run_metadata: {}, infons: [], quality_report: { stats: {} } }
              return { ...r, resultJson: { ...base, infons: [...(base.infons || []), ...normalizedYield] } }
            })
          }
        }
        if (finish) {
          const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const sliced = extractFirstJSONObject(raw) || raw
          const { ok, value } = tryParseJSON(sliced)
          if (ok) {
            // 计算当前对话轮次和信息元次序（中文注释）
            const sessionObj = get().getCurrentSession()
            const messageCount = (sessionObj?.messages || []).length
            const messageRound = Math.floor(messageCount / 2) + 1 // 每轮对话包含用户和助手消息
            const currentRuns = get().getCurrentInfonRuns()
            const completedRuns = currentRuns.filter(r => r.status === 'done')
            const infonIndex = completedRuns.reduce((sum, r) => sum + (r.resultJson?.infons?.length || 0), 0)

            const normalized = normalizeInfonOutput(value, {
              recordTimeISO: nowISO,
              defaultModality: 'image',
              sessionId: session.id,
              messageRound,
              infonIndex,
              infonType: 'image'
            })
            get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'done', progress: 100, resultJson: normalized }))
          } else {
            get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
          }
        }
      })
    } catch (err) {
      const aborted = err && err.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 发送消息（中文注释）：整合上下文，调用本地 OpenAI 兼容接口并流式追加助手回复
  async sendMessage(text) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 如历史对话包含图片，则改走多模态 /api/chat，并将图片并入上下文（中文注释）
    const hasHistoricalImages = (session.messages || []).some(m => Array.isArray(m.images) && m.images.length > 0)
    const provider = get().customProviders?.[get().model]
    if (hasHistoricalImages && !provider) {
      return await get().sendMessageWithImages(text, [])
    }

    // 写入用户消息
    const userMsgId = generateId()
    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: text,
      createdAt: Date.now(),
    })

    // 预创建助手空消息用于流式写入
    const assistantMsgId = generateId()
    get()._appendMessage(session.id, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      reasoning: '', // 思考过程（中文注释）：与最终回复分开存储
      phase: 'thinking', // thinking -> answering -> done
      streaming: true,
      createdAt: Date.now(),
    })

    // 组装上下文（中文注释）：将当前会话全部消息转换为 OpenAI Chat messages
    const payloadMessages = get().getCurrentSession().messages.map(m => ({
      role: m.role,
      content: m.content,
    }))

    // 发起请求（中文注释）：使用 AbortController 以支持停止
    const controller = new AbortController()
    set({ isGenerating: true, abortController: controller })

    try {
      // 依据当前模型选择不同的基址与鉴权（中文注释）
      const provider = get().customProviders?.[get().model]
      const baseUrl = provider ? provider.baseUrl : get().baseUrl
      const headers = { 'Content-Type': 'application/json' }
      if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`

      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: get().model,
          messages: payloadMessages,
          temperature: 0.7,
          stream: true,
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        // 错误处理：写入错误信息
        const textErr = await res.text().catch(() => '')
        get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: textErr || 'Request failed', content: m.content }))
        set({ isGenerating: false, abortController: null })
        return
      }

      const reader = res.body?.getReader()
      if (!reader) {
        get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: 'No stream', content: m.content }))
        set({ isGenerating: false, abortController: null })
        return
      }

      // 在本地维护一个 think 状态以拆分 <think> ... </think>
      let inThink = false

      await streamOpenAIResponse(reader, ({ content, reasoning, finish }) => {
        if (reasoning) {
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, reasoning: (m.reasoning || '') + reasoning }))
        }

        if (typeof content === 'string' && content.length) {
          let rest = content
          while (rest && rest.length) {
            if (inThink) {
              const endIdx = rest.indexOf('</think>')
              if (endIdx >= 0) {
                const head = rest.slice(0, endIdx)
                const tail = rest.slice(endIdx + 8)
                if (head) {
                  get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, reasoning: (m.reasoning || '') + head }))
                }
                inThink = false
                rest = tail
                continue
              } else {
                get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, reasoning: (m.reasoning || '') + rest }))
                rest = ''
                break
              }
            } else {
              const startIdx = rest.indexOf('<think>')
              if (startIdx >= 0) {
                const before = rest.slice(0, startIdx)
                const tail = rest.slice(startIdx + 7)
                if (before) {
                  get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, content: (m.content || '') + before, phase: 'answering' }))
                }
                inThink = true
                rest = tail
                continue
              } else {
                get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, content: (m.content || '') + rest, phase: 'answering' }))
                rest = ''
                break
              }
            }
          }
        }

        if (finish) {
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, phase: 'done' }))
          // 助手回复完成后，启动对助手消息的信息元提取（中文注释）
          try { get().startMessageInfons(assistantMsgId) } catch (_) {}
        }
      })

    } catch (err) {
      // 被中止或网络错误
      const msg = (err && (err.name === 'AbortError')) ? 'Aborted' : 'Network error'
      get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: msg }))
    } finally {
      set({ isGenerating: false, abortController: null })
    }

    // 返回用户消息 ID（中文注释）：用于后续处理
    return userMsgId
  },

  // 发送带图片的多模态消息（中文注释）：调用 Ollama 原生 /api/chat，携带 images(base64) 并流式读取
  async sendMessageWithImages(text, imageDataUrls) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 写入用户消息（中文注释）：包含图片预览（data URL）
    const userMsgId = generateId()
    const imgs = Array.isArray(imageDataUrls) ? imageDataUrls.filter(Boolean) : []
    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: text,
      images: imgs,
      createdAt: Date.now(),
    })

    // 如果是自定义提供商，暂不支持图片（中文注释）：直接写入错误并返回
    const provider = get().customProviders?.[get().model]
    if (provider) {
      const assistantMsgId = generateId()
      get()._appendMessage(session.id, {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        reasoning: '',
        phase: 'done',
        streaming: false,
        error: 'Image messages are not supported for this model',
        createdAt: Date.now(),
      })
      return userMsgId
    }

    // 预创建助手空消息用于流式写入
    const assistantMsgId = generateId()
    get()._appendMessage(session.id, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      reasoning: '',
      phase: 'thinking',
      streaming: true,
      createdAt: Date.now(),
    })

    // 将历史消息转换为 Ollama /api/chat 的 messages（中文注释）：携带 images 时去掉 data: 前缀
    const stripDataUrl = (s) => {
      if (typeof s !== 'string') return s
      const i = s.indexOf(',')
      if (i >= 0 && s.slice(0, i).includes('base64')) return s.slice(i + 1)
      return s
    }

    const sessionMsgs = get().getCurrentSession().messages
    let lastImageIdx = -1
    for (let i = sessionMsgs.length - 1; i >= 0; i--) {
      const m = sessionMsgs[i]
      if (Array.isArray(m.images) && m.images.length > 0 && m.role === 'user') { lastImageIdx = i; break }
    }
    const history = sessionMsgs.map((m, idx) => {
      const o = { role: m.role, content: m.content }
      if (idx === lastImageIdx && Array.isArray(m.images) && m.images.length) {
        o.images = m.images.map(stripDataUrl)
      }
      return o
    })

    // 计算 /api 基址（中文注释）：将 baseUrl 的 /v1 替换为 /api
    const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')

    const controller = new AbortController()
    set({ isGenerating: true, abortController: controller })
    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: get().model,
          messages: history,
          stream: true,
          options: { temperature: 0.2 }
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const textErr = await res.text().catch(() => '')
        get().
          _updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: textErr || 'Request failed', content: m.content }))
        set({ isGenerating: false, abortController: null })
        return
      }

      const reader = res.body?.getReader()
      if (!reader) {
        get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: 'No stream', content: m.content }))
        set({ isGenerating: false, abortController: null })
        return
      }

      await streamOllamaChatResponse(reader, ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, content: (m.content || '') + content, phase: 'answering' }))
        }
        if (finish) {
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, phase: 'done' }))
          // 助手回复完成后，启动对助手消息的信息元提取（中文注释）
          try { get().startMessageInfons(assistantMsgId) } catch (_) {}
        }
      })

    } catch (err) {
      const msg = (err && (err.name === 'AbortError')) ? 'Aborted' : 'Network error'
      get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: msg }))
    } finally {
      set({ isGenerating: false, abortController: null })
    }

    // 返回用户消息 ID（中文注释）
    return userMsgId
  },

  // 停止生成（中文注释）：调用 AbortController 取消流
  stopGenerating() {
    const { abortController } = get()
    try { abortController?.abort() } catch (_) { }
    set({ isGenerating: false, abortController: null })
  },

  // 重新生成（中文注释）：删除最后一条助手消息并复用最后一条用户消息再次生成
  async regenerateLast() {
    const session = get().getCurrentSession()
    if (!session) return
    // 找到最后的用户消息
    const lastUserIndex = [...session.messages].reverse().findIndex(m => m.role === 'user')
    if (lastUserIndex === -1) return
    const idxFromEnd = lastUserIndex
    const userIdx = session.messages.length - 1 - idxFromEnd
    const lastUser = session.messages[userIdx]

    // 如果最后一条是助手，先移除（中文注释）：保持一问一答结构
    set((state) => ({
      sessions: state.sessions.map(s => {
        if (s.id !== session.id) return s
        const msgs = [...s.messages]
        if (msgs.length > userIdx + 1 && msgs[userIdx + 1].role === 'assistant') {
          msgs.splice(userIdx + 1, 1)
        }
        return { ...s, messages: msgs, updatedAt: Date.now() }
      })
    }))

    await get().sendMessage(lastUser.content)
  },

}))


