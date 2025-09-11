import { create } from 'zustand'
import { systemPrompt as privacySystemPrompt, prefix as privacyPrefix, suffix as privacySuffix, formatSpec as privacyFormatSpec } from './templates/privacyInferenceTemplate'

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
  model: 'qwen2.5vl:7b',
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
        }
      })

    } catch (err) {
      // 被中止或网络错误
      const msg = (err && (err.name === 'AbortError')) ? 'Aborted' : 'Network error'
      get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: msg }))
    } finally {
      set({ isGenerating: false, abortController: null })
    }
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
      return
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
        }
      })

    } catch (err) {
      const msg = (err && (err.name === 'AbortError')) ? 'Aborted' : 'Network error'
      get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: msg }))
    } finally {
      set({ isGenerating: false, abortController: null })
    }
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

  // 固定模板隐私推断（中文注释）：结合历史上下文、未发送输入与选中属性，使用指定模型一次性返回JSON结果
  async runPrivacyInference(pendingText, overrideModel, selectedAttributes) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) throw new Error('No active session')

    // 组装历史上下文（中文注释）：使用当前会话全部消息，保留角色用于溯源（与右侧证据图一一对应）
    const history = (session.messages || []).map((m, idx) => {
      const role = m.role || 'user'
      const content = typeof m.content === 'string' ? m.content : ''
      return `[${idx}](${role}) ${content}`
    }).join('\n')

    // 模板要素（中文注释）：system + user（Prefix + 历史 + 未发送 + Suffix），强制JSON输出（从模板模块导入）
    const systemPrompt = privacySystemPrompt
    const prefix = privacyPrefix
    const suffix = privacySuffix
    const formatSpec = privacyFormatSpec

    const pending = (pendingText || '').trim()
    const attrs = Array.isArray(selectedAttributes) ? selectedAttributes.filter(Boolean) : []
    const attrsBlock = attrs.length ? `\n\n[SELECTED_ATTRIBUTES]\n${attrs.join('\n')}` : `\n\n[SELECTED_ATTRIBUTES]\n` 
    const userContent = `${prefix}\n\n[HISTORY]\n${history}\n\n[PENDING]\n${pending}${attrsBlock}\n\n${suffix}${formatSpec}`

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userContent },
    ]

    const modelToUse = overrideModel || get().model
    // 仅允许本地模型（中文注释）
    const isCustom = !!get().customProviders?.[modelToUse]
    if (isCustom) {
      throw new Error('Privacy inference requires local model')
    }

    // 发起一次性请求（中文注释）：OpenAI 兼容 /chat/completions，关闭流
    const res = await fetch(`${get().baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelToUse,
        messages,
        temperature: 0.1,
        stream: false,
      })
    })

    if (!res.ok) {
      const textErr = await res.text().catch(() => '')
      throw new Error(textErr || 'Inference request failed')
    }

    // 解析响应（中文注释）：兼容 OpenAI choices[0].message.content
    const json = await res.json().catch(() => ({}))
    const content = json?.choices?.[0]?.message?.content || ''

    // 尝试提取JSON（中文注释）：宽松截取第一个 { 到最后一个 }
    const extractJson = (s) => {
      if (typeof s !== 'string') return null
      const first = s.indexOf('{')
      const last = s.lastIndexOf('}')
      if (first >= 0 && last >= first) {
        const sub = s.slice(first, last + 1)
        try { return JSON.parse(sub) } catch (_) { return null }
      }
      return null
    }

    const parsed = extractJson(content)
    if (!parsed) throw new Error('Model did not return valid JSON')
    return parsed
  },
}))


