# Ollama.md 文件处理
_Exported on 2026/2/8 at GMT+8 18:01:28 from Cursor (2.2.44)_

---

**User**

@cursor_ollama.md 继续处理

---

**Cursor**

```javascript
// streamOpenAIResponse:
if (contentDelta || reasoningDelta) onDelta({ content: contentDelta, reasoning: reasoningDelta, finish: null })
if (finish) onDelta({ content: '', reasoning: '', finish })

// streamOllamaChatResponse:
if (delta) onDelta({ content: delta, reasoning: '', finish: null })
if (finish) onDelta({ content: '', reasoning: '', finish })
```

我找到了卡住的根本原因。让我先分析一下问题链：

**核心问题**：`streamOpenAIResponse` 和 `streamOllamaChatResponse` 中调用 `onDelta` 时**没有 `await`**，但 `infonSlice.js` 传入的回调是 `async` 函数。这意味着如果 `finish` 回调中任何代码抛出错误（比如新增的 `deduplicateAndMergeInfons` 中），错误会被静默吞掉，run 永远停留在 `running` 状态。

让我逐步修复：

```javascript
/**
 * 流式响应处理工具模块
 * 处理 OpenAI 和 Ollama 的 SSE 流式响应
 */

/**
 * 解析 OpenAI SSE 流：将 response.body 按行解析 data: 片段
 * @param {ReadableStreamDefaultReader} reader - 流读取器
 * @param {Function} onDelta - 回调函数，接收 {content, reasoning, finish}
 */
export async function streamOpenAIResponse(reader, onDelta) {
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
          // 必须 await 异步回调，确保错误能正确传播
          if (contentDelta || reasoningDelta) await onDelta({ content: contentDelta, reasoning: reasoningDelta, finish: null })
          if (finish) await onDelta({ content: '', reasoning: '', finish })
        } catch (_) {
          // 忽略不可解析的行
        }
      }
    }
  }
}

/**
 * 解析 Ollama /api/chat 流：逐行解析 JSON，并处理"全量快照"或"增量 token"两种格式
 * @param {ReadableStreamDefaultReader} reader - 流读取器
 * @param {Function} onDelta - 回调函数，接收 {content, reasoning, finish}
 */
export async function streamOllamaChatResponse(reader, onDelta) {
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

```

```javascript
/**
 * 流式响应处理工具模块
 * 处理 OpenAI 和 Ollama 的 SSE 流式响应
 */

/**
 * 解析 OpenAI SSE 流：将 response.body 按行解析 data: 片段
 * @param {ReadableStreamDefaultReader} reader - 流读取器
 * @param {Function} onDelta - 回调函数，接收 {content, reasoning, finish}
 */
export async function streamOpenAIResponse(reader, onDelta) {
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
          // 必须 await 异步回调，确保错误能正确传播
          if (contentDelta || reasoningDelta) await onDelta({ content: contentDelta, reasoning: reasoningDelta, finish: null })
          if (finish) await onDelta({ content: '', reasoning: '', finish })
        } catch (_) {
          // 忽略不可解析的行
        }
      }
    }
  }
}

/**
 * 解析 Ollama /api/chat 流：逐行解析 JSON，并处理"全量快照"或"增量 token"两种格式
 * @param {ReadableStreamDefaultReader} reader - 流读取器
 * @param {Function} onDelta - 回调函数，接收 {content, reasoning, finish}
 */
export async function streamOllamaChatResponse(reader, onDelta) {
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
          // 必须 await 异步回调，确保错误能正确传播
          if (delta) await onDelta({ content: delta, reasoning: '', finish: null })
        }
        if (finish) await onDelta({ content: '', reasoning: '', finish })
      } catch (_) {
        // 忽略不可解析的行
      }
    }
  }
}

```

现在给 `infonSlice.js` 的 finish handler 添加 try/catch 保护，确保无论如何都会设置 run 状态：

```javascript
// 信息元提取 Slice
import { buildSystemPrompt } from '../../templates/infons.js'
import { generateId, tryParseJSON, extractFirstJSONObject, computeHashId, normalizeInfonOutput } from '../utils'
import { streamOpenAIResponse, streamOllamaChatResponse } from '../streamUtils'
import { incrementalExtractInfons } from '../infonParser'
import { deduplicateAndMergeInfons } from '../infonMerge'
import { getExistingInfons, createInfonRun, getModelApiConfig } from './infonHelpers'
import { stripDataUrl } from './messageHelpers'
import { getModelModalities } from '../../utils/modelUtils'

function buildInfonSystemPrompt(modalities, nowISO, options = {}) {
  const { currentRound = 1, existingInfons = [] } = options
  return buildSystemPrompt({
    modalities, includeExamples: false, currentRound, existingInfons,
    extraInstructions: `System time (ISO8601) = ${nowISO}. Set run_metadata.record_time to this value. For each situation and infon, if record_time is missing, set it to this value. Only set occur_time when it is explicitly expressed; otherwise omit.`
  })
}

export const createInfonSlice = (set, get) => ({
  // 状态
  infonSessions: {},
  lastPendingTextHash: null,
  lastPendingImageHashes: [],
  infonParsers: {},
  highlightedInfon: null,

  setHighlightedInfon(infon, run) { set({ highlightedInfon: infon ? { infon, run } : null }) },

  _getOrCreateInfonSession(sessionId) {
    let box = get().infonSessions?.[sessionId]
    if (!box) {
      box = { runs: [] }
      set(s => ({ infonSessions: { ...(s.infonSessions || {}), [sessionId]: box } }))
    }
    return box
  },

  _appendInfonRun(sessionId, run) {
    set(s => {
      const cur = s.infonSessions?.[sessionId] || { runs: [] }
      return { infonSessions: { ...(s.infonSessions || {}), [sessionId]: { runs: [...cur.runs, run] } } }
    })
  },

  _updateInfonRun(sessionId, runId, updater) {
    set(s => {
      const box = s.infonSessions?.[sessionId]
      if (!box) return {}
      return { infonSessions: { ...s.infonSessions, [sessionId]: { runs: box.runs.map(r => r.id === runId ? updater(r) : r) } } }
    })
  },

  getCurrentInfonRuns() {
    const session = get().getCurrentSession()
    return session ? (get().infonSessions?.[session.id]?.runs || []) : []
  },

  clearAllPendingInfons() {
    get()._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return
    try {
      const runs = (get().infonSessions?.[session.id]?.runs) || []
      runs.forEach(r => { if (r.targetType === 'pending' && r.status === 'running') try { r.controller?.abort?.() } catch (_) {} })
    } catch (_) {}
    set(s => {
      const box = s.infonSessions?.[session.id] || { runs: [] }
      return { infonSessions: { ...(s.infonSessions || {}), [session.id]: { runs: box.runs.filter(r => r.targetType !== 'pending') } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
  },

  adoptPendingInfonsToMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session || !messageId) return { adopted: 0, runIds: [] }
    let adopted = 0
    const adoptedRunIds = []
    set(s => {
      const box = s.infonSessions?.[session.id]
      if (!box) return {}
      const runs = box.runs.map(r => {
        if (r.targetType === 'pending') { adopted++; adoptedRunIds.push(r.id); return { ...r, targetType: 'message', targetKey: messageId } }
        return r
      })
      return { infonSessions: { ...s.infonSessions, [session.id]: { runs } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
    return { adopted, runIds: adoptedRunIds }
  },

  // 触发信息元提取（消息发送后）
  triggerInfonsForMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session) return
    const msg = session.messages.find(m => m.id === messageId)
    if (!msg || msg.role !== 'user') return
    
    const text = typeof msg.content === 'string' ? msg.content : ''
    const imgs = msg.images || []
    const audios = msg.audios || []
    
    if (text.trim()) get()._startTextInfonRun({ targetType: 'message', targetKey: messageId, text })
    if (imgs.length) imgs.forEach((dataUrl, idx) => get()._startImageInfonRun({ targetType: 'message', targetKey: messageId, dataUrl, imageIndex: idx }))
    if (audios.length) audios.forEach((audio, idx) => {
      const audioHash = computeHashId(audio.id + (audio.transcript || ''))
      get()._startAudioInfonRun({ targetType: 'message', targetKey: messageId, audio, audioIndex: idx, _hash: audioHash })
    })
  },

  // 别名方法：供外部调用
  startMessageInfons(messageId) { get().triggerInfonsForMessage(messageId) },

  // 启动 pending 信息元提取（用户输入时）
  startPendingInfons(text, images = [], audios = []) {
    const session = get().getCurrentSession()
    if (!session) return

    const runs = get().infonSessions?.[session.id]?.runs || []

    // 文本处理：计算 hash 判断是否需要重新提取
    const textHash = text?.trim() ? computeHashId(text.trim()) : null
    const lastTextHash = get().lastPendingTextHash
    if (textHash && textHash !== lastTextHash) {
      // 先中止并移除旧的 pending 文本 runs
      runs.filter(r => r.targetType === 'pending' && r.modality === 'text').forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return {
          infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'text')) } },
          lastPendingTextHash: textHash
        }
      })
      get()._startTextInfonRun({ targetType: 'pending', targetKey: 'pending', text: text.trim() })
    }

    // 图片处理：检查新增图片
    const imageHashes = (images || []).map(img => {
      const url = typeof img === 'string' ? img : img.url
      return computeHashId(url)
    })
    const lastImageHashes = get().lastPendingImageHashes || []
    const newImages = imageHashes.filter(h => !lastImageHashes.includes(h))
    // 移除已删除的图片对应的 runs
    const removedImages = lastImageHashes.filter(h => !imageHashes.includes(h))
    if (removedImages.length > 0) {
      runs.filter(r => r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash)).forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return { infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash))) } } }
      })
    }
    if (newImages.length > 0) {
      set({ lastPendingImageHashes: imageHashes })
      images.forEach((img, idx) => {
        const url = typeof img === 'string' ? img : img.url
        const hash = computeHashId(url)
        if (newImages.includes(hash)) {
          get()._startImageInfonRun({ targetType: 'pending', targetKey: 'pending', dataUrl: url, imageIndex: idx, _hash: hash })
        }
      })
    }

    // 音频处理
    audios.forEach((audio, idx) => {
      if (!audio.transcript?.trim()) return
      const audioHash = computeHashId(audio.id + audio.transcript)
      const existingRuns = get().infonSessions?.[session.id]?.runs || []
      const alreadyRunning = existingRuns.some(r => r.targetType === 'pending' && r._hash === audioHash && (r.status === 'running' || r.status === 'done'))
      if (!alreadyRunning) {
        get()._startAudioInfonRun({ targetType: 'pending', targetKey: 'pending', audio, audioIndex: idx, _hash: audioHash })
      }
    })
  },

  // 中止 pending 信息元提取
  abortPendingInfons() {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = get().infonSessions?.[session.id]?.runs || []
    runs.forEach(r => {
      if (r.targetType === 'pending' && r.status === 'running' && r.controller) {
        try { r.controller.abort() } catch (_) {}
      }
    })
  },

  // 文本信息元提取
  async _startTextInfonRun({ targetType, targetKey, text }) {
    const session = get().getCurrentSession()
    if (!session) return

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'text' })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const { provider, baseUrl, headers, maxTokens } = getModelApiConfig(get, configuredModel)
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['text'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons in compact format. Input text:\n\n${text}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (currentRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
          
          const performParsing = async () => {
            const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            if (!curRun) return
            const { state: newState, yielded } = await incrementalExtractInfons(curRun.buffer || '', get().infonParsers?.[runId] || null)
            set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
            if (yielded?.length > 0) {
              get()._updateInfonRun(session.id, runId, r => {
                const infons = [...(r.resultJson?.infons || [])]
                yielded.forEach(ni => {
                  const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                  idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                })
                return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
              })
            }
            lastParseTime = Date.now()
          }
          
          if (parseTimer) clearTimeout(parseTimer)
          if (buffer.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
          const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const parserState = get().infonParsers?.[runId]
          let finalInfons = [], parseSuccess = false
          
          if (parserState?.isCompact) {
            const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
            const result = parseCompactInfonsFormat(raw)
            if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
          } else {
            const sliced = extractFirstJSONObject(raw) || raw
            const { ok, value } = tryParseJSON(sliced)
            if (ok) {
              const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'text', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
              finalInfons = normalized.infons || []; parseSuccess = true
            }
          }
          
          if (parseSuccess && finalInfons.length) {
            const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
            // === 主记忆流：写入向量索引库 ===
            console.log('[MemoryStream] text ingest hook:', deduplicated.length, 'infons, session:', session.id, 'round:', currentRound)
            try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] text ingest error:', e) }
          } else {
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 图像信息元提取
  async _startImageInfonRun({ targetType, targetKey, dataUrl, imageIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'image', imageIndex, _hash })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().imageParsingModel || 'gemma3:12b'
    const customProviders = get().customProviders
    
    if (!getModelModalities(configuredModel, customProviders).image) {
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Image not supported' }))
      return
    }
    
    const provider = customProviders?.[configuredModel]
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['image'], nowISO, { currentRound, existingInfons })
    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      let res
      if (provider) {
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: [{ type: 'text', text: 'Extract Situation Theory infons in compact format.' }, { type: 'image_url', image_url: { url: dataUrl } }] },
        ]
        const isOmni = configuredModel.toLowerCase().includes('omni')
        res = await fetch(`${provider.baseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...(provider.apiKey ? { 'Authorization': `Bearer ${provider.apiKey}` } : {}), 'Connection': 'keep-alive' },
          body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: isOmni ? 2000 : 4096, top_p: 0.95 }),
          signal: controller.signal, keepalive: true
        })
      } else {
        const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: 'Extract Situation Theory infons in compact format.', images: [stripDataUrl(dataUrl)] },
        ]
        res = await fetch(`${apiBase}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: configuredModel, messages, stream: true, options: { temperature: 0 } }),
          signal: controller.signal,
        })
      }
      
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      const streamHandler = provider ? streamOpenAIResponse : streamOllamaChatResponse
      await streamHandler(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          try {
            const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            const buffer = (curRun?.buffer || '') + content
            get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
            
            const { state: newState, yielded } = await incrementalExtractInfons(buffer, get().infonParsers?.[runId] || null)
            set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
            
            if (yielded?.length > 0) {
              get()._updateInfonRun(session.id, runId, r => {
                const infons = [...(r.resultJson?.infons || [])]
                yielded.forEach(ni => {
                  const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                  idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                })
                return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
              })
            }
          } catch (contentErr) {
            console.error('[InfonSlice] image content handler error:', contentErr)
          }
        }
        if (finish) {
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'image', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] image ingest hook:', deduplicated.length, 'infons')
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] image ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] image finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 音频信息元提取
  async _startAudioInfonRun({ targetType, targetKey, audio, audioIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const transcript = (audio.transcript || '').trim()
    if (!transcript) {
      const errorRun = { ...createInfonRun({ targetType, targetKey, modality: 'audio', audioIndex, _hash }), status: 'error', error: 'No transcript' }
      get()._appendInfonRun(session.id, errorRun)
      return
    }

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'audio', audioIndex, _hash })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const { baseUrl, headers, maxTokens } = getModelApiConfig(get, configuredModel)
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['audio'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons in compact format. Audio transcript:\n\n${transcript}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (curRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
          
          const performParsing = async () => {
            const run = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            if (!run) return
            const { state: newState, yielded } = await incrementalExtractInfons(run.buffer || '', get().infonParsers?.[runId] || null)
            set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
            if (yielded?.length > 0) {
              get()._updateInfonRun(session.id, runId, r => {
                const infons = [...(r.resultJson?.infons || [])]
                yielded.forEach(ni => {
                  const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                  idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                })
                return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
              })
            }
            lastParseTime = Date.now()
          }
          
          if (parseTimer) clearTimeout(parseTimer)
          if (buffer.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
          const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const parserState = get().infonParsers?.[runId]
          let finalInfons = [], parseSuccess = false
          
          if (parserState?.isCompact) {
            const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
            const result = parseCompactInfonsFormat(raw)
            if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
          } else {
            const sliced = extractFirstJSONObject(raw) || raw
            const { ok, value } = tryParseJSON(sliced)
            if (ok) {
              const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'audio', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
              finalInfons = normalized.infons || []; parseSuccess = true
            }
          }
          
          if (parseSuccess && finalInfons.length) {
            const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
            // === 主记忆流：写入向量索引库 ===
            console.log('[MemoryStream] audio ingest hook:', deduplicated.length, 'infons')
            try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] audio ingest error:', e) }
          } else {
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },
})

```

同样给文本和音频的 finish handler 也加上 try/catch 保护：

```javascript
// 信息元提取 Slice
import { buildSystemPrompt } from '../../templates/infons.js'
import { generateId, tryParseJSON, extractFirstJSONObject, computeHashId, normalizeInfonOutput } from '../utils'
import { streamOpenAIResponse, streamOllamaChatResponse } from '../streamUtils'
import { incrementalExtractInfons } from '../infonParser'
import { deduplicateAndMergeInfons } from '../infonMerge'
import { getExistingInfons, createInfonRun, getModelApiConfig } from './infonHelpers'
import { stripDataUrl } from './messageHelpers'
import { getModelModalities } from '../../utils/modelUtils'

function buildInfonSystemPrompt(modalities, nowISO, options = {}) {
  const { currentRound = 1, existingInfons = [] } = options
  return buildSystemPrompt({
    modalities, includeExamples: false, currentRound, existingInfons,
    extraInstructions: `System time (ISO8601) = ${nowISO}. Set run_metadata.record_time to this value. For each situation and infon, if record_time is missing, set it to this value. Only set occur_time when it is explicitly expressed; otherwise omit.`
  })
}

export const createInfonSlice = (set, get) => ({
  // 状态
  infonSessions: {},
  lastPendingTextHash: null,
  lastPendingImageHashes: [],
  infonParsers: {},
  highlightedInfon: null,

  setHighlightedInfon(infon, run) { set({ highlightedInfon: infon ? { infon, run } : null }) },

  _getOrCreateInfonSession(sessionId) {
    let box = get().infonSessions?.[sessionId]
    if (!box) {
      box = { runs: [] }
      set(s => ({ infonSessions: { ...(s.infonSessions || {}), [sessionId]: box } }))
    }
    return box
  },

  _appendInfonRun(sessionId, run) {
    set(s => {
      const cur = s.infonSessions?.[sessionId] || { runs: [] }
      return { infonSessions: { ...(s.infonSessions || {}), [sessionId]: { runs: [...cur.runs, run] } } }
    })
  },

  _updateInfonRun(sessionId, runId, updater) {
    set(s => {
      const box = s.infonSessions?.[sessionId]
      if (!box) return {}
      return { infonSessions: { ...s.infonSessions, [sessionId]: { runs: box.runs.map(r => r.id === runId ? updater(r) : r) } } }
    })
  },

  getCurrentInfonRuns() {
    const session = get().getCurrentSession()
    return session ? (get().infonSessions?.[session.id]?.runs || []) : []
  },

  clearAllPendingInfons() {
    get()._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return
    try {
      const runs = (get().infonSessions?.[session.id]?.runs) || []
      runs.forEach(r => { if (r.targetType === 'pending' && r.status === 'running') try { r.controller?.abort?.() } catch (_) {} })
    } catch (_) {}
    set(s => {
      const box = s.infonSessions?.[session.id] || { runs: [] }
      return { infonSessions: { ...(s.infonSessions || {}), [session.id]: { runs: box.runs.filter(r => r.targetType !== 'pending') } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
  },

  adoptPendingInfonsToMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session || !messageId) return { adopted: 0, runIds: [] }
    let adopted = 0
    const adoptedRunIds = []
    set(s => {
      const box = s.infonSessions?.[session.id]
      if (!box) return {}
      const runs = box.runs.map(r => {
        if (r.targetType === 'pending') { adopted++; adoptedRunIds.push(r.id); return { ...r, targetType: 'message', targetKey: messageId } }
        return r
      })
      return { infonSessions: { ...s.infonSessions, [session.id]: { runs } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
    return { adopted, runIds: adoptedRunIds }
  },

  // 触发信息元提取（消息发送后）
  triggerInfonsForMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session) return
    const msg = session.messages.find(m => m.id === messageId)
    if (!msg || msg.role !== 'user') return
    
    const text = typeof msg.content === 'string' ? msg.content : ''
    const imgs = msg.images || []
    const audios = msg.audios || []
    
    if (text.trim()) get()._startTextInfonRun({ targetType: 'message', targetKey: messageId, text })
    if (imgs.length) imgs.forEach((dataUrl, idx) => get()._startImageInfonRun({ targetType: 'message', targetKey: messageId, dataUrl, imageIndex: idx }))
    if (audios.length) audios.forEach((audio, idx) => {
      const audioHash = computeHashId(audio.id + (audio.transcript || ''))
      get()._startAudioInfonRun({ targetType: 'message', targetKey: messageId, audio, audioIndex: idx, _hash: audioHash })
    })
  },

  // 别名方法：供外部调用
  startMessageInfons(messageId) { get().triggerInfonsForMessage(messageId) },

  // 启动 pending 信息元提取（用户输入时）
  startPendingInfons(text, images = [], audios = []) {
    const session = get().getCurrentSession()
    if (!session) return

    const runs = get().infonSessions?.[session.id]?.runs || []

    // 文本处理：计算 hash 判断是否需要重新提取
    const textHash = text?.trim() ? computeHashId(text.trim()) : null
    const lastTextHash = get().lastPendingTextHash
    if (textHash && textHash !== lastTextHash) {
      // 先中止并移除旧的 pending 文本 runs
      runs.filter(r => r.targetType === 'pending' && r.modality === 'text').forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return {
          infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'text')) } },
          lastPendingTextHash: textHash
        }
      })
      get()._startTextInfonRun({ targetType: 'pending', targetKey: 'pending', text: text.trim() })
    }

    // 图片处理：检查新增图片
    const imageHashes = (images || []).map(img => {
      const url = typeof img === 'string' ? img : img.url
      return computeHashId(url)
    })
    const lastImageHashes = get().lastPendingImageHashes || []
    const newImages = imageHashes.filter(h => !lastImageHashes.includes(h))
    // 移除已删除的图片对应的 runs
    const removedImages = lastImageHashes.filter(h => !imageHashes.includes(h))
    if (removedImages.length > 0) {
      runs.filter(r => r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash)).forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return { infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash))) } } }
      })
    }
    if (newImages.length > 0) {
      set({ lastPendingImageHashes: imageHashes })
      images.forEach((img, idx) => {
        const url = typeof img === 'string' ? img : img.url
        const hash = computeHashId(url)
        if (newImages.includes(hash)) {
          get()._startImageInfonRun({ targetType: 'pending', targetKey: 'pending', dataUrl: url, imageIndex: idx, _hash: hash })
        }
      })
    }

    // 音频处理
    audios.forEach((audio, idx) => {
      if (!audio.transcript?.trim()) return
      const audioHash = computeHashId(audio.id + audio.transcript)
      const existingRuns = get().infonSessions?.[session.id]?.runs || []
      const alreadyRunning = existingRuns.some(r => r.targetType === 'pending' && r._hash === audioHash && (r.status === 'running' || r.status === 'done'))
      if (!alreadyRunning) {
        get()._startAudioInfonRun({ targetType: 'pending', targetKey: 'pending', audio, audioIndex: idx, _hash: audioHash })
      }
    })
  },

  // 中止 pending 信息元提取
  abortPendingInfons() {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = get().infonSessions?.[session.id]?.runs || []
    runs.forEach(r => {
      if (r.targetType === 'pending' && r.status === 'running' && r.controller) {
        try { r.controller.abort() } catch (_) {}
      }
    })
  },

  // 文本信息元提取
  async _startTextInfonRun({ targetType, targetKey, text }) {
    const session = get().getCurrentSession()
    if (!session) return

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'text' })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const { provider, baseUrl, headers, maxTokens } = getModelApiConfig(get, configuredModel)
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['text'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons in compact format. Input text:\n\n${text}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (currentRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
          
          const performParsing = async () => {
            try {
              const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
              if (!curRun) return
              const { state: newState, yielded } = await incrementalExtractInfons(curRun.buffer || '', get().infonParsers?.[runId] || null)
              set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
              if (yielded?.length > 0) {
                get()._updateInfonRun(session.id, runId, r => {
                  const infons = [...(r.resultJson?.infons || [])]
                  yielded.forEach(ni => {
                    const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                    idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                  })
                  return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
                })
              }
              lastParseTime = Date.now()
            } catch (parseErr) {
              console.error('[InfonSlice] text parsing error:', parseErr)
            }
          }
          
          if (parseTimer) clearTimeout(parseTimer)
          if (buffer.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'text', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] text ingest hook:', deduplicated.length, 'infons, session:', session.id, 'round:', currentRound)
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] text ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] text finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 图像信息元提取
  async _startImageInfonRun({ targetType, targetKey, dataUrl, imageIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'image', imageIndex, _hash })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().imageParsingModel || 'gemma3:12b'
    const customProviders = get().customProviders
    
    if (!getModelModalities(configuredModel, customProviders).image) {
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Image not supported' }))
      return
    }
    
    const provider = customProviders?.[configuredModel]
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['image'], nowISO, { currentRound, existingInfons })
    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      let res
      if (provider) {
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: [{ type: 'text', text: 'Extract Situation Theory infons in compact format.' }, { type: 'image_url', image_url: { url: dataUrl } }] },
        ]
        const isOmni = configuredModel.toLowerCase().includes('omni')
        res = await fetch(`${provider.baseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...(provider.apiKey ? { 'Authorization': `Bearer ${provider.apiKey}` } : {}), 'Connection': 'keep-alive' },
          body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: isOmni ? 2000 : 4096, top_p: 0.95 }),
          signal: controller.signal, keepalive: true
        })
      } else {
        const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: 'Extract Situation Theory infons in compact format.', images: [stripDataUrl(dataUrl)] },
        ]
        res = await fetch(`${apiBase}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: configuredModel, messages, stream: true, options: { temperature: 0 } }),
          signal: controller.signal,
        })
      }
      
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      const streamHandler = provider ? streamOpenAIResponse : streamOllamaChatResponse
      await streamHandler(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          try {
            const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            const buffer = (curRun?.buffer || '') + content
            get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
            
            const { state: newState, yielded } = await incrementalExtractInfons(buffer, get().infonParsers?.[runId] || null)
            set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
            
            if (yielded?.length > 0) {
              get()._updateInfonRun(session.id, runId, r => {
                const infons = [...(r.resultJson?.infons || [])]
                yielded.forEach(ni => {
                  const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                  idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                })
                return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
              })
            }
          } catch (contentErr) {
            console.error('[InfonSlice] image content handler error:', contentErr)
          }
        }
        if (finish) {
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'image', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] image ingest hook:', deduplicated.length, 'infons')
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] image ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] image finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 音频信息元提取
  async _startAudioInfonRun({ targetType, targetKey, audio, audioIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const transcript = (audio.transcript || '').trim()
    if (!transcript) {
      const errorRun = { ...createInfonRun({ targetType, targetKey, modality: 'audio', audioIndex, _hash }), status: 'error', error: 'No transcript' }
      get()._appendInfonRun(session.id, errorRun)
      return
    }

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'audio', audioIndex, _hash })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const { baseUrl, headers, maxTokens } = getModelApiConfig(get, configuredModel)
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['audio'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons in compact format. Audio transcript:\n\n${transcript}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (curRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
          
          const performParsing = async () => {
            const run = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            if (!run) return
            const { state: newState, yielded } = await incrementalExtractInfons(run.buffer || '', get().infonParsers?.[runId] || null)
            set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
            if (yielded?.length > 0) {
              get()._updateInfonRun(session.id, runId, r => {
                const infons = [...(r.resultJson?.infons || [])]
                yielded.forEach(ni => {
                  const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                  idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                })
                return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
              })
            }
            lastParseTime = Date.now()
          }
          
          if (parseTimer) clearTimeout(parseTimer)
          if (buffer.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
          const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const parserState = get().infonParsers?.[runId]
          let finalInfons = [], parseSuccess = false
          
          if (parserState?.isCompact) {
            const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
            const result = parseCompactInfonsFormat(raw)
            if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
          } else {
            const sliced = extractFirstJSONObject(raw) || raw
            const { ok, value } = tryParseJSON(sliced)
            if (ok) {
              const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'audio', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
              finalInfons = normalized.infons || []; parseSuccess = true
            }
          }
          
          if (parseSuccess && finalInfons.length) {
            const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
            // === 主记忆流：写入向量索引库 ===
            console.log('[MemoryStream] audio ingest hook:', deduplicated.length, 'infons')
            try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] audio ingest error:', e) }
          } else {
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },
})

```

现在对音频的 handler 也同样处理：

```javascript
// 信息元提取 Slice
import { buildSystemPrompt } from '../../templates/infons.js'
import { generateId, tryParseJSON, extractFirstJSONObject, computeHashId, normalizeInfonOutput } from '../utils'
import { streamOpenAIResponse, streamOllamaChatResponse } from '../streamUtils'
import { incrementalExtractInfons } from '../infonParser'
import { deduplicateAndMergeInfons } from '../infonMerge'
import { getExistingInfons, createInfonRun, getModelApiConfig } from './infonHelpers'
import { stripDataUrl } from './messageHelpers'
import { getModelModalities } from '../../utils/modelUtils'

function buildInfonSystemPrompt(modalities, nowISO, options = {}) {
  const { currentRound = 1, existingInfons = [] } = options
  return buildSystemPrompt({
    modalities, includeExamples: false, currentRound, existingInfons,
    extraInstructions: `System time (ISO8601) = ${nowISO}. Set run_metadata.record_time to this value. For each situation and infon, if record_time is missing, set it to this value. Only set occur_time when it is explicitly expressed; otherwise omit.`
  })
}

export const createInfonSlice = (set, get) => ({
  // 状态
  infonSessions: {},
  lastPendingTextHash: null,
  lastPendingImageHashes: [],
  infonParsers: {},
  highlightedInfon: null,

  setHighlightedInfon(infon, run) { set({ highlightedInfon: infon ? { infon, run } : null }) },

  _getOrCreateInfonSession(sessionId) {
    let box = get().infonSessions?.[sessionId]
    if (!box) {
      box = { runs: [] }
      set(s => ({ infonSessions: { ...(s.infonSessions || {}), [sessionId]: box } }))
    }
    return box
  },

  _appendInfonRun(sessionId, run) {
    set(s => {
      const cur = s.infonSessions?.[sessionId] || { runs: [] }
      return { infonSessions: { ...(s.infonSessions || {}), [sessionId]: { runs: [...cur.runs, run] } } }
    })
  },

  _updateInfonRun(sessionId, runId, updater) {
    set(s => {
      const box = s.infonSessions?.[sessionId]
      if (!box) return {}
      return { infonSessions: { ...s.infonSessions, [sessionId]: { runs: box.runs.map(r => r.id === runId ? updater(r) : r) } } }
    })
  },

  getCurrentInfonRuns() {
    const session = get().getCurrentSession()
    return session ? (get().infonSessions?.[session.id]?.runs || []) : []
  },

  clearAllPendingInfons() {
    get()._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return
    try {
      const runs = (get().infonSessions?.[session.id]?.runs) || []
      runs.forEach(r => { if (r.targetType === 'pending' && r.status === 'running') try { r.controller?.abort?.() } catch (_) {} })
    } catch (_) {}
    set(s => {
      const box = s.infonSessions?.[session.id] || { runs: [] }
      return { infonSessions: { ...(s.infonSessions || {}), [session.id]: { runs: box.runs.filter(r => r.targetType !== 'pending') } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
  },

  adoptPendingInfonsToMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session || !messageId) return { adopted: 0, runIds: [] }
    let adopted = 0
    const adoptedRunIds = []
    set(s => {
      const box = s.infonSessions?.[session.id]
      if (!box) return {}
      const runs = box.runs.map(r => {
        if (r.targetType === 'pending') { adopted++; adoptedRunIds.push(r.id); return { ...r, targetType: 'message', targetKey: messageId } }
        return r
      })
      return { infonSessions: { ...s.infonSessions, [session.id]: { runs } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
    return { adopted, runIds: adoptedRunIds }
  },

  // 触发信息元提取（消息发送后）
  triggerInfonsForMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session) return
    const msg = session.messages.find(m => m.id === messageId)
    if (!msg || msg.role !== 'user') return
    
    const text = typeof msg.content === 'string' ? msg.content : ''
    const imgs = msg.images || []
    const audios = msg.audios || []
    
    if (text.trim()) get()._startTextInfonRun({ targetType: 'message', targetKey: messageId, text })
    if (imgs.length) imgs.forEach((dataUrl, idx) => get()._startImageInfonRun({ targetType: 'message', targetKey: messageId, dataUrl, imageIndex: idx }))
    if (audios.length) audios.forEach((audio, idx) => {
      const audioHash = computeHashId(audio.id + (audio.transcript || ''))
      get()._startAudioInfonRun({ targetType: 'message', targetKey: messageId, audio, audioIndex: idx, _hash: audioHash })
    })
  },

  // 别名方法：供外部调用
  startMessageInfons(messageId) { get().triggerInfonsForMessage(messageId) },

  // 启动 pending 信息元提取（用户输入时）
  startPendingInfons(text, images = [], audios = []) {
    const session = get().getCurrentSession()
    if (!session) return

    const runs = get().infonSessions?.[session.id]?.runs || []

    // 文本处理：计算 hash 判断是否需要重新提取
    const textHash = text?.trim() ? computeHashId(text.trim()) : null
    const lastTextHash = get().lastPendingTextHash
    if (textHash && textHash !== lastTextHash) {
      // 先中止并移除旧的 pending 文本 runs
      runs.filter(r => r.targetType === 'pending' && r.modality === 'text').forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return {
          infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'text')) } },
          lastPendingTextHash: textHash
        }
      })
      get()._startTextInfonRun({ targetType: 'pending', targetKey: 'pending', text: text.trim() })
    }

    // 图片处理：检查新增图片
    const imageHashes = (images || []).map(img => {
      const url = typeof img === 'string' ? img : img.url
      return computeHashId(url)
    })
    const lastImageHashes = get().lastPendingImageHashes || []
    const newImages = imageHashes.filter(h => !lastImageHashes.includes(h))
    // 移除已删除的图片对应的 runs
    const removedImages = lastImageHashes.filter(h => !imageHashes.includes(h))
    if (removedImages.length > 0) {
      runs.filter(r => r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash)).forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return { infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash))) } } }
      })
    }
    if (newImages.length > 0) {
      set({ lastPendingImageHashes: imageHashes })
      images.forEach((img, idx) => {
        const url = typeof img === 'string' ? img : img.url
        const hash = computeHashId(url)
        if (newImages.includes(hash)) {
          get()._startImageInfonRun({ targetType: 'pending', targetKey: 'pending', dataUrl: url, imageIndex: idx, _hash: hash })
        }
      })
    }

    // 音频处理
    audios.forEach((audio, idx) => {
      if (!audio.transcript?.trim()) return
      const audioHash = computeHashId(audio.id + audio.transcript)
      const existingRuns = get().infonSessions?.[session.id]?.runs || []
      const alreadyRunning = existingRuns.some(r => r.targetType === 'pending' && r._hash === audioHash && (r.status === 'running' || r.status === 'done'))
      if (!alreadyRunning) {
        get()._startAudioInfonRun({ targetType: 'pending', targetKey: 'pending', audio, audioIndex: idx, _hash: audioHash })
      }
    })
  },

  // 中止 pending 信息元提取
  abortPendingInfons() {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = get().infonSessions?.[session.id]?.runs || []
    runs.forEach(r => {
      if (r.targetType === 'pending' && r.status === 'running' && r.controller) {
        try { r.controller.abort() } catch (_) {}
      }
    })
  },

  // 文本信息元提取
  async _startTextInfonRun({ targetType, targetKey, text }) {
    const session = get().getCurrentSession()
    if (!session) return

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'text' })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const { provider, baseUrl, headers, maxTokens } = getModelApiConfig(get, configuredModel)
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['text'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons in compact format. Input text:\n\n${text}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (currentRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
          
          const performParsing = async () => {
            try {
              const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
              if (!curRun) return
              const { state: newState, yielded } = await incrementalExtractInfons(curRun.buffer || '', get().infonParsers?.[runId] || null)
              set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
              if (yielded?.length > 0) {
                get()._updateInfonRun(session.id, runId, r => {
                  const infons = [...(r.resultJson?.infons || [])]
                  yielded.forEach(ni => {
                    const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                    idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                  })
                  return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
                })
              }
              lastParseTime = Date.now()
            } catch (parseErr) {
              console.error('[InfonSlice] text parsing error:', parseErr)
            }
          }
          
          if (parseTimer) clearTimeout(parseTimer)
          if (buffer.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'text', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] text ingest hook:', deduplicated.length, 'infons, session:', session.id, 'round:', currentRound)
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] text ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] text finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 图像信息元提取
  async _startImageInfonRun({ targetType, targetKey, dataUrl, imageIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'image', imageIndex, _hash })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().imageParsingModel || 'gemma3:12b'
    const customProviders = get().customProviders
    
    if (!getModelModalities(configuredModel, customProviders).image) {
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Image not supported' }))
      return
    }
    
    const provider = customProviders?.[configuredModel]
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['image'], nowISO, { currentRound, existingInfons })
    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      let res
      if (provider) {
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: [{ type: 'text', text: 'Extract Situation Theory infons in compact format.' }, { type: 'image_url', image_url: { url: dataUrl } }] },
        ]
        const isOmni = configuredModel.toLowerCase().includes('omni')
        res = await fetch(`${provider.baseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...(provider.apiKey ? { 'Authorization': `Bearer ${provider.apiKey}` } : {}), 'Connection': 'keep-alive' },
          body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: isOmni ? 2000 : 4096, top_p: 0.95 }),
          signal: controller.signal, keepalive: true
        })
      } else {
        const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: 'Extract Situation Theory infons in compact format.', images: [stripDataUrl(dataUrl)] },
        ]
        res = await fetch(`${apiBase}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: configuredModel, messages, stream: true, options: { temperature: 0 } }),
          signal: controller.signal,
        })
      }
      
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      const streamHandler = provider ? streamOpenAIResponse : streamOllamaChatResponse
      await streamHandler(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          try {
            const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            const buffer = (curRun?.buffer || '') + content
            get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
            
            const { state: newState, yielded } = await incrementalExtractInfons(buffer, get().infonParsers?.[runId] || null)
            set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
            
            if (yielded?.length > 0) {
              get()._updateInfonRun(session.id, runId, r => {
                const infons = [...(r.resultJson?.infons || [])]
                yielded.forEach(ni => {
                  const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                  idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                })
                return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
              })
            }
          } catch (contentErr) {
            console.error('[InfonSlice] image content handler error:', contentErr)
          }
        }
        if (finish) {
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'image', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] image ingest hook:', deduplicated.length, 'infons')
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] image ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] image finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 音频信息元提取
  async _startAudioInfonRun({ targetType, targetKey, audio, audioIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const transcript = (audio.transcript || '').trim()
    if (!transcript) {
      const errorRun = { ...createInfonRun({ targetType, targetKey, modality: 'audio', audioIndex, _hash }), status: 'error', error: 'No transcript' }
      get()._appendInfonRun(session.id, errorRun)
      return
    }

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'audio', audioIndex, _hash })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const { baseUrl, headers, maxTokens } = getModelApiConfig(get, configuredModel)
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['audio'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons in compact format. Audio transcript:\n\n${transcript}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (curRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
          
          const performParsing = async () => {
            try {
              const run = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
              if (!run) return
              const { state: newState, yielded } = await incrementalExtractInfons(run.buffer || '', get().infonParsers?.[runId] || null)
              set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
              if (yielded?.length > 0) {
                get()._updateInfonRun(session.id, runId, r => {
                  const infons = [...(r.resultJson?.infons || [])]
                  yielded.forEach(ni => {
                    const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                    idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                  })
                  return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
                })
              }
              lastParseTime = Date.now()
            } catch (parseErr) {
              console.error('[InfonSlice] audio parsing error:', parseErr)
            }
          }
          
          if (parseTimer) clearTimeout(parseTimer)
          if (buffer.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'audio', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] audio ingest hook:', deduplicated.length, 'infons')
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] audio ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] audio finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },
})

```

现在添加僵尸 run 清理机制，在 `AgentPage` 中检测超时的 run 并自动清理：

我需要在 `infonSlice` 中添加一个僵尸 run 清理方法，让 AgentPage 可以在挂载时调用：

```javascript
// 信息元提取 Slice
import { buildSystemPrompt } from '../../templates/infons.js'
import { generateId, tryParseJSON, extractFirstJSONObject, computeHashId, normalizeInfonOutput } from '../utils'
import { streamOpenAIResponse, streamOllamaChatResponse } from '../streamUtils'
import { incrementalExtractInfons } from '../infonParser'
import { deduplicateAndMergeInfons } from '../infonMerge'
import { getExistingInfons, createInfonRun, getModelApiConfig } from './infonHelpers'
import { stripDataUrl } from './messageHelpers'
import { getModelModalities } from '../../utils/modelUtils'

function buildInfonSystemPrompt(modalities, nowISO, options = {}) {
  const { currentRound = 1, existingInfons = [] } = options
  return buildSystemPrompt({
    modalities, includeExamples: false, currentRound, existingInfons,
    extraInstructions: `System time (ISO8601) = ${nowISO}. Set run_metadata.record_time to this value. For each situation and infon, if record_time is missing, set it to this value. Only set occur_time when it is explicitly expressed; otherwise omit.`
  })
}

export const createInfonSlice = (set, get) => ({
  // 状态
  infonSessions: {},
  lastPendingTextHash: null,
  lastPendingImageHashes: [],
  infonParsers: {},
  highlightedInfon: null,

  setHighlightedInfon(infon, run) { set({ highlightedInfon: infon ? { infon, run } : null }) },

  _getOrCreateInfonSession(sessionId) {
    let box = get().infonSessions?.[sessionId]
    if (!box) {
      box = { runs: [] }
      set(s => ({ infonSessions: { ...(s.infonSessions || {}), [sessionId]: box } }))
    }
    return box
  },

  _appendInfonRun(sessionId, run) {
    set(s => {
      const cur = s.infonSessions?.[sessionId] || { runs: [] }
      return { infonSessions: { ...(s.infonSessions || {}), [sessionId]: { runs: [...cur.runs, run] } } }
    })
  },

  _updateInfonRun(sessionId, runId, updater) {
    set(s => {
      const box = s.infonSessions?.[sessionId]
      if (!box) return {}
      return { infonSessions: { ...s.infonSessions, [sessionId]: { runs: box.runs.map(r => r.id === runId ? updater(r) : r) } } }
    })
  },

  getCurrentInfonRuns() {
    const session = get().getCurrentSession()
    return session ? (get().infonSessions?.[session.id]?.runs || []) : []
  },

  // 清理僵尸 runs：超过指定时间仍处于 running 的 run 标记为 error
  cleanupZombieRuns(maxAgeMs = 120000) {
    const session = get().getCurrentSession()
    if (!session) return 0
    const runs = get().infonSessions?.[session.id]?.runs || []
    const now = Date.now()
    let cleaned = 0
    runs.forEach(run => {
      if (run.status === 'running' && run.createdAt && (now - run.createdAt) > maxAgeMs) {
        // 尝试中止
        if (run.controller) try { run.controller.abort() } catch (_) {}
        get()._updateInfonRun(session.id, run.id, r => ({ ...r, status: 'error', error: 'Timeout (zombie cleanup)' }))
        cleaned++
      }
    })
    if (cleaned > 0) console.log(`[InfonSlice] Cleaned up ${cleaned} zombie runs`)
    return cleaned
  },

  clearAllPendingInfons() {
    get()._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return
    try {
      const runs = (get().infonSessions?.[session.id]?.runs) || []
      runs.forEach(r => { if (r.targetType === 'pending' && r.status === 'running') try { r.controller?.abort?.() } catch (_) {} })
    } catch (_) {}
    set(s => {
      const box = s.infonSessions?.[session.id] || { runs: [] }
      return { infonSessions: { ...(s.infonSessions || {}), [session.id]: { runs: box.runs.filter(r => r.targetType !== 'pending') } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
  },

  adoptPendingInfonsToMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session || !messageId) return { adopted: 0, runIds: [] }
    let adopted = 0
    const adoptedRunIds = []
    set(s => {
      const box = s.infonSessions?.[session.id]
      if (!box) return {}
      const runs = box.runs.map(r => {
        if (r.targetType === 'pending') { adopted++; adoptedRunIds.push(r.id); return { ...r, targetType: 'message', targetKey: messageId } }
        return r
      })
      return { infonSessions: { ...s.infonSessions, [session.id]: { runs } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
    return { adopted, runIds: adoptedRunIds }
  },

  // 触发信息元提取（消息发送后）
  triggerInfonsForMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session) return
    const msg = session.messages.find(m => m.id === messageId)
    if (!msg || msg.role !== 'user') return
    
    const text = typeof msg.content === 'string' ? msg.content : ''
    const imgs = msg.images || []
    const audios = msg.audios || []
    
    if (text.trim()) get()._startTextInfonRun({ targetType: 'message', targetKey: messageId, text })
    if (imgs.length) imgs.forEach((dataUrl, idx) => get()._startImageInfonRun({ targetType: 'message', targetKey: messageId, dataUrl, imageIndex: idx }))
    if (audios.length) audios.forEach((audio, idx) => {
      const audioHash = computeHashId(audio.id + (audio.transcript || ''))
      get()._startAudioInfonRun({ targetType: 'message', targetKey: messageId, audio, audioIndex: idx, _hash: audioHash })
    })
  },

  // 别名方法：供外部调用
  startMessageInfons(messageId) { get().triggerInfonsForMessage(messageId) },

  // 启动 pending 信息元提取（用户输入时）
  startPendingInfons(text, images = [], audios = []) {
    const session = get().getCurrentSession()
    if (!session) return

    const runs = get().infonSessions?.[session.id]?.runs || []

    // 文本处理：计算 hash 判断是否需要重新提取
    const textHash = text?.trim() ? computeHashId(text.trim()) : null
    const lastTextHash = get().lastPendingTextHash
    if (textHash && textHash !== lastTextHash) {
      // 先中止并移除旧的 pending 文本 runs
      runs.filter(r => r.targetType === 'pending' && r.modality === 'text').forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return {
          infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'text')) } },
          lastPendingTextHash: textHash
        }
      })
      get()._startTextInfonRun({ targetType: 'pending', targetKey: 'pending', text: text.trim() })
    }

    // 图片处理：检查新增图片
    const imageHashes = (images || []).map(img => {
      const url = typeof img === 'string' ? img : img.url
      return computeHashId(url)
    })
    const lastImageHashes = get().lastPendingImageHashes || []
    const newImages = imageHashes.filter(h => !lastImageHashes.includes(h))
    // 移除已删除的图片对应的 runs
    const removedImages = lastImageHashes.filter(h => !imageHashes.includes(h))
    if (removedImages.length > 0) {
      runs.filter(r => r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash)).forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return { infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash))) } } }
      })
    }
    if (newImages.length > 0) {
      set({ lastPendingImageHashes: imageHashes })
      images.forEach((img, idx) => {
        const url = typeof img === 'string' ? img : img.url
        const hash = computeHashId(url)
        if (newImages.includes(hash)) {
          get()._startImageInfonRun({ targetType: 'pending', targetKey: 'pending', dataUrl: url, imageIndex: idx, _hash: hash })
        }
      })
    }

    // 音频处理
    audios.forEach((audio, idx) => {
      if (!audio.transcript?.trim()) return
      const audioHash = computeHashId(audio.id + audio.transcript)
      const existingRuns = get().infonSessions?.[session.id]?.runs || []
      const alreadyRunning = existingRuns.some(r => r.targetType === 'pending' && r._hash === audioHash && (r.status === 'running' || r.status === 'done'))
      if (!alreadyRunning) {
        get()._startAudioInfonRun({ targetType: 'pending', targetKey: 'pending', audio, audioIndex: idx, _hash: audioHash })
      }
    })
  },

  // 中止 pending 信息元提取
  abortPendingInfons() {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = get().infonSessions?.[session.id]?.runs || []
    runs.forEach(r => {
      if (r.targetType === 'pending' && r.status === 'running' && r.controller) {
        try { r.controller.abort() } catch (_) {}
      }
    })
  },

  // 文本信息元提取
  async _startTextInfonRun({ targetType, targetKey, text }) {
    const session = get().getCurrentSession()
    if (!session) return

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'text' })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const { provider, baseUrl, headers, maxTokens } = getModelApiConfig(get, configuredModel)
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['text'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons in compact format. Input text:\n\n${text}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (currentRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
          
          const performParsing = async () => {
            try {
              const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
              if (!curRun) return
              const { state: newState, yielded } = await incrementalExtractInfons(curRun.buffer || '', get().infonParsers?.[runId] || null)
              set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
              if (yielded?.length > 0) {
                get()._updateInfonRun(session.id, runId, r => {
                  const infons = [...(r.resultJson?.infons || [])]
                  yielded.forEach(ni => {
                    const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                    idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                  })
                  return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
                })
              }
              lastParseTime = Date.now()
            } catch (parseErr) {
              console.error('[InfonSlice] text parsing error:', parseErr)
            }
          }
          
          if (parseTimer) clearTimeout(parseTimer)
          if (buffer.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'text', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] text ingest hook:', deduplicated.length, 'infons, session:', session.id, 'round:', currentRound)
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] text ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] text finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 图像信息元提取
  async _startImageInfonRun({ targetType, targetKey, dataUrl, imageIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'image', imageIndex, _hash })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().imageParsingModel || 'gemma3:12b'
    const customProviders = get().customProviders
    
    if (!getModelModalities(configuredModel, customProviders).image) {
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Image not supported' }))
      return
    }
    
    const provider = customProviders?.[configuredModel]
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['image'], nowISO, { currentRound, existingInfons })
    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      let res
      if (provider) {
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: [{ type: 'text', text: 'Extract Situation Theory infons in compact format.' }, { type: 'image_url', image_url: { url: dataUrl } }] },
        ]
        const isOmni = configuredModel.toLowerCase().includes('omni')
        res = await fetch(`${provider.baseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...(provider.apiKey ? { 'Authorization': `Bearer ${provider.apiKey}` } : {}), 'Connection': 'keep-alive' },
          body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: isOmni ? 2000 : 4096, top_p: 0.95 }),
          signal: controller.signal, keepalive: true
        })
      } else {
        const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: 'Extract Situation Theory infons in compact format.', images: [stripDataUrl(dataUrl)] },
        ]
        res = await fetch(`${apiBase}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: configuredModel, messages, stream: true, options: { temperature: 0 } }),
          signal: controller.signal,
        })
      }
      
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      const streamHandler = provider ? streamOpenAIResponse : streamOllamaChatResponse
      await streamHandler(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          try {
            const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            const buffer = (curRun?.buffer || '') + content
            get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
            
            const { state: newState, yielded } = await incrementalExtractInfons(buffer, get().infonParsers?.[runId] || null)
            set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
            
            if (yielded?.length > 0) {
              get()._updateInfonRun(session.id, runId, r => {
                const infons = [...(r.resultJson?.infons || [])]
                yielded.forEach(ni => {
                  const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                  idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                })
                return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
              })
            }
          } catch (contentErr) {
            console.error('[InfonSlice] image content handler error:', contentErr)
          }
        }
        if (finish) {
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'image', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] image ingest hook:', deduplicated.length, 'infons')
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] image ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] image finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 音频信息元提取
  async _startAudioInfonRun({ targetType, targetKey, audio, audioIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const transcript = (audio.transcript || '').trim()
    if (!transcript) {
      const errorRun = { ...createInfonRun({ targetType, targetKey, modality: 'audio', audioIndex, _hash }), status: 'error', error: 'No transcript' }
      get()._appendInfonRun(session.id, errorRun)
      return
    }

    const currentRound = Math.floor((session.messages?.length || 0) / 2) + 1
    const existingInfons = getExistingInfons(get().getCurrentInfonRuns())
    const run = createInfonRun({ targetType, targetKey, modality: 'audio', audioIndex, _hash })
    const runId = run.id
    get()._appendInfonRun(session.id, run)

    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const { baseUrl, headers, maxTokens } = getModelApiConfig(get, configuredModel)
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['audio'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons in compact format. Audio transcript:\n\n${transcript}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      if (!res.ok) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Request failed' })); return }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          const curRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (curRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, r => ({ ...r, buffer }))
          
          const performParsing = async () => {
            try {
              const run = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
              if (!run) return
              const { state: newState, yielded } = await incrementalExtractInfons(run.buffer || '', get().infonParsers?.[runId] || null)
              set(s => ({ infonParsers: { ...s.infonParsers, [runId]: newState } }))
              if (yielded?.length > 0) {
                get()._updateInfonRun(session.id, runId, r => {
                  const infons = [...(r.resultJson?.infons || [])]
                  yielded.forEach(ni => {
                    const idx = infons.findIndex(i => i._objIndex === ni._objIndex)
                    idx >= 0 ? infons[idx] = { ...infons[idx], ...ni } : infons.push(ni)
                  })
                  return { ...r, status: 'running', resultJson: { ...r.resultJson, infons } }
                })
              }
              lastParseTime = Date.now()
            } catch (parseErr) {
              console.error('[InfonSlice] audio parsing error:', parseErr)
            }
          }
          
          if (parseTimer) clearTimeout(parseTimer)
          if (buffer.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            const parserState = get().infonParsers?.[runId]
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              const { ok, value } = tryParseJSON(sliced)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'audio', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              console.log('[MemoryStream] audio ingest hook:', deduplicated.length, 'infons')
              try { await get().ingestInfonsToMemory?.(deduplicated, session.id, currentRound) } catch (e) { console.error('[MemoryStream] audio ingest error:', e) }
            } else {
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
            }
          } catch (finishErr) {
            console.error('[InfonSlice] audio finish handler error:', finishErr)
            get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: String(finishErr?.message || 'Finish handler error') }))
          }
        }
      })
    } catch (err) {
      const aborted = err?.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },
})

```

现在在 AgentPage 中添加僵尸清理的 useEffect：

---

**User**

继续

