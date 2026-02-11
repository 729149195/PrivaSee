// 信息元提取 Slice
import { buildSystemPrompt } from '../../templates/infons.js'
import { generateId, tryParseJSON, extractFirstJSONObject, computeHashId, normalizeInfonOutput } from '../utils'
import { streamOpenAIResponse } from '../streamUtils'
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
      const pendingRunIds = runs.filter(r => r.targetType === 'pending').map(r => r.id).filter(Boolean)
      runs.forEach(r => { if (r.targetType === 'pending' && r.status === 'running') try { r.controller?.abort?.() } catch (_) {} })
      if (pendingRunIds.length > 0) {
        get().removeMemoryInfonsByRunIds?.(session.id, pendingRunIds, 'pending')
      }
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
    if (adoptedRunIds.length > 0) {
      get().promotePendingMemoryInfons?.(session.id, adoptedRunIds, messageId)
    }
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
      const removedPendingTextRunIds = runs
        .filter(r => r.targetType === 'pending' && r.modality === 'text')
        .map(r => r.id)
        .filter(Boolean)
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
      if (removedPendingTextRunIds.length > 0) {
        get().removeMemoryInfonsByRunIds?.(session.id, removedPendingTextRunIds, 'pending')
      }
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
      const removedPendingImageRunIds = runs
        .filter(r => r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash))
        .map(r => r.id)
        .filter(Boolean)
      runs.filter(r => r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash)).forEach(r => {
        if (r.status === 'running' && r.controller) try { r.controller.abort() } catch (_) {}
      })
      set(s => {
        const box = s.infonSessions?.[session.id] || { runs: [] }
        return { infonSessions: { ...s.infonSessions, [session.id]: { runs: box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'image' && removedImages.includes(r._hash))) } } }
      })
      if (removedPendingImageRunIds.length > 0) {
        get().removeMemoryInfonsByRunIds?.(session.id, removedPendingImageRunIds, 'pending')
      }
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
    // 60 秒超时：防止 Ollama 加载模型或无响应时 UI 永久卡住
    const timeoutId = setTimeout(() => { try { controller.abort() } catch (_) {} }, 60000)
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      console.log('[InfonSlice] text extraction start: model=%s, url=%s/chat/completions', configuredModel, baseUrl)
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      clearTimeout(timeoutId)
      console.log('[InfonSlice] text extraction response: status=%d', res.status)
      if (!res.ok) {
        const errBody = await res.text().catch(() => '')
        console.error('[InfonSlice] text extraction failed: status=%d, body=%s', res.status, errBody.slice(0, 200))
        get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: `Request failed (${res.status})` }))
        return
      }
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
              const infonsForMemory = deduplicated.map(inf => ({
                ...inf,
                _memory_target_type: targetType,
                _memory_target_key: targetKey,
                _memory_run_id: runId,
                _memory_modality: 'text',
              }))
              console.log('[MemoryStream] text ingest hook:', deduplicated.length, 'infons, session:', session.id, 'round:', currentRound)
              try { await get().ingestInfonsToMemory?.(infonsForMemory, session.id, currentRound) } catch (e) { console.error('[MemoryStream] text ingest error:', e) }
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
      if (!aborted) console.error('[InfonSlice] text extraction error:', err)
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : String(err?.message || 'Network error') }))
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
    const timeoutId = setTimeout(() => { try { controller.abort() } catch (_) {} }, 90000) // 图片提取 90s 超时
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      let res
      if (provider) {
        console.log('[InfonSlice] image extraction start (provider): model=%s, url=%s/chat/completions', configuredModel, provider.baseUrl)
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
        // 统一走 OpenAI 兼容端点 /v1/chat/completions，避免 /api/chat 在反向代理下流式连接被截断
        const ollamaBaseUrl = get().baseUrl || '/v1'
        console.log('[InfonSlice] image extraction start (ollama-compat): model=%s, url=%s/chat/completions', configuredModel, ollamaBaseUrl)
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: [
            { type: 'text', text: 'Extract Situation Theory infons in compact format.' },
            { type: 'image_url', image_url: { url: dataUrl } }
          ] },
        ]
        res = await fetch(`${ollamaBaseUrl}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Connection': 'keep-alive' },
          body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: 4096, top_p: 0.95 }),
          signal: controller.signal, keepalive: true
        })
      }
      
      clearTimeout(timeoutId)
      console.log('[InfonSlice] image extraction response: status=%d', res.status)
      if (!res.ok) {
        const errBody = await res.text().catch(() => '')
        console.error('[InfonSlice] image extraction failed: status=%d, body=%s', res.status, errBody.slice(0, 200))
        get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: `Request failed (${res.status})` }))
        return
      }
      const reader = res.body?.getReader()
      if (!reader) { get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: 'No stream' })); return }

      // 统一使用 OpenAI SSE 解析器（Ollama 也走 /v1/chat/completions 兼容端点）
      const streamHandler = streamOpenAIResponse

      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50
      let tokenCount = 0

      await streamHandler(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          tokenCount++
          if (tokenCount <= 3 || tokenCount % 20 === 0) {
            console.log('[InfonSlice] image stream token #%d: %s', tokenCount, JSON.stringify(content).slice(0, 100))
          }
          try {
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
                  console.log('[InfonSlice] image incremental yielded %d infons', yielded.length, yielded.map(i => i.iid || i._objIndex))
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
                console.error('[InfonSlice] image parsing error:', parseErr)
              }
            }

            if (parseTimer) clearTimeout(parseTimer)
            // 优化：仅当当前 token 包含换行符（可能完成了一行）或距上次解析超过去抖间隔时触发
            if (content.includes('\n') || Date.now() - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
            else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
          } catch (contentErr) {
            console.error('[InfonSlice] image content handler error:', contentErr)
          }
        }
        if (finish) {
          console.log('[InfonSlice] image stream FINISH received, total tokens: %d', tokenCount)
          if (parseTimer) clearTimeout(parseTimer)
          try {
            const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
            console.log('[InfonSlice] image raw buffer length: %d, first 500 chars:\n%s', raw.length, raw.slice(0, 500))
            const parserState = get().infonParsers?.[runId]
            console.log('[InfonSlice] image parserState: isCompact=%s, formatDetected=%s', parserState?.isCompact, parserState?.formatDetected)
            let finalInfons = [], parseSuccess = false
            
            if (parserState?.isCompact) {
              const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
              const result = parseCompactInfonsFormat(raw)
              console.log('[InfonSlice] image compact parse result: %d infons', result?.infons?.length || 0)
              if (result?.infons) { finalInfons = result.infons; parseSuccess = true }
            } else {
              const sliced = extractFirstJSONObject(raw) || raw
              console.log('[InfonSlice] image JSON parse attempt, sliced length: %d', sliced.length)
              const { ok, value } = tryParseJSON(sliced)
              console.log('[InfonSlice] image JSON parse ok=%s', ok)
              if (ok) {
                const normalized = normalizeInfonOutput(value, { recordTimeISO: nowISO, defaultModality: 'image', sessionId: session.id, messageRound: currentRound, infonIndex: 0, infonType: 'desc' })
                finalInfons = normalized.infons || []; parseSuccess = true
              }
            }
            
            console.log('[InfonSlice] image final: parseSuccess=%s, finalInfons=%d', parseSuccess, finalInfons.length)
            if (parseSuccess && finalInfons.length) {
              const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
              get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'done', progress: 100, resultJson: { infons: deduplicated } }))
              // === 主记忆流：写入向量索引库 ===
              const infonsForMemory = deduplicated.map(inf => ({
                ...inf,
                _memory_target_type: targetType,
                _memory_target_key: targetKey,
                _memory_run_id: runId,
                _memory_modality: 'image',
              }))
              console.log('[MemoryStream] image ingest hook:', deduplicated.length, 'infons')
              try { await get().ingestInfonsToMemory?.(infonsForMemory, session.id, currentRound) } catch (e) { console.error('[MemoryStream] image ingest error:', e) }
            } else {
              console.error('[InfonSlice] image extraction failed: Invalid JSON output. Raw:\n%s', raw.slice(0, 1000))
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
      if (!aborted) console.error('[InfonSlice] image extraction error:', err)
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : String(err?.message || 'Network error') }))
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
    const timeoutId = setTimeout(() => { try { controller.abort() } catch (_) {} }, 60000) // 60s 超时
    get()._updateInfonRun(session.id, runId, r => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { ...headers, 'Connection': 'keep-alive' },
        body: JSON.stringify({ model: configuredModel, messages, temperature: 0, stream: true, max_tokens: maxTokens, top_p: 0.95 }),
        signal: controller.signal, keepalive: true
      })
      clearTimeout(timeoutId)
      if (!res.ok) {
        const errBody = await res.text().catch(() => '')
        console.error('[InfonSlice] audio extraction failed: status=%d', res.status)
        get()._updateInfonRun(session.id, runId, r => ({ ...r, status: 'error', error: `Request failed (${res.status})` }))
        return
      }
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
              const infonsForMemory = deduplicated.map(inf => ({
                ...inf,
                _memory_target_type: targetType,
                _memory_target_key: targetKey,
                _memory_run_id: runId,
                _memory_modality: 'audio',
              }))
              console.log('[MemoryStream] audio ingest hook:', deduplicated.length, 'infons')
              try { await get().ingestInfonsToMemory?.(infonsForMemory, session.id, currentRound) } catch (e) { console.error('[MemoryStream] audio ingest error:', e) }
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
      if (!aborted) console.error('[InfonSlice] audio extraction error:', err)
      get()._updateInfonRun(session.id, runId, r => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : String(err?.message || 'Network error') }))
    }
  },
})
