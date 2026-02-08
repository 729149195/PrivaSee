/**
 * 信息元提取辅助函数
 * 提供信息元运行的流式处理和解析的通用逻辑
 */

import { generateId, tryParseJSON, extractFirstJSONObject, normalizeInfonOutput } from '../utils.js'
import { streamOpenAIResponse, streamOllamaChatResponse } from '../streamUtils.js'
import { incrementalExtractInfons } from '../infonParser.js'
import { deduplicateAndMergeInfons } from '../infonMerge.js'

/**
 * 获取已完成运行的所有信息元
 */
export function getExistingInfons(currentRuns) {
  const completedRuns = currentRuns.filter(r => r.status === 'done')
  const existingInfons = []
  completedRuns.forEach(r => {
    if (r.resultJson?.infons) {
      existingInfons.push(...r.resultJson.infons)
    }
  })
  return existingInfons
}

/**
 * 创建信息元运行对象
 */
export function createInfonRun({ targetType, targetKey, modality, imageIndex, audioIndex, _hash }) {
  return {
    id: generateId(),
    targetType,
    targetKey,
    modality,
    ...(imageIndex !== undefined && { imageIndex }),
    ...(audioIndex !== undefined && { audioIndex }),
    ...(_hash && { _hash }),
    status: 'running',
    progress: 0,
    buffer: '',
    resultJson: null,
    createdAt: Date.now(),
    controller: null,
  }
}

/**
 * 处理流式 infon 解析的通用回调
 */
export function createStreamHandler({ get, set, sessionId, runId, existingInfons, nowISO, modality }) {
  let parseTimer = null
  let lastParseTime = 0
  const PARSE_DEBOUNCE_MS = 50

  const performParsing = async () => {
    const currentRun = get().infonSessions?.[sessionId]?.runs.find(x => x.id === runId)
    if (!currentRun) return
    
    const buffer = currentRun.buffer || ''
    const parserState = get().infonParsers?.[runId] || null
    const { state: newState, yielded } = await incrementalExtractInfons(buffer, parserState)
    
    set(state => ({
      infonParsers: {
        ...state.infonParsers,
        [runId]: newState
      }
    }))
    
    if (yielded && yielded.length > 0) {
      get()._updateInfonRun(sessionId, runId, (r) => {
        const currentInfons = r.resultJson?.infons || []
        const updatedInfons = [...currentInfons]
        
        yielded.forEach(newInfon => {
          const objIndex = newInfon._objIndex
          if (objIndex !== undefined) {
            const existingIndex = updatedInfons.findIndex(inf => inf._objIndex === objIndex)
            if (existingIndex >= 0) {
              updatedInfons[existingIndex] = { ...updatedInfons[existingIndex], ...newInfon }
            } else {
              updatedInfons.push(newInfon)
            }
          } else {
            updatedInfons.push(newInfon)
          }
        })
        
        return {
          ...r,
          status: 'running',
          resultJson: { ...r.resultJson, infons: updatedInfons }
        }
      })
    }
    
    lastParseTime = Date.now()
  }

  return async ({ content, finish }) => {
    if (typeof content === 'string' && content.length) {
      const currentRun = get().infonSessions?.[sessionId]?.runs.find(x => x.id === runId)
      const buffer = (currentRun?.buffer || '') + content
      get()._updateInfonRun(sessionId, runId, (r) => ({ ...r, buffer }))
      
      const now = Date.now()
      const timeSinceLastParse = now - lastParseTime
      
      if (parseTimer) clearTimeout(parseTimer)
      
      const hasCompleteLine = buffer.includes('\n')
      if (hasCompleteLine || timeSinceLastParse >= PARSE_DEBOUNCE_MS) {
        await performParsing()
      } else {
        parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
      }
    }
    
    if (finish) {
      if (parseTimer) clearTimeout(parseTimer)
      await handleInfonFinish({ get, sessionId, runId, existingInfons, nowISO, modality })
    }
  }
}

/**
 * 处理 infon 提取完成的通用逻辑
 */
async function handleInfonFinish({ get, sessionId, runId, existingInfons, nowISO, modality }) {
  const raw = get().infonSessions?.[sessionId]?.runs.find(x => x.id === runId)?.buffer || ''
  const parserState = get().infonParsers?.[runId]
  const isCompact = parserState?.isCompact
  
  let finalInfons = []
  let parseSuccess = false
  
  if (isCompact) {
    const { parseCompactInfonsFormat } = await import('../../templates/infons.js')
    const result = parseCompactInfonsFormat(raw)
    if (result && Array.isArray(result.infons)) {
      finalInfons = result.infons
      parseSuccess = true
    }
  } else {
    const sliced = extractFirstJSONObject(raw) || raw
    const { ok, value } = tryParseJSON(sliced)
    if (ok) {
      const session = get().getCurrentSession()
      const normalized = normalizeInfonOutput(value, {
        recordTimeISO: nowISO,
        defaultModality: modality,
        sessionId: sessionId,
        messageRound: Math.floor(((session?.messages || []).length) / 2) + 1,
        infonIndex: 0,
        infonType: 'desc'
      })
      finalInfons = normalized.infons || []
      parseSuccess = true
    }
  }
  
  if (parseSuccess && finalInfons.length > 0) {
    const deduplicated = deduplicateAndMergeInfons(finalInfons, existingInfons)
    const finalResult = { infons: deduplicated }
    get()._updateInfonRun(sessionId, runId, (r) => ({ ...r, status: 'done', progress: 100, resultJson: finalResult }))
    
    // === 主记忆流：将信息元写入向量索引库 (含 Top-K 关联绑定) ===
    try {
      const session = get().getCurrentSession()
      const roundNum = Math.floor(((session?.messages || []).length) / 2) + 1
      console.log('[MemoryStream] helper ingest hook:', deduplicated.length, 'infons, session:', sessionId, 'round:', roundNum)
      await get().ingestInfonsToMemory?.(deduplicated, sessionId, roundNum)
    } catch (memErr) {
      console.error('[MemoryStream] 信息元写入记忆流失败:', memErr)
    }
  } else {
    get()._updateInfonRun(sessionId, runId, (r) => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
  }
}

/**
 * 执行 API 请求并处理流式响应
 */
export async function executeInfonRequest({
  get, set, sessionId, runId, controller,
  apiUrl, headers, requestBody, existingInfons, nowISO, modality,
  useOllamaStream = false
}) {
  try {
    const res = await fetch(apiUrl, {
      method: 'POST',
      headers: { ...headers, 'Connection': 'keep-alive' },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
      keepalive: true
    })
    
    if (!res.ok) {
      const errText = await res.text().catch(() => '')
      get()._updateInfonRun(sessionId, runId, (r) => ({ ...r, status: 'error', error: errText || 'Request failed' }))
      return
    }
    
    const reader = res.body?.getReader()
    if (!reader) {
      get()._updateInfonRun(sessionId, runId, (r) => ({ ...r, status: 'error', error: 'No stream' }))
      return
    }
    
    const streamHandler = useOllamaStream ? streamOllamaChatResponse : streamOpenAIResponse
    const onDelta = createStreamHandler({ get, set, sessionId, runId, existingInfons, nowISO, modality })
    await streamHandler(reader, onDelta)
  } catch (err) {
    const aborted = err && err.name === 'AbortError'
    get()._updateInfonRun(sessionId, runId, (r) => ({
      ...r,
      status: aborted ? 'aborted' : 'error',
      error: aborted ? undefined : 'Network error'
    }))
  }
}

/**
 * 获取模型的 API 配置
 */
export function getModelApiConfig(get, modelId) {
  const provider = get().customProviders?.[modelId]
  const baseUrl = provider ? provider.baseUrl : get().baseUrl
  const headers = { 'Content-Type': 'application/json' }
  if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`
  
  const isOmniModel = modelId.toLowerCase().includes('omni')
  const maxTokens = isOmniModel ? 2000 : 4096
  
  return { provider, baseUrl, headers, maxTokens, isApiModel: !!provider }
}
