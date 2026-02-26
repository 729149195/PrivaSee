// 隐私推理 Slice
import {
  collectInfonsForInference,
  mergeRisks, cleanAndParseBuffer, parseSSELine
} from './privacyHelpers'

function _buildPrivacyInfonDebugRows(list = [], sourceLabel = '') {
  return (Array.isArray(list) ? list : []).map((inf, idx) => ({
    idx,
    source: sourceLabel,
    iid: inf?.iid || '',
    infon_type: inf?.infon_type || '',
    entity: inf?.entity || '',
    attribute: inf?.attribute || '',
    temporal: inf?.temporal || '',
    spatial: inf?.spatial || '',
    relation_name: inf?.relation_name || '',
    arg_refs: Array.isArray(inf?.arg_refs) ? inf.arg_refs.join(', ') : '',
    retrieval_similarity: inf?.retrieval_similarity ?? '',
  }))
}

export const createPrivacySlice = (set, get) => ({
  privacyInferences: {},
  privacyParsers: {},
  privacyIdMaps: {},

  async startPrivacyInference() {
    const session = get().getCurrentSession()
    if (!session) return
    const { selectedLaw, infonSessions } = get()
    if (!selectedLaw?.data) return
    const allInfons = collectInfonsForInference(infonSessions, session.id)
    if (!allInfons.length) return
    
    const previousRisks = get().privacyInferences?.[session.id]?.status === 'done'
      ? (get().privacyInferences[session.id].risks || []) : []
    const abortController = new AbortController()
    
    set(s => ({
      privacyInferences: { ...s.privacyInferences, [session.id]: {
        status: 'running', risks: [], buffer: '', abortController,
        lawKey: selectedLaw.key, previousRisks, createdAt: Date.now(), updatedAt: Date.now()
      }},
      privacyParsers: { ...(s.privacyParsers || {}), [session.id]: null }
    }))
    
    let parseTimer = null
    let aborted = false
    
    try {
      // 记忆流触发检测（5s 超时）
      let memoryInfons = []
      try {
        memoryInfons = await Promise.race([
          get().triggerCheckAndRetrieve?.(allInfons) || Promise.resolve([]),
          new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 5000))
        ]) || []
      } catch (_) {}
      
      const infons = memoryInfons.length > 0 ? [...memoryInfons, ...allInfons] : allInfons
      // 调试输出：每次隐私风险分析都打印参与推理的信息元（当前 + 回溯插入 + 最终输入）
      try {
        const currentRows = _buildPrivacyInfonDebugRows(allInfons, 'current')
        const memoryRows = _buildPrivacyInfonDebugRows(memoryInfons, 'memory_retrieved')
        const finalRows = [
          ...memoryRows,
          ...currentRows,
        ].map((row, idx) => ({ ...row, final_order: idx }))
        console.groupCollapsed(
          `[PrivacyInference] session=${session.id} current=${allInfons.length} memory=${memoryInfons.length} total=${infons.length}`
        )
        if (currentRows.length > 0) {
          console.log('[PrivacyInference] current infons')
          console.table(currentRows)
        } else {
          console.log('[PrivacyInference] current infons: (empty)')
        }
        if (memoryRows.length > 0) {
          console.log('[PrivacyInference] memory retrieved infons')
          console.table(memoryRows)
        } else {
          console.log('[PrivacyInference] memory retrieved infons: (empty)')
        }
        if (finalRows.length > 0) {
          console.log('[PrivacyInference] final infons for analysis (input order)')
          console.table(finalRows)
        } else {
          console.log('[PrivacyInference] final infons for analysis: (empty)')
        }
        console.log('[PrivacyInference] final infon payload', infons)
        console.groupEnd()
      } catch (_) {}
      const model = get().infonPrivacyInferenceModel || 'deepseek-chat'
      const think = !!get().infonPrivacyInferenceThinkMode
      const provider = get().customProviders?.[model]
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const apiKey = provider?.apiKey || ''
      const maxTokens = model.toLowerCase().includes('omni') ? 2000 : 4096
      
      const { fillPromptTemplate } = await import('../../templates/inference.js')
      const { prompt, lawIdMap, infonIdMap, isEmpty, emptyReason } = fillPromptTemplate(infons, selectedLaw.data)
      
      if (isEmpty) {
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: {
          status: 'done', risks: [], buffer: '', emptyReason,
          lawKey: selectedLaw.key, createdAt: Date.now(), updatedAt: Date.now()
        }}}))
        return
      }
      
      set(s => ({ privacyIdMaps: { ...s.privacyIdMaps, [session.id]: { lawIdMap, infonIdMap } } }))
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Connection': 'keep-alive', ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}) },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], stream: true, temperature: 0.5, max_tokens: maxTokens, top_p: 0.9, think }),
        signal: abortController.signal, keepalive: true
      })
      if (!response.ok) throw new Error(`API error: ${response.status}`)
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let lastParseTime = 0
      
      const performParsing = async () => {
        if (aborted) return
        const cleaned = buffer.replace(/<think>[\s\S]*?<\/think>/gi, '')
        const { incrementalExtractRisks, resolveRiskIds } = await import('../../templates/inference.js')
        if (aborted) return
        const parserState = get().privacyParsers?.[session.id] || null
        const { state: newState, yielded } = incrementalExtractRisks(cleaned, parserState)
        set(s => ({ privacyParsers: { ...s.privacyParsers, [session.id]: newState } }))
        if (yielded?.length > 0) {
          const idMaps = get().privacyIdMaps?.[session.id]
          const resolved = yielded.map(r => idMaps ? resolveRiskIds(r, idMaps.lawIdMap, idMaps.infonIdMap) : r)
          set(s => {
            const updated = mergeRisks(s.privacyInferences?.[session.id]?.risks || [], resolved)
            return { privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'running', risks: updated, buffer, updatedAt: Date.now() } } }
          })
        }
        lastParseTime = Date.now()
      }
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value, { stream: true }).split('\n')) {
          const delta = parseSSELine(line)
          if (!delta) continue
          buffer += delta
          if (parseTimer) clearTimeout(parseTimer)
          if (Date.now() - lastParseTime >= 50) await performParsing()
          else parseTimer = setTimeout(performParsing, 50)
        }
      }
      
      if (parseTimer) { clearTimeout(parseTimer); parseTimer = null }
      if (buffer.length > 0 && !buffer.endsWith('\n')) buffer += '\n'
      await performParsing()
      
      // 最终解析：先试 JSON，再试紧凑格式
      const parseResult = cleanAndParseBuffer(buffer)
      if (parseResult.success) {
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'done', risks: parseResult.risks, buffer: parseResult.cleanBuffer, abortController: null, updatedAt: Date.now() } } }))
        return
      }
      
      const cur = get().privacyInferences?.[session.id]
      if (!cur?.risks?.length && buffer.length > 0 && parseResult.isCompact) {
        try {
          const { parseCompactFormat, resolveRiskIds } = await import('../../templates/inference.js')
          const compactResult = parseCompactFormat(buffer.replace(/<think>[\s\S]*?<\/think>/gi, '').trim())
          if (compactResult?.risks?.length > 0) {
            const idMaps = get().privacyIdMaps?.[session.id]
            const resolved = compactResult.risks.map(r => idMaps ? resolveRiskIds(r, idMaps.lawIdMap, idMaps.infonIdMap) : r)
            set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'done', risks: resolved, abortController: null, updatedAt: Date.now() } } }))
            return
          }
        } catch (_) {}
      }
      
      if (!cur?.risks?.length && !buffer.length) {
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'error', error: 'No response', abortController: null, updatedAt: Date.now() } } }))
        return
      }
      set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'done', abortController: null, updatedAt: Date.now() } } }))
      
    } catch (err) {
      aborted = true
      if (parseTimer) { clearTimeout(parseTimer); parseTimer = null }
      if (err.name === 'AbortError') {
        const prev = get().privacyInferences?.[session.id]?.previousRisks || []
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: prev.length ? 'done' : 'aborted', risks: prev, abortController: null, updatedAt: Date.now() } } }))
      } else {
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'error', error: err.message, abortController: null, updatedAt: Date.now() } } }))
      }
    }
  },
  
  abortPrivacyInference() {
    const session = get().getCurrentSession()
    if (!session) return
    const inf = get().privacyInferences?.[session.id]
    if (inf?.abortController) try { inf.abortController.abort() } catch (_) {}
  },
  
  clearPrivacyInference() {
    const session = get().getCurrentSession()
    if (!session) return
    set(s => {
      const inf = { ...s.privacyInferences }, par = { ...s.privacyParsers }
      delete inf[session.id]; delete par[session.id]
      return { privacyInferences: inf, privacyParsers: par }
    })
  },
  
  clearCurrentInferenceAndRestore() {
    const session = get().getCurrentSession()
    if (!session) return
    const cur = get().privacyInferences?.[session.id]
    if (!cur) return
    if (cur.status === 'running') {
      if (cur.abortController) try { cur.abortController.abort() } catch (_) {}
    } else if (cur.status === 'done') {
      const prev = cur.previousRisks || []
      set(s => ({
        privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: prev.length ? 'done' : 'idle', risks: prev, buffer: '', updatedAt: Date.now() } },
        privacyParsers: { ...s.privacyParsers, [session.id]: null }
      }))
    }
  },
})
