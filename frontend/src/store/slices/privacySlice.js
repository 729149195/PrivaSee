// 隐私推理 Slice
import {
  collectInfonsForInference,
  mergeRisks, cleanAndParseBuffer, parseSSELine
} from './privacyHelpers'

export const createPrivacySlice = (set, get) => ({
  // 状态
  privacyInferences: {},
  privacyParsers: {},

  // 启动隐私推理
  async startPrivacyInference() {
    const session = get().getCurrentSession()
    if (!session) return
    
    const { selectedLaw, infonSessions } = get()
    if (!selectedLaw?.data) return
    
    const allInfons = collectInfonsForInference(infonSessions, session.id)
    if (!allInfons.length) return
    
    const previousInference = get().privacyInferences?.[session.id]
    const previousRisks = previousInference?.status === 'done' ? (previousInference.risks || []) : []
    const abortController = new AbortController()
    
    set(s => ({
      privacyInferences: {
        ...s.privacyInferences,
        [session.id]: {
          status: 'running', risks: [], buffer: '', abortController,
          lawKey: selectedLaw.key, previousRisks, createdAt: Date.now(), updatedAt: Date.now()
        }
      },
      privacyParsers: { ...(s.privacyParsers || {}), [session.id]: null }
    }))
    
    try {
      const configuredModel = get().infonPrivacyInferenceModel || 'deepseek-chat'
      const provider = get().customProviders?.[configuredModel]
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const apiKey = provider?.apiKey || ''
      const isOmni = configuredModel.toLowerCase().includes('omni')
      const maxTokens = isOmni ? 2000 : 4096
      
      const { fillPromptTemplate } = await import('../../templates/inference.js')
      const prompt = fillPromptTemplate(allInfons, selectedLaw.data, null, [])
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Connection': 'keep-alive', ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}) },
        body: JSON.stringify({ model: configuredModel, messages: [{ role: 'user', content: prompt }], stream: true, temperature: 0.5, max_tokens: maxTokens, top_p: 0.9 }),
        signal: abortController.signal, keepalive: true
      })
      
      if (!response.ok) throw new Error(`API error: ${response.status}`)
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let parseTimer = null, lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 50
      
      const performParsing = async () => {
        let cleanedBuffer = buffer.replace(/<think>[\s\S]*?<\/think>/gi, '')
        const { incrementalExtractRisks } = await import('../../templates/inference.js')
        const parserState = get().privacyParsers?.[session.id] || null
        const { state: newState, yielded } = incrementalExtractRisks(cleanedBuffer, parserState)
        
        set(s => ({ privacyParsers: { ...s.privacyParsers, [session.id]: newState } }))
        
        if (yielded?.length > 0) {
          set(s => {
            const currentRisks = s.privacyInferences?.[session.id]?.risks || []
            const updatedRisks = mergeRisks(currentRisks, yielded)
            return { privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'running', risks: updatedRisks, buffer, updatedAt: Date.now() } } }
          })
        }
        lastParseTime = Date.now()
      }
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        for (const line of chunk.split('\n')) {
          const contentDelta = parseSSELine(line)
          if (!contentDelta) continue
          buffer += contentDelta
          const now = Date.now()
          if (parseTimer) clearTimeout(parseTimer)
          if (now - lastParseTime >= PARSE_DEBOUNCE_MS) await performParsing()
          else parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
        }
      }
      
      if (parseTimer) clearTimeout(parseTimer)
      if (buffer.length > 0 && !buffer.endsWith('\n')) buffer += '\n'
      await performParsing()
      
      const parseResult = cleanAndParseBuffer(buffer)
      
      if (parseResult.success) {
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'done', risks: parseResult.risks, buffer: parseResult.cleanBuffer, abortController: null, updatedAt: Date.now() } } }))
        return
      }
      
      const currentState = get().privacyInferences?.[session.id]
      if (!currentState?.risks?.length && !buffer.length) {
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'error', error: 'No response', abortController: null, updatedAt: Date.now() } } }))
        return
      }
      
      set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'done', abortController: null, updatedAt: Date.now() } } }))
      
    } catch (err) {
      if (err.name === 'AbortError') {
        const currentState = get().privacyInferences?.[session.id]
        const previousRisks = currentState?.previousRisks || []
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: previousRisks.length ? 'done' : 'aborted', risks: previousRisks, abortController: null, updatedAt: Date.now() } } }))
      } else {
        set(s => ({ privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: 'error', error: err.message, abortController: null, updatedAt: Date.now() } } }))
      }
    }
  },
  
  abortPrivacyInference() {
    const session = get().getCurrentSession()
    if (!session) return
    const inference = get().privacyInferences?.[session.id]
    if (inference?.abortController) try { inference.abortController.abort() } catch (_) {}
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
    const currentInference = get().privacyInferences?.[session.id]
    if (!currentInference) return
    
    if (currentInference.status === 'running') {
      if (currentInference.abortController) try { currentInference.abortController.abort() } catch (_) {}
    } else if (currentInference.status === 'done') {
      const previousRisks = currentInference.previousRisks || []
      set(s => ({
        privacyInferences: { ...s.privacyInferences, [session.id]: { ...s.privacyInferences[session.id], status: previousRisks.length ? 'done' : 'idle', risks: previousRisks, previousRisks: previousRisks.length ? previousRisks : undefined, buffer: '', updatedAt: Date.now() } },
        privacyParsers: { ...s.privacyParsers, [session.id]: null }
      }))
    }
  },
})
