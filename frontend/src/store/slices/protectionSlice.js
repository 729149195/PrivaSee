// 保护建议 Slice
import { parseSSELine } from './privacyHelpers'

export const createProtectionSlice = (set, get) => ({
  protectionSuggestions: {},

  async generateProtectionSuggestions(text, editingMessageId = null) {
    const session = get().getCurrentSession()
    if (!session || !text?.trim()) return
    
    const currentInference = get().privacyInferences?.[session.id]
    if (!currentInference || currentInference.status !== 'done') {
      set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'error', error: '请先完成隐私推理分析', suggestions: [] } } }))
      return
    }
    
    const privacyRisks = currentInference.risks || []
    if (!privacyRisks.length) {
      set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'done', suggestions: [], error: null } } }))
      return
    }
    const runs = get().infonSessions?.[session.id]?.runs || []
    const allInfons = [], supersededIids = new Set()
    runs.forEach(run => {
      if (run.status === 'done' || run.status === 'running') {
        const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
        allInfons.push(...infons)
        infons.forEach(inf => { if (Array.isArray(inf._supersedes)) inf._supersedes.forEach(id => supersededIids.add(id)) })
      }
    })
    const validInfons = allInfons.filter(i => i.iid && !supersededIids.has(i.iid))
    
    const abortController = new AbortController()
    set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'running', suggestions: [], error: null, abortController } } }))
    
    let parseTimer = null
    try {
      const model = get().protectionSuggestionModel || 'deepseek-chat'
      const provider = get().customProviders?.[model]
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const apiKey = provider?.apiKey || ''
      if (!apiUrl) throw new Error('未配置API地址')
      
      const { fillProtectionPrompt, incrementalExtractSuggestions } = await import('../../templates/protection.js')
      const prompt = fillProtectionPrompt(text, privacyRisks, validInfons)
      const maxTokens = model.toLowerCase().includes('omni') ? 2000 : 4096
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}) },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], stream: true, temperature: 0.7, max_tokens: maxTokens }),
        signal: abortController.signal
      })
      if (!response.ok) throw new Error(`API error: ${response.status}`)
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '', parser = null
      const allSuggestions = new Map()
      let lastParseTime = 0
      
      const performParsing = () => {
        // 移除 <think> 块后解析
        const cleaned = buffer.replace(/<think>[\s\S]*?<\/think>/gi, '')
        const result = incrementalExtractSuggestions(cleaned, parser)
        parser = result.state
        
        for (const sug of result.yielded) {
          allSuggestions.set(sug._objIndex ?? allSuggestions.size, sug)
        }
        
        if (result.yielded.length > 0) {
          const arr = Array.from(allSuggestions.values()).sort((a, b) => {
            const order = { 'high_privacy': 0, 'balanced': 1, 'low_privacy': 2 }
            return (order[a.level] || 999) - (order[b.level] || 999)
          })
          set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { ...s.protectionSuggestions?.[session.id], status: 'running', suggestions: arr, error: null, abortController } } }))
        }
        lastParseTime = Date.now()
      }
      
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value, { stream: true }).split('\n')) {
          // 复用 parseSSELine：自动处理 content 和 reasoning_content
          // 只将 content 加入 buffer（reasoning_content 被跳过）
          const delta = parseSSELine(line)
          if (!delta) continue
          buffer += delta
          if (parseTimer) clearTimeout(parseTimer)
          if (Date.now() - lastParseTime >= 80) performParsing()
          else parseTimer = setTimeout(performParsing, 80)
        }
      }
      
      if (parseTimer) clearTimeout(parseTimer)
      if (buffer.length > 0 && !buffer.endsWith('\n')) buffer += '\n'
      performParsing()
      
      // 最终后备解析
      let finalSuggestions = Array.from(allSuggestions.values())
      if (finalSuggestions.length === 0 && buffer.trim()) {
        const { parseCompactProtectionFormat } = await import('../../templates/protection.js')
        const cleanBuf = buffer.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
        const parsed = parseCompactProtectionFormat(cleanBuf)
        if (parsed?.length > 0) finalSuggestions = parsed.map((s, i) => ({ ...s, _objIndex: i }))
      }
      
      finalSuggestions = finalSuggestions.map(s => ({ ...s, _isComplete: true })).sort((a, b) => {
        const order = { 'high_privacy': 0, 'balanced': 1, 'low_privacy': 2 }
        return (order[a.level] || 999) - (order[b.level] || 999)
      })
      
      if (finalSuggestions.length === 0) {
        set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'error', error: '未能解析保护建议，请重试', suggestions: [], abortController: null } } }))
      } else {
        set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'done', suggestions: finalSuggestions, error: null, abortController: null } } }))
      }
      
    } catch (err) {
      if (parseTimer) clearTimeout(parseTimer)
      const msg = err.name === 'AbortError' ? '请求已中止' : (err instanceof TypeError ? '网络连接失败' : err.message)
      set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'error', error: msg, suggestions: [], abortController: null } } }))
    }
  },
  
  abortProtectionSuggestions() {
    const session = get().getCurrentSession()
    if (!session) return
    const sug = get().protectionSuggestions?.[session.id]
    if (sug?.abortController) try { sug.abortController.abort() } catch (_) {}
  },
  
  clearProtectionSuggestions() {
    const session = get().getCurrentSession()
    if (!session) return
    set(s => { const n = { ...s.protectionSuggestions }; delete n[session.id]; return { protectionSuggestions: n } })
  },
})
