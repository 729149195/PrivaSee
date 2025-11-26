// 保护建议 Slice

export const createProtectionSlice = (set, get) => ({
  // 状态
  protectionSuggestions: {},

  // 生成保护建议
  async generateProtectionSuggestions(text, editingMessageId = null) {
    const session = get().getCurrentSession()
    if (!session || !text?.trim()) return
    
    const currentInference = get().privacyInferences?.[session.id]
    if (!currentInference || currentInference.status !== 'done') {
      set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'error', error: '请先完成隐私推理分析', suggestions: [] } } }))
      return
    }
    
    const privacyRisks = currentInference.risks || []
    const runs = get().infonSessions?.[session.id]?.runs || []
    const allInfons = [], supersededIids = new Set()
    
    runs.forEach(run => {
      if (run.status === 'done' || run.status === 'running') {
        const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
        allInfons.push(...infons)
        infons.forEach(infon => {
          if (Array.isArray(infon._supersedes)) infon._supersedes.forEach(id => supersededIids.add(id))
        })
      }
    })
    
    const validInfons = allInfons.filter(i => i.iid && !supersededIids.has(i.iid))
    const abortController = new AbortController()
    
    set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'running', suggestions: [], error: null, abortController } } }))
    
    try {
      const configuredModel = get().protectionSuggestionModel || 'qwen2.5:7b-instruct'
      const provider = get().customProviders?.[configuredModel]
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const apiKey = provider?.apiKey || ''
      if (!apiUrl) throw new Error('未配置API地址')
      
      const { fillProtectionPrompt, incrementalExtractSuggestions } = await import('../../templates/protection.js')
      const prompt = fillProtectionPrompt(text, privacyRisks, validInfons)
      const isOmni = configuredModel.toLowerCase().includes('omni')
      const maxTokens = isOmni ? 2000 : 4096
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}) },
        body: JSON.stringify({ model: configuredModel, messages: [{ role: 'user', content: prompt }], stream: true, temperature: 0.7, max_tokens: maxTokens }),
        signal: abortController.signal
      })
      
      if (!response.ok) throw new Error(`API error: ${response.status}`)
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '', parser = null
      const allSuggestions = new Map()
      
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6)
          if (data === '[DONE]') continue
          
          try {
            const parsed = JSON.parse(data)
            const content = parsed?.choices?.[0]?.delta?.content || ''
            if (content) {
              buffer += content
              const result = incrementalExtractSuggestions(buffer, parser)
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
            }
          } catch (_) {}
        }
      }
      
      // 最终解析：使用完整 buffer 进行解析（流式解析可能跳过未完成行）
      const { parseCompactProtectionFormat } = await import('../../templates/protection.js')
      let finalSuggestions = Array.from(allSuggestions.values())
      
      // 如果流式解析没有结果，尝试完整解析
      if (finalSuggestions.length === 0 && buffer.trim()) {
        const parsed = parseCompactProtectionFormat(buffer)
        if (parsed && parsed.length > 0) {
          finalSuggestions = parsed.map((s, i) => ({ ...s, _objIndex: i }))
        }
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
      const errorMsg = err.name === 'AbortError' ? '请求已中止' : (err instanceof TypeError ? '网络连接失败' : err.message)
      set(s => ({ protectionSuggestions: { ...s.protectionSuggestions, [session.id]: { status: 'error', error: errorMsg, suggestions: [], abortController: null } } }))
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
    set(s => {
      const newSug = { ...s.protectionSuggestions }
      delete newSug[session.id]
      return { protectionSuggestions: newSug }
    })
  },
})
