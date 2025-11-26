// 会话管理 Slice
import { createEmptySession } from '../utils'
import { getDefaultModelsConfig } from '../../config/defaultModelsConfig'
import { deleteSessionFiles } from '../../utils/fileStorage'
import { SESSION_TITLE_SYSTEM_PROMPT, buildSessionTitleUserPrompt, cleanGeneratedTitle, generateOcrTitle } from '../../templates/sessionTitle.js'

export const createSessionSlice = (set, get) => ({
  // 状态
  sessions: [createEmptySession()],
  currentSessionId: null,
  isGenerating: false,
  abortController: null,

  _ensureCurrentSession() {
    const { sessions, currentSessionId } = get()
    if (sessions.length === 0) {
      const newSession = createEmptySession()
      set({ sessions: [newSession], currentSessionId: newSession.id })
      return
    }
    if (!currentSessionId && sessions.length > 0) set({ currentSessionId: sessions[0].id })
  },

  getCurrentSession() {
    const { sessions, currentSessionId } = get()
    return sessions.find(s => s.id === currentSessionId) || null
  },

  createSession() {
    const newSession = createEmptySession()
    set(s => ({ sessions: [newSession, ...s.sessions], currentSessionId: newSession.id }))
  },

  switchSession(id) { set({ currentSessionId: id }) },

  deleteSession(id) {
    set(s => {
      const nextSessions = s.sessions.filter(x => x.id !== id)
      if (nextSessions.length === 0) {
        const newSession = createEmptySession()
        return { sessions: [newSession], currentSessionId: newSession.id, sessionKeywords: {} }
      }
      const nextCurrent = s.currentSessionId === id ? nextSessions[0]?.id : s.currentSessionId
      const updatedKeywords = { ...s.sessionKeywords }
      delete updatedKeywords[id]
      return { sessions: nextSessions, currentSessionId: nextCurrent, sessionKeywords: updatedKeywords }
    })
    deleteSessionFiles(id).catch(err => console.error('[deleteSession] 清理失败:', err))
  },

  renameSession(id, title) {
    set(s => ({ sessions: s.sessions.map(x => x.id === id ? { ...x, title, updatedAt: Date.now() } : x) }))
  },

  async generateSessionTitle(sessionId) {
    const session = get().sessions.find(s => s.id === sessionId)
    if (!session || !session.title.startsWith('New chat')) return
    
    const firstUserMessage = session.messages.find(msg => msg.role === 'user')
    if (!firstUserMessage) return
    
    // OCR 消息特殊命名
    if (firstUserMessage.files?.length > 0 && firstUserMessage.commands?.length > 0) {
      const title = generateOcrTitle(firstUserMessage.commands, firstUserMessage.files)
      get().renameSession(sessionId, title)
      return
    }
    
    // 提取消息内容
    const contentParts = []
    if (typeof firstUserMessage.content === 'string') {
      const text = firstUserMessage.content.replace(/<audio>([\s\S]*?)<\/audio>/gi, '$1').trim()
      if (text) contentParts.push(text)
    } else if (Array.isArray(firstUserMessage.content)) {
      const texts = firstUserMessage.content.filter(p => p.type === 'text').map(p => p.text)
      if (texts.length) contentParts.push(texts.join(' '))
    }
    
    // 提取图片分析
    const imageAnalysisMap = firstUserMessage.imageAnalysis || {}
    ;(firstUserMessage.images || []).forEach(url => {
      const analysis = imageAnalysisMap[url]?.trim()?.slice(0, 200)
      if (analysis) contentParts.push(analysis)
    })
    
    const content = contentParts.join(' ')
    if (!content.trim()) return
    
    try {
      const configuredModel = get().infonExtractionModel || 'deepseek-chat'
      const provider = get().customProviders?.[configuredModel]
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const headers = { 'Content-Type': 'application/json' }
      if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: configuredModel,
          messages: [
            { role: 'system', content: SESSION_TITLE_SYSTEM_PROMPT },
            { role: 'user', content: buildSessionTitleUserPrompt(content.slice(0, 500)) }
          ],
          temperature: 0.7,
          max_tokens: 20
        })
      })
      
      if (!response.ok) return
      const data = await response.json()
      const title = cleanGeneratedTitle(data.choices?.[0]?.message?.content?.trim())
      if (title) get().renameSession(sessionId, title)
    } catch (error) {
      console.error('[Session Title] 生成失败:', error)
    }
  },

  _appendMessage(sessionId, message) {
    set(s => ({
      sessions: s.sessions.map(x => x.id !== sessionId ? x : { ...x, updatedAt: Date.now(), messages: [...x.messages, message] })
    }))
  },

  _updateMessage(sessionId, messageId, updater) {
    set(s => ({
      sessions: s.sessions.map(x => {
        if (x.id !== sessionId) return x
        return { ...x, messages: x.messages.map(m => m.id === messageId ? updater(m) : m), updatedAt: Date.now() }
      })
    }))
  },

  stopGenerating() {
    const { abortController } = get()
    try { abortController?.abort() } catch (_) {}
    set({ isGenerating: false, abortController: null })
  },
})
