// 用户与历史数据 Slice
import { loadUserSessions, saveUserSessions } from '../../users/historyStorage'
import { getDefaultModelsConfig } from '../../config/defaultModelsConfig'
import { createEmptySession } from '../utils'

export const createUserSlice = (set, get) => ({
  currentUserId: null,

  setCurrentUser: (userId) => {
    // 切换用户前清空记忆流前端缓存 (防止用户间数据串扰)
    set({
      currentUserId: userId,
      memoryStreamLastIngest: null,
      memoryRetrievedInfons: [],
      memoryTriggerResult: null,
      memoryBacktraceCache: {},
      memoryStreamStatus: null,
      memoryVisualizationData: null,
    })
    if (userId) get()._loadUserHistory(userId)
  },
  
  clearCurrentUser: () => {
    const { currentUserId } = get()
    if (currentUserId) get()._saveUserHistory(currentUserId)
    const emptySession = createEmptySession()
    set({
      currentUserId: null, sessions: [emptySession], currentSessionId: emptySession.id,
      model: getDefaultModelsConfig().conversationModel, infonSessions: {},
      privacyInferences: {},
      // 切换用户时清空记忆流前端状态 (后端数据按用户隔离，不受影响)
      memoryStreamLastIngest: null,
      memoryRetrievedInfons: [],
      memoryTriggerResult: null,
      memoryBacktraceCache: {},
      memoryStreamStatus: null,
      memoryVisualizationData: null,
    })
  },
  
  clearAllData: () => {
    // 清空当前用户的后端记忆流数据 (按用户隔离)
    try { get().clearMemoryStream?.() } catch (_) {}
    
    const emptySession = createEmptySession()
    set({
      sessions: [emptySession], currentSessionId: emptySession.id,
      model: getDefaultModelsConfig().conversationModel, infonSessions: {},
      privacyInferences: {}, protectionSuggestions: {},
      customPrivacyItems: [], selectedPrivacyItems: [],
      // 清空主记忆流前端状态
      memoryStreamLastIngest: null,
      memoryRetrievedInfons: [],
      memoryTriggerResult: null,
      memoryBacktraceCache: {},
      memoryStreamStatus: null,
      memoryVisualizationData: null,
    })
  },

  _loadUserHistory(userId) {
    try {
      const data = loadUserSessions(userId, getDefaultModelsConfig())
      if (data?.sessions?.length > 0) {
        set({
          sessions: data.sessions,
          infonSessions: data.infonSessions || {},
          privacyInferences: data.privacyInferences || {},
          currentSessionId: data.sessions[0]?.id || null,
          customPrivacyItems: data.customPrivacyItems || [],
          selectedLawIdx: data.selectedLawIdx ?? 0,
          selectedPrivacyItems: data.selectedPrivacyItems || [],
          model: data.conversationModel || getDefaultModelsConfig().conversationModel,
          infonExtractionModel: data.infonExtractionModel || getDefaultModelsConfig().infonExtractionModel,
          infonPrivacyInferenceModel: data.infonPrivacyInferenceModel || getDefaultModelsConfig().infonPrivacyInferenceModel,
          imageParsingModel: data.imageParsingModel || getDefaultModelsConfig().imageParsingModel,
          protectionSuggestionModel: data.protectionSuggestionModel || getDefaultModelsConfig().protectionSuggestionModel,
          autoPrivacyInference: data.autoPrivacyInference ?? true
        })
      } else {
        const newSession = createEmptySession()
        set({
          sessions: [newSession], currentSessionId: newSession.id,
          infonSessions: {}, privacyInferences: {},
          customPrivacyItems: [], selectedLawIdx: 0, selectedPrivacyItems: []
        })
      }
    } catch (error) {
      console.error('[PrivaSee] 加载用户历史失败:', error)
    }
  },
  
  _saveUserHistory(userId) {
    try {
      const state = get()
      const serializableInferences = {}
      Object.keys(state.privacyInferences).forEach(sessionId => {
        const inference = state.privacyInferences[sessionId]
        if (inference) {
          const { abortController, ...rest } = inference
          serializableInferences[sessionId] = rest
        }
      })
      
      // 构建主记忆流元数据 (轻量, 不含向量)
      const memoryStreamMeta = state.memoryStreamLastIngest ? {
        lastIngest: {
          ingested_count: state.memoryStreamLastIngest.ingested_count,
          total_in_store: state.memoryStreamLastIngest.total_in_store,
        },
      } : null
      
      saveUserSessions(
        userId, state.sessions, state.infonSessions, serializableInferences,
        state.customPrivacyItems, state.selectedLawIdx, state.selectedPrivacyItems,
        state.model, state.infonExtractionModel,
        state.infonPrivacyInferenceModel, state.imageParsingModel, state.protectionSuggestionModel,
        state.autoPrivacyInference, memoryStreamMeta
      )
    } catch (error) {
      console.error('[PrivaSee] 保存用户历史失败:', error)
    }
  },
  
  saveCurrentUserHistory() {
    const { currentUserId } = get()
    if (currentUserId) get()._saveUserHistory(currentUserId)
  },
})
