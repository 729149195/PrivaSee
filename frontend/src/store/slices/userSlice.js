// 用户与历史数据 Slice
import { loadUserSessions, saveUserSessions } from '../../users/historyStorage'
import { getDefaultModelsConfig } from '../../config/defaultModelsConfig'
import { createEmptySession } from '../utils'

export const createUserSlice = (set, get) => ({
  currentUserId: null,

  setCurrentUser: (userId) => {
    set({ currentUserId: userId })
    if (userId) get()._loadUserHistory(userId)
  },
  
  clearCurrentUser: () => {
    const { currentUserId } = get()
    if (currentUserId) get()._saveUserHistory(currentUserId)
    const emptySession = createEmptySession()
    set({
      currentUserId: null, sessions: [emptySession], currentSessionId: emptySession.id,
      model: getDefaultModelsConfig().conversationModel, infonSessions: {},
      privacyInferences: {}, sessionKeywords: {}
    })
  },
  
  clearAllData: () => {
    const emptySession = createEmptySession()
    set({
      sessions: [emptySession], currentSessionId: emptySession.id,
      model: getDefaultModelsConfig().conversationModel, infonSessions: {},
      privacyInferences: {}, sessionKeywords: {}, protectionSuggestions: {},
      customPrivacyItems: [], selectedPrivacyItems: []
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
          sessionKeywords: data.sessionKeywords || {},
          currentSessionId: data.sessions[0]?.id || null,
          customPrivacyItems: data.customPrivacyItems || [],
          selectedLawIdx: data.selectedLawIdx ?? 0,
          selectedPrivacyItems: data.selectedPrivacyItems || [],
          model: data.conversationModel || getDefaultModelsConfig().conversationModel,
          directInferenceModel: data.directInferenceModel || getDefaultModelsConfig().directInferenceModel,
          infonExtractionModel: data.infonExtractionModel || getDefaultModelsConfig().infonExtractionModel,
          infonPrivacyInferenceModel: data.infonPrivacyInferenceModel || getDefaultModelsConfig().infonPrivacyInferenceModel,
          imageParsingModel: data.imageParsingModel || getDefaultModelsConfig().imageParsingModel,
          protectionSuggestionModel: data.protectionSuggestionModel || getDefaultModelsConfig().protectionSuggestionModel,
          inferenceMode: data.inferenceMode || getDefaultModelsConfig().inferenceMode,
          autoPrivacyInference: data.autoPrivacyInference ?? true
        })
      } else {
        const newSession = createEmptySession()
        set({
          sessions: [newSession], currentSessionId: newSession.id,
          infonSessions: {}, privacyInferences: {}, sessionKeywords: {},
          customPrivacyItems: [], selectedLawIdx: 0, selectedPrivacyItems: [],
          inferenceMode: 'extract'
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
      
      saveUserSessions(
        userId, state.sessions, state.infonSessions, serializableInferences,
        state.customPrivacyItems, state.selectedLawIdx, state.selectedPrivacyItems,
        state.model, state.directInferenceModel, state.infonExtractionModel,
        state.infonPrivacyInferenceModel, state.imageParsingModel, state.protectionSuggestionModel,
        state.inferenceMode, state.sessionKeywords, state.autoPrivacyInference
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
