// 历史数据持久化工具
// 为每个登录用户单独保存会话历史和信息元数据

// 保存用户的所有会话数据
export function saveUserSessions(userId, sessions, infonSessions, privacyInferences, customPrivacyItems, selectedLawIdx, selectedPrivacyItems, conversationModel, directInferenceModel, infonExtractionModel, infonPrivacyInferenceModel, imageParsingModel, protectionSuggestionModel, inferenceMode, sessionKeywords) {
  if (!userId) return
  
  try {
    const key = `privasee_history_${userId}`
    
    // 将sessionKeywords中的Set转换为数组以便JSON序列化
    const serializedKeywords = {}
    if (sessionKeywords && typeof sessionKeywords === 'object') {
      Object.entries(sessionKeywords).forEach(([sessionId, keywordSet]) => {
        if (keywordSet instanceof Set) {
          serializedKeywords[sessionId] = Array.from(keywordSet)
        } else if (Array.isArray(keywordSet)) {
          serializedKeywords[sessionId] = keywordSet
        }
      })
    }
    
    const data = {
      sessions: sessions || [],
      infonSessions: infonSessions || {},
      privacyInferences: privacyInferences || {},
      customPrivacyItems: customPrivacyItems || [],
      selectedLawIdx: selectedLawIdx ?? 0,
      selectedPrivacyItems: selectedPrivacyItems || [],
      // 模型配置
      conversationModel: conversationModel || 'deepseek-chat',
      directInferenceModel: directInferenceModel || 'deepseek-chat',
      infonExtractionModel: infonExtractionModel || 'deepseek-chat',
      infonPrivacyInferenceModel: infonPrivacyInferenceModel || 'deepseek-chat',
      imageParsingModel: imageParsingModel || 'gemma3:12b',
      protectionSuggestionModel: protectionSuggestionModel || 'deepseek-chat',
      inferenceMode: inferenceMode || 'extract', // 保存推断模式
      sessionKeywords: serializedKeywords, // 保存关键词（数组格式）
      savedAt: Date.now()
    }
    
    localStorage.setItem(key, JSON.stringify(data))
    console.log(`[PrivaSee] 已保存用户 ${userId} 的会话数据（包含 ${Object.keys(serializedKeywords).length} 个会话的关键词）`)
  } catch (error) {
    console.error('[PrivaSee] 保存会话数据失败:', error)
  }
}

// 加载用户的所有会话数据
export function loadUserSessions(userId, defaultModelsConfig = {}) {
  if (!userId) return null
  
  try {
    const key = `privasee_history_${userId}`
    const data = localStorage.getItem(key)
    
    if (!data) return null
    
    const parsed = JSON.parse(data)
    
    // 将sessionKeywords中的数组转换回Set
    const deserializedKeywords = {}
    if (parsed.sessionKeywords && typeof parsed.sessionKeywords === 'object') {
      Object.entries(parsed.sessionKeywords).forEach(([sessionId, keywordArray]) => {
        if (Array.isArray(keywordArray)) {
          deserializedKeywords[sessionId] = new Set(keywordArray)
        }
      })
    }
    
    console.log(`[PrivaSee] 已加载用户 ${userId} 的会话数据（包含 ${Object.keys(deserializedKeywords).length} 个会话的关键词）`)
    
    return {
      sessions: parsed.sessions || [],
      infonSessions: parsed.infonSessions || {},
      privacyInferences: parsed.privacyInferences || {},
      customPrivacyItems: parsed.customPrivacyItems || [],
      selectedLawIdx: parsed.selectedLawIdx ?? 0,
      selectedPrivacyItems: parsed.selectedPrivacyItems || [],
      // 模型配置
      conversationModel: parsed.conversationModel || defaultModelsConfig.conversationModel || 'deepseek-chat',
      directInferenceModel: parsed.directInferenceModel || defaultModelsConfig.directInferenceModel || 'deepseek-chat',
      infonExtractionModel: parsed.infonExtractionModel || defaultModelsConfig.infonExtractionModel || 'deepseek-chat',
      infonPrivacyInferenceModel: parsed.infonPrivacyInferenceModel || defaultModelsConfig.infonPrivacyInferenceModel || 'deepseek-chat',
      imageParsingModel: parsed.imageParsingModel || defaultModelsConfig.imageParsingModel || 'gemma3:12b',
      protectionSuggestionModel: parsed.protectionSuggestionModel || defaultModelsConfig.protectionSuggestionModel || 'deepseek-chat',
      inferenceMode: parsed.inferenceMode || defaultModelsConfig.inferenceMode || 'extract', // 加载推断模式
      sessionKeywords: deserializedKeywords, // 加载关键词（Set格式）
      savedAt: parsed.savedAt
    }
  } catch (error) {
    console.error('[PrivaSee] 加载会话数据失败:', error)
    return null
  }
}

// 清除用户的所有会话数据
export function clearUserSessions(userId) {
  if (!userId) return
  
  try {
    const key = `privasee_history_${userId}`
    localStorage.removeItem(key)
    console.log(`[PrivaSee] 已清除用户 ${userId} 的会话数据`)
  } catch (error) {
    console.error('[PrivaSee] 清除会话数据失败:', error)
  }
}

// 获取所有用户的历史数据大小（用于显示存储使用情况）
export function getUserHistorySize(userId) {
  if (!userId) return 0
  
  try {
    const key = `privasee_history_${userId}`
    const data = localStorage.getItem(key)
    
    if (!data) return 0
    
    // 计算字符串字节大小（粗略估算）
    return new Blob([data]).size
  } catch (error) {
    return 0
  }
}

// 导出用户数据（JSON 格式）
export function exportUserData(userId) {
  const data = loadUserSessions(userId)
  if (!data) return null
  
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `privasee-history-${userId}-${Date.now()}.json`
  link.click()
  URL.revokeObjectURL(url)
  
  return data
}

// 导入用户数据（从 JSON 文件）
export function importUserData(userId, file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result)
        
        // 验证数据格式
        if (!data.sessions || !Array.isArray(data.sessions)) {
          throw new Error('无效的数据格式')
        }
        
        // 保存导入的数据
        saveUserSessions(
          userId, 
          data.sessions, 
          data.infonSessions, 
          data.privacyInferences,
          data.customPrivacyItems,
          data.selectedLawIdx,
          data.selectedPrivacyItems,
          data.directInferenceModel,
          data.infonExtractionModel,
          data.infonPrivacyInferenceModel,
          data.imageParsingModel,
          data.protectionSuggestionModel,
          data.inferenceMode,
          data.sessionKeywords || {}
        )
        resolve(data)
      } catch (error) {
        reject(error)
      }
    }
    
    reader.onerror = () => reject(new Error('读取文件失败'))
    reader.readAsText(file)
  })
}

