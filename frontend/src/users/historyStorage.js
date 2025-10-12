// 历史数据持久化工具
// 为每个登录用户单独保存会话历史和信息元数据

// 保存用户的所有会话数据
export function saveUserSessions(userId, sessions, infonSessions, privacyInferences, customPrivacyItems, selectedLawIdx, selectedPrivacyItems) {
  if (!userId) return
  
  try {
    const key = `privasee_history_${userId}`
    const data = {
      sessions: sessions || [],
      infonSessions: infonSessions || {},
      privacyInferences: privacyInferences || {},
      customPrivacyItems: customPrivacyItems || [],
      selectedLawIdx: selectedLawIdx ?? 0,
      selectedPrivacyItems: selectedPrivacyItems || [],
      savedAt: Date.now()
    }
    
    localStorage.setItem(key, JSON.stringify(data))
    console.log(`[PrivaSee] 已保存用户 ${userId} 的会话数据`)
  } catch (error) {
    console.error('[PrivaSee] 保存会话数据失败:', error)
  }
}

// 加载用户的所有会话数据
export function loadUserSessions(userId) {
  if (!userId) return null
  
  try {
    const key = `privasee_history_${userId}`
    const data = localStorage.getItem(key)
    
    if (!data) return null
    
    const parsed = JSON.parse(data)
    console.log(`[PrivaSee] 已加载用户 ${userId} 的会话数据`)
    
    return {
      sessions: parsed.sessions || [],
      infonSessions: parsed.infonSessions || {},
      privacyInferences: parsed.privacyInferences || {},
      customPrivacyItems: parsed.customPrivacyItems || [],
      selectedLawIdx: parsed.selectedLawIdx ?? 0,
      selectedPrivacyItems: parsed.selectedPrivacyItems || [],
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
          data.selectedPrivacyItems
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

