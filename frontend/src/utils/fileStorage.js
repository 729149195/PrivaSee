/**
 * IndexedDB 文件存储工具
 * 用于持久化存储 OCR 文件，支持刷新后恢复
 */

const DB_NAME = 'PrivaSeeFileStorage'
const DB_VERSION = 1
const STORE_NAME = 'ocrFiles'

/**
 * 打开 IndexedDB 数据库
 */
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    
    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result
      
      // 创建对象存储（如果不存在）
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        // 使用复合键：sessionId + messageId + fileId
        const objectStore = db.createObjectStore(STORE_NAME, { keyPath: 'key' })
        objectStore.createIndex('sessionId', 'sessionId', { unique: false })
        objectStore.createIndex('messageId', 'messageId', { unique: false })
      }
    }
  })
}

/**
 * 生成存储键
 */
function makeKey(sessionId, messageId, fileId) {
  return `${sessionId}:${messageId}:${fileId}`
}

/**
 * 存储文件到 IndexedDB
 * @param {string} sessionId - 会话ID
 * @param {string} messageId - 消息ID
 * @param {string} fileId - 文件ID
 * @param {File} file - File 对象
 */
export async function saveFile(sessionId, messageId, fileId, file) {
  try {
    // 先获取文件内容（在事务外完成异步操作）
    const arrayBuffer = await file.arrayBuffer()
    
    // 再开启事务进行同步操作
    const db = await openDB()
    const transaction = db.transaction([STORE_NAME], 'readwrite')
    const objectStore = transaction.objectStore(STORE_NAME)
    
    // 将 File 对象转换为可存储的格式
    const fileData = {
      key: makeKey(sessionId, messageId, fileId),
      sessionId,
      messageId,
      fileId,
      name: file.name,
      size: file.size,
      type: file.type,
      lastModified: file.lastModified,
      // 存储文件内容为 ArrayBuffer
      arrayBuffer: arrayBuffer,
      timestamp: Date.now()
    }
    
    await new Promise((resolve, reject) => {
      const request = objectStore.put(fileData)
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
    
    console.log('[FileStorage] 文件已保存:', { sessionId, messageId, fileId, name: file.name })
    db.close()
  } catch (error) {
    console.error('[FileStorage] 保存文件失败:', error)
    throw error
  }
}

/**
 * 从 IndexedDB 恢复文件
 * @param {string} sessionId - 会话ID
 * @param {string} messageId - 消息ID
 * @param {string} fileId - 文件ID
 * @returns {File|null} - 恢复的 File 对象
 */
export async function loadFile(sessionId, messageId, fileId) {
  try {
    const db = await openDB()
    const transaction = db.transaction([STORE_NAME], 'readonly')
    const objectStore = transaction.objectStore(STORE_NAME)
    
    const fileData = await new Promise((resolve, reject) => {
      const request = objectStore.get(makeKey(sessionId, messageId, fileId))
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    
    db.close()
    
    if (!fileData) {
      console.warn('[FileStorage] 文件不存在:', { sessionId, messageId, fileId })
      return null
    }
    
    // 从 ArrayBuffer 重建 File 对象
    const file = new File(
      [fileData.arrayBuffer],
      fileData.name,
      {
        type: fileData.type,
        lastModified: fileData.lastModified
      }
    )
    
    console.log('[FileStorage] 文件已恢复:', { sessionId, messageId, fileId, name: file.name })
    return file
  } catch (error) {
    console.error('[FileStorage] 恢复文件失败:', error)
    return null
  }
}

/**
 * 批量保存文件（优化版：单个事务处理所有文件）
 * @param {string} sessionId - 会话ID
 * @param {string} messageId - 消息ID
 * @param {Array} files - 文件数组 [{id, file}]
 */
export async function saveFiles(sessionId, messageId, files) {
  if (!files || files.length === 0) return
  
  try {
    // 第一步：并行获取所有文件的 ArrayBuffer（在事务外完成）
    const fileDataList = await Promise.all(
      files.map(async ({ id, file }) => ({
        key: makeKey(sessionId, messageId, id),
        sessionId,
        messageId,
        fileId: id,
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: file.lastModified,
        arrayBuffer: await file.arrayBuffer(),
        timestamp: Date.now()
      }))
    )
    
    // 第二步：使用单个事务批量保存（避免事务超时）
    const db = await openDB()
    const transaction = db.transaction([STORE_NAME], 'readwrite')
    const objectStore = transaction.objectStore(STORE_NAME)
    
    // 批量添加到对象存储
    const promises = fileDataList.map(fileData => 
      new Promise((resolve, reject) => {
        const request = objectStore.put(fileData)
        request.onsuccess = () => resolve()
        request.onerror = () => reject(request.error)
      })
    )
    
    await Promise.all(promises)
    
    console.log('[FileStorage] 批量保存完成:', { sessionId, messageId, count: files.length })
    db.close()
  } catch (error) {
    console.error('[FileStorage] 批量保存文件失败:', error)
    throw error
  }
}

/**
 * 批量恢复文件
 * @param {string} sessionId - 会话ID
 * @param {string} messageId - 消息ID
 * @param {Array} fileIds - 文件ID数组
 * @returns {Object} - 文件映射 {fileId: File}
 */
export async function loadFiles(sessionId, messageId, fileIds) {
  const fileMap = {}
  
  for (const fileId of fileIds) {
    const file = await loadFile(sessionId, messageId, fileId)
    if (file) {
      fileMap[fileId] = file
    }
  }
  
  return fileMap
}

/**
 * 删除消息的所有文件
 * @param {string} sessionId - 会话ID
 * @param {string} messageId - 消息ID
 */
export async function deleteMessageFiles(sessionId, messageId) {
  try {
    const db = await openDB()
    const transaction = db.transaction([STORE_NAME], 'readwrite')
    const objectStore = transaction.objectStore(STORE_NAME)
    const index = objectStore.index('messageId')
    
    // 查找所有属于该消息的文件
    const keys = await new Promise((resolve, reject) => {
      const request = index.getAllKeys(messageId)
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    
    // 删除所有找到的文件
    for (const key of keys) {
      await new Promise((resolve, reject) => {
        const request = objectStore.delete(key)
        request.onsuccess = () => resolve()
        request.onerror = () => reject(request.error)
      })
    }
    
    console.log('[FileStorage] 已删除消息的所有文件:', { sessionId, messageId, count: keys.length })
    db.close()
  } catch (error) {
    console.error('[FileStorage] 删除文件失败:', error)
  }
}

/**
 * 删除会话的所有文件
 * @param {string} sessionId - 会话ID
 */
export async function deleteSessionFiles(sessionId) {
  try {
    const db = await openDB()
    const transaction = db.transaction([STORE_NAME], 'readwrite')
    const objectStore = transaction.objectStore(STORE_NAME)
    const index = objectStore.index('sessionId')
    
    // 查找所有属于该会话的文件
    const keys = await new Promise((resolve, reject) => {
      const request = index.getAllKeys(sessionId)
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    
    // 删除所有找到的文件
    for (const key of keys) {
      await new Promise((resolve, reject) => {
        const request = objectStore.delete(key)
        request.onsuccess = () => resolve()
        request.onerror = () => reject(request.error)
      })
    }
    
    console.log('[FileStorage] 已删除会话的所有文件:', { sessionId, count: keys.length })
    db.close()
  } catch (error) {
    console.error('[FileStorage] 删除会话文件失败:', error)
  }
}

/**
 * 清理旧文件（超过指定天数）
 * @param {number} days - 保留天数
 */
export async function cleanOldFiles(days = 7) {
  try {
    const db = await openDB()
    const transaction = db.transaction([STORE_NAME], 'readwrite')
    const objectStore = transaction.objectStore(STORE_NAME)
    
    const cutoffTime = Date.now() - (days * 24 * 60 * 60 * 1000)
    let deletedCount = 0
    
    const cursor = await new Promise((resolve, reject) => {
      const request = objectStore.openCursor()
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    
    let current = cursor
    while (current) {
      if (current.value.timestamp < cutoffTime) {
        await new Promise((resolve, reject) => {
          const request = current.delete()
          request.onsuccess = () => resolve()
          request.onerror = () => reject(request.error)
        })
        deletedCount++
      }
      current = await new Promise((resolve, reject) => {
        const request = current.continue()
        request.onsuccess = () => resolve(request.result)
        request.onerror = () => reject(request.error)
      })
    }
    
    console.log('[FileStorage] 清理旧文件完成:', { deletedCount, days })
    db.close()
  } catch (error) {
    console.error('[FileStorage] 清理旧文件失败:', error)
  }
}

/**
 * 获取存储使用情况
 */
export async function getStorageInfo() {
  try {
    const db = await openDB()
    const transaction = db.transaction([STORE_NAME], 'readonly')
    const objectStore = transaction.objectStore(STORE_NAME)
    
    const count = await new Promise((resolve, reject) => {
      const request = objectStore.count()
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    
    let totalSize = 0
    const cursor = await new Promise((resolve, reject) => {
      const request = objectStore.openCursor()
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    
    let current = cursor
    while (current) {
      totalSize += current.value.size || 0
      current = await new Promise((resolve, reject) => {
        const request = current.continue()
        request.onsuccess = () => resolve(request.result)
        request.onerror = () => reject(request.error)
      })
    }
    
    db.close()
    
    return {
      fileCount: count,
      totalSize: totalSize,
      totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2)
    }
  } catch (error) {
    console.error('[FileStorage] 获取存储信息失败:', error)
    return { fileCount: 0, totalSize: 0, totalSizeMB: 0 }
  }
}

