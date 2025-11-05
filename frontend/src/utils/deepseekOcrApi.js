import { getOcrCommandById, getOcrSystemPrompt } from './ocrCommands'

const readFileAsDataUrl = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = () => {
    const result = reader.result
    if (typeof result === 'string') {
      resolve(result)
    } else {
      reject(new Error('无法读取文件内容'))
    }
  }
  reader.onerror = () => reject(reader.error || new Error('文件读取失败'))
  reader.readAsDataURL(file)
})

const normalizeBaseUrl = (baseUrl) => {
  if (!baseUrl || typeof baseUrl !== 'string') return ''
  return baseUrl.replace(/\/$/, '')
}

const buildUserContent = (instruction, dataUrl) => ([
  {
    type: 'text',
    text: instruction
  },
  {
    type: 'image_url',
    image_url: {
      url: dataUrl
    }
  }
])

export async function callDeepseekOcr({
  file,
  commandId,
  provider,
  signal,
  onProgress,
  question,
  resolution = 'gundam', // 默认使用 gundam 分辨率模式
  uploadedFilename = null  // 已上传的文件名
}) {
  if (!file && !uploadedFilename) {
    throw new Error('未提供需要识别的文件')
  }

  if (!provider) {
    throw new Error('未找到 DeepSeek OCR 的 API 配置')
  }

  const baseUrl = normalizeBaseUrl(provider.baseUrl)
  if (!baseUrl) {
    throw new Error('DeepSeek OCR API 基础地址无效')
  }

  const command = getOcrCommandById(commandId)
  if (!command) {
    throw new Error('暂不支持所选的 OCR 功能')
  }

  const reportProgress = (value, stage) => {
    if (typeof onProgress === 'function') {
      onProgress({ value, stage })
    }
  }

  reportProgress(5, '准备文件...')

  // 使用 FormData 上传文件（适配新后端）
  const formData = new FormData()
  
  // 如果提供了已上传的文件名，使用它；否则上传新文件
  if (uploadedFilename) {
    formData.append('uploaded_filename', uploadedFilename)
  } else {
    formData.append('file', file)
  }
  
  formData.append('function', commandId)
  formData.append('resolution', resolution)
  formData.append('save_results', 'false')
  
  // 如果是视觉问答，添加自定义问题
  if (command.id === 'visual_qa' && question) {
    formData.append('question', question)
  }

  reportProgress(25, '调用 OCR 接口...')

  const headers = {}
  if (provider.apiKey) {
    headers['Authorization'] = `Bearer ${provider.apiKey}`
  }

  let response
  try {
    response = await fetch(`${baseUrl}/process`, {
      method: 'POST',
      headers,
      body: formData,
      signal
    })
  } catch (error) {
    throw new Error(`网络请求失败: ${error.message || error}`)
  }

  if (!response.ok) {
    let errMessage = `HTTP ${response.status}`
    try {
      const errJson = await response.json()
      if (errJson?.error) {
        errMessage = errJson.error
      } else if (errJson?.message) {
        errMessage = errJson.message
      }
    } catch (_) {
      try {
        const errText = await response.text()
        if (errText) errMessage = errText
      } catch (_) {}
    }
    throw new Error(`OCR 处理失败: ${errMessage}`)
  }

  reportProgress(70, '解析结果...')

  let data
  try {
    data = await response.json()
  } catch (error) {
    throw new Error(`解析响应失败: ${error.message || error}`)
  }

  if (!data.success) {
    throw new Error(data.error || 'OCR 处理失败')
  }

  const text = data.text?.trim() || ''
  
  // 如果是视觉问答且没有文本，给出更友好的提示
  if (!text && command.id === 'visual_qa') {
    console.warn('[deepseekOcrApi] 视觉问答返回空结果，可能是问题不明确或图片无法理解')
  }

  reportProgress(100, '处理完成')

  return {
    text: text || '(未识别到内容)', // 如果为空，返回提示文本
    raw: data,
    command,
    metadata: data.metadata
  }
}

/**
 * 流式调用 DeepSeek OCR (SSE)
 */
export async function callDeepseekOcrStream({
  file,
  commandId,
  provider,
  signal,
  onProgress,
  onContent,
  question,
  resolution = 'gundam',
  uploadedFilename = null,  // 已上传的文件名
  messages = null  // 历史消息列表
}) {
  if (!file && !uploadedFilename) {
    throw new Error('未提供需要识别的文件')
  }

  if (!provider) {
    throw new Error('未找到 DeepSeek OCR 的 API 配置')
  }

  const baseUrl = normalizeBaseUrl(provider.baseUrl)
  if (!baseUrl) {
    throw new Error('DeepSeek OCR API 基础地址无效')
  }

  const command = getOcrCommandById(commandId)
  if (!command) {
    throw new Error('暂不支持所选的 OCR 功能')
  }

  // 准备 FormData
  const formData = new FormData()
  
  // 如果提供了已上传的文件名，使用它；否则上传新文件
  if (uploadedFilename) {
    formData.append('uploaded_filename', uploadedFilename)
  } else {
    formData.append('file', file)
  }
  
  formData.append('function', commandId)
  formData.append('resolution', resolution)
  
  if (command.id === 'visual_qa' && question) {
    formData.append('question', question)
  }
  
  // 如果提供了历史消息，将其序列化为 JSON 并添加到 FormData
  if (messages && Array.isArray(messages) && messages.length > 0) {
    formData.append('messages', JSON.stringify(messages))
  }

  const headers = {}
  if (provider.apiKey) {
    headers['Authorization'] = `Bearer ${provider.apiKey}`
  }

  let reader = null
  
  try {
    // 发起 SSE 请求
    const response = await fetch(`${baseUrl}/process/stream`, {
      method: 'POST',
      headers,
      body: formData,
      signal
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    // 处理 SSE 流
    reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let fullText = ''
    let metadata = null

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6))
          
          switch (data.type) {
            case 'start':
            case 'progress':
              if (onProgress) {
                onProgress({
                  value: data.progress || 0,
                  stage: data.stage || data.message || ''
                })
              }
              break
              
            case 'content':
              if (data.text) {
                fullText += data.text
                if (onContent) {
                  onContent(data.text) // 逐块发送
                }
              }
              break
              
            case 'done':
              metadata = data.metadata
              break
              
            case 'error':
              throw new Error(data.error || 'OCR 处理失败')
          }
        }
      }
    }

    return {
      text: fullText,
      command,
      metadata
    }
  } catch (error) {
    // 检查是否是用户中断
    if (error.name === 'AbortError' || signal?.aborted) {
      const abortError = new Error('用户已停止处理')
      abortError.name = 'AbortError'
      throw abortError
    }
    throw error
  } finally {
    // 清理资源
    if (reader) {
      try {
        reader.releaseLock()
      } catch (e) {
        // 忽略释放锁的错误
      }
    }
  }
}

export async function callDeepseekOcrBatch({ files, commandId, provider, signal, onProgress }) {
  if (!Array.isArray(files) || files.length === 0) {
    throw new Error('未提供需要处理的文件列表')
  }

  const results = []
  for (let i = 0; i < files.length; i += 1) {
    const file = files[i]
    const progressBase = Math.floor((i / files.length) * 100)
    const result = await callDeepseekOcr({
      file,
      commandId,
      provider,
      signal,
      onProgress: (info) => {
        const value = progressBase + Math.round(info.value / files.length)
        const stage = `处理第 ${i + 1} 个文件: ${info.stage}`
        if (typeof onProgress === 'function') {
          onProgress({ value, stage })
        }
      }
    })
    results.push({ file, ...result })
  }

  return results
}

