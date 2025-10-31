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
  question
}) {
  if (!file) {
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

  reportProgress(5, '读取文件...')
  const dataUrl = await readFileAsDataUrl(file)

  reportProgress(25, '调用 OCR 接口...')

  const headers = { 'Content-Type': 'application/json' }
  if (provider.apiKey) {
    headers['Authorization'] = `Bearer ${provider.apiKey}`
  }

  const instruction = command.id === 'visual_qa' && question
    ? `${command.instruction}\n问题: ${question}`
    : command.instruction

  const payload = {
    model: provider.modelId || 'deepseek-ai/DeepSeek-OCR',
    messages: [
      {
        role: 'system',
        content: getOcrSystemPrompt()
      },
      {
        role: 'user',
        content: buildUserContent(instruction, dataUrl)
      }
    ],
    temperature: 0.0,
    max_tokens: 2048
  }

  let response
  try {
    response = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal
    })
  } catch (error) {
    throw new Error(`网络请求失败: ${error.message || error}`)
  }

  if (!response.ok) {
    let errMessage = `HTTP ${response.status}`
    try {
      const errJson = await response.json()
      if (errJson?.error?.message) {
        errMessage = errJson.error.message
      } else if (errJson?.error) {
        errMessage = typeof errJson.error === 'string' ? errJson.error : JSON.stringify(errJson.error)
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

  const text = data?.choices?.[0]?.message?.content?.trim()
  if (!text) {
    throw new Error('OCR 接口未返回文本内容')
  }

  reportProgress(100, '处理完成')

  return {
    text,
    raw: data,
    command
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

