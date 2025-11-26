/**
 * 消息发送辅助函数
 * 提供消息处理和流式响应的通用逻辑
 */

import { generateId } from '../utils.js'
import { streamOpenAIResponse, streamOllamaChatResponse } from '../streamUtils.js'

/**
 * 创建用户消息对象
 */
export function createUserMessage({ text, audios = [], images = [], files = [], commands = [], imageAnalysis = {} }) {
  let messageContent = text
  
  // 添加音频转写
  if (audios.length > 0) {
    const audioTranscripts = audios
      .filter(audio => audio.transcript && audio.transcript.trim())
      .map(audio => `<audio>${audio.transcript.trim()}</audio>`)
      .join('\n')
    if (audioTranscripts) {
      messageContent = [text, audioTranscripts].filter(Boolean).join('\n\n')
    }
  }
  
  return {
    id: generateId(),
    role: 'user',
    content: messageContent,
    ...(audios.length > 0 && { audios }),
    ...(images.length > 0 && { images }),
    ...(files.length > 0 && { files }),
    ...(commands.length > 0 && { commands }),
    ...(Object.keys(imageAnalysis).length > 0 && { imageAnalysis }),
    createdAt: Date.now(),
  }
}

/**
 * 创建助手消息对象（用于流式写入）
 */
export function createAssistantMessage() {
  return {
    id: generateId(),
    role: 'assistant',
    content: '',
    reasoning: '',
    phase: 'thinking',
    streaming: true,
    createdAt: Date.now(),
  }
}

/**
 * 处理 <think> 标签的流式内容解析
 */
export function createThinkTagHandler(get, sessionId, assistantMsgId) {
  let inThink = false
  
  return ({ content, reasoning, finish }) => {
    if (reasoning) {
      get()._updateMessage(sessionId, assistantMsgId, (m) => ({
        ...m,
        reasoning: (m.reasoning || '') + reasoning
      }))
    }

    if (typeof content === 'string' && content.length) {
      let rest = content
      while (rest && rest.length) {
        if (inThink) {
          const endIdx = rest.indexOf('</think>')
          if (endIdx >= 0) {
            const head = rest.slice(0, endIdx)
            const tail = rest.slice(endIdx + 8)
            if (head) {
              get()._updateMessage(sessionId, assistantMsgId, (m) => ({
                ...m,
                reasoning: (m.reasoning || '') + head
              }))
            }
            inThink = false
            rest = tail
            continue
          } else {
            get()._updateMessage(sessionId, assistantMsgId, (m) => ({
              ...m,
              reasoning: (m.reasoning || '') + rest
            }))
            rest = ''
            break
          }
        } else {
          const startIdx = rest.indexOf('<think>')
          if (startIdx >= 0) {
            const before = rest.slice(0, startIdx)
            const tail = rest.slice(startIdx + 7)
            if (before) {
              get()._updateMessage(sessionId, assistantMsgId, (m) => ({
                ...m,
                content: (m.content || '') + before,
                phase: 'answering'
              }))
            }
            inThink = true
            rest = tail
            continue
          } else {
            get()._updateMessage(sessionId, assistantMsgId, (m) => ({
              ...m,
              content: (m.content || '') + rest,
              phase: 'answering'
            }))
            rest = ''
            break
          }
        }
      }
    }

    if (finish) {
      get()._updateMessage(sessionId, assistantMsgId, (m) => ({
        ...m,
        streaming: false,
        phase: 'done'
      }))
    }
  }
}

/**
 * 获取模型配置（max_tokens 等）
 */
export function getModelConfig(modelName) {
  const lowerName = modelName.toLowerCase()
  const isOmni = lowerName.includes('omni')
  const isVL = lowerName.includes('vl') && !isOmni
  
  return {
    isOmni,
    isVL,
    maxTokens: isOmni ? 2000 : 4096,
    // VL 模型处理图片时不能使用流式
    canStreamWithImages: isOmni || !isVL,
  }
}

/**
 * 执行流式聊天请求
 */
export async function executeStreamingChat({
  get, set, sessionId, assistantMsgId, controller,
  apiUrl, headers, requestBody, useOllamaStream = false
}) {
  try {
    const res = await fetch(apiUrl, {
      method: 'POST',
      headers: { ...headers, 'Connection': 'keep-alive' },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
      keepalive: true
    })

    if (!res.ok) {
      const textErr = await res.text().catch(() => '')
      get()._updateMessage(sessionId, assistantMsgId, (m) => ({
        ...m,
        streaming: false,
        error: textErr || 'Request failed',
        content: m.content
      }))
      set({ isGenerating: false, abortController: null })
      return false
    }

    const reader = res.body?.getReader()
    if (!reader) {
      get()._updateMessage(sessionId, assistantMsgId, (m) => ({
        ...m,
        streaming: false,
        error: 'No stream',
        content: m.content
      }))
      set({ isGenerating: false, abortController: null })
      return false
    }

    const streamHandler = useOllamaStream ? streamOllamaChatResponse : streamOpenAIResponse
    const onDelta = createThinkTagHandler(get, sessionId, assistantMsgId)
    await streamHandler(reader, onDelta)
    return true
  } catch (err) {
    const msg = (err && err.name === 'AbortError') ? 'Aborted' : `Network error: ${err?.message || 'Unknown'}`
    get()._updateMessage(sessionId, assistantMsgId, (m) => ({
      ...m,
      streaming: false,
      error: msg
    }))
    return false
  } finally {
    set({ isGenerating: false, abortController: null })
  }
}

/**
 * 去除 data URL 前缀（用于 Ollama 格式）
 */
export function stripDataUrl(s) {
  if (typeof s !== 'string') return s
  const i = s.indexOf(',')
  if (i >= 0 && s.slice(0, i).includes('base64')) return s.slice(i + 1)
  return s
}

/**
 * 构建 Ollama 格式的历史消息（处理图片）
 * @param {Array} messages - 会话消息列表
 * @param {string} excludeAssistantId - 要排除的助手消息ID
 * @returns {Array} Ollama 格式的历史消息
 */
export function buildOllamaHistory(messages, excludeAssistantId) {
  // 过滤掉空的 assistant 消息
  const filteredMsgs = messages.filter(m => {
    if (m.id === excludeAssistantId) return false
    if (m.role === 'assistant' && (!m.content || m.content.trim() === '')) return false
    return true
  })
  
  // 找到最后一个包含图片的用户消息索引
  let lastImageIdx = -1
  for (let i = filteredMsgs.length - 1; i >= 0; i--) {
    const m = filteredMsgs[i]
    if (Array.isArray(m.images) && m.images.length > 0 && m.role === 'user') {
      lastImageIdx = i
      break
    }
  }
  
  // 构建历史消息（只保留最后一个图片消息的图片）
  return filteredMsgs.map((m, idx) => {
    const o = { role: m.role, content: m.content }
    if (idx === lastImageIdx && Array.isArray(m.images) && m.images.length) {
      o.images = m.images.map(stripDataUrl)
    }
    return o
  })
}

/**
 * 创建 Ollama 流式响应处理器
 * @param {Function} get - zustand get 函数
 * @param {string} sessionId - 会话ID
 * @param {string} assistantMsgId - 助手消息ID
 * @returns {Function} 响应处理回调
 */
export function createOllamaStreamHandler(get, sessionId, assistantMsgId) {
  return ({ content, finish }) => {
    if (typeof content === 'string' && content.length) {
      get()._updateMessage(sessionId, assistantMsgId, (m) => ({
        ...m,
        content: (m.content || '') + content,
        phase: 'answering'
      }))
    }
    if (finish) {
      get()._updateMessage(sessionId, assistantMsgId, (m) => ({
        ...m,
        streaming: false,
        phase: 'done'
      }))
    }
  }
}
