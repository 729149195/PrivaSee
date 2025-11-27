/**
 * 消息发送 Slice
 * 统一管理消息发送、重新生成等功能
 */

import { generateId } from '../utils'
import { streamOpenAIResponse, streamOllamaChatResponse } from '../streamUtils'
import { getModelApiConfig } from './infonHelpers'
import {
  createUserMessage,
  createAssistantMessage,
  createThinkTagHandler,
  buildOllamaHistory,
  createOllamaStreamHandler,
} from './messageHelpers'
import { getModelModalities } from '../../utils/modelUtils'
import { callDeepseekOcrStream } from '../../utils/deepseekOcrApi'
import { saveFiles, loadFiles } from '../../utils/fileStorage'

/**
 * 通用的流式请求错误处理器
 */
const handleStreamError = (get, set, sessionId, assistantMsgId, error) => {
  const msg = error?.name === 'AbortError' ? 'Aborted' : `Network error: ${error?.message || '未知错误'}`
  get()._updateMessage(sessionId, assistantMsgId, m => ({
    ...m,
    streaming: false,
    error: msg
  }))
  set({ isGenerating: false, abortController: null })
}

/**
 * 通用的流式请求处理器
 */
const handleStreamResponse = async (get, set, sessionId, assistantMsgId, res, useOllama = false) => {
  if (!res.ok) {
    const textErr = await res.text().catch(() => '')
    get()._updateMessage(sessionId, assistantMsgId, m => ({
      ...m,
      streaming: false,
      error: textErr || 'Request failed'
    }))
    set({ isGenerating: false, abortController: null })
    return false
  }

  const reader = res.body?.getReader()
  if (!reader) {
    get()._updateMessage(sessionId, assistantMsgId, m => ({
      ...m,
      streaming: false,
      error: 'No stream'
    }))
    set({ isGenerating: false, abortController: null })
    return false
  }

  const onDelta = useOllama 
    ? createOllamaStreamHandler(get, sessionId, assistantMsgId)
    : createThinkTagHandler(get, sessionId, assistantMsgId)
  
  const streamFn = useOllama ? streamOllamaChatResponse : streamOpenAIResponse
  await streamFn(reader, onDelta)
  return true
}

export const createMessageSlice = (set, get) => ({
  /**
   * 发送纯文本消息
   */
  async sendMessage(text, audioDataArray = []) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 检查是否需要使用带图片的发送方法
    const hasHistoricalImages = (session.messages || []).some(m => Array.isArray(m.images) && m.images.length > 0)
    const hasHistoricalAudios = (session.messages || []).some(m => Array.isArray(m.audios) && m.audios.length > 0)
    const hasAudios = Array.isArray(audioDataArray) && audioDataArray.length > 0
    
    if (hasHistoricalImages || hasHistoricalAudios || hasAudios) {
      return await get().sendMessageWithImages(text, [], audioDataArray)
    }

    // 创建消息
    const audios = Array.isArray(audioDataArray) ? audioDataArray.filter(Boolean) : []
    const userMsg = createUserMessage({ text, audios })
    get()._appendMessage(session.id, userMsg)

    const assistantMsg = createAssistantMessage()
    get()._appendMessage(session.id, assistantMsg)

    // 构建请求
    const payloadMessages = get().getCurrentSession().messages.map(m => ({ role: m.role, content: m.content }))
    const controller = new AbortController()
    set({ isGenerating: true, abortController: controller })

    // 异步执行流式请求
    ;(async () => {
      try {
        const { baseUrl, headers, maxTokens } = getModelApiConfig(get, get().model)
        const res = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: { ...headers, 'Connection': 'keep-alive' },
          body: JSON.stringify({
            model: get().model,
            messages: payloadMessages,
            temperature: 0.7,
            stream: true,
            max_tokens: maxTokens,
            top_p: 0.9
          }),
          signal: controller.signal,
          keepalive: true
        })

        await handleStreamResponse(get, set, session.id, assistantMsg.id, res)
      } catch (err) {
        handleStreamError(get, set, session.id, assistantMsg.id, err)
      } finally {
        set({ isGenerating: false, abortController: null })
      }
    })()

    return userMsg.id
  },

  /**
   * 发送带图片的消息
   */
  async sendMessageWithImages(text, imageDataUrls, audioDataArray = [], imageAnalysisMap = {}) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 创建用户消息
    const userMsgId = generateId()
    const imgs = Array.isArray(imageDataUrls) ? imageDataUrls.filter(Boolean) : []
    const audios = Array.isArray(audioDataArray) ? audioDataArray.filter(Boolean) : []

    let messageContent = text
    if (audios.length > 0) {
      const audioTranscripts = audios
        .filter(a => a.transcript?.trim())
        .map(a => `<audio>${a.transcript.trim()}</audio>`)
        .join('\n')
      if (audioTranscripts) {
        messageContent = [text, audioTranscripts].filter(Boolean).join('\n\n')
      }
    }

    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: messageContent,
      images: imgs,
      audios,
      imageAnalysis: imageAnalysisMap,
      createdAt: Date.now()
    })

    // 检查模型是否支持图片
    const currentModel = get().model
    const customProviders = get().customProviders
    const modalities = getModelModalities(currentModel, customProviders)

    if (!modalities.image) {
      const assistantMsgId = generateId()
      get()._appendMessage(session.id, {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        phase: 'done',
        streaming: false,
        error: 'Image not supported',
        createdAt: Date.now()
      })
      return userMsgId
    }

    // 创建助手消息
    const assistantMsgId = generateId()
    get()._appendMessage(session.id, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      reasoning: '',
      phase: 'thinking',
      streaming: true,
      createdAt: Date.now()
    })

    const controller = new AbortController()
    set({ isGenerating: true, abortController: controller })

    const provider = customProviders?.[currentModel]

    // 自定义 Provider
    if (provider) {
      ;(async () => {
        try {
          const sessionMsgs = get().getCurrentSession().messages
          const filteredMsgs = sessionMsgs.filter(m => 
            m.id !== assistantMsgId && !(m.role === 'assistant' && !m.content?.trim())
          )

          // 构建多模态消息格式
          const payloadMessages = filteredMsgs.map(m => {
            if (m.role === 'user' && Array.isArray(m.images) && m.images.length > 0) {
              const contentArray = []
              if (m.content?.trim()) contentArray.push({ type: 'text', text: m.content })
              m.images.forEach(img => contentArray.push({ type: 'image_url', image_url: { url: img } }))
              return { role: m.role, content: contentArray }
            }
            return { role: m.role, content: m.content }
          })

          const baseUrl = provider.baseUrl
          const headers = { 'Content-Type': 'application/json' }
          if (provider.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`

          // 确定流式选项
          const hasImages = filteredMsgs.some(m => Array.isArray(m.images) && m.images.length > 0)
          const modelName = currentModel.toLowerCase()
          const isOmni = modelName.includes('omni')
          const isVL = modelName.includes('vl') && !isOmni
          const useStreaming = !(hasImages && isVL)
          const maxTokens = isOmni ? 2000 : 4096

          const requestBody = (hasImages && isVL)
            ? { model: currentModel, messages: payloadMessages, temperature: 0.3, max_tokens: 2000 }
            : { model: currentModel, messages: payloadMessages, temperature: 0.7, stream: true, max_tokens: maxTokens, top_p: 0.9 }

          const res = await fetch(`${baseUrl}/chat/completions`, {
            method: 'POST',
            headers,
            body: JSON.stringify(requestBody),
            signal: controller.signal
          })

          if (!useStreaming && hasImages && isVL) {
            // 非流式响应
            if (res.ok) {
              const result = await res.json()
              const content = result.choices?.[0]?.message?.content || ''
              get()._updateMessage(session.id, assistantMsgId, m => ({
                ...m,
                content,
                streaming: false,
                phase: 'done'
              }))
            } else {
              const textErr = await res.text().catch(() => '')
              get()._updateMessage(session.id, assistantMsgId, m => ({
                ...m,
                streaming: false,
                error: textErr || 'Request failed'
              }))
            }
            set({ isGenerating: false, abortController: null })
            return
          }

          await handleStreamResponse(get, set, session.id, assistantMsgId, res)
        } catch (err) {
          handleStreamError(get, set, session.id, assistantMsgId, err)
        } finally {
          set({ isGenerating: false, abortController: null })
        }
      })()
      return userMsgId
    }

    // 本地 Ollama 模型
    const sessionMsgs = get().getCurrentSession().messages
    const history = buildOllamaHistory(sessionMsgs, assistantMsgId)
    const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')

    ;(async () => {
      try {
        const res = await fetch(`${apiBase}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: currentModel,
            messages: history,
            stream: true,
            options: { temperature: 0.2 }
          }),
          signal: controller.signal,
        })

        await handleStreamResponse(get, set, session.id, assistantMsgId, res, true)
      } catch (err) {
        handleStreamError(get, set, session.id, assistantMsgId, err)
      } finally {
        set({ isGenerating: false, abortController: null })
      }
    })()

    return userMsgId
  },

  /**
   * 发送 DeepSeek OCR 消息
   */
  async sendMessageWithDeepSeekOCR(text, selectedCommands, selectedFiles, resolution = 'gundam') {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    const userMsgId = generateId()
    const fileMetadata = selectedFiles.map(f => ({
      id: f.id,
      name: f.name,
      size: f.size,
      type: f.type,
      serverFilename: f.serverFilename || null
    }))
    
    // 保存文件对象引用
    const fileObjectsMap = {}
    selectedFiles.forEach(f => { if (f.file) fileObjectsMap[f.id] = f.file })
    set(s => ({
      ocrFileObjects: {
        ...s.ocrFileObjects,
        [session.id]: { ...s.ocrFileObjects?.[session.id], [userMsgId]: fileObjectsMap }
      }
    }))

    // 异步保存文件
    ;(async () => {
      try {
        const filesToSave = selectedFiles.filter(f => f.file).map(f => ({ id: f.id, file: f.file }))
        if (filesToSave.length > 0) await saveFiles(session.id, userMsgId, filesToSave)
      } catch (e) { console.error('[OCR] 保存文件失败:', e) }
    })()

    // 添加用户消息
    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: text || '',
      files: fileMetadata,
      commands: selectedCommands,
      createdAt: Date.now()
    })

    const provider = get().customProviders?.[get().model]
    if (!provider) throw new Error(`模型 ${get().model} 配置不存在`)

    // 添加助手消息
    const assistantMsgId = generateId()
    get()._appendMessage(session.id, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      streaming: true,
      phase: 'thinking',
      createdAt: Date.now()
    })
    get()._updateMessage(session.id, userMsgId, m => ({
      ...m,
      ocrStatus: 'processing',
      ocrProgress: 0
    }))

    const controller = new AbortController()
    set({ isGenerating: true, abortController: controller })
    let currentContent = ''

    try {
      // 构建历史消息
      const maxHistoryMessages = 6
      const historyMessages = []
      const previousMessages = (session.messages || []).slice(0, -1).slice(-maxHistoryMessages)
      for (const msg of previousMessages) {
        if ((msg.role === 'user' || msg.role === 'assistant') && msg.content?.trim()) {
          historyMessages.push({ role: msg.role, content: msg.content.slice(0, 1000).trim() })
        }
      }

      let filesToProcess = selectedFiles
      let commandsToUse = selectedCommands

      // 如果没有文件，尝试使用历史消息中的文件
      if (selectedFiles.length === 0 && text?.trim()) {
        const previousMsgs = session.messages.slice(0, -1)
        let recentFileMessage = null
        for (let i = previousMsgs.length - 1; i >= 0; i--) {
          if (previousMsgs[i].role === 'user' && previousMsgs[i].files?.length > 0) {
            recentFileMessage = previousMsgs[i]
            break
          }
        }

        if (recentFileMessage) {
          const messageFileObjects = get().ocrFileObjects?.[session.id]?.[recentFileMessage.id] || {}
          if (Object.keys(messageFileObjects).length === 0) {
            try {
              const fileIds = recentFileMessage.files.map(f => f.id)
              const restoredFiles = await loadFiles(session.id, recentFileMessage.id, fileIds)
              filesToProcess = recentFileMessage.files.map(f => ({
                ...f,
                file: restoredFiles[f.id],
                uploadStatus: 'completed'
              }))
            } catch (e) { console.error('[OCR] 恢复文件失败:', e) }
          } else {
            filesToProcess = recentFileMessage.files.map(f => ({
              ...f,
              file: messageFileObjects[f.id],
              uploadStatus: 'completed'
            }))
          }
          commandsToUse = recentFileMessage.commands?.length > 0
            ? recentFileMessage.commands
            : [{ id: 'visual_qa', label: '视觉问答', icon: '💬' }]
        } else {
          get()._updateMessage(session.id, assistantMsgId, m => ({
            ...m,
            content: '💡 DeepSeek-OCR 模式专门用于文档/图片处理。请上传文件后再提问。',
            streaming: false,
            phase: 'done'
          }))
          set({ isGenerating: false, abortController: null })
          return userMsgId
        }
      }

      // 上传文件
      for (let i = 0; i < filesToProcess.length; i++) {
        const fileData = filesToProcess[i]
        if (fileData.serverFilename) continue
        if (fileData.file) {
          const { uploadFile } = await import('../../utils/fileUpload')
          const result = await uploadFile(fileData.file, provider, p => {
            get()._updateMessage(session.id, userMsgId, m => ({
              ...m,
              ocrStatus: 'uploading',
              ocrProgress: Math.round(p * 0.2),
              ocrStage: `上传 ${i + 1}/${filesToProcess.length}`
            }))
          })
          if (result.success) {
            fileData.serverFilename = result.filename
            get()._updateMessage(session.id, userMsgId, m => ({
              ...m,
              files: m.files.map(f => f.id === fileData.id ? { ...f, serverFilename: result.filename } : f)
            }))
          } else throw new Error('上传失败')
        }
      }

      // 处理文件
      for (let i = 0; i < filesToProcess.length; i++) {
        const fileData = filesToProcess[i]
        const command = commandsToUse[i] || commandsToUse[0]

        if (filesToProcess.length > 1) {
          if (i > 0) currentContent += '\n\n'
          currentContent += `## 📄 ${fileData.name}\n\n`
          get()._updateMessage(session.id, assistantMsgId, m => ({ ...m, content: currentContent }))
        }

        try {
          await callDeepseekOcrStream({
            file: fileData.serverFilename ? null : fileData.file,
            uploadedFilename: fileData.serverFilename,
            commandId: command.id,
            provider,
            resolution,
            question: text || undefined,
            messages: historyMessages.length > 0 ? historyMessages : null,
            signal: controller.signal,
            onProgress: ({ value, stage }) => {
              const progress = Math.round(((i + value / 100) / filesToProcess.length) * 100)
              get()._updateMessage(session.id, userMsgId, m => ({
                ...m,
                ocrStatus: 'processing',
                ocrProgress: progress,
                ocrStage: stage
              }))
            },
            onContent: chunk => {
              currentContent += chunk
              get()._updateMessage(session.id, assistantMsgId, m => ({
                ...m,
                content: currentContent,
                streaming: true
              }))
            }
          })
        } catch (e) {
          if (e.name === 'AbortError') throw e
          currentContent += `处理出错：${e.message}`
          get()._updateMessage(session.id, assistantMsgId, m => ({ ...m, content: currentContent }))
        }
      }

      // 完成处理
      get()._updateMessage(session.id, assistantMsgId, m => ({
        ...m,
        streaming: false,
        phase: 'done'
      }))
      get()._updateMessage(session.id, userMsgId, m => ({
        ...m,
        ocrStatus: 'completed',
        ocrProgress: 100
      }))
      set({ isGenerating: false, abortController: null })
      return userMsgId

    } catch (error) {
      const isAborted = error.name === 'AbortError' || controller.signal.aborted
      get()._updateMessage(session.id, assistantMsgId, m => ({
        ...m,
        content: isAborted ? (currentContent || '已停止') : `处理失败：${error.message}`,
        streaming: false,
        phase: 'done',
        error: isAborted ? undefined : error.message
      }))
      get()._updateMessage(session.id, userMsgId, m => ({
        ...m,
        ocrStatus: isAborted ? 'aborted' : 'error',
        ocrError: isAborted ? undefined : error.message
      }))
      set({ isGenerating: false, abortController: null })
    }
  },

  /**
   * 重新生成最后一条消息
   */
  async regenerateLast() {
    const session = get().getCurrentSession()
    if (!session) return

    const lastUserIndex = [...session.messages].reverse().findIndex(m => m.role === 'user')
    if (lastUserIndex === -1) return
    const userIdx = session.messages.length - 1 - lastUserIndex
    const lastUser = session.messages[userIdx]

    // 移除上一次的助手回复
    set(s => ({
      sessions: s.sessions.map(x => {
        if (x.id !== session.id) return x
        const msgs = [...x.messages]
        if (msgs.length > userIdx + 1 && msgs[userIdx + 1].role === 'assistant') {
          msgs.splice(userIdx + 1, 1)
        }
        return { ...x, messages: msgs, updatedAt: Date.now() }
      })
    }))

    // OCR 消息重新生成
    if (lastUser.files?.length > 0 && lastUser.commands?.length > 0) {
      const messageFileObjects = get().ocrFileObjects?.[session.id]?.[lastUser.id] || {}
      let restoredFiles = messageFileObjects
      if (Object.keys(restoredFiles).length === 0) {
        try {
          restoredFiles = await loadFiles(session.id, lastUser.id, lastUser.files.map(f => f.id))
        } catch (e) { console.error('[regenerate] 恢复文件失败:', e) }
      }

      const provider = get().customProviders?.[get().model]
      if (!provider) throw new Error('模型配置不存在')

      const filesToProcess = await Promise.all(lastUser.files.map(async f => {
        if (f.serverFilename) return { ...f, file: restoredFiles[f.id], uploadStatus: 'success' }
        const fileObj = restoredFiles[f.id]
        if (fileObj) {
          const { uploadFile } = await import('../../utils/fileUpload')
          const result = await uploadFile(fileObj, provider, () => {})
          return result.success
            ? { ...f, file: fileObj, uploadStatus: 'success', serverFilename: result.filename }
            : { ...f, uploadStatus: 'error', uploadError: '上传失败' }
        }
        return { ...f, uploadStatus: 'error', uploadError: '文件丢失' }
      }))

      const failed = filesToProcess.filter(f => f.uploadStatus === 'error')
      if (failed.length) throw new Error(`文件上传失败: ${failed.map(f => f.name).join(', ')}`)

      await get().sendMessageWithDeepSeekOCR(
        lastUser.content,
        lastUser.commands,
        filesToProcess,
        get().currentResolution || 'gundam'
      )
    } else {
      await get().sendMessage(lastUser.content)
    }
  },
})
