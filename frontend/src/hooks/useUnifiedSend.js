import { useCallback } from 'react'
import { useStore } from '../store'

/**
 * 统一消息发送 Hook
 * 消除 AgentPage 中 handleSend 和 handleLandingSend 的重复代码
 * 
 * @param {object} params - 配置参数
 */
export function useUnifiedSend({
  currentSession,
  model,
  inferenceMode,
  infonSessions,
  sendMessage,
  startMessageInfons,
  generateSessionTitle,
  getCurrentSession,
  lastInferenceRunCountRef,
  isAdoptingPendingRef,
}) {
  /**
   * 判断是否是 OCR 模式
   */
  const isOcrMode = useCallback(() => {
    return model === 'deepseek-ocr' || model === 'deepseek-ocr-local'
  }, [model])

  /**
   * 提取图片 analysis 数据（直接推理模式）
   */
  const extractImageAnalysis = useCallback((images) => {
    const imageAnalysisMap = {}
    if (inferenceMode === 'direct' && images.length > 0) {
      images.forEach(img => {
        const imgObj = typeof img === 'string' ? { url: img } : img
        if (imgObj.url && imgObj.analysis) {
          imageAnalysisMap[imgObj.url] = imgObj.analysis
        }
      })
    }
    return imageAnalysisMap
  }, [inferenceMode])

  /**
   * 获取 pending run IDs（用于签名计算）
   */
  const getPendingRunIds = useCallback(() => {
    const currentRuns = infonSessions?.[currentSession?.id]?.runs || []
    return currentRuns
      .filter(run => run.targetType === 'pending' && run.status === 'done')
      .map(r => r.id)
      .sort()
  }, [currentSession?.id, infonSessions])

  /**
   * 采纳 pending infons 到消息
   */
  const adoptPendingInfons = useCallback(async (userId, pendingRunIds, logPrefix = '[Send]') => {
    try {
      if (pendingRunIds.length > 0) {
        const messageSignature = pendingRunIds.join('|')
        lastInferenceRunCountRef.current = messageSignature
        console.log(`${logPrefix} 提前更新签名，避免重复推理`, { signature: messageSignature })
      }

      const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
      if (result.adopted === 0 && inferenceMode === 'extract') {
        // 没有 pending infons，需要重新提取（仅在提取信息元模式下）
        startMessageInfons?.(userId)
      }
    } catch (_) {}
  }, [inferenceMode, startMessageInfons, lastInferenceRunCountRef])

  /**
   * 自动生成会话标题（仅在第一条消息后）
   */
  const autoGenerateTitle = useCallback(() => {
    setTimeout(() => {
      const session = getCurrentSession()
      if (session) {
        const userMessages = session.messages.filter(msg => msg.role === 'user')
        console.log('[AgentPage] 检查是否需要生成标题，用户消息数:', userMessages.length)
        if (userMessages.length === 1) {
          console.log('[AgentPage] 触发标题生成')
          generateSessionTitle?.(session.id)
        }
      }
    }, 1000)
  }, [getCurrentSession, generateSessionTitle])

  /**
   * 统一发送消息
   * @param {object} params - 发送参数
   * @param {string} params.text - 文本内容
   * @param {Array} params.images - 图片列表
   * @param {Array} params.audios - 音频列表
   * @param {Array} params.files - 文件列表（OCR 模式）
   * @param {object|null} params.command - 命令（OCR 模式）
   * @param {string} params.resolution - 分辨率（OCR 模式）
   * @param {object} params.callbacks - 清理回调
   * @param {boolean} params.sendLocked - 是否被锁定
   */
  const unifiedSend = useCallback(async ({
    text = '',
    images = [],
    audios = [],
    files = [],
    command = null,
    resolution = 'gundam',
    callbacks = {},
    sendLocked = false,
  }) => {
    // 检查锁定状态
    if (sendLocked) return null

    const trimmedText = text.trim()
    const imgs = images.map(img => typeof img === 'string' ? img : img.url)
    const audioList = [...audios]
    const hasImages = imgs.length > 0
    const hasAudios = audioList.length > 0
    const hasFiles = files.length > 0
    const hasCommand = command != null

    // 检查是否有内容
    if (!trimmedText && !hasImages && !hasAudios && !hasFiles && !hasCommand) {
      return null
    }

    // 获取 pending run IDs
    const pendingRunIds = getPendingRunIds()

    // 设置采纳标志
    isAdoptingPendingRef.current = true

    // 提取图片分析数据
    const imageAnalysisMap = extractImageAnalysis(images)

    let userId = null

    // OCR 模式
    if (isOcrMode()) {
      const commandsToSend = command ? [command] : []
      
      // 先清空输入
      callbacks.clearInput?.()
      callbacks.clearCommand?.()
      callbacks.clearFiles?.()
      callbacks.resetResolution?.()
      useStore.getState().setPendingImages([])

      // 发送消息
      userId = await useStore.getState().sendMessageWithDeepSeekOCR(
        trimmedText, 
        commandsToSend, 
        files, 
        resolution
      )
      await adoptPendingInfons(userId, pendingRunIds, '[OCRSend]')
    }
    // 带图片或音频
    else if (hasImages || hasAudios) {
      userId = await useStore.getState().sendMessageWithImages(
        trimmedText, 
        imgs, 
        audioList, 
        imageAnalysisMap
      )
      await adoptPendingInfons(userId, pendingRunIds)

      // 清空输入
      callbacks.clearInput?.()
      callbacks.clearImages?.()
      callbacks.clearAudios?.()
      useStore.getState().setPendingImages([])
    }
    // 纯文本
    else {
      userId = await sendMessage(trimmedText, audioList)
      await adoptPendingInfons(userId, pendingRunIds)

      // 清空输入
      callbacks.clearInput?.()
      callbacks.clearImages?.()
      callbacks.clearAudios?.()
      useStore.getState().setPendingImages([])
    }

    // 自动生成标题
    autoGenerateTitle()

    return userId
  }, [
    getPendingRunIds,
    isAdoptingPendingRef,
    extractImageAnalysis,
    isOcrMode,
    adoptPendingInfons,
    sendMessage,
    autoGenerateTitle,
  ])

  return {
    unifiedSend,
    isOcrMode,
    extractImageAnalysis,
    getPendingRunIds,
    adoptPendingInfons,
    autoGenerateTitle,
  }
}
