import { useCallback } from 'react'
import { useStore } from '../store'

/**
 * 消息发送 Hook
 * 统一处理不同模式（普通/图片/OCR）的消息发送逻辑
 * 
 * @param {object} params - 配置参数
 * @param {object} params.currentSession - 当前会话对象
 * @param {string} params.model - 当前模型
 * @param {object} params.infonSessions - 信息元会话
 * @param {React.MutableRefObject} params.lastInferenceRunCountRef - 推理签名 ref
 * @param {React.MutableRefObject} params.isAdoptingPendingRef - 采纳 pending 标志 ref
 */
export function useSendMessage({
  currentSession,
  model,
  infonSessions,
  lastInferenceRunCountRef,
  isAdoptingPendingRef,
}) {
  /**
   * 准备发送前的公共逻辑
   * @returns {string[]} pending run IDs
   */
  const prepareSend = useCallback(() => {
    const currentRuns = infonSessions?.[currentSession?.id]?.runs || []
    const pendingRunIds = currentRuns
      .filter(run => run.targetType === 'pending' && run.status === 'done')
      .map(r => r.id)
      .sort()

    // 设置采纳标志，防止 useEffect 清空 pending
    isAdoptingPendingRef.current = true

    return pendingRunIds
  }, [currentSession?.id, infonSessions, isAdoptingPendingRef])

  /**
   * 发送后采纳 pending infons
   */
  const adoptPendingInfons = useCallback(async (userId, pendingRunIds) => {
    if (pendingRunIds.length > 0) {
      const messageSignature = pendingRunIds.join('|')
      lastInferenceRunCountRef.current = messageSignature
      console.log('[Send] 提前更新签名，避免重复推理', { signature: messageSignature })
    }

    const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
    
    if (result.adopted === 0) {
      // 没有 pending infons，需要重新提取
      useStore.getState().startMessageInfons?.(userId)
    }
  }, [lastInferenceRunCountRef])

  /**
   * 判断是否是 OCR 模式
   */
  const isOcrMode = useCallback(() => {
    return model === 'deepseek-ocr' || model === 'deepseek-ocr-local'
  }, [model])

  /**
   * 发送普通文本消息
   */
  const sendTextMessage = useCallback(async ({
    text,
    audios = [],
    onComplete,
  }) => {
    const pendingRunIds = prepareSend()
    const userId = await useStore.getState().sendMessage(text, audios)
    
    try {
      await adoptPendingInfons(userId, pendingRunIds)
    } catch (_) {}
    
    onComplete?.()
    return userId
  }, [prepareSend, adoptPendingInfons])

  /**
   * 发送带图片的消息
   */
  const sendImageMessage = useCallback(async ({
    text,
    images = [],
    audios = [],
    onComplete,
  }) => {
    const pendingRunIds = prepareSend()
    
    // 提取图片 URL
    const imgs = images.map(img => typeof img === 'string' ? img : img.url)
    
    const userId = await useStore.getState().sendMessageWithImages(text, imgs, audios, {})
    
    try {
      await adoptPendingInfons(userId, pendingRunIds)
    } catch (_) {}
    
    onComplete?.()
    return userId
  }, [prepareSend, adoptPendingInfons])

  /**
   * 发送 OCR 消息
   */
  const sendOcrMessage = useCallback(async ({
    text,
    commands = [],
    files = [],
    resolution = 'gundam',
    onComplete,
  }) => {
    const pendingRunIds = prepareSend()
    
    const userId = await useStore.getState().sendMessageWithDeepSeekOCR(text, commands, files, resolution)
    
    try {
      await adoptPendingInfons(userId, pendingRunIds)
    } catch (_) {}
    
    onComplete?.()
    return userId
  }, [prepareSend, adoptPendingInfons])

  /**
   * 统一发送接口
   */
  const send = useCallback(async ({
    text = '',
    images = [],
    audios = [],
    files = [],
    commands = [],
    resolution = 'gundam',
    onComplete,
  }) => {
    const hasImages = images.length > 0
    const hasAudios = audios.length > 0
    const hasFiles = files.length > 0
    const hasCommand = commands.length > 0

    // 检查是否有内容
    const trimmedText = text.trim()
    if (!trimmedText && !hasImages && !hasAudios && !hasFiles && !hasCommand) {
      return null
    }

    // OCR 模式
    if (isOcrMode()) {
      return sendOcrMessage({
        text: trimmedText,
        commands,
        files,
        resolution,
        onComplete,
      })
    }

    // 带图片或音频
    if (hasImages || hasAudios) {
      return sendImageMessage({
        text: trimmedText,
        images,
        audios,
        onComplete,
      })
    }

    // 纯文本
    return sendTextMessage({
      text: trimmedText,
      audios,
      onComplete,
    })
  }, [isOcrMode, sendOcrMessage, sendImageMessage, sendTextMessage])

  return {
    send,
    sendTextMessage,
    sendImageMessage,
    sendOcrMessage,
    isOcrMode,
  }
}
