import { useEffect, useRef, useState } from 'react'

/**
 * Pending 信息元提取防抖 Hook
 * 从 AgentPage 中提取的 1.5 秒防抖逻辑
 * 
 * @param {object} params - 配置参数
 */
export function usePendingDebounce({
  // 当前状态
  hasMessages,
  input,
  landingInput,
  selectedImages,
  selectedAudios,
  selectedFiles,
  model,
  // 编辑状态
  editingMessageId,
  editingContent,
  editingImages,
  editingAudios,
  originalEditingContent,
  originalEditingImages,
  originalEditingAudios,
  // 模式
  autoPrivacyInference,
  // 回调
  startPendingInfons,
  clearAllPendingInfons,
  markExpiringInfons,
  // Refs
  isAdoptingPendingRef,
}) {
  const pendingTimerRef = useRef(null)
  const [isWaitingForDebounce, setIsWaitingForDebounce] = useState(false)

  useEffect(() => {
    // 如果未启用自动隐私保护，跳过
    if (!autoPrivacyInference) return

    // 清理上一个定时器
    if (pendingTimerRef.current) {
      clearTimeout(pendingTimerRef.current)
      pendingTimerRef.current = null
    }

    const isEditing = editingMessageId !== null

    // 编辑模式
    if (isEditing) {
      const textToUse = (editingContent || '').trim()
      const imgs = [...(editingImages || [])]
      const audios = [...(editingAudios || [])]

      // 检查内容是否修改
      const hasContentChanged =
        editingContent !== originalEditingContent ||
        JSON.stringify(editingImages) !== JSON.stringify(originalEditingImages) ||
        JSON.stringify(editingAudios) !== JSON.stringify(originalEditingAudios)

      // 内容未修改，清空 pending
      if (!hasContentChanged) {
        setIsWaitingForDebounce(false)
        if (!isAdoptingPendingRef?.current) {
          try { clearAllPendingInfons?.() } catch (_) {}
        }
        return
      }

      // 标记即将过期的信息元
      markExpiringInfons?.()

      // 无有效内容
      if (!textToUse && imgs.length === 0 && audios.length === 0) {
        setIsWaitingForDebounce(false)
        if (!isAdoptingPendingRef?.current) {
          try { clearAllPendingInfons?.() } catch (_) {}
        }
        return
      }

      // 启动防抖
      setIsWaitingForDebounce(true)
      pendingTimerRef.current = setTimeout(() => {
        try {
          startPendingInfons?.(textToUse, imgs, audios)
          setIsWaitingForDebounce(false)
        } catch (_) {}
        pendingTimerRef.current = null
      }, 1500)

      return () => {
        if (pendingTimerRef.current) {
          clearTimeout(pendingTimerRef.current)
          pendingTimerRef.current = null
        }
      }
    }

    // 非编辑模式
    const textToUse = hasMessages ? (input || '').trim() : (landingInput || '').trim()
    const imgs = [...(selectedImages || [])]
    const audios = [...(selectedAudios || [])]

    // 检查是否有有效内容（OCR 模式的命令标签不算有效内容）
    const isOcrMode = model === 'deepseek-ocr' || model === 'deepseek-ocr-local'
    const hasValidContent = textToUse ||
      imgs.length > 0 ||
      audios.length > 0 ||
      (selectedFiles?.length > 0 && !isOcrMode)

    if (!hasValidContent) {
      setIsWaitingForDebounce(false)
      if (!isAdoptingPendingRef?.current) {
        try { clearAllPendingInfons?.() } catch (_) {}
      }
      return
    }

    // 启动防抖
    setIsWaitingForDebounce(true)
    pendingTimerRef.current = setTimeout(() => {
      try {
        startPendingInfons?.(textToUse, imgs, audios)
        setIsWaitingForDebounce(false)
      } catch (_) {}
      pendingTimerRef.current = null
    }, 1500)

    return () => {
      if (pendingTimerRef.current) {
        clearTimeout(pendingTimerRef.current)
        pendingTimerRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    input,
    landingInput,
    selectedImages,
    selectedAudios,
    selectedFiles,
    hasMessages,
    editingMessageId,
    editingContent,
    editingImages,
    editingAudios,
    originalEditingContent,
    originalEditingImages,
    originalEditingAudios,
    autoPrivacyInference,
    model,
  ])

  return { isWaitingForDebounce, setIsWaitingForDebounce }
}
