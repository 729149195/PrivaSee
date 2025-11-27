import { useEffect, useRef } from 'react'
import { useStore } from '../store'

/**
 * 隐私推理自动触发 Hook
 * 从 AgentPage 中提取的核心推理触发逻辑
 * 
 * @param {object} params - 配置参数
 * @param {object} params.currentSession - 当前会话对象
 * @param {string} params.inferenceMode - 推断模式 ('extract' | 'direct')
 * @param {object} params.selectedLaw - 当前选中的法律
 * @param {string} params.input - 主输入框内容
 * @param {string} params.landingInput - 着陆页输入框内容
 * @param {Array} params.selectedAudios - 已选择的音频
 * @param {Array} params.selectedImages - 已选择的图片
 * @param {string|null} params.editingMessageId - 正在编辑的消息 ID
 * @param {string} params.editingContent - 编辑内容
 * @param {Array} params.editingAudios - 编辑音频
 * @param {Array} params.editingImages - 编辑图片
 * @param {string} params.originalEditingContent - 原始编辑内容
 * @param {Array} params.originalEditingAudios - 原始编辑音频
 * @param {Array} params.originalEditingImages - 原始编辑图片
 * @param {React.MutableRefObject} params.isAdoptingPendingRef - 采纳 pending 标志 ref
 * @param {React.MutableRefObject} params.lastInferenceRunCountRef - 上次推理签名 ref
 */
export function usePrivacyAutoInference({
  currentSession,
  inferenceMode,
  selectedLaw,
  input,
  landingInput,
  selectedAudios,
  selectedImages,
  editingMessageId,
  editingContent,
  editingAudios,
  editingImages,
  originalEditingContent,
  originalEditingAudios,
  originalEditingImages,
  isAdoptingPendingRef,
  lastInferenceRunCountRef,
}) {
  const {
    autoPrivacyInference,
    infonSessions,
    privacyInferences,
    sessionKeywords,
    abortPrivacyInference,
    startPrivacyInference,
    clearProtectionSuggestions,
    clearCurrentInferenceAndRestore,
    setPendingUserInput,
    setPendingAudios,
    setPendingImages,
  } = useStore()

  const currentSessionId = currentSession?.id

  // 记录上次推理时使用的法律key
  const lastInferenceLawKeyRef = useRef(null)

  // 会话切换时重置推理记录
  useEffect(() => {
    lastInferenceRunCountRef.current = ''
    lastInferenceLawKeyRef.current = null
  }, [currentSessionId, lastInferenceRunCountRef])

  // 计算当前签名（直接推断模式）
  const computeDirectModeSignature = () => {
    const isEditing = editingMessageId !== null
    const hasContentChanged = isEditing && (
      editingContent !== originalEditingContent ||
      JSON.stringify(editingImages) !== JSON.stringify(originalEditingImages) ||
      JSON.stringify(editingAudios) !== JSON.stringify(originalEditingAudios)
    )

    let pendingInput = ''
    let pendingAudios = []
    let pendingImages = []

    if (isEditing) {
      if (hasContentChanged) {
        pendingInput = (editingContent || '').trim()
        pendingAudios = editingAudios || []
        pendingImages = editingImages || []
      }
    } else {
      pendingInput = (input || landingInput || '').trim()
      pendingAudios = selectedAudios || []
      pendingImages = selectedImages || []
    }

    const audioHash = pendingAudios
      .map(audio => `${audio.id}:${audio.transcript?.length || 0}:${(audio.transcript || '').slice(0, 30)}`)
      .join('|')
    const imageHash = pendingImages
      .map(img => {
        const imgObj = typeof img === 'string' ? { id: 'legacy', url: img, status: 'done' } : img
        return `${imgObj.id}:${imgObj.status}:${imgObj.analysis?.length || 0}`
      })
      .join('|')
    const pendingHash = pendingInput ? `pending:${pendingInput.length}:${pendingInput.slice(0, 50)}` : ''
    
    return {
      signature: [pendingHash, audioHash, imageHash].filter(Boolean).join('||'),
      pendingInput,
      pendingAudios,
      pendingImages,
      hasContentChanged,
      isEditing,
    }
  }

  // 检查图片是否都已完成分析
  const checkAllImagesAnalyzed = (images) => {
    return images.every(img => {
      const imgObj = typeof img === 'string' ? { status: 'done' } : img
      return imgObj.status === 'done' || imgObj.status === 'error'
    })
  }

  // 清空当前推理结果
  const clearCurrentInference = () => {
    const privacyInferences = useStore.getState().privacyInferences || {}
    useStore.setState({
      privacyInferences: {
        ...privacyInferences,
        [currentSessionId]: {
          status: 'idle',
          risks: [],
          buffer: '',
          abortController: null,
          createdAt: Date.now(),
          updatedAt: Date.now()
        }
      }
    })
  }

  // 清空当前会话的关键词
  const clearSessionKeywords = () => {
    const currentSessionKeywords = useStore.getState().sessionKeywords?.[currentSessionId]
    if (currentSessionKeywords && currentSessionKeywords.size > 0) {
      const updatedKeywords = { ...useStore.getState().sessionKeywords }
      delete updatedKeywords[currentSessionId]
      useStore.setState({ sessionKeywords: updatedKeywords })
    }
  }

  // 主要推理触发逻辑
  useEffect(() => {
    if (!autoPrivacyInference) return
    if (!currentSessionId || !selectedLaw) return

    const currentInference = privacyInferences?.[currentSessionId]
    const isInferenceRunning = currentInference?.status === 'running'

    // 直接推断模式
    if (inferenceMode === 'direct') {
      const userMessages = (currentSession?.messages || []).filter(msg => msg.role === 'user')
      const {
        signature: currentSignature,
        pendingInput,
        pendingAudios,
        pendingImages,
        hasContentChanged,
        isEditing,
      } = computeDirectModeSignature()

      // 没有 pending 内容的情况处理
      if (!pendingInput && pendingAudios.length === 0 && pendingImages.length === 0) {
        // 编辑模式但内容未变化，保持当前状态
        if (isEditing && !hasContentChanged) {
          return
        }

        // 首次加载有消息但无推理结果，触发推理
        if (userMessages.length > 0 && !lastInferenceRunCountRef.current && 
            currentInference?.status !== 'done' && currentInference?.status !== 'running') {
          lastInferenceRunCountRef.current = 'initial'
          clearProtectionSuggestions?.()
          setPendingUserInput('')
          setPendingAudios([])
          setPendingImages([])

          const timer = setTimeout(() => {
            startPrivacyInference?.(null)
          }, 800)
          return () => clearTimeout(timer)
        }

        // 输入被清空
        if (lastInferenceRunCountRef.current && lastInferenceRunCountRef.current !== 'initial') {
          if (isAdoptingPendingRef.current) {
            // 发送消息后的清空，保留推理结果
            isAdoptingPendingRef.current = false
          } else {
            // 手动清空，清除当前推理并恢复上一次结果
            clearCurrentInferenceAndRestore?.()
            setPendingUserInput('')
            setPendingAudios([])
            setPendingImages([])
            clearSessionKeywords()
            lastInferenceRunCountRef.current = ''
          }
        }
        return
      }

      // 检测到变化，触发推理
      if (currentSignature !== lastInferenceRunCountRef.current && currentSignature) {
        // 图片还在处理中，等待完成
        if (pendingImages.length > 0 && !checkAllImagesAnalyzed(pendingImages)) {
          if (isInferenceRunning) {
            abortPrivacyInference?.()
          }
          clearCurrentInference()
          clearProtectionSuggestions?.()
          clearSessionKeywords()
          return
        }

        // 所有图片分析完成，触发推理
        lastInferenceRunCountRef.current = currentSignature

        if (isInferenceRunning) {
          abortPrivacyInference?.()
        }
        clearCurrentInference()
        clearProtectionSuggestions?.()

        setPendingUserInput(pendingInput)
        setPendingAudios(pendingAudios)
        setPendingImages(pendingImages)

        const timer = setTimeout(() => {
          startPrivacyInference?.(editingMessageId)
        }, 800)
        return () => clearTimeout(timer)
      }
      return
    }

    // 提取信息元模式
    const runs = infonSessions?.[currentSessionId]?.runs || []
    const hasAnyRunningInfons = runs.some(run => run.status === 'running')
    const pendingRuns = runs.filter(run => run.targetType === 'pending' && run.status === 'done' && !run.expiring)
    const messageRuns = runs.filter(run => run.targetType === 'message' && run.status === 'done' && !run.expiring)

    let currentSignature = ''
    let infonType = ''
    if (pendingRuns.length > 0) {
      currentSignature = pendingRuns.map(r => r.id).sort().join('|')
      infonType = 'pending'
    } else if (messageRuns.length > 0) {
      currentSignature = messageRuns.map(r => r.id).sort().join('|')
      infonType = 'message'
    }

    if (!currentSignature) return

    // 初始化：已有推理结果，记录签名
    if (!lastInferenceRunCountRef.current && currentInference?.status === 'done') {
      lastInferenceRunCountRef.current = currentSignature
      return
    }

    // 新信息元完成，触发推理
    if (currentSignature !== lastInferenceRunCountRef.current && !hasAnyRunningInfons) {
      lastInferenceRunCountRef.current = currentSignature

      if (isInferenceRunning) {
        abortPrivacyInference?.()
      }
      clearProtectionSuggestions?.()

      const timer = setTimeout(() => {
        startPrivacyInference?.(null)
      }, 300)
      return () => clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    infonSessions?.[currentSessionId]?.runs,
    currentSessionId,
    selectedLaw?.key,
    inferenceMode,
    autoPrivacyInference,
    input,
    landingInput,
    selectedAudios,
    selectedImages,
    editingMessageId,
    editingContent,
    editingAudios,
    editingImages,
    originalEditingContent,
    originalEditingImages,
    originalEditingAudios,
  ])

  // 信息元开始重新提取时中止推理
  useEffect(() => {
    if (!currentSessionId || inferenceMode === 'direct') return

    const runs = infonSessions?.[currentSessionId]?.runs || []
    const currentInference = privacyInferences?.[currentSessionId]
    const isInferenceRunning = currentInference?.status === 'running'
    const hasAnyRunningInfons = runs.some(
      run => run.status === 'running' && (run.targetType === 'pending' || run.targetType === 'message')
    )

    if (hasAnyRunningInfons && isInferenceRunning) {
      abortPrivacyInference?.()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    infonSessions?.[currentSessionId]?.runs,
    currentSessionId,
    privacyInferences?.[currentSessionId]?.status,
    inferenceMode,
  ])

  // 切换法律时中止推理
  useEffect(() => {
    if (!currentSessionId) return

    const currentInference = privacyInferences?.[currentSessionId]
    const isInferenceRunning = currentInference?.status === 'running'

    if (!lastInferenceLawKeyRef.current && selectedLaw?.key) {
      lastInferenceLawKeyRef.current = selectedLaw.key
      return
    }

    if (selectedLaw?.key && selectedLaw.key !== lastInferenceLawKeyRef.current && isInferenceRunning) {
      abortPrivacyInference?.()
    }

    if (selectedLaw?.key) {
      lastInferenceLawKeyRef.current = selectedLaw.key
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLaw?.key, currentSessionId])
}
