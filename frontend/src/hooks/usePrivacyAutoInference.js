import { useEffect, useRef } from 'react'
import { useStore } from '../store'

/**
 * 隐私推理自动触发 Hook
 * 从 AgentPage 中提取的核心推理触发逻辑（提取信息元模式）
 * 
 * @param {object} params - 配置参数
 * @param {object} params.currentSession - 当前会话对象
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
    abortPrivacyInference,
    startPrivacyInference,
    clearProtectionSuggestions,
  } = useStore()

  const currentSessionId = currentSession?.id

  // 记录上次推理时使用的法律key
  const lastInferenceLawKeyRef = useRef(null)

  // 会话切换时重置推理记录
  useEffect(() => {
    lastInferenceRunCountRef.current = ''
    lastInferenceLawKeyRef.current = null
  }, [currentSessionId, lastInferenceRunCountRef])

  // 主要推理触发逻辑（提取信息元模式）
  useEffect(() => {
    if (!autoPrivacyInference) return
    if (!currentSessionId || !selectedLaw) return

    const currentInference = privacyInferences?.[currentSessionId]
    const isInferenceRunning = currentInference?.status === 'running'

    // 提取信息元模式
    const runs = infonSessions?.[currentSessionId]?.runs || []
    const hasAnyRunningInfons = runs.some(run => run.status === 'running')
    const pendingRuns = runs.filter(run => run.targetType === 'pending' && run.status === 'done' && !run.expiring)
    const messageRuns = runs.filter(run => run.targetType === 'message' && run.status === 'done' && !run.expiring)

    let currentSignature = ''
    if (pendingRuns.length > 0) {
      currentSignature = pendingRuns.map(r => r.id).sort().join('|')
    } else if (messageRuns.length > 0) {
      currentSignature = messageRuns.map(r => r.id).sort().join('|')
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
        startPrivacyInference?.()
      }, 300)
      return () => clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    infonSessions?.[currentSessionId]?.runs,
    currentSessionId,
    selectedLaw?.key,
    autoPrivacyInference,
  ])

  // 信息元开始重新提取时中止推理
  useEffect(() => {
    if (!currentSessionId) return

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
