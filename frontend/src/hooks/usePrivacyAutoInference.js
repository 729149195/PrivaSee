import { useEffect, useRef } from 'react'
import { useStore } from '../store'

/**
 * 隐私推理自动触发 Hook
 * 当信息元提取完成时自动触发隐私推理
 */
export function usePrivacyAutoInference({
  currentSession,
  selectedLaw,
  lastInferenceRunCountRef,
}) {
  const {
    autoPrivacyInference, infonSessions, privacyInferences,
    abortPrivacyInference, startPrivacyInference,
    clearPrivacyInference, clearProtectionSuggestions,
  } = useStore()

  const currentSessionId = currentSession?.id
  const lastLawKeyRef = useRef(null)
  // 用 ref 管理 timer，避免 effect re-run 时被 cleanup 清除
  // （记忆流写入会更新 runs 引用导致 effect 重跑）
  const timerRef = useRef(null)

  const clearTimer = () => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null }
  }

  // 会话切换时重置 + 组件卸载时清理
  useEffect(() => {
    lastInferenceRunCountRef.current = ''
    lastLawKeyRef.current = null
    clearTimer()
  }, [currentSessionId, lastInferenceRunCountRef])

  useEffect(() => clearTimer, [])

  // 主触发逻辑：信息元完成时启动推理
  useEffect(() => {
    if (!autoPrivacyInference || !currentSessionId || !selectedLaw) return

    const inference = privacyInferences?.[currentSessionId]
    const runs = infonSessions?.[currentSessionId]?.runs || []
    const hasRunning = runs.some(r => r.status === 'running')
    const doneRuns = runs.filter(r =>
      (r.targetType === 'pending' || r.targetType === 'message') && r.status === 'done' && !r.expiring
    )

    const sig = doneRuns.map(r => r.id).sort().join('|')
    if (!sig) return

    // 初次加载已有结果，只记录签名
    if (!lastInferenceRunCountRef.current && inference?.status === 'done') {
      lastInferenceRunCountRef.current = sig
      return
    }

    // 签名变化 + 无正在运行的提取 → 触发推理
    if (sig !== lastInferenceRunCountRef.current && !hasRunning) {
      lastInferenceRunCountRef.current = sig
      if (inference?.status === 'running') abortPrivacyInference?.()
      clearPrivacyInference?.()
      clearProtectionSuggestions?.()
      clearTimer()
      timerRef.current = setTimeout(() => { timerRef.current = null; startPrivacyInference?.() }, 300)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [infonSessions?.[currentSessionId]?.runs, currentSessionId, selectedLaw?.key, autoPrivacyInference])

  // 新 infon 提取开始时，中止推理 + 取消待执行 timer
  useEffect(() => {
    if (!currentSessionId) return
    const runs = infonSessions?.[currentSessionId]?.runs || []
    const hasRunning = runs.some(r => r.status === 'running' && (r.targetType === 'pending' || r.targetType === 'message'))
    if (!hasRunning) return
    if (privacyInferences?.[currentSessionId]?.status === 'running') abortPrivacyInference?.()
    clearTimer()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [infonSessions?.[currentSessionId]?.runs, currentSessionId])

  // 切换法律时中止推理
  useEffect(() => {
    if (!currentSessionId || !selectedLaw?.key) return
    if (!lastLawKeyRef.current) { lastLawKeyRef.current = selectedLaw.key; return }
    if (selectedLaw.key !== lastLawKeyRef.current && privacyInferences?.[currentSessionId]?.status === 'running') {
      abortPrivacyInference?.()
    }
    lastLawKeyRef.current = selectedLaw.key
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLaw?.key, currentSessionId])
}
