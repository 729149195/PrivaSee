import { useEffect, useState, useMemo, useCallback } from 'react'
import { isModelMultimodal as checkModelMultimodal } from '../utils/modelUtils'
import { estimateTokens } from '../utils/tokenEstimation'

/**
 * 模型上下文 Hook
 * 处理模型选择、多模态检测、上下文窗口计算等逻辑
 * 
 * @param {object} params - 配置参数
 */
export function useModelContext({
  model,
  models,
  baseUrl,
  customProviders,
  currentSession,
  selectedImages,
  setModel,
}) {
  const [maxContextTokens, setMaxContextTokens] = useState(null)

  // 多模态能力检测
  const isModelMultimodal = useCallback(
    (id) => checkModelMultimodal(id, customProviders),
    [customProviders]
  )

  // 当前模型是否多模态
  const currentModelIsMultimodal = useMemo(
    () => isModelMultimodal(model),
    [model, isModelMultimodal]
  )

  // 上下文是否已含图片
  const contextHasImages = useMemo(() => {
    const msgs = currentSession?.messages || []
    return msgs.some((m) => Array.isArray(m?.images) && m.images.length > 0)
  }, [currentSession?.messages])

  // 当上下文或 pending 存在图片时，强制主模型为多模态
  useEffect(() => {
    try {
      const hasPendingImages = selectedImages.length > 0
      const needMultimodal = Boolean(contextHasImages || hasPendingImages)
      if (!needMultimodal) return
      if (model && isModelMultimodal(model)) return
      
      const list = [model, ...(models || [])].filter((v, i, a) => v && a.indexOf(v) === i)
      const preferred = 'gemma3:12b'
      const mm = list.filter((id) => isModelMultimodal(id))
      
      if (mm.includes(preferred)) setModel?.(preferred)
      else if (mm.length) setModel?.(mm[0])
    } catch (_) {}
  }, [model, models, contextHasImages, selectedImages, setModel, isModelMultimodal])

  // 获取模型的上下文窗口大小
  useEffect(() => {
    const fetchCtx = async () => {
      try {
        let ctxVal = null

        // 优先检查 API 模型预定义值
        const apiProvider = customProviders?.[model]
        if (apiProvider && typeof apiProvider.contextLength === 'number') {
          ctxVal = apiProvider.contextLength
        }

        // 查询 Ollama /api/show
        if (!ctxVal) {
          const apiBase = (baseUrl || '').replace(/\/?v1\/?$/, '/api')
          try {
            const res = await fetch(`${apiBase}/show`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: model }),
            })
            if (res.ok) {
              const j = await res.json().catch(() => ({}))
              const pickNum = (v) => {
                if (typeof v === 'number') return v
                if (typeof v === 'string') {
                  const n = parseInt(v, 10)
                  return Number.isFinite(n) ? n : null
                }
                return null
              }
              ctxVal = (
                pickNum(j?.parameters?.num_ctx) ||
                pickNum(j?.details?.context_length) ||
                pickNum(j?.model_info?.context) ||
                pickNum(j?.model_info?.num_ctx) ||
                pickNum(j?.context) ||
                null
              )
              // 扫描 model_info 中的可能键
              if (!ctxVal && j?.model_info) {
                for (const [k, v] of Object.entries(j.model_info)) {
                  if (/(context|num_ctx|max_context|max_tokens)/i.test(String(k))) {
                    const n = pickNum(v)
                    if (n && n > 0) { ctxVal = n; break }
                  }
                }
              }
            }
          } catch (_) {}
        }

        // 回退：OpenAI 兼容 /v1/models
        if (!ctxVal) {
          try {
            const res2 = await fetch(`${baseUrl}/models`, { method: 'GET' })
            if (res2.ok) {
              const j2 = await res2.json().catch(() => ({}))
              const list = Array.isArray(j2?.data) ? j2.data : 
                (Array.isArray(j2) ? j2 : (Array.isArray(j2?.models) ? j2.models : []))
              const m = list.find((it) => (it?.id || it?.name || it) === model)
              if (m) {
                const pickNum = (v) => {
                  if (typeof v === 'number') return v
                  if (typeof v === 'string') {
                    const n = parseInt(v, 10)
                    return Number.isFinite(n) ? n : null
                  }
                  return null
                }
                ctxVal = (
                  pickNum(m?.context_length) ||
                  pickNum(m?.max_context) ||
                  pickNum(m?.tokenLimit) ||
                  pickNum(m?.max_tokens) ||
                  pickNum(m?.max_input_tokens) ||
                  pickNum(m?.details?.context_length) ||
                  pickNum(m?.parameters?.num_ctx) ||
                  null
                )
              }
            }
          } catch (_) {}
        }

        if (typeof ctxVal === 'number' && ctxVal > 0) {
          setMaxContextTokens(ctxVal)
        }
      } catch (_) {}
    }
    fetchCtx()
  }, [baseUrl, model, customProviders])

  // 计算已使用的 tokens
  const contextTokensUsed = useMemo(
    () => estimateTokens(currentSession?.messages || [], model),
    [currentSession?.messages, model]
  )

  // 计算使用百分比
  const contextPercent = useMemo(() => {
    if (typeof maxContextTokens !== 'number' || maxContextTokens <= 0) return 0
    return Math.min(100, Math.round((contextTokensUsed / Math.max(1, maxContextTokens)) * 100))
  }, [contextTokensUsed, maxContextTokens])

  // 上下文标签文本
  const contextLabel = useMemo(() => {
    if (typeof maxContextTokens === 'number' && maxContextTokens > 0) {
      return `${contextTokensUsed}/${maxContextTokens} est.`
    }
    return `${contextTokensUsed} est.`
  }, [contextTokensUsed, maxContextTokens])

  return {
    maxContextTokens,
    contextTokensUsed,
    contextPercent,
    contextLabel,
    isModelMultimodal,
    currentModelIsMultimodal,
    contextHasImages,
  }
}
