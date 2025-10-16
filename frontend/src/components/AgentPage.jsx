import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'
import { Splitter, Progress, Spin } from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import WordCloud from './WordCloud'
import LawTree from './LawTree'
import Timeline from './Timeline'
import PrivacyRiskAnalysis from './PrivacyRiskAnalysis'
import UserAuth from '../users/UserAuth'
import { useUserStore } from '../users/userStore'
import PrivacyModeIndicator from '../users/PrivacyModeIndicator'

// 导入提取的工具函数
import { estimateTokens } from '../utils/tokenEstimation'
import { isModelMultimodal as checkModelMultimodal } from '../utils/modelUtils'
import { buildInfonIndex } from '../utils/infonUtils'

// 导入提取的组件
import InfonLegend from './agent/InfonLegend'
import ImagePreviewModal from './agent/ImagePreviewModal'
import InfonRunCard from './agent/InfonRunCard'
import ChatSessionItem from './agent/ChatSessionItem'
import ModelPickerToolbar from './agent/ModelPickerToolbar'
import MessageBubble from './agent/MessageBubble'
import LandingView from './agent/LandingView'
import MessageComposer from './agent/MessageComposer'

// 导入提取的 Hook
import { useImageSelection } from '../hooks/useImageSelection'
import { useInfonHighlight } from '../hooks/useInfonHighlight.jsx'
import { useSessionDragDrop } from '../hooks/useSessionDragDrop'
import { useMessageHandlers } from '../hooks/useMessageHandlers'


export default function AgentPage() {
  const {
    baseUrl,
    model,
    models,
    customProviders,
    addApiModel,
    sessions,
    currentSessionId,
    isGenerating,
    createSession,
    switchSession,
    deleteSession,
    renameSession,
    getCurrentSession,
    sendMessage,
    stopGenerating,
    _ensureCurrentSession,
    fetchModels,
    setModel,
    // 信息元提取
    startPendingInfons,
    abortPendingInfons,
    startMessageInfons,
    clearAllPendingInfons,
    infonSessions,
    // 隐私推理
    privacyInferences,
    startPrivacyInference,
    abortPrivacyInference,
    selectedLaw,
  } = useStore()

  // 用户状态：从用户 store 获取
  const { currentUser, isLoggedIn } = useUserStore()
  const { setCurrentUser, clearCurrentUser } = useStore()

  // 当前会话对象
  const currentSession = getCurrentSession()
  
  // 使用提取的 Hook
  const {
    selectedImages,
    setSelectedImages,
    previewImage,
    setPreviewImage,
    handlePickImages,
    removeSelectedImage,
  } = useImageSelection()

  const {
    getMessageInfons,
    getMessageRelations,
    getPendingInfons,
    pendingHighlights,
    pendingRelations,
    pendingInfonIndex,
    renderHighlightedText
  } = useInfonHighlight(currentSession, infonSessions)

  const {
    draggingSessionId,
    reorderedSessions,
    handleDragStartSession,
    handleDragOverSession,
    handleDropSession,
    handleDragEndSession,
    setSessionRef
  } = useSessionDragDrop()
  
  const lastInferenceRunCountRef = useRef('')

  const {
    editingMessageId,
    editingContent,
    setEditingContent,
    editingImages,
    setEditingImages,
    originalEditingContent,
    originalEditingImages,
    isAdoptingPendingRef,
    handleCopyMessage,
    handleEditMessage,
    markExpiringInfons,
    handleCancelEdit,
    handleSaveEdit,
    handleRetry
  } = useMessageHandlers(
    getCurrentSession,
    infonSessions,
    privacyInferences,
    sendMessage,
    startMessageInfons,
    clearAllPendingInfons,
    lastInferenceRunCountRef
  )

  // 同步用户登录状态到主 store：用于控制历史数据持久化
  useEffect(() => {
    if (isLoggedIn && currentUser?.id) {
      setCurrentUser(currentUser.id)
    } else {
      clearCurrentUser()
    }
  }, [isLoggedIn, currentUser, setCurrentUser, clearCurrentUser])

  // 多模态能力检测（使用提取的工具函数）
  const isModelMultimodal = useCallback((id) => checkModelMultimodal(id, customProviders), [customProviders])

  // 当前模型是否多模态
  const currentModelIsMultimodal = useMemo(() => isModelMultimodal(model), [model, isModelMultimodal])

  // 上下文是否已含图片
  const contextHasImages = useMemo(() => {
    const msgs = currentSession?.messages || []
    return msgs.some((m) => Array.isArray(m?.images) && m.images.length > 0)
  }, [currentSession?.messages])

  // 初始化当前会话：确保存在 currentSessionId
  useEffect(() => { _ensureCurrentSession() }, [_ensureCurrentSession])

  const [input, setInput] = useState('')
  const [landingInput, setLandingInput] = useState('')
  const mainScrollRef = useRef(null) // 主滚动区域
  const leftPaneScrollRef = useRef(null) // 左侧面板滚动区域
  const [maxContextTokens, setMaxContextTokens] = useState(null)
  // 1.5秒 防抖计时器
  const pendingTimerRef = useRef(null)
  // 追踪是否正在等待防抖：用于锁定发送按钮
  const [isWaitingForDebounce, setIsWaitingForDebounce] = useState(false)
  // 时间线选中的时间：用于筛选 WordCloud 中的信息元
  const [selectedTime, setSelectedTime] = useState(null)
  
  // 左侧栏编辑状态：用于追踪正在编辑的 session 和编辑的标题
  const [editingSessionId, setEditingSessionId] = useState(null)
  const [editingTitle, setEditingTitle] = useState('')
  
  // 记录上次推理时使用的法律key：用于检测法律切换
  const lastInferenceLawKeyRef = useRef(null)
  // 记录上次推理完成的时间戳：用于防止频繁触发
  const lastInferenceCompleteTimeRef = useRef(0)
  // 记录上次推理时的活跃信息元签名：用于检测"数量未变但内容替换"的情况
  const lastInferenceInfonSignatureRef = useRef('')
  
  // 会话切换时重置时间选择和推理记录（中文注释）
  useEffect(() => {
    setSelectedTime(null)
    lastInferenceRunCountRef.current = '' // 重置 message run IDs
    lastInferenceLawKeyRef.current = null // 重置法律记录
    lastInferenceCompleteTimeRef.current = 0 // 重置完成时间
    lastInferenceInfonSignatureRef.current = '' // 重置信息元签名
  }, [currentSessionId])

  // 隐私推理自动触发逻辑（中文注释）：信息元提取完成后自动触发
  // 核心逻辑：pending 或 message 信息元提取完成就触发推理
  useEffect(() => {
    if (!currentSession?.id || !selectedLaw) return
    
    const runs = infonSessions?.[currentSession.id]?.runs || []
    const currentInference = privacyInferences?.[currentSession.id]
    const isInferenceRunning = currentInference?.status === 'running'
    
    // 检查是否有任何信息元正在提取（message 或 pending）
    const hasAnyRunningInfons = runs.some(run => run.status === 'running')
    
    // 优先使用 pending infons（输入框中），其次使用 message infons（已发送）
    // 排除即将过期的信息元（用户正在编辑消息时）
    const pendingRuns = runs.filter(run => run.targetType === 'pending' && run.status === 'done' && !run.expiring)
    const messageRuns = runs.filter(run => run.targetType === 'message' && run.status === 'done' && !run.expiring)
    
    // 生成信息元签名：优先 pending，没有 pending 则用 message
    let currentSignature = ''
    let infonType = ''
    if (pendingRuns.length > 0) {
      currentSignature = pendingRuns.map(r => r.id).sort().join('|')
      infonType = 'pending'
    } else if (messageRuns.length > 0) {
      currentSignature = messageRuns.map(r => r.id).sort().join('|')
      infonType = 'message'
    }
    
    // 如果没有任何信息元，直接返回
    if (!currentSignature) return
    
    // 初始化：如果是刷新进入且已有推理结果，直接记录当前签名，不触发推理
    if (!lastInferenceRunCountRef.current && currentInference?.status === 'done') {
      console.log('[Privacy Inference] 初始化：已有推理结果，记录签名')
      lastInferenceRunCountRef.current = currentSignature
      return
    }
    
    // 检测到新的信息元完成，触发推理（无论当前推理状态）
    if (currentSignature !== lastInferenceRunCountRef.current && !hasAnyRunningInfons) {
      console.log('[Privacy Inference] 信息元提取完成，触发推理', {
        type: infonType,
        signature: currentSignature,
        lastSignature: lastInferenceRunCountRef.current,
        inferenceStatus: currentInference?.status
      })
      lastInferenceRunCountRef.current = currentSignature
      
      // 如果推理正在运行，先中止
      if (isInferenceRunning) {
        console.log('[Privacy Inference] 中止当前推理')
        abortPrivacyInference?.()
      }
      
      // 直接调用推理，和长按 law 按钮一样的逻辑
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
  ])

  // 推理中止逻辑1：任何信息元开始重新提取时（含 pending/message），若推理运行则立刻中止并恢复上次结果（中文注释）
  useEffect(() => {
    if (!currentSession?.id) return
    
    const runs = infonSessions?.[currentSession.id]?.runs || []
    const currentInference = privacyInferences?.[currentSession.id]
    const isInferenceRunning = currentInference?.status === 'running'
    
    // 检查是否有 pending 或 message 类型的信息元正在提取
    const hasAnyRunningInfons = runs.some(run => run.status === 'running' && (run.targetType === 'pending' || run.targetType === 'message'))
    
    // 如果任一信息元开始提取，且推理正在运行，中止推理
    if (hasAnyRunningInfons && isInferenceRunning) {
      console.log('[Privacy Inference] 中止推理：信息元开始重新提取（pending/message）')
      abortPrivacyInference?.()
    }
    
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    infonSessions?.[currentSessionId]?.runs,
    currentSessionId,
    privacyInferences?.[currentSessionId]?.status
  ])

  // 推理中止逻辑2：切换法律时，中止正在进行的推理（中文注释）
  useEffect(() => {
    if (!currentSession?.id) return
    
    const currentInference = privacyInferences?.[currentSession.id]
    const isInferenceRunning = currentInference?.status === 'running'
    
    // 记录当前选中的法律
    if (!lastInferenceLawKeyRef.current && selectedLaw?.key) {
      lastInferenceLawKeyRef.current = selectedLaw.key
      return
    }
    
    // 如果法律改变，且推理正在运行，中止推理
    if (selectedLaw?.key && selectedLaw.key !== lastInferenceLawKeyRef.current && isInferenceRunning) {
      console.log('[Privacy Inference] 中止推理：切换法律')
      abortPrivacyInference?.()
    }
    
    // 更新记录的法律
    if (selectedLaw?.key) {
      lastInferenceLawKeyRef.current = selectedLaw.key
    }
    
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLaw?.key, currentSessionId])

  // 监听信息元提取结果，当首次出现 SIT 类型时自动更新对话标题（中文注释）
  useEffect(() => {
    if (!currentSession?.id) return
    
    const runs = (infonSessions?.[currentSession.id]?.runs) || []
    // 只检查已完成的 run
    const doneRuns = runs.filter(r => r.status === 'done')
    
    for (const run of doneRuns) {
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      for (const infon of infons) {
        const type = String(infon.infon_type || '').toUpperCase()
        if (type === 'SIT' && infon.description) {
          // 获取当前会话标题
          const currentTitle = currentSession.title || ''
          // 如果标题是默认的 "New chat"，则更新为 SIT 的 description
          if (currentTitle === 'New chat') {
            const newTitle = String(infon.description).slice(0, 50) // 限制长度
            renameSession?.(currentSession.id, newTitle)
            return // 只更新一次
          }
        }
      }
    }
  }, [currentSession?.id, currentSession?.title, infonSessions, renameSession])

  // 计算当前会话的信息元数据（用于PrivacyRiskAnalysis）（中文注释）
  const wordData = useMemo(() => {
    if (!currentSession?.id) return []
    const runs = (infonSessions?.[currentSession.id]?.runs) || []
    const allInfons = []
    
    for (const run of runs) {
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      infons.forEach((infon) => {
        const type = String(infon.infon_type || '').toUpperCase()
        if (type === 'SIT') return
        allInfons.push(infon)
      })
    }
    
    return allInfons
  }, [currentSession?.id, infonSessions])

  // 获取当前会话的隐私推理结果（中文注释）
  const inference = useMemo(() => (currentSession ? privacyInferences?.[currentSession.id] : null), [currentSession, privacyInferences])

  // 默认注册 DeepSeek 示例：仅添加一次，已存在则跳过
  useEffect(() => {
    try {
      useStore.getState().addApiModel?.({ id: 'deepseek-chat', baseUrl: 'https://api.deepseek.com/v1', apiKey: 'sk-8c2ee9474f2f44f5969dcd5de280e634' })
    } catch (_) { }
  }, [])
  const contextTokensUsed = useMemo(() => estimateTokens(currentSession?.messages || [], model), [currentSession?.messages, model])
  const contextPercent = useMemo(() => {
    if (typeof maxContextTokens !== 'number' || maxContextTokens <= 0) return 0
    return Math.min(100, Math.round((contextTokensUsed / Math.max(1, maxContextTokens)) * 100))
  }, [contextTokensUsed, maxContextTokens])
  const hasMessages = useMemo(() => (currentSession?.messages || []).length > 0, [currentSession?.messages])

  // 自动滚动到底部（中文注释）：流式时保持跟随
  useEffect(() => {
    const mainEl = mainScrollRef.current
    const leftEl = leftPaneScrollRef.current
    if (mainEl) mainEl.scrollTop = mainEl.scrollHeight
    if (leftEl) leftEl.scrollTop = leftEl.scrollHeight
  }, [currentSession?.messages?.length, isGenerating])
  
  // 刷新重进或切换 chat 时自动滚动到底部（中文注释）
  useEffect(() => {
    const mainEl = mainScrollRef.current
    const leftEl = leftPaneScrollRef.current
    if (!currentSession?.messages?.length) return
    // 延迟滚动，确保 DOM 渲染完成
    const timer = setTimeout(() => {
      if (mainEl) mainEl.scrollTop = mainEl.scrollHeight
      if (leftEl) leftEl.scrollTop = leftEl.scrollHeight
    }, 100)
    return () => clearTimeout(timer)
  }, [currentSessionId, currentSession?.messages?.length])

  // 拉取模型列表（中文注释）：页面挂载时
  useEffect(() => { fetchModels?.() }, [fetchModels])

  // 当上下文或 pending 存在图片时，强制主模型为多模态（中文注释）
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
    } catch (_) { }
  }, [model, models, contextHasImages, selectedImages, setModel, isModelMultimodal])

  // 根据当前模型查询实际上下文窗口（中文注释）：优先 /api/show，其次 /v1/models
  useEffect(() => {
    const fetchCtx = async () => {
      try {
        const apiBase = (baseUrl || '').replace(/\/?v1\/?$/, '/api')
        let ctxVal = null

        // 优先：Ollama /api/show
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
            // 额外扫描 model_info 中的可能键（如 llama.context_length 等）
            if (!ctxVal && j && typeof j === 'object' && j.model_info && typeof j.model_info === 'object') {
              for (const [k, v] of Object.entries(j.model_info)) {
                if (/(context|num_ctx|max_context|max_tokens)/i.test(String(k))) {
                  const n = pickNum(v)
                  if (n && n > 0) { ctxVal = n; break }
                }
              }
            }
          }
        } catch (_) { }

        // 回退：OpenAI 兼容 /v1/models（某些服务会返回 context_length 等）
        if (!ctxVal) {
          try {
            const res2 = await fetch(`${baseUrl}/models`, { method: 'GET' })
            if (res2.ok) {
              const j2 = await res2.json().catch(() => ({}))
              const list = Array.isArray(j2?.data) ? j2.data : (Array.isArray(j2) ? j2 : (Array.isArray(j2?.models) ? j2.models : []))
              const m = (list || []).find((it) => (it?.id || it?.name || it) === model)
              if (m) {
                const pickNum = (v) => {
                  if (typeof v === 'number') return v
                  if (typeof v === 'string') { const n = parseInt(v, 10); return Number.isFinite(n) ? n : null }
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
          } catch (_) { }
        }

        if (typeof ctxVal === 'number' && ctxVal > 0) setMaxContextTokens(ctxVal)
      } catch (_) { }
    }
    fetchCtx()
  }, [baseUrl, model])

  const contextLabel = useMemo(() => {
    if (typeof maxContextTokens === 'number' && maxContextTokens > 0) {
      return `${contextTokensUsed}/${maxContextTokens} est.`
    }
    return `${contextTokensUsed} est.`
  }, [contextTokensUsed, maxContextTokens])


  // 发送锁定状态与阶段检测（中文注释）：计算当前隐私保护流程进度
  const sendLockState = useMemo(() => {
    if (!currentSession?.id) return { locked: false, stage: 'ready', label: 'Send' }
    
    // 正在等待防抖（用户输入中）
    if (isWaitingForDebounce) {
      return { locked: true, stage: 'waiting', label: 'Preparing...' }
    }
    
    const runs = infonSessions?.[currentSession.id]?.runs || []
    // 检查 pending 和 message 级别的信息元提取
    const hasRunningPendingInfons = runs.some(run => run.status === 'running' && run.targetType === 'pending')
    const hasRunningMessageInfons = runs.some(run => run.status === 'running' && run.targetType === 'message')
    const currentInference = privacyInferences?.[currentSession.id]
    const isInferenceRunning = currentInference?.status === 'running'
    
    // Pending 信息元提取中（用户刚输入完，还在提取）
    if (hasRunningPendingInfons) {
      return { locked: true, stage: 'extracting', label: 'Extracting Infons...' }
    }
    // Message 信息元提取中
    if (hasRunningMessageInfons) {
      return { locked: true, stage: 'extracting', label: 'Extracting Infons...' }
    }
    // 隐私推理中
    if (isInferenceRunning) {
      return { locked: true, stage: 'analyzing', label: 'Privacy Analyzing...' }
    }
    
    // 严格检查：只要存在任何 pending 信息元（无论状态），且没有完成推理，就锁定
    const hasPendingInfons = runs.some(run => run.targetType === 'pending')
    const hasCompletedInference = currentInference?.status === 'done'
    
    // 如果有 pending 信息元但推理未完成，保持锁定
    if (hasPendingInfons && selectedLaw) {
      if (!hasCompletedInference) {
        // 判断当前处于哪个等待阶段
        const hasDonePending = runs.some(run => run.targetType === 'pending' && run.status === 'done')
        if (hasDonePending) {
          return { locked: true, stage: 'waiting', label: 'Privacy Analyzing...' }
        } else {
          return { locked: true, stage: 'waiting', label: 'Extracting Infons...' }
        }
      }
    }
    
    return { locked: false, stage: 'ready', label: 'Send' }
  }, [currentSession?.id, infonSessions, privacyInferences, selectedLaw, isWaitingForDebounce])

  const handleSend = async () => {
    // 隐私保护流程未完成时禁止发送（中文注释）
    if (sendLockState.locked) return
    
    const text = (input || '').trim()
    const imgs = [...selectedImages]
    const hasImages = imgs.length > 0
    if (!text && !hasImages) return
    
    // 在发送前，先获取当前的 pending runs 用于后续签名计算
    const currentRuns = infonSessions?.[currentSession.id]?.runs || []
    const pendingRunIds = currentRuns
      .filter(run => run.targetType === 'pending' && run.status === 'done')
      .map(r => r.id)
      .sort()
    
    // 设置标志，防止useEffect清空pending
    isAdoptingPendingRef.current = true
    
    if (hasImages) {
      const userId = await useStore.getState().sendMessageWithImages(text, imgs)
      try {
        // 如果有 pending infons，先更新签名，再 adopt（避免时序问题）
        if (pendingRunIds.length > 0) {
          const messageSignature = pendingRunIds.join('|')
          lastInferenceRunCountRef.current = messageSignature
          console.log('[Send] 提前更新签名，避免重复推理', { signature: messageSignature })
        }
        
        const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
        if (result.adopted === 0) {
          // 没有 pending infons，需要重新提取
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 清空输入和图片（中文注释）
      isAdoptingPendingRef.current = false
      setInput('')
      setSelectedImages([])
    } else {
      const userId = await sendMessage(text)
      try {
        // 如果有 pending infons，先更新签名，再 adopt（避免时序问题）
        if (pendingRunIds.length > 0) {
          const messageSignature = pendingRunIds.join('|')
          lastInferenceRunCountRef.current = messageSignature
          console.log('[Send] 提前更新签名，避免重复推理', { signature: messageSignature })
        }
        
        const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
        if (result.adopted === 0) {
          // 没有 pending infons，需要重新提取
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 清空输入和图片（中文注释）
      isAdoptingPendingRef.current = false
      setInput('')
      setSelectedImages([])
    }
    
    // 注意：adoptPendingInfonsToMessage 已经处理了 pending infons，无需再清空
  }

  const handleLandingSend = async () => {
    // 隐私保护流程未完成时禁止发送（中文注释）
    if (sendLockState.locked) return
    
    const text = (landingInput || '').trim()
    const imgs = [...selectedImages]
    const hasImages = imgs.length > 0
    if (!text && !hasImages) return
    
    // 在发送前，先获取当前的 pending runs 用于后续签名计算
    const currentRuns = infonSessions?.[currentSession.id]?.runs || []
    const pendingRunIds = currentRuns
      .filter(run => run.targetType === 'pending' && run.status === 'done')
      .map(r => r.id)
      .sort()
    
    // 设置标志，防止useEffect清空pending
    isAdoptingPendingRef.current = true
    
    if (hasImages) {
      const userId = await useStore.getState().sendMessageWithImages(text, imgs)
      try {
        // 如果有 pending infons，先更新签名，再 adopt（避免时序问题）
        if (pendingRunIds.length > 0) {
          const messageSignature = pendingRunIds.join('|')
          lastInferenceRunCountRef.current = messageSignature
          console.log('[LandingSend] 提前更新签名，避免重复推理', { signature: messageSignature })
        }
        
        const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
        if (result.adopted === 0) {
          // 没有 pending infons，需要重新提取
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 清空输入和图片（中文注释）
      isAdoptingPendingRef.current = false
      setLandingInput('')
      setSelectedImages([])
    } else {
      const userId = await sendMessage(text)
      try {
        // 如果有 pending infons，先更新签名，再 adopt（避免时序问题）
        if (pendingRunIds.length > 0) {
          const messageSignature = pendingRunIds.join('|')
          lastInferenceRunCountRef.current = messageSignature
          console.log('[LandingSend] 提前更新签名，避免重复推理', { signature: messageSignature })
        }
        
        const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
        if (result.adopted === 0) {
          // 没有 pending infons，需要重新提取
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 清空输入和图片（中文注释）
      isAdoptingPendingRef.current = false
      setLandingInput('')
      setSelectedImages([])
    }
    
    // 注意：adoptPendingInfonsToMessage 已经处理了 pending infons，无需再清空
  }


  // 输入变化时，立刻中止 pending 的提取（但不清除结果，等新的提取覆盖）
  useEffect(() => {
    try { 
      abortPendingInfons?.(false)
    } catch (_) {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, landingInput, editingContent])

  // 1.5秒 防抖：在用户停止输入后启动 pending 提取（中文注释）
  // 支持主输入框和编辑框两种模式
  useEffect(() => {
    // 检查是否在编辑模式
    const isEditing = editingMessageId !== null
    
    // 编辑模式下：只响应编辑内容的变化，忽略主输入框
    if (isEditing) {
      const textToUse = (editingContent || '').trim()
      const imgs = [...editingImages]
      
      if (pendingTimerRef.current) {
        clearTimeout(pendingTimerRef.current)
        pendingTimerRef.current = null
      }
      
      // 检查内容是否真的修改了
      const hasContentChanged = 
        editingContent !== originalEditingContent || 
        JSON.stringify(editingImages) !== JSON.stringify(originalEditingImages)
      
      // 如果内容未修改，清空pending并返回
      if (!hasContentChanged) {
        setIsWaitingForDebounce(false)
        if (!isAdoptingPendingRef.current) {
          try { clearAllPendingInfons?.() } catch (_) {}
        }
        return
      }
      
      // 内容已修改：立即标记即将过期的信息元
      markExpiringInfons?.()
      
      // 若无输入也无图片，清空pending并返回
      if (!textToUse && imgs.length === 0) {
        setIsWaitingForDebounce(false)
        if (!isAdoptingPendingRef.current) {
          try { clearAllPendingInfons?.() } catch (_) {}
        }
        return
      }
      
      // 标记正在等待防抖
      setIsWaitingForDebounce(true)
      
      // 启动新的提取
      pendingTimerRef.current = setTimeout(() => {
        try { 
          clearAllPendingInfons?.()
          startPendingInfons?.(textToUse, imgs)
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
    
    // 非编辑模式：处理主输入框和landing输入框
    const textToUse = (hasMessages ? (input || '').trim() : (landingInput || '').trim())
    const imgs = [...selectedImages]
    
    if (pendingTimerRef.current) {
      clearTimeout(pendingTimerRef.current)
      pendingTimerRef.current = null
    }
    
    // 若无输入也无图片，则清空旧的 pending 并返回（中文注释）
    // 但如果正在采纳pending信息元（发送消息过程中），则不清空
    if (!textToUse && imgs.length === 0) {
      setIsWaitingForDebounce(false)
      if (!isAdoptingPendingRef.current) {
        try { clearAllPendingInfons?.() } catch (_) {}
      }
      return
    }
    
    // 标记正在等待防抖
    setIsWaitingForDebounce(true)
    
    // 启动新的提取前，清除旧的 pending 结果
    pendingTimerRef.current = setTimeout(() => {
      try { 
        clearAllPendingInfons?.() // 先清空旧结果
        startPendingInfons?.(textToUse, imgs) // 再启动新提取
        setIsWaitingForDebounce(false) // 防抖结束
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
  }, [input, landingInput, selectedImages, hasMessages, editingMessageId, editingContent, editingImages, originalEditingContent, originalEditingImages])

  // 会话拖拽排序事件处理（包装 Hook 函数以提交状态）
  const onDropSession = (id) => (e) => {
    const list = handleDropSession(sessions)(id)(e)
    if (list) {
      // 提交重排
      useStore.setState({ sessions: list })
    }
  }

  return (
    <div className={styles.shell}>
      {/* 左侧：侧边栏 */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarTop}>
          <button className={styles.newBtn} onClick={createSession}>
            <PlusOutlined className={styles.newBtnIcon} />
            <span>New chat</span>
          </button>
        </div>
        {/* 无痕模式提示（中文注释） */}
        <PrivacyModeIndicator />
        <div className={styles.sidebarScroll}>
          {(reorderedSessions || sessions).map((s) => (
            <ChatSessionItem
              key={s.id}
              session={s}
              currentSessionId={currentSessionId}
              editingSessionId={editingSessionId}
              editingTitle={editingTitle}
              draggingSessionId={draggingSessionId}
              onSwitch={switchSession}
              onRename={renameSession}
              onDelete={deleteSession}
              onEditStart={(id, title) => {
                setEditingSessionId(id)
                setEditingTitle(title)
              }}
              onEditEnd={() => {
                setEditingSessionId(null)
                setEditingTitle('')
              }}
              setEditingTitle={setEditingTitle}
              onDragStart={handleDragStartSession(sessions)}
              onDragOver={handleDragOverSession(sessions)}
              onDrop={onDropSession}
              onDragEnd={handleDragEndSession}
              setRef={setSessionRef}
            />
          ))}
        </div>
        <div className={styles.sidebarBottom}>
          {/* <div className={styles.kv}><span>Base URL</span><span>{baseUrl}</span></div> */}
          <div className={styles.kv}><span>Model</span><span>{model}</span></div>
          <div className={styles.contextSection}>
            <div className={styles.contextInfo} style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <div style={{ flex: 1 }}>
                <div className={styles.contextLabel}>Context window<span className={styles.contextText}>{contextLabel}</span></div>
                <Progress percent={contextPercent} size="small" className={styles.contextProgress} />
              </div>
              {/* 用户登录入口（中文注释）：放在 Context window 右上角 */}
              <div style={{ marginLeft: '8px' }}>
                <UserAuth />
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* 右侧：主区域 */}
      <section className={styles.main}>
        <div className={styles.scroll} ref={mainScrollRef}>
          {/* 顶部：左上角模型选择器 */}
          <ModelPickerToolbar
            model={model}
            models={models}
            customProviders={customProviders}
            setModel={setModel}
            addApiModel={addApiModel}
            contextHasImages={contextHasImages}
            selectedImagesCount={selectedImages.length}
          />
          <Splitter className={styles.splitterRoot}>
            <Splitter.Panel style={{ overflow: 'hidden', position: 'relative', display: 'flex', flexDirection: 'column' }}>
              {/* 信息元类型图例 */}
              <InfonLegend />
              <div className={styles.leftPaneScroll} ref={leftPaneScrollRef} style={{ flex: 1, overflow: 'auto' }}>
                {hasMessages ? (
                  <div className={styles.column}>
                    {(currentSession?.messages || []).map((m) => {
                      const isUser = m.role === 'user'
                      const messageRelations = getMessageRelations(m.id)
                      const infonList = getMessageInfons(m.id)
                      const infonIndex = buildInfonIndex(infonList)
                      
                      return (
                        <MessageBubble
                          key={m.id}
                          message={m}
                          isUser={isUser}
                          editingMessageId={editingMessageId}
                          editingContent={editingContent}
                          setEditingContent={setEditingContent}
                          editingImages={editingImages}
                          setEditingImages={setEditingImages}
                          originalEditingContent={originalEditingContent}
                          originalEditingImages={originalEditingImages}
                          onCopy={handleCopyMessage}
                          onEdit={handleEditMessage}
                          onSaveEdit={handleSaveEdit}
                          onCancelEdit={handleCancelEdit}
                          onRetry={handleRetry}
                          isGenerating={isGenerating}
                          renderHighlightedText={renderHighlightedText}
                          messageRelations={messageRelations}
                          infonIndex={infonIndex}
                          pendingHighlights={pendingHighlights}
                          pendingRelations={pendingRelations}
                          pendingInfonIndex={pendingInfonIndex}
                          sendLockState={sendLockState}
                        />
                      )
                    })}
                  </div>
                ) : (
                  <LandingView
                    landingInput={landingInput}
                    setLandingInput={setLandingInput}
                    onSend={handleLandingSend}
                    selectedImages={selectedImages}
                    setSelectedImages={setSelectedImages}
                    onRemoveImage={removeSelectedImage}
                    onImageClick={setPreviewImage}
                    sendLockState={sendLockState}
                    pendingHighlights={pendingHighlights}
                    pendingRelations={pendingRelations}
                    pendingInfonIndex={pendingInfonIndex}
                    currentModelIsMultimodal={currentModelIsMultimodal}
                  />
                )}
              </div>

              {/* 底部输入条：固定于左侧面板底部 */}
              {(currentSession && (currentSession.messages || []).length > 0) && (
                <MessageComposer
                  input={input}
                  setInput={setInput}
                  onSend={handleSend}
                  selectedImages={selectedImages}
                  setSelectedImages={setSelectedImages}
                  onRemoveImage={removeSelectedImage}
                  onImageClick={setPreviewImage}
                  isGenerating={isGenerating}
                  onStop={stopGenerating}
                  sendLockState={sendLockState}
                  pendingHighlights={pendingHighlights}
                  pendingRelations={pendingRelations}
                  pendingInfonIndex={pendingInfonIndex}
                  currentModelIsMultimodal={currentModelIsMultimodal}
                  isEditingMessage={editingMessageId !== null}
                />
              )}
            </Splitter.Panel>
            <Splitter.Panel defaultSize="35%" min="25%" max="50%">
              <div className={styles.rightPaneScroll}>
                <div className={styles.rightPaneHeader}>
                  <div className={styles.rightPaneTitle}>Privacy inference</div>
                </div>
                <div className={styles.rightPaneBody}>
                  {/* 法规 treemap 可视化（中文注释） */}
                  <LawTree />
                  {/* 隐私风险分析组件（中文注释） */}
                  <PrivacyRiskAnalysis
                    inference={inference}
                    selectedLaw={selectedLaw}
                  />
                  {/* 时间线组件（中文注释）：用于按时间筛选信息元 */}
                  <Timeline onTimeSelect={setSelectedTime} />
                  {/* 信息元词云可视化（中文注释） */}
                  <WordCloud selectedTime={selectedTime} />
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)', marginBottom: 8, paddingLeft: 4 }}>
                      Infons Results
                    </div>
                    <div className={styles.infonRuns}>
                      {(() => {
                        const runs = (infonSessions?.[currentSession?.id]?.runs) || []
                        if (!runs.length) return <div className={styles.infonEmpty}>No infons yet</div>
                        const sorted = [...runs].sort((a, b) => b.createdAt - a.createdAt)
                        return sorted.map((r) => <InfonRunCard key={r.id} run={r} />)
                      })()}
                    </div>
                  </div>
                </div>
              </div>
            </Splitter.Panel>
          </Splitter>
        </div>
      </section>

      {/* 图片预览 Modal */}
      <ImagePreviewModal previewImage={previewImage} onClose={() => setPreviewImage(null)} />
    </div>
  )
}