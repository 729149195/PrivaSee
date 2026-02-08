import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'
import { Splitter, Progress, Spin, Tooltip, Button } from 'antd'
import { PlusOutlined, LeftOutlined, RightOutlined } from '@ant-design/icons'
import WordCloud from './WordCloud'
import LawTree from './LawTree'
import Timeline from './Timeline'
import PrivacyRiskAnalysis from './PrivacyRiskAnalysis'
import PrivacyProtectionSuggestions from './PrivacyProtectionSuggestions'
import UserAuth from '../users/UserAuth'
import { useUserStore } from '../users/userStore'
import PrivacyModeIndicator from '../users/PrivacyModeIndicator'

// 导入提取的工具函数
import { buildInfonIndex } from '../utils/infonUtils'

// 导入提取的组件
import InfonLegend from './agent/InfonLegend'
import ImagePreviewModal from './agent/ImagePreviewModal'
import ChatSessionItem from './agent/ChatSessionItem'
import ModelPickerToolbar from './agent/ModelPickerToolbar'
import MessageBubble from './agent/MessageBubble'
import LandingView from './agent/LandingView'
import MessageComposer from './agent/MessageComposer'

// 导入提取的 Hook
import { useImageSelection } from '../hooks/useImageSelection'
import { useImageAnalysis } from '../hooks/useImageAnalysis'
import { useInfonHighlight } from '../hooks/useInfonHighlight.jsx'
import { useSessionDragDrop } from '../hooks/useSessionDragDrop'
import { useMessageHandlers } from '../hooks/useMessageHandlers'
import { usePrivacyAutoInference } from '../hooks/usePrivacyAutoInference'
import { useAutoScroll } from '../hooks/useAutoScroll'
import { useModelContext } from '../hooks/useModelContext'
import { usePendingDebounce } from '../hooks/usePendingDebounce'
import { useUnifiedSend } from '../hooks/useUnifiedSend'
import MemoryStreamDebugPanel from './MemoryStreamDebugPanel'


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
    generateSessionTitle,
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
    clearCurrentInferenceAndRestore,
    selectedLaw,
    // 隐私保护建议
    protectionSuggestions,
    generateProtectionSuggestions,
    clearProtectionSuggestions,
    // 自动推理开关
    autoPrivacyInference,
    setSelectedLaw,
  } = useStore()

  // 用户状态：从用户 store 获取
  const { currentUser, isLoggedIn } = useUserStore()
  const { setCurrentUser, clearCurrentUser } = useStore()

  // 当前会话对象
  const currentSession = getCurrentSession()
  
  // 初始化默认法律（确保即使右边栏未展开也能正常推理）
  useEffect(() => {
    const initDefaultLaw = async () => {
      // 如果已经有选中的法律，跳过初始化
      if (selectedLaw) return
      
      // 加载默认法律 (PIPL)
      try {
        const res = await fetch('./law/PIPL.json')
        const lawData = await res.json()
        setSelectedLaw('PIPL', lawData)
        console.log('[AgentPage] 初始化默认法律: PIPL')
      } catch (error) {
        console.error('[AgentPage] 加载默认法律失败:', error)
      }
    }
    
    initDefaultLaw()
  }, [selectedLaw, setSelectedLaw]) // eslint-disable-line react-hooks/exhaustive-deps
  
  // 使用提取的 Hook
  const {
    selectedImages,
    setSelectedImages,
    previewImage,
    setPreviewImage,
    handlePickImages,
    removeSelectedImage,
  } = useImageSelection()
  
  // 使用图片分析 Hook
  const { processImageUpload } = useImageAnalysis()

  const {
    getMessageInfons,
    getMessageRelations,
    getPendingInfons,
    pendingHighlights,
    pendingRelations,
    pendingInfonIndex,
    renderHighlightedText
  } = useInfonHighlight(currentSession, infonSessions, privacyInferences)

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
    editingAudios,
    setEditingAudios,
    editingFiles,
    setEditingFiles,
    editingCommands,
    setEditingCommands,
    originalEditingContent,
    originalEditingImages,
    originalEditingAudios,
    originalEditingFiles,
    originalEditingCommands,
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
    lastInferenceRunCountRef,
    startPrivacyInference
  )

  // 使用模型上下文 Hook（多模态检测、上下文窗口计算）
  const {
    contextLabel,
    contextPercent,
    isModelMultimodal,
    currentModelIsMultimodal,
    contextHasImages,
  } = useModelContext({
    model,
    models,
    baseUrl,
    customProviders,
    currentSession,
    selectedImages,
    setModel,
  })

  // 使用自动滚动 Hook
  const { mainScrollRef, leftPaneScrollRef } = useAutoScroll({
    currentSession,
    currentSessionId,
    isGenerating,
  })

  // 同步用户登录状态到主 store：用于控制历史数据持久化
  useEffect(() => {
    if (isLoggedIn && currentUser?.id) {
      setCurrentUser(currentUser.id)
    } else {
      clearCurrentUser()
    }
  }, [isLoggedIn, currentUser, setCurrentUser, clearCurrentUser])

  // 初始化当前会话：确保存在 currentSessionId
  useEffect(() => { _ensureCurrentSession() }, [_ensureCurrentSession])

  const [input, setInput] = useState('')
  const [landingInput, setLandingInput] = useState('')
  const [selectedAudios, setSelectedAudios] = useState([]) // 已选择的音频
  const [selectedFiles, setSelectedFiles] = useState([]) // 已选择的文件（deepseek-ocr模式）
  const [selectedCommand, setSelectedCommand] = useState(null) // 已选择的命令（deepseek-ocr模式）
  const [selectedResolution, setSelectedResolution] = useState('gundam') // 已选择的分辨率模式（deepseek-ocr模式）
  // 时间线选中的时间：用于筛选 WordCloud 中的信息元
  const [selectedTime, setSelectedTime] = useState(null)
  
  // 右边栏显示/隐藏状态
  const [rightPanelVisible, setRightPanelVisible] = useState(false)
  
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

  // 用于直接推断模式：只监听 user 消息数量，不监听 assistant 消息
  const userMessageCount = useMemo(() => {
    return (currentSession?.messages || []).filter(msg => msg.role === 'user').length
  }, [currentSession?.messages])

  // hasMessages 标志（需要在 usePendingDebounce 之前定义）
  const hasMessages = useMemo(() => (currentSession?.messages || []).length > 0, [currentSession?.messages])

  // 使用防抖 Hook（提取信息元模式下的1.5秒防抖）
  const { isWaitingForDebounce } = usePendingDebounce({
    hasMessages,
    input,
    landingInput,
    selectedImages,
    selectedAudios,
    selectedFiles,
    model,
    editingMessageId,
    editingContent,
    editingImages,
    editingAudios,
    originalEditingContent,
    originalEditingImages,
    originalEditingAudios,
    autoPrivacyInference,
    startPendingInfons,
    clearAllPendingInfons,
    markExpiringInfons,
    isAdoptingPendingRef,
  })

  // 使用隐私推理自动触发 Hook
  usePrivacyAutoInference({
    currentSession,
    selectedLaw,
    lastInferenceRunCountRef,
  })

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
  
  // 获取当前会话的保护建议（中文注释）
  const suggestions = useMemo(() => (currentSession ? protectionSuggestions?.[currentSession.id] : null), [currentSession, protectionSuggestions])

  // 拉取模型列表（中文注释）：页面挂载时
  useEffect(() => { fetchModels?.() }, [fetchModels])

  // 发送锁定状态与阶段检测（中文注释）：计算当前隐私保护流程进度
  const sendLockState = useMemo(() => {
    if (!currentSession?.id) return { locked: false, stage: 'ready', label: 'Send' }
    
    // 正在等待防抖（用户输入中）
    if (isWaitingForDebounce) {
      return { locked: true, stage: 'waiting', label: 'Preparing...' }
    }
    
    // 首先检查是否有图片正在处理中
    const isEditing = editingMessageId !== null
    const imagesToCheck = isEditing ? editingImages : selectedImages
    const hasProcessingImages = imagesToCheck.some(img => {
      const imgObj = typeof img === 'string' ? { status: 'done' } : img
      return imgObj.status === 'uploading' || imgObj.status === 'analyzing'
    })
    
    if (hasProcessingImages) {
      return { locked: true, stage: 'analyzing', label: 'Processing Images...' }
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
  }, [
    currentSession?.id, 
    infonSessions, 
    privacyInferences, 
    selectedLaw, 
    isWaitingForDebounce, 
    editingMessageId, 
    editingImages, 
    selectedImages
  ])

  const removeSelectedAudio = (index) => {
    setSelectedAudios((prev) => prev.filter((_, i) => i !== index))
  }

  const handleTranscriptChange = (audioId, newTranscript) => {
    setSelectedAudios((prev) => 
      prev.map(audio => 
        audio.id === audioId ? { ...audio, transcript: newTranscript } : audio
      )
    )
  }

  const handleEditingTranscriptChange = (audioId, newTranscript) => {
    setEditingAudios((prev) => 
      prev.map(audio => 
        audio.id === audioId ? { ...audio, transcript: newTranscript } : audio
      )
    )
  }

  const handleSend = async () => {
    // 隐私保护流程未完成时禁止发送（中文注释）
    if (sendLockState.locked) return

    const text = (input || '').trim()
    // 提取图片 URL（兼容字符串和对象格式）
    const imgs = selectedImages.map(img => typeof img === 'string' ? img : img.url)
    const audios = [...selectedAudios]
    const hasImages = imgs.length > 0
    const hasAudios = audios.length > 0
    const hasFiles = selectedFiles.length > 0
    const hasCommand = selectedCommand != null

    // 检查是否有内容（文本、图片、音频、文件或命令）
    if (!text && !hasImages && !hasAudios && !hasFiles && !hasCommand) return
    
    // 在发送前，先获取当前的 pending runs 用于后续签名计算
    const currentRuns = infonSessions?.[currentSession.id]?.runs || []
    const pendingRunIds = currentRuns
      .filter(run => run.targetType === 'pending' && run.status === 'done')
      .map(r => r.id)
      .sort()
    
    // 设置标志，防止useEffect清空pending
    isAdoptingPendingRef.current = true
    
    // 判断是否是 OCR 模式（包括 API 和本地版本）
    const isOcrMode = model === 'deepseek-ocr' || model === 'deepseek-ocr-local'
    
    if (isOcrMode) {
      // OCR 模式：处理命令和文件
      // 立即清空输入、命令、文件和分辨率（不等待处理完成）
      const commandsToSend = selectedCommand ? [selectedCommand] : []
      const filesToSend = selectedFiles
      const resolutionToSend = selectedResolution
      const textToSend = text
      
      setInput('')
      setSelectedCommand(null)
      setSelectedFiles([])
      setSelectedResolution('gundam') // 重置为默认分辨率
      
      // 异步处理 OCR（不阻塞UI）
      const userId = await useStore.getState().sendMessageWithDeepSeekOCR(textToSend, commandsToSend, filesToSend, resolutionToSend)
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
      // 标志会在 useEffect 中检测并重置
    } else if (hasImages || hasAudios) {
      const userId = await useStore.getState().sendMessageWithImages(text, imgs, audios, {})
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
      // 清空输入、图片和音频（中文注释）
      setInput('')
      setSelectedImages([])
      setSelectedAudios([])
      // 标志会在 useEffect 中检测并重置
    } else {
      const userId = await sendMessage(text, audios)
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
      // 清空输入、图片和音频（中文注释）
      setInput('')
      setSelectedImages([])
      setSelectedAudios([])
      // 标志会在 useEffect 中检测并重置
    }
    
    // 自动生成会话标题（仅在第一条消息后）
    // 延迟执行，确保消息已经添加到 session
    setTimeout(() => {
      const currentSession = getCurrentSession()
      if (currentSession) {
        const userMessages = currentSession.messages.filter(msg => msg.role === 'user')
        console.log('[AgentPage] 检查是否需要生成标题，用户消息数:', userMessages.length)
        if (userMessages.length === 1) {
          console.log('[AgentPage] 触发标题生成')
          generateSessionTitle?.(currentSession.id)
        }
      }
    }, 1000)
    
    // 注意：adoptPendingInfonsToMessage 已经处理了 pending infons，无需再清空
  }

  const handleLandingSend = async () => {
    // 隐私保护流程未完成时禁止发送（中文注释）
    if (sendLockState.locked) return
    
    const text = (landingInput || '').trim()
    // 提取图片 URL（兼容字符串和对象格式）
    const imgs = selectedImages.map(img => typeof img === 'string' ? img : img.url)
    const audios = [...selectedAudios]
    const hasImages = imgs.length > 0
    const hasAudios = audios.length > 0
    const hasFiles = selectedFiles.length > 0
    const hasCommand = selectedCommand != null
    
    // 检查是否有内容（文本、图片、音频、文件或命令）
    if (!text && !hasImages && !hasAudios && !hasFiles && !hasCommand) return
    
    // 在发送前，先获取当前的 pending runs 用于后续签名计算
    const currentRuns = infonSessions?.[currentSession.id]?.runs || []
    const pendingRunIds = currentRuns
      .filter(run => run.targetType === 'pending' && run.status === 'done')
      .map(r => r.id)
      .sort()
    
    // 设置标志，防止useEffect清空pending
    isAdoptingPendingRef.current = true
    
    // 判断是否是 OCR 模式（包括 API 和本地版本）
    const isOcrMode = model === 'deepseek-ocr' || model === 'deepseek-ocr-local'
    
    if (isOcrMode) {
      // OCR 模式：处理命令和文件
      // 立即清空输入、命令、文件和分辨率（不等待处理完成）
      const commandsToSend = selectedCommand ? [selectedCommand] : []
      const filesToSend = selectedFiles
      const resolutionToSend = selectedResolution
      const textToSend = text
      
      setLandingInput('')
      setSelectedCommand(null)
      setSelectedFiles([])
      setSelectedResolution('gundam') // 重置为默认分辨率
      
      // 异步处理 OCR（不阻塞UI）
      const userId = await useStore.getState().sendMessageWithDeepSeekOCR(textToSend, commandsToSend, filesToSend, resolutionToSend)
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
      // 标志会在 useEffect 中检测并重置
    } else if (hasImages || hasAudios) {
      const userId = await useStore.getState().sendMessageWithImages(text, imgs, audios, {})
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
      // 清空输入、图片和音频（中文注释）
      setLandingInput('')
      setSelectedImages([])
      setSelectedAudios([])
      // 标志会在 useEffect 中检测并重置
    } else {
      const userId = await sendMessage(text, audios)
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
      // 清空输入、图片和音频（中文注释）
      setLandingInput('')
      setSelectedImages([])
      setSelectedAudios([])
      // 标志会在 useEffect 中检测并重置
    }
    
    // 自动生成会话标题（仅在第一条消息后）
    // 延迟执行，确保消息已经添加到 session
    setTimeout(() => {
      const currentSession = getCurrentSession()
      if (currentSession) {
        const userMessages = currentSession.messages.filter(msg => msg.role === 'user')
        console.log('[AgentPage] 检查是否需要生成标题，用户消息数:', userMessages.length)
        if (userMessages.length === 1) {
          console.log('[AgentPage] 触发标题生成')
          generateSessionTitle?.(currentSession.id)
        }
      }
    }, 1000)
    
    // 注意：adoptPendingInfonsToMessage 已经处理了 pending infons，无需再清空
  }


  // 生成隐私保护建议（中文注释）
  const handleGenerateSuggestions = useCallback(() => {
    // 检查是否在编辑模式
    const isEditing = editingMessageId !== null
    
    let textToUse = ''
    if (isEditing) {
      // 编辑模式：使用编辑框的内容
      textToUse = (editingContent || '').trim()
    } else {
      // 非编辑模式：使用主输入框或landing输入框的内容
      textToUse = (hasMessages ? (input || '').trim() : (landingInput || '').trim())
    }
    
    if (!textToUse) {
      console.warn('[Protection] 没有文本可供生成建议')
      return
    }
    
    generateProtectionSuggestions?.(textToUse, editingMessageId)
  }, [editingMessageId, editingContent, input, landingInput, hasMessages, generateProtectionSuggestions])
  
  // 应用隐私保护建议（中文注释）
  const handleApplySuggestion = useCallback((suggestion) => {
    if (!suggestion || !suggestion.modified_text) {
      console.warn('[Protection] 建议无效')
      return
    }
    
    const modifiedText = suggestion.modified_text
    
    // 检查是否在编辑模式
    const isEditing = editingMessageId !== null
    
    if (isEditing) {
      // 编辑模式：更新编辑框内容
      setEditingContent(modifiedText)
      console.log('[Protection] 已应用建议到编辑框')
    } else {
      // 非编辑模式：更新主输入框或landing输入框
      if (hasMessages) {
        setInput(modifiedText)
        console.log('[Protection] 已应用建议到主输入框')
      } else {
        setLandingInput(modifiedText)
        console.log('[Protection] 已应用建议到landing输入框')
      }
    }
    
    // 清除当前的建议（可选，让用户可以继续查看其他建议）
    // clearProtectionSuggestions?.()
  }, [editingMessageId, setEditingContent, hasMessages, setInput, setLandingInput])

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
          <Splitter className={styles.splitterRoot} key={rightPanelVisible ? 'with-panel' : 'no-panel'}>
            <Splitter.Panel style={{ overflow: 'hidden', position: 'relative', display: 'flex', flexDirection: 'column' }}>
              {/* 信息元类型图例 */}
              <InfonLegend />
              
              {/* 右边栏切换按钮 */}
              {!rightPanelVisible && (
                <div style={{ 
                  position: 'absolute', 
                  top: '16px', 
                  right: '16px', 
                  zIndex: 10 
                }}>
                  <Tooltip title="显示右边栏">
                    <Button
                      type="text"
                      icon={<LeftOutlined />}
                      onClick={() => setRightPanelVisible(true)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: '32px',
                        height: '32px',
                        borderRadius: '6px',
                        background: 'var(--color-bg-secondary)',
                        border: '1px solid var(--color-border-light)',
                        color: 'var(--color-text-secondary)',
                        transition: 'all 0.2s'
                      }}
                    />
                  </Tooltip>
                </div>
              )}
              
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
                          editingAudios={editingAudios}
                          setEditingAudios={setEditingAudios}
                          editingFiles={editingFiles}
                          setEditingFiles={setEditingFiles}
                          editingCommands={editingCommands}
                          setEditingCommands={setEditingCommands}
                          originalEditingContent={originalEditingContent}
                          originalEditingImages={originalEditingImages}
                          originalEditingAudios={originalEditingAudios}
                          originalEditingFiles={originalEditingFiles}
                          originalEditingCommands={originalEditingCommands}
                          onEditingTranscriptChange={handleEditingTranscriptChange}
                          onCopy={handleCopyMessage}
                          onEdit={handleEditMessage}
                          onSaveEdit={handleSaveEdit}
                          onCancelEdit={handleCancelEdit}
                          onRetry={handleRetry}
                          onImageClick={setPreviewImage}
                          isGenerating={isGenerating}
                          renderHighlightedText={renderHighlightedText}
                          messageRelations={messageRelations}
                          infonIndex={infonIndex}
                          pendingHighlights={pendingHighlights}
                          processImageUpload={processImageUpload}
                          pendingRelations={pendingRelations}
                          pendingInfonIndex={pendingInfonIndex}
                          sendLockState={sendLockState}
                          currentModelIsMultimodal={currentModelIsMultimodal}
                          model={model}
                          selectedResolution={selectedResolution}
                          setSelectedResolution={setSelectedResolution}
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
                    selectedAudios={selectedAudios}
                    setSelectedAudios={setSelectedAudios}
                    onRemoveAudio={removeSelectedAudio}
                    onTranscriptChange={handleTranscriptChange}
                    isGenerating={isGenerating}
                    onStop={stopGenerating}
                    sendLockState={sendLockState}
                    pendingHighlights={pendingHighlights}
                    pendingRelations={pendingRelations}
                    pendingInfonIndex={pendingInfonIndex}
                    currentModelIsMultimodal={currentModelIsMultimodal}
                    renderHighlightedText={renderHighlightedText}
                    processImageUpload={processImageUpload}
                    model={model}
                    selectedFiles={selectedFiles}
                    setSelectedFiles={setSelectedFiles}
                    onRemoveFile={(index) => setSelectedFiles((prev) => prev.filter((_, i) => i !== index))}
                    selectedCommand={selectedCommand}
                    setSelectedCommand={setSelectedCommand}
                    selectedResolution={selectedResolution}
                    setSelectedResolution={setSelectedResolution}
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
                  selectedAudios={selectedAudios}
                  setSelectedAudios={setSelectedAudios}
                  onRemoveAudio={removeSelectedAudio}
                  onTranscriptChange={handleTranscriptChange}
                  isGenerating={isGenerating}
                  onStop={stopGenerating}
                  sendLockState={sendLockState}
                  pendingHighlights={pendingHighlights}
                  pendingRelations={pendingRelations}
                  pendingInfonIndex={pendingInfonIndex}
                  currentModelIsMultimodal={currentModelIsMultimodal}
                  isEditingMessage={editingMessageId !== null}
                  renderHighlightedText={renderHighlightedText}
                  processImageUpload={processImageUpload}
                  model={model}
                  selectedFiles={selectedFiles}
                  setSelectedFiles={setSelectedFiles}
                  onRemoveFile={(index) => setSelectedFiles((prev) => prev.filter((_, i) => i !== index))}
                  selectedCommand={selectedCommand}
                  setSelectedCommand={setSelectedCommand}
                  selectedResolution={selectedResolution}
                  setSelectedResolution={setSelectedResolution}
                />
              )}
            </Splitter.Panel>
            {rightPanelVisible && (
              <Splitter.Panel defaultSize="28%" min="20%" max="45%">
                <div className={styles.rightPaneScroll}>
                  <div className={styles.rightPaneHeader}>
                    <div className={styles.rightPaneTitle} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span>Privacy inference</span>
                      
                      {/* 隐藏右边栏按钮 */}
                      <Tooltip title="隐藏右边栏">
                        <Button
                          type="text"
                          icon={<RightOutlined />}
                          onClick={() => setRightPanelVisible(false)}
                          size="small"
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'var(--color-text-tertiary)',
                            transition: 'all 0.2s'
                          }}
                        />
                      </Tooltip>
                    </div>
                  </div>
                <div className={styles.rightPaneBody}>
                  {/* 法规 treemap 可视化（中文注释） */}
                  <LawTree />
                  {/* 隐私风险分析组件（中文注释） */}
                  <PrivacyRiskAnalysis
                    inference={inference}
                    selectedLaw={selectedLaw}
                  />
                  {/* 隐私保护修改建议组件（中文注释） */}
                  <PrivacyProtectionSuggestions
                    suggestions={suggestions}
                    onApplySuggestion={handleApplySuggestion}
                    onGenerateSuggestions={handleGenerateSuggestions}
                    hasInference={inference?.status === 'done' && inference?.risks?.length > 0}
                    inferenceStatus={inference?.status}
                    hasRisks={inference?.risks?.length > 0}
                    hasEditingText={
                      editingMessageId !== null 
                        ? (editingContent || '').trim().length > 0
                        : (hasMessages ? (input || '').trim().length > 0 : (landingInput || '').trim().length > 0)
                    }
                  />
                  {/* 时间线组件（中文注释）：用于按时间筛选信息元 */}
                  <Timeline onTimeSelect={setSelectedTime} />
                  {/* 信息元词云可视化（中文注释） */}
                  <WordCloud selectedTime={selectedTime} />
                </div>
              </div>
            </Splitter.Panel>
            )}
          </Splitter>
        </div>
      </section>

      {/* 图片预览 Modal */}
      <ImagePreviewModal previewImage={previewImage} onClose={() => setPreviewImage(null)} />

      {/* 主记忆流调试面板（浮动，不影响布局，删除此行即可移除） */}
      <MemoryStreamDebugPanel />
    </div>
  )
}