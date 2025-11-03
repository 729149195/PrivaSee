import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'
import { Splitter, Progress, Spin, Switch, Tooltip, Button } from 'antd'
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
import { useImageAnalysis } from '../hooks/useImageAnalysis'
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
    sessionKeywords,
    // 隐私保护建议
    protectionSuggestions,
    generateProtectionSuggestions,
    clearProtectionSuggestions,
    // 推断模式
    inferenceMode,
    setInferenceMode,
    setPendingUserInput,
    setPendingAudios,
    setPendingImages,
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
  
  // 使用图片分析 Hook
  const { processImageUpload } = useImageAnalysis(inferenceMode)

  const {
    getMessageInfons,
    getMessageRelations,
    getPendingInfons,
    pendingHighlights,
    pendingRelations,
    pendingInfonIndex,
    renderHighlightedText
  } = useInfonHighlight(currentSession, infonSessions, inferenceMode, privacyInferences, sessionKeywords)

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
    inferenceMode,
    startPrivacyInference
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
  const [selectedAudios, setSelectedAudios] = useState([]) // 已选择的音频
  const [selectedFiles, setSelectedFiles] = useState([]) // 已选择的文件（deepseek-ocr模式）
  const [selectedCommand, setSelectedCommand] = useState(null) // 已选择的命令（deepseek-ocr模式）
  const [selectedResolution, setSelectedResolution] = useState('gundam') // 已选择的分辨率模式（deepseek-ocr模式）
  const mainScrollRef = useRef(null) // 主滚动区域
  const leftPaneScrollRef = useRef(null) // 左侧面板滚动区域
  const [maxContextTokens, setMaxContextTokens] = useState(null)
  // 1.5秒 防抖计时器
  const pendingTimerRef = useRef(null)
  // 追踪是否正在等待防抖：用于锁定发送按钮
  const [isWaitingForDebounce, setIsWaitingForDebounce] = useState(false)
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

  // 隐私推理自动触发逻辑（中文注释）：信息元提取完成后或用户输入后自动触发
  // 核心逻辑：
  // - 提取信息元模式：pending 或 message 信息元提取完成就触发推理
  // - 直接推断模式：ONLY当pending输入框或编辑框内容变化时触发推理（发送消息不触发重新推理）
  useEffect(() => {
    if (!currentSession?.id || !selectedLaw) return
    
    const currentInference = privacyInferences?.[currentSession.id]
    const isInferenceRunning = currentInference?.status === 'running'
    
    // 直接推断模式：只监听pending输入的变化（不监听已发送消息的变化）
    if (inferenceMode === 'direct') {
      const userMessages = (currentSession.messages || []).filter(msg => msg.role === 'user')
      
      // 获取pending输入：优先使用编辑内容（如果在编辑模式），否则使用主输入框或landing输入框
      const isEditing = editingMessageId !== null
      
      // 检查编辑内容是否有变化
      const hasContentChanged = isEditing && (
        editingContent !== originalEditingContent || 
        JSON.stringify(editingImages) !== JSON.stringify(originalEditingImages) ||
        JSON.stringify(editingAudios) !== JSON.stringify(originalEditingAudios)
      )
      
      let pendingInput = ''
      let pendingAudios = []
      let pendingImages = []
      if (isEditing) {
        // 编辑模式：使用编辑内容（且与原始内容不同时才算有效的pending）
        if (hasContentChanged) {
          pendingInput = (editingContent || '').trim()
          pendingAudios = editingAudios || []
          pendingImages = editingImages || []
        }
      } else {
        // 非编辑模式：使用主输入框或landing输入框
        pendingInput = (input || landingInput || '').trim()
        pendingAudios = selectedAudios || []
        pendingImages = selectedImages || []
      }
      
      // 生成签名：基于pending输入、音频转写内容和图片分析内容
      // 这样发送消息后不会触发重新推理，只有pending内容变化才会触发
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
      const currentSignature = [pendingHash, audioHash, imageHash].filter(Boolean).join('||')
      
      // 如果没有pending输入、没有pending音频、也没有pending图片
      if (!pendingInput && pendingAudios.length === 0 && pendingImages.length === 0) {
        // 特殊情况：编辑模式但内容未变化，不执行任何操作，保持当前状态
        if (isEditing && !hasContentChanged) {
          console.log('[Privacy Inference] 直接推断模式：编辑模式但内容未变化，保持当前推理状态')
          return
        }
        
        // 如果有已发送消息且之前没有推理过，执行一次推理
        if (userMessages.length > 0 && !lastInferenceRunCountRef.current && currentInference?.status !== 'done' && currentInference?.status !== 'running') {
          console.log('[Privacy Inference] 直接推断模式：首次加载，有消息但无推理结果，触发推理')
          lastInferenceRunCountRef.current = 'initial'
          
          // 清空隐私保护建议
          clearProtectionSuggestions?.()
          
          // 清空 pendingUserInput、pendingAudios 和 pendingImages
          useStore.getState().setPendingUserInput('')
          useStore.getState().setPendingAudios([])
          useStore.getState().setPendingImages([])
          
          // 延迟触发推理
          const timer = setTimeout(() => {
            startPrivacyInference?.(null)
          }, 800)
          return () => clearTimeout(timer)
        }
        
        // 输入被清空：如果之前有推理（非initial状态）
        // 但如果是发送消息后自动清空（isAdoptingPendingRef.current为true），不要清除推理和关键词
        if (lastInferenceRunCountRef.current && lastInferenceRunCountRef.current !== 'initial') {
          if (isAdoptingPendingRef.current) {
            console.log('[Privacy Inference] 直接推断模式：发送消息后清空输入，保留推理结果和关键词')
            // 发送消息后的清空，保留推理结果和关键词
            // 重置标志
            isAdoptingPendingRef.current = false
          } else {
            console.log('[Privacy Inference] 直接推断模式：输入被手动清空，清除当前推理并恢复上一次结果')
            
            // 清除当前推理并恢复到上一次结果
            clearCurrentInferenceAndRestore?.()
            
            // 清空 pendingUserInput、pendingAudios 和 pendingImages
            useStore.getState().setPendingUserInput('')
            useStore.getState().setPendingAudios([])
            useStore.getState().setPendingImages([])
            
            // 清空当前会话的关键词（因为没有pending内容了）
            const currentSessionKeywords = useStore.getState().sessionKeywords?.[currentSession.id]
            if (currentSessionKeywords && currentSessionKeywords.size > 0) {
              console.log('[Privacy Inference] 输入被手动清空，清空关键词')
              const updatedKeywords = { ...useStore.getState().sessionKeywords }
              delete updatedKeywords[currentSession.id]
              useStore.setState({ sessionKeywords: updatedKeywords })
            }
            
            // 清空签名
            lastInferenceRunCountRef.current = ''
          }
        }
        
        return
      }
      
      // 检测到pending输入、音频或图片变化，触发推理
      if (currentSignature !== lastInferenceRunCountRef.current && currentSignature) {
        // 检查图片是否都已完成分析
        const allImagesAnalyzed = pendingImages.every(img => {
          const imgObj = typeof img === 'string' ? { status: 'done' } : img
          return imgObj.status === 'done' || imgObj.status === 'error'
        })
        
        // 如果有图片还在处理中：中止当前推理，但不触发新推理
        if (pendingImages.length > 0 && !allImagesAnalyzed) {
          console.log('[Privacy Inference] 直接推断模式：图片正在处理中，中止当前推理并等待完成')
          
          // 如果推理正在运行，先中止
          if (isInferenceRunning) {
            console.log('[Privacy Inference] 直接推理模式：中止当前推理')
            abortPrivacyInference?.()
          }
          
          // 直接推理模式：清空当前推理结果（不保留 previousRisks）
          console.log('[Privacy Inference] 直接推理模式：清空当前推理结果')
          const privacyInferences = useStore.getState().privacyInferences || {}
          useStore.setState({
            privacyInferences: {
              ...privacyInferences,
              [currentSession.id]: {
                status: 'idle',
                risks: [],
                buffer: '',
                abortController: null,
                createdAt: Date.now(),
                updatedAt: Date.now()
              }
            }
          })
          
          // 清空隐私保护建议
          clearProtectionSuggestions?.()
          
          // 清空当前会话的关键词
          const currentSessionKeywords = useStore.getState().sessionKeywords?.[currentSession.id]
          if (currentSessionKeywords && currentSessionKeywords.size > 0) {
            console.log('[Privacy Inference] 直接推理模式：清空旧关键词，等待图片处理完成')
            const updatedKeywords = { ...useStore.getState().sessionKeywords }
            delete updatedKeywords[currentSession.id]
            useStore.setState({ sessionKeywords: updatedKeywords })
          }
          
          // 不更新签名，等待图片处理完成后再触发
          return
        }
        
        // 所有图片都已完成分析，可以触发推理
        console.log('[Privacy Inference] 直接推断模式：检测到pending输入、音频或图片变化，触发推理', {
          signature: currentSignature,
          lastSignature: lastInferenceRunCountRef.current,
          messageCount: userMessages.length,
          pendingInputLength: pendingInput.length,
          pendingAudiosCount: pendingAudios.length,
          pendingImagesCount: pendingImages.length,
          inferenceStatus: currentInference?.status,
          editingMessageId: editingMessageId
        })
        lastInferenceRunCountRef.current = currentSignature
        
        // 如果推理正在运行，先中止
        if (isInferenceRunning) {
          console.log('[Privacy Inference] 直接推理模式：中止当前推理')
          abortPrivacyInference?.()
        }
        
        // 直接推理模式：清空当前推理结果（不保留 previousRisks）
        console.log('[Privacy Inference] 直接推理模式：清空当前推理结果')
        const privacyInferences = useStore.getState().privacyInferences || {}
        useStore.setState({
          privacyInferences: {
            ...privacyInferences,
            [currentSession.id]: {
              status: 'idle',
              risks: [],
              buffer: '',
              abortController: null,
              createdAt: Date.now(),
              updatedAt: Date.now()
            }
          }
        })
        
        // 清空隐私保护建议
        clearProtectionSuggestions?.()
        
        // 不要清空关键词，让新推理的结果自然覆盖（保持旧消息的高亮直到新推理完成）
        // 新推理会基于所有消息重新提取关键词，自动替换旧的关键词集合
        
        // 立即同步设置 pendingUserInput、pendingAudios 和 pendingImages（使用 useStore.getState() 确保同步）
        useStore.getState().setPendingUserInput(pendingInput)
        useStore.getState().setPendingAudios(pendingAudios)
        useStore.getState().setPendingImages(pendingImages)
        
        // 延迟触发推理，传递editingMessageId以排除正在编辑的消息
        const timer = setTimeout(() => {
          console.log('[Privacy Inference] 触发推理，当前 pendingUserInput:', useStore.getState().pendingUserInput?.substring(0, 50), 'pendingAudios:', useStore.getState().pendingAudios?.length, 'pendingImages:', useStore.getState().pendingImages?.length)
          startPrivacyInference?.(editingMessageId)
        }, 800)
        return () => clearTimeout(timer)
      }
      
      return
    }
    
    // 提取信息元模式：监听信息元提取状态
    const runs = infonSessions?.[currentSession.id]?.runs || []
    
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
      
      // 清空隐私保护建议
      clearProtectionSuggestions?.()
      
      // 直接调用推理，和长按 law 按钮一样的逻辑
      const timer = setTimeout(() => {
        startPrivacyInference?.(null)
      }, 300)
      return () => clearTimeout(timer)
    }
    
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    infonSessions?.[currentSessionId]?.runs,
    userMessageCount, // 只监听 user 消息数量，不监听 assistant 消息
    currentSessionId,
    selectedLaw?.key,
    inferenceMode,
    input, // 直接推断模式需要监听pending输入
    landingInput,
    selectedAudios, // 直接推断模式：监听音频数据
    selectedImages, // 直接推断模式：监听图片数据
    editingMessageId, // 直接推断模式：监听编辑状态
    editingContent, // 直接推断模式：监听编辑内容
    editingAudios, // 直接推断模式：监听编辑音频
    editingImages, // 直接推断模式：监听编辑图片
    originalEditingContent,
    originalEditingImages,
    originalEditingAudios,
  ])

  // 推理中止逻辑1：任何信息元开始重新提取时（含 pending/message），若推理运行则立刻中止并恢复上次结果（中文注释）
  // 注意：仅在提取信息元模式下有效
  useEffect(() => {
    if (!currentSession?.id) return
    
    // 直接推断模式：跳过此逻辑
    if (inferenceMode === 'direct') return
    
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
    privacyInferences?.[currentSessionId]?.status,
    inferenceMode,
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
  
  // 获取当前会话的保护建议（中文注释）
  const suggestions = useMemo(() => (currentSession ? protectionSuggestions?.[currentSession.id] : null), [currentSession, protectionSuggestions])

  // 内置模型已在 defaultApiModelsConfig.js 中配置，无需手动注册
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
  
  // 监听最后一条消息内容变化，自动滚动（用于流式响应）
  useEffect(() => {
    const mainEl = mainScrollRef.current
    const leftEl = leftPaneScrollRef.current
    const messages = currentSession?.messages || []
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      // 当最后一条消息是流式状态或正在处理时，持续滚动
      if (lastMessage?.streaming || isGenerating) {
        if (mainEl) mainEl.scrollTop = mainEl.scrollHeight
        if (leftEl) leftEl.scrollTop = leftEl.scrollHeight
      }
    }
  }, [currentSession?.messages, isGenerating])
  
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

  // 更新pending用户输入状态（用于直接推断模式）
  useEffect(() => {
    if (inferenceMode === 'direct') {
      // 如果在编辑模式，使用编辑内容；否则使用主输入框或landing输入框
      const isEditing = editingMessageId !== null
      let pendingText = ''
      if (isEditing) {
        pendingText = (editingContent || '').trim()
      } else {
        pendingText = hasMessages ? (input || '').trim() : (landingInput || '').trim()
      }
      setPendingUserInput(pendingText)
    } else {
      setPendingUserInput('')
    }
  }, [input, landingInput, hasMessages, inferenceMode, setPendingUserInput, editingMessageId, editingContent])

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

  // 根据当前模型查询实际上下文窗口（中文注释）：优先 API模型预定义值，其次 /api/show，再次 /v1/models
  useEffect(() => {
    const fetchCtx = async () => {
      try {
        let ctxVal = null

        // 优先：检查是否是 API 模型，有预定义的 contextLength
        const apiProvider = customProviders?.[model]
        if (apiProvider && typeof apiProvider.contextLength === 'number') {
          ctxVal = apiProvider.contextLength
        }

        // 如果不是 API 模型或没有预定义值，则尝试查询 Ollama /api/show
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
        }

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
  }, [baseUrl, model, customProviders])

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
    
    // 直接推断模式：跳过信息元相关的锁定检查
    if (inferenceMode === 'direct') {
      // 检查是否有图片正在处理中
      const isEditing = editingMessageId !== null
      const imagesToCheck = isEditing ? editingImages : selectedImages
      const hasProcessingImages = imagesToCheck.some(img => {
        const imgObj = typeof img === 'string' ? { status: 'done' } : img
        return imgObj.status === 'uploading' || imgObj.status === 'analyzing'
      })
      
      if (hasProcessingImages) {
        return { locked: true, stage: 'analyzing', label: 'Processing Images...' }
      }
      
      const currentInference = privacyInferences?.[currentSession.id]
      const isInferenceRunning = currentInference?.status === 'running'
      const hasCompletedInference = currentInference?.status === 'done'
      
      // 检查隐私推理状态
      if (isInferenceRunning) {
        return { locked: true, stage: 'analyzing', label: 'Privacy Analyzing...' }
      }
      
      // 检查是否有 pending 输入但推理未完成
      // 获取当前的 pending 输入
      let pendingInput = ''
      let pendingAudios = []
      let pendingImages = []
      
      if (isEditing) {
        const hasContentChanged = 
          editingContent !== originalEditingContent || 
          JSON.stringify(editingImages) !== JSON.stringify(originalEditingImages) ||
          JSON.stringify(editingAudios) !== JSON.stringify(originalEditingAudios)
        
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
      
      // 如果有 pending 输入但推理未完成，保持锁定
      const hasPendingContent = pendingInput || pendingAudios.length > 0 || pendingImages.length > 0
      
      if (hasPendingContent && selectedLaw && !hasCompletedInference) {
        return { locked: true, stage: 'waiting', label: 'Preparing...' }
      }
      
      return { locked: false, stage: 'ready', label: 'Send' }
    }
    
    // 提取信息元模式：完整的流程检查
    // 首先检查是否有图片正在处理中（任何模式下都需要等待图片处理完成）
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
    inferenceMode, 
    editingMessageId, 
    editingImages, 
    selectedImages,
    editingContent,
    originalEditingContent,
    editingAudios,
    originalEditingAudios,
    originalEditingImages,
    input,
    landingInput,
    selectedAudios
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
    
    // 提取图片 analysis 数据（直接推理模式）
    const imageAnalysisMap = {}
    if (inferenceMode === 'direct' && hasImages) {
      selectedImages.forEach(img => {
        const imgObj = typeof img === 'string' ? { url: img } : img
        if (imgObj.url && imgObj.analysis) {
          imageAnalysisMap[imgObj.url] = imgObj.analysis
        }
      })
    }
    
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
      useStore.getState().setPendingImages([])
      
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
        if (result.adopted === 0 && inferenceMode === 'extract') {
          // 没有 pending infons，需要重新提取（仅在提取信息元模式下）
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 标志会在 useEffect 中检测并重置
    } else if (hasImages || hasAudios) {
      const userId = await useStore.getState().sendMessageWithImages(text, imgs, audios, imageAnalysisMap)
      try {
        // 如果有 pending infons，先更新签名，再 adopt（避免时序问题）
        if (pendingRunIds.length > 0) {
          const messageSignature = pendingRunIds.join('|')
          lastInferenceRunCountRef.current = messageSignature
          console.log('[Send] 提前更新签名，避免重复推理', { signature: messageSignature })
        }

        const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
        if (result.adopted === 0 && inferenceMode === 'extract') {
          // 没有 pending infons，需要重新提取（仅在提取信息元模式下）
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 清空输入、图片和音频（中文注释）
      setInput('')
      setSelectedImages([])
      setSelectedAudios([])
      // 清空 pending 图片（直接推断模式）
      useStore.getState().setPendingImages([])
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
        if (result.adopted === 0 && inferenceMode === 'extract') {
          // 没有 pending infons，需要重新提取（仅在提取信息元模式下）
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 清空输入、图片和音频（中文注释）
      setInput('')
      setSelectedImages([])
      setSelectedAudios([])
      // 清空 pending 图片（直接推断模式）
      useStore.getState().setPendingImages([])
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
    
    // 提取图片 analysis 数据（直接推理模式）
    const imageAnalysisMap = {}
    if (inferenceMode === 'direct' && hasImages) {
      selectedImages.forEach(img => {
        const imgObj = typeof img === 'string' ? { url: img } : img
        if (imgObj.url && imgObj.analysis) {
          imageAnalysisMap[imgObj.url] = imgObj.analysis
        }
      })
    }
    
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
      useStore.getState().setPendingImages([])
      
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
        if (result.adopted === 0 && inferenceMode === 'extract') {
          // 没有 pending infons，需要重新提取（仅在提取信息元模式下）
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 标志会在 useEffect 中检测并重置
    } else if (hasImages || hasAudios) {
      const userId = await useStore.getState().sendMessageWithImages(text, imgs, audios, imageAnalysisMap)
      try {
        // 如果有 pending infons，先更新签名，再 adopt（避免时序问题）
        if (pendingRunIds.length > 0) {
          const messageSignature = pendingRunIds.join('|')
          lastInferenceRunCountRef.current = messageSignature
          console.log('[LandingSend] 提前更新签名，避免重复推理', { signature: messageSignature })
        }
        
        const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
        if (result.adopted === 0 && inferenceMode === 'extract') {
          // 没有 pending infons，需要重新提取（仅在提取信息元模式下）
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 清空输入、图片和音频（中文注释）
      setLandingInput('')
      setSelectedImages([])
      setSelectedAudios([])
      // 清空 pending 图片（直接推断模式）
      useStore.getState().setPendingImages([])
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
        if (result.adopted === 0 && inferenceMode === 'extract') {
          // 没有 pending infons，需要重新提取（仅在提取信息元模式下）
          startMessageInfons?.(userId)
        }
      } catch (_) {}
      // 清空输入、图片和音频（中文注释）
      setLandingInput('')
      setSelectedImages([])
      setSelectedAudios([])
      // 清空 pending 图片（直接推断模式）
      useStore.getState().setPendingImages([])
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

  // 1.5秒 防抖：在用户停止输入后启动 pending 提取（中文注释）
  // 支持主输入框和编辑框两种模式
  useEffect(() => {
    // 直接推断模式：跳过整个防抖逻辑，推理由专门的 useEffect 处理
    if (inferenceMode === 'direct') {
      return
    }
    
    // 检查是否在编辑模式
    const isEditing = editingMessageId !== null
    
    // 编辑模式下：只响应编辑内容的变化，忽略主输入框
    if (isEditing) {
      const textToUse = (editingContent || '').trim()
      const imgs = [...editingImages]
      const audios = [...(editingAudios || [])]
      
      if (pendingTimerRef.current) {
        clearTimeout(pendingTimerRef.current)
        pendingTimerRef.current = null
      }
      
      // 检查内容是否真的修改了
      const hasContentChanged = 
        editingContent !== originalEditingContent || 
        JSON.stringify(editingImages) !== JSON.stringify(originalEditingImages) ||
        JSON.stringify(editingAudios) !== JSON.stringify(originalEditingAudios)
      
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
      
      // 若无输入也无图片也无音频，清空pending并返回
      if (!textToUse && imgs.length === 0 && audios.length === 0) {
        setIsWaitingForDebounce(false)
        if (!isAdoptingPendingRef.current) {
          try { clearAllPendingInfons?.() } catch (_) {}
        }
        return
      }
      
      // 标记正在等待防抖（仅提取信息元模式）
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
    
    // 非编辑模式：处理主输入框和landing输入框
    const textToUse = (hasMessages ? (input || '').trim() : (landingInput || '').trim())
    const imgs = [...selectedImages]
    const audios = [...selectedAudios]
    
    if (pendingTimerRef.current) {
      clearTimeout(pendingTimerRef.current)
      pendingTimerRef.current = null
    }
    
    // 若无输入也无图片也无音频也无文件，也无有效的命令标签，则清空旧的 pending 并返回（中文注释）
    // 但如果正在采纳pending信息元（发送消息过程中），则不清空
    // deepseek-ocr模式下，只有命令标签而没有实际文本内容时，也不触发隐私推理
    const hasValidContent = textToUse ||
      imgs.length > 0 ||
      audios.length > 0 ||
      selectedFiles.length > 0 ||
      ((model === 'deepseek-ocr' || model === 'deepseek-ocr-local') ? false : false) // OCR模式的命令标签不算有效内容

    if (!hasValidContent) {
      setIsWaitingForDebounce(false)
      if (!isAdoptingPendingRef.current) {
        try { clearAllPendingInfons?.() } catch (_) {}
      }
      return
    }
    
    // 启动新的提取（不清空，让 startPendingInfons 自己处理）
    // 标记正在等待防抖（仅提取信息元模式）
    setIsWaitingForDebounce(true)
    pendingTimerRef.current = setTimeout(() => {
      try { 
        startPendingInfons?.(textToUse, imgs, audios) // 启动新提取
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
  }, [input, landingInput, selectedImages, selectedAudios, hasMessages, editingMessageId, editingContent, editingImages, editingAudios, originalEditingContent, originalEditingImages, originalEditingAudios])

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
              {/* 信息元类型图例 - 仅在提取模式下显示 */}
              {inferenceMode !== 'direct' && <InfonLegend />}
              
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
                          inferenceMode={inferenceMode}
                          processImageUpload={processImageUpload}
                          pendingRelations={pendingRelations}
                          pendingInfonIndex={pendingInfonIndex}
                          sendLockState={sendLockState}
                          currentModelIsMultimodal={currentModelIsMultimodal}
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
                    inferenceMode={inferenceMode}
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
                  inferenceMode={inferenceMode}
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
              <Splitter.Panel defaultSize="35%" min="25%" max="50%">
                <div className={styles.rightPaneScroll}>
                  <div className={styles.rightPaneHeader}>
                    <div className={styles.rightPaneTitle} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        Privacy inference
                        {/* 推断模式开关（中文注释）：低调样式，紧贴标题右侧 */}
                        <Tooltip title={inferenceMode === 'direct' ? '直接推断：跳过信息元提取，直接对输入进行隐私推断' : '提取信息元：先提取信息元，再基于信息元进行隐私推断'}>
                          <div style={{ 
                          display: 'inline-flex', 
                          alignItems: 'center', 
                          gap: 6, 
                          marginLeft: 12,
                          fontSize: 11, 
                          color: 'var(--color-text-tertiary)',
                          opacity: 0.7,
                          transition: 'opacity 0.2s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                        onMouseLeave={(e) => e.currentTarget.style.opacity = '0.7'}
                        >
                          <span style={{ userSelect: 'none', whiteSpace: 'nowrap' }}>
                            {inferenceMode === 'direct' ? '直接推断' : '提取信息元'}
                          </span>
                          <Switch 
                            size="small"
                            checked={inferenceMode === 'direct'}
                            onChange={(checked) => {
                              const newMode = checked ? 'direct' : 'extract'
                              setInferenceMode(newMode)
                            }}
                          />
                        </div>
                      </Tooltip>
                      </div>
                      
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
                    hasInference={inference?.status === 'done'}
                    hasEditingText={
                      editingMessageId !== null 
                        ? (editingContent || '').trim().length > 0
                        : (hasMessages ? (input || '').trim().length > 0 : (landingInput || '').trim().length > 0)
                    }
                  />
                  {/* 信息元相关组件（中文注释）：仅在提取信息元模式下显示 */}
                  {inferenceMode === 'extract' && (
                    <>
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
                    </>
                  )}
                </div>
              </div>
            </Splitter.Panel>
            )}
          </Splitter>
        </div>
      </section>

      {/* 图片预览 Modal */}
      <ImagePreviewModal previewImage={previewImage} onClose={() => setPreviewImage(null)} />
    </div>
  )
}