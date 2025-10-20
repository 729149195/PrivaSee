import { useState, useRef } from 'react'
import { message as antdMessage } from 'antd'
import { useStore } from '../store'

/**
 * 消息处理逻辑 Hook
 * 处理消息的发送、编辑、复制、重试等操作
 * @param {object} getCurrentSession - 获取当前会话的函数
 * @param {object} infonSessions - 信息元会话对象
 * @param {object} privacyInferences - 隐私推理对象
 * @param {function} sendMessage - 发送消息的函数
 * @param {function} startMessageInfons - 开始消息信息元提取的函数
 * @param {function} clearAllPendingInfons - 清空所有 pending 信息元的函数
 * @param {object} lastInferenceRunCountRef - 上次推理 run count 的 ref
 */
export function useMessageHandlers(
  getCurrentSession,
  infonSessions,
  privacyInferences,
  sendMessage,
  startMessageInfons,
  clearAllPendingInfons,
  lastInferenceRunCountRef,
  inferenceMode,
  startPrivacyInference
) {
  const [editingMessageId, setEditingMessageId] = useState(null)
  const [editingContent, setEditingContent] = useState('')
  const [editingImages, setEditingImages] = useState([])
  const [editingAudios, setEditingAudios] = useState([])
  const [originalEditingContent, setOriginalEditingContent] = useState('') // 保存原始内容
  const [originalEditingImages, setOriginalEditingImages] = useState([]) // 保存原始图片
  const [originalEditingAudios, setOriginalEditingAudios] = useState([]) // 保存原始音频
  const isAdoptingPendingRef = useRef(false)

  /**
   * 复制消息内容
   */
  const handleCopyMessage = (content) => {
    navigator.clipboard.writeText(content).then(() => {
      antdMessage.success('已复制到剪贴板')
    }).catch(() => {
      antdMessage.error('复制失败')
    })
  }

  /**
   * 辅助函数：从消息内容中移除 <audio>...</audio> 标签
   * 编辑时只显示纯文本部分，音频通过AudioTag组件单独显示和编辑
   */
  const removeAudioTags = (content) => {
    if (typeof content !== 'string') return content
    // 移除所有 <audio>...</audio> 标签及其内容
    return content.replace(/<audio>[\s\S]*?<\/audio>/gi, '').trim()
  }

  /**
   * 开始编辑消息：进入编辑模式（不立即标记expiring）
   */
  const handleEditMessage = (messageId, content, images, audios) => {
    // 从content中移除音频标签，编辑时只编辑纯文本
    const contentWithoutAudio = removeAudioTags(content)
    
    setEditingMessageId(messageId)
    setEditingContent(contentWithoutAudio)
    setEditingImages(images || [])
    setEditingAudios(audios || [])
    // 保存原始内容、图片和音频，用于判断是否发生变化
    setOriginalEditingContent(contentWithoutAudio)
    setOriginalEditingImages(images || [])
    setOriginalEditingAudios(audios || [])
    
    console.log('[EditMessage] 进入编辑模式', { messageId, hasAudios: audios?.length })
  }
  
  /**
   * 标记即将过期的信息元：在开始提取pending信息元时调用
   */
  const markExpiringInfons = () => {
    if (!editingMessageId) return
    
    const session = getCurrentSession()
    if (session) {
      const currentInfonSession = infonSessions?.[session.id]
      if (currentInfonSession?.runs) {
        // 找到该消息的索引
        const messageIndex = session.messages.findIndex(m => m.id === editingMessageId)
        if (messageIndex !== -1) {
          // 获取该消息及后续消息的ID
          const affectedMessageIds = session.messages.slice(messageIndex).map(m => m.id)
          
          // 检查是否已经标记过
          const alreadyMarked = currentInfonSession.runs.some(run => 
            run.targetType === 'message' && 
            affectedMessageIds.includes(run.targetKey) && 
            run.expiring
          )
          
          if (!alreadyMarked) {
            // 标记这些消息的信息元为即将过期
            const updatedRuns = currentInfonSession.runs.map(run => {
              if (run.targetType === 'message' && affectedMessageIds.includes(run.targetKey)) {
                return { ...run, expiring: true }
              }
              return run
            })
            
            useStore.setState({
              infonSessions: {
                ...infonSessions,
                [session.id]: { ...currentInfonSession, runs: updatedRuns }
              }
            })
            
            console.log('[MarkExpiring] 标记即将过期的信息元', { affectedMessageIds })
          }
        }
      }
    }
  }

  /**
   * 取消编辑：移除"即将过期"标签，恢复原状态
   */
  const handleCancelEdit = () => {
    const session = getCurrentSession()
    if (session) {
      const currentInfonSession = infonSessions?.[session.id]
      if (currentInfonSession?.runs) {
        // 移除所有即将过期的标记
        const restoredRuns = currentInfonSession.runs.map(run => {
          if (run.expiring) {
            const { expiring, ...rest } = run
            return rest
          }
          return run
        })
        
        useStore.setState({
          infonSessions: {
            ...infonSessions,
            [session.id]: { ...currentInfonSession, runs: restoredRuns }
          }
        })
        
        console.log('[CancelEdit] 恢复即将过期的信息元')
      }
    }
    
    setEditingMessageId(null)
    setEditingContent('')
    setEditingImages([])
    setEditingAudios([])
    setOriginalEditingContent('')
    setOriginalEditingImages([])
    setOriginalEditingAudios([])
    // 清除 pending 信息元
    clearAllPendingInfons?.()
  }

  /**
   * 保存编辑
   */
  const handleSaveEdit = async () => {
    if (!editingMessageId) return
    
    const text = editingContent.trim()
    if (!text && editingImages.length === 0 && editingAudios.length === 0) {
      antdMessage.warning('消息内容不能为空')
      return
    }

    // 获取当前 session
    const session = getCurrentSession()
    if (!session) return

    // 找到要编辑的消息及其后续消息
    const messageIndex = session.messages.findIndex(m => m.id === editingMessageId)
    if (messageIndex === -1) return

    // 删除该消息及其后续的所有消息
    const newMessages = session.messages.slice(0, messageIndex)
    const deletedMessages = session.messages.slice(messageIndex)
    const deletedMessageIds = new Set(deletedMessages.map(m => m.id))
    
    // 更新 session 的消息列表
    const sessions = useStore.getState().sessions
    const updatedSessions = sessions.map(s => {
      if (s.id === session.id) {
        return { ...s, messages: newMessages }
      }
      return s
    })
    
    // 清理被删除消息的信息元和所有即将过期的信息元
    const currentInfonSession = infonSessions?.[session.id]
    if (currentInfonSession?.runs) {
      const filteredRuns = currentInfonSession.runs.filter(run => {
        // 删除属于被删除消息的 runs
        if (run.targetType === 'message' && deletedMessageIds.has(run.targetKey)) {
          return false
        }
        // 删除所有即将过期的 runs
        if (run.expiring) {
          return false
        }
        return true
      })
      
      useStore.setState({
        infonSessions: {
          ...infonSessions,
          [session.id]: { ...currentInfonSession, runs: filteredRuns }
        }
      })
      
      console.log('[SaveEdit] 删除被删除消息的信息元和即将过期的信息元')
    }
    
    // 获取当前推理结果
    const currentPrivacyInference = privacyInferences?.[session.id]
    
    // 在直接推理模式下，清空关键词（因为消息被修改，旧关键词可能不再适用）
    if (inferenceMode === 'direct') {
      const sessionKeywords = useStore.getState().sessionKeywords || {}
      if (sessionKeywords[session.id]) {
        const updatedKeywords = { ...sessionKeywords }
        delete updatedKeywords[session.id]
        useStore.setState({ sessionKeywords: updatedKeywords })
        console.log('[SaveEdit] 直接推理模式：清空关键词，等待重新推理')
      }
    }
    
    // 在提取信息元模式下清空推理结果
    if (inferenceMode === 'extract') {
      // 清空隐私推理结果和隐私保护建议
      if (currentPrivacyInference) {
        useStore.setState({
          privacyInferences: {
            ...privacyInferences,
            [session.id]: {
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
      
      // 清空隐私保护建议
      const protectionSuggestions = useStore.getState().protectionSuggestions
      if (protectionSuggestions?.[session.id]) {
        const newSuggestions = { ...protectionSuggestions }
        delete newSuggestions[session.id]
        useStore.setState({ protectionSuggestions: newSuggestions })
      }
      
      // 重置推理记录，以便在信息元提取完成后触发自动推理
      lastInferenceRunCountRef.current = ''
      console.log('[SaveEdit] 提取信息元模式：清空隐私推理结果、保护建议并重置推理记录')
    } else {
      // 直接推理模式：保留推理结果和高亮，只清空签名（与handleSend保持一致）
      // 如果pending输入框有内容，会触发新的推理；否则保持当前推理结果
      lastInferenceRunCountRef.current = ''
      console.log('[SaveEdit] 直接推理模式：保留推理结果，仅重置签名')
    }
    
    // 更新 store 的 sessions
    useStore.setState({ sessions: updatedSessions })

    // 在发送前，先获取当前的 pending runs
    const currentRuns = infonSessions?.[session.id]?.runs || []
    const pendingRunIds = currentRuns
      .filter(run => run.targetType === 'pending' && run.status === 'done')
      .map(r => r.id)
      .sort()
    
    // 设置标志，防止useEffect清空pending
    isAdoptingPendingRef.current = true
    
    // 发送新消息（包含音频）
    if (editingImages.length > 0 || editingAudios.length > 0) {
      const userId = await useStore.getState().sendMessageWithImages(text, editingImages, editingAudios)
      
      // 只在提取信息元模式下处理信息元相关逻辑
      if (inferenceMode === 'extract') {
        try {
          const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
          if (result.adopted === 0) {
            // 没有 pending infons，需要重新提取
            startMessageInfons?.(userId)
          }
          console.log('[SaveEdit] 信息元处理完成', { adopted: result.adopted, hasPending: pendingRunIds.length > 0, mode: inferenceMode })
        } catch (_) {}
      } else {
        console.log('[SaveEdit] 直接推理模式：跳过信息元处理')
      }
    } else {
      const userId = await sendMessage(text, editingAudios)
      
      // 只在提取信息元模式下处理信息元相关逻辑
      if (inferenceMode === 'extract') {
        try {
          const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
          if (result.adopted === 0) {
            // 没有 pending infons，需要重新提取
            startMessageInfons?.(userId)
          }
          console.log('[SaveEdit] 信息元处理完成', { adopted: result.adopted, hasPending: pendingRunIds.length > 0, mode: inferenceMode })
        } catch (_) {}
      } else {
        console.log('[SaveEdit] 直接推理模式：跳过信息元处理')
      }
    }

    // 清除标志并清理编辑状态（不调用handleCancelEdit，避免清空pending导致自动推理失败）
    isAdoptingPendingRef.current = false
    
    // 手动清理编辑状态
    setEditingMessageId(null)
    setEditingContent('')
    setEditingImages([])
    setEditingAudios([])
    setOriginalEditingContent('')
    setOriginalEditingImages([])
    setOriginalEditingAudios([])
    
    // 注意：不清空pending infons，因为它们已经被adopt到message了
    // clearAllPendingInfons?.() // 这里不调用
    
    // 在直接推理模式下，由于清空了关键词，需要触发推理以重新提取
    if (inferenceMode === 'direct') {
      // 清空 pendingUserInput 和 pendingAudios，确保推理使用所有已发送的消息
      useStore.getState().setPendingUserInput('')
      useStore.getState().setPendingAudios([])
      
      // 延迟触发推理，确保消息已经保存
      setTimeout(() => {
        const updatedSession = useStore.getState().getCurrentSession()
        if (updatedSession && updatedSession.messages.length > 0) {
          console.log('[SaveEdit] 直接推理模式：触发推理以重新提取关键词')
          startPrivacyInference?.(null) // 编辑已完成，传null
        }
      }, 300)
    }
    
    antdMessage.success('消息已更新并重新生成')
  }

  /**
   * 重试生成：根据点击的assistant消息进行重新生成
   * @param {string} assistantMessageId - 要重新生成的assistant消息的ID
   */
  const handleRetry = async (assistantMessageId) => {
    const session = getCurrentSession()
    if (!session || !session.messages || session.messages.length === 0) return

    const messages = session.messages
    
    // 找到该assistant消息的索引
    const assistantIndex = messages.findIndex(m => m.id === assistantMessageId)
    if (assistantIndex === -1) return
    
    // 向前查找对应的user消息（通常就是前一条）
    let userIndex = -1
    for (let i = assistantIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        userIndex = i
        break
      }
    }
    if (userIndex === -1) return
    
    const userMessage = messages[userIndex]
    const oldUserMessageId = userMessage.id
    
    // 判断这是否是最后一轮对话
    // 如果该assistant消息是最后一条消息，或者该assistant消息后面只有其他assistant消息（没有user消息），则认为是最后一轮
    const isLastRound = assistantIndex === messages.length - 1 || 
                        !messages.slice(assistantIndex + 1).some(m => m.role === 'user')
    
    console.log('[Retry] 重新生成', {
      assistantMessageId,
      userMessageId: oldUserMessageId,
      isLastRound,
      assistantIndex,
      userIndex
    })

    // 保存该用户消息的信息元
    const currentInfonSession = infonSessions?.[session.id]
    let savedUserInfonRuns = []
    if (currentInfonSession?.runs) {
      savedUserInfonRuns = currentInfonSession.runs.filter(run => 
        run.targetType === 'message' && run.targetKey === oldUserMessageId
      )
    }

    // 删除从用户消息开始的所有消息（包括用户和该用户后面的所有消息）
    const updatedMessages = messages.slice(0, userIndex)
    const deletedMessages = messages.slice(userIndex)
    const deletedMessageIds = deletedMessages.map(m => m.id)

    // 清理被删除消息的信息元
    if (currentInfonSession?.runs) {
      const filteredRuns = currentInfonSession.runs.filter(run => {
        if (run.targetType === 'message' && deletedMessageIds.includes(run.targetKey)) {
          return false
        }
        return true
      })
      
      useStore.setState({
        infonSessions: {
          ...infonSessions,
          [session.id]: { ...currentInfonSession, runs: filteredRuns }
        }
      })
    }
    
    // 如果不是最后一轮，需要清空隐私推理结果并重新推理
    // 如果是最后一轮，保持当前隐私推理状态不变
    if (!isLastRound) {
      console.log('[Retry] 不是最后一轮，清空隐私推理结果')
      const currentPrivacyInference = privacyInferences?.[session.id]
      if (currentPrivacyInference) {
        useStore.setState({
          privacyInferences: {
            ...privacyInferences,
            [session.id]: {
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
      
      // 重置推理记录，以便自动触发新的推理
      lastInferenceRunCountRef.current = ''
    } else {
      console.log('[Retry] 是最后一轮，保持隐私推理状态')
    }

    // 更新 session
    const sessions = useStore.getState().sessions
    const updatedSessions = sessions.map(s => {
      if (s.id === session.id) {
        return { ...s, messages: updatedMessages }
      }
      return s
    })
    useStore.setState({ sessions: updatedSessions })

    // 重新发送用户消息
    const hasImages = Array.isArray(userMessage.images) && userMessage.images.length > 0
    let newUserMessageId
    if (hasImages) {
      newUserMessageId = await useStore.getState().sendMessageWithImages(userMessage.content, userMessage.images)
    } else {
      newUserMessageId = await sendMessage(userMessage.content)
    }

    // 迁移信息元到新的用户消息（如果有保存的信息元）
    if (savedUserInfonRuns.length > 0 && newUserMessageId) {
      const updatedRuns = savedUserInfonRuns.map(run => ({
        ...run,
        targetKey: newUserMessageId // 更新到新的消息ID
      }))
      
      const latestInfonSession = useStore.getState().infonSessions?.[session.id]
      if (latestInfonSession) {
        useStore.setState({
          infonSessions: {
            ...useStore.getState().infonSessions,
            [session.id]: {
              ...latestInfonSession,
              runs: [...latestInfonSession.runs, ...updatedRuns]
            }
          }
        })
        
        // 如果是最后一轮，更新推理签名以避免触发自动推理
        if (isLastRound) {
          // 立即更新签名（user信息元已迁移）
          const finalRuns = [...latestInfonSession.runs, ...updatedRuns]
          const messageRuns = finalRuns.filter(
            run => run.targetType === 'message' && run.status === 'done'
          )
          if (messageRuns.length > 0) {
            const newSignature = messageRuns.map(r => r.id).sort().join('|')
            lastInferenceRunCountRef.current = newSignature
            console.log('[Retry] 最后一轮：迁移信息元后更新签名', { newSignature })
          }
        }
      }
    }
    
    // 如果是最后一轮，需要持续监听信息元变化并更新签名，避免agent信息元提取完成时触发推理
    if (isLastRound) {
      console.log('[Retry] 最后一轮：启动信息元监听，防止触发推理')
      
      // 记录agent生成完成前的最后一次检查时间
      let lastCheckTime = Date.now()
      let stableCount = 0 // 连续稳定的次数
      
      const checkInterval = setInterval(() => {
        const finalInfonSession = useStore.getState().infonSessions?.[session.id]
        const currentSession = useStore.getState().sessions?.find(s => s.id === session.id)
        
        if (finalInfonSession?.runs && currentSession) {
          // 检查是否有正在运行的信息元提取
          const hasRunningInfons = finalInfonSession.runs.some(run => run.status === 'running')
          // 检查是否正在生成回复
          const isGeneratingNow = useStore.getState().isGenerating
          
          // 如果没有正在运行的信息元提取且不在生成中，认为稳定
          if (!hasRunningInfons && !isGeneratingNow) {
            stableCount++
          } else {
            stableCount = 0
          }
          
          // 计算当前签名
          const allMessageRuns = finalInfonSession.runs.filter(
            run => run.targetType === 'message' && run.status === 'done'
          )
          if (allMessageRuns.length > 0) {
            const newSignature = allMessageRuns.map(r => r.id).sort().join('|')
            
            // 如果签名与当前记录不同，更新签名
            if (lastInferenceRunCountRef.current !== newSignature) {
              lastInferenceRunCountRef.current = newSignature
              console.log('[Retry] 最后一轮：检测到信息元变化，更新签名', { 
                newSignature,
                hasRunningInfons,
                isGeneratingNow
              })
            }
          }
          
          // 如果连续3次检查都稳定（没有运行中的任务），则停止监听
          if (stableCount >= 3) {
            console.log('[Retry] 最后一轮：信息元已稳定，停止监听')
            clearInterval(checkInterval)
          }
          
          // 超过30秒强制停止
          if (Date.now() - lastCheckTime > 30000) {
            console.log('[Retry] 最后一轮：超时，停止监听')
            clearInterval(checkInterval)
          }
        }
      }, 500) // 每500ms检查一次
      
      // 设置超时保护，避免内存泄漏
      setTimeout(() => {
        clearInterval(checkInterval)
      }, 35000)
    }
  }

  return {
    editingMessageId,
    editingContent,
    setEditingContent,
    editingImages,
    setEditingImages,
    editingAudios,
    setEditingAudios,
    originalEditingContent,
    originalEditingImages,
    originalEditingAudios,
    isAdoptingPendingRef,
    handleCopyMessage,
    handleEditMessage,
    markExpiringInfons,
    handleCancelEdit,
    handleSaveEdit,
    handleRetry
  }
}

