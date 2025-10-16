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
  lastInferenceRunCountRef
) {
  const [editingMessageId, setEditingMessageId] = useState(null)
  const [editingContent, setEditingContent] = useState('')
  const [editingImages, setEditingImages] = useState([])
  const [savedMessageInfons, setSavedMessageInfons] = useState(null)
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
   * 开始编辑消息：暂时移除该消息的信息元
   */
  const handleEditMessage = (messageId, content, images) => {
    setEditingMessageId(messageId)
    setEditingContent(content || '')
    setEditingImages(images || [])
    
    // 保存并暂时移除该消息的信息元
    const session = getCurrentSession()
    if (session) {
      const currentInfonSession = infonSessions?.[session.id]
      if (currentInfonSession?.runs) {
        // 找到该消息的所有信息元runs
        const messageRuns = currentInfonSession.runs.filter(run => 
          run.targetType === 'message' && run.targetKey === messageId
        )
        
        if (messageRuns.length > 0) {
          // 保存
          setSavedMessageInfons({ messageId, runs: messageRuns })
          
          // 从infonSessions中移除
          const filteredRuns = currentInfonSession.runs.filter(run => 
            !(run.targetType === 'message' && run.targetKey === messageId)
          )
          
          useStore.setState({
            infonSessions: {
              ...infonSessions,
              [session.id]: { ...currentInfonSession, runs: filteredRuns }
            }
          })
        }
      }
    }
  }

  /**
   * 取消编辑：恢复原消息的信息元
   */
  const handleCancelEdit = () => {
    // 恢复之前保存的信息元
    if (savedMessageInfons) {
      const session = getCurrentSession()
      if (session) {
        const currentInfonSession = infonSessions?.[session.id]
        if (currentInfonSession) {
          useStore.setState({
            infonSessions: {
              ...infonSessions,
              [session.id]: {
                ...currentInfonSession,
                runs: [...currentInfonSession.runs, ...savedMessageInfons.runs]
              }
            }
          })
        }
      }
      setSavedMessageInfons(null)
    }
    
    setEditingMessageId(null)
    setEditingContent('')
    setEditingImages([])
    // 清除 pending 信息元
    clearAllPendingInfons?.()
  }

  /**
   * 保存编辑
   */
  const handleSaveEdit = async () => {
    if (!editingMessageId) return
    
    const text = editingContent.trim()
    if (!text && editingImages.length === 0) {
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
    
    // 清理被删除消息的信息元
    const currentInfonSession = infonSessions?.[session.id]
    if (currentInfonSession?.runs) {
      const filteredRuns = currentInfonSession.runs.filter(run => {
        // 保留不属于被删除消息的 runs
        if (run.targetType === 'message' && deletedMessageIds.has(run.targetKey)) {
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
    
    // 清空隐私推理结果
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
    
    // 更新 store 的 sessions
    useStore.setState({ sessions: updatedSessions })

    // 在发送前，先获取当前的 pending runs 用于后续签名计算
    const currentRuns = infonSessions?.[session.id]?.runs || []
    const pendingRunIds = currentRuns
      .filter(run => run.targetType === 'pending' && run.status === 'done')
      .map(r => r.id)
      .sort()
    
    // 设置标志，防止useEffect清空pending
    isAdoptingPendingRef.current = true
    
    // 发送新消息
    if (editingImages.length > 0) {
      const userId = await useStore.getState().sendMessageWithImages(text, editingImages)
      try {
        // 如果有 pending infons，先更新签名，再 adopt（避免时序问题）
        if (pendingRunIds.length > 0) {
          const messageSignature = pendingRunIds.join('|')
          lastInferenceRunCountRef.current = messageSignature
          console.log('[SaveEdit] 提前更新签名，避免重复推理', { signature: messageSignature })
        }
        
        const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
        if (result.adopted === 0) {
          // 没有 pending infons，需要重新提取
          startMessageInfons?.(userId)
        }
      } catch (_) {}
    } else {
      const userId = await sendMessage(text)
      try {
        // 如果有 pending infons，先更新签名，再 adopt（避免时序问题）
        if (pendingRunIds.length > 0) {
          const messageSignature = pendingRunIds.join('|')
          lastInferenceRunCountRef.current = messageSignature
          console.log('[SaveEdit] 提前更新签名，避免重复推理', { signature: messageSignature })
        }
        
        const result = useStore.getState().adoptPendingInfonsToMessage?.(userId) || { adopted: 0, runIds: [] }
        if (result.adopted === 0) {
          // 没有 pending infons，需要重新提取
          startMessageInfons?.(userId)
        }
      } catch (_) {}
    }

    // 清除标志并清理编辑状态
    isAdoptingPendingRef.current = false
    handleCancelEdit()
    
    antdMessage.success('消息已更新并重新生成')
  }

  /**
   * 重试生成：保存用户消息的信息元，删除用户和助手消息，重新发送，然后迁移信息元
   */
  const handleRetry = async () => {
    const session = getCurrentSession()
    if (!session || !session.messages || session.messages.length === 0) return

    const messages = session.messages
    const lastUserIndex = [...messages].reverse().findIndex(m => m.role === 'user')
    if (lastUserIndex === -1) return

    const actualIndex = messages.length - 1 - lastUserIndex
    const lastUserMessage = messages[actualIndex]
    const oldUserMessageId = lastUserMessage.id

    // 保存该用户消息的信息元
    const currentInfonSession = infonSessions?.[session.id]
    let savedUserInfonRuns = []
    if (currentInfonSession?.runs) {
      savedUserInfonRuns = currentInfonSession.runs.filter(run => 
        run.targetType === 'message' && run.targetKey === oldUserMessageId
      )
    }

    // 删除从用户消息开始的所有消息（包括用户和助手消息）
    const updatedMessages = messages.slice(0, actualIndex)
    const deletedMessages = messages.slice(actualIndex)
    const deletedMessageIds = deletedMessages.map(m => m.id)

    // 清理被删除消息的信息元（暂时）
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
    
    // 清空隐私推理结果
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
    const hasImages = Array.isArray(lastUserMessage.images) && lastUserMessage.images.length > 0
    let newUserMessageId
    if (hasImages) {
      newUserMessageId = await useStore.getState().sendMessageWithImages(lastUserMessage.content, lastUserMessage.images)
    } else {
      newUserMessageId = await sendMessage(lastUserMessage.content)
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
      }
    }
  }

  return {
    editingMessageId,
    editingContent,
    setEditingContent,
    editingImages,
    setEditingImages,
    isAdoptingPendingRef,
    handleCopyMessage,
    handleEditMessage,
    handleCancelEdit,
    handleSaveEdit,
    handleRetry
  }
}

