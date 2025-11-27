import { useEffect, useRef } from 'react'

/**
 * 自动滚动 Hook
 * 统一管理聊天区域的自动滚动逻辑
 * 
 * @param {object} params - 配置参数
 * @param {object} params.currentSession - 当前会话
 * @param {string} params.currentSessionId - 当前会话 ID
 * @param {boolean} params.isGenerating - 是否正在生成
 * @returns {object} refs - 滚动区域的 refs
 */
export function useAutoScroll({ currentSession, currentSessionId, isGenerating }) {
  const mainScrollRef = useRef(null)
  const leftPaneScrollRef = useRef(null)

  // 消息数量变化或生成状态变化时滚动到底部
  useEffect(() => {
    const mainEl = mainScrollRef.current
    const leftEl = leftPaneScrollRef.current
    if (mainEl) mainEl.scrollTop = mainEl.scrollHeight
    if (leftEl) leftEl.scrollTop = leftEl.scrollHeight
  }, [currentSession?.messages?.length, isGenerating])

  // 监听最后一条消息内容变化（用于流式响应）
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

  // 刷新重进或切换 chat 时滚动到底部
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

  return { mainScrollRef, leftPaneScrollRef }
}
