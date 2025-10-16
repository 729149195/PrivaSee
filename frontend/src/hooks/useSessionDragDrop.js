import { useState, useRef } from 'react'

/**
 * 会话拖拽排序 Hook
 * 管理会话列表的拖拽排序功能
 */
export function useSessionDragDrop() {
  const [draggingSessionId, setDraggingSessionId] = useState(null)
  const [reorderedSessions, setReorderedSessions] = useState(null)
  const chatItemRefs = useRef({})
  const lastReorderRef = useRef({ fromId: null, index: -1 })

  /**
   * 测量所有会话项的位置
   */
  const measureRects = () => {
    const result = {}
    try {
      const entries = Object.entries(chatItemRefs.current || {})
      for (const [id, el] of entries) {
        if (el && el.getBoundingClientRect) result[id] = el.getBoundingClientRect()
      }
    } catch (_) {}
    return result
  }

  /**
   * FLIP 动画
   */
  const animateFLIP = (prevRects, nextRects) => {
    try {
      const ids = Object.keys(nextRects || {})
      for (const id of ids) {
        const el = chatItemRefs.current[id]
        const prev = prevRects?.[id]
        const next = nextRects?.[id]
        if (!el || !prev || !next) continue
        const dy = prev.top - next.top
        if (!dy) continue
        el.style.transition = 'transform 0s'
        el.style.transform = `translateY(${dy}px)`
        requestAnimationFrame(() => {
          el.style.transition = 'transform 150ms ease'
          el.style.transform = ''
        })
      }
    } catch (_) {}
  }

  /**
   * 拖拽开始事件
   */
  const handleDragStartSession = (sessions) => (id) => (e) => {
    try {
      setDraggingSessionId(id)
      setReorderedSessions([...sessions])
      if (e?.dataTransfer) {
        e.dataTransfer.effectAllowed = 'move'
        e.dataTransfer.setData('text/plain', id)
      }
    } catch (_) {}
  }

  /**
   * 拖拽中事件
   */
  const handleDragOverSession = (sessions) => (id) => (e) => {
    e.preventDefault()
    if (e?.dataTransfer) e.dataTransfer.dropEffect = 'move'
    const fromId = draggingSessionId
    if (!fromId) return
    const targetEl = e.currentTarget
    const rect = targetEl?.getBoundingClientRect?.()
    if (!rect) return
    const y = e.clientY
    const ratio = (y - rect.top) / Math.max(1, rect.height)
    // 40%/60% 阈值：避免在中间区域频繁抖动
    const beforeZone = ratio < 0.4
    const afterZone = ratio > 0.6
    if (!beforeZone && !afterZone) return

    const list = Array.isArray(reorderedSessions) ? [...reorderedSessions] : [...sessions]
    const fromIdx = list.findIndex((s) => s.id === fromId)
    const targetIdx = list.findIndex((s) => s.id === id)
    if (fromIdx === -1 || targetIdx === -1) return

    // 计算期望插入索引：在目标项前或后
    let desiredIndex = beforeZone ? targetIdx : targetIdx + 1
    desiredIndex = Math.max(0, Math.min(desiredIndex, list.length))
    // 移除源项后索引校正
    const shiftAdjustedIndex = desiredIndex - (fromIdx < desiredIndex ? 1 : 0)
    if (shiftAdjustedIndex === fromIdx) return
    // 重复目标去重：同一 fromId + index 不重复执行
    if (lastReorderRef.current.fromId === fromId && lastReorderRef.current.index === shiftAdjustedIndex) return
    lastReorderRef.current = { fromId, index: shiftAdjustedIndex }

    const prevRects = measureRects()
    const [moved] = list.splice(fromIdx, 1)
    list.splice(shiftAdjustedIndex, 0, moved)
    setReorderedSessions(list)
    requestAnimationFrame(() => {
      const nextRects = measureRects()
      animateFLIP(prevRects, nextRects)
    })
  }

  /**
   * 放置事件
   */
  const handleDropSession = (sessions) => (id) => (e) => {
    e.preventDefault()
    const fromId = draggingSessionId || (e?.dataTransfer ? e.dataTransfer.getData('text/plain') : null)
    setDraggingSessionId(null)
    if (!fromId) { 
      setReorderedSessions(null)
      return 
    }
    const list = Array.isArray(reorderedSessions) ? reorderedSessions : sessions
    return list
  }

  /**
   * 拖拽结束事件
   */
  const handleDragEndSession = () => {
    setDraggingSessionId(null)
    setReorderedSessions(null)
    lastReorderRef.current = { fromId: null, index: -1 }
  }

  /**
   * 设置会话项的 ref
   */
  const setSessionRef = (id, el) => {
    if (el) chatItemRefs.current[id] = el
    else delete chatItemRefs.current[id]
  }

  return {
    draggingSessionId,
    reorderedSessions,
    handleDragStartSession,
    handleDragOverSession,
    handleDropSession,
    handleDragEndSession,
    setSessionRef
  }
}

