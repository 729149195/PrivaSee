import React, { useMemo, useState, useRef, useEffect } from 'react'
import { useStore } from '../store'
import styles from './Timeline.module.css'

// 时间线组件（中文注释）：扁平横向时间线，用于按 record_time 筛选信息元
export default function Timeline({ onTimeSelect }) {
  const { getCurrentSession, infonSessions } = useStore()
  const session = getCurrentSession()
  const runs = useMemo(() => (session ? (infonSessions?.[session.id]?.runs || []) : []), [session, infonSessions])
  
  // 当前选中的时间点（中文注释）
  const [selectedTime, setSelectedTime] = useState(null)
  
  // 保存已显示的时间节点（中文注释）：用于流式显示
  const displayedTimesRef = useRef(new Set())
  
  // 拖拽状态（中文注释）
  const [isDragging, setIsDragging] = useState(false)
  const [scrollLeft, setScrollLeft] = useState(0)
  const [startX, setStartX] = useState(0)
  const containerRef = useRef(null)
  
  // 会话切换时清空已显示的时间节点（中文注释）
  useEffect(() => {
    displayedTimesRef.current.clear()
  }, [session?.id])
  
  // 从所有信息元中提取时间点（中文注释）
  const timelineData = useMemo(() => {
    const timeMap = new Map() // key: record_time, value: { time, count, infons: [], hasExpiring }
    
    for (const run of runs) {
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      const isExpiring = run.expiring === true // 检查run是否即将过期
      
      infons.forEach((infon) => {
        const recordTime = infon.record_time
        if (!recordTime) return // 跳过没有 record_time 的信息元
        
        const timeKey = String(recordTime)
        if (!timeMap.has(timeKey)) {
          timeMap.set(timeKey, {
            time: recordTime,
            count: 0,
            infons: [],
            hasExpiring: false
          })
        }
        
        const timeData = timeMap.get(timeKey)
        timeData.count += 1
        timeData.infons.push(infon)
        // 如果这个时间点包含任何即将过期的信息元，标记为hasExpiring
        if (isExpiring) {
          timeData.hasExpiring = true
        }
      })
    }
    
    // 转换为数组并按时间排序（中文注释）
    const timeArray = Array.from(timeMap.values())
    
    // 尝试将时间字符串转换为时间戳进行排序（中文注释）
    timeArray.sort((a, b) => {
      const timeA = new Date(a.time).getTime()
      const timeB = new Date(b.time).getTime()
      if (!isNaN(timeA) && !isNaN(timeB)) {
        return timeA - timeB
      }
      // 如果无法转换为日期，则按字符串排序
      return String(a.time).localeCompare(String(b.time))
    })
    
    // 标记新节点（中文注释）：未在已显示集合中的节点
    const result = timeArray.map(item => {
      const timeKey = String(item.time)
      const isNew = !displayedTimesRef.current.has(timeKey)
      return { ...item, isNew }
    })
    
    return result
  }, [runs])
  
  // 更新已显示的时间节点集合（中文注释）
  useEffect(() => {
    timelineData.forEach(item => {
      const timeKey = String(item.time)
      displayedTimesRef.current.add(timeKey)
    })
  }, [timelineData])
  
  // 刷新重进或数据更新时自动滚动到最右边（中文注释）
  useEffect(() => {
    const container = containerRef.current
    if (!container || timelineData.length === 0) return
    // 延迟滚动，确保 DOM 渲染完成
    const timer = setTimeout(() => {
      container.scrollLeft = container.scrollWidth
    }, 100)
    return () => clearTimeout(timer)
  }, [session?.id, timelineData.length])
  
  // 处理时间点点击（中文注释）
  const handleTimeClick = (timeData) => {
    if (selectedTime === timeData.time) {
      // 再次点击同一个节点，取消选择（恢复全显）
      setSelectedTime(null)
      onTimeSelect?.(null)
    } else {
      setSelectedTime(timeData.time)
      onTimeSelect?.(timeData.time)
    }
  }
  
  // 处理空白区域点击（中文注释）
  const handleBackgroundClick = (e) => {
    // 仅当点击的是背景本身（不是节点）时才触发
    if (e.target === e.currentTarget) {
      setSelectedTime(null)
      onTimeSelect?.(null)
    }
  }
  
  // 拖拽事件处理（中文注释）
  const handleMouseDown = (e) => {
    if (!containerRef.current) return
    setIsDragging(true)
    setStartX(e.pageX - containerRef.current.offsetLeft)
    setScrollLeft(containerRef.current.scrollLeft)
  }

  const handleMouseMove = (e) => {
    if (!isDragging || !containerRef.current) return
    e.preventDefault()
    const x = e.pageX - containerRef.current.offsetLeft
    const walk = (x - startX) * 2 // 拖拽速度倍数
    containerRef.current.scrollLeft = scrollLeft - walk
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleMouseLeave = () => {
    setIsDragging(false)
  }
  
  // 格式化时间显示（中文注释）：返回 { date, time } 对象
  const formatTime = (time) => {
    const dateObj = new Date(time)
    if (!isNaN(dateObj.getTime())) {
      // 是有效的日期，分别格式化日期和时间
      const date = dateObj.toLocaleDateString('zh-CN', { 
        month: 'short', 
        day: 'numeric'
      })
      const timeStr = dateObj.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit'
      })
      return { date, time: timeStr }
    }
    // 不是有效日期，直接显示原始字符串
    const str = String(time)
    const truncated = str.length > 15 ? str.substring(0, 15) + '...' : str
    return { date: truncated, time: '' }
  }
  
  if (timelineData.length === 0) {
    return (
      <div className={styles.timelineWrapper}>
        <div className={styles.timelineTitle}>Timeline</div>
        <div className={styles.timelineRoot}>
          <div className={styles.timelineEmpty}>No timeline data</div>
        </div>
      </div>
    )
  }
  
  return (
    <div className={styles.timelineWrapper}>
      <div className={styles.timelineTitle}>Timeline</div>
      <div 
        className={styles.timelineRoot} 
        onClick={handleBackgroundClick}
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        <div className={styles.timelineContainer}>
          <div className={styles.timelineLine} />
          <div className={styles.timelineNodes}>
            {timelineData.map((timeData) => {
              const isSelected = selectedTime === timeData.time
              const isNew = timeData.isNew
              const isExpiring = timeData.hasExpiring
              const { date, time } = formatTime(timeData.time)
              return (
                <div
                  key={String(timeData.time)}
                  className={`${styles.timelineNode} ${isSelected ? styles.timelineNodeActive : ''} ${isNew ? styles.timelineNodeNew : ''} ${isExpiring ? styles.timelineNodeExpiring : ''}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    if (!isDragging) {
                      handleTimeClick(timeData)
                    }
                  }}
                  title={`${timeData.time} (${timeData.count} infons)${isExpiring ? ' - 即将过期' : ''}`}
                  style={isExpiring ? { opacity: 0.4, filter: 'grayscale(80%)' } : undefined}
                >
                  <div className={styles.timelineDot}>
                    <div className={styles.timelineDotCount}>{timeData.count}</div>
                  </div>
                  <div className={styles.timelineLabel}>
                    <div className={styles.timelineLabelDate}>{date}</div>
                    {time && <div className={styles.timelineLabelTime}>{time}</div>}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

