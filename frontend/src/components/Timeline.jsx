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
  
  // 会话切换时清空已显示的时间节点（中文注释）
  useEffect(() => {
    displayedTimesRef.current.clear()
  }, [session?.id])
  
  // 从所有信息元中提取时间点（中文注释）
  const timelineData = useMemo(() => {
    const timeMap = new Map() // key: record_time, value: { time, count, infons: [] }
    
    for (const run of runs) {
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      
      infons.forEach((infon) => {
        const recordTime = infon.record_time
        if (!recordTime) return // 跳过没有 record_time 的信息元
        
        const timeKey = String(recordTime)
        if (!timeMap.has(timeKey)) {
          timeMap.set(timeKey, {
            time: recordTime,
            count: 0,
            infons: []
          })
        }
        
        const timeData = timeMap.get(timeKey)
        timeData.count += 1
        timeData.infons.push(infon)
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
  
  // 格式化时间显示（中文注释）
  const formatTime = (time) => {
    const date = new Date(time)
    if (!isNaN(date.getTime())) {
      // 是有效的日期，格式化为简洁形式
      return date.toLocaleDateString('zh-CN', { 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    }
    // 不是有效日期，直接显示原始字符串（截断过长的字符串）
    const str = String(time)
    return str.length > 20 ? str.substring(0, 20) + '...' : str
  }
  
  if (timelineData.length === 0) {
    return null // 没有时间数据时不显示
  }
  
  return (
    <div className={styles.timelineRoot} onClick={handleBackgroundClick}>
      <div className={styles.timelineHeader}>
        <div className={styles.timelineTitle}>Timeline</div>
        <div className={styles.timelineHint}>
          {selectedTime ? 'Click the same node or blank area to restore full display' : 'Click the node to filter the information elements at this time'}
        </div>
      </div>
      <div className={styles.timelineContainer}>
        <div className={styles.timelineLine} />
        <div className={styles.timelineNodes}>
          {timelineData.map((timeData) => {
            const isSelected = selectedTime === timeData.time
            const isNew = timeData.isNew
            return (
              <div
                key={String(timeData.time)}
                className={`${styles.timelineNode} ${isSelected ? styles.timelineNodeActive : ''} ${isNew ? styles.timelineNodeNew : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  handleTimeClick(timeData)
                }}
                title={`${timeData.time} (${timeData.count} infons`}
              >
                <div className={styles.timelineDot}>
                  <div className={styles.timelineDotInner} />
                </div>
                <div className={styles.timelineLabel}>
                  <div className={styles.timelineLabelTime}>{formatTime(timeData.time)}</div>
                  <div className={styles.timelineLabelCount}>{timeData.count}</div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

