import React, { useMemo, useState, useRef, useEffect } from 'react'
import { useStore } from '../store'
import styles from './Timeline.module.css'

// 时间线组件（中文注释）：紧凑横向时间线，直接显示时间标签
export default function Timeline({ onTimeSelect }) {
  const { getCurrentSession, infonSessions } = useStore()
  const session = getCurrentSession()
  const runs = useMemo(() => (session ? (infonSessions?.[session.id]?.runs || []) : []), [session, infonSessions])
  
  const [selectedTime, setSelectedTime] = useState(null)
  const displayedTimesRef = useRef(new Set())
  const containerRef = useRef(null)
  
  // 会话切换时清空（中文注释）
  useEffect(() => {
    displayedTimesRef.current.clear()
    setSelectedTime(null)
  }, [session?.id])
  
  // 提取时间点数据（中文注释）
  const timelineData = useMemo(() => {
    const timeMap = new Map()
    
    for (const run of runs) {
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      const isExpiring = run.expiring === true
      
      infons.forEach((infon) => {
        const recordTime = infon.record_time
        if (!recordTime) return
        
        const timeKey = String(recordTime)
        if (!timeMap.has(timeKey)) {
          timeMap.set(timeKey, { time: recordTime, count: 0, infons: [], hasExpiring: false })
        }
        
        const timeData = timeMap.get(timeKey)
        timeData.count += 1
        timeData.infons.push(infon)
        if (isExpiring) timeData.hasExpiring = true
      })
    }
    
    const timeArray = Array.from(timeMap.values())
    timeArray.sort((a, b) => {
      const timeA = new Date(a.time).getTime()
      const timeB = new Date(b.time).getTime()
      if (!isNaN(timeA) && !isNaN(timeB)) return timeA - timeB
      return String(a.time).localeCompare(String(b.time))
    })
    
    return timeArray.map(item => {
      const timeKey = String(item.time)
      const isNew = !displayedTimesRef.current.has(timeKey)
      return { ...item, isNew }
    })
  }, [runs])
  
  // 更新已显示集合（中文注释）
  useEffect(() => {
    timelineData.forEach(item => {
      displayedTimesRef.current.add(String(item.time))
    })
  }, [timelineData])
  
  // 自动滚动到最右（中文注释）
  useEffect(() => {
    const container = containerRef.current
    if (!container || timelineData.length === 0) return
    const timer = setTimeout(() => {
      container.scrollLeft = container.scrollWidth
    }, 100)
    return () => clearTimeout(timer)
  }, [session?.id, timelineData.length])
  
  // 点击处理（中文注释）
  const handleTimeClick = (timeData) => {
    if (selectedTime === timeData.time) {
      setSelectedTime(null)
      onTimeSelect?.(null)
    } else {
      setSelectedTime(timeData.time)
      onTimeSelect?.(timeData.time)
    }
  }
  
  // 格式化时间显示（中文注释）
  const formatTime = (time) => {
    const dateObj = new Date(time)
    if (!isNaN(dateObj.getTime())) {
      const month = dateObj.getMonth() + 1
      const day = dateObj.getDate()
      const hour = dateObj.getHours().toString().padStart(2, '0')
      const min = dateObj.getMinutes().toString().padStart(2, '0')
      return { date: `${month}/${day}`, time: `${hour}:${min}` }
    }
    const str = String(time)
    const truncated = str.length > 10 ? str.substring(0, 10) + '…' : str
    return { date: truncated, time: '' }
  }
  
  if (timelineData.length === 0) {
    return (
      <div className={styles.timelineWrapper}>
        <div className={styles.timelineHeader}>
          <span className={styles.timelineTitle}>Timeline</span>
        </div>
        <div className={styles.timelineEmpty}>No timeline data</div>
      </div>
    )
  }
  
  return (
    <div className={styles.timelineWrapper}>
      <div className={styles.timelineHeader}>
        <span className={styles.timelineTitle}>Timeline</span>
        <span className={styles.timelineCount}>{timelineData.length} points</span>
      </div>
      <div className={styles.timelineRoot} ref={containerRef}>
        <div className={styles.timelineLine} />
        <div className={styles.timelineNodes}>
          {timelineData.map((timeData) => {
            const isSelected = selectedTime === timeData.time
            const isExpiring = timeData.hasExpiring
            const { date, time } = formatTime(timeData.time)
            
            return (
              <div
                key={String(timeData.time)}
                className={`${styles.timelineNode} ${isSelected ? styles.active : ''} ${timeData.isNew ? styles.new : ''}`}
                onClick={() => handleTimeClick(timeData)}
                style={isExpiring ? { opacity: 0.4 } : undefined}
              >
                <div className={`${styles.dot} ${isSelected ? styles.dotActive : ''}`}>
                  {timeData.count}
                </div>
                <div className={styles.label}>
                  <span className={styles.labelDate}>{date}</span>
                  {time && <span className={styles.labelTime}>{time}</span>}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
