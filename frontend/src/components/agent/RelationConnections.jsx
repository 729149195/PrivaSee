import React, { useLayoutEffect, useRef, useState } from 'react'
import styles from '../AgentPage.module.css'

/**
 * 连线组件：根据关系信息元画连线连接标签和高亮文本
 * @param {string} messageId - 消息 ID
 * @param {Array} relations - 关系信息元数组
 * @param {object} infonIndex - 信息元索引
 */
const RelationConnections = ({ messageId, relations, infonIndex }) => {
  const [connections, setConnections] = useState([])
  const containerRef = useRef(null)

  useLayoutEffect(() => {
    if (!containerRef.current || !relations.length) return

    const container = containerRef.current.parentElement
    if (!container) return

    const newConnections = []
    const containerRect = container.getBoundingClientRect()

    relations.forEach(({ infon }, relIdx) => {
      const relatedInfons = infon.arg_refs || []
      
      // 找到关系标签的位置
      const tagSelector = `.${styles.relationTag}`
      const allTags = container.querySelectorAll(tagSelector)
      const tagEl = allTags[relIdx]
      if (!tagEl) return

      const tagRect = tagEl.getBoundingClientRect()
      const tagX = tagRect.left - containerRect.left + tagRect.width / 2
      const tagY = tagRect.bottom - containerRect.top

      relatedInfons.forEach((argRef) => {
        // 查找对应的高亮元素
        const highlightEl = container.querySelector(`[data-infon-id="${argRef}"][data-relation-id="${infon.iid}"]`)
        if (!highlightEl) return

        const highlightRect = highlightEl.getBoundingClientRect()
        const highlightX = highlightRect.left - containerRect.left + highlightRect.width / 2
        const highlightY = highlightRect.top - containerRect.top

        // 计算贝塞尔曲线控制点
        const dx = highlightX - tagX
        const dy = highlightY - tagY
        const controlY = tagY + dy * 0.5

        newConnections.push({
          relationId: infon.iid,
          argRef: argRef,
          startX: tagX,
          startY: tagY,
          endX: highlightX,
          endY: highlightY,
          controlY: controlY,
        })
      })
    })

    setConnections(newConnections)
  }, [relations, infonIndex, messageId])

  if (!connections.length) return null

  return (
    <svg 
      ref={containerRef}
      className={styles.relationConnections} 
      style={{ 
        position: 'absolute', 
        top: 0, 
        left: 0, 
        width: '100%', 
        height: '100%', 
        pointerEvents: 'none',
        zIndex: 1,
        overflow: 'visible'
      }}
    >
      {connections.map((conn, i) => {
        const path = `M ${conn.startX} ${conn.startY} Q ${conn.startX} ${conn.controlY}, ${conn.endX} ${conn.endY}`
        return (
          <g key={i}>
            <path 
              d={path}
              fill="none"
              stroke="rgba(91, 141, 239, 0.3)"
              strokeWidth="1.5"
              strokeDasharray="3,3"
            />
            <circle 
              cx={conn.endX} 
              cy={conn.endY} 
              r={3}
              fill="rgba(91, 141, 239, 0.4)"
              stroke="rgba(91, 141, 239, 0.6)"
              strokeWidth="1"
            />
          </g>
        )
      })}
    </svg>
  )
}

export default RelationConnections

