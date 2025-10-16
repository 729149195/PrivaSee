import React from 'react'
import { Spin } from 'antd'
import styles from '../AgentPage.module.css'

/**
 * 信息元提取任务结果卡片组件
 * @param {object} run - 信息元提取任务对象
 */
const InfonRunCard = ({ run }) => {
  const title = run.modality === 'text' ? 'Text' : `Image${Number.isFinite(run.imageIndex) ? ` #${run.imageIndex + 1}` : ''}`
  const status = run.status
  const allInfons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
  const isExpiring = run.expiring === true // 检查是否即将过期

  return (
    <div 
      className={styles.infonRunCard}
      style={isExpiring ? { 
        opacity: 0.4, 
        filter: 'grayscale(80%)',
        position: 'relative'
      } : undefined}
      title={isExpiring ? '即将过期' : undefined}
    >
      <div className={styles.infonRunHeader}>
        <div className={styles.infonRunTitle}>
          {title}
          <span style={{ marginLeft: '8px', fontSize: '12px' }}>
            {status === 'running' && <Spin size="small" />}
            {status === 'done' && <span style={{ color: '#52c41a' }}>✓</span>}
            {status === 'error' && <span style={{ color: '#ff4d4f' }}>✕</span>}
            {status === 'aborted' && <span style={{ color: '#faad14' }}>⏸</span>}
          </span>
        </div>
        <div className={styles.infonRunMeta}>{run.targetType}</div>
      </div>
      {status === 'error' && run.error ? (
        <div className={styles.infonError}>{run.error}</div>
      ) : null}
      {allInfons.length > 0 ? (
        <details className={styles.infonDetails}>
          <summary className={styles.infonDetailsSummary}>Infons ({allInfons.length})</summary>
          <div className={styles.infonJsonList}>
            {allInfons.map((infon, idx) => (
              <div key={idx} className={styles.infonItem}>
                <div className={styles.infonType}>{infon.infon_type || 'Unknown'}</div>
                <pre className={styles.infonJsonCode}>{JSON.stringify(infon, null, 2)}</pre>
              </div>
            ))}
          </div>
        </details>
      ) : null}
      <details className={styles.infonDetails}>
        <summary className={styles.infonDetailsSummary}>Raw stream</summary>
        <pre className={styles.infonJsonCode}>{run.buffer || ''}</pre>
      </details>
    </div>
  )
}

export default InfonRunCard

