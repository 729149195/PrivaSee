import React, { useMemo, useState, useEffect, useRef, memo } from 'react'
import styles from './AgentPage.module.css'
import { useStore } from '../store'

/**
 * 打字机效果 Hook - 分块显示文本，优化流式性能
 * 流式期间直接显示文本（因为文本本身已经在流式到达）；
 * 完成后如果之前未完全显示，则快速追赶到完整文本。
 */
function useTypewriter(text, isComplete, speed = 15) {
  const [displayedText, setDisplayedText] = useState('')
  const rafRef = useRef(null)
  const prevCompleteRef = useRef(false)
  
  useEffect(() => {
    const currentText = text || ''
    
    // 流式期间（isComplete=false）：直接显示当前文本，无需逐字动画
    // 因为文本本身已经在从 API 逐步到达
    if (!isComplete) {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      if (displayedText !== currentText) {
        setDisplayedText(currentText)
      }
      prevCompleteRef.current = false
      return
    }
    
    // 刚完成时（从 streaming → complete 转换）：
    // 如果文本差距太大，用快速动画追赶；否则直接显示
    if (isComplete && !prevCompleteRef.current) {
      prevCompleteRef.current = true
      const remaining = currentText.length - displayedText.length
      if (remaining > 50) {
        // 大量剩余文本，快速分块追赶
        let pos = displayedText.length
        const CHUNK = Math.max(5, Math.ceil(remaining / 20)) // 约 20 帧内完成
        const animate = () => {
          pos = Math.min(pos + CHUNK, currentText.length)
          setDisplayedText(currentText.slice(0, pos))
          if (pos < currentText.length) {
            rafRef.current = requestAnimationFrame(animate)
          } else {
            rafRef.current = null
          }
        }
        rafRef.current = requestAnimationFrame(animate)
        return () => {
          if (rafRef.current) {
            cancelAnimationFrame(rafRef.current)
            rafRef.current = null
          }
        }
      } else {
        // 差距不大，直接设置
        if (displayedText !== currentText) {
          setDisplayedText(currentText)
        }
        return
      }
    }
    
    // 已完成且之前也是完成状态（文本没变化），直接同步
    if (displayedText !== currentText) {
      setDisplayedText(currentText)
    }
  }, [text, isComplete, displayedText, speed])
  
  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [])
  
  return displayedText
}

/**
 * 将 inference_chain 文本按序号分段
 * 例如："1) xxx 2) yyy" => ["1) xxx", "2) yyy"]
 */
function parseInferenceChain(text) {
  if (!text || typeof text !== 'string') return []
  
  // 匹配序号模式，如 "1)", "2)", "10)" 等
  const pattern = /(\d+\))/g
  const segments = []
  let lastIndex = 0
  let match
  
  while ((match = pattern.exec(text)) !== null) {
    // 如果不是第一个匹配，先保存前一段
    if (match.index > 0 && lastIndex < match.index) {
      const prevSegment = text.substring(lastIndex, match.index).trim()
      if (prevSegment) {
        segments.push(prevSegment)
      }
    }
    lastIndex = match.index
  }
  
  // 添加最后一段
  if (lastIndex < text.length) {
    const finalSegment = text.substring(lastIndex).trim()
    if (finalSegment) {
      segments.push(finalSegment)
    }
  }
  
  // 如果没有找到序号，返回原文本
  if (segments.length === 0 && text.trim()) {
    segments.push(text.trim())
  }
  
  return segments
}

/**
 * 单个风险卡片组件 - 带打字机效果（使用 React.memo 优化）
 */
const RiskCard = memo(function RiskCard({ risk, idx, infonMap }) {
  const isPartial = risk._isComplete === false
  const isComplete = risk._isComplete !== false
  const uniqueKey = risk._objIndex ?? idx
  
  const riskLevel = risk.risk_level || 'UNKNOWN'
  const lawNodeName = risk.law_node_name || 'Loading...'
  const usedInfons = risk.used_infons || []
  
  // 使用打字机效果显示文本字段
  const privacyExposure = useTypewriter(
    risk.privacy_exposure || '',
    isComplete,
    15
  )
  const inferenceChain = useTypewriter(
    risk.inference_chain || '',
    isComplete,
    15
  )
  
  // 解析 inference_chain 为分段数组
  const inferenceChainSegments = useMemo(
    () => parseInferenceChain(inferenceChain),
    [inferenceChain]
  )
  
  // 获取关联的信息元（去重 iid，避免 React key 冲突）
  const usedIids = useMemo(() => {
    const raw = Array.isArray(usedInfons) ? usedInfons.map(x => (typeof x === 'string' ? x : x?.iid)).filter(Boolean) : []
    return [...new Set(raw)]
  }, [usedInfons])
  const relatedInfons = useMemo(
    () => usedIids.map(iid => infonMap.get(iid)).filter(Boolean),
    [usedIids, infonMap]
  )
  
  // 获取边框颜色
  const borderColor = riskLevel === 'HIGH' ? '#ef4444' : riskLevel === 'MEDIUM' ? '#f59e0b' : riskLevel === 'LOW' ? '#10b981' : '#94a3b8'
  
  return (
    <div 
      key={uniqueKey}
      className={styles.riskItem}
      style={{ 
        flex: '1 1 180px',
        minWidth: 160,
        maxWidth: 280,
        padding: '8px 10px', 
        borderRadius: 6, 
        background: 'var(--color-bg-tertiary)',
        border: '1px solid var(--color-border-light)',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        opacity: isPartial ? 0.85 : 1,
        transition: 'opacity 0.3s ease'
      }}
    >
      {/* 标题行：风险等级 + 法律类别名称 */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginBottom: 4 }}>
        <span style={{ 
          fontSize: 9, 
          fontWeight: 700, 
          padding: '1px 5px', 
          borderRadius: 3,
          background: borderColor,
          color: '#fff',
          flexShrink: 0,
          lineHeight: '14px'
        }}>
          {riskLevel}
        </span>
        <span style={{ 
          fontSize: 11, 
          fontWeight: 600, 
          color: 'var(--color-text-secondary)',
          lineHeight: '14px',
          wordBreak: 'break-word'
        }}>
          {lawNodeName}
        </span>
        {isPartial && (
          <span className={styles.analyzingDot} style={{ width: 4, height: 4, flexShrink: 0, marginTop: 5 }}></span>
        )}
      </div>
      
      {/* 中文说明 */}
      {(privacyExposure || (isPartial && !risk.privacy_exposure)) && (
        <div style={{ 
          fontSize: 10, 
          color: 'var(--color-text-tertiary)', 
          marginBottom: 4,
          fontStyle: isPartial && !risk.privacy_exposure ? 'italic' : 'normal',
          lineHeight: 1.3
        }}>
          {privacyExposure || 'Analyzing...'}
        </div>
      )}
      
      {/* 推理链（更紧凑） */}
      {(inferenceChain || (isPartial && !risk.inference_chain)) && (
        <div style={{ 
          fontSize: 9, 
          color: 'var(--color-text-tertiary)', 
          marginBottom: 4,
          fontStyle: isPartial && !risk.inference_chain ? 'italic' : 'normal',
          lineHeight: 1.3
        }}>
          {inferenceChain ? (
            inferenceChainSegments.length > 0 ? (
              inferenceChainSegments.map((segment, segIdx) => (
                <div key={segIdx} style={{ marginBottom: segIdx < inferenceChainSegments.length - 1 ? 2 : 0 }}>
                  {segment}
                </div>
              ))
            ) : (
              inferenceChain
            )
          ) : (
            'Streaming...'
          )}
        </div>
      )}
      
      {/* 相关信息元列表 - 更紧凑 */}
      {relatedInfons.length > 0 && (
        <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px dashed var(--color-border-light)' }}>
          <div style={{ fontSize: 9, color: 'var(--color-text-tertiary)', marginBottom: 3 }}>
            Related Infons ({relatedInfons.length})
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {relatedInfons.map((infon, infonIdx) => {
              const keyword = getInfonKeyword(infon)
              const color = getInfonColor(infon.infon_type)
              const infonType = String(infon.infon_type || '').toUpperCase()
              const isRelation = infonType === 'REL'
              
              // 过滤掉 SIT 类型
              if (infonType === 'SIT') {
                return null
              }
              
              const hasAssociations = Array.isArray(infon.associations) && infon.associations.length > 0
              const hasEvidencePointer = !!infon.evidence_pointer
              
              return (
                <div
                  key={infon.iid || infonIdx}
                  title={hasAssociations 
                    ? `${keyword}\n\nAssociations (${infon.associations.length}):\n${infon.associations.map(a => `  ${a.iid} (sim: ${a.similarity})`).join('\n')}${hasEvidencePointer ? `\n\nEvidence: ${infon.evidence_pointer}` : ''}`
                    : (hasEvidencePointer ? `${keyword}\nEvidence: ${infon.evidence_pointer}` : keyword)
                  }
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 2,
                    padding: '2px 5px',
                    borderRadius: isRelation ? 6 : 3,
                    background: isRelation ? 'rgba(255, 255, 255, 0.95)' : `${color}15`,
                    border: `1px solid ${color}`,
                    borderStyle: isRelation ? 'dashed' : 'solid',
                    fontSize: 8,
                    fontWeight: 600,
                    color: color,
                    maxWidth: '100%',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    cursor: hasAssociations ? 'help' : 'default',
                  }}
                >
                  {keyword}
                  {hasAssociations && (
                    <span style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      background: `${color}30`,
                      fontSize: 7,
                      fontWeight: 700,
                      lineHeight: 1,
                      flexShrink: 0,
                    }}>
                      {infon.associations.length}
                    </span>
                  )}
                  {hasEvidencePointer && (
                    <span style={{ fontSize: 7, opacity: 0.6, flexShrink: 0 }}>
                      *
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
})

// 信息元类型颜色映射（中文注释）
const getInfonColor = (infonType) => {
  const colors = {
    DESC: '#3b82f6',  // 描述（实体+属性）：蓝色
    SCEN: '#10b981',  // 场景（时间+位置）：翠绿色
    REL: '#8b5cf6',   // 关系：紫色
    SIT: '#f59e0b',   // 情景：琥珀色
  }
  return colors[String(infonType).toUpperCase()] || '#64748b'
}

// 提取信息元关键词（中文注释）
const getInfonKeyword = (infon) => {
  if (!infon || typeof infon !== 'object') return 'Unknown'
  const t = String(infon.infon_type || '').toUpperCase()
  if (t === 'DESC') {
    const attribute = infon.attribute ?? ''
    const entity = infon.entity ?? ''
    return attribute || entity || 'Description'
  }
  if (t === 'SCEN') {
    const temporal = infon.temporal ?? ''
    const spatial = infon.spatial ?? ''
    return temporal || spatial || 'Scenario'
  }
  if (t === 'REL') return String(infon.relation_name ?? 'Relation')
  if (t === 'SIT') return String(infon.description ?? 'Situation')
  return t || 'Unknown'
}

// Privacy Risk Analysis组件：用于显示隐私风险分析结果（自动推断版本）
export default function PrivacyRiskAnalysis({ 
  inference, 
  selectedLaw
}) {
  const { getCurrentSession, infonSessions } = useStore()
  const session = getCurrentSession()
  
  // 获取所有信息元映射（中文注释）
  const infonMap = useMemo(() => {
    const runs = session ? (infonSessions?.[session.id]?.runs || []) : []
    const map = new Map()
    
    for (const run of runs) {
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      
      infons.forEach((infon) => {
        if (infon.iid) {
          map.set(infon.iid, infon)
        }
      })
    }
    
    return map
  }, [session, infonSessions])
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)' }}>
            Privacy Risk Analysis
          </span>
          {inference?.status === 'running' && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className={styles.analyzingDot}></span>
              <span style={{ fontSize: 11, color: 'var(--color-accent-primary)', fontWeight: 500 }}>
                Analyzing...
              </span>
            </span>
          )}
        </div>
        {!selectedLaw && (
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
            (Please select a law in the Law Tree first)
          </div>
        )}
      </div>
      <div className={styles.wordCloudRoot}>
      
      {inference && inference.risks && inference.risks.length > 0 ? (() => {
        // 过滤掉无效的风险项：
        // 1. reason 包含"未提及"或"not mentioned"
        // 2. risk_level 无效（空或不是 HIGH/MEDIUM/LOW）
        // 3. 没有关联的 infon
        const validRisks = inference.risks.filter(risk => {
          const reason = (risk.inference_chain || risk.reason || '').toLowerCase()
          const level = (risk.risk_level || '').toUpperCase()
          const hasValidLevel = ['HIGH', 'MEDIUM', 'LOW'].includes(level)
          const hasInvalidReason = reason.includes('未提及') || reason.includes('not mentioned') || reason.includes('无相关')
          const hasInfons = Array.isArray(risk.used_infons) && risk.used_infons.length > 0
          
          return hasValidLevel && !hasInvalidReason && hasInfons
        })
        
        if (validRisks.length === 0) {
          return (
            <div className={styles.infonEmpty} style={{ padding: 12, textAlign: 'center' }}>
              No matching privacy risks found
            </div>
          )
        }
        
        return (
          <details className={styles.wordCloudDetails} open={inference.status === 'running' || inference.status === 'done'}>
            <summary className={styles.wordCloudDetailsSummary}>
              Inference Results ({validRisks.length} risk{validRisks.length > 1 ? 's' : ''})
              {inference.status === 'running' && (
                <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--color-accent-primary)' }}>
                  (streaming...)
                </span>
              )}
            </summary>
            <div className={styles.wordCloudDetailsContent} style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', 
              gap: 6 
            }}>
              {validRisks.map((risk, idx) => (
                <RiskCard
                  key={risk._objIndex ?? idx}
                  risk={risk}
                  idx={idx}
                  infonMap={infonMap}
                />
              ))}
            </div>
          </details>
        )
      })() : inference?.status === 'running' ? (
        <div style={{ padding: 20, fontSize: 11, color: 'var(--color-text-tertiary)', textAlign: 'center' }}>
          <div style={{ marginBottom: 8, fontWeight: 600, color: 'var(--color-accent-primary)' }}>
            Analyzing privacy risks...
          </div>
          <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>
            This may take a few seconds to start streaming results
          </div>
        </div>
      ) : inference?.status === 'error' ? (
        <div style={{ padding: 12, fontSize: 11, color: '#ef4444', textAlign: 'center' }}>
          Inference error: {inference.error}
        </div>
      ) : (
        <div className={styles.infonEmpty} style={{ padding: 12, textAlign: 'center' }}>
          No inference yet
        </div>
      )}
      </div>
    </div>
  )
}

