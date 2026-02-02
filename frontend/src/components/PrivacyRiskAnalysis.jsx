import React, { useMemo, useState, useEffect, useRef } from 'react'
import styles from './AgentPage.module.css'
import { useStore } from '../store'

/**
 * 打字机效果 Hook - 逐字显示文本
 */
function useTypewriter(text, isComplete, speed = 15) {
  const [displayedText, setDisplayedText] = useState('')
  const animationIdRef = useRef(null)
  
  useEffect(() => {
    const currentText = text || ''
    
    // 如果文本已完整，直接显示全部
    if (isComplete) {
      if (displayedText !== currentText) {
        setDisplayedText(currentText)
      }
      return
    }
    
    // 如果文本为空，清空显示
    if (!currentText) {
      if (displayedText !== '') {
        setDisplayedText('')
      }
      return
    }
    
    // 取消之前的动画
    if (animationIdRef.current) {
      clearTimeout(animationIdRef.current)
      animationIdRef.current = null
    }
    
    // 如果显示的文本已经和当前文本一致，不做任何事
    if (displayedText === currentText) {
      return
    }
    
    // 如果显示的文本比当前文本长（回退情况），直接更新
    if (displayedText.length > currentText.length) {
      setDisplayedText(currentText)
      return
    }
    
    // 逐字显示剩余字符
    const remainingText = currentText.slice(displayedText.length)
    if (remainingText.length > 0) {
      animationIdRef.current = setTimeout(() => {
        setDisplayedText(currentText.slice(0, displayedText.length + 1))
      }, speed)
    }
    
    return () => {
      if (animationIdRef.current) {
        clearTimeout(animationIdRef.current)
        animationIdRef.current = null
      }
    }
  }, [text, isComplete, displayedText, speed])
  
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
 * 单个风险卡片组件 - 带打字机效果
 */
function RiskCard({ risk, idx, infonMap }) {
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
  
  // 获取关联的信息元
  const usedIids = Array.isArray(usedInfons) ? usedInfons.map(x => (typeof x === 'string' ? x : x?.iid)).filter(Boolean) : []
  const relatedInfons = usedIids.map(iid => infonMap.get(iid)).filter(Boolean)
  
  return (
    <div 
      key={uniqueKey}
      className={styles.riskItem}
      style={{ 
        flex: '0 0 calc(50% - 3px)',
        padding: 12, 
        borderRadius: 8, 
        background: 'var(--color-bg-tertiary)',
        border: `1px solid ${riskLevel === 'HIGH' ? '#ef4444' : riskLevel === 'MEDIUM' ? '#f59e0b' : riskLevel === 'LOW' ? '#10b981' : '#94a3b8'}`,
        opacity: isPartial ? 0.85 : 1,
        transition: 'opacity 0.3s ease'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
        <span style={{ 
          fontSize: 10, 
          fontWeight: 700, 
          padding: '2px 6px', 
          borderRadius: 4,
          background: riskLevel === 'HIGH' ? '#ef4444' : riskLevel === 'MEDIUM' ? '#f59e0b' : riskLevel === 'LOW' ? '#10b981' : '#94a3b8',
          color: '#fff'
        }}>
          {riskLevel}
        </span>
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-secondary)' }}>
          {lawNodeName}
        </span>
        {isPartial && (
          <span style={{ 
            fontSize: 10, 
            color: 'var(--color-accent-primary)', 
            fontStyle: 'italic',
            display: 'flex',
            alignItems: 'center',
            gap: 4
          }}>
            <span className={styles.analyzingDot} style={{ width: 4, height: 4 }}></span>
            streaming...
          </span>
        )}
      </div>
      {(privacyExposure || (isPartial && !risk.privacy_exposure)) && (
        <div style={{ 
          fontSize: 11, 
          color: 'var(--color-text-primary)', 
          marginBottom: 6,
          fontStyle: isPartial && !risk.privacy_exposure ? 'italic' : 'normal',
          minHeight: '1.5em'
        }}>
          {privacyExposure || 'Analyzing...'}
        </div>
      )}
      {(inferenceChain || (isPartial && !risk.inference_chain)) && (
        <div style={{ 
          fontSize: 10, 
          color: 'var(--color-text-tertiary)', 
          marginBottom: 6,
          fontStyle: isPartial && !risk.inference_chain ? 'italic' : 'normal',
          minHeight: '1.5em'
        }}>
          {inferenceChain ? (
            inferenceChainSegments.length > 0 ? (
              inferenceChainSegments.map((segment, segIdx) => (
                <div key={segIdx} style={{ marginBottom: segIdx < inferenceChainSegments.length - 1 ? 4 : 0 }}>
                  {segment}
                </div>
              ))
            ) : (
              inferenceChain
            )
          ) : (
            'Streaming reasoning...'
          )}
        </div>
      )}
      
      {/* 相关信息元列表 */}
      {relatedInfons.length > 0 && (
        <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px dashed var(--color-border-light)' }}>
          <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', marginBottom: 4, fontWeight: 600 }}>
            Related Infons ({relatedInfons.length})
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {relatedInfons.map((infon, infonIdx) => {
              const keyword = getInfonKeyword(infon)
              const color = getInfonColor(infon.infon_type)
              const infonType = String(infon.infon_type || '').toUpperCase()
              const isRelation = infonType === 'REL'
              
              // 过滤掉 SIT 类型
              if (infonType === 'SIT') {
                return null
              }
              
              return (
                <div
                  key={infon.iid || infonIdx}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    padding: '4px 8px',
                    borderRadius: isRelation ? 8 : 4,
                    background: isRelation ? 'rgba(255, 255, 255, 0.95)' : `${color}26`,
                    border: `1px solid ${color}`,
                    borderStyle: isRelation ? 'dashed' : 'solid',
                    fontSize: 9,
                    fontWeight: isRelation ? 700 : 600,
                    color: color,
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)'
                  }}
                >
                  {keyword}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

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
      
      {inference && inference.risks && inference.risks.length > 0 ? (
        <details className={styles.wordCloudDetails} open={inference.status === 'running' || inference.status === 'done'}>
          <summary className={styles.wordCloudDetailsSummary}>
            Inference Results ({inference.risks.length} risk{inference.risks.length > 1 ? 's' : ''})
            {inference.status === 'running' && (
              <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--color-accent-primary)' }}>
                (streaming...)
              </span>
            )}
          </summary>
          <div className={styles.wordCloudDetailsContent} style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {inference.risks.map((risk, idx) => (
              <RiskCard
                key={risk._objIndex ?? idx}
                risk={risk}
                idx={idx}
                infonMap={infonMap}
              />
            ))}
          </div>
        </details>
      ) : inference?.status === 'running' ? (
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

