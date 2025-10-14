import React from 'react'
import styles from './AgentPage.module.css'
import WordCloud from './WordCloud'
import { Tooltip } from 'antd'

// Privacy Risk Analysis组件：用于显示隐私风险分析结果（自动推断版本）
export default function PrivacyRiskAnalysis({ 
  inference, 
  selectedLaw
}) {
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
            {inference.risks.map((risk, idx) => {
              // 支持部分数据：某些字段可能还未流式输出
              const isPartial = risk._isComplete === false
              const riskLevel = risk.risk_level || 'UNKNOWN'
              const lawNodeName = risk.law_node_name || 'Loading...'
              const privacyExposure = risk.privacy_exposure || (isPartial ? 'Analyzing...' : 'N/A')
              const inferenceChain = risk.inference_chain || (isPartial ? 'Streaming reasoning...' : '')
              const usedInfons = risk.used_infons || []
              const usedIids = Array.isArray(usedInfons) ? usedInfons.map(x => (typeof x === 'string' ? x : x?.iid)).filter(Boolean) : []
              const uniqueKey = risk._objIndex ?? idx
              
              return (
                <Tooltip 
                  key={uniqueKey}
                  title={usedIids.length > 0 ? (
                    <WordCloud selectedTime={null} filterIids={usedIids} compact={true} />
                  ) : null}
                  placement={(idx % 2 === 0) ? 'right' : 'left'}
                  mouseEnterDelay={0.1}
                  classNames={{ root: `${styles.riskTooltipOverlay} ${(idx % 2 === 0) ? styles.riskTooltipRight : styles.riskTooltipLeft}` }}
                  arrow={false}
                >
                <div 
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
                  {privacyExposure && (
                    <div style={{ 
                      fontSize: 11, 
                      color: 'var(--color-text-primary)', 
                      marginBottom: 6,
                      fontStyle: isPartial && !risk.privacy_exposure ? 'italic' : 'normal'
                    }}>
                      {privacyExposure}
                    </div>
                  )}
                  {inferenceChain && (
                    <div style={{ 
                      fontSize: 10, 
                      color: 'var(--color-text-tertiary)', 
                      marginBottom: 4,
                      fontStyle: isPartial && !risk.inference_chain ? 'italic' : 'normal'
                    }}>
                      {inferenceChain}
                    </div>
                  )}
                </div>
                </Tooltip>
              )
            })}
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

