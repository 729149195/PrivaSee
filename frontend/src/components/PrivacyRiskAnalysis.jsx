import React from 'react'
import styles from './AgentPage.module.css'

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
            {inference.risks.map((risk, idx) => (
              <div 
                key={idx} 
                className={styles.riskItem}
                style={{ 
                  flex: '0 0 calc(50% - 3px)',
                  padding: 12, 
                  borderRadius: 8, 
                  background: 'var(--color-bg-tertiary)',
                  border: `1px solid ${risk.risk_level === 'HIGH' ? '#ef4444' : risk.risk_level === 'MEDIUM' ? '#f59e0b' : '#10b981'}`
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
                  <span style={{ 
                    fontSize: 10, 
                    fontWeight: 700, 
                    padding: '2px 6px', 
                    borderRadius: 4,
                    background: risk.risk_level === 'HIGH' ? '#ef4444' : risk.risk_level === 'MEDIUM' ? '#f59e0b' : '#10b981',
                    color: '#fff'
                  }}>
                    {risk.risk_level}
                  </span>
                  <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-text-secondary)' }}>
                    Confidence: {(risk.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: 4 }}>
                  {risk.law_node_name}
                </div>
                <div style={{ fontSize: 11, color: 'var(--color-text-primary)', marginBottom: 6 }}>
                  {risk.privacy_exposure}
                </div>
                <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', marginBottom: 4 }}>
                  {risk.inference_chain}
                </div>
                {risk.used_infons && risk.used_infons.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
                    <span style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>Used Infons:</span>
                    {risk.used_infons.map((infon, i) => (
                      <span 
                        key={i}
                        style={{ 
                          fontSize: 10, 
                          padding: '2px 6px', 
                          borderRadius: 4,
                          background: 'var(--color-bg-secondary)',
                          border: '1px solid var(--color-border-light)',
                          color: 'var(--color-text-secondary)'
                        }}
                      >
                        [{infon.type}] {infon.keyword}
                      </span>
                    ))}
                  </div>
                )}
              </div>
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

