import React, { useState } from 'react'
import { Button, Tooltip } from 'antd'
import { CheckCircleOutlined, ThunderboltOutlined, SafetyOutlined } from '@ant-design/icons'
import styles from './AgentPage.module.css'

/**
 * 隐私保护修改建议组件
 * 在Privacy Risk Analysis下方显示，提供不同级别的隐私保护修改建议
 * 
 * @param {Object} suggestions - 建议数据对象 { status, suggestions, error }
 * @param {Function} onApplySuggestion - 应用建议的回调函数
 * @param {Function} onGenerateSuggestions - 生成建议的回调函数
 * @param {boolean} hasInference - 是否已完成隐私推理
 * @param {boolean} hasEditingText - 是否有正在编辑的文本
 */
export default function PrivacyProtectionSuggestions({ 
  suggestions,
  onApplySuggestion,
  onGenerateSuggestions,
  hasInference,
  hasEditingText
}) {
  const [expandedLevel, setExpandedLevel] = useState(null)

  // 获取级别图标和颜色
  const getLevelConfig = (level) => {
    switch (level) {
      case 'high_privacy':
        return {
          icon: <SafetyOutlined />,
          color: '#10b981',
          bgColor: 'rgba(16, 185, 129, 0.1)',
          borderColor: '#10b981'
        }
      case 'balanced':
        return {
          icon: <CheckCircleOutlined />,
          color: '#f59e0b',
          bgColor: 'rgba(245, 158, 11, 0.1)',
          borderColor: '#f59e0b'
        }
      case 'low_privacy':
        return {
          icon: <ThunderboltOutlined />,
          color: '#3b82f6',
          bgColor: 'rgba(59, 130, 246, 0.1)',
          borderColor: '#3b82f6'
        }
      default:
        return {
          icon: null,
          color: '#94a3b8',
          bgColor: 'rgba(148, 163, 184, 0.1)',
          borderColor: '#94a3b8'
        }
    }
  }

  // 如果没有完成隐私推理，显示提示
  if (!hasInference) {
    return (
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)', marginBottom: 8 }}>
          Privacy Protection Suggestions
        </div>
        <div className={styles.wordCloudRoot}>
          <div className={styles.infonEmpty} style={{ padding: 12, textAlign: 'center', fontSize: 11 }}>
            Please complete Privacy Risk Analysis first
          </div>
        </div>
      </div>
    )
  }

  // 如果没有正在编辑的文本，显示提示
  if (!hasEditingText) {
    return (
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)', marginBottom: 8 }}>
          Privacy Protection Suggestions
        </div>
        <div className={styles.wordCloudRoot}>
          <div className={styles.infonEmpty} style={{ padding: 12, textAlign: 'center', fontSize: 11 }}>
            Start typing or editing a message to get suggestions
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)' }}>
          Privacy Protection Suggestions
        </span>
        {suggestions?.status === 'running' && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className={styles.analyzingDot}></span>
            <span style={{ fontSize: 11, color: 'var(--color-accent-primary)', fontWeight: 500 }}>
              Generating...
            </span>
          </span>
        )}
      </div>

      <div className={styles.wordCloudRoot}>
        {/* 生成按钮 */}
        {(!suggestions || suggestions.status === 'idle' || suggestions.status === 'error') && (
          <div style={{ padding: 12, textAlign: 'center' }}>
            <Button 
              type="primary" 
              size="small"
              onClick={onGenerateSuggestions}
              loading={suggestions?.status === 'running'}
            >
              生成隐私保护建议
            </Button>
            {suggestions?.status === 'error' && (
              <div style={{ marginTop: 8, fontSize: 11, color: '#ef4444' }}>
                生成失败: {suggestions.error}
              </div>
            )}
          </div>
        )}

        {/* 建议列表（支持流式渲染） */}
        {(suggestions?.status === 'running' || suggestions?.status === 'done') && suggestions.suggestions && suggestions.suggestions.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {suggestions.suggestions.map((suggestion, idx) => {
              const config = getLevelConfig(suggestion.level)
              const isExpanded = expandedLevel === suggestion.level
              
              // 支持流式渲染：检查建议是否完整
              const isPartial = suggestion._isComplete === false
              const level = suggestion.level || 'unknown'
              const label = suggestion.label || (isPartial ? 'Generating...' : 'Unknown Level')
              const privacyScore = suggestion.privacy_score || (isPartial ? '...' : 'N/A')
              const utilityScore = suggestion.utility_score || (isPartial ? '...' : 'N/A')
              const changesSummary = suggestion.changes_summary || (isPartial ? 'Analyzing modifications...' : '')
              const modifiedText = suggestion.modified_text || (isPartial ? 'Generating protected text...' : '')
              const uniqueKey = suggestion._objIndex ?? idx
              
              return (
                <div 
                  key={uniqueKey}
                  style={{
                    padding: 12,
                    borderRadius: 8,
                    background: config.bgColor,
                    border: `1px solid ${config.borderColor}`,
                    opacity: isPartial ? 0.85 : 1,
                    transition: 'all 0.3s ease'
                  }}
                >
                  {/* 标题栏 */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ color: config.color, fontSize: 14 }}>
                        {config.icon}
                      </span>
                      <span style={{ 
                        fontWeight: 600, 
                        fontSize: 12, 
                        color: 'var(--color-text-primary)',
                        fontStyle: isPartial && !suggestion.label ? 'italic' : 'normal'
                      }}>
                        {label}
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
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {/* 评分标签 */}
                      <Tooltip title="隐私保护程度">
                        <span style={{
                          fontSize: 10,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: '#10b981',
                          color: '#fff',
                          fontWeight: 600,
                          opacity: isPartial && !suggestion.privacy_score ? 0.6 : 1
                        }}>
                          🔒 {privacyScore}
                        </span>
                      </Tooltip>
                      <Tooltip title="模型效用">
                        <span style={{
                          fontSize: 10,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: '#3b82f6',
                          color: '#fff',
                          fontWeight: 600,
                          opacity: isPartial && !suggestion.utility_score ? 0.6 : 1
                        }}>
                          ⚡ {utilityScore}
                        </span>
                      </Tooltip>
                    </div>
                  </div>

                  {/* 修改说明 */}
                  {changesSummary && (
                    <div style={{ 
                      fontSize: 11, 
                      color: 'var(--color-text-secondary)', 
                      marginBottom: 8,
                      lineHeight: 1.5,
                      fontStyle: isPartial && !suggestion.changes_summary ? 'italic' : 'normal'
                    }}>
                      {changesSummary}
                    </div>
                  )}

                  {/* 展开/收起按钮 */}
                  <div style={{ display: 'flex', gap: 8 }}>
                    <Button 
                      size="small"
                      onClick={() => setExpandedLevel(isExpanded ? null : suggestion.level)}
                      style={{ fontSize: 11 }}
                      disabled={isPartial}
                    >
                      {isExpanded ? '收起预览' : '预览修改'}
                    </Button>
                    <Button 
                      size="small"
                      type="primary"
                      onClick={() => onApplySuggestion(suggestion)}
                      style={{ fontSize: 11 }}
                      disabled={isPartial}
                    >
                      一键应用
                    </Button>
                  </div>

                  {/* 展开的修改预览 */}
                  {isExpanded && (
                    <div style={{ 
                      marginTop: 8, 
                      padding: 8, 
                      background: 'var(--color-bg-secondary)', 
                      borderRadius: 6,
                      fontSize: 11,
                      color: 'var(--color-text-primary)',
                      maxHeight: 200,
                      overflow: 'auto',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      fontStyle: isPartial && !suggestion.modified_text ? 'italic' : 'normal'
                    }}>
                      {modifiedText}
                    </div>
                  )}

                  {/* 移除的风险列表 */}
                  {suggestion.removed_risks && suggestion.removed_risks.length > 0 && (
                    <div style={{ 
                      marginTop: 8, 
                      fontSize: 10, 
                      color: 'var(--color-text-tertiary)',
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: 4
                    }}>
                      <span>移除风险:</span>
                      {suggestion.removed_risks.map((risk, riskIdx) => (
                        <span 
                          key={riskIdx}
                          style={{
                            padding: '2px 6px',
                            borderRadius: 4,
                            background: 'var(--color-bg-tertiary)',
                            border: '1px solid var(--color-border-secondary)'
                          }}
                        >
                          {risk}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {/* 正在生成但还没有结果 */}
        {suggestions?.status === 'running' && (!suggestions.suggestions || suggestions.suggestions.length === 0) && (
          <div style={{ padding: 20, fontSize: 11, color: 'var(--color-text-tertiary)', textAlign: 'center' }}>
            <div style={{ marginBottom: 8, fontWeight: 600, color: 'var(--color-accent-primary)' }}>
              正在生成隐私保护建议...
            </div>
            <div style={{ fontSize: 10 }}>
              正在分析您的文本并生成不同级别的保护方案
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

