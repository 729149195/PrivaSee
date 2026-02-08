import React, { useState, useEffect, useRef } from 'react'
import { Button } from 'antd'
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
/**
 * 打字机效果 Hook - 逐字显示文本
 * @param {string} text - 要显示的文本
 * @param {boolean} isComplete - 文本是否已完整接收
 * @param {number} speed - 打字速度（毫秒/字符）
 */
function useTypewriter(text, isComplete, speed = 15) {
  const [displayedText, setDisplayedText] = useState('')
  const lastCompleteTextRef = useRef('')
  const animationIdRef = useRef(null)
  
  useEffect(() => {
    const currentText = text || ''
    
    // 如果文本已完整，直接显示全部
    if (isComplete) {
      if (displayedText !== currentText) {
        setDisplayedText(currentText)
        lastCompleteTextRef.current = currentText
      }
      return
    }
    
    // 如果文本为空，清空显示
    if (!currentText) {
      if (displayedText !== '') {
        setDisplayedText('')
        lastCompleteTextRef.current = ''
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
 * 单个建议卡片组件 - 带打字机效果
 */
function SuggestionCard({ suggestion, idx, config, onApply }) {
  const isPartial = suggestion._isComplete === false
  const isComplete = suggestion._isComplete !== false
  
  // 使用打字机效果显示各个字段
  const label = useTypewriter(suggestion.label || '', isComplete, 10)
  const changesSummary = useTypewriter(suggestion.changes_summary || '', isComplete, 15)
  const modifiedText = useTypewriter(suggestion.modified_text || '', isComplete, 15)
  
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
        transition: 'all 0.3s ease',
        position: 'relative'
      }}
    >
      {/* 标题栏 */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
          <span style={{ color: config.color, fontSize: 14 }}>
            {config.icon}
          </span>
          <span style={{ 
            fontWeight: 600, 
            fontSize: 12, 
            color: 'var(--color-text-primary)',
            fontStyle: isPartial && !suggestion.label ? 'italic' : 'normal',
            minHeight: '1.2em' // 避免布局跳动
          }}>
            {label || (isPartial ? 'Generating...' : 'Unknown Level')}
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
        {/* 一键应用按钮（右上角） */}
        <Button 
          size="small"
          onClick={() => onApply(suggestion)}
          disabled={isPartial}
          style={{ 
            fontSize: 11,
            height: 24,
            padding: '0 12px',
            borderRadius: 6,
            fontWeight: 500,
            background: isPartial ? 'var(--color-bg-secondary)' : config.color,
            borderColor: config.color,
            color: isPartial ? 'var(--color-text-tertiary)' : '#fff',
            opacity: isPartial ? 0.5 : 0.75,
            boxShadow: isPartial ? 'none' : '0 1px 3px rgba(0, 0, 0, 0.08)',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (!isPartial) {
              e.currentTarget.style.opacity = '0.95'
              e.currentTarget.style.transform = 'translateY(-1px)'
              e.currentTarget.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.12)'
            }
          }}
          onMouseLeave={(e) => {
            if (!isPartial) {
              e.currentTarget.style.opacity = '0.75'
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.08)'
            }
          }}
        >
          一键应用
        </Button>
      </div>

      {/* 修改说明 */}
      {(changesSummary || (isPartial && !suggestion.changes_summary)) && (
        <div style={{ 
          fontSize: 11, 
          color: 'var(--color-text-secondary)', 
          marginBottom: 8,
          lineHeight: 1.5,
          fontStyle: isPartial && !suggestion.changes_summary ? 'italic' : 'normal',
          minHeight: '1.5em' // 避免布局跳动
        }}>
          {changesSummary || 'Analyzing modifications...'}
        </div>
      )}

      {/* 修改预览（默认展开） */}
      {(modifiedText || (isPartial && !suggestion.modified_text)) && (
        <div style={{ 
          marginBottom: 8, 
          padding: 8, 
          background: 'var(--color-bg-secondary)', 
          borderRadius: 6,
          fontSize: 11,
          color: 'var(--color-text-primary)',
          maxHeight: 200,
          overflow: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          fontStyle: isPartial && !suggestion.modified_text ? 'italic' : 'normal',
          minHeight: '3em' // 避免布局跳动
        }}>
          {modifiedText || 'Generating protected text...'}
        </div>
      )}

      {/* 移除的风险列表 */}
      {suggestion.removed_risks && suggestion.removed_risks.length > 0 && (
        <div style={{ 
          marginTop: 0, 
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
}

export default function PrivacyProtectionSuggestions({ 
  suggestions,
  onApplySuggestion,
  onGenerateSuggestions,
  hasInference,
  hasEditingText,
  inferenceStatus,
  hasRisks,
}) {

  // 获取级别图标和颜色
  const getLevelConfig = (level) => {
    switch (level) {
      case 'high_privacy':
        return {
          icon: <SafetyOutlined />,
          color: '#f59e0b', // 橙色
          bgColor: 'rgba(245, 158, 11, 0.1)',
          borderColor: '#f59e0b'
        }
      case 'balanced':
        return {
          icon: <CheckCircleOutlined />,
          color: '#3b82f6', // 蓝色
          bgColor: 'rgba(59, 130, 246, 0.1)',
          borderColor: '#3b82f6'
        }
      case 'low_privacy':
        return {
          icon: <ThunderboltOutlined />,
          color: '#10b981', // 绿色
          bgColor: 'rgba(16, 185, 129, 0.1)',
          borderColor: '#10b981'
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

  // 推理完成但无风险 → 不需要修改建议
  if (inferenceStatus === 'done' && !hasRisks) {
    return (
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)', marginBottom: 8 }}>
          Privacy Protection Suggestions
        </div>
        <div className={styles.wordCloudRoot}>
          <div className={styles.infonEmpty} style={{ padding: 12, textAlign: 'center', fontSize: 11, color: '#10b981' }}>
            ✓ No privacy risks detected — no modifications needed
          </div>
        </div>
      </div>
    )
  }

  // 推理未完成，显示提示
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
              size="middle"
              onClick={onGenerateSuggestions}
              loading={suggestions?.status === 'running'}
              className={styles.generateSuggestionsButton}
              style={{
                fontSize: 12,
                height: 32,
                padding: '0 20px',
                borderRadius: 8,
                fontWeight: 600,
                background: 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%)',
                borderColor: 'transparent',
                color: '#fff',
                opacity: 0.85,
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)'
                e.currentTarget.style.opacity = '1'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.opacity = '0.85'
              }}
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
            {[...suggestions.suggestions]
              .sort((a, b) => {
                // 确保排序：高隐私 → 平衡 → 低隐私
                const orderMap = { 'high_privacy': 0, 'balanced': 1, 'low_privacy': 2 }
                return (orderMap[a.level] || 999) - (orderMap[b.level] || 999)
              })
              .map((suggestion, idx) => {
                const config = getLevelConfig(suggestion.level)
                
                return (
                  <SuggestionCard
                    key={suggestion._objIndex ?? idx}
                    suggestion={suggestion}
                    idx={idx}
                    config={config}
                    onApply={onApplySuggestion}
                  />
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

