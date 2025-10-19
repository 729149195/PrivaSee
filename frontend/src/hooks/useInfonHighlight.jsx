import React, { useMemo } from 'react'
import styles from '../components/AgentPage.module.css'
import { getInfonColor, getMatchKeywords, buildInfonIndex, getRelatedInfons } from '../utils/infonUtils'

/**
 * 信息元高亮 Hook
 * 处理消息和输入框的信息元高亮逻辑
 * @param {object} currentSession - 当前会话对象
 * @param {object} infonSessions - 信息元会话对象
 * @param {string} inferenceMode - 推理模式 ('extract' | 'direct')
 * @param {object} privacyInferences - 隐私推理结果
 * @param {object} sessionKeywords - 持久化的关键词集合 { [sessionId]: Set<string> }
 */
export function useInfonHighlight(currentSession, infonSessions, inferenceMode = 'extract', privacyInferences = {}, sessionKeywords = {}) {
  /**
   * 获取消息的所有信息元（排除即将过期的）
   * 包含文本和音频模态的信息元，因为音频转录也需要在文本中高亮
   */
  const getMessageInfons = (messageId) => {
    if (!currentSession?.id) return []
    const runs = (infonSessions?.[currentSession.id]?.runs) || []
    // 排除即将过期的信息元
    // 包含text和audio模态，因为音频转录文本也需要高亮
    const messageRuns = runs.filter(r => 
      r.targetType === 'message' && 
      r.targetKey === messageId && 
      (r.modality === 'text' || r.modality === 'audio') && 
      !r.expiring
    )
    const allInfons = messageRuns.flatMap(r => {
      const infons = Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons : []
      return infons.map(infon => ({ infon, run: r }))
    })
    return allInfons
  }

  /**
   * 获取消息的关系信息元
   */
  const getMessageRelations = (messageId) => {
    const infonList = getMessageInfons(messageId)
    return infonList.filter(({ infon }) => String(infon.infon_type || '').toUpperCase() === 'REL')
  }

  /**
   * 获取 pending 状态的所有信息元
   * 包含文本和音频模态的信息元
   */
  const getPendingInfons = useMemo(() => {
    if (!currentSession?.id) return []
    const runs = (infonSessions?.[currentSession.id]?.runs) || []
    // 包含text和audio模态
    const pendingRuns = runs.filter(r => r.targetType === 'pending' && (r.modality === 'text' || r.modality === 'audio'))
    const allInfons = pendingRuns.flatMap(r => {
      const infons = Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons : []
      return infons.map(infon => ({ infon, run: r }))
    })
    return allInfons
  }, [currentSession?.id, infonSessions])

  /**
   * 构建 pending 高亮数据
   * 在直接推理模式下，使用持久化的 sessionKeywords 进行高亮
   */
  const pendingHighlights = useMemo(() => {
    const highlights = []
    
    // 提取信息元模式：使用信息元进行高亮
    if (inferenceMode === 'extract') {
      if (!getPendingInfons.length) return []
      
      const infonIndex = buildInfonIndex(getPendingInfons)
      
      getPendingInfons.forEach(({ infon }) => {
        const keywords = getMatchKeywords(infon)
        const color = getInfonColor(infon.infon_type)
        keywords.forEach(kw => {
          highlights.push({ keyword: kw, color })
        })
        
        // 关系信息元的关联高亮
        const related = getRelatedInfons(infon, infonIndex)
        related.forEach(relInfon => {
          const relKeywords = getMatchKeywords(relInfon)
          const relColor = getInfonColor(relInfon.infon_type)
          relKeywords.forEach(kw => {
            highlights.push({ keyword: kw, color: relColor })
          })
        })
      })
    }
    
    // 直接推理模式：使用持久化的 sessionKeywords 进行高亮（累积的、不会消失）
    if (inferenceMode === 'direct' && currentSession?.id) {
      const keywords = sessionKeywords?.[currentSession.id]
      
      if (keywords && keywords instanceof Set && keywords.size > 0) {
        // 为每个关键词生成高亮（使用蓝色）
        const highlightColor = '#3b82f6' // 统一使用蓝色表示直接推理模式下的高亮
        keywords.forEach(keyword => {
          highlights.push({ keyword, color: highlightColor })
        })
      }
    }
    
    return highlights
  }, [getPendingInfons, inferenceMode, currentSession?.id, sessionKeywords])

  /**
   * 获取 pending 的关系信息元
   */
  const pendingRelations = useMemo(() => {
    return getPendingInfons.filter(({ infon }) => String(infon.infon_type || '').toUpperCase() === 'REL')
  }, [getPendingInfons])

  /**
   * 获取 pending 的 infon 索引
   */
  const pendingInfonIndex = useMemo(() => {
    return buildInfonIndex(getPendingInfons)
  }, [getPendingInfons])

  /**
   * 渲染带高亮的文本
   */
  const renderHighlightedText = (text, messageId) => {
    const textStr = String(text || '')
    
    // 收集所有需要高亮的关键词及其颜色
    const highlights = []
    
    // 提取信息元模式：使用信息元进行高亮
    if (inferenceMode === 'extract') {
      const infonList = getMessageInfons(messageId)
      if (!infonList.length) return textStr
      
      const infonIndex = buildInfonIndex(infonList)
      
      infonList.forEach(({ infon }) => {
        const keywords = getMatchKeywords(infon)
        const color = getInfonColor(infon.infon_type)
        keywords.forEach(kw => {
          highlights.push({ keyword: kw, color, infon })
        })
        
        // 如果是关系信息元，也高亮其关联的信息元
        const related = getRelatedInfons(infon, infonIndex)
        related.forEach(relInfon => {
          const relKeywords = getMatchKeywords(relInfon)
          const relColor = getInfonColor(relInfon.infon_type)
          relKeywords.forEach(kw => {
            highlights.push({ keyword: kw, color: relColor, infon: relInfon, fromRelation: infon.iid })
          })
        })
      })
    }
    
    // 直接推理模式：使用持久化的 sessionKeywords 进行高亮（累积的、不会消失）
    if (inferenceMode === 'direct' && currentSession?.id) {
      const keywords = sessionKeywords?.[currentSession.id]
      
      if (keywords && keywords instanceof Set && keywords.size > 0) {
        // 为每个关键词生成高亮（使用蓝色）
        const highlightColor = '#3b82f6' // 统一使用蓝色表示直接推理模式下的高亮
        keywords.forEach(keyword => {
          highlights.push({ keyword, color: highlightColor })
        })
      }
    }
    
    if (!highlights.length) return textStr
    
    // 按关键词长度降序排序，优先匹配长关键词
    highlights.sort((a, b) => b.keyword.length - a.keyword.length)
    
    // 构建正则表达式：匹配所有关键词
    const uniqueKeywords = [...new Set(highlights.map(h => h.keyword))]
    const pattern = uniqueKeywords.map(kw => kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
    if (!pattern) return textStr
    
    const regex = new RegExp(`(${pattern})`, 'gi')
    const parts = textStr.split(regex)
    
    return parts.map((part, i) => {
      if (i % 2 === 1) {
        // 匹配的部分：找到对应的颜色
        const match = highlights.find(h => h.keyword.toLowerCase() === part.toLowerCase())
        if (match) {
          // 为关联的信息元添加 data 属性，用于连线（仅在extract模式下）
          const dataAttrs = match.infon && match.fromRelation 
            ? { 'data-infon-id': match.infon.iid, 'data-relation-id': match.fromRelation } 
            : match.infon 
            ? { 'data-infon-id': match.infon.iid }
            : {}
          return <mark key={i} className={styles.infonHighlight} style={{ backgroundColor: match.color + '20', color: match.color }} {...dataAttrs}>{part}</mark>
        }
      }
      return part
    })
  }

  return {
    getMessageInfons,
    getMessageRelations,
    getPendingInfons,
    pendingHighlights,
    pendingRelations,
    pendingInfonIndex,
    renderHighlightedText
  }
}

