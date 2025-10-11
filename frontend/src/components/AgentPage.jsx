import React, { useEffect, useRef, useState, useMemo, useLayoutEffect } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'
import MarkdownMessage from './MarkdownMessage'
import { Splitter, Select, Button, Upload, Progress, Spin, Input, Modal, Popconfirm, message as antdMessage, Tooltip } from 'antd'
import { SendOutlined, StopOutlined, CameraOutlined, PlusOutlined, EditOutlined, DeleteOutlined, CopyOutlined, RedoOutlined, CheckOutlined, CloseOutlined } from '@ant-design/icons'
import WordCloud from './WordCloud'
import LawTree from './LawTree'
import Timeline from './Timeline'
import HighlightInput from './HighlightInput'
import PrivacyRiskAnalysis from './PrivacyRiskAnalysis'
import UserAuth from '../users/UserAuth'
import { useUserStore } from '../users/userStore'
import PrivacyModeIndicator from '../users/PrivacyModeIndicator'

// 连线组件（中文注释）：根据关系信息元画连线连接标签和高亮文本
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
      
      // 找到关系标签的位置（中文注释）
      const tagSelector = `.${styles.relationTag}`
      const allTags = container.querySelectorAll(tagSelector)
      const tagEl = allTags[relIdx]
      if (!tagEl) return

      const tagRect = tagEl.getBoundingClientRect()
      const tagX = tagRect.left - containerRect.left + tagRect.width / 2
      const tagY = tagRect.bottom - containerRect.top

      relatedInfons.forEach((argRef) => {
        // 查找对应的高亮元素（中文注释）
        const highlightEl = container.querySelector(`[data-infon-id="${argRef}"][data-relation-id="${infon.iid}"]`)
        if (!highlightEl) return

        const highlightRect = highlightEl.getBoundingClientRect()
        const highlightX = highlightRect.left - containerRect.left + highlightRect.width / 2
        const highlightY = highlightRect.top - containerRect.top

        // 计算贝塞尔曲线控制点（中文注释）
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


export default function AgentPage() {
  const {
    baseUrl,
    model,
    models,
    customProviders,
    addApiModel,
    sessions,
    currentSessionId,
    isGenerating,
    createSession,
    switchSession,
    deleteSession,
    renameSession,
    getCurrentSession,
    sendMessage,
    stopGenerating,
    regenerateLast,
    _ensureCurrentSession,
    fetchModels,
    setModel,
    // 信息元提取（中文注释）
    startPendingInfons,
    abortPendingInfons,
    startMessageInfons,
    clearAllPendingInfons,
    infonSessions,
    // 隐私推理（中文注释）
    privacyInferences,
    startPrivacyInference,
    abortPrivacyInference,
    selectedLaw,
  } = useStore()

  // 用户状态（中文注释）：从用户 store 获取
  const { currentUser, isLoggedIn } = useUserStore()
  const { setCurrentUser, clearCurrentUser } = useStore()

  // 当前会话对象（中文注释）：需在引用它的 useMemo 之前定义
  const currentSession = getCurrentSession()

  // 同步用户登录状态到主 store（中文注释）：用于控制历史数据持久化
  useEffect(() => {
    if (isLoggedIn && currentUser?.id) {
      setCurrentUser(currentUser.id)
    } else {
      clearCurrentUser()
    }
  }, [isLoggedIn, currentUser, setCurrentUser, clearCurrentUser])

  // 多模态能力检测（中文注释）：基于模型 ID 关键词 + 自定义提供商回退
  const isModelMultimodal = React.useCallback((id) => {
    try {
      if (!id) return false
      // 先基于模型 ID 关键词判断（中文注释）：优先识别已知多模态家族
      const s = String(id).toLowerCase()
      if (/(vision|vl|multi.?modal|llava|idefics|qwen[-_]?vl|gpt-?4o|xcomposer|internvl|minicpm-?v|pixtral|gemma[-_]?3)/.test(s)) return true
      // 自定义提供商（OpenAI 兼容 API）通常不支持图片；若上面未命中则认为是文本（中文注释）
      if (customProviders?.[id]) return false
      return false
    } catch (_) {
      return false
    }
  }, [customProviders])

  // 当前模型是否多模态（中文注释）
  const currentModelIsMultimodal = useMemo(() => isModelMultimodal(model), [model, isModelMultimodal])

  // 上下文是否已含图片（中文注释）
  const contextHasImages = useMemo(() => {
    const msgs = currentSession?.messages || []
    return msgs.some((m) => Array.isArray(m?.images) && m.images.length > 0)
  }, [currentSession?.messages])

  // 初始化当前会话（中文注释）：确保存在 currentSessionId
  useEffect(() => { _ensureCurrentSession() }, [_ensureCurrentSession])

  const [input, setInput] = useState('')
  const [landingInput, setLandingInput] = useState('')
  // 图片选择状态（中文注释）：保存 data URL 用于预览与发送
  const [selectedImages, setSelectedImages] = useState([])
  const listRef = useRef(null)
  const [maxContextTokens, setMaxContextTokens] = useState(null)
  // 属性级展示不再需要画像索引（中文注释）
  const [apiModalOpen, setApiModalOpen] = useState(false)
  const [apiModelId, setApiModelId] = useState('')
  const [apiBaseUrl, setApiBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  // 500ms 防抖计时器（中文注释）
  const pendingTimerRef = useRef(null)
  // 记录上次推断时的信息元数量（中文注释）：防止重复推断
  const lastInferenceRunCountRef = useRef(0)
  // 图片预览 Modal（中文注释）
  const [previewImage, setPreviewImage] = useState(null)
  // 时间线选中的时间（中文注释）：用于筛选 WordCloud 中的信息元
  const [selectedTime, setSelectedTime] = useState(null)
  
  // 消息编辑状态（中文注释）
  const [editingMessageId, setEditingMessageId] = useState(null)
  const [editingContent, setEditingContent] = useState('')
  const [editingImages, setEditingImages] = useState([])
  const [savedMessageInfons, setSavedMessageInfons] = useState(null) // 保存被编辑消息的信息元
  
  // Pending 信息元提取的状态（用于编辑时的实时提取）
  const [editingPendingInfonIndex, setEditingPendingInfonIndex] = useState({})
  // 左侧栏编辑状态（中文注释）：用于追踪正在编辑的 session 和编辑的标题
  const [editingSessionId, setEditingSessionId] = useState(null)
  const [editingTitle, setEditingTitle] = useState('')
  
  // 会话切换时重置时间选择（中文注释）
  useEffect(() => {
    setSelectedTime(null)
    lastInferenceRunCountRef.current = 0 // 重置推断计数器
  }, [currentSessionId])

  // 自动隐私推断（中文注释）：当没有正在提取的信息元时自动启动
  useEffect(() => {
    if (!currentSession || !selectedLaw) return
    
    // 检查是否有正在运行的信息元提取
    const runs = infonSessions?.[currentSession.id]?.runs || []
    const hasRunningInfons = runs.some(run => run.status === 'running')
    
    // 检查当前的推断状态
    const currentInference = privacyInferences?.[currentSession.id]
    const isInferenceRunning = currentInference?.status === 'running'
    
    // 如果有信息元正在提取，且推断正在运行，则中止推断
    if (hasRunningInfons && isInferenceRunning) {
      abortPrivacyInference?.()
      return
    }
    
    // 统计完成的信息元runs数量
    const doneRuns = runs.filter(run => run.status === 'done' && run.resultJson?.infons?.length > 0)
    const currentRunCount = doneRuns.length
    
    // 检查是否有新的信息元（与上次推断时相比）
    const hasNewInfons = currentRunCount > lastInferenceRunCountRef.current
    
    // 如果没有信息元正在提取，有新的信息元数据，且推断未运行，则启动推断
    if (!hasRunningInfons && hasNewInfons && !isInferenceRunning && currentRunCount > 0) {
      // 使用延迟避免频繁触发
      const timer = setTimeout(() => {
        lastInferenceRunCountRef.current = currentRunCount // 更新记录
        startPrivacyInference?.()
      }, 1000)
      return () => clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [infonSessions, currentSessionId, selectedLaw, privacyInferences])

  // 监听信息元提取结果，当首次出现 SIT 类型时自动更新对话标题（中文注释）
  useEffect(() => {
    if (!currentSession?.id) return
    
    const runs = (infonSessions?.[currentSession.id]?.runs) || []
    // 只检查已完成的 run
    const doneRuns = runs.filter(r => r.status === 'done')
    
    for (const run of doneRuns) {
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      for (const infon of infons) {
        const type = String(infon.infon_type || '').toUpperCase()
        if (type === 'SIT' && infon.description) {
          // 获取当前会话标题
          const currentTitle = currentSession.title || ''
          // 如果标题是默认的 "New chat"，则更新为 SIT 的 description
          if (currentTitle === 'New chat') {
            const newTitle = String(infon.description).slice(0, 50) // 限制长度
            renameSession?.(currentSession.id, newTitle)
            return // 只更新一次
          }
        }
      }
    }
  }, [currentSession?.id, currentSession?.title, infonSessions, renameSession])

  // 计算当前会话的信息元数据（用于PrivacyRiskAnalysis）（中文注释）
  const wordData = useMemo(() => {
    if (!currentSession?.id) return []
    const runs = (infonSessions?.[currentSession.id]?.runs) || []
    const allInfons = []
    
    for (const run of runs) {
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      infons.forEach((infon) => {
        const type = String(infon.infon_type || '').toUpperCase()
        if (type === 'SIT') return
        allInfons.push(infon)
      })
    }
    
    return allInfons
  }, [currentSession?.id, infonSessions])

  // 获取当前会话的隐私推理结果（中文注释）
  const inference = useMemo(() => (currentSession ? privacyInferences?.[currentSession.id] : null), [currentSession, privacyInferences])

  // 默认注册 DeepSeek 示例（中文注释）：仅添加一次，已存在则跳过
  useEffect(() => {
    try {
      useStore.getState().addApiModel?.({ id: 'deepseek-chat', baseUrl: 'https://api.deepseek.com/v1', apiKey: 'sk-8c2ee9474f2f44f5969dcd5de280e634' })
    } catch (_) { }
  }, [])

  // 估算上下文 token（中文注释）：字符数/4 + 每张图固定加权（经验值）
  const estimateTokens = (messages) => {
    if (!Array.isArray(messages)) return 0
    let sum = 0
    for (const m of messages) {
      if (m && typeof m.content === 'string') sum += Math.ceil(m.content.length / 4)
      if (Array.isArray(m.images)) sum += m.images.length * 512
    }
    return sum
  }
  const contextTokensUsed = useMemo(() => estimateTokens(currentSession?.messages || []), [currentSession?.messages])
  const contextPercent = useMemo(() => {
    if (typeof maxContextTokens !== 'number' || maxContextTokens <= 0) return 0
    return Math.min(100, Math.round((contextTokensUsed / Math.max(1, maxContextTokens)) * 100))
  }, [contextTokensUsed, maxContextTokens])
  const hasMessages = useMemo(() => (currentSession?.messages || []).length > 0, [currentSession?.messages])

  // 自动滚动到底部（中文注释）：流式时保持跟随
  useEffect(() => {
    const el = listRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [currentSession?.messages?.length, isGenerating])

  // 拉取模型列表（中文注释）：页面挂载时
  useEffect(() => { fetchModels?.() }, [fetchModels])

  // 当上下文或 pending 存在图片时，强制主模型为多模态（中文注释）
  useEffect(() => {
    try {
      const hasPendingImages = selectedImages.length > 0
      const needMultimodal = Boolean(contextHasImages || hasPendingImages)
      if (!needMultimodal) return
      if (model && isModelMultimodal(model)) return
      const list = [model, ...(models || [])].filter((v, i, a) => v && a.indexOf(v) === i)
      const preferred = 'gemma3:12b'
      const mm = list.filter((id) => isModelMultimodal(id))
      if (mm.includes(preferred)) setModel?.(preferred)
      else if (mm.length) setModel?.(mm[0])
    } catch (_) { }
  }, [model, models, contextHasImages, selectedImages, setModel, isModelMultimodal])

  // 根据当前模型查询实际上下文窗口（中文注释）：优先 /api/show，其次 /v1/models
  useEffect(() => {
    const fetchCtx = async () => {
      try {
        const apiBase = (baseUrl || '').replace(/\/?v1\/?$/, '/api')
        let ctxVal = null

        // 优先：Ollama /api/show
        try {
          const res = await fetch(`${apiBase}/show`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: model }),
          })
          if (res.ok) {
            const j = await res.json().catch(() => ({}))
            const pickNum = (v) => {
              if (typeof v === 'number') return v
              if (typeof v === 'string') {
                const n = parseInt(v, 10)
                return Number.isFinite(n) ? n : null
              }
              return null
            }
            ctxVal = (
              pickNum(j?.parameters?.num_ctx) ||
              pickNum(j?.details?.context_length) ||
              pickNum(j?.model_info?.context) ||
              pickNum(j?.model_info?.num_ctx) ||
              pickNum(j?.context) ||
              null
            )
            // 额外扫描 model_info 中的可能键（如 llama.context_length 等）
            if (!ctxVal && j && typeof j === 'object' && j.model_info && typeof j.model_info === 'object') {
              for (const [k, v] of Object.entries(j.model_info)) {
                if (/(context|num_ctx|max_context|max_tokens)/i.test(String(k))) {
                  const n = pickNum(v)
                  if (n && n > 0) { ctxVal = n; break }
                }
              }
            }
          }
        } catch (_) { }

        // 回退：OpenAI 兼容 /v1/models（某些服务会返回 context_length 等）
        if (!ctxVal) {
          try {
            const res2 = await fetch(`${baseUrl}/models`, { method: 'GET' })
            if (res2.ok) {
              const j2 = await res2.json().catch(() => ({}))
              const list = Array.isArray(j2?.data) ? j2.data : (Array.isArray(j2) ? j2 : (Array.isArray(j2?.models) ? j2.models : []))
              const m = (list || []).find((it) => (it?.id || it?.name || it) === model)
              if (m) {
                const pickNum = (v) => {
                  if (typeof v === 'number') return v
                  if (typeof v === 'string') { const n = parseInt(v, 10); return Number.isFinite(n) ? n : null }
                  return null
                }
                ctxVal = (
                  pickNum(m?.context_length) ||
                  pickNum(m?.max_context) ||
                  pickNum(m?.tokenLimit) ||
                  pickNum(m?.max_tokens) ||
                  pickNum(m?.max_input_tokens) ||
                  pickNum(m?.details?.context_length) ||
                  pickNum(m?.parameters?.num_ctx) ||
                  null
                )
              }
            }
          } catch (_) { }
        }

        if (typeof ctxVal === 'number' && ctxVal > 0) setMaxContextTokens(ctxVal)
      } catch (_) { }
    }
    fetchCtx()
  }, [baseUrl, model])

  const contextLabel = useMemo(() => {
    if (typeof maxContextTokens === 'number' && maxContextTokens > 0) {
      return `${contextTokensUsed}/${maxContextTokens} est.`
    }
    return `${contextTokensUsed} est.`
  }, [contextTokensUsed, maxContextTokens])

  // 信息元类型对应的高亮颜色（中文注释）
  const getInfonColor = (infonType) => {
    const colors = {
      DESC: '#3b82f6',  // 描述（实体+属性）：蓝色
      SCEN: '#10b981',  // 场景（时间+位置）：翠绿色
      REL: '#8b5cf6',   // 关系：紫色
      SIT: '#f59e0b',   // 情景：琥珀色
    }
    return colors[String(infonType).toUpperCase()] || '#64748b'
  }

  // 从信息元中提取用于匹配的关键词（中文注释）
  const getMatchKeywords = (infon) => {
    if (!infon || typeof infon !== 'object') return []
    const keywords = []
    const t = String(infon.infon_type || '').toUpperCase()
    
    if (t === 'DESC') {
      // 描述：提取实体和属性作为关键词（优先属性，因为属性是实际值）
      if (infon.attribute) keywords.push(String(infon.attribute))
      if (infon.entity) keywords.push(String(infon.entity))
    } else if (t === 'SCEN') {
      // 场景：提取时间和空间作为关键词（优先时间）
      if (infon.temporal) keywords.push(String(infon.temporal))
      if (infon.spatial) keywords.push(String(infon.spatial))
    } else if (t === 'REL' && infon.relation_name) {
      keywords.push(String(infon.relation_name))
    }
    // SIT 不提取关键词用于高亮
    
    return keywords.filter(k => k && k.trim())
  }

  // 获取消息的所有信息元（中文注释）：包括该消息对应的所有 run 的所有 infons
  const getMessageInfons = (messageId) => {
    if (!currentSession?.id) return []
    const runs = (infonSessions?.[currentSession.id]?.runs) || []
    const messageRuns = runs.filter(r => r.targetType === 'message' && r.targetKey === messageId && r.modality === 'text')
    const allInfons = messageRuns.flatMap(r => {
      const infons = Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons : []
      return infons.map(infon => ({ infon, run: r }))
    })
    return allInfons
  }

  // 构建信息元索引（中文注释）：用于快速查找 iid 对应的信息元
  const buildInfonIndex = (infonList) => {
    const index = {}
    infonList.forEach(({ infon }) => {
      if (infon.iid) index[infon.iid] = infon
    })
    return index
  }

  // 收集关系信息元关联的所有信息元（中文注释）
  const getRelatedInfons = (infon, infonIndex) => {
    const related = []
    if (String(infon.infon_type || '').toUpperCase() === 'REL' && Array.isArray(infon.arg_refs)) {
      infon.arg_refs.forEach(ref => {
        if (infonIndex[ref]) related.push(infonIndex[ref])
      })
    }
    return related
  }

  // 渲染带高亮的文本（中文注释）：自动高亮所有信息元
  const renderHighlightedText = (text, messageId) => {
    const textStr = String(text || '')
    const infonList = getMessageInfons(messageId)
    if (!infonList.length) return textStr
    
    const infonIndex = buildInfonIndex(infonList)
    
    // 收集所有需要高亮的关键词及其颜色（中文注释）
    const highlights = []
    infonList.forEach(({ infon }) => {
      const keywords = getMatchKeywords(infon)
      const color = getInfonColor(infon.infon_type)
      keywords.forEach(kw => {
        highlights.push({ keyword: kw, color, infon })
      })
      
      // 如果是关系信息元，也高亮其关联的信息元（中文注释）
      const related = getRelatedInfons(infon, infonIndex)
      related.forEach(relInfon => {
        const relKeywords = getMatchKeywords(relInfon)
        const relColor = getInfonColor(relInfon.infon_type)
        relKeywords.forEach(kw => {
          highlights.push({ keyword: kw, color: relColor, infon: relInfon, fromRelation: infon.iid })
        })
      })
    })
    
    if (!highlights.length) return textStr
    
    // 按关键词长度降序排序，优先匹配长关键词（中文注释）
    highlights.sort((a, b) => b.keyword.length - a.keyword.length)
    
    // 构建正则表达式：匹配所有关键词（中文注释）
    const uniqueKeywords = [...new Set(highlights.map(h => h.keyword))]
    const pattern = uniqueKeywords.map(kw => kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
    if (!pattern) return textStr
    
    const regex = new RegExp(`(${pattern})`, 'gi')
    const parts = textStr.split(regex)
    
    return parts.map((part, i) => {
      if (i % 2 === 1) {
        // 匹配的部分：找到对应的颜色（中文注释）
        const match = highlights.find(h => h.keyword.toLowerCase() === part.toLowerCase())
        if (match) {
          // 为关联的信息元添加 data 属性，用于连线（中文注释）
          const dataAttrs = match.fromRelation ? { 'data-infon-id': match.infon.iid, 'data-relation-id': match.fromRelation } : { 'data-infon-id': match.infon.iid }
          return <mark key={i} className={styles.infonHighlight} style={{ backgroundColor: match.color + '20', color: match.color }} {...dataAttrs}>{part}</mark>
        }
      }
      return part
    })
  }

  // 获取消息的关系信息元（中文注释）
  const getMessageRelations = (messageId) => {
    const infonList = getMessageInfons(messageId)
    return infonList.filter(({ infon }) => String(infon.infon_type || '').toUpperCase() === 'REL')
  }

  // 获取 pending 状态的所有信息元（中文注释）：用于输入框实时高亮
  const getPendingInfons = useMemo(() => {
    if (!currentSession?.id) return []
    const runs = (infonSessions?.[currentSession.id]?.runs) || []
    const pendingRuns = runs.filter(r => r.targetType === 'pending' && r.modality === 'text')
    const allInfons = pendingRuns.flatMap(r => {
      const infons = Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons : []
      return infons.map(infon => ({ infon, run: r }))
    })
    return allInfons
  }, [currentSession?.id, infonSessions])

  // 构建 pending 高亮数据（中文注释）：转换为 HighlightInput 需要的格式
  const pendingHighlights = useMemo(() => {
    if (!getPendingInfons.length) return []
    
    const infonIndex = buildInfonIndex(getPendingInfons)
    const highlights = []
    
    getPendingInfons.forEach(({ infon }) => {
      const keywords = getMatchKeywords(infon)
      const color = getInfonColor(infon.infon_type)
      keywords.forEach(kw => {
        highlights.push({ keyword: kw, color })
      })
      
      // 关系信息元的关联高亮（中文注释）
      const related = getRelatedInfons(infon, infonIndex)
      related.forEach(relInfon => {
        const relKeywords = getMatchKeywords(relInfon)
        const relColor = getInfonColor(relInfon.infon_type)
        relKeywords.forEach(kw => {
          highlights.push({ keyword: kw, color: relColor })
        })
      })
    })
    
    return highlights
  }, [getPendingInfons])

  // 获取 pending 的关系信息元（中文注释）
  const pendingRelations = useMemo(() => {
    return getPendingInfons.filter(({ infon }) => String(infon.infon_type || '').toUpperCase() === 'REL')
  }, [getPendingInfons])

  // 获取 pending 的 infon 索引（中文注释）
  const pendingInfonIndex = useMemo(() => {
    return buildInfonIndex(getPendingInfons)
  }, [getPendingInfons])

  // 属性级展示（中文注释）：不需要默认选择



  const handleSend = async () => {
    const text = (input || '').trim()
    const imgs = [...selectedImages]
    const hasImages = imgs.length > 0
    if (!text && !hasImages) return
    
    // 清空输入和图片（中文注释）
    setInput('')
    setSelectedImages([])
    
    if (hasImages) {
      const userId = await useStore.getState().sendMessageWithImages(text, imgs)
      try {
        const adopted = useStore.getState().adoptPendingInfonsToMessage?.(userId) || 0
        if (!adopted) startMessageInfons?.(userId)
      } catch (_) {}
    } else {
      const userId = await sendMessage(text)
      try {
        const adopted = useStore.getState().adoptPendingInfonsToMessage?.(userId) || 0
        if (!adopted) startMessageInfons?.(userId)
      } catch (_) {}
    }
    
    // 发送后立即清空 pending 信息元，移除关系标签显示（中文注释）
    try {
      clearAllPendingInfons?.()
    } catch (_) {}
  }

  const handleLandingSend = async () => {
    const text = (landingInput || '').trim()
    const imgs = [...selectedImages]
    const hasImages = imgs.length > 0
    if (!text && !hasImages) return
    
    // 清空输入和图片（中文注释）
    setLandingInput('')
    setSelectedImages([])
    
    if (hasImages) {
      const userId = await useStore.getState().sendMessageWithImages(text, imgs)
      try {
        const adopted = useStore.getState().adoptPendingInfonsToMessage?.(userId) || 0
        if (!adopted) startMessageInfons?.(userId)
      } catch (_) {}
    } else {
      const userId = await sendMessage(text)
      try {
        const adopted = useStore.getState().adoptPendingInfonsToMessage?.(userId) || 0
        if (!adopted) startMessageInfons?.(userId)
      } catch (_) {}
    }
    
    // 发送后立即清空 pending 信息元，移除关系标签显示（中文注释）
    try {
      clearAllPendingInfons?.()
    } catch (_) {}
  }

  // 复制消息内容（中文注释）
  const handleCopyMessage = (content) => {
    navigator.clipboard.writeText(content).then(() => {
      antdMessage.success('已复制到剪贴板')
    }).catch(() => {
      antdMessage.error('复制失败')
    })
  }

  // 开始编辑消息（中文注释）：暂时移除该消息的信息元
  const handleEditMessage = (messageId, content, images) => {
    setEditingMessageId(messageId)
    setEditingContent(content || '')
    setEditingImages(images || [])
    
    // 保存并暂时移除该消息的信息元
    const session = getCurrentSession()
    if (session) {
      const currentInfonSession = infonSessions?.[session.id]
      if (currentInfonSession?.runs) {
        // 找到该消息的所有信息元runs
        const messageRuns = currentInfonSession.runs.filter(run => 
          run.targetType === 'message' && run.targetKey === messageId
        )
        
        if (messageRuns.length > 0) {
          // 保存
          setSavedMessageInfons({ messageId, runs: messageRuns })
          
          // 从infonSessions中移除
          const filteredRuns = currentInfonSession.runs.filter(run => 
            !(run.targetType === 'message' && run.targetKey === messageId)
          )
          
          useStore.setState({
            infonSessions: {
              ...infonSessions,
              [session.id]: { ...currentInfonSession, runs: filteredRuns }
            }
          })
        }
      }
    }
    
    // 信息元提取会由useEffect自动触发
  }

  // 取消编辑（中文注释）：恢复原消息的信息元
  const handleCancelEdit = () => {
    // 恢复之前保存的信息元
    if (savedMessageInfons) {
      const session = getCurrentSession()
      if (session) {
        const currentInfonSession = infonSessions?.[session.id]
        if (currentInfonSession) {
          useStore.setState({
            infonSessions: {
              ...infonSessions,
              [session.id]: {
                ...currentInfonSession,
                runs: [...currentInfonSession.runs, ...savedMessageInfons.runs]
              }
            }
          })
        }
      }
      setSavedMessageInfons(null)
    }
    
    setEditingMessageId(null)
    setEditingContent('')
    setEditingImages([])
    // 清除 pending 信息元
    clearAllPendingInfons?.()
  }

  // 保存编辑（中文注释）
  const handleSaveEdit = async () => {
    if (!editingMessageId) return
    
    const text = editingContent.trim()
    if (!text && editingImages.length === 0) {
      antdMessage.warning('消息内容不能为空')
      return
    }

    // 获取当前 session
    const session = getCurrentSession()
    if (!session) return

    // 找到要编辑的消息及其后续消息
    const messageIndex = session.messages.findIndex(m => m.id === editingMessageId)
    if (messageIndex === -1) return

    // 删除该消息及其后续的所有消息
    const newMessages = session.messages.slice(0, messageIndex)
    const deletedMessages = session.messages.slice(messageIndex) // 被删除的消息
    const deletedMessageIds = new Set(deletedMessages.map(m => m.id))
    
    // 更新 session 的消息列表
    const updatedSessions = sessions.map(s => {
      if (s.id === session.id) {
        return { ...s, messages: newMessages }
      }
      return s
    })
    
    // 清理被删除消息的信息元（中文注释）
    const currentInfonSession = infonSessions?.[session.id]
    if (currentInfonSession?.runs) {
      const filteredRuns = currentInfonSession.runs.filter(run => {
        // 保留不属于被删除消息的 runs
        if (run.targetType === 'message' && deletedMessageIds.has(run.targetKey)) {
          return false // 删除这个 run
        }
        return true // 保留这个 run
      })
      
      useStore.setState({
        infonSessions: {
          ...infonSessions,
          [session.id]: { ...currentInfonSession, runs: filteredRuns }
        }
      })
    }
    
    // 清空隐私推理结果（中文注释）：因为消息改变了，需要重新推理
    const currentPrivacyInference = privacyInferences?.[session.id]
    if (currentPrivacyInference) {
      useStore.setState({
        privacyInferences: {
          ...privacyInferences,
          [session.id]: {
            status: 'idle',
            risks: [],
            buffer: '',
            abortController: null,
            createdAt: Date.now(),
            updatedAt: Date.now()
          }
        }
      })
    }
    
    // 更新 store 的 sessions
    useStore.setState({ sessions: updatedSessions })

    // 采纳 pending 信息元
    // 注意：这里我们将 pending 信息元转移到即将发送的新消息
    
    // 发送新消息
    if (editingImages.length > 0) {
      const userId = await useStore.getState().sendMessageWithImages(text, editingImages)
      try {
        const adopted = useStore.getState().adoptPendingInfonsToMessage?.(userId) || 0
        if (!adopted) startMessageInfons?.(userId)
      } catch (_) {}
    } else {
      const userId = await sendMessage(text)
      try {
        const adopted = useStore.getState().adoptPendingInfonsToMessage?.(userId) || 0
        if (!adopted) startMessageInfons?.(userId)
      } catch (_) {}
    }

    // 清理编辑状态
    handleCancelEdit()
    
    antdMessage.success('消息已更新并重新生成')
  }

  // 重试生成（中文注释）：保存用户消息的信息元，删除用户和助手消息，重新发送，然后迁移信息元
  const handleRetry = async () => {
    const session = getCurrentSession()
    if (!session || !session.messages || session.messages.length === 0) return

    const messages = session.messages
    const lastUserIndex = [...messages].reverse().findIndex(m => m.role === 'user')
    if (lastUserIndex === -1) return

    const actualIndex = messages.length - 1 - lastUserIndex
    const lastUserMessage = messages[actualIndex]
    const oldUserMessageId = lastUserMessage.id

    // 保存该用户消息的信息元
    const currentInfonSession = infonSessions?.[session.id]
    let savedUserInfonRuns = []
    if (currentInfonSession?.runs) {
      savedUserInfonRuns = currentInfonSession.runs.filter(run => 
        run.targetType === 'message' && run.targetKey === oldUserMessageId
      )
    }

    // 删除从用户消息开始的所有消息（包括用户和助手消息）
    const updatedMessages = messages.slice(0, actualIndex)
    const deletedMessages = messages.slice(actualIndex)
    const deletedMessageIds = deletedMessages.map(m => m.id)

    // 清理被删除消息的信息元（暂时）
    if (currentInfonSession?.runs) {
      const filteredRuns = currentInfonSession.runs.filter(run => {
        if (run.targetType === 'message' && deletedMessageIds.includes(run.targetKey)) {
          return false
        }
        return true
      })
      
      useStore.setState({
        infonSessions: {
          ...infonSessions,
          [session.id]: { ...currentInfonSession, runs: filteredRuns }
        }
      })
    }
    
    // 清空隐私推理结果
    const currentPrivacyInference = privacyInferences?.[session.id]
    if (currentPrivacyInference) {
      useStore.setState({
        privacyInferences: {
          ...privacyInferences,
          [session.id]: {
            status: 'idle',
            risks: [],
            buffer: '',
            abortController: null,
            createdAt: Date.now(),
            updatedAt: Date.now()
          }
        }
      })
    }

    // 更新 session
    const updatedSessions = sessions.map(s => {
      if (s.id === session.id) {
        return { ...s, messages: updatedMessages }
      }
      return s
    })
    useStore.setState({ sessions: updatedSessions })

    // 重新发送用户消息
    const hasImages = Array.isArray(lastUserMessage.images) && lastUserMessage.images.length > 0
    let newUserMessageId
    if (hasImages) {
      newUserMessageId = await useStore.getState().sendMessageWithImages(lastUserMessage.content, lastUserMessage.images)
    } else {
      newUserMessageId = await sendMessage(lastUserMessage.content)
    }

    // 迁移信息元到新的用户消息（如果有保存的信息元）
    if (savedUserInfonRuns.length > 0 && newUserMessageId) {
      const updatedRuns = savedUserInfonRuns.map(run => ({
        ...run,
        targetKey: newUserMessageId // 更新到新的消息ID
      }))
      
      const latestInfonSession = useStore.getState().infonSessions?.[session.id]
      if (latestInfonSession) {
        useStore.setState({
          infonSessions: {
            ...useStore.getState().infonSessions,
            [session.id]: {
              ...latestInfonSession,
              runs: [...latestInfonSession.runs, ...updatedRuns]
            }
          }
        })
      }
    }
  }

  // 处理图片选择（中文注释）：将文件读取为 data URL 后加入队列
  const handlePickImages = async (e) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    const toDataUrl = (file) => new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
    try {
      const urls = await Promise.all(files.map(toDataUrl))
      setSelectedImages((prev) => [...prev, ...urls])
    } catch (_) { }
    e.target.value = ''
  }

  const removeSelectedImage = (idx) => {
    setSelectedImages((prev) => prev.filter((_, i) => i !== idx))
  }

  // 输入变化时，立刻中止 pending 的提取（中文注释）
  useEffect(() => {
    try { abortPendingInfons?.(false) } catch (_) {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, landingInput, editingContent])

  // 500ms 防抖：在用户停止输入 500ms 后启动 pending 提取（中文注释）
  // 支持主输入框和编辑框两种模式
  useEffect(() => {
    // 优先使用编辑模式的内容（如果正在编辑）
    const isEditing = editingMessageId !== null
    const textToUse = isEditing 
      ? (editingContent || '').trim()
      : (hasMessages ? (input || '').trim() : (landingInput || '').trim())
    const imgs = isEditing ? [...editingImages] : [...selectedImages]
    
    if (pendingTimerRef.current) {
      clearTimeout(pendingTimerRef.current)
      pendingTimerRef.current = null
    }
    // 若无输入也无图片，则不启动（中文注释）
    if (!textToUse && imgs.length === 0) return
    pendingTimerRef.current = setTimeout(() => {
      try { startPendingInfons?.(textToUse, imgs) } catch (_) {}
      pendingTimerRef.current = null
    }, 1500)
    return () => {
      if (pendingTimerRef.current) {
        clearTimeout(pendingTimerRef.current)
        pendingTimerRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, landingInput, selectedImages, hasMessages, editingMessageId, editingContent, editingImages])

  return (
    <div className={styles.shell}>
      {/* 左侧：侧边栏 */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarTop}>
          <button className={styles.newBtn} onClick={createSession}>
            <PlusOutlined className={styles.newBtnIcon} />
            <span>New chat</span>
          </button>
        </div>
        {/* 无痕模式提示（中文注释） */}
        <PrivacyModeIndicator />
        <div className={styles.sidebarScroll}>
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`${styles.chatItem} ${s.id === currentSessionId ? styles.chatItemActive : ''}`}
              onClick={() => {
                if (editingSessionId !== s.id) {
                  switchSession(s.id)
                }
              }}
              title={editingSessionId === s.id ? '' : s.title}
            >
              <div className={styles.chatItemHeader}>
                <div className={styles.chatItemInfo}>
                  {editingSessionId === s.id ? (
                    <Input
                      className={styles.chatNameInput}
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onPressEnter={(e) => {
                        e.stopPropagation()
                        const newTitle = editingTitle.trim()
                        if (newTitle && newTitle !== s.title) {
                          renameSession(s.id, newTitle)
                        }
                        setEditingSessionId(null)
                        setEditingTitle('')
                      }}
                      onBlur={() => {
                        const newTitle = editingTitle.trim()
                        if (newTitle && newTitle !== s.title) {
                          renameSession(s.id, newTitle)
                        }
                        setEditingSessionId(null)
                        setEditingTitle('')
                      }}
                      onClick={(e) => e.stopPropagation()}
                      autoFocus
                      size="small"
                    />
                  ) : (
                    <div className={styles.chatName}>{s.title}</div>
                  )}
                  {editingSessionId !== s.id && (
                    <div className={styles.chatMeta}>{new Date(s.updatedAt).toLocaleString()}</div>
                  )}
                </div>
                <div className={styles.chatActions}>
                  <button 
                    className={styles.iconBtn} 
                    onClick={(e) => { 
                      e.stopPropagation()
                      setEditingSessionId(s.id)
                      setEditingTitle(s.title)
                    }} 
                    title="Rename"
                  >
                    <EditOutlined />
                  </button>
                  <Popconfirm
                    title="删除对话"
                    description="确定要删除这个对话吗？"
                    onConfirm={(e) => {
                      e?.stopPropagation()
                      deleteSession(s.id)
                    }}
                    onCancel={(e) => e?.stopPropagation()}
                    okText="删除"
                    cancelText="取消"
                    placement="right"
                  >
                    <button 
                      className={styles.iconBtn} 
                      onClick={(e) => e.stopPropagation()} 
                      title="Delete"
                    >
                      <DeleteOutlined />
                    </button>
                  </Popconfirm>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className={styles.sidebarBottom}>
          {/* <div className={styles.kv}><span>Base URL</span><span>{baseUrl}</span></div> */}
          <div className={styles.kv}><span>Model</span><span>{model}</span></div>
          <div className={styles.contextSection}>
            <div className={styles.contextInfo} style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <div style={{ flex: 1 }}>
                <div className={styles.contextLabel}>Context window<span className={styles.contextText}>{contextLabel}</span></div>
                <Progress percent={contextPercent} size="small" className={styles.contextProgress} />
              </div>
              {/* 用户登录入口（中文注释）：放在 Context window 右上角 */}
              <div style={{ marginLeft: '8px', marginTop: '-4px' }}>
                <UserAuth />
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* 右侧：主区域 */}
      <section className={styles.main}>
        <div className={styles.scroll} ref={listRef}>
          {/* 顶部：左上角模型选择器 */}
          <div className={styles.toolbar}>
            <div className={styles.modelPicker}>
              <Select
                style={{ minWidth: 220 }}
                value={model}
                onChange={(v) => {
                  const requireMultimodal = Boolean(contextHasImages || (selectedImages.length > 0))
                  if (requireMultimodal && !isModelMultimodal(v)) {
                    message.warning('Cannot switch to a non-multimodal model when images exist in context or pending')
                    return
                  }
                  setModel?.(v)
                }}
                options={(() => {
                  const requireMultimodal = Boolean(contextHasImages || (selectedImages.length > 0))
                  return [model, ...(models || [])]
                    .filter((v, i, a) => v && a.indexOf(v) === i)
                    .map((v) => ({
                      label: `${v}${isModelMultimodal(v) ? ' (multimodal)' : ' (text-only)'}`,
                      value: v,
                      disabled: (requireMultimodal && !isModelMultimodal(v))
                    }))
                })()}
              />
              <Button onClick={() => setApiModalOpen(true)}>Add API model</Button>
            </div>
          </div>
          <Modal
            title="Add API model"
            open={apiModalOpen}
            onCancel={() => setApiModalOpen(false)}
            onOk={() => {
              try {
                useStore.getState().addApiModel({ id: apiModelId.trim(), baseUrl: apiBaseUrl.trim(), apiKey: apiKey.trim() })
                setApiModalOpen(false)
              } catch (_) { }
            }}
          >
            <div style={{ display: 'grid', gap: 8 }}>
              <Input placeholder="Model ID" value={apiModelId} onChange={(e) => setApiModelId(e.target.value)} />
              <Input placeholder="Base URL" value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} />
              <Input.Password placeholder="API Key" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            </div>
          </Modal>
          <Splitter className={styles.splitterRoot}>
            <Splitter.Panel style={{ overflow: 'hidden', position: 'relative', display: 'flex', flexDirection: 'column' }}>
              {/* 信息元类型图例（中文注释） */}
              <div className={styles.infonLegend}>
                <div className={styles.legendItem}>
                  <span className={styles.legendDot} style={{ backgroundColor: getInfonColor('DESC') }}></span>
                  <span className={styles.legendLabel}>Description (DESC)</span>
                </div>
                <div className={styles.legendItem}>
                  <span className={styles.legendDot} style={{ backgroundColor: getInfonColor('SCEN') }}></span>
                  <span className={styles.legendLabel}>Scenario (SCEN)</span>
                </div>
                <div className={styles.legendItem}>
                  <span className={styles.legendDot} style={{ backgroundColor: getInfonColor('REL') }}></span>
                  <span className={styles.legendLabel}>Relation (REL)</span>
                </div>
              </div>
              <div className={styles.leftPaneScroll} ref={listRef} style={{ flex: 1, overflow: 'auto' }}>
                {hasMessages ? (
                  <div className={styles.column}>
                    {(currentSession?.messages || []).map((m) => {
                      const isUser = m.role === 'user'
                      const messageRelations = getMessageRelations(m.id)
                      const infonList = getMessageInfons(m.id)
                      const infonIndex = buildInfonIndex(infonList)
                      
                      return (
                        <div key={m.id} className={`${styles.msgRow} ${isUser ? styles.rowUser : styles.rowAssistant}`}>
                          {isUser ? (
                            <>
                              <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                                {editingMessageId === m.id ? (
                                  // 编辑模式（中文注释）：使用和主输入框相同的样式
                                  <div className={styles.editingComposer}>
                                    {/* 图片预览 */}
                                    {editingImages.length > 0 && (
                                      <div className={styles.composerPreviews}>
                                        {editingImages.map((src, imgIdx) => (
                                          <div key={imgIdx} className={styles.composerPreviewItem}>
                                            <img src={src} alt={`img-${imgIdx}`} className={styles.composerPreviewImg} />
                                            <button
                                              className={styles.composerPreviewRemove}
                                              onClick={() => setEditingImages(editingImages.filter((_, i) => i !== imgIdx))}
                                            >✕</button>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                    {/* 输入框 */}
                                    <div className={styles.composerRow}>
                                      <HighlightInput
                                        className={styles.composerInput}
                                        value={editingContent}
                                        onChange={setEditingContent}
                                        placeholder="编辑消息..."
                                        highlights={pendingHighlights}
                                        autoSize={{ minRows: 2, maxRows: 10 }}
                                      />
                                    </div>
                                    {/* Pending关系标签显示 */}
                                    {pendingRelations.length > 0 && (
                                      <div className={styles.relationTags} style={{ marginTop: 8 }}>
                                        {pendingRelations.map(({ infon }, idx) => {
                                          const relatedInfons = getRelatedInfons(infon, pendingInfonIndex)
                                          const color = getInfonColor('REL')
                                          
                                          return (
                                            <div key={idx} className={styles.relationTag} style={{ borderColor: color }}>
                                              <span className={styles.relationTagName} style={{ color: color }}>
                                                {infon.relation_name || 'Relation'}
                                              </span>
                                              <span className={styles.relationTagArgs}>
                                                {relatedInfons.map((rel, ri) => {
                                                  const relColor = getInfonColor(rel.infon_type)
                                                  const keywords = getMatchKeywords(rel)
                                                  const label = keywords[0] || rel.iid
                                                  return (
                                                    <React.Fragment key={ri}>
                                                      {ri > 0 && <span className={styles.relationTagSep}>→</span>}
                                                      <span className={styles.relationTagArg} style={{ color: relColor }}>
                                                        {label}
                                                      </span>
                                                    </React.Fragment>
                                                  )
                                                })}
                                              </span>
                                            </div>
                                          )
                                        })}
                                      </div>
                                    )}
                                    {/* 操作按钮 */}
                                    <div style={{ display: 'flex', gap: 8, marginTop: 12, justifyContent: 'flex-end' }}>
                                      <Button size="small" icon={<CheckOutlined />} onClick={handleSaveEdit} type="primary">保存并重新生成</Button>
                                      <Button size="small" icon={<CloseOutlined />} onClick={handleCancelEdit}>取消</Button>
                                    </div>
                                  </div>
                                ) : (
                                  // 正常显示模式（中文注释）
                                  <>
                                    <div className={`${styles.msgBubble} ${styles.msgBubbleUser}`} style={{ position: 'relative' }}>
                                      {messageRelations.length > 0 && (
                                        <RelationConnections messageId={m.id} relations={messageRelations} infonIndex={infonIndex} />
                                      )}
                                      <div className={styles.msgContent} style={{ position: 'relative', zIndex: 2 }}>{renderHighlightedText(m.content, m.id)}</div>
                                      {Array.isArray(m.images) && m.images.length > 0 && (
                                        <div className={styles.msgImages}>
                                          {m.images.map((src, imgIdx) => (
                                            <img key={imgIdx} src={src} alt={`img-${imgIdx}`} className={styles.msgImage} />
                                          ))}
                                        </div>
                                      )}
                                      {/* 关系标签（中文注释） */}
                                      {messageRelations.length > 0 && (
                                        <div className={styles.relationTags}>
                                          {messageRelations.map(({ infon }, idx) => {
                                            const relatedInfons = getRelatedInfons(infon, infonIndex)
                                            const color = getInfonColor('REL')
                                            
                                            return (
                                              <div key={idx} className={styles.relationTag} style={{ borderColor: color }}>
                                                <span className={styles.relationTagName} style={{ color: color }}>
                                                  {infon.relation_name || 'Relation'}
                                                </span>
                                                <span className={styles.relationTagArgs}>
                                                  {relatedInfons.map((rel, ri) => {
                                                    const relColor = getInfonColor(rel.infon_type)
                                                    const keywords = getMatchKeywords(rel)
                                                    const label = keywords[0] || rel.iid
                                                    return (
                                                      <React.Fragment key={ri}>
                                                        {ri > 0 && <span className={styles.relationTagSep}>→</span>}
                                                        <span className={styles.relationTagArg} style={{ color: relColor }}>
                                                          {label}
                                                        </span>
                                                      </React.Fragment>
                                                    )
                                                  })}
                                                </span>
                                              </div>
                                            )
                                          })}
                                        </div>
                                      )}
                                    </div>
                                    {/* 用户消息操作按钮（中文注释） */}
                                    <div className={styles.messageActions} style={{ justifyContent: 'flex-end' }}>
                                      <Tooltip title="复制">
                                        <Button 
                                          type="text" 
                                          size="small" 
                                          icon={<CopyOutlined />}
                                          onClick={() => handleCopyMessage(m.content)}
                                          className={styles.messageActionBtn}
                                        />
                                      </Tooltip>
                                      <Tooltip title="编辑">
                                        <Button 
                                          type="text" 
                                          size="small" 
                                          icon={<EditOutlined />}
                                          onClick={() => handleEditMessage(m.id, m.content, m.images)}
                                          className={styles.messageActionBtn}
                                          disabled={isGenerating}
                                        />
                                      </Tooltip>
                                    </div>
                                  </>
                                )}
                              </div>
                              <div className={styles.avatar}>U</div>
                            </>
                          ) : (
                            <>
                              <div className={styles.avatar}>A</div>
                              <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                                <div className={`${styles.msgBubble} ${styles.msgBubbleAssistant}`} style={{ position: 'relative' }}>
                                  {messageRelations.length > 0 && (
                                    <RelationConnections messageId={m.id} relations={messageRelations} infonIndex={infonIndex} />
                                  )}
                                  {m.reasoning && (
                                    <div className={styles.reasoningBox}>
                                      <div className={styles.reasoningTitle}>Thinking</div>
                                      <div className={styles.reasoningBody}>
                                        <MarkdownMessage content={m.reasoning} />
                                      </div>
                                    </div>
                                  )}
                                  <div className={styles.msgContent}>
                                    <div className={styles.assistantTextHighlight} style={{ position: 'relative', zIndex: 2 }}>
                                      {renderHighlightedText(m.content, m.id)}
                                    </div>
                                  </div>
                                  {/* 关系标签（中文注释） */}
                                  {messageRelations.length > 0 && (
                                    <div className={styles.relationTags}>
                                      {messageRelations.map(({ infon }, idx) => {
                                        const relatedInfons = getRelatedInfons(infon, infonIndex)
                                        const color = getInfonColor('REL')
                                        
                                        return (
                                          <div key={idx} className={styles.relationTag} style={{ borderColor: color }}>
                                            <span className={styles.relationTagName} style={{ color: color }}>
                                              {infon.relation_name || 'Relation'}
                                            </span>
                                            <span className={styles.relationTagArgs}>
                                              {relatedInfons.map((rel, ri) => {
                                                const relColor = getInfonColor(rel.infon_type)
                                                const keywords = getMatchKeywords(rel)
                                                const label = keywords[0] || rel.iid
                                                return (
                                                  <React.Fragment key={ri}>
                                                    {ri > 0 && <span className={styles.relationTagSep}>→</span>}
                                                    <span className={styles.relationTagArg} style={{ color: relColor }}>
                                                      {label}
                                                    </span>
                                                  </React.Fragment>
                                                )
                                              })}
                                            </span>
                                          </div>
                                        )
                                      })}
                                    </div>
                                  )}
                                  {m.streaming ? <div className={styles.cursor}>▍</div> : null}
                                  {m.error ? <div className={styles.error}>Error: {m.error}</div> : null}
                                </div>
                                {/* 助手消息操作按钮（中文注释） */}
                                {!m.streaming && (
                                  <div className={styles.messageActions}>
                                    <Tooltip title="复制">
                                      <Button 
                                        type="text" 
                                        size="small" 
                                        icon={<CopyOutlined />}
                                        onClick={() => handleCopyMessage(m.content)}
                                        className={styles.messageActionBtn}
                                      />
                                    </Tooltip>
                                    <Tooltip title="重新生成">
                                      <Button 
                                        type="text" 
                                        size="small" 
                                        icon={<RedoOutlined />}
                                        onClick={handleRetry}
                                        className={styles.messageActionBtn}
                                        disabled={isGenerating}
                                      />
                                    </Tooltip>
                                  </div>
                                )}
                              </div>
                            </>
                          )}
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <div className={styles.landing}>
                    <div className={styles.landingTitle}>How can I help you today?</div>
                    <div className={styles.landingSearch}>
                      {selectedImages.length > 0 && (
                        <div className={styles.composerPreviews}>
                          {selectedImages.map((src, i) => (
                            <div key={i} className={styles.composerPreviewItem}>
                              <img 
                                src={src} 
                                alt={`preview-${i}`} 
                                className={styles.composerPreviewImg} 
                                onClick={() => setPreviewImage(src)}
                                style={{ cursor: 'pointer' }}
                              />
                              <button className={styles.composerPreviewRemove} onClick={(e) => { e.stopPropagation(); removeSelectedImage(i); }}>✕</button>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className={styles.landingInputArea}>
                        <div className={styles.landingControls}>
                          <Upload
                            disabled={!currentModelIsMultimodal}
                            multiple
                            accept="image/*"
                            showUploadList={false}
                            beforeUpload={(file) => {
                              const reader = new FileReader()
                              reader.onload = () => setSelectedImages((prev) => [...prev, reader.result])
                              reader.readAsDataURL(file)
                              return Upload.LIST_IGNORE
                            }}
                          >
                            <Button icon={<CameraOutlined />} disabled={!currentModelIsMultimodal} title={currentModelIsMultimodal ? '' : 'Current model does not support images'} />
                          </Upload>
                          <HighlightInput
                            className={styles.landingInput}
                            placeholder="Type your question..."
                            value={landingInput}
                            onChange={setLandingInput}
                            onPressEnter={handleLandingSend}
                            highlights={pendingHighlights}
                            autoSize={{ minRows: 1, maxRows: 6 }}
                          />
                          <Button type="primary" icon={<SendOutlined />} onClick={handleLandingSend} disabled={!landingInput.trim() && selectedImages.length === 0} />
                        </div>
                        {/* Pending 关系标签（中文注释） */}
                        {pendingRelations.length > 0 && (
                          <div className={styles.relationTags} style={{ marginTop: '8px' }}>
                            {pendingRelations.map(({ infon }, idx) => {
                              const relatedInfons = getRelatedInfons(infon, pendingInfonIndex)
                              const color = getInfonColor('REL')
                              
                              return (
                                <div key={idx} className={styles.relationTag} style={{ borderColor: color }}>
                                  <span className={styles.relationTagName} style={{ color: color }}>
                                    {infon.relation_name || 'Relation'}
                                  </span>
                                  <span className={styles.relationTagArgs}>
                                    {relatedInfons.map((rel, ri) => {
                                      const relColor = getInfonColor(rel.infon_type)
                                      const keywords = getMatchKeywords(rel)
                                      const label = keywords[0] || rel.iid
                                      return (
                                        <React.Fragment key={ri}>
                                          {ri > 0 && <span className={styles.relationTagSep}>→</span>}
                                          <span className={styles.relationTagArg} style={{ color: relColor }}>
                                            {label}
                                          </span>
                                        </React.Fragment>
                                      )
                                    })}
                                  </span>
                                </div>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* 底部输入条（中文注释）：固定于左侧面板底部 */}
              {(currentSession && (currentSession.messages || []).length > 0) && (
                <div className={styles.composerDock}>
                  <div className={styles.composer}>
                    {/* 隐藏的图片选择器（中文注释）：通过按钮触发 */}
                    <input
                      id="image-picker"
                      type="file"
                      accept="image/*"
                      multiple
                      style={{ display: 'none' }}
                      onChange={handlePickImages}
                    />
                    {/* 预览总在输入框上方（中文注释） */}
                    {selectedImages.length > 0 && (
                      <div className={styles.composerPreviews}>
                        {selectedImages.map((src, i) => (
                          <div key={i} className={styles.composerPreviewItem}>
                            <img 
                              src={src} 
                              alt={`preview-${i}`} 
                              className={styles.composerPreviewImg} 
                              onClick={() => setPreviewImage(src)}
                              style={{ cursor: 'pointer' }}
                            />
                            <button className={styles.composerPreviewRemove} onClick={(e) => { e.stopPropagation(); removeSelectedImage(i); }}>✕</button>
                          </div>
                        ))}
                      </div>
                    )}
                    <div className={styles.composerRow}>
                      <HighlightInput
                        className={styles.composerInput}
                        placeholder="Message ChatGPT"
                        value={input}
                        onChange={setInput}
                        onPressEnter={handleSend}
                        highlights={pendingHighlights}
                        autoSize={{ minRows: 1, maxRows: 6 }}
                      />
                      <div className={styles.composerButtons}>
                        <Upload
                          disabled={!currentModelIsMultimodal}
                          multiple
                          accept="image/*"
                          showUploadList={false}
                          beforeUpload={(file) => {
                            const reader = new FileReader()
                            reader.onload = () => setSelectedImages((prev) => [...prev, reader.result])
                            reader.readAsDataURL(file)
                            return Upload.LIST_IGNORE
                          }}
                        >
                          <Button icon={<CameraOutlined />} disabled={!currentModelIsMultimodal} title={currentModelIsMultimodal ? '' : 'Current model does not support images'} />
                        </Upload>
                        {!isGenerating ? (
                          <Button type="primary" icon={<SendOutlined />} disabled={!input.trim() && selectedImages.length === 0} onClick={handleSend} />
                        ) : (
                          <Button danger icon={<StopOutlined />} onClick={stopGenerating}>Stop</Button>
                        )}
                      </div>
                    </div>
                    {/* Pending 关系标签（中文注释） */}
                    {pendingRelations.length > 0 && (
                      <div className={styles.relationTags} style={{ marginTop: '8px' }}>
                        {pendingRelations.map(({ infon }, idx) => {
                          const relatedInfons = getRelatedInfons(infon, pendingInfonIndex)
                          const color = getInfonColor('REL')
                          
                          return (
                            <div key={idx} className={styles.relationTag} style={{ borderColor: color }}>
                              <span className={styles.relationTagName} style={{ color: color }}>
                                {infon.relation_name || 'Relation'}
                              </span>
                              <span className={styles.relationTagArgs}>
                                {relatedInfons.map((rel, ri) => {
                                  const relColor = getInfonColor(rel.infon_type)
                                  const keywords = getMatchKeywords(rel)
                                  const label = keywords[0] || rel.iid
                                  return (
                                    <React.Fragment key={ri}>
                                      {ri > 0 && <span className={styles.relationTagSep}>→</span>}
                                      <span className={styles.relationTagArg} style={{ color: relColor }}>
                                        {label}
                                      </span>
                                    </React.Fragment>
                                  )
                                })}
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </div>
                  <div className={styles.disclaimer}>Model streams responses. Context comes from this chat history.</div>
                </div>
              )}
            </Splitter.Panel>
            <Splitter.Panel defaultSize="35%" min="25%" max="50%">
              <div className={styles.rightPaneScroll}>
                <div className={styles.rightPaneHeader}>
                  <div className={styles.rightPaneTitle}>Privacy inference</div>
                </div>
                <div className={styles.rightPaneBody}>
                  {/* 法规 treemap 可视化（中文注释） */}
                  <LawTree />
                  {/* 隐私风险分析组件（中文注释） */}
                  <PrivacyRiskAnalysis
                    inference={inference}
                    selectedLaw={selectedLaw}
                  />
                  {/* 时间线组件（中文注释）：用于按时间筛选信息元 */}
                  <Timeline onTimeSelect={setSelectedTime} />
                  {/* 信息元词云可视化（中文注释） */}
                  <WordCloud selectedTime={selectedTime} />
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)', marginBottom: 8, paddingLeft: 4 }}>
                      Infons Results
                    </div>
                    <div className={styles.infonRuns}>
                      {(() => {
                        const runs = (infonSessions?.[currentSession?.id]?.runs) || []
                        if (!runs.length) return <div className={styles.infonEmpty}>No infons yet</div>
                      const sorted = [...runs].sort((a, b) => b.createdAt - a.createdAt)
                      return sorted.map((r) => {
                        const title = r.modality === 'text' ? 'Text' : `Image${Number.isFinite(r.imageIndex) ? ` #${r.imageIndex + 1}` : ''}`
                        const status = r.status
                        const percent = status === 'done' ? 100 : (status === 'running' ? 66 : (status === 'error' ? 0 : 0))
                        const allInfons = Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons : []
                        return (
                          <div key={r.id} className={styles.infonRunCard}>
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
                              <div className={styles.infonRunMeta}>{r.targetType}</div>
                            </div>
                            {status === 'error' && r.error ? (
                              <div className={styles.infonError}>{r.error}</div>
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
                              <pre className={styles.infonJsonCode}>{r.buffer || ''}</pre>
                            </details>
                          </div>
                        )
                        })
                      })()}
                    </div>
                  </div>
                </div>
              </div>
            </Splitter.Panel>
          </Splitter>
        </div>
      </section>

      {/* 图片预览 Modal（中文注释） */}
      <Modal
        open={!!previewImage}
        onCancel={() => setPreviewImage(null)}
        footer={null}
        width="90vw"
        centered
        className={styles.imagePreviewModal}
      >
        {previewImage && (
          <div className={styles.imagePreviewContainer}>
            <img src={previewImage} alt="Preview" className={styles.imagePreviewImg} />
          </div>
        )}
      </Modal>
    </div>
  )
}