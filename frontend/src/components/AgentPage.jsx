import React, { useEffect, useRef, useState, useMemo } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'
import MarkdownMessage from './MarkdownMessage'
import { Splitter, Select, Button, Upload, Progress, Input, Modal, message, Checkbox } from 'antd'
import { SendOutlined, StopOutlined, CameraOutlined } from '@ant-design/icons'


export default function AgentPage() {
  const {
    baseUrl,
    model,
    models,
    runPrivacyInference,
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
  } = useStore()

  // 当前会话对象（中文注释）：需在引用它的 useMemo 之前定义
  const currentSession = getCurrentSession()

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
  const [inferenceModel, setInferenceModel] = useState('')
  const [inferenceLoading, setInferenceLoading] = useState(false)
  const [inferenceError, setInferenceError] = useState('')
  const [inferenceResult, setInferenceResult] = useState(null)
  // 属性级展示不再需要画像索引（中文注释）
  const [apiModalOpen, setApiModalOpen] = useState(false)
  const [apiModelId, setApiModelId] = useState('')
  const [apiBaseUrl, setApiBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')

  // 隐私标签（中文注释）：固定列表，UI 文本使用英文；中文含义见注释
  const PRIVACY_TAGS = useMemo(() => [
    // 基础生理/外观（中文注释）
    { value: 'skin_color', label: 'Skin Color' }, // 皮肤颜色
    { value: 'gender', label: 'Gender' }, // 性别
    { value: 'hair_color', label: 'Hair Color' }, // 头发颜色
    { value: 'race', label: 'Race' }, // 种族
    { value: 'age_group', label: 'Age Group' }, // 年龄段
    { value: 'safe', label: 'Safe' }, // 安全
    { value: 'face_c', label: 'Face (C)' }, // 脸部（C）
    { value: 'face_p', label: 'Face (P)' }, // 脸部（P）
    { value: 'eye_color', label: 'Eye Color' }, // 眼睛颜色
    { value: 'weight_group', label: 'Weight Group' }, // 体重组
    { value: 'height_group', label: 'Height Group' }, // 身高组
    { value: 'culture', label: 'Culture' }, // 文化
    { value: 'landmark', label: 'Landmark' }, // 地标
    { value: 'occupation', label: 'Occupation' }, // 职业
    { value: 'visited_location_p', label: 'Visited Location (P)' }, // 访问地点（P）
    { value: 'datetime_of_activity', label: 'Date/Time of Activity' }, // 活动日期/时间
    { value: 'social_circle', label: 'Social Circle' }, // 社交圈
    { value: 'spectators', label: 'Spectators' }, // 观众
    { value: 'professional_circle', label: 'Professional Circle' }, // 专业圈
    { value: 'semi_nudity', label: 'Semi-nudity' }, // 半裸
    { value: 'full_name', label: 'Full Name' }, // 全名
    { value: 'similar_view', label: 'Similar view' }, // 相似视角
    { value: 'sports', label: 'Sports' }, // 运动
    { value: 'general_opinion', label: 'General Opinion' }, // 总体意见
    { value: 'personal_relationships', label: 'Personal Relationships' }, // 人际关系
    { value: 'work_occasion', label: 'Work Occasion' }, // 工作场合
    { value: 'tickets', label: 'Tickets' }, // 票
    { value: 'handwriting', label: 'Handwriting' }, // 字迹
    { value: 'sexual_orientation', label: 'Sexual Orientation' }, // 性取向
    { value: 'tattoo', label: 'Tattoo' }, // 刺青
    { value: 'hobbies', label: 'Hobbies' }, // 爱好
    { value: 'license_plate_c', label: 'License Plate (C)' }, // 车牌（C）
    { value: 'traditional_clothing', label: 'Traditional clothing' }, // 传统服饰
    { value: 'medical_treatment', label: 'Medical Treatment' }, // 医疗治疗
    { value: 'competitors', label: 'Competitors' }, // 竞争对手
    { value: 'signature', label: 'Signature' }, // 签名
    { value: 'religion', label: 'Religion' }, // 宗教
    { value: 'passport', label: 'Passport' }, // 护照
    { value: 'receipts', label: 'Receipts' }, // 发票
    { value: 'first_name', label: 'First Name' }, // 名字
    { value: 'last_name', label: 'Last Name' }, // 姓氏
    { value: 'nationality', label: 'Nationality' }, // 国籍
    { value: 'personal_occasion', label: 'Personal Occasion' }, // 个人场合
    { value: 'physical_disability', label: 'Physical disability' }, // 身体残疾
    { value: 'license_plate_p', label: 'License Plate (P)' }, // 车牌（P）
    { value: 'date_of_birth', label: 'Date of Birth' }, // 出生日期
    { value: 'vehicle_ownership', label: 'Vehicle Ownership' }, // 车辆拥有
    { value: 'mail', label: 'Mail' }, // 邮件
    { value: 'phone_number', label: 'Phone no.' }, // 电话号码
    { value: 'visited_location_c', label: 'Visited Location (C)' }, // 访问地点（C）
    { value: 'education_history', label: 'Education history' }, // 教育背景
    { value: 'online_conversations', label: 'Online conversations' }, // 在线对话
    { value: 'username', label: 'Username' }, // 用户名
    { value: 'marital_status', label: 'Marital status' }, // 婚姻状况
    { value: 'home_address_c', label: 'Home address (C)' }, // 家庭地址（C）
    { value: 'political_opinion', label: 'Political Opinion' }, // 政治观点
    { value: 'credit_card', label: 'Credit card' }, // 信用卡
    { value: 'email_address', label: 'Email address' }, // 电子邮件地址
    { value: 'student_license', label: 'Student License' }, // 学生证
    { value: 'drivers_license', label: 'Drivers License' }, // 驾照
    { value: 'medical_history', label: 'Medical History' }, // 医疗历史
    { value: 'place_of_birth', label: 'Place of Birth' }, // 出生地
    { value: 'email_content', label: 'Email content' }, // 电子邮件内容
    { value: 'legal_involvement', label: 'Legal involvement' }, // 法律参与
    { value: 'nudity', label: 'Nudity' }, // 裸露
    { value: 'fingerprint', label: 'Fingerprint' }, // 指纹
    { value: 'national_identification', label: 'National Identification' }, // 国家身份证
    { value: 'home_address_p', label: 'Home address (P)' }, // 家庭地址（P）
  ], [])

  // 复选框选中项（中文注释）：默认全不选，常显
  const [selectedPrivacyTags, setSelectedPrivacyTags] = useState([])

  // 默认注册 DeepSeek 示例（中文注释）：仅添加一次，已存在则跳过
  useEffect(() => {
    try {
      useStore.getState().addApiModel?.({ id: 'deepseek-chat', baseUrl: 'https://api.deepseek.com/v1', apiKey: 'sk-8c2ee9474f2f44f5969dcd5de280e634' })
    } catch (_) {}
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
    } catch (_) {}
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

  // 属性级展示（中文注释）：不需要默认选择

  // 右侧推断模型默认值与主模型解耦（中文注释）：优先选择 gemma3:12b，若不存在则回退到本地第一个
  useEffect(() => {
    try {
      const locals = (models || []).filter((id) => !customProviders?.[id])
      if (!inferenceModel) {
        const preferred = 'gemma3:12b'
        if (locals.includes(preferred)) setInferenceModel(preferred)
        else if (locals.length) setInferenceModel(locals[0])
        else setInferenceModel(preferred)
      }
    } catch (_) {}
  }, [models, customProviders, inferenceModel])

  // 当上下文或 pending 存在图片时，强制右侧使用多模态模型（中文注释）
  useEffect(() => {
    try {
      const locals = (models || []).filter((id) => !customProviders?.[id])
      const hasPendingImages = selectedImages.length > 0
      const needMultimodal = Boolean(contextHasImages || hasPendingImages)
      if (!needMultimodal) return
      if (inferenceModel && isModelMultimodal(inferenceModel)) return
      const preferred = 'gemma3:12b'
      const mmList = locals.filter((id) => isModelMultimodal(id))
      if (mmList.includes(preferred)) setInferenceModel(preferred)
      else if (mmList.length) setInferenceModel(mmList[0])
    } catch (_) {}
  }, [models, customProviders, inferenceModel, contextHasImages, selectedImages])

  // 证据网络图（中文注释）：以最近消息为节点，依据 evidence_refs 共现构建边；宽度自适应
  const EvidenceGraph = ({ inferenceResult, messages, pendingText, pendingHas, pendingTooltip }) => {
    const wrapRef = useRef(null)
    const [size, setSize] = useState({ w: 300, h: 200 })
    useEffect(() => {
      if (!wrapRef.current) return
      const ro = new ResizeObserver((entries) => {
        for (const e of entries) {
          const cr = e.contentRect
          const w = Math.max(260, cr.width)
          const h = Math.min(420, Math.max(180, 60 + Math.min(30, messages.length) * 10))
          setSize({ w, h })
        }
      })
      ro.observe(wrapRef.current)
      return () => { try { ro.disconnect() } catch (_) {} }
    }, [messages?.length])

    try {
      const ev = Array.isArray(inferenceResult?.evidence) ? inferenceResult.evidence : []
      const evMap = new Map(ev.map((e) => [e?.id, e]).filter(([k,v]) => !!k))
      const infoUnits = Array.isArray(inferenceResult?.info_units) ? inferenceResult.info_units : []
      const attributes = Array.isArray(inferenceResult?.attributes) ? inferenceResult.attributes.filter(a => (a?.best?.value !== 'N/A')) : []
      const edgesAll = Array.isArray(inferenceResult?.graph?.edges) ? inferenceResult.graph.edges : []
      const graphByAttribute = Array.isArray(inferenceResult?.graph_by_attribute) ? inferenceResult.graph_by_attribute : []

      const recent = (messages || []).slice(-30)
      // 层次布局（中文注释）：父层=消息节点；中层=信息元；末层=画像
      let layerMsg = recent.map((m, idx) => ({ id: `msg:${idx}`, idx }))
      const hasPending = !!pendingHas
      if (hasPending) layerMsg = [...layerMsg, { id: 'pending', idx: 'pending' }]
      const layerInfo = infoUnits.map((u, idx) => ({ id: u.id || `info:${idx}`, label: String(u.label || ''), from: Array.isArray(u.from_messages) ? u.from_messages : [] }))
      const layerAttributes = attributes.map((a) => ({ id: `attr:${String(a?.name || '')}`, name: String(a?.name || '') }))

      const pad = 24
      const extraColSpacing = 28
      const W = size.w
      const colW = Math.max(240, Math.floor((W - pad*4) / 3))
      const colX = [pad, pad + colW + pad + extraColSpacing, pad + (colW + pad)*2 + extraColSpacing*2]
      const gapY = 36

      // 根据节点数量动态增加高度，避免垂直拥挤（中文注释）
      const maxColCount = Math.max(layerMsg.length, layerInfo.length, layerAttributes.length)
      const desiredH = Math.min(560, Math.max(240, pad*2 + Math.max(1, maxColCount) * gapY))
      const H = Math.max(size.h, desiredH)

      const columnHeights = [layerMsg.length, layerInfo.length, layerAttributes.length].map(cnt => Math.max(1, cnt) * gapY)
      const topOffsets = columnHeights.map(h => Math.max(pad, Math.floor((H - h) / 2)))

      const msgPos = layerMsg.map((m, i) => ({ x: colX[0], y: topOffsets[0] + i * gapY }))
      const infoPos = layerInfo.map((n, i) => ({ x: colX[1], y: topOffsets[1] + i * gapY }))
      const attrPos = layerAttributes.map((n, i) => ({ x: colX[2], y: topOffsets[2] + i * gapY }))

      // 连接：msg->info 来自 info_units.from_messages，提供多重回退（中文注释）
      const msgToInfo = []
      layerInfo.forEach((inf, infIdx) => {
        let linked = false
        // 1) 直接使用 from_messages（期望是全局索引，Recent 列表内需换算）
        for (const mi of (inf.from || [])) {
          if (Number.isFinite(mi)) {
            const globalToRecent = (messages || []).length - recent.length
            const local = mi - globalToRecent
            if (local >= 0 && local < recent.length) {
              msgToInfo.push({ si: local, ti: infIdx, w: 1 })
              linked = true
            }
          }
        }
        // 2) 依据 evidence_refs 回退
        if (!linked) {
          const refs = Array.isArray(infoUnits[infIdx]?.evidence_refs) ? infoUnits[infIdx].evidence_refs : []
          for (const rid of refs) {
            const e = evMap.get(rid)
            if (e && Number.isFinite(e.message_index)) {
              const globalToRecent = (messages || []).length - recent.length
              const local = e.message_index - globalToRecent
              if (local >= 0 && local < recent.length) {
                msgToInfo.push({ si: local, ti: infIdx, w: 1 })
                linked = true
                break
              }
            }
          }
        }
        // 3) 文本相似度极简回退：用 label 在 recent 内容中模糊匹配
        if (!linked && inf.label) {
          const label = String(inf.label).toLowerCase()
          let bestIdx = -1
          let bestScore = 0
          recent.forEach((m, i) => {
            const t = String(m?.content || '').toLowerCase()
            const score = (label && t.includes(label)) ? Math.min(1, Math.max(0.2, Math.sqrt(label.length / Math.max(8, t.length)))) : 0
            if (score > bestScore) { bestScore = score; bestIdx = i }
          })
          if (bestIdx >= 0) {
            msgToInfo.push({ si: bestIdx, ti: infIdx, w: 1 })
            linked = true
          }
        }
        // 4) 仅有 pending 时兜底：强制连接到 pending（权重最低）（中文注释）
        if (!linked && hasPending) {
          const pendingIndex = layerMsg.length - 1
          msgToInfo.push({ si: pendingIndex, ti: infIdx, w: 1 })
          linked = true
        }
      })
      // 追加 pending->info（中文注释）：依据 info_units.evidence_refs 命中 source=pending
      if (hasPending) {
        const pendingIndex = layerMsg.length - 1
        layerInfo.forEach((inf, infIdx) => {
          const refs = Array.isArray(infoUnits[infIdx]?.evidence_refs) ? infoUnits[infIdx].evidence_refs : []
          const hit = refs.some((rid) => evMap.get(rid)?.source === 'pending')
          if (hit) msgToInfo.push({ si: pendingIndex, ti: infIdx, w: 1 })
        })
      }
      const infoToAttribute = []
      const clamp16 = (v) => Math.max(1, Math.min(6, Number(v) || 1))
      const addEdge = (si, ti, w) => {
        if (si < 0 || ti < 0) return
        if (!infoToAttribute.some(x => x.si === si && x.ti === ti)) {
          infoToAttribute.push({ si, ti, w: clamp16(w) })
        }
      }
      const infoIndexById = new Map(layerInfo.map((n, i) => [n.id, i]))
      const attrIndexByName = new Map(layerAttributes.map((n, i) => [n.name, i]))

      // 1) 依据 graph.edges 中的 info_to_attribute
      edgesAll.forEach((e) => {
        if (e?.type === 'info_to_attribute') {
          const si = infoIndexById.has(e.source) ? infoIndexById.get(e.source) : layerInfo.findIndex((n) => n.id === e.source)
          let ti = -1
          if (attrIndexByName.has(e.target)) ti = attrIndexByName.get(e.target)
          else {
            ti = layerAttributes.findIndex((n) => n.id === e.target || n.name === e.target || n.id === `attr:${e.target}`)
          }
          if (si >= 0 && ti >= 0) addEdge(si, ti, e?.weight ?? 1)
        }
      })

      // 1.1) 解析包含 attributes 数组的边（当 e.type 未标注为 info_to_attribute 时）
      edgesAll.forEach((e) => {
        if (Array.isArray(e?.attributes) && e.attributes.length) {
          const si = infoIndexById.has(e.source) ? infoIndexById.get(e.source) : layerInfo.findIndex((n) => n.id === e.source)
          e.attributes.forEach((attrName) => {
            const ti = attrIndexByName.has(attrName) ? attrIndexByName.get(attrName) : layerAttributes.findIndex((n) => n.name === attrName)
            if (si >= 0 && ti >= 0) addEdge(si, ti, e?.weight ?? 1)
          })
        }
      })

      // 2) graph_by_attribute 指定的边
      graphByAttribute.forEach((ga) => {
        const ti = attrIndexByName.has(ga?.name) ? attrIndexByName.get(ga.name) : layerAttributes.findIndex((n) => n.name === ga?.name)
        if (ti < 0) return
        const edges = Array.isArray(ga?.edges) ? ga.edges : []
        edges.forEach((e) => {
          const si = infoIndexById.has(e.source) ? infoIndexById.get(e.source) : layerInfo.findIndex((n) => n.id === e.source)
          if (si >= 0) addEdge(si, ti, e?.weight ?? 1)
        })
      })

      // 3) 依据 attributes[].info_units 生成连线
      attributes.forEach((a, ai) => {
        const uids = Array.isArray(a?.info_units) ? a.info_units : []
        uids.forEach((uid) => {
          const si = infoIndexById.has(uid) ? infoIndexById.get(uid) : layerInfo.findIndex((n) => n.id === uid)
          const ti = ai
          if (si >= 0 && ti >= 0) addEdge(si, ti, (a?.exposure_contribution ?? 1))
        })
      })

      // 4) 文本匹配兜底：info.label 与 attribute 的 best/top_predictions rationale 共现
      attributes.forEach((a, ai) => {
        const texts = []
        if (a?.best?.rationale) texts.push(String(a.best.rationale).toLowerCase())
        if (Array.isArray(a?.top_predictions)) {
          a.top_predictions.forEach(tp => { if (tp?.rationale) texts.push(String(tp.rationale).toLowerCase()) })
        }
        if (!texts.length) return
        layerInfo.forEach((inf, si) => {
          const k = String(inf.label || '').toLowerCase()
          if (!k) return
          const hit = texts.some(t => t.includes(k))
          if (hit) {
            const weight = (a?.exposure_contribution ?? 1) + 1.5
            addEdge(si, ai, weight)
          }
        })
      })

      // 5) 最终兜底：确保每个属性至少有一条入边，选取贡献度最高的前1-2个信息元
      attributes.forEach((a, ai) => {
        const hasAny = infoToAttribute.some(e => e.ti === ai)
        if (!hasAny && layerInfo.length) {
          const pairs = layerInfo.map((inf, si) => ({ si, c: Number(inf?.exposure_contribution) || 1 }))
          pairs.sort((p,q) => q.c - p.c)
          pairs.slice(0, Math.min(2, pairs.length)).forEach(p => addEdge(p.si, ai, (a?.exposure_contribution ?? 1)))
        }
      })

      // 信息元间连接（中文注释）：依据 edges 中 type=info_to_info
      const infoToInfo = []
      edgesAll.forEach((e) => {
        if (e?.type === 'info_to_info') {
          const si = layerInfo.findIndex((n) => n.id === e.source)
          const ti = layerInfo.findIndex((n) => n.id === e.target)
          if (si >= 0 && ti >= 0) {
            infoToInfo.push({ si, ti, w: Math.max(1, Math.min(6, Number(e?.weight) || 1)) })
          }
        }
      })

      // 风险高亮：依据 privacy_risks.contributors 与 trigger_combination（中文注释）
      const riskyInfo = new Set()
      try {
        const risks = Array.isArray(inferenceResult?.privacy_risks) ? inferenceResult.privacy_risks : []
        for (const r of risks) {
          for (const id of (Array.isArray(r?.contributors) ? r.contributors : [])) {
            const idx = layerInfo.findIndex((n) => n.id === id)
            if (idx >= 0) riskyInfo.add(idx)
          }
        }
      } catch (_) {}

      // 权重映射（中文注释）：线宽与透明度
      // 加强连线粗细对比度（中文注释）：使用对数缩放映射
      const wAll = [...infoToAttribute.map(e=>e.w), ...infoToInfo.map(e=>e.w), ...msgToInfo.map(e=>e.w)]
      const wMin = wAll.length ? Math.min(...wAll) : 1
      const wMax = wAll.length ? Math.max(...wAll) : 1
      const scaleOpacity = (w) => {
        if (wMax === wMin) return 0.7
        const n = (Math.log(1 + (w - wMin)) / Math.log(1 + (wMax - wMin)))
        return 0.35 + 0.65 * n
      }
      const scaleStroke = (w) => {
        if (wMax === wMin) return 1.4
        const n = (Math.log(1 + (w - wMin)) / Math.log(1 + (wMax - wMin)))
        return 0.8 + 4.2 * n
      }

      // 曲线函数（中文注释）：三次贝塞尔，加入水平弯曲与垂直偏移以降低重叠
      const curve = (x1,y1,x2,y2,k=0) => {
        const dx = Math.max(24, Math.min(160, (x2 - x1) * 0.45))
        const oy = k
        return `M ${x1},${y1} C ${x1+dx},${y1+oy} ${x2-dx},${y2-oy} ${x2},${y2}`
      }

      return (
        <div ref={wrapRef} style={{ width: '100%' }}>
          <svg width={W} height={H} style={{ border: '1px solid #e5e7eb', borderRadius: 8, background: '#ffffff' }}>
            <defs>
              <marker id="arrow-dark" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#334155" /></marker>
              <marker id="arrow-gray" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" /></marker>
              <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="0" dy="1" stdDeviation="1" floodColor="#0f172a" floodOpacity="0.15"/></filter>
            </defs>

            {(() => {
              // 分组给同一目标的边增加上下错位，减少重叠（中文注释）
              const groups = new Map()
              msgToInfo.forEach((e) => { const k = `to:${e.ti}`; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(e) })
              const paths = []
              groups.forEach((arr) => {
                const n = arr.length
                arr.forEach((e, idx) => {
                  const offset = (idx - (n-1)/2) * 6
                  paths.push(
                    <path key={`m2i-${e.si}-${e.ti}-${idx}`} d={curve(msgPos[e.si].x+16, msgPos[e.si].y, infoPos[e.ti].x-46, infoPos[e.ti].y, offset)} stroke="#94a3b8" strokeWidth={scaleStroke(e.w)} opacity={scaleOpacity(e.w)} fill="none" markerEnd="url(#arrow-gray)">
                      <title>{`message -> ${layerInfo[e.ti]?.label || 'info'}`}</title>
                    </path>
                  )
                })
              })
              return paths
            })()}
            {(() => {
              const groups = new Map()
              infoToInfo.forEach((e) => { const k = `${e.si}->${e.ti}`; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(e) })
              const paths = []
              groups.forEach((arr, key) => {
                const n = arr.length
                arr.forEach((e, idx) => {
                  const offset = (idx - (n-1)/2) * 6
                  paths.push(
                    <path key={`i2i-${key}-${idx}`} d={curve(infoPos[e.si].x+46, infoPos[e.si].y, infoPos[e.ti].x-46, infoPos[e.ti].y, offset)} stroke="#64748b" strokeWidth={scaleStroke(e.w)} opacity={scaleOpacity(e.w)} fill="none" markerEnd="url(#arrow-dark)">
                      <title>{`${layerInfo[e.si]?.label || 'info'} -> ${layerInfo[e.ti]?.label || 'info'} (w=${e.w})`}</title>
                    </path>
                  )
                })
              })
              return paths
            })()}
            {(() => {
              const groups = new Map()
              infoToAttribute.forEach((e) => { const k = `toAttr:${e.ti}`; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(e) })
              const paths = []
              groups.forEach((arr) => {
                const n = arr.length
                arr.forEach((e, idx) => {
                  const offset = (idx - (n-1)/2) * 6
                  paths.push(
                    <path key={`i2a-${e.si}-${e.ti}-${idx}`} d={curve(infoPos[e.si].x+46, infoPos[e.si].y, attrPos[e.ti].x-16, attrPos[e.ti].y, offset)} stroke="#334155" strokeWidth={scaleStroke(e.w)} opacity={scaleOpacity(e.w)} fill="none" markerEnd="url(#arrow-dark)">
                      <title>{`${layerInfo[e.si]?.label || 'info'} -> ${layerAttributes[e.ti]?.name || 'attribute'} (w=${e.w})`}</title>
                    </path>
                  )
                })
              })
              return paths
            })()}
            {layerMsg.map((m, i) => {
              const isPendingNode = m.id === 'pending'
              const fill = isPendingNode ? '#f59e0b' : '#0ea5e9'
              const titleText = isPendingNode ? (pendingTooltip || '') : String((messages[i]?.content || '')).slice(0, 400)
              const roleText = isPendingNode ? 'P' : (messages[i]?.role === 'user' ? 'U' : 'A')
              return (
                <g key={m.id} filter="url(#shadow)">
                  <circle cx={msgPos[i].x} cy={msgPos[i].y} r={8} fill={fill}>
                    <title>{titleText}</title>
                  </circle>
                  <text x={msgPos[i].x + 12} y={msgPos[i].y + 4} fontSize={10} fill="#334155">{roleText}</text>
                </g>
              )
            })}
            {layerInfo.map((n, i) => {
              const label = String(n.label || 'info')
              // 根据文本长度自适应卡片宽度：最小 120，最大 220（中文注释）
              const w = Math.max(120, Math.min(220, 12 * Math.ceil(label.length * 0.9)))
              return (
                <g key={n.id} filter="url(#shadow)">
                  <rect x={infoPos[i].x - w/2} y={infoPos[i].y - 16} width={w} height={32} rx={8} ry={8} fill={riskyInfo.has(i) ? '#fee2e2' : '#e6f4f1'} stroke={riskyInfo.has(i) ? '#ef4444' : '#10a37f'} />
                  <text x={infoPos[i].x} y={infoPos[i].y} fontSize={11} fill="#334155" textAnchor="middle" dominantBaseline="middle">{label}</text>
                  <title>{`info: ${label}`}</title>
                </g>
              )
            })}
            {layerAttributes.map((n, i) => (
              <g key={n.id} filter="url(#shadow)">
                <circle cx={attrPos[i].x} cy={attrPos[i].y} r={10} fill={'#10a37f'} />
                <text x={attrPos[i].x + 12} y={attrPos[i].y + 4} fontSize={10} fill="#334155">{n.name || 'attr'}</text>
                <title>{`attribute: ${n.name || ''}`}</title>
              </g>
            ))}
          </svg>
        </div>
      )
    } catch (_) {
      return null
    }
  }

  const handleSend = async () => {
    const text = (input || '').trim()
    const hasImages = selectedImages.length > 0
    if (!text && !hasImages) return
    setInput('')
    if (hasImages) {
      const imgs = [...selectedImages]
      setSelectedImages([])
      await useStore.getState().sendMessageWithImages(text, imgs)
    } else {
      await sendMessage(text)
    }
  }

  const handleLandingSend = async () => {
    const text = (landingInput || '').trim()
    const hasImages = selectedImages.length > 0
    if (!text && !hasImages) return
    setLandingInput('')
    if (hasImages) {
      const imgs = [...selectedImages]
      setSelectedImages([])
      await useStore.getState().sendMessageWithImages(text, imgs)
    } else {
      await sendMessage(text)
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

  // 渲染 pending 卡片（中文注释）：根据推断结果标红
  const renderPendingCard = () => {
    let hasPendingRisk = false
    try {
      const evidence = Array.isArray(inferenceResult?.evidence) ? inferenceResult.evidence : []
      const pendingIds = new Set(evidence.filter((e) => e?.source === 'pending').map((e) => e?.id).filter(Boolean))
      const edges = Array.isArray(inferenceResult?.graph?.edges) ? inferenceResult.graph.edges : []
      hasPendingRisk = edges.some((e) => (Array.isArray(e?.evidence_refs) ? e.evidence_refs : []).some((r) => pendingIds.has(r)))
    } catch (_) {}
    return (
      <div style={{ border: `1px solid ${hasPendingRisk ? '#ef4444' : '#e5e7eb'}`, borderRadius: 8, padding: 8, background: hasPendingRisk ? '#fff1f1' : '#ffffff' }}>
        <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>pending</div>
        {selectedImages.length > 0 ? (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {selectedImages.map((src, k) => (
              <img key={k} src={src} alt={`pending-img-${k}`} style={{ width: 64, height: 64, objectFit: 'cover', borderRadius: 6, border: '1px solid #e5e7eb' }} />
            ))}
          </div>
        ) : null}
        <div style={{ fontSize: 13, color: '#0f172a', whiteSpace: 'pre-wrap' }}>
          {((input || landingInput || '')).slice(0, 200)}{((input || landingInput || '')).length > 200 ? '…' : ''}
        </div>
      </div>
    )
  }

  return (
    <div className={styles.shell}>
      {/* 左侧：侧边栏 */}
      <aside className={styles.sidebar}>
        <div className={styles.sidebarTop}>
          <button className={styles.newBtn} onClick={createSession}>New chat</button>
        </div>
        <div className={styles.sidebarScroll}>
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`${styles.chatItem} ${s.id === currentSessionId ? styles.chatItemActive : ''}`}
              onClick={() => switchSession(s.id)}
              title={s.title}
            >
              <div className={styles.chatName}>{s.title}</div>
              <div className={styles.chatMeta}>{new Date(s.updatedAt).toLocaleString()}</div>
              <div className={styles.chatActions}>
                <button className={styles.iconBtn} onClick={(e) => { e.stopPropagation(); const t = prompt('Rename'); if (t) renameSession(s.id, t) }}>✎</button>
                <button className={styles.iconBtn} onClick={(e) => { e.stopPropagation(); if (confirm('Delete this chat?')) deleteSession(s.id) }}>🗑</button>
              </div>
            </div>
          ))}
        </div>
        <div className={styles.sidebarBottom}>
          {/* <div className={styles.kv}><span>Base URL</span><span>{baseUrl}</span></div> */}
          <div className={styles.kv}><span>Model</span><span>{model}</span></div>
        </div>
      </aside>

      {/* 右侧：主区域 */}
      <section className={styles.main}>
        <div className={styles.scroll} ref={listRef}>
          {/* 顶部：左上角模型选择器，右侧保留空白对齐 */}
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
              } catch (_) {}
            }}
          >
            <div style={{ display: 'grid', gap: 8 }}>
              <Input placeholder="Model ID" value={apiModelId} onChange={(e) => setApiModelId(e.target.value)} />
              <Input placeholder="Base URL" value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} />
              <Input.Password placeholder="API Key" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
              <div style={{ fontSize: 12, color: '#64748b' }}>Note: Privacy inference requires local model.</div>
            </div>
          </Modal>
          <Splitter className={styles.splitterRoot}>
            <Splitter.Panel style={{ overflow: 'hidden' }}>
                <div className={styles.leftPaneScroll} ref={listRef}>
                  {hasMessages ? (
                    <div className={styles.column}>
                      {(currentSession?.messages || []).map((m) => {
                        const isUser = m.role === 'user'
                        return (
                          <div key={m.id} className={`${styles.msgRow} ${isUser ? styles.rowUser : styles.rowAssistant}`}>
                            {isUser ? (
                              <>
                                <div className={`${styles.msgBubble} ${styles.msgBubbleUser}`}>
                                  <div className={styles.msgContent}>{m.content}</div>
                                  {Array.isArray(m.images) && m.images.length > 0 && (
                                    <div className={styles.msgImages}>
                                      {m.images.map((src, i) => (
                                        <img key={i} src={src} alt={`img-${i}`} className={styles.msgImage} />
                                      ))}
                                    </div>
                                  )}
                                </div>
                                <div className={styles.avatar}>U</div>
                              </>
                            ) : (
                              <>
                                <div className={styles.avatar}>A</div>
                                <div className={`${styles.msgBubble} ${styles.msgBubbleAssistant}`}>
                                  {m.reasoning && (
                                    <div className={styles.reasoningBox}>
                                      <div className={styles.reasoningTitle}>Thinking</div>
                                      <div className={styles.reasoningBody}>
                                        <MarkdownMessage content={m.reasoning} />
                                      </div>
                                    </div>
                                  )}
                                  <div className={styles.msgContent}>
                                    <MarkdownMessage content={m.content} />
                                  </div>
                                  {m.streaming ? <div className={styles.cursor}>▍</div> : null}
                                  {m.error ? <div className={styles.error}>Error: {m.error}</div> : null}
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
                                <img src={src} alt={`preview-${i}`} className={styles.composerPreviewImg} />
                                <button className={styles.composerPreviewRemove} onClick={() => removeSelectedImage(i)}>✕</button>
                              </div>
                            ))}
                          </div>
                        )}
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
                          <Input.TextArea
                            className={styles.landingInput}
                            placeholder="Type your question..."
                            value={landingInput}
                            onChange={(e) => setLandingInput(e.target.value)}
                            onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); handleLandingSend() } }}
                            autoSize={{ minRows: 1, maxRows: 6 }}
                          />
                          <Button type="primary" icon={<SendOutlined />} onClick={handleLandingSend} disabled={!landingInput.trim() && selectedImages.length === 0} />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
            </Splitter.Panel>
            <Splitter.Panel defaultSize="28%" min="18%" max="45%">
                <div className={styles.rightPaneScroll}>
                  <div style={{ padding: 16 }}>
                    <div style={{ fontWeight: 600, marginBottom: 8 }}>Context window</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                      <Progress percent={contextPercent} size="small" style={{ flex: 1 }} />
                      <div style={{ color: '#64748b', fontSize: 12 }}>{contextLabel}</div>
                    </div>
                    <div style={{ color: '#475569', fontSize: 12, marginBottom: 8 }}>Evidence graph</div>
                    <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, background: '#ffffff' }}>
                      {inferenceResult ? (
                        <EvidenceGraph
                          inferenceResult={inferenceResult}
                          messages={(currentSession?.messages || [])}
                          pendingText={(input || landingInput || '')}
                          pendingHas={Boolean((input || '').trim() || (landingInput || '').trim() || selectedImages.length)}
                          pendingTooltip={`${(input || landingInput || '').slice(0, 180)}${((input || landingInput || '').length > 180) ? '…' : ''}${selectedImages.length ? `\n[${selectedImages.length} image(s) selected]` : ''}`}
                        />
                      ) : (
                        <div style={{ fontSize: 12, color: '#64748b' }}>Run inference to view the evidence graph.</div>
                      )}
                    </div>
                    {/* 实时显示未发送输入与已选图片（中文注释） */}
                    {(((input || '').trim().length > 0) || ((landingInput || '').trim().length > 0) || selectedImages.length > 0) && renderPendingCard()}
                    {/* 隐私推断控制区（中文注释）：单独选择模型 + 推断按钮 + 结果显示 */}
                    <div style={{ height: 12 }} />
                    <div style={{ borderTop: '1px solid #e5e7eb', margin: '12px 0' }} />
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                      <div style={{ fontWeight: 600 }}>Privacy inference</div>
                    </div>
                    {/* 隐私标签常显（中文注释）：自适应列，宽度占满，无滚动条 */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                      <div style={{ fontSize: 12, color: '#64748b' }}>Privacy tags (optional)</div>
                      <div style={{ fontSize: 12, color: '#64748b' }}>{selectedPrivacyTags.length} selected</div>
                    </div>
                    <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, background: '#ffffff', marginBottom: 8 }}>
                      <Checkbox.Group value={selectedPrivacyTags} onChange={setSelectedPrivacyTags}>
                        <div className={styles.privacyGrid}>
                          {PRIVACY_TAGS.map(opt => (
                            <Checkbox key={opt.value} value={opt.value} className={styles.privacyCheckbox}>{opt.label}</Checkbox>
                          ))}
                        </div>
                      </Checkbox.Group>
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                      {(() => {
                        const localModels = (models || [])
                          .filter((v, i, a) => v && a.indexOf(v) === i)
                          .filter((id) => !customProviders?.[id])
                        const requireMultimodal = Boolean(contextHasImages || (selectedImages.length > 0))
                        const currentVal = (inferenceModel && localModels.includes(inferenceModel))
                          ? inferenceModel
                          : (localModels[0] || '')
                        return (
                          <Select
                            style={{ minWidth: 220 }}
                            value={currentVal}
                            onChange={(v) => {
                              if (requireMultimodal && !isModelMultimodal(v)) {
                                message.warning('Cannot switch to a non-multimodal model when images exist in context or pending')
                                return
                              }
                              setInferenceModel(v)
                            }}
                            options={localModels.map((v) => ({ label: `${v}${isModelMultimodal(v) ? ' (multimodal)' : ' (text-only)'}`, value: v, disabled: (requireMultimodal && !isModelMultimodal(v)) }))}
                          />
                        )
                      })()}
                      <Button
                        type="primary"
                        loading={inferenceLoading}
                        onClick={async () => {
                          try {
                            setInferenceError('')
                            setInferenceLoading(true)
                            const pending = input || landingInput || ''
                            const localModels = (models || []).filter((id) => !customProviders?.[id])
                            const usedModel = (inferenceModel && localModels.includes(inferenceModel)) ? inferenceModel : (localModels[0] || '')
                            const requireMultimodal = Boolean(contextHasImages || (selectedImages.length > 0))
                            if (requireMultimodal && !isModelMultimodal(usedModel)) {
                              message.warning('A multimodal model is required when images exist in context or pending')
                              return
                            }
                            const result = await runPrivacyInference(pending, usedModel, selectedPrivacyTags)
                            setInferenceResult(result)
                          } catch (e) {
                            setInferenceResult(null)
                            setInferenceError(String(e?.message || e || 'Inference failed'))
                          } finally {
                            setInferenceLoading(false)
                          }
                        }}
                        disabled={(() => { const ls=(models||[]).filter((id)=>!customProviders?.[id]); const requireMultimodal=Boolean(contextHasImages||(selectedImages.length>0)); const used=(inferenceModel&&ls.includes(inferenceModel))?inferenceModel:(ls[0]||''); if(!used) return true; if(requireMultimodal && !isModelMultimodal(used)) return true; return false })()}
                      >Infer</Button>
                    </div>
                    {inferenceError ? (
                      <div style={{ color: '#b91c1c', fontSize: 12, marginBottom: 8 }}>Error: {inferenceError}</div>
                    ) : null}
                    {inferenceResult ? (
                      <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, background: '#ffffff' }}>
                        <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Attributes</div>
                        <div style={{ display: 'grid', gap: 6, marginBottom: 8 }}>
                          {(Array.isArray(inferenceResult?.attributes) ? inferenceResult.attributes : [])
                            .filter(a => a && a.best && a.best.value !== 'N/A')
                            .map((a, idx) => (
                              <div key={a?.name || idx} style={{ border: '2px solid #e5e7eb', borderRadius: 8, padding: 10, background: '#ffffff' }}>
                                <div style={{ fontSize: 13 }}>
                                  <b>{String(a?.name || '')}</b> · {String(a?.best?.value)}
                                </div>
                                <div style={{ fontSize: 12, color: '#334155' }}>confidence: {Number.isFinite(a?.best?.confidence) ? Number(a.best.confidence).toFixed(2) : '-'}</div>
                                {a?.best?.rationale ? <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>{String(a.best.rationale)}</div> : null}
                                {Array.isArray(a?.top_predictions) && a.top_predictions.length ? (
                                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 6 }}>
                                    top-3: {a.top_predictions.map((tp) => `${tp?.value} (${Number.isFinite(tp?.confidence) ? Number(tp.confidence).toFixed(2) : '-'})`).join(' · ')}
                                  </div>
                                ) : null}
                              </div>
                            ))}
                        </div>
                        {/* 修改建议列表（中文注释）：trade-off 视角的规避建议 */}
                        {Array.isArray(inferenceResult?.mitigation_suggestions) && inferenceResult.mitigation_suggestions.length > 0 ? (
                          <>
                            <div style={{ fontSize: 12, color: '#64748b', margin: '8px 0 6px' }}>Mitigation suggestions</div>
                            <div style={{ display: 'grid', gap: 8 }}>
                              {inferenceResult.mitigation_suggestions.map((s, idx) => (
                                <div key={s?.id || idx} style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, background: '#f8fafc' }}>
                                  <div style={{ fontSize: 13, color: '#0f172a', marginBottom: 4 }}>{String(s?.description || '')}</div>
                                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 12, color: '#334155' }}>
                                    {Array.isArray(s?.targeted_info_units) && s.targeted_info_units.length > 0 ? (
                                      <div>targets: {s.targeted_info_units.join(', ')}</div>
                                    ) : null}
                                    {Array.isArray(s?.suggested_transformations) && s.suggested_transformations.length > 0 ? (
                                      <div>transforms: {s.suggested_transformations.join(', ')}</div>
                                    ) : null}
                                    {Number.isFinite(s?.expected_risk_reduction) ? (
                                      <div>risk↓: {s.expected_risk_reduction}</div>
                                    ) : null}
                                    {s?.expected_utility_impact ? (
                                      <div>utility impact: {String(s.expected_utility_impact)}</div>
                                    ) : null}
                                    {s?.tradeoffs ? (
                                      <div>trade-offs: privacy {s.tradeoffs?.privacy ?? '-'} / utility {s.tradeoffs?.utility ?? '-'} / complexity {s.tradeoffs?.complexity ?? '-'}</div>
                                    ) : null}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </>
                        ) : null}

                        <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6, marginTop: 8 }}>Structured JSON</div>
                        <pre style={{ maxHeight: 440, overflow: 'auto', background: '#0b1020', color: '#e2e8f0', padding: 8, borderRadius: 8, border: '1px solid #111827' }}>
{JSON.stringify(inferenceResult, null, 2)}
                        </pre>
                      </div>
                    ) : null}
                  </div>
                </div>
            </Splitter.Panel>
          </Splitter>
        </div>

        {/* 底部输入条（中文注释）：固定于底部，圆角胶囊样式 */}
        {(currentSession && (currentSession.messages || []).length > 0) && (
          <div className={styles.composerDock}>
            <div className={styles.composerShadow} />
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
                      <img src={src} alt={`preview-${i}`} className={styles.composerPreviewImg} />
                      <button className={styles.composerPreviewRemove} onClick={() => removeSelectedImage(i)}>✕</button>
                    </div>
                  ))}
                </div>
              )}
              <div className={styles.composerRow}>
                <Input.TextArea
                  className={styles.composerInput}
                  placeholder="Message ChatGPT"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); handleSend() } }}
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
            </div>
            <div className={styles.disclaimer}>Model streams responses. Context comes from this chat history.</div>
          </div>
        )}
      </section>
    </div>
  )
}