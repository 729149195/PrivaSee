import React, { useEffect, useRef, useState, useMemo } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'
import MarkdownMessage from './MarkdownMessage'
import { Splitter, Select, Button, Upload, Progress, Input, Modal } from 'antd'
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

  // 初始化当前会话（中文注释）：确保存在 currentSessionId
  useEffect(() => { _ensureCurrentSession() }, [_ensureCurrentSession])

  const currentSession = getCurrentSession()
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
  const [apiModalOpen, setApiModalOpen] = useState(false)
  const [apiModelId, setApiModelId] = useState('deepseek-chat')
  const [apiBaseUrl, setApiBaseUrl] = useState('https://api.deepseek.com/v1')
  const [apiKey, setApiKey] = useState('sk-8c2ee9474f2f44f5969dcd5de280e634')

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
                onChange={(v) => setModel?.(v)}
                options={[model, ...(models || [])]
                  .filter((v, i, a) => v && a.indexOf(v) === i)
                  .map((v) => ({ label: v, value: v }))}
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
                            <Button icon={<CameraOutlined />} />
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
                    <div style={{ color: '#475569', fontSize: 12, marginBottom: 8 }}>Messages in context</div>
                    <div style={{ display: 'grid', gap: 8 }}>
                      {(currentSession?.messages || []).slice(-30).map((m, i) => (
                        <div key={m.id || i} style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, background: '#ffffff' }}>
                          {/* 折叠为网络节点（中文注释）：仅展示节点占位，后续可视化接入 */}
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ width: 10, height: 10, borderRadius: 9999, background: '#94a3b8' }} />
                            <div style={{ fontSize: 12, color: '#334155' }}>{m.role}</div>
                            <div style={{ fontSize: 12, color: '#64748b' }}>
                              {(m.content || '').slice(0, 24)}{(m.content || '').length > 24 ? '…' : ''}
                            </div>
                          </div>
                        </div>
                      ))}
                      {/* 若已有推断结果，则在此展示网络图（中文注释） */}
                      {inferenceResult ? (
                        <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, background: '#ffffff' }}>
                          {(() => {
                            try {
                              const g = inferenceResult?.graph || {}
                              const nodes = Array.isArray(g.nodes) ? g.nodes : []
                              const edges = Array.isArray(g.edges) ? g.edges : []
                              const evidence = Array.isArray(inferenceResult?.evidence) ? inferenceResult.evidence : []
                              const pendingIds = new Set(evidence.filter((e) => e?.source === 'pending').map((e) => e?.id).filter(Boolean))
                              const idToIdx = new Map()
                              nodes.forEach((n, idx) => idToIdx.set(n?.id, idx))
                              const W = 280, H = 180, cx = W/2, cy = H/2, R = Math.min(W,H)/2 - 20
                              const coords = nodes.map((_, i) => {
                                const t = (i / Math.max(1, nodes.length)) * Math.PI * 2
                                return { x: cx + R * Math.cos(t), y: cy + R * Math.sin(t) }
                              })
                              const dangerIdx = new Set()
                              edges.forEach((e) => {
                                const refs = Array.isArray(e?.evidence_refs) ? e.evidence_refs : []
                                const hit = refs.some((r) => pendingIds.has(r))
                                if (hit) {
                                  const si = idToIdx.get(e?.source)
                                  const ti = idToIdx.get(e?.target)
                                  if (si != null) dangerIdx.add(si)
                                  if (ti != null) dangerIdx.add(ti)
                                }
                              })
                              return (
                                <svg width={280} height={180} style={{ border: '1px solid #e5e7eb', borderRadius: 6, background: '#ffffff' }}>
                                  {edges.map((e, i) => {
                                    const si = idToIdx.get(e?.source)
                                    const ti = idToIdx.get(e?.target)
                                    if (si == null || ti == null) return null
                                    const w = Math.max(1, Math.min(6, Number(e?.weight) || 1))
                                    const refs = Array.isArray(e?.evidence_refs) ? e.evidence_refs : []
                                    const warn = refs.some((r) => pendingIds.has(r))
                                    return <line key={i} x1={coords[si].x} y1={coords[si].y} x2={coords[ti].x} y2={coords[ti].y} stroke={warn ? '#ef4444' : '#64748b'} strokeWidth={w} opacity={0.7} />
                                  })}
                                  {nodes.map((n, i) => (
                                    <g key={i}>
                                      <circle cx={coords[i].x} cy={coords[i].y} r={8} fill={dangerIdx.has(i) ? '#ef4444' : '#10a37f'} />
                                      <text x={coords[i].x + 10} y={coords[i].y + 4} fontSize={10} fill="#334155">{String(n?.label || n?.id || 'node')}</text>
                                    </g>
                                  ))}
                                </svg>
                              )
                            } catch (_) {
                              return null
                            }
                          })()}
                        </div>
                      ) : null}
                      {/* 实时显示未发送输入与已选图片（中文注释） */}
                      {(((input || '').trim().length > 0) || ((landingInput || '').trim().length > 0) || selectedImages.length > 0) && (
                        (() => {
                          // 根据推断结果标红 pending（中文注释）
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
                        })()
                      )}
                    </div>
                    {/* 隐私推断控制区（中文注释）：单独选择模型 + 推断按钮 + 结果显示 */}
                    <div style={{ height: 12 }} />
                    <div style={{ borderTop: '1px solid #e5e7eb', margin: '12px 0' }} />
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                      <div style={{ fontWeight: 600 }}>Privacy inference</div>
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                      {(() => {
                        const localModels = [model, ...(models || [])]
                          .filter((v, i, a) => v && a.indexOf(v) === i)
                          .filter((id) => !customProviders?.[id])
                        const currentVal = localModels.includes(inferenceModel || model)
                          ? (inferenceModel || model)
                          : (localModels[0] || '')
                        return (
                          <Select
                            style={{ minWidth: 220 }}
                            value={currentVal}
                            onChange={(v) => setInferenceModel(v)}
                            options={localModels.map((v) => ({ label: v, value: v }))}
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
                            const result = await runPrivacyInference(pending, inferenceModel || model)
                            setInferenceResult(result)
                          } catch (e) {
                            setInferenceResult(null)
                            setInferenceError(String(e?.message || e || 'Inference failed'))
                          } finally {
                            setInferenceLoading(false)
                          }
                        }}
                      >Infer</Button>
                    </div>
                    {inferenceError ? (
                      <div style={{ color: '#b91c1c', fontSize: 12, marginBottom: 8 }}>Error: {inferenceError}</div>
                    ) : null}
                    {inferenceResult ? (
                      <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, background: '#ffffff' }}>
                        <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Top profiles</div>
                        <div style={{ display: 'grid', gap: 6, marginBottom: 8 }}>
                          {(Array.isArray(inferenceResult?.top_profiles) ? inferenceResult.top_profiles : []).slice(0,3).map((p, idx) => (
                            <div key={idx} style={{ border: '1px solid #e5e7eb', borderRadius: 6, padding: 8 }}>
                              <div style={{ fontSize: 13 }}>
                                <b>#{idx+1}</b> · {String(p?.location || '-')}, {String(p?.age ?? '-')}, {String(p?.gender || '-')}
                              </div>
                              <div style={{ fontSize: 12, color: '#334155' }}>confidence: {typeof p?.confidence === 'number' ? p.confidence.toFixed(2) : String(p?.confidence || '-')}</div>
                              {p?.rationale ? <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>{String(p.rationale)}</div> : null}
                            </div>
                          ))}
                        </div>
                        {/* 简单网络图（中文注释）：根据 graph.edges.weight 设置线宽；含 pending 文本时标红 */}
                        {(() => {
                          try {
                            const g = inferenceResult?.graph || {}
                            const nodes = Array.isArray(g.nodes) ? g.nodes : []
                            const edges = Array.isArray(g.edges) ? g.edges : []
                            const idToIdx = new Map()
                            nodes.forEach((n, idx) => idToIdx.set(n?.id, idx))
                            const W = 280, H = 180, cx = W/2, cy = H/2, R = Math.min(W,H)/2 - 20
                            const coords = nodes.map((_, i) => {
                              const t = (i / Math.max(1, nodes.length)) * Math.PI * 2
                              return { x: cx + R * Math.cos(t), y: cy + R * Math.sin(t) }
                            })
                            const pendingText = (input || landingInput || '').trim()
                            const isPending = (label) => pendingText && typeof label === 'string' && label.length && (label.toLowerCase().includes(pendingText.toLowerCase().slice(0, 8)))
                            return (
                              <svg width={W} height={H} style={{ border: '1px solid #e5e7eb', borderRadius: 6, background: '#ffffff', marginBottom: 8 }}>
                                {edges.map((e, i) => {
                                  const si = idToIdx.get(e?.source)
                                  const ti = idToIdx.get(e?.target)
                                  if (si == null || ti == null) return null
                                  const w = Math.max(1, Math.min(6, Number(e?.weight) || 1))
                                  return <line key={i} x1={coords[si].x} y1={coords[si].y} x2={coords[ti].x} y2={coords[ti].y} stroke="#64748b" strokeWidth={w} opacity={0.7} />
                                })}
                                {nodes.map((n, i) => {
                                  const red = isPending(n?.label)
                                  return (
                                    <g key={i}>
                                      <circle cx={coords[i].x} cy={coords[i].y} r={8} fill={red ? '#ef4444' : '#10a37f'} />
                                      <text x={coords[i].x + 10} y={coords[i].y + 4} fontSize={10} fill="#334155">{String(n?.label || n?.id || 'node')}</text>
                                    </g>
                                  )
                                })}
                              </svg>
                            )
                          } catch (_) {
                            return null
                          }
                        })()}
                        <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Structured JSON</div>
                        <pre style={{ maxHeight: 220, overflow: 'auto', background: '#0b1020', color: '#e2e8f0', padding: 8, borderRadius: 8, border: '1px solid #111827' }}>
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
                    <Button icon={<CameraOutlined />} />
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