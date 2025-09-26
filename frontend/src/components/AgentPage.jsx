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
  // 属性级展示不再需要画像索引（中文注释）
  const [apiModalOpen, setApiModalOpen] = useState(false)
  const [apiModelId, setApiModelId] = useState('')
  const [apiBaseUrl, setApiBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')



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

  // 属性级展示（中文注释）：不需要默认选择



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
    return (
      <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, background: '#ffffff' }}>
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
          <div className={styles.contextSection}>
            <div className={styles.contextInfo}>
              <div className={styles.contextLabel}>Context window<span className={styles.contextText}>{contextLabel}</span></div>
              <Progress percent={contextPercent} size="small" className={styles.contextProgress} />
            </div>
          </div>
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
                <div className={styles.rightPaneHeader}>
                  <div className={styles.rightPaneTitle}>Privacy inference</div>
                </div>
                <div className={styles.rightPaneBody}>
                  {/* 实时显示未发送输入与已选图片（中文注释） */}
                  {(((input || '').trim().length > 0) || ((landingInput || '').trim().length > 0) || selectedImages.length > 0) && renderPendingCard()}
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