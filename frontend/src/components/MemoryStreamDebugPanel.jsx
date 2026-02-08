/**
 * MemoryStreamDebugPanel - 主记忆流与关联回溯可视化调试面板
 * 
 * 浮动式调试面板，不影响现有布局。
 * 移除方式：删除本文件 + 移除 AgentPage.jsx 中的 import 和 <MemoryStreamDebugPanel /> 即可。
 */
import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useStore } from '../store'

// ======================== 样式常量（与系统浅色主题一致） ========================

const C = {
  bg: '#ffffff',
  bgCard: '#f8fafc',
  bgHover: '#f1f5f9',
  border: '#e2e8f0',
  borderMed: '#cbd5e1',
  accent: '#0ea5e9',
  accentLight: '#e0f2fe',
  accentHover: '#0284c7',
  text: '#0f172a',
  textSec: '#475569',
  textTer: '#94a3b8',
  green: '#16a34a',
  greenBg: '#f0fdf4',
  red: '#dc2626',
  redBg: '#fef2f2',
  yellow: '#ca8a04',
  yellowBg: '#fefce8',
  blue: '#3b82f6',
  blueBg: '#eff6ff',
  purple: '#8b5cf6',
  purpleBg: '#f5f3ff',
  teal: '#0d9488',
  tealBg: '#f0fdfa',
  shadow: '0 4px 24px rgba(0,0,0,0.10), 0 1px 4px rgba(0,0,0,0.06)',
  font: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
}

const fabStyle = {
  position: 'fixed',
  bottom: 16,
  right: 16,
  zIndex: 99999,
  width: 36,
  height: 36,
  borderRadius: '50%',
  background: C.accent,
  color: '#fff',
  border: 'none',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: 16,
  fontWeight: 700,
  boxShadow: '0 2px 8px rgba(14,165,233,0.35)',
  transition: 'transform 0.2s, background 0.2s',
}

const panelStyle = {
  position: 'fixed',
  bottom: 60,
  right: 16,
  zIndex: 99998,
  width: 480,
  height: 560,
  background: C.bg,
  borderRadius: 12,
  border: `1px solid ${C.border}`,
  boxShadow: C.shadow,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  fontFamily: C.font,
  fontSize: 12,
  color: C.text,
}

// ======================== 小工具组件 ========================

function Badge({ children, color = C.accent, bg }) {
  return (
    <span style={{
      display: 'inline-block',
      padding: '1px 7px',
      borderRadius: 10,
      background: bg || color + '14',
      color,
      fontSize: 10,
      fontWeight: 600,
      border: `1px solid ${color}30`,
      whiteSpace: 'nowrap',
      lineHeight: '16px',
    }}>
      {children}
    </span>
  )
}

function Stat({ label, value, color = C.text }) {
  return (
    <div style={{ textAlign: 'center', flex: 1 }}>
      <div style={{ fontSize: 18, fontWeight: 700, color, lineHeight: 1.2 }}>{value}</div>
      <div style={{ fontSize: 9, color: C.textTer, marginTop: 2 }}>{label}</div>
    </div>
  )
}

function SectionTitle({ children }) {
  return <div style={{ fontSize: 10, fontWeight: 700, color: C.textTer, textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 6, marginTop: 12 }}>{children}</div>
}

function EmptyHint({ text }) {
  return <div style={{ color: C.textTer, fontSize: 11, textAlign: 'center', padding: '20px 0' }}>{text}</div>
}

function SimScore({ value }) {
  const pct = (value * 100).toFixed(1)
  const color = value >= 0.85 ? C.red : value >= 0.5 ? C.yellow : C.teal
  return <span style={{ fontSize: 10, fontWeight: 700, color, flexShrink: 0 }}>{pct}%</span>
}

function InfonMini({ infon, similarity }) {
  const type = String(infon.infon_type || '').toUpperCase()
  const colors = { DESC: C.blue, SCEN: C.green, REL: C.purple }
  const c = colors[type] || C.textTer
  const entity = infon.entity || infon.temporal || infon.relation_name || ''
  const attr = infon.attribute || infon.spatial || (infon.arg_refs || []).join(', ') || ''
  return (
    <div style={{
      background: C.bgCard,
      borderRadius: 6,
      padding: '5px 10px',
      border: `1px solid ${C.border}`,
      display: 'flex',
      alignItems: 'center',
      gap: 6,
      minWidth: 0,
    }}>
      <Badge color={c}>{type || '?'}</Badge>
      <div style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 11 }}>
        <span style={{ fontWeight: 600 }}>{entity}</span>
        {attr && <span style={{ color: C.textSec }}>{' : '}{attr}</span>}
      </div>
      {similarity !== undefined && <SimScore value={similarity} />}
    </div>
  )
}

function InputRow({ value, onChange, onSubmit, placeholder, buttonText, loading, disabled }) {
  return (
    <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
      <input
        value={value}
        onChange={e => onChange(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && onSubmit()}
        placeholder={placeholder}
        style={{
          flex: 1,
          background: C.bgCard,
          border: `1px solid ${C.border}`,
          borderRadius: 6,
          color: C.text,
          padding: '6px 10px',
          fontSize: 12,
          outline: 'none',
          fontFamily: C.font,
        }}
      />
      <button
        onClick={onSubmit}
        disabled={loading || disabled}
        style={{
          background: C.accentLight,
          color: C.accent,
          border: `1px solid ${C.accent}40`,
          borderRadius: 6,
          padding: '5px 14px',
          cursor: loading || disabled ? 'default' : 'pointer',
          fontSize: 11,
          fontWeight: 600,
          opacity: loading || disabled ? 0.5 : 1,
          fontFamily: C.font,
        }}
      >
        {loading ? '...' : buttonText}
      </button>
    </div>
  )
}

// ======================== Tab: Store ========================

function StoreTab() {
  const { memoryStreamStatus, memoryStreamLastIngest, fetchMemoryStreamStatus, clearMemoryStream } = useStore()
  const [clearing, setClearing] = useState(false)

  useEffect(() => { fetchMemoryStreamStatus() }, [fetchMemoryStreamStatus])

  const handleClear = async () => {
    setClearing(true)
    await clearMemoryStream()
    await fetchMemoryStreamStatus()
    setClearing(false)
  }

  const status = memoryStreamStatus
  const last = memoryStreamLastIngest

  return (
    <div style={{ padding: '10px 14px', overflow: 'auto', flex: 1 }}>
      {/* Stats */}
      <div style={{ display: 'flex', padding: '10px 0 14px', borderBottom: `1px solid ${C.border}` }}>
        <Stat label="Total Infons" value={status?.total_infons ?? '-'} color={C.blue} />
        <Stat label="Index Size" value={status?.index_size ?? '-'} color={C.teal} />
        <Stat label="Embed Dim" value={status?.embedding_dim ?? '-'} color={C.purple} />
        <Stat label="Status" value={status?.status === 'ok' ? 'OK' : status?.status || '-'} color={status?.status === 'ok' ? C.green : C.red} />
      </div>

      {/* Last ingest */}
      <SectionTitle>Last Ingest</SectionTitle>
      {last ? (
        <div>
          <div style={{ display: 'flex', gap: 14, marginBottom: 8, fontSize: 11 }}>
            <span>Ingested: <b style={{ color: C.green }}>{last.ingested_count}</b></span>
            <span>Skipped: <b style={{ color: C.yellow }}>{last.skipped_count}</b></span>
            <span>Store: <b style={{ color: C.blue }}>{last.total_in_store}</b></span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 200, overflow: 'auto' }}>
            {(last.ingested || []).map((item) => (
              <div key={item.iid} style={{ background: C.bgCard, borderRadius: 6, padding: '5px 10px', border: `1px solid ${C.border}`, fontSize: 11 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 600 }}>{item.iid}</span>
                  {item.associations?.length > 0 && <Badge color={C.teal}>{item.associations.length} assoc</Badge>}
                </div>
                <div style={{ color: C.textTer, fontSize: 10, marginTop: 2 }}>ptr: {item.evidence_pointer || 'n/a'}</div>
                {item.associations?.length > 0 && (
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 3 }}>
                    {item.associations.map((a, i) => (
                      <span key={i} style={{ fontSize: 9, color: C.teal }}>{a.iid} ({(a.similarity * 100).toFixed(1)}%)</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <EmptyHint text="No ingest yet" />
      )}

      {/* Clear */}
      <div style={{ marginTop: 16, textAlign: 'center' }}>
        <button
          onClick={handleClear}
          disabled={clearing}
          style={{
            background: clearing ? C.bgCard : C.redBg,
            color: C.red,
            border: `1px solid ${C.red}30`,
            borderRadius: 6,
            padding: '5px 20px',
            cursor: clearing ? 'wait' : 'pointer',
            fontSize: 11,
            fontWeight: 600,
            fontFamily: C.font,
          }}
        >
          {clearing ? 'Clearing...' : 'Clear All Memory Stream'}
        </button>
      </div>
    </div>
  )
}

// ======================== Tab: Triggers ========================

function TriggersTab() {
  const { memoryTriggerResult, memoryRetrievedInfons } = useStore()
  const result = memoryTriggerResult

  return (
    <div style={{ padding: '10px 14px', overflow: 'auto', flex: 1 }}>
      <SectionTitle>Last Trigger Check</SectionTitle>
      {result ? (
        <>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6, marginBottom: 10,
            padding: '4px 12px', borderRadius: 6,
            background: result.triggered ? C.redBg : C.greenBg,
            border: `1px solid ${result.triggered ? C.red : C.green}25`,
          }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: result.triggered ? C.red : C.green }} />
            <span style={{ fontWeight: 600, fontSize: 11, color: result.triggered ? C.red : C.green }}>
              {result.triggered ? 'TRIGGERED' : 'NOT TRIGGERED'}
            </span>
          </div>

          {result.triggers?.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
              {result.triggers.map((t, i) => {
                const type = t.trigger_type || ''
                let detail = ''
                let labelColor = C.accent
                if (type === 'quasi_identifier_combination') {
                  detail = `${t.categories_count} categories: ${(t.categories || []).join(', ')}`
                  labelColor = C.yellow
                } else if (type === 'refinement_detection') {
                  detail = `max sim: ${(t.max_similarity * 100).toFixed(1)}% (threshold: ${(t.threshold * 100).toFixed(0)}%), infon: ${t.triggered_infon_iid || '-'}`
                  labelColor = C.red
                } else if (type === 'sensitive_domain_hit') {
                  detail = `domains: ${(t.domains_hit || []).join(', ')}`
                  labelColor = C.purple
                }
                return (
                  <div key={i} style={{ background: C.bgCard, borderRadius: 6, padding: '6px 10px', border: `1px solid ${C.border}` }}>
                    <Badge color={labelColor}>{type.replace(/_/g, ' ')}</Badge>
                    <div style={{ fontSize: 10, color: C.textSec, marginTop: 3 }}>{detail}</div>
                  </div>
                )
              })}
            </div>
          )}

          <SectionTitle>Retrieved Infons ({memoryRetrievedInfons.length})</SectionTitle>
          {memoryRetrievedInfons.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {memoryRetrievedInfons.map((inf) => (
                <InfonMini key={inf.iid} infon={inf} similarity={inf.retrieval_similarity} />
              ))}
            </div>
          ) : (
            <EmptyHint text="No infons retrieved" />
          )}
        </>
      ) : (
        <EmptyHint text="No trigger check performed yet. Triggers run automatically before privacy inference." />
      )}
    </div>
  )
}

// ======================== Tab: Associations ========================

// ======================== Tab: Search ========================

function SearchTab() {
  const { searchMemoryStream, memoryStreamLoading } = useStore()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [searched, setSearched] = useState(false)

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return
    const r = await searchMemoryStream(query.trim(), 10)
    setResults(r || [])
    setSearched(true)
  }, [query, searchMemoryStream])

  return (
    <div style={{ padding: '10px 14px', overflow: 'auto', flex: 1 }}>
      <SectionTitle>Vector Similarity Search</SectionTitle>
      <InputRow
        value={query} onChange={setQuery} onSubmit={handleSearch}
        placeholder="Search text (e.g. 北京 地址)"
        buttonText="Search" loading={memoryStreamLoading} disabled={!query.trim()}
      />

      {searched && (
        <>
          <div style={{ fontSize: 10, color: C.textTer, marginBottom: 6 }}>
            {results.length} result{results.length !== 1 ? 's' : ''}
          </div>
          {results.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {results.map((inf) => (
                <div key={inf.iid} style={{ background: C.bgCard, borderRadius: 6, padding: '6px 10px', border: `1px solid ${C.border}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0, overflow: 'hidden' }}>
                      <Badge color={C.blue}>{inf.infon_type || '?'}</Badge>
                      <span style={{ fontWeight: 600, fontSize: 11 }}>{inf.entity || inf.temporal || inf.relation_name || ''}</span>
                      <span style={{ color: C.textSec, fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {inf.attribute || inf.spatial || ''}
                      </span>
                    </div>
                    <SimScore value={inf.similarity} />
                  </div>
                  <div style={{ fontSize: 9, color: C.textTer, marginTop: 3, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                    <span>iid: {inf.iid}</span>
                    <span>session: {inf.session_id}</span>
                    <span>R{inf.round_num}</span>
                    <span>{inf.modality}</span>
                  </div>
                  {inf.associations?.length > 0 && (
                    <div style={{ fontSize: 9, color: C.teal, marginTop: 2 }}>
                      assoc: {inf.associations.map(a => `${a.iid}(${(a.similarity * 100).toFixed(0)}%)`).join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <EmptyHint text="No results found" />
          )}
        </>
      )}
    </div>
  )
}

// ======================== Tab: Map (向量可视化 - Canvas 高性能渲染) ========================

const TYPE_COLORS = {
  DESC: { fill: '#3b82f6', stroke: '#2563eb', label: 'DESC' },
  SCEN: { fill: '#16a34a', stroke: '#15803d', label: 'SCEN' },
  REL:  { fill: '#8b5cf6', stroke: '#7c3aed', label: 'REL' },
  _default: { fill: '#94a3b8', stroke: '#64748b', label: '?' },
}

/** Canvas 绘制引擎 — 万级节点无压力 */
function drawCanvas(ctx, data, pointMap, transform, dimensions, showEdges, showDensity, selected, hoveredIid) {
  const { width, height } = dimensions
  const dpr = window.devicePixelRatio || 1

  ctx.save()
  ctx.clearRect(0, 0, width * dpr, height * dpr)
  ctx.scale(dpr, dpr)

  const pad = 24
  const innerW = width - pad * 2
  const innerH = height - pad * 2

  // 应用 zoom transform
  ctx.translate(transform.x, transform.y)
  ctx.scale(transform.k, transform.k)

  const xScale = x => pad + x * innerW
  const yScale = y => pad + (1 - y) * innerH

  // ---- 密度热力图 (平滑径向渐变) ----
  if (showDensity && data.points.length > 4) {
    // 用离屏 canvas 绘制热力图，然后叠加到主画布
    const heatCanvas = document.createElement('canvas')
    heatCanvas.width = innerW
    heatCanvas.height = innerH
    const hctx = heatCanvas.getContext('2d')

    // 每个点画一个径向渐变圆，自然叠加形成平滑热力
    const baseRadius = Math.max(30, Math.min(80, 300 / Math.sqrt(data.points.length)))

    for (const p of data.points) {
      const px = p.x * innerW
      const py = (1 - p.y) * innerH
      const grad = hctx.createRadialGradient(px, py, 0, px, py, baseRadius)
      grad.addColorStop(0, 'rgba(14, 165, 233, 0.12)')
      grad.addColorStop(0.4, 'rgba(14, 165, 233, 0.06)')
      grad.addColorStop(1, 'rgba(14, 165, 233, 0)')
      hctx.fillStyle = grad
      hctx.fillRect(px - baseRadius, py - baseRadius, baseRadius * 2, baseRadius * 2)
    }

    ctx.drawImage(heatCanvas, pad, pad)
  }

  // ---- 关联边 ----
  if (showEdges && data.edges?.length) {
    for (const edge of data.edges) {
      const s = pointMap.get(edge.source)
      const t = pointMap.get(edge.target)
      if (!s || !t) continue

      const sim = edge.similarity || 0
      const isHighlight = selected && (edge.source === selected || edge.target === selected)

      if (isHighlight) {
        ctx.strokeStyle = sim >= 0.5 ? 'rgba(239,68,68,0.6)' : 'rgba(249,115,22,0.5)'
        ctx.lineWidth = Math.max(1, sim * 3)
      } else {
        ctx.strokeStyle = sim >= 0.8 ? 'rgba(239,68,68,0.25)'
          : sim >= 0.5 ? 'rgba(245,158,11,0.20)'
          : 'rgba(148,163,184,0.12)'
        ctx.lineWidth = Math.max(0.5, sim * 1.5)
      }

      ctx.beginPath()
      ctx.moveTo(xScale(s.x), yScale(s.y))
      ctx.lineTo(xScale(t.x), yScale(t.y))

      if (!isHighlight && sim < 0.5) {
        ctx.setLineDash([3, 3])
      } else {
        ctx.setLineDash([])
      }
      ctx.stroke()
    }
    ctx.setLineDash([])
  }

  // ---- 信息元点 ----
  const selectedAssocIids = new Set()
  if (selected) {
    const sp = pointMap.get(selected)
    if (sp?.associations) {
      sp.associations.forEach(a => selectedAssocIids.add(a.iid))
    }
  }

  for (const p of data.points) {
    const tc = TYPE_COLORS[String(p.infon_type).toUpperCase()] || TYPE_COLORS._default
    const px = xScale(p.x)
    const py = yScale(p.y)
    const isSel = p.iid === selected
    const isAssoc = selectedAssocIids.has(p.iid)
    const isHov = p.iid === hoveredIid

    let r = isSel ? 8 : isAssoc ? 6 : isHov ? 6 : 4
    // 数据量大时缩小点
    if (data.points.length > 500) r = Math.max(2, r - 1)
    if (data.points.length > 2000) r = Math.max(1.5, r - 1)

    ctx.globalAlpha = isSel || isAssoc || isHov ? 1.0 : 0.75

    const modality = p.modality || 'text'

    // 选中的光晕
    if (isSel) {
      ctx.fillStyle = tc.fill + '20'
      ctx.beginPath()
      ctx.arc(px, py, r + 6, 0, Math.PI * 2)
      ctx.fill()
    }

    ctx.fillStyle = tc.fill
    ctx.strokeStyle = isSel ? '#0f172a' : isAssoc ? '#f59e0b' : tc.stroke
    ctx.lineWidth = isSel ? 2.5 : isAssoc ? 2 : 1

    if (modality === 'image') {
      // 菱形
      ctx.beginPath()
      ctx.moveTo(px, py - r)
      ctx.lineTo(px + r, py)
      ctx.lineTo(px, py + r)
      ctx.lineTo(px - r, py)
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
    } else if (modality === 'audio') {
      // 三角
      ctx.beginPath()
      ctx.moveTo(px, py - r)
      ctx.lineTo(px + r, py + r * 0.7)
      ctx.lineTo(px - r, py + r * 0.7)
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
    } else {
      // 圆
      ctx.beginPath()
      ctx.arc(px, py, r, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    }

    // 选中 / 关联 / hover 时显示标签
    if (isSel || isAssoc || isHov) {
      ctx.globalAlpha = 1
      const label = p.entity || p.iid
      ctx.font = `${isSel ? 'bold ' : ''}${isSel ? 10 : 9}px -apple-system, sans-serif`
      ctx.fillStyle = '#0f172a'
      ctx.textAlign = 'center'
      ctx.fillText(label, px, py - r - 4, 120)
    }
  }

  ctx.globalAlpha = 1
  ctx.restore()
}

function MapTab() {
  const {
    fetchVisualizationData,
    memoryVisualizationData,
    memoryVisualizationLoading,
    memoryStreamStatus,
    fetchMemoryStreamStatus,
  } = useStore()

  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const transformRef = useRef({ x: 0, y: 0, k: 1 })
  const [method, setMethod] = useState('auto')
  const [hovered, setHovered] = useState(null)     // hovered point data
  const [selected, setSelected] = useState(null)   // selected iid
  const [showEdges, setShowEdges] = useState(true)
  const [showDensity, setShowDensity] = useState(true)
  const [dimensions, setDimensions] = useState({ width: 450, height: 300 })

  const data = memoryVisualizationData
  const totalInfons = memoryStreamStatus?.total_infons ?? 0

  // 首次 mount 获取状态
  useEffect(() => { fetchMemoryStreamStatus() }, [fetchMemoryStreamStatus])

  // 加载数据
  const handleLoad = useCallback(() => {
    fetchVisualizationData(method)
  }, [fetchVisualizationData, method])

  // 状态加载后自动渲染
  useEffect(() => {
    if (totalInfons > 0 && !data) handleLoad()
  }, [totalInfons])  // eslint-disable-line react-hooks/exhaustive-deps

  // 容器尺寸
  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      if (width > 0 && height > 0) setDimensions({ width: Math.floor(width), height: Math.floor(height) })
    })
    obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  // iid → point 映射
  const pointMap = useMemo(() => {
    if (!data?.points) return new Map()
    return new Map(data.points.map(p => [p.iid, p]))
  }, [data])

  // 空间索引：给定 canvas 坐标找最近的点
  const findPointAt = useCallback((cx, cy) => {
    if (!data?.points?.length) return null
    const { width, height } = dimensions
    const pad = 24
    const innerW = width - pad * 2
    const innerH = height - pad * 2
    const t = transformRef.current

    let bestDist = 144 // 12px² 命中半径
    let bestPt = null
    for (const p of data.points) {
      const px = (pad + p.x * innerW) * t.k + t.x
      const py = (pad + (1 - p.y) * innerH) * t.k + t.y
      const dist = (cx - px) ** 2 + (cy - py) ** 2
      if (dist < bestDist) {
        bestDist = dist
        bestPt = p
      }
    }
    return bestPt
  }, [data, dimensions])

  // Canvas 重绘
  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    // 数据为空时清空画布，防止用户切换后残留旧内容
    if (!data?.points?.length) {
      const dpr = window.devicePixelRatio || 1
      ctx.clearRect(0, 0, canvas.width * dpr, canvas.height * dpr)
      return
    }
    drawCanvas(ctx, data, pointMap, transformRef.current, dimensions, showEdges, showDensity, selected, hovered?.iid)
  }, [data, pointMap, dimensions, showEdges, showDensity, selected, hovered])

  useEffect(() => { redraw() }, [redraw])

  // Canvas 交互事件
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Mouse move → hover
    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect()
      const pt = findPointAt(e.clientX - rect.left, e.clientY - rect.top)
      setHovered(pt)
      canvas.style.cursor = pt ? 'pointer' : 'grab'
    }

    // Click → select
    const onClick = (e) => {
      const rect = canvas.getBoundingClientRect()
      const pt = findPointAt(e.clientX - rect.left, e.clientY - rect.top)
      setSelected(prev => pt ? (prev === pt.iid ? null : pt.iid) : null)
    }

    // Wheel → zoom
    const onWheel = (e) => {
      e.preventDefault()
      const t = transformRef.current
      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      const newK = Math.max(0.3, Math.min(20, t.k * factor))
      transformRef.current = {
        k: newK,
        x: mx - (mx - t.x) * (newK / t.k),
        y: my - (my - t.y) * (newK / t.k),
      }
      redraw()
    }

    // Drag → pan
    let dragging = false, dragStart = { x: 0, y: 0 }, dragOrigin = { x: 0, y: 0 }
    const onDown = (e) => {
      if (findPointAt(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top)) return
      dragging = true
      dragStart = { x: e.clientX, y: e.clientY }
      dragOrigin = { ...transformRef.current }
      canvas.style.cursor = 'grabbing'
    }
    const onDrag = (e) => {
      if (!dragging) return
      transformRef.current = {
        ...transformRef.current,
        x: dragOrigin.x + (e.clientX - dragStart.x),
        y: dragOrigin.y + (e.clientY - dragStart.y),
      }
      redraw()
    }
    const onUp = () => { dragging = false; canvas.style.cursor = 'grab' }

    // 双击 → reset zoom
    const onDblClick = () => {
      transformRef.current = { x: 0, y: 0, k: 1 }
      redraw()
    }

    canvas.addEventListener('mousemove', onMove)
    canvas.addEventListener('click', onClick)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('mousedown', onDown)
    window.addEventListener('mousemove', onDrag)
    window.addEventListener('mouseup', onUp)
    canvas.addEventListener('dblclick', onDblClick)

    return () => {
      canvas.removeEventListener('mousemove', onMove)
      canvas.removeEventListener('click', onClick)
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('mousedown', onDown)
      window.removeEventListener('mousemove', onDrag)
      window.removeEventListener('mouseup', onUp)
      canvas.removeEventListener('dblclick', onDblClick)
    }
  }, [findPointAt, redraw])

  // 选中的完整信息
  const selectedPoint = selected ? pointMap.get(selected) : null
  const stats = data?.stats

  // 构建摘要文字
  const summaryText = useMemo(() => {
    if (!data) return ''
    const parts = [`${data.displayed} pts`]
    if (data.sampled) parts.push(`sampled from ${data.total}`)
    parts.push(data.method.toUpperCase())
    if (data.perplexity) parts.push(`perp=${data.perplexity}`)
    if (stats?.compute_time) parts.push(`${stats.compute_time}s`)
    return parts.join(' · ')
  }, [data, stats])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      {/* 控制栏 */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px',
        borderBottom: `1px solid ${C.border}`, flexWrap: 'wrap',
      }}>
        <select value={method} onChange={e => setMethod(e.target.value)}
          style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, border: `1px solid ${C.border}`, background: C.bgCard, color: C.text, fontFamily: C.font }}>
          <option value="auto">Auto</option>
          <option value="tsne">t-SNE</option>
          <option value="pca">PCA</option>
        </select>
        <label style={{ fontSize: 10, color: C.textSec, display: 'flex', alignItems: 'center', gap: 2 }}>
          <input type="checkbox" checked={showEdges} onChange={e => setShowEdges(e.target.checked)} style={{ width: 11, height: 11 }} />
          Edges
        </label>
        <label style={{ fontSize: 10, color: C.textSec, display: 'flex', alignItems: 'center', gap: 2 }}>
          <input type="checkbox" checked={showDensity} onChange={e => setShowDensity(e.target.checked)} style={{ width: 11, height: 11 }} />
          Density
        </label>
        <button onClick={handleLoad} disabled={memoryVisualizationLoading || totalInfons === 0}
          style={{ marginLeft: 'auto', fontSize: 10, padding: '3px 10px', borderRadius: 4, background: C.accentLight, color: C.accent, border: `1px solid ${C.accent}40`, cursor: memoryVisualizationLoading ? 'wait' : 'pointer', fontWeight: 600, opacity: memoryVisualizationLoading || totalInfons === 0 ? 0.5 : 1, fontFamily: C.font }}>
          {memoryVisualizationLoading ? 'Computing…' : '↻ Refresh'}
        </button>
      </div>

      {/* 图例 + 统计 */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '3px 12px',
        borderBottom: `1px solid ${C.border}`, fontSize: 9, color: C.textSec, flexWrap: 'wrap',
      }}>
        {Object.entries(TYPE_COLORS).filter(([k]) => k !== '_default').map(([key, val]) => (
          <span key={key} style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: val.fill, border: `1px solid ${val.stroke}`, display: 'inline-block' }} />
            {val.label}
            {stats?.type_counts?.[key] != null && <span style={{ color: C.textTer }}>({stats.type_counts[key]})</span>}
          </span>
        ))}
        <span style={{ color: C.textTer }}>◇img △audio ●text</span>
        {summaryText && <span style={{ marginLeft: 'auto', color: C.textTer }}>{summaryText}</span>}
      </div>

      {/* Canvas 画布 */}
      <div ref={containerRef} style={{ flex: 1, position: 'relative', overflow: 'hidden', minHeight: 0 }}>
        {!data?.points?.length && !memoryVisualizationLoading && (
          <EmptyHint text={totalInfons === 0
            ? 'No infons in memory stream yet'
            : 'Click "↻ Refresh" to visualize the vector space'
          } />
        )}
        {memoryVisualizationLoading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(255,255,255,0.85)', zIndex: 2,
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ color: C.accent, fontWeight: 600, fontSize: 12 }}>Computing projection…</div>
              <div style={{ color: C.textTer, fontSize: 10, marginTop: 4 }}>Method & perplexity auto-selected</div>
            </div>
          </div>
        )}
        <canvas
          ref={canvasRef}
          width={dimensions.width * (window.devicePixelRatio || 1)}
          height={dimensions.height * (window.devicePixelRatio || 1)}
          style={{ display: 'block', width: dimensions.width, height: dimensions.height, cursor: 'grab' }}
        />

        {/* Hover tooltip */}
        {hovered && (
          <div style={{
            position: 'absolute', top: 8, left: 8,
            background: 'rgba(255,255,255,0.96)',
            border: `1px solid ${C.border}`, borderRadius: 8,
            padding: '8px 12px', maxWidth: 220,
            boxShadow: '0 2px 12px rgba(0,0,0,0.10)',
            pointerEvents: 'none', zIndex: 3,
            fontSize: 10, lineHeight: 1.5,
          }}>
            <div style={{ fontWeight: 700, fontSize: 11, marginBottom: 4, display: 'flex', gap: 6, alignItems: 'center' }}>
              <Badge color={(TYPE_COLORS[String(hovered.infon_type).toUpperCase()] || TYPE_COLORS._default).fill}>
                {hovered.infon_type}
              </Badge>
              <Badge color={C.textTer}>{hovered.modality}</Badge>
            </div>
            <div style={{ fontWeight: 600, color: C.text }}>{hovered.entity || '(no entity)'}</div>
            {hovered.attribute && <div style={{ color: C.textSec }}>{hovered.attribute}</div>}
            <div style={{ color: C.textTer, fontSize: 9, marginTop: 4 }}>
              iid: {hovered.iid}
            </div>
            <div style={{ color: C.textTer, fontSize: 9 }}>
              session: {hovered.session_id} · R{hovered.round_num}
            </div>
            {hovered.associations?.length > 0 && (
              <div style={{ color: C.teal, fontSize: 9, marginTop: 3 }}>
                {hovered.associations.length} association{hovered.associations.length > 1 ? 's' : ''}
              </div>
            )}
          </div>
        )}

        {/* 右下角快捷提示 */}
        <div style={{
          position: 'absolute', bottom: 4, right: 8,
          fontSize: 8, color: C.textTer, pointerEvents: 'none',
        }}>
          scroll=zoom · drag=pan · dblclick=reset
        </div>
      </div>

      {/* 选中信息元详情 */}
      {selectedPoint && (
        <div style={{
          borderTop: `1px solid ${C.border}`,
          padding: '8px 14px', maxHeight: 120, overflow: 'auto',
          background: C.bgCard, fontSize: 10,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <Badge color={(TYPE_COLORS[String(selectedPoint.infon_type).toUpperCase()] || TYPE_COLORS._default).fill}>
              {selectedPoint.infon_type}
            </Badge>
            <span style={{ fontWeight: 700, fontSize: 11 }}>{selectedPoint.entity || selectedPoint.iid}</span>
            {selectedPoint.attribute && <span style={{ color: C.textSec }}>: {selectedPoint.attribute}</span>}
          </div>
          <div style={{ color: C.textTer, fontSize: 9, marginBottom: 3 }}>
            embed: &quot;{selectedPoint.text_for_embedding}&quot; · {selectedPoint.modality} · R{selectedPoint.round_num}
          </div>
          {selectedPoint.associations?.length > 0 && (
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {selectedPoint.associations.map((a, i) => (
                <span key={i}
                  onClick={() => setSelected(a.iid)}
                  style={{
                    fontSize: 9, color: C.teal, cursor: 'pointer',
                    textDecoration: 'underline', textDecorationColor: C.teal + '40',
                  }}
                >
                  → {a.iid} ({(a.similarity * 100).toFixed(0)}%)
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ======================== 主面板 ========================

const TABS = [
  { key: 'map', label: '🗺 Map' },
  { key: 'store', label: 'Store' },
  { key: 'triggers', label: 'Triggers' },
  { key: 'search', label: 'Search' },
]

export default function MemoryStreamDebugPanel() {
  const [open, setOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('map')
  // 监听用户变化：用户切换时 key 变化 → 所有子组件强制重新挂载
  // 这样所有 useEffect 重新触发、本地状态重置、Canvas 清空
  const currentUserId = useStore(s => s.currentUserId)

  useEffect(() => {
    if (!open) return
    const handleKey = (e) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [open])

  return (
    <>
      {/* FAB toggle */}
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          ...fabStyle,
          transform: open ? 'rotate(45deg)' : 'rotate(0deg)',
          background: open ? C.accentHover : C.accent,
        }}
        title="Memory Stream Debug Panel"
      >
        {open ? '\u00D7' : '\u29C9'}
      </button>

      {/* Panel */}
      {open && (
        <div style={panelStyle}>
          {/* Header */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            padding: '10px 14px',
            borderBottom: `1px solid ${C.border}`,
            background: C.bgCard,
          }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>Memory Stream</span>
            <span style={{ fontSize: 10, color: C.textTer, marginLeft: 8 }}>
              {currentUserId
                ? `👤 ${currentUserId}`
                : `👤 anonymous (${useStore.getState()._getMemoryUserId?.().slice(-10) || '?'})`
              }
            </span>
          </div>

          {/* Tab bar */}
          <div style={{
            display: 'flex',
            borderBottom: `1px solid ${C.border}`,
            background: C.bg,
          }}>
            {TABS.map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                style={{
                  flex: 1,
                  padding: '7px 0',
                  background: 'transparent',
                  border: 'none',
                  borderBottom: activeTab === tab.key ? `2px solid ${C.accent}` : '2px solid transparent',
                  color: activeTab === tab.key ? C.accent : C.textTer,
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                  fontFamily: C.font,
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content — key 随 currentUserId 变化，强制子组件重新挂载 */}
          <div key={currentUserId ?? '_anon'} style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {activeTab === 'map' && <MapTab />}
            {activeTab === 'store' && <StoreTab />}
            {activeTab === 'triggers' && <TriggersTab />}
            {activeTab === 'search' && <SearchTab />}
          </div>
        </div>
      )}
    </>
  )
}
