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
  width: 40,
  height: 40,
  borderRadius: 10,
  background: C.bg,
  color: C.textSec,
  border: `1px solid ${C.borderMed}`,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  boxShadow: '0 2px 8px rgba(15,23,42,0.10)',
  transition: 'all 0.16s ease',
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

function formatTs(ts) {
  if (!ts) return '-'
  const d = new Date(ts)
  const date = d.toLocaleDateString()
  const time = d.toLocaleTimeString()
  return `${date} ${time}`
}

function iidAlias(iid) {
  const raw = String(iid || '')
  if (!raw) return '-'
  const idx = raw.indexOf('__u')
  return idx > 0 ? raw.slice(0, idx) : raw
}

function shortenText(text, max = 34) {
  const raw = String(text || '')
  if (raw.length <= max) return raw
  const head = Math.max(8, Math.floor(max * 0.55))
  const tail = Math.max(6, max - head - 1)
  return `${raw.slice(0, head)}…${raw.slice(-tail)}`
}

function shortenHead(text, max = 24) {
  const raw = String(text || '')
  if (raw.length <= max) return raw
  return `${raw.slice(0, Math.max(1, max - 1))}…`
}

function sanitizeDisplayText(text) {
  // 去掉 embedding 文本里夹带的 iid / 作用域 token（例如 desc_r1_3__uuser_xxx...）
  // 这类 token 对调试可视化很干扰；原始 iid 仍可在详情面板中查看。
  const raw = String(text || '')
  if (!raw) return ''
  return raw
    // remove scoped infon ids like "desc_r1_3__uuser_xxx_ssession_r1" (and unscoped "desc_r1_3")
    .replace(/\b(?:desc|scen|rel)_r\d+_\d+(?:__u[a-zA-Z0-9_-]+)?\b/g, '')
    // remove leftover scope tokens if they appear alone
    .replace(/\b__u[a-zA-Z0-9_-]+\b/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

function getInfonPrimaryText(inf) {
  // Search/列表标题优先显示实际 embedding 文本（最贴近“内容是什么”）
  const embed = sanitizeDisplayText(inf?.text_for_embedding || '')
  if (embed) return embed
  const type = String(inf?.infon_type || '').toUpperCase()
  if (type === 'REL') {
    return inf?.relation_name || inf?.relation || ''
  }
  return inf?.entity || inf?.temporal || inf?.relation_name || ''
}

function useInfonLookup() {
  const {
    memoryVisualizationData,
    memoryRetrievedInfons,
    memoryStreamLastIngest,
    memoryBacktraceCache,
  } = useStore()

  const infonByIid = useMemo(() => {
    const map = new Map()

    ;(memoryVisualizationData?.points || []).forEach(p => { if (p?.iid) map.set(p.iid, p) })
    ;(memoryRetrievedInfons || []).forEach(p => { if (p?.iid) map.set(p.iid, p) })
    ;(memoryStreamLastIngest?.ingested || []).forEach(p => { if (p?.iid) map.set(p.iid, p) })

    // backtrace cache often includes enriched associations (with text_for_embedding)
    Object.values(memoryBacktraceCache || {}).forEach(bt => {
      ;(bt?.associations || []).forEach(a => { if (a?.iid) map.set(a.iid, a) })
      if (bt?.iid) map.set(bt.iid, bt)
    })

    return map
  }, [memoryVisualizationData, memoryRetrievedInfons, memoryStreamLastIngest, memoryBacktraceCache])

  const resolve = useCallback((iid) => infonByIid.get(iid), [infonByIid])
  return { resolve }
}

function InfonChip({ iid, resolve, color = C.teal, maxLen = 22 }) {
  const inf = resolve?.(iid)
  const content = inf ? (getInfonPrimaryText(inf) || getEmbeddingLabel(inf)) : ''
  const label = content ? shortenHead(content, maxLen) : iidAlias(iid)
  return (
    <span
      title={inf ? `${content}\n${iid}` : String(iid || '')}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        border: `1px solid ${C.border}`,
        background: C.bg,
        borderRadius: 10,
        padding: '1px 6px',
        fontSize: 9,
        color,
        maxWidth: 220,
      }}
    >
      {label || '-'}
    </span>
  )
}

function ReadableIid({ iid, color = C.textSec, showRaw = true }) {
  const raw = String(iid || '')
  const alias = iidAlias(raw)
  const needRaw = showRaw && raw && alias !== raw
  return (
    <span title={raw || alias} style={{ display: 'inline-flex', alignItems: 'baseline', gap: 4, minWidth: 0 }}>
      <span style={{ color, fontWeight: 600, wordBreak: 'break-all' }}>{alias || '-'}</span>
      {needRaw && <span style={{ color: C.textTer, fontSize: 9 }}>{shortenText(raw, 30)}</span>}
    </span>
  )
}

function IidList({ label, iids, color = C.teal, resolve }) {
  const uniq = Array.from(new Set((iids || []).filter(Boolean))).slice(0, 8)
  if (uniq.length === 0) return null
  return (
    <div style={{ marginTop: 2 }}>
      <span style={{ fontSize: 9, color: C.textTer }}>{label}:</span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 2 }}>
        {uniq.map(iid => (
          <InfonChip key={iid} iid={iid} resolve={resolve} color={color} />
        ))}
      </div>
    </div>
  )
}

function SimScore({ value }) {
  const pct = (value * 100).toFixed(1)
  const color = value >= 0.85 ? C.red : value >= 0.5 ? C.yellow : C.teal
  return <span style={{ fontSize: 10, fontWeight: 700, color, flexShrink: 0 }}>{pct}%</span>
}

function getTypeBadgeColor(infonType) {
  const type = String(infonType || '').toUpperCase()
  if (type === 'DESC') return C.blue
  if (type === 'SCEN') return C.green
  if (type === 'REL') return C.purple
  return C.textTer
}

function getEmbeddingLabel(p) {
  // 节点内部 label 优先展示 embedding 文本（对调试最直观），避免满屏 iid
  const embedText = sanitizeDisplayText(p?.text_for_embedding || '')
  return (
    embedText ||
    p?.entity ||
    p?.attribute ||
    p?.temporal ||
    p?.spatial ||
    p?.relation_name ||
    p?.iid ||
    ''
  )
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

function PanelToggleIcon({ open }) {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
      <rect x="2.25" y="3" width="10.5" height="8.5" rx="1.8" stroke={open ? C.accent : C.textSec} strokeWidth="1.6" />
      <rect x="5.25" y="6" width="10.5" height="8.5" rx="1.8" stroke={open ? C.accent : C.textSec} strokeWidth="1.6" opacity="0.9" />
    </svg>
  )
}

// ======================== Tab: Store ========================

function StoreTab() {
  const { memoryStreamStatus, memoryStreamLastIngest, fetchMemoryStreamStatus, clearMemoryStream } = useStore()
  const { resolve } = useInfonLookup()
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
                  <ReadableIid iid={item.iid} color={C.text} />
                  {item.associations?.length > 0 && <Badge color={C.teal}>{item.associations.length} assoc</Badge>}
                </div>
                <div style={{ color: C.textTer, fontSize: 10, marginTop: 2 }}>ptr: {item.evidence_pointer || 'n/a'}</div>
                {item.associations?.length > 0 && (
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 3 }}>
                    {item.associations.map((a, i) => (
                      <span key={i} style={{ fontSize: 9, color: C.teal }}>
                        <InfonChip iid={a.iid} resolve={resolve} color={C.teal} /> ({(a.similarity * 100).toFixed(1)}%)
                      </span>
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
                  detail = `max sim: ${(t.max_similarity * 100).toFixed(1)}% (threshold: ${(t.threshold * 100).toFixed(0)}%), infon: ${iidAlias(t.triggered_infon_iid) || '-'}`
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

function TraceTab() {
  const {
    memoryAssociationEvents,
    memoryBacktraceCache,
    memoryRetrievedInfons,
    memoryStreamLastIngest,
    queryBacktrace,
  } = useStore()
  const { resolve } = useInfonLookup()
  const [iidInput, setIidInput] = useState('')
  const [querying, setQuerying] = useState(false)
  const [selectedIid, setSelectedIid] = useState('')

  const candidateIids = useMemo(() => {
    const set = new Set()
    ;(memoryRetrievedInfons || []).forEach(inf => { if (inf?.iid) set.add(inf.iid) })
    ;(memoryStreamLastIngest?.ingested || []).forEach(item => { if (item?.iid) set.add(item.iid) })
    ;(memoryAssociationEvents || []).forEach(evt => {
      if (evt?.payload?.iid) set.add(evt.payload.iid)
      ;(evt?.payload?.retrieved_iids || []).forEach(iid => set.add(iid))
      ;(evt?.payload?.association_iids || []).forEach(iid => set.add(iid))
      ;(evt?.payload?.linked_iids || []).forEach(iid => set.add(iid))
    })
    return Array.from(set).slice(0, 40)
  }, [memoryRetrievedInfons, memoryStreamLastIngest, memoryAssociationEvents])

  const selectedBacktrace = selectedIid ? memoryBacktraceCache?.[selectedIid] : null

  const handleBacktrace = useCallback(async () => {
    const iid = iidInput.trim()
    if (!iid) return
    setQuerying(true)
    const result = await queryBacktrace(iid)
    setSelectedIid(iid)
    if (!result) {
      // 保留选中，便于用户看到 miss 事件和输入值
      setSelectedIid(iid)
    }
    setQuerying(false)
  }, [iidInput, queryBacktrace])

  const handleQuickPick = useCallback(async (iid) => {
    if (!iid) return
    setIidInput(iid)
    setQuerying(true)
    await queryBacktrace(iid)
    setSelectedIid(iid)
    setQuerying(false)
  }, [queryBacktrace])

  const typeStyle = (type) => {
    if (type === 'backtrace_query') return { color: C.teal, bg: C.tealBg, label: 'BACKTRACE' }
    if (type === 'backtrace_cache_hit') return { color: C.blue, bg: C.blueBg, label: 'CACHE' }
    if (type === 'backtrace_miss') return { color: C.red, bg: C.redBg, label: 'MISS' }
    if (type === 'trigger_check') return { color: C.purple, bg: C.purpleBg, label: 'TRIGGER' }
    if (type === 'ingest_association_bind') return { color: C.green, bg: C.greenBg, label: 'INGEST' }
    return { color: C.textSec, bg: C.bgCard, label: 'EVENT' }
  }

  return (
    <div style={{ padding: '10px 14px', overflow: 'auto', flex: 1 }}>
      <SectionTitle>Association Backtrace Probe</SectionTitle>
      <InputRow
        value={iidInput}
        onChange={setIidInput}
        onSubmit={handleBacktrace}
        placeholder="Input infon iid to backtrace"
        buttonText="Backtrace"
        loading={querying}
        disabled={!iidInput.trim()}
      />

      {candidateIids.length > 0 && (
        <>
          <div style={{ fontSize: 10, color: C.textTer, marginBottom: 6 }}>Quick pick iid</div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 10 }}>
            {candidateIids.slice(0, 12).map(iid => (
              <button
                key={iid}
                onClick={() => handleQuickPick(iid)}
                style={{
                  fontSize: 9,
                  padding: '2px 6px',
                  borderRadius: 10,
                  border: `1px solid ${C.border}`,
                  background: selectedIid === iid ? C.accentLight : C.bgCard,
                  color: selectedIid === iid ? C.accent : C.textSec,
                  cursor: 'pointer',
                }}
              >
                {iidAlias(iid)}
              </button>
            ))}
          </div>
        </>
      )}

      <SectionTitle>Selected Backtrace</SectionTitle>
      {selectedIid ? (
        selectedBacktrace ? (
          <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 8, padding: '8px 10px', marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 700 }}><ReadableIid iid={selectedIid} color={C.text} /></div>
            <div style={{ marginTop: 3, fontSize: 10, color: C.textSec }}>
              evidence: {selectedBacktrace.evidence_pointer || 'n/a'}
            </div>
            <div style={{ marginTop: 6, fontSize: 10, color: C.textTer }}>
              associations: {selectedBacktrace.associations?.length || 0}
            </div>
            {selectedBacktrace.associations?.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 4 }}>
                {selectedBacktrace.associations.slice(0, 8).map((a, idx) => (
                  <div key={`${a.iid}_${idx}`} style={{ fontSize: 10, color: C.teal }}>
                    → <InfonChip iid={a.iid} resolve={resolve} color={C.teal} /> ({((a.similarity || 0) * 100).toFixed(1)}%)
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div style={{ marginBottom: 12 }}>
            <EmptyHint text={`No backtrace result for ${selectedIid}`} />
          </div>
        )
      ) : (
        <EmptyHint text="Query or quick-pick an iid to inspect backtrace result" />
      )}

      <SectionTitle>Association Timeline ({memoryAssociationEvents?.length || 0})</SectionTitle>
      {memoryAssociationEvents?.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {memoryAssociationEvents.map(evt => {
            const style = typeStyle(evt.type)
            return (
              <div key={evt.id} style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 8, padding: '7px 10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                  <Badge color={style.color} bg={style.bg}>{style.label}</Badge>
                  <span style={{ fontSize: 10, fontWeight: 600, color: C.text }}>{evt.title || evt.type}</span>
                  <span style={{ marginLeft: 'auto', fontSize: 9, color: C.textTer }}>{formatTs(evt.ts)}</span>
                </div>
                {evt.detail && <div style={{ fontSize: 10, color: C.textSec, marginBottom: 3 }}>{evt.detail}</div>}
                {(evt.payload?.trigger_types || []).length > 0 && (
                  <div style={{ fontSize: 9, color: C.purple }}>
                    trigger: {evt.payload.trigger_types.join(', ')}
                  </div>
                )}
                <IidList label="query" iids={evt.payload?.query_iids} color={C.textSec} resolve={resolve} />
                <IidList label="retrieved" iids={evt.payload?.retrieved_iids} color={C.teal} resolve={resolve} />
                <IidList label="assoc" iids={evt.payload?.association_iids} color={C.teal} resolve={resolve} />
                <IidList label="linked" iids={evt.payload?.linked_iids} color={C.green} resolve={resolve} />
              </div>
            )
          })}
        </div>
      ) : (
        <EmptyHint text="No association events yet" />
      )}
    </div>
  )
}

// ======================== Tab: Search ========================

function SearchTab() {
  const { searchMemoryStream, memoryStreamLoading } = useStore()
  const { resolve } = useInfonLookup()
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
                      <Badge color={getTypeBadgeColor(inf.infon_type)}>{String(inf.infon_type || '?').toUpperCase()}</Badge>
                      <span style={{ fontWeight: 600, fontSize: 11 }}>
                        {getInfonPrimaryText(inf) || iidAlias(inf.iid)}
                      </span>
                      {String(inf.infon_type || '').toUpperCase() !== 'REL' && (
                        <span style={{ color: C.textSec, fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {inf.attribute || inf.spatial || ''}
                        </span>
                      )}
                    </div>
                    <SimScore value={inf.similarity} />
                  </div>
                  {String(inf.infon_type || '').toUpperCase() === 'REL' && (
                    <div style={{ marginTop: 4, fontSize: 9, color: C.textSec }}>
                      <span style={{ color: C.textTer }}>args:</span>{' '}
                      {Array.isArray(inf.arg_refs) && inf.arg_refs.length > 0
                        ? inf.arg_refs.slice(0, 4).map((iid, idx) => (
                          <span key={`${iid}_${idx}`} style={{ marginRight: 6 }}>
                            <ReadableIid iid={iid} color={C.purple} />
                          </span>
                        ))
                        : <span style={{ color: C.textTer }}>n/a</span>
                      }
                    </div>
                  )}
                  <div style={{ fontSize: 9, color: C.textTer, marginTop: 3, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                    <span>iid: <ReadableIid iid={inf.iid} color={C.textSec} /></span>
                    <span>session: {inf.session_id}</span>
                    <span>R{inf.round_num}</span>
                    <span>{inf.modality}</span>
                  </div>
                  {inf.associations?.length > 0 && (
                    <div style={{ marginTop: 2 }}>
                      <div style={{ fontSize: 9, color: C.teal }}>
                        assoc:
                      </div>
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 2 }}>
                        {inf.associations.slice(0, 6).map((a, idx) => (
                          <span key={`${a.iid}_${idx}`} style={{ fontSize: 9, color: C.teal, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                            <InfonChip iid={a.iid} resolve={resolve} color={C.teal} /> ({(a.similarity * 100).toFixed(0)}%)
                          </span>
                        ))}
                      </div>
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
  DESC: { fill: '#dbeafe', stroke: '#2563eb', text: '#1d4ed8', label: 'DESC' },
  SCEN: { fill: '#dcfce7', stroke: '#15803d', text: '#166534', label: 'SCEN' },
  REL:  { fill: '#ede9fe', stroke: '#7c3aed', text: '#6d28d9', label: 'REL' },
  _default: { fill: '#e2e8f0', stroke: '#64748b', text: '#334155', label: '?' },
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

  const drawRoundedRect = (x, y, w, h, r) => {
    const rr = Math.max(0, Math.min(r, w / 2, h / 2))
    ctx.beginPath()
    ctx.moveTo(x + rr, y)
    ctx.lineTo(x + w - rr, y)
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr)
    ctx.lineTo(x + w, y + h - rr)
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h)
    ctx.lineTo(x + rr, y + h)
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr)
    ctx.lineTo(x, y + rr)
    ctx.quadraticCurveTo(x, y, x + rr, y)
    ctx.closePath()
  }

  const drawNode = (p) => {
    const tc = TYPE_COLORS[String(p.infon_type).toUpperCase()] || TYPE_COLORS._default
    const px = xScale(p.x)
    const py = yScale(p.y)
    const isSel = p.iid === selected
    const isAssoc = selectedAssocIids.has(p.iid)
    const isHov = p.iid === hoveredIid

    const showNodeLabel = isSel || isAssoc || isHov
    let label = ''
    let w = showNodeLabel ? (isSel ? 54 : 46) : 8
    let h = showNodeLabel ? (isSel ? 20 : 18) : 8
    const radius = showNodeLabel ? 5 : 2.5
    let fontSize = isSel ? 10 : 9

    if (showNodeLabel) {
      label = String(getEmbeddingLabel(p))
      const maxLen = isSel ? 18 : 14
      // 节点标签只保留开头，避免尾部把 iid/噪音又带回来
      label = shortenHead(label, maxLen)
      ctx.font = `${isSel ? 'bold ' : ''}${fontSize}px -apple-system, sans-serif`
      const tw = Math.min(120, ctx.measureText(label).width)
      w = Math.max(w, tw + 12)
      h = isSel ? 20 : 18
    }

    ctx.globalAlpha = showNodeLabel ? 1.0 : 0.75

    // 选中的光晕
    if (isSel) {
      ctx.fillStyle = 'rgba(14,165,233,0.16)'
      drawRoundedRect(px - w / 2 - 4, py - h / 2 - 4, w + 8, h + 8, 8)
      ctx.fill()
    }

    ctx.fillStyle = tc.fill
    ctx.strokeStyle = isSel ? '#0f172a' : isAssoc ? '#f59e0b' : tc.stroke
    ctx.lineWidth = isSel ? 2.5 : isAssoc ? 2 : 1

    // 数据量大时，未高亮节点进一步缩小
    if (!showNodeLabel && data.points.length > 500) {
      w = 6
      h = 6
    }
    if (!showNodeLabel && data.points.length > 2000) {
      w = 5
      h = 5
    }

    drawRoundedRect(px - w / 2, py - h / 2, w, h, radius)
    ctx.fill()
    ctx.stroke()

    // 选中 / 关联 / hover 时显示内嵌标签（文字标在节点内部）
    if (showNodeLabel) {
      ctx.globalAlpha = 1
      ctx.font = `${isSel ? 'bold ' : ''}${fontSize}px -apple-system, sans-serif`
      ctx.fillStyle = tc.text || '#0f172a'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(label, px, py, w - 10)
    }
  }

  // 先画普通节点，再画高亮节点，避免被小节点遮挡
  const normalNodes = []
  const focusNodes = []
  for (const p of data.points) {
    const isFocus = p.iid === selected || selectedAssocIids.has(p.iid) || p.iid === hoveredIid
    if (isFocus) focusNodes.push(p)
    else normalNodes.push(p)
  }
  normalNodes.forEach(drawNode)
  focusNodes.forEach(drawNode)

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
            <span style={{ width: 10, height: 7, borderRadius: 2, background: val.fill, border: `1px solid ${val.stroke}`, display: 'inline-block' }} />
            {val.label}
            {stats?.type_counts?.[key] != null && <span style={{ color: C.textTer }}>({stats.type_counts[key]})</span>}
          </span>
        ))}
        <span style={{ color: C.textTer, marginLeft: 8 }}>Edges:</span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
          <span style={{ width: 14, borderTop: '2px solid rgba(239,68,68,0.35)' }} />
          <span style={{ color: C.textTer }}>strong</span>
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
          <span style={{ width: 14, borderTop: '2px solid rgba(245,158,11,0.3)' }} />
          <span style={{ color: C.textTer }}>medium</span>
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
          <span style={{ width: 14, borderTop: '2px dashed rgba(148,163,184,0.5)' }} />
          <span style={{ color: C.textTer }}>weak</span>
        </span>
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
            <span style={{ fontWeight: 700, fontSize: 11 }}>
              {getEmbeddingLabel(selectedPoint) || selectedPoint.entity || ''}
            </span>
            {selectedPoint.attribute && <span style={{ color: C.textSec }}>: {selectedPoint.attribute}</span>}
          </div>
          <div style={{ color: C.textTer, fontSize: 9, marginBottom: 3 }}>
            iid: <ReadableIid iid={selectedPoint.iid} color={C.textTer} />
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
                  → {sanitizeDisplayText(getEmbeddingLabel(pointMap.get(a.iid) || { iid: a.iid })) || iidAlias(a.iid)} ({(a.similarity * 100).toFixed(0)}%)
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
  { key: 'map', label: 'User Profile' },
  { key: 'trace', label: 'Trace' },
  { key: 'store', label: 'Store' },
  { key: 'triggers', label: 'Triggers' },
  { key: 'search', label: 'Search' },
]

export default function MemoryStreamDebugPanel() {
  const [open, setOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('map')
  const panelRef = useRef(null)
  const fabRef = useRef(null)
  // 监听用户变化：用户切换时 key 变化 → 所有子组件强制重新挂载
  // 这样所有 useEffect 重新触发、本地状态重置、Canvas 清空
  const currentUserId = useStore(s => s.currentUserId)

  useEffect(() => {
    if (!open) return
    const handleKey = (e) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [open])

  // 点击面板外区域自动收起
  useEffect(() => {
    if (!open) return
    const handleOutside = (e) => {
      const target = e.target
      if (panelRef.current?.contains(target)) return
      if (fabRef.current?.contains(target)) return
      setOpen(false)
    }
    window.addEventListener('mousedown', handleOutside)
    window.addEventListener('touchstart', handleOutside, { passive: true })
    return () => {
      window.removeEventListener('mousedown', handleOutside)
      window.removeEventListener('touchstart', handleOutside)
    }
  }, [open])

  return (
    <>
      {/* FAB toggle */}
      <button
        ref={fabRef}
        onClick={() => setOpen(v => !v)}
        style={{
          ...fabStyle,
          borderColor: open ? C.accent + '80' : C.borderMed,
          boxShadow: open ? '0 4px 14px rgba(14,165,233,0.18)' : fabStyle.boxShadow,
          background: open ? C.accentLight : C.bg,
        }}
        title={open ? 'Close Memory Stream panel' : 'Open Memory Stream panel'}
      >
        <PanelToggleIcon open={open} />
      </button>

      {/* Panel */}
      {open && (
        <div ref={panelRef} style={panelStyle}>
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
            {activeTab === 'trace' && <TraceTab />}
            {activeTab === 'store' && <StoreTab />}
            {activeTab === 'triggers' && <TriggersTab />}
            {activeTab === 'search' && <SearchTab />}
          </div>
        </div>
      )}
    </>
  )
}
