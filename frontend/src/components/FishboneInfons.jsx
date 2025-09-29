import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'

// 鱼骨图（中文注释）：基于当前会话的流式信息元提取，实时绘制时间轴鱼骨
export default function FishboneInfons() {
  const { getCurrentSession, infonSessions } = useStore()
  const session = getCurrentSession()
  const runs = useMemo(() => (session ? (infonSessions?.[session.id]?.runs || []) : []), [session, infonSessions])
  // 选中信息元（中文注释）：点击节点后在下方表格显示
  const [selectedInfon, setSelectedInfon] = useState(null)

  // 容器尺寸观测（中文注释）：自适应宽度计算布局
  const containerRef = useRef(null)
  const [containerWidth, setContainerWidth] = useState(640)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.max(320, Math.floor(entry.contentRect.width))
        setContainerWidth(w)
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // 将 run 归入“对话轮次”（中文注释）：每个用户消息作为一轮；pending 属于下一轮的空心主分支
  const rounds = useMemo(() => {
    if (!session) return []
    const messages = Array.isArray(session.messages) ? session.messages : []
    const userMsgs = messages.filter((m) => m.role === 'user')
    const messageById = new Map(messages.map((m) => [m.id, m]))
    // 将所有消息映射到其所在“轮”（中文注释）：每个用户消息启动新轮，后续助手消息归属该轮
    const messageIdToRound = new Map()
    let roundIdx = -1
    for (const m of messages) {
      if (m.role === 'user') roundIdx++
      if (roundIdx < 0) roundIdx = 0
      messageIdToRound.set(m.id, roundIdx)
    }

    const buckets = new Map() // roundIndex -> { index, pending, userRuns: [], assistantRuns: [] }
    const ensureBucket = (index, pending) => {
      if (!buckets.has(index)) buckets.set(index, { index, pending: !!pending, userRuns: [], assistantRuns: [] })
      const b = buckets.get(index)
      // 如果已有 bucket，pending 取或（中文注释）
      b.pending = b.pending || !!pending
      return b
    }

    for (const r of runs) {
      if (!r) continue
      if (r.targetType === 'message') {
        let idx = messageIdToRound.get(r.targetKey)
        if (typeof idx !== 'number') idx = Math.max(0, userMsgs.length - 1)
        const msg = messageById.get(r.targetKey)
        const bucket = ensureBucket(idx, false)
        if (msg && msg.role === 'assistant') bucket.assistantRuns.push(r)
        else bucket.userRuns.push(r)
      } else if (r.targetType === 'pending') {
        const idx = userMsgs.length // 下一轮（尚未发送）
        ensureBucket(idx, true).userRuns.push(r)
      }
    }

    const list = Array.from(buckets.values()).sort((a, b) => a.index - b.index)
    // 计算该轮的总信息元数量（中文注释）
    list.forEach((b) => {
      let c = 0
      for (const r of [...b.userRuns, ...b.assistantRuns]) {
        c += Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons.length : 0
      }
      b.infonCount = c
    })
    return list
  }, [session, runs])

  // 提取信息元关键词（中文注释）：按类型选择代表性字段
  const getInfonKeyword = (infon) => {
    if (!infon || typeof infon !== 'object') return 'Unknown'
    const t = String(infon.infon_type || '').toUpperCase()
    if (t === 'IND') {
      if (Array.isArray(infon.names) && infon.names.length) return String(infon.names[0])
      return 'IND'
    }
    if (t === 'PAR') return String(infon.value ?? 'PAR')
    if (t === 'TIM') return String(infon.temporal_value ?? 'TIM')
    if (t === 'LOC') return String(infon.spatial_value ?? (Array.isArray(infon.bbox) ? 'bbox' : 'LOC'))
    if (t === 'REL') return String(infon.relation_name ?? 'REL')
    if (t === 'TYP') return String(infon.type_name ?? 'TYP')
    if (t === 'SIT') return 'SIT'
    return t || 'Unknown'
  }

  // 文本测量（中文注释）：使用离屏 canvas 估计标签宽度，fallback 到字符粗略估计
  const measureText = (() => {
    let ctx = null
    let font = '11px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Noto Sans, "Apple Color Emoji", "Segoe UI Emoji"'
    return (text) => {
      try {
        if (!ctx) {
          const canvas = document.createElement('canvas')
          ctx = canvas.getContext('2d')
          ctx.font = font
        }
        ctx.font = font
        const m = ctx.measureText(String(text || ''))
        return Math.ceil(m.width)
      } catch (_) {
        const s = String(text || '')
        // 粗略：ASCII ~6px/char，CJK ~11px/char（中文注释）
        let w = 0
        for (const ch of s) {
          const code = ch.charCodeAt(0)
          w += (code < 128) ? 6 : 11
        }
        return w
      }
    }
  })()

  // 将文本裁剪到指定像素宽度，超出以省略号结尾（中文注释）
  function truncateToPx(text, maxPx) {
    const s = String(text || '')
    if (maxPx <= 0) return '…'
    if (measureText(s) <= maxPx) return s
    let out = ''
    for (let i = 0; i < s.length; i++) {
      const next = out + s[i]
      if (measureText(next + '…') > maxPx) break
      out = next
    }
    return out ? (out + '…') : '…'
  }

  // 几何参数（中文注释）：竖直时间轴 + 45° 主分叉（上下交错）
  const width = containerWidth
  const leftPad = 24
  const rightPad = 24
  const topPad = 16
  const bottomPad = 24
  const diagOffset = Math.max(90, Math.min(220, Math.floor(width * 0.28))) // 主分叉在 x/y 各偏移
  const mainDiagLen = diagOffset // 45° 对角线，dx=dy
  const rowGap = Math.max(110, Math.floor(diagOffset * 0.9)) // 轴上相邻锚点间距
  const subSlantLen = 16 // 小分叉短斜长度
  const maxNodesPerRun = 8
  const nodeSize = 12 // 节点方块尺寸（中文注释）
  const labelGap = 8 // 节点到文本的水平间距（中文注释）
  const labelMinGap = 10 // 相邻文本间的最小间隔（中文注释）
  const leafDiagStep = 16 // 叶子沿主分叉的最小间距（像素，按对角线方向）（中文注释）

  // 预先计算整体高度（中文注释）：扫描所有分叉端点 y 范围
  const axisX = Math.round(width / 2)
  const startY = topPad + diagOffset // 顶部预留，避免向上的分叉超出
  let topMost = topPad
  let bottomMost = startY
  for (let ri = 0; ri < rounds.length; ri++) {
    const anchorY = startY + ri * rowGap
    const endY = anchorY - mainDiagLen // 右侧分支也向上45°（中文注释）
    const hasRuns = (((rounds[ri]?.userRuns || []).length) + ((rounds[ri]?.assistantRuns || []).length)) > 0
    if (!hasRuns) {
      topMost = Math.min(topMost, endY)
      bottomMost = Math.max(bottomMost, anchorY)
    } else {
      // 即使没有 infons 也要考虑小分叉短斜（中文注释）
      topMost = Math.min(topMost, endY - subSlantLen)
      bottomMost = Math.max(bottomMost, anchorY)
    }
  }
  const height = Math.max(bottomMost + bottomPad, startY + bottomPad)

  if (!rounds.length) {
    return (
      <div ref={containerRef} className={styles.fishboneRoot}>
        <div className={styles.infonEmpty}>No inference yet</div>
      </div>
    )
  }

  // 计算主轴锚点（中文注释）：沿竖直轴等距分布
  const anchorStepY = rowGap

  return (
    <div ref={containerRef} className={styles.fishboneRoot}>
      <svg className={styles.fishboneSvg} width="100%" height={height}>
        {/* 竖直时间轴（中文注释） */}
        <line x1={axisX} y1={topPad / 2} x2={axisX} y2={height - bottomPad / 2} className={styles.fishboneAxis} />

        {rounds.map((round, ri) => {
          const anchorX = axisX
          const anchorY = startY + ri * anchorStepY
          const sideLeft = (ri % 2) === 0 // 左右交替（中文注释）
          const endX = sideLeft ? (anchorX - mainDiagLen) : (anchorX + mainDiagLen)
          const endY = anchorY - mainDiagLen // 一律向上45°（中文注释）
          const clusterR = Math.max(12, Math.min(30, 12 + (round.infonCount || 0) * 1.2))
          const runsInRound = [...round.userRuns].sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0))
          const countRuns = runsInRound.length

          // 45 度直线参数化（中文注释）
          const dx = endX - anchorX
          const dy = endY - anchorY

          return (
            <g key={`round-${ri}`}>
              {/* 主分叉（中文注释） */}
              <line x1={anchorX} y1={anchorY} x2={endX} y2={endY} className={styles.fishboneBranch} />

              {/* 轴-主分叉交点的空心/实心圆（中文注释） */}
              <circle cx={anchorX} cy={anchorY} r={clusterR} className={`${styles.fishboneCluster} ${round.pending ? styles.fishboneClusterHollow : styles.fishboneClusterSolid}`} />
              {/* 模型回复（assistantRuns）的信息元：绘制在圆形内部（中文注释） */}
              {(() => {
                const asRuns = [...round.assistantRuns]
                const infons = asRuns.flatMap((r) => Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons : [])
                const maxN = 10
                const shown = infons.slice(0, maxN)
                if (!shown.length) return null
                const s = 8 // 方块边长
                const g = 6 // 间距
                const cols = Math.min(3, shown.length)
                const rows = Math.ceil(shown.length / cols)
                const gridW = cols * s + (cols - 1) * g
                const gridH = rows * s + (rows - 1) * g
                const startX = anchorX - gridW / 2
                const startY2 = anchorY - gridH / 2
                return (
                  <g>
                    {shown.map((infon, idx) => {
                      const col = idx % cols
                      const row = Math.floor(idx / cols)
                      const nx = Math.round(startX + col * (s + g))
                      const ny = Math.round(startY2 + row * (s + g))
                      const label = getInfonKeyword(infon)
                      return (
                        <g key={`as-${ri}-${idx}`}>
                          <rect x={nx - s / 2} y={ny - s / 2} width={s} height={s} rx={1.5} ry={1.5} className={`${styles.fishboneNodeSquare} ${styles.fishboneClickable}`} onClick={() => setSelectedInfon(infon)} />
                          <title>{label}</title>
                        </g>
                      )
                    })}
                  </g>
                )
              })()}

              {/* 小分叉：每个信息元独立叶子分支（中文注释） */}
              {countRuns > 0 ? runsInRound.map((run, i) => {
                const tRun = (i + 1) / (countRuns + 1)
                const baseX = Math.round(anchorX + dx * tRun)
                const baseY = Math.round(anchorY + dy * tRun)
                const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
                const list = infons.slice(0, maxNodesPerRun)
                const tStep = leafDiagStep / Math.max(1, mainDiagLen)

                // 生成对称偏移序列（中文注释）：中心优先，向两侧展开
                const offsets = list.map((_, idx) => idx - (list.length - 1) / 2)

                const leaves = list.map((infon, idx) => {
                  const t = Math.max(0.05, Math.min(0.95, tRun + offsets[idx] * tStep))
                  const bx = Math.round(anchorX + dx * t)
                  const by = Math.round(anchorY + dy * t)
                  const slantEndX = bx + (sideLeft ? -subSlantLen : subSlantLen)
                  const slantEndY = by - subSlantLen
                  const label = getInfonKeyword(infon)
                  const labelW = measureText(label)
                  const baseNeed = (nodeSize / 2) + 4 + labelGap + labelW + 8
                  const stubMax = sideLeft ? (slantEndX - leftPad) : (width - rightPad - slantEndX)
                  const stubLen = Math.max(60, Math.min(stubMax, baseNeed))
                  const stubEndX = slantEndX + (sideLeft ? -stubLen : stubLen)
                  const stubEndY = slantEndY

                  const nx = stubEndX
                  const ny = stubEndY

                  const maxLabelPx = Math.max(0, stubLen - ((nodeSize / 2) + 4 + labelGap))
                  const textShown = truncateToPx(label, maxLabelPx)
                  const confidence = Math.max(0, Math.min(1, Number(infon?.confidence ?? 0)))
                  const barW = nodeSize * 1.6
                  const barX = nx - barW / 2
                  const barY = ny + nodeSize / 2 + 2

                  return { bx, by, slantEndX, slantEndY, stubEndX, stubEndY, nx, ny, textShown, confidence, infon }
                })

                // 运行中：在该 run 的末尾再放置一个旋转圈（中文注释）
                const spinnerPos = (() => {
                  if (run.status !== 'running') return null
                  if (leaves.length > 0) {
                    const last = leaves[leaves.length - 1]
                    return { x: last.stubEndX + (sideLeft ? -8 : 8), y: last.stubEndY }
                  }
                  const slantEndX = baseX + (sideLeft ? -subSlantLen : subSlantLen)
                  const slantEndY = baseY - subSlantLen
                  const stubEndX = slantEndX + (sideLeft ? -60 : 60)
                  const stubEndY = slantEndY
                  return { x: stubEndX + (sideLeft ? -8 : 8), y: stubEndY }
                })()

                return (
                  <g key={run.id}>
                    {leaves.map((leaf, li) => (
                      <g key={`${run.id}-leaf-${li}`}>
                        <line x1={leaf.bx} y1={leaf.by} x2={leaf.slantEndX} y2={leaf.slantEndY} className={styles.fishboneSubBranch} />
                        <line x1={leaf.slantEndX} y1={leaf.slantEndY} x2={leaf.stubEndX} y2={leaf.stubEndY} className={styles.fishboneSubStub} />
                        <rect x={leaf.nx - nodeSize / 2} y={leaf.ny - nodeSize / 2} width={nodeSize} height={nodeSize} rx={2} ry={2} className={`${styles.fishboneNodeSquare} ${styles.fishboneClickable}`} onClick={() => setSelectedInfon(leaf.infon)} />
                        <rect x={leaf.nx - nodeSize * 0.8} y={leaf.ny + nodeSize / 2 + 2} width={nodeSize * 1.6} height={2} rx={1} ry={1} className={styles.fishboneProgressBg} />
                        <rect x={leaf.nx - nodeSize * 0.8} y={leaf.ny + nodeSize / 2 + 2} width={nodeSize * 1.6 * leaf.confidence} height={2} rx={1} ry={1} className={styles.fishboneProgressFill} />
                        <text x={leaf.nx + (sideLeft ? -labelGap : labelGap)} y={leaf.ny} dominant-baseline="middle" className={styles.fishboneNodeText} textAnchor={sideLeft ? 'end' : 'start'}>{leaf.textShown}</text>
                      </g>
                    ))}
                    {spinnerPos ? (
                      <g className={styles.fishboneSpinnerGroup} transform={`translate(${spinnerPos.x},${spinnerPos.y})`}>
                        <circle cx={0} cy={0} r={6} className={styles.fishboneSpinnerCircle} />
                      </g>
                    ) : null}
                  </g>
                )
              }) : null}
            </g>
          )
        })}
      </svg>
      {selectedInfon ? (
        <div className={styles.fishboneDetails}>
          <div className={styles.fishboneDetailsTitle}>Infon details</div>
          <table className={styles.fishboneTable}>
            <tbody>
              {(() => {
                const preferred = [
                  'iid','infon_type','record_time','occur_time','confidence','support',
                  'names','value','data_type','temporal_value','granularity','spatial_value','bbox',
                  'relation_name','arity','arg_refs','arg_types','type_name','category','modality','context_span'
                ]
                const keys = Array.from(new Set([...preferred, ...Object.keys(selectedInfon || {})]))
                return keys.map((k) => {
                  if (!(k in (selectedInfon || {}))) return null
                  const v = selectedInfon[k]
                  const isString = typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
                  const str = isString ? String(v) : JSON.stringify(v, null, 2)
                  return (
                    <tr key={k}>
                      <td className={styles.fishboneKey}>{k}</td>
                      <td className={styles.fishboneValue}>{isString ? str : (<pre className={styles.fishbonePre}>{str}</pre>)}</td>
                    </tr>
                  )
                })
              })()}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  )
}


