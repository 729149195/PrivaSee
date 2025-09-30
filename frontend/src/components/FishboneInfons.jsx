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
  const bottomPad = 10
  const diagOffset = Math.max(90, Math.min(220, Math.floor(width * 0.28))) // 主分叉在 x/y 各偏移
  const mainDiagLen = diagOffset // 45° 对角线，dx=dy
  const rowGap = Math.max(100, Math.floor(diagOffset)) // 轴上相邻锚点间距
  const subSlantLen = 16 // 小分叉短斜长度
  const maxNodesPerRun = Infinity
  const nodeSize = 12 // 节点方块尺寸（中文注释）
  const labelGap = 12 // 节点到文本的水平间距（中文注释）
  const labelMinGap = 18 // 相邻文本间的最小间隔（中文注释）
  const leafDiagStep = 16 // 叶子沿主分叉的最小间距（像素，按对角线方向）（中文注释）
  const subTrunkStart = 40 // 次级主分支起始到第一叶子的距离（px）（中文注释）
  const subTrunkStep = Math.max(18, Math.min(32, Math.floor(width * 0.03))) // 自适应密度（中文注释）
  const leafLen = Math.max(18, Math.min(34, Math.floor(width * 0.03) + 8)) // 自适应叶子长度（中文注释）

  // 预先计算整体高度（中文注释）：扫描所有分叉端点 y 范围
  const axisX = Math.round(width / 2)
  const startY = topPad + diagOffset // 顶部预留，避免向上的分叉超出
  let topMost = topPad
  let bottomMost = startY
  for (let ri = 0; ri < rounds.length; ri++) {
    const anchorY = startY + ri * rowGap
    const endY = anchorY - mainDiagLen // 45° 主分叉向上（中文注释）
    const runsInRound = [...(rounds[ri]?.userRuns || [])]
    const maxInf = runsInRound.reduce((m, r) => Math.max(m, Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons.length : 0), 0)
    const trunkLen = subTrunkStart + subTrunkStep * Math.max(0, maxInf - 1)
    const topCandidate = endY - trunkLen - leafLen
    // 同时考虑中轴大气泡半径（中文注释）
    const infonCount = rounds[ri]?.infonCount || 0
    const clusterR_est = Math.max(12, Math.min(30, 12 + infonCount * 1.2))
    topMost = Math.min(topMost, topCandidate, anchorY - clusterR_est)
    bottomMost = Math.max(bottomMost, anchorY + clusterR_est)
  }
  // 将顶部留白强制调节为“2.5 个节点高度”（中文注释）：允许向上/向下微移
  const desiredTopMargin = Math.round(nodeSize * 0.5)
  const vShift = (-35 - topMost)
  const height = Math.max(bottomMost + bottomPad + vShift, startY + bottomPad + vShift)

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
        <line x1={axisX} y1={topPad / 2 + vShift} x2={axisX} y2={height - bottomPad / 2} className={styles.fishboneAxis} />

        {rounds.map((round, ri) => {
          const anchorX = axisX
          const anchorY = (startY + vShift) + ri * anchorStepY
          const sideLeft = (ri % 2) === 0 // 左右交替（中文注释）
          const endX = sideLeft ? (anchorX - mainDiagLen) : (anchorX + mainDiagLen)
          const endY = anchorY - mainDiagLen // 一律向上45°（中文注释）
          const clusterR = Math.max(12, Math.min(30, 12 + (round.infonCount || 0) * 1.2))
          const runsInRound = [...round.userRuns].sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0))
          const countRuns = runsInRound.length

          // 45 度直线参数化（中文注释）
          const dx = endX - anchorX
          const dy = endY - anchorY

          // 自适应：根据该轮所有 run 的标签宽度，放大“次主分支”之间的间距，避免文本遮盖（中文注释）
          const maxLabelAcrossRuns = (() => {
            let maxW = 0
            for (const run of runsInRound) {
              const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
              for (const infon of infons) {
                const t = String(infon?.infon_type || '').toUpperCase()
                if (t === 'SIT') continue
                const label = getInfonKeyword(infon)
                const maxLabelPx = Math.max(40, leafLen - (nodeSize / 2 + 4 + labelGap))
                const textShown = truncateToPx(label, maxLabelPx)
                const w = measureText(textShown)
                if (w > maxW) maxW = w
              }
            }
            return maxW
          })()

          // 将像素级最小步长投影到主分叉对角线的参数 t（中文注释）
          const minRunStepPx = Math.max(leafDiagStep, Math.floor(maxLabelAcrossRuns + labelGap))
          const minRunStepT = minRunStepPx / Math.max(1, mainDiagLen)
          const uniformDeltaT = countRuns > 0 ? (1 / (countRuns + 1)) : 1
          const requiredT = minRunStepT * countRuns
          // 如果空间允许，采用更大的步长；否则回退到均匀分布（中文注释）
          const deltaTRun = requiredT < 1 ? Math.max(minRunStepT, uniformDeltaT) : uniformDeltaT

          return (
            <g key={`round-${ri}`}>
              {/* 主分叉（中文注释） */}
              <line x1={anchorX} y1={anchorY} x2={endX} y2={endY} className={styles.fishboneBranch} />

              {/* 轴-主分叉交点的大气泡（中文注释） */}
              <circle cx={anchorX} cy={anchorY} r={clusterR} className={`${styles.fishboneCluster} ${round.pending ? styles.fishboneClusterHollow : styles.fishboneClusterSolid}`} />
              {/* 模型回复（assistantRuns）的信息元：绘制在圆形内部（中文注释） */}
              {(() => {
                const asRuns = [...round.assistantRuns]
                const infons = asRuns.flatMap((r) => Array.isArray(r?.resultJson?.infons) ? r.resultJson.infons : [])
                const maxN = 24
                const shown = infons.slice(0, maxN)
                if (!shown.length) return null
                const rPad = 3
                const rLarge = Math.max(12, clusterR - 1)
                // 根据数量自适应缩放小气泡半径（中文注释）
                let rSmall = Math.max(4, Math.min(8, Math.floor(rLarge * 0.24)))
                rSmall = Math.max(3, Math.min(rSmall, Math.floor((rLarge * 0.5) / Math.sqrt(shown.length + 1))))
                const rAvail = Math.max(2, rLarge - rPad - rSmall)

                // 初始位置：Fermat 螺旋（黄金角）+ 半径开方分布（中文注释）
                const golden = Math.PI * (3 - Math.sqrt(5))
                const pts = shown.map((_, idx) => {
                  const t = (idx + 0.5) / Math.max(1, shown.length)
                  const angle = idx * golden
                  const rad = Math.sqrt(t) * rAvail
                  return { x: anchorX + rad * Math.cos(angle), y: anchorY + rad * Math.sin(angle) }
                })

                // 位置松弛：简单斥力+向心+边界收敛，确保最小间距（中文注释）
                const minDist = rSmall * 2 + 2
                const minDist2 = minDist * minDist
                const iters = Math.min(60, 20 + shown.length * 2)
                for (let iter = 0; iter < iters; iter++) {
                  for (let i = 0; i < pts.length; i++) {
                    for (let j = i + 1; j < pts.length; j++) {
                      let dx = pts[j].x - pts[i].x
                      let dy = pts[j].y - pts[i].y
                      let d2 = dx * dx + dy * dy
                      if (d2 < 1e-6) { dx = (Math.random() - 0.5) * 0.01; dy = (Math.random() - 0.5) * 0.01; d2 = dx * dx + dy * dy }
                      if (d2 < minDist2) {
                        const d = Math.sqrt(d2)
                        const push = (minDist - d) * 0.5
                        const ux = dx / d
                        const uy = dy / d
                        pts[i].x -= ux * push
                        pts[i].y -= uy * push
                        pts[j].x += ux * push
                        pts[j].y += uy * push
                      }
                    }
                  }
                  for (let i = 0; i < pts.length; i++) {
                    // 向心
                    pts[i].x += (anchorX - pts[i].x) * 0.05
                    pts[i].y += (anchorY - pts[i].y) * 0.05
                    // 边界约束
                    const dx = pts[i].x - anchorX
                    const dy = pts[i].y - anchorY
                    const d = Math.sqrt(dx * dx + dy * dy) || 1
                    if (d > rAvail) {
                      const s = rAvail / d
                      pts[i].x = anchorX + dx * s
                      pts[i].y = anchorY + dy * s
                    }
                  }
                }

                return (
                  <g>
                    <circle cx={anchorX} cy={anchorY} r={rLarge} className={styles.fishboneBubbleLargeRing} />
                    {shown.map((infon, idx) => {
                      const nx = Math.round(pts[idx].x)
                      const ny = Math.round(pts[idx].y)
                      const label = getInfonKeyword(infon)
                      return (
                        <g key={`as-${ri}-${idx}`}>
                          <circle cx={nx} cy={ny} r={rSmall} className={`${styles.fishboneBubbleSmall} ${styles.fishboneClickable}`} onClick={() => setSelectedInfon(infon)} />
                          <title>{label}</title>
                        </g>
                      )
                    })}
                  </g>
                )
              })()}

              {/* 小分叉：每个信息元独立叶子分支（中文注释） */}
              {countRuns > 0 ? runsInRound.map((run, i) => {
                const tRun = deltaTRun * (i + 1)
                const baseX = Math.round(anchorX + dx * tRun)
                const baseY = Math.round(anchorY + dy * tRun)
                const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
                const listAll = infons
                // 将 infon 按类型分组：SIT 在交界点；其余（包含 REL）作为叶子，REL 渲染为圆形（中文注释）
                const sits = []
                const leavesInput = [] // { infon, isRel }
                for (let idx = 0; idx < listAll.length; idx++) {
                  const infon = listAll[idx]
                  const t = String(infon?.infon_type || '').toUpperCase()
                  if (t === 'SIT') sits.push(infon)
                  else leavesInput.push({ infon, isRel: t === 'REL' })
                }
                // 构建竖直次级主分支，并沿其布置 45° 叶子（中文注释）
                const trunkLen = subTrunkStart + subTrunkStep * Math.max(0, leavesInput.length - 1)
                const trunkTopX = baseX
                const trunkTopY = baseY - trunkLen
                // 叶子：记录文本包围盒（中文注释）
                const leaves = leavesInput.map((item, idx) => {
                  const infon = item.infon
                  const attachY = baseY - (subTrunkStart + idx * subTrunkStep)
                  const attachX = baseX
                  const nx = attachX + (sideLeft ? -leafLen : leafLen)
                  const ny = attachY - leafLen
                  const label = getInfonKeyword(infon)
                  const maxLabelPx = Math.max(40, leafLen - (nodeSize / 2 + 4 + labelGap))
                  const textShown = truncateToPx(label, maxLabelPx)
                  const confidence = Math.max(0, Math.min(1, Number(infon?.confidence ?? 0)))
                  // 文本包围盒估计（中文注释）
                  const textWidth = measureText(textShown)
                  const textHeight = 12
                  const textX = nx + (sideLeft ? -labelGap : labelGap)
                  const textY = ny
                  const boxLeft = sideLeft ? (textX - textWidth) : textX
                  const boxRight = sideLeft ? textX : (textX + textWidth)
                  const boxTop = textY - textHeight / 2
                  const boxBottom = textY + textHeight / 2
                  return { trunkX: attachX, trunkY: attachY, nx, ny, textShown, confidence, infon, textX, textY, textWidth, textHeight, boxLeft, boxRight, boxTop, boxBottom, isRel: !!item.isRel }
                })

                // SIT 情景：在交界点绘制圆形与 description 文本（中文注释）
                const sitLabels = sits.map((infon) => {
                  const label = String(infon?.description ?? 'SIT')
                  const textShown = truncateToPx(label, Math.max(60, Math.floor(leafLen * 2)))
                  const textWidth = measureText(textShown)
                  const textHeight = 12
                  const textX = baseX + (sideLeft ? -labelGap : labelGap)
                  const textY = baseY
                  const boxLeft = sideLeft ? (textX - textWidth) : textX
                  const boxRight = sideLeft ? textX : (textX + textWidth)
                  const boxTop = textY - textHeight / 2
                  const boxBottom = textY + textHeight / 2
                  return { infon, textShown, textX, textY, textWidth, textHeight, boxLeft, boxRight, boxTop, boxBottom }
                })

                // 运行中：在该 run 的末尾再放置一个旋转圈（中文注释）
                const spinnerPos = (() => {
                  if (run.status !== 'running') return null
                  if (leaves.length > 0) {
                    const last = leaves[leaves.length - 1]
                    return { x: last.nx + (sideLeft ? -8 : 8), y: last.ny }
                  }
                  const nx = baseX + (sideLeft ? -leafLen : leafLen)
                  const ny = baseY - leafLen
                  return { x: nx + (sideLeft ? -8 : 8), y: ny }
                })()

                return (
                  <g key={run.id}>
                    {/* 竖直次级主分支 */}
                    <line x1={baseX} y1={baseY} x2={trunkTopX} y2={trunkTopY} className={styles.fishboneSubTrunk} />
                    {/* 信息元节点：常规方形，REL 为圆形；均带文本与进度（中文注释）*/}
                    {leaves.map((leaf, li) => (
                      <g key={`${run.id}-leaf-${li}`}>
                        <line x1={leaf.trunkX} y1={leaf.trunkY} x2={leaf.nx} y2={leaf.ny} className={styles.fishboneSubBranch} />
                        {leaf.isRel ? (
                          <circle cx={leaf.nx} cy={leaf.ny} r={nodeSize / 2} className={`${styles.fishboneRelCircle} ${styles.fishboneClickable}`} onClick={() => setSelectedInfon(leaf.infon)} />
                        ) : (
                          <rect x={leaf.nx - nodeSize / 2} y={leaf.ny - nodeSize / 2} width={nodeSize} height={nodeSize} rx={2} ry={2} className={`${styles.fishboneNodeSquare} ${styles.fishboneClickable}`} onClick={() => setSelectedInfon(leaf.infon)} />
                        )}
                        <text x={leaf.textX} y={leaf.textY} dominantBaseline="middle" className={styles.fishboneNodeText} textAnchor={sideLeft ? 'end' : 'start'}>{leaf.textShown}</text>
                      </g>
                    ))}
                    {/* SIT 情景节点：交界点圆形 + description 文本（中文注释） */}
                    {sitLabels.map((s, si) => (
                      <g key={`${run.id}-sit-${si}`} className={styles.fishboneClickable} onClick={() => setSelectedInfon(s.infon)}>
                        <circle cx={baseX} cy={baseY} r={nodeSize / 2} className={styles.fishboneSITCircle} />
                        <text x={s.textX} y={s.textY} dominantBaseline="middle" className={styles.fishboneSITText} textAnchor={sideLeft ? 'end' : 'start'}>{s.textShown}</text>
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
                  'relation_name','arity','arg_refs','arg_types','type_name','category','description','context_span'
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


