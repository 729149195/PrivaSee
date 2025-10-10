import React, { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { useStore } from '../store'

const LAWS = [
  { key: 'PIPL', label: 'PIPL', file: './law/PIPL.json' },
  { key: 'GDPR', label: 'GDPR', file: './law/GDPR.json' },
  { key: 'CCPA_CPRA', label: 'CCPA/CPRA', file: './law/CCPA_CPRA.json' },
]

async function fetchLawData(file) {
  const res = await fetch(file)
  return await res.json()
}

export default function LawTree() {
  const [lawIdx, setLawIdx] = useState(0)
  const [lawData, setLawData] = useState([null, null, null])
  const containerRef = useRef(null)
  const svgRef = useRef(null)
  const [size, setSize] = useState({ width: 928, height: 600 })
  
  // 从 store 获取推理结果和相关方法（中文注释）
  const { getCurrentSession, privacyInferences, setSelectedLaw } = useStore()
  const session = getCurrentSession()
  const inference = useMemo(() => (session ? privacyInferences?.[session.id] : null), [session, privacyInferences])

  // 预加载三份数据
  useEffect(() => {
    LAWS.forEach((law, idx) => {
      if (!lawData[idx]) {
        fetchLawData(law.file).then(data => {
          setLawData(prev => {
            const arr = [...prev]
            arr[idx] = data
            return arr
          })
        })
      }
    })
    // eslint-disable-next-line
  }, [])
  
  // 当法律数据或索引变化时，更新 store 中的选中法律（中文注释）
  useEffect(() => {
    if (lawData[lawIdx]) {
      setSelectedLaw(LAWS[lawIdx].key, lawData[lawIdx])
    }
  }, [lawIdx, lawData, setSelectedLaw])

  // 容器自适应
  useEffect(() => {
    const update = () => {
      if (!containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const w = Math.max(320, Math.floor(rect.width))
      const h = Math.max(400, Math.floor(rect.width * 0.65))
      setSize({ width: w, height: h })
    }
    update()
    const ro = new ResizeObserver(update)
    if (containerRef.current) ro.observe(containerRef.current)
    window.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [])

  // 构建风险映射：精确匹配到最小叶子节点（中文注释）
  const riskMap = useMemo(() => {
    const map = new Map() // key: 节点路径或名称, value: { level, confidence, risks, isLeaf }
    
    if (!inference || !inference.risks || !lawData[lawIdx]) return map
    
    // 构建法律树的所有节点路径映射
    const nodePathMap = new Map() // key: 完整路径字符串, value: 节点对象
    const nodeNameMap = new Map() // key: 节点名称, value: 节点对象数组（可能有重名）
    
    function traverseTree(node, path = []) {
      const currentPath = [...path, node.name]
      const pathKey = currentPath.join(' > ')
      
      nodePathMap.set(pathKey, { node, path: currentPath, isLeaf: !node.children || node.children.length === 0 })
      
      // 按名称索引（支持查找）
      if (!nodeNameMap.has(node.name)) {
        nodeNameMap.set(node.name, [])
      }
      nodeNameMap.get(node.name).push({ node, path: currentPath, isLeaf: !node.children || node.children.length === 0 })
      
      if (node.children) {
        node.children.forEach(child => traverseTree(child, currentPath))
      }
    }
    
    traverseTree(lawData[lawIdx])
    
    // 匹配推理结果到法律节点
    inference.risks.forEach(risk => {
      const lawPath = risk.law_path || ''
      const nodeName = risk.law_node_name || ''
      
      let matchedNode = null
      
      // 策略1：优先使用完整路径精确匹配
      if (lawPath) {
        // 尝试完全匹配
        if (nodePathMap.has(lawPath)) {
          matchedNode = nodePathMap.get(lawPath)
        } else {
          // 尝试模糊匹配（路径可能格式不同）
          const normalizedPath = lawPath.replace(/\s*[>›→]\s*/g, ' > ').trim()
          if (nodePathMap.has(normalizedPath)) {
            matchedNode = nodePathMap.get(normalizedPath)
          } else {
            // 尝试部分路径匹配（从最后一级开始往上）
            for (const [key, value] of nodePathMap.entries()) {
              if (key.endsWith(nodeName) || normalizedPath.includes(key)) {
                matchedNode = value
                break
              }
            }
          }
        }
      }
      
      // 策略2：使用节点名称匹配，优先匹配叶子节点
      if (!matchedNode && nodeName) {
        const candidates = nodeNameMap.get(nodeName) || []
        if (candidates.length > 0) {
          // 优先选择叶子节点
          const leafNodes = candidates.filter(c => c.isLeaf)
          matchedNode = leafNodes.length > 0 ? leafNodes[0] : candidates[0]
        } else {
          // 尝试部分匹配
          for (const [name, nodes] of nodeNameMap.entries()) {
            if (name.includes(nodeName) || nodeName.includes(name)) {
              const leafNodes = nodes.filter(n => n.isLeaf)
              matchedNode = leafNodes.length > 0 ? leafNodes[0] : nodes[0]
              break
            }
          }
        }
      }
      
      if (matchedNode) {
        const key = matchedNode.node.name
        const levelPriority = { HIGH: 3, MEDIUM: 2, LOW: 1 }
        
        if (!map.has(key)) {
          map.set(key, {
            level: risk.risk_level,
            confidence: risk.confidence,
            risks: [risk],
            isLeaf: matchedNode.isLeaf,
            path: matchedNode.path
          })
        } else {
          const existing = map.get(key)
          if ((levelPriority[risk.risk_level] || 0) > (levelPriority[existing.level] || 0)) {
            existing.level = risk.risk_level
          }
          existing.confidence = Math.max(existing.confidence, risk.confidence)
          existing.risks.push(risk)
        }
      }
    })
    
    return map
  }, [inference, lawData, lawIdx])

  // 绘制
  useEffect(() => {
    const data = lawData[lawIdx]
    if (!data || !svgRef.current) return

    const width = size.width
    const height = size.height
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // 边框颜色：优先 CSS 变量，回退到浅灰
    const strokeColor =
      (getComputedStyle(document.documentElement).getPropertyValue('--color-border-strong') || '').trim() ||
      (getComputedStyle(document.documentElement).getPropertyValue('--color-border-light') || '').trim() ||
      '#334155' // slate-700
    
    // 风险颜色映射（中文注释）
    const getRiskColor = (level) => {
      switch (level) {
        case 'HIGH': return '#ef4444'    // 红色
        case 'MEDIUM': return '#f59e0b'  // 橙色
        case 'LOW': return '#10b981'     // 绿色
        default: return null
      }
    }

    // —— 层级 + “均分权重”（每个父节点把自己的 value 平均分给直接子节点）—— //
    const root = d3.hierarchy(data)
    root.value = 1
    root.each(node => {
      if (node.children && node.children.length) {
        const share = node.value / node.children.length
        node.children.forEach(c => { c.value = share })
      }
    })

    // —— icicle：纵向为“值”（高度），横向为“深度”（厚度） —— //
    const visibleDepth = Math.max(1, root.height)   // 可见层（depth≥1）
    const step = width / visibleDepth               // 每层横向厚度，正好铺满宽度

    d3.partition().size([height, (root.height + 1) * step])(root)

    // 去掉根的占位带：整体左移一个 step
    root.each(d => { d.y0 -= step; d.y1 -= step })

    // 只渲染 depth>0
    const nodes = root.descendants().filter(d => d.depth > 0)

    // 初始焦点
    let focus = root

    // SVG 容器（增加边距以避免边框被裁剪）
    const margin = 4 // 边框留白
    svg
      .attr('viewBox', `${-margin} ${-margin} ${width + margin * 2} ${height + margin * 2}`)
      .attr('width', '100%')
      .attr('height', height)
      .attr('preserveAspectRatio', 'none')
      .attr('style', 'display:block; max-width:100%; height:auto; font:13px var(--font-family-main);')

    // 背景：点击返回根
    svg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'transparent')
      .on('click', () => clicked(root))

    // 像素取整，避免 1px 缝
    const px = v => Math.round(v)

    // 绘制
    const cell = svg.append('g')
      .attr('class', 'cells')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('transform', d => `translate(${px(d.y0)},${px(d.x0)})`)

      const rect = cell.append('rect')
      .attr('width',  d => Math.max(1, px(d.y1) - px(d.y0)))
      .attr('height', d => Math.max(1, px(d.x1) - px(d.x0)))
      .attr('fill', d => {
        // 根据风险等级填充颜色（中文注释）
        const nodeName = d.data.name
        const risk = riskMap.get(nodeName)
        if (risk) {
          const color = getRiskColor(risk.level)
          return color ? `${color}20` : 'transparent' // 20% 透明度
        }
        return 'transparent'
      })
      .style('pointer-events', 'all')    // ← 明确允许接收点击/hover
      .attr('stroke', d => {
        // 根据风险等级设置边框颜色（中文注释）
        const nodeName = d.data.name
        const risk = riskMap.get(nodeName)
        if (risk) {
          return getRiskColor(risk.level)
        }
        return strokeColor
      })
      .attr('stroke-width', d => {
        // 有风险的节点边框加粗（中文注释）
        const nodeName = d.data.name
        const risk = riskMap.get(nodeName)
        return risk ? 1.5 : 0.75
      })
      .attr('shape-rendering', 'geometricPrecision')
      .attr('vector-effect', 'non-scaling-stroke')
      .attr('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation()
        clicked(d)
      })

    const text = cell.append('text')
      .style('user-select', 'none')
      .attr('pointer-events', 'none')
      .attr('x', 6)
      .attr('y', 18)
      .attr('fill', (getComputedStyle(document.documentElement).getPropertyValue('--color-text-primary') || '#0f172a').trim() || '#0f172a')
      .attr('font-size', 13)
      .attr('font-weight', 600)
      .attr('fill-opacity', d => +labelVisible(d))
      .text(d => d.data.name)

    // tooltip（中文注释）：如果有风险，显示风险信息
    cell.append('title')
      .text(d => {
        const path = d.ancestors().map(d => d.data.name).reverse().join(' / ')
        const risk = riskMap.get(d.data.name)
        if (risk) {
          return `${path}\n\nRisk Level: ${risk.level}\nConfidence: ${(risk.confidence * 100).toFixed(0)}%\nRisk Count: ${risk.risks.length}`
        }
        return path
      })

    // —— 缩放：点击进入；再次点击同一节点→回父级；点击空白→回根 —— //
    function clicked(p) {
      focus = (p === focus && p.parent) ? p.parent : p

      // 以“去根后的起点”为基准；焦点为 root 时不补回根带
      const baseY = Math.max(0, focus.y0)

      root.each(d => {
        d.target = {
          // 垂直（值）归一化到整个高度
          x0: (d.x0 - focus.x0) / (focus.x1 - focus.x0) * height,
          x1: (d.x1 - focus.x0) / (focus.x1 - focus.x0) * height,
          // 水平（深度）仅平移（保持每层厚度不缩放）
          y0: d.y0 - baseY,
          y1: d.y1 - baseY
        }
      })

      const t = svg.selectAll('.cells g').transition().duration(750)
        .attr('transform', d => `translate(${px(d.target.y0)},${px(d.target.x0)})`)

      rect.transition(t)
        .attr('width',  d => Math.max(1, px(d.target.y1) - px(d.target.y0)))
        .attr('height', d => Math.max(1, px(d.target.x1) - px(d.target.x0)))

      text.transition(t)
        .attr('fill-opacity', d => +labelVisible(d.target))
    }

    function labelVisible(d) {
      const w = (d.y1 - d.y0)
      const h = (d.x1 - d.x0)
      return w > 38 && h > 18
    }
  }, [lawData, lawIdx, size, riskMap])

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)', marginBottom: 8, paddingLeft: 4 }}>
        Law Tree
      </div>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          minHeight: 200,
          background: 'var(--color-bg-secondary)',
          borderRadius: 16,
          boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
          border: '1px solid var(--color-border-light)',
          padding: 12
        }}
      >
        <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
          {LAWS.map((law, idx) => (
            <div
              key={law.key}
              onClick={() => setLawIdx(idx)}
              style={{
                cursor: 'pointer',
                padding: '6px 6px',
                borderRadius: 12,
                fontWeight: 600,
                fontSize: 12,
                color: lawIdx === idx ? 'var(--color-accent-primary)' : 'var(--color-text-secondary)',
                background: lawIdx === idx ? 'var(--color-accent-light)' : 'transparent',
                border: lawIdx === idx ? '1.5px solid var(--color-accent-primary)' : '1.5px solid transparent',
                boxShadow: lawIdx === idx ? '0 2px 8px rgba(14,165,233,0.08)' : 'none',
                transition: 'all 0.18s',
              }}
            >
              {law.label}
            </div>
          ))}
        </div>
        <svg ref={svgRef} style={{ width: '100%', height: size.height, display: 'block' }} />
      </div>
    </div>
  )
}
