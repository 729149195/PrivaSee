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

  // 构建风险映射：精确匹配到最小叶子节点，并向上传播高亮（中文注释）
  const riskMap = useMemo(() => {
    const map = new Map() // key: 节点路径或名称, value: { level, confidence, risks, isLeaf }
    
    if (!inference || !inference.risks || !lawData[lawIdx]) return map
    
    // 检查推理结果是否与当前选中的法律匹配（中文注释）
    const currentLawKey = LAWS[lawIdx].key
    if (inference.lawKey && inference.lawKey !== currentLawKey) {
      // 推理结果是针对其他法律的，不显示高亮
      return map
    }
    
    // 构建法律树的所有节点路径映射
    const nodePathMap = new Map() // key: 完整路径字符串, value: 节点对象
    const nodeNameMap = new Map() // key: 节点名称, value: 节点对象数组（可能有重名）
    const nodeByName = new Map() // key: 节点名称, value: 节点对象（用于快速查找）
    
    function traverseTree(node, path = [], parent = null) {
      const currentPath = [...path, node.name]
      const pathKey = currentPath.join(' > ')
      
      const nodeInfo = { 
        node, 
        path: currentPath, 
        isLeaf: !node.children || node.children.length === 0,
        parent: parent
      }
      
      nodePathMap.set(pathKey, nodeInfo)
      nodeByName.set(node.name, nodeInfo)
      
      // 按名称索引（支持查找）
      if (!nodeNameMap.has(node.name)) {
        nodeNameMap.set(node.name, [])
      }
      nodeNameMap.get(node.name).push(nodeInfo)
      
      if (node.children) {
        node.children.forEach(child => traverseTree(child, currentPath, node))
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
          // 尝试简单的包含匹配
          for (const [name, nodes] of nodeNameMap.entries()) {
            if (name.includes(nodeName) || nodeName.includes(name)) {
              const leafNodes = nodes.filter(n => n.isLeaf)
              matchedNode = leafNodes.length > 0 ? leafNodes[0] : nodes[0]
              break
            }
          }
          
          // 如果还没匹配到，尝试智能分段匹配（针对斜杠分隔的名称）
          if (!matchedNode && (nodeName.includes('/') || nodeName.includes('／'))) {
            let bestMatch = null
            let bestScore = 0
            
            // 将节点名称按斜杠分段
            const nodeNameParts = nodeName.split(/[\/／]/).map(p => p.trim()).filter(p => p)
            
            for (const [name, nodes] of nodeNameMap.entries()) {
              // 只对叶子节点进行模糊匹配
              const leafNodes = nodes.filter(n => n.isLeaf)
              if (leafNodes.length === 0) continue
              
              const nameParts = name.split(/[\/／]/).map(p => p.trim()).filter(p => p)
              
              // 计算匹配分数：每个段之间的相似度
              let totalScore = 0
              let matchedParts = 0
              
              for (const nodePart of nodeNameParts) {
                let maxPartScore = 0
                for (const namePart of nameParts) {
                  // 计算编辑距离相似度或包含关系
                  if (nodePart === namePart) {
                    maxPartScore = 1.0  // 完全匹配
                  } else if (nodePart.includes(namePart) || namePart.includes(nodePart)) {
                    maxPartScore = Math.max(maxPartScore, 0.8)  // 包含关系
                  } else {
                    // 计算字符重叠度（简单相似度）
                    const overlap = [...nodePart].filter(c => namePart.includes(c)).length
                    const similarity = overlap / Math.max(nodePart.length, namePart.length)
                    maxPartScore = Math.max(maxPartScore, similarity * 0.6)
                  }
                }
                if (maxPartScore > 0.3) {  // 至少30%相似才计入
                  totalScore += maxPartScore
                  matchedParts++
                }
              }
              
              // 整体分数 = 平均相似度 * 匹配段数比例
              const avgScore = matchedParts > 0 ? totalScore / nodeNameParts.length : 0
              const coverageBonus = matchedParts / nodeNameParts.length
              const finalScore = avgScore * coverageBonus
              
              // 如果匹配度超过50%，认为可能匹配
              if (finalScore >= 0.5 && finalScore > bestScore) {
                bestMatch = leafNodes[0]
                bestScore = finalScore
              }
            }
            
            if (bestMatch) {
              matchedNode = bestMatch
              console.log(`Fuzzy matched "${nodeName}" to "${bestMatch.node.name}" (score: ${bestScore.toFixed(2)})`)
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
            path: matchedNode.path,
            node: matchedNode.node
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
    
    // 向上传播高亮到父节点（中文注释）：第二级及以上的节点也要高亮
    const propagatedMap = new Map(map)
    const levelPriority = { HIGH: 3, MEDIUM: 2, LOW: 1 }
    
    for (const [nodeName, riskInfo] of map.entries()) {
      // 获取该节点的路径，向上遍历所有父节点
      const path = riskInfo.path
      if (path && path.length > 1) {
        // 从当前节点向上到根节点，逐级传播高亮
        for (let i = path.length - 2; i >= 0; i--) {
          const parentName = path[i]
          const parentNode = nodeByName.get(parentName)
          
          if (!parentNode) continue
          
          if (!propagatedMap.has(parentName)) {
            // 父节点还没有风险，创建一个继承自子节点的风险
            propagatedMap.set(parentName, {
              level: riskInfo.level,
              confidence: riskInfo.confidence,
              risks: [], // 父节点的risks为空，表示是从子节点继承的
              isLeaf: false,
              path: parentNode.path,
              node: parentNode.node,
              inherited: true // 标记为继承的高亮
            })
          } else {
            // 父节点已有风险，更新为子节点中的最高级别
            const existing = propagatedMap.get(parentName)
            if ((levelPriority[riskInfo.level] || 0) > (levelPriority[existing.level] || 0)) {
              existing.level = riskInfo.level
              existing.confidence = Math.max(existing.confidence, riskInfo.confidence)
            }
          }
        }
      }
    }
    
    return propagatedMap
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
          return color ? `${color}cc` : 'transparent' // 80% 不透明度，更显眼的填充高亮
        }
        return 'transparent'
      })
      .style('pointer-events', 'all')    // ← 明确允许接收点击/hover
      .attr('stroke', strokeColor)  // 统一使用默认边框颜色
      .attr('stroke-width', 0.75)   // 统一边框宽度
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

    // tooltip（中文注释）：如果有风险，显示风险信息，区分直接风险和继承风险
    cell.append('title')
      .text(d => {
        const path = d.ancestors().map(d => d.data.name).reverse().join(' / ')
        const risk = riskMap.get(d.data.name)
        if (risk) {
          const riskType = risk.inherited ? '(Inherited from children)' : '(Direct match)'
          const riskCount = risk.risks && risk.risks.length > 0 ? `\nDirect Risk Count: ${risk.risks.length}` : ''
          return `${path}\n\nRisk Level: ${risk.level} ${riskType}\nConfidence: ${(risk.confidence * 100).toFixed(0)}%${riskCount}`
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
          padding: 6
        }}
      >
        <div style={{ display: 'flex', gap: 8, marginBottom: 6, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'center' }}>
          {LAWS.map((law, idx) => (
            <div
              key={law.key}
              onClick={() => setLawIdx(idx)}
              style={{
                cursor: 'pointer',
                padding: '6px 6px',
                borderRadius: 10,
                fontWeight: 600,
                fontSize: 12,
                color: lawIdx === idx ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                background: lawIdx === idx ? 'var(--color-bg-tertiary)' : 'transparent',
                borderWidth: 0.5,
                borderStyle: 'solid',
                borderColor: 'var(--color-border-light)',
                boxShadow: lawIdx === idx ? '0 0 0 1px #334155' : 'none',
                transition: 'background-color 0.18s, color 0.18s',
              }}
            >
              {law.label}
            </div>
          ))}
          {inference && inference.lawKey && inference.lawKey !== LAWS[lawIdx].key && (
            <div style={{ 
              fontSize: 11, 
              color: 'var(--color-text-tertiary)',
              fontStyle: 'italic',
              padding: '4px 8px',
              background: 'var(--color-bg-tertiary)',
              borderRadius: 6,
            }}>
              ⚠️ Inference results are for {inference.lawKey}
            </div>
          )}
        </div>
        <svg ref={svgRef} style={{ width: '100%', height: size.height, display: 'block' }} />
      </div>
    </div>
  )
}
