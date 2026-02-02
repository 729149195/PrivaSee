import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'
import * as d3 from 'd3'

// 信息元词云可视化（中文注释）：分区布局 + 关系连线
export default function WordCloud({ selectedTime = null, filterIids = null, compact = false, fontScale = null }) {
  const { getCurrentSession, infonSessions } = useStore()
  const session = getCurrentSession()
  const runs = useMemo(() => (session ? (infonSessions?.[session.id]?.runs || []) : []), [session, infonSessions])
  
  // Tooltip 状态（中文注释）
  const [tooltip, setTooltip] = useState({ visible: false, infon: null, x: 0, y: 0 })
  
  // 显示模式：compact（紧凑圆点）或 full（显示文字）（中文注释）
  const [displayMode, setDisplayMode] = useState('auto') // 'auto' | 'compact' | 'full'
  
  // SVG 容器引用（中文注释）
  const svgRef = useRef(null)
  const containerRef = useRef(null)
  const [containerWidth, setContainerWidth] = useState(800)
  const [containerHeight, setContainerHeight] = useState(350)
  
  // 保存节点位置（中文注释）
  const nodePositionsRef = useRef(new Map())
  const lineKeysRef = useRef(new Set())

  // 信息元类型配置（中文注释）
  const typeConfig = {
    DESC: { label: 'Description', color: '#3b82f6', yRatio: 0.2 },  // 上方区域
    SCEN: { label: 'Scenario', color: '#10b981', yRatio: 0.8 },     // 下方区域
    REL: { label: 'Relation', color: '#8b5cf6', yRatio: 0.5 }       // 中间区域
  }

  const getInfonColor = (infonType) => {
    return typeConfig[String(infonType).toUpperCase()]?.color || '#64748b'
  }

  const getInfonKeyword = (infon) => {
    if (!infon || typeof infon !== 'object') return 'Unknown'
    const t = String(infon.infon_type || '').toUpperCase()
    if (t === 'DESC') {
      const attribute = infon.attribute ?? ''
      const entity = infon.entity ?? ''
      return attribute || entity || 'Description'
    }
    if (t === 'SCEN') {
      const temporal = infon.temporal ?? ''
      const spatial = infon.spatial ?? ''
      return temporal || spatial || 'Scenario'
    }
    if (t === 'REL') return String(infon.relation_name ?? 'Relation')
    if (t === 'SIT') return String(infon.description ?? 'Situation')
    return t || 'Unknown'
  }

  // 从所有 runs 中提取信息元数据（中文注释）
  const { wordData, relations, infonMap } = useMemo(() => {
    const filterSet = (Array.isArray(filterIids) && filterIids.length > 0) ? new Set(filterIids) : null
    const allInfons = []
    const relationInfons = []
    const infonById = new Map()
    const supersededIids = new Set()
    
    for (const run of runs) {
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      infons.forEach((infon) => {
        if (Array.isArray(infon._supersedes)) {
          infon._supersedes.forEach(oldIid => supersededIids.add(oldIid))
        }
      })
    }
    
    for (const run of runs) {
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      const isExpiring = run.expiring === true
      
      infons.forEach((infon) => {
        const type = String(infon.infon_type || '').toUpperCase()
        const iid = infon.iid
        
        if (iid && supersededIids.has(iid)) return
        if (filterSet && (!iid || !filterSet.has(iid))) return
        if (iid) infonById.set(iid, infon)
        if (selectedTime !== null && infon.record_time !== selectedTime) return
        if (type === 'SIT') return
        
        const item = {
          infon,
          keyword: getInfonKeyword(infon),
          type: type,
          color: getInfonColor(infon.infon_type),
          confidence: Math.max(0.3, Math.min(1, Number(infon?.confidence ?? 0.7))),
          runId: run.id,
          targetType: run.targetType,
          isPending: run.targetType === 'pending',
          iid: iid,
          isRelation: type === 'REL',
          isExpiring: isExpiring,
          count: 1,
          infons: [infon],
          iids: iid ? [iid] : []
        }
        
        if (type === 'REL') {
          relationInfons.push(item)
        } else {
          allInfons.push(item)
        }
      })
    }

    const grouped = new Map()
    allInfons.forEach((item) => {
      const key = `${item.keyword}-${item.type}`
      if (!grouped.has(key)) {
        grouped.set(key, { ...item })
      } else {
        const existing = grouped.get(key)
        existing.count += 1
        existing.confidence = Math.max(existing.confidence, item.confidence)
        existing.infons.push(item.infon)
        existing.isPending = existing.isPending || item.isPending
        existing.isExpiring = existing.isExpiring || item.isExpiring
        if (item.iid) existing.iids.push(item.iid)
      }
    })

    return {
      wordData: [...Array.from(grouped.values()), ...relationInfons],
      relations: relationInfons,
      infonMap: infonById
    }
  }, [runs, selectedTime, filterIids])

  // 容器尺寸观测（中文注释）
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(Math.max(320, Math.floor(entry.contentRect.width)))
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // 根据节点数量动态调整高度（中文注释）
  useEffect(() => {
    const nodeCount = wordData.length
    let baseHeight = compact ? 200 : 300
    if (!compact && nodeCount > 15) {
      baseHeight = Math.min(500, 300 + Math.floor((nodeCount - 15) / 5) * 30)
    }
    setContainerHeight(baseHeight)
  }, [wordData.length, compact])

  // 会话切换时清空缓存（中文注释）
  useEffect(() => {
    nodePositionsRef.current.clear()
    lineKeysRef.current.clear()
  }, [session?.id])

  // 判断是否使用紧凑模式（中文注释）
  const useCompactNodes = useMemo(() => {
    if (displayMode === 'compact') return true
    if (displayMode === 'full') return false
    // auto 模式：节点多时使用紧凑模式
    return wordData.length > 30
  }, [displayMode, wordData.length])

  // D3 分区布局渲染（中文注释）
  useEffect(() => {
    if (!svgRef.current || !wordData.length) return
    
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = containerWidth
    const height = containerHeight
    const padding = 20
    const nodeCount = wordData.length

    // 创建主容器组（中文注释）
    const mainGroup = svg.append('g').attr('class', 'main-group')
    
    // 添加缩放功能（中文注释）
    const zoom = d3.zoom()
      .scaleExtent([0.3, 5])
      .on('zoom', (event) => mainGroup.attr('transform', event.transform))
    
    svg.call(zoom)
      .on('dblclick.zoom', () => {
        svg.transition().duration(300).call(zoom.transform, d3.zoomIdentity)
      })
    svg.style('cursor', 'move')

    // 按类型分组节点（中文注释）
    const nodesByType = { DESC: [], SCEN: [], REL: [] }
    wordData.forEach(d => {
      if (nodesByType[d.type]) nodesByType[d.type].push(d)
    })

    // 计算节点大小（中文注释）：紧凑模式用小圆点，否则用文字
    const compactMode = useCompactNodes
    const nodeRadius = compactMode ? 6 : null
    
    // 根据节点数量动态调整字体大小（中文注释）
    const maxCount = Math.max(...wordData.map(d => d.count), 1)
    let minFontSize, maxFontSize
    if (nodeCount > 50) {
      minFontSize = 7; maxFontSize = 10
    } else if (nodeCount > 35) {
      minFontSize = 8; maxFontSize = 11
    } else if (nodeCount > 25) {
      minFontSize = 9; maxFontSize = 12
    } else if (nodeCount > 15) {
      minFontSize = 10; maxFontSize = 13
    } else {
      minFontSize = 11; maxFontSize = 15
    }
    const fontSizeScale = d3.scaleSqrt().domain([1, maxCount]).range([minFontSize, maxFontSize])

    // 测量文本尺寸（中文注释）
    const tempGroup = mainGroup.append('g').style('visibility', 'hidden')
    
    // 为节点分配位置（分区布局）（中文注释）
    const nodes = []
    const centerX = width / 2
    
    Object.entries(nodesByType).forEach(([type, items]) => {
      if (items.length === 0) return
      
      const config = typeConfig[type]
      const baseY = height * config.yRatio
      const itemCount = items.length
      
      // 计算该类型区域的可用宽度和行数（中文注释）
      const availableWidth = width - padding * 2
      
      items.forEach((d, i) => {
        const baseFontSize = fontSizeScale(d.count)
        const fontSize = baseFontSize * (0.7 + d.confidence * 0.3)
        
        let actualTextWidth, actualTextHeight
        if (compactMode) {
          actualTextWidth = nodeRadius * 2
          actualTextHeight = nodeRadius * 2
        } else {
          const tempText = tempGroup.append('text')
            .attr('font-size', d.isRelation ? Math.max(8, fontSize * 0.85) : fontSize)
            .attr('font-weight', d.isRelation ? 700 : 600)
            .text(d.keyword)
          const bbox = tempText.node().getBBox()
          actualTextWidth = bbox.width
          actualTextHeight = bbox.height
          tempText.remove()
        }
        
        const nodeKey = `${d.keyword}-${d.type}`
        const savedPos = nodePositionsRef.current.get(nodeKey)
        
        let initX, initY
        if (savedPos) {
          initX = savedPos.x
          initY = savedPos.y
        } else {
          // 分区布局：在各自区域内横向分布（中文注释）
          const spacing = compactMode ? 25 : Math.max(actualTextWidth + 20, 50)
          const itemsPerRow = Math.max(1, Math.floor(availableWidth / spacing))
          const row = Math.floor(i / itemsPerRow)
          const col = i % itemsPerRow
          const rowWidth = Math.min(itemsPerRow, itemCount - row * itemsPerRow) * spacing
          const startX = centerX - rowWidth / 2 + spacing / 2
          
          initX = startX + col * spacing
          initY = baseY + row * (compactMode ? 22 : 28) - (Math.ceil(itemCount / itemsPerRow) - 1) * (compactMode ? 11 : 14)
          
          // 添加一点随机偏移使布局更自然（中文注释）
          initX += (Math.random() - 0.5) * 10
          initY += (Math.random() - 0.5) * 8
        }
        
        nodes.push({
          ...d, fontSize, actualTextWidth, actualTextHeight,
          x: initX, y: initY, vx: 0, vy: 0,
          id: nodes.length, nodeKey, isNew: !savedPos,
          targetY: baseY // 用于力导向的目标 Y 位置
        })
      })
    })
    
    tempGroup.remove()

    // 构建连线数据（中文注释）
    const links = []
    const iidToNodeIndex = new Map()
    nodes.forEach((node, idx) => {
      if (node.iid) iidToNodeIndex.set(node.iid, idx)
      if (node.iids) node.iids.forEach(iid => iidToNodeIndex.set(iid, idx))
    })
    
    relations.forEach(rel => {
      const argRefs = rel.infon?.arg_refs || []
      if (argRefs.length < 2) return
      const relNodeIdx = iidToNodeIndex.get(rel.iid)
      if (relNodeIdx === undefined) return
      
      argRefs.forEach(argRef => {
        const argNodeIdx = iidToNodeIndex.get(argRef)
        if (argNodeIdx !== undefined) {
          links.push({ source: argNodeIdx, target: relNodeIdx, distance: 80 })
        }
      })
    })

    // 力导向模拟（中文注释）：使用较弱的力，主要保持分区布局
    const hasNewNodes = nodes.some(n => n.isNew)
    
    // 根据节点数量调整力参数（中文注释）
    const chargeStrength = nodeCount > 40 ? -80 : nodeCount > 25 ? -60 : -40
    const collideRadius = compactMode 
      ? () => nodeRadius + 4
      : d => Math.max(d.actualTextWidth, d.actualTextHeight) / 2 + 6
    
    const simulation = d3.forceSimulation(nodes)
      .force('charge', d3.forceManyBody().strength(chargeStrength))
      .force('collide', d3.forceCollide().radius(collideRadius).iterations(4))
      .force('link', d3.forceLink(links).distance(d => d.distance).strength(0.3))
      .force('x', d3.forceX(centerX).strength(0.02))
      .force('y', d3.forceY(d => d.targetY).strength(0.08)) // 强力拉向目标 Y 位置保持分区
      .alpha(hasNewNodes ? 0.8 : 0.3)
      .alphaDecay(0.02)
      .velocityDecay(0.5)
      .on('tick', () => {
        nodes.forEach(d => {
          // 边界约束（中文注释）
          const halfW = compactMode ? nodeRadius + 2 : (d.actualTextWidth + 12) / 2
          const halfH = compactMode ? nodeRadius + 2 : (d.actualTextHeight + 12) / 2
          d.x = Math.max(padding + halfW, Math.min(width - padding - halfW, d.x))
          d.y = Math.max(padding + halfH, Math.min(height - padding - halfH, d.y))
          
          if (d.nodeKey) {
            nodePositionsRef.current.set(d.nodeKey, { x: d.x, y: d.y, vx: d.vx || 0, vy: d.vy || 0 })
          }
        })
      })

    // 拖拽行为（中文注释）
    const drag = d3.drag()
      .on('start', function(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x; d.fy = d.y
        d3.select(this).style('cursor', 'grabbing')
      })
      .on('drag', function(event, d) { d.fx = event.x; d.fy = event.y })
      .on('end', function(event, d) {
        if (!event.active) simulation.alphaTarget(0)
        d.fx = null; d.fy = null
        d3.select(this).style('cursor', 'grab')
      })

    // 创建区域分隔线和标签（中文注释）
    if (!compact) {
      const regionGroup = mainGroup.append('g').attr('class', 'regions')
      
      // DESC 和 REL 之间的分隔
      const descRelY = height * 0.35
      regionGroup.append('line')
        .attr('x1', padding).attr('x2', width - padding)
        .attr('y1', descRelY).attr('y2', descRelY)
        .attr('stroke', 'var(--color-border)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4').attr('opacity', 0.5)
      
      // REL 和 SCEN 之间的分隔
      const relScenY = height * 0.65
      regionGroup.append('line')
        .attr('x1', padding).attr('x2', width - padding)
        .attr('y1', relScenY).attr('y2', relScenY)
        .attr('stroke', 'var(--color-border)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4').attr('opacity', 0.5)
      
      // 区域标签
      const labelStyle = { fontSize: 10, opacity: 0.6, fontWeight: 600 }
      regionGroup.append('text')
        .attr('x', padding + 5).attr('y', height * 0.1)
        .attr('fill', typeConfig.DESC.color)
        .attr('font-size', labelStyle.fontSize).attr('opacity', labelStyle.opacity).attr('font-weight', labelStyle.fontWeight)
        .text(`DESC (${nodesByType.DESC.length})`)
      regionGroup.append('text')
        .attr('x', padding + 5).attr('y', height * 0.5)
        .attr('fill', typeConfig.REL.color)
        .attr('font-size', labelStyle.fontSize).attr('opacity', labelStyle.opacity).attr('font-weight', labelStyle.fontWeight)
        .text(`REL (${nodesByType.REL.length})`)
      regionGroup.append('text')
        .attr('x', padding + 5).attr('y', height * 0.9)
        .attr('fill', typeConfig.SCEN.color)
        .attr('font-size', labelStyle.fontSize).attr('opacity', labelStyle.opacity).attr('font-weight', labelStyle.fontWeight)
        .text(`SCEN (${nodesByType.SCEN.length})`)
    }

    // 创建节点元素（中文注释）
    const wordGroups = mainGroup.selectAll('.word-group')
      .data(nodes).enter().append('g')
      .attr('class', 'word-group')
      .style('cursor', 'grab')
      .style('opacity', d => d.isNew ? 0 : 1)
      .call(drag)
    
    wordGroups.filter(d => d.isNew)
      .transition().duration(400).style('opacity', 1)

    if (compactMode) {
      // 紧凑模式：小圆点（中文注释）
      wordGroups.append('circle')
        .attr('r', d => nodeRadius + (d.count > 1 ? Math.min(d.count, 4) : 0))
        .attr('fill', d => d.color)
        .attr('fill-opacity', d => d.isExpiring ? 0.2 : 0.8)
        .attr('stroke', d => d.color)
        .attr('stroke-width', d => d.isRelation ? 2 : 1)
        .attr('stroke-dasharray', d => d.isRelation ? '2,2' : '0')
    } else {
      // 完整模式：显示文字（中文注释）
      wordGroups.append('rect')
        .attr('rx', d => d.isRelation ? 8 : 4)
        .attr('ry', d => d.isRelation ? 8 : 4)
        .attr('fill', d => d.isRelation ? 'rgba(255, 255, 255, 0.95)' : d.color)
        .attr('fill-opacity', d => d.isExpiring ? 0.05 : (d.isRelation ? 1 : 0.15))
        .attr('stroke', d => d.color)
        .attr('stroke-width', d => d.isRelation ? 1.5 : 1)
        .attr('stroke-dasharray', d => d.isRelation ? '3,3' : '0')
        .attr('width', d => d.actualTextWidth + 12)
        .attr('height', d => d.actualTextHeight + 10)
        .attr('x', d => -(d.actualTextWidth + 12) / 2)
        .attr('y', d => -(d.actualTextHeight + 10) / 2)
        .style('opacity', d => d.isExpiring ? 0.4 : 1)

      wordGroups.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('fill', d => d.color)
        .attr('font-size', d => d.isRelation ? Math.max(8, d.fontSize * 0.85) : d.fontSize)
        .attr('font-weight', d => d.isRelation ? 700 : 600)
        .attr('opacity', d => d.isExpiring ? 0.3 : (d.isRelation ? 1 : (0.7 + d.confidence * 0.3)))
        .text(d => d.keyword)

      // 计数徽章（中文注释）
      wordGroups.filter(d => !d.isRelation && d.count > 1)
        .append('circle')
        .attr('cx', d => (d.actualTextWidth + 12) / 2 - 2)
        .attr('cy', d => -(d.actualTextHeight + 10) / 2 + 2)
        .attr('r', 6).attr('fill', d => d.color).attr('opacity', 0.9)

      wordGroups.filter(d => !d.isRelation && d.count > 1)
        .append('text')
        .attr('x', d => (d.actualTextWidth + 12) / 2 - 2)
        .attr('y', d => -(d.actualTextHeight + 10) / 2 + 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('fill', '#fff').attr('font-size', 8).attr('font-weight', 700)
        .text(d => d.count)
    }

    // 关系连线（中文注释）
    const relationGroup = mainGroup.insert('g', ':first-child').attr('class', 'relation-lines')
    const defs = mainGroup.append('defs')
    defs.append('marker').attr('id', 'arrowhead')
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('refX', 4).attr('refY', 2).attr('orient', 'auto')
      .append('polygon').attr('points', '0 0, 4 2, 0 4').attr('fill', 'rgba(139, 92, 246, 0.4)')
    defs.append('marker').attr('id', 'arrowhead-hl')
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('refX', 4).attr('refY', 2).attr('orient', 'auto')
      .append('polygon').attr('points', '0 0, 4 2, 0 4').attr('fill', 'rgba(139, 92, 246, 0.9)')

    // 构建邻接表和连线数据（中文注释）
    const iidToNode = new Map()
    nodes.forEach(node => {
      if (node.iid) iidToNode.set(node.iid, node)
      if (node.iids) node.iids.forEach(iid => iidToNode.set(iid, node))
    })

    const adjacencyMap = new Map()
    const lineData = []
    
    relations.forEach((rel, idx) => {
      const argRefs = rel.infon?.arg_refs || []
      if (argRefs.length === 0) return
      const relNode = iidToNode.get(rel.iid)
      if (!relNode) return

      let availableIndex = 0
      argRefs.forEach((ref) => {
        const argNode = iidToNode.get(ref)
        if (!argNode) return

        const lineKey = `${rel.iid}:${ref}`
        const isNewLine = !lineKeysRef.current.has(lineKey)

        lineData.push({
          relNode, argNode, isFirst: availableIndex === 0, index: availableIndex,
          delay: idx * 60 + availableIndex * 30, relNodeId: relNode.id, argNodeId: argNode.id,
          lineKey, isNewLine
        })

        if (!adjacencyMap.has(relNode.id)) adjacencyMap.set(relNode.id, new Set())
        if (!adjacencyMap.has(argNode.id)) adjacencyMap.set(argNode.id, new Set())
        adjacencyMap.get(relNode.id).add(argNode.id)
        adjacencyMap.get(argNode.id).add(relNode.id)

        availableIndex += 1
      })
    })
    
    lineData.forEach(d => { if (d.isNewLine) lineKeysRef.current.add(d.lineKey) })

    // BFS 查找可达节点（中文注释）
    const findReachableNodes = (startNodeId, maxDepth = 2) => {
      const reachable = new Set([startNodeId])
      const queue = [{ nodeId: startNodeId, depth: 0 }]
      while (queue.length > 0) {
        const { nodeId: current, depth } = queue.shift()
        if (depth >= maxDepth) continue
        const neighbors = adjacencyMap.get(current) || new Set()
        for (const neighbor of neighbors) {
          if (!reachable.has(neighbor)) {
            reachable.add(neighbor)
            queue.push({ nodeId: neighbor, depth: depth + 1 })
          }
        }
      }
      return reachable
    }

    // 连线路径（中文注释）
    const linePaths = relationGroup.selectAll('.relation-line')
      .data(lineData).enter().append('path')
      .attr('class', 'relation-line')
      .attr('fill', 'none')
      .attr('stroke', 'rgba(139, 92, 246, 0.25)')
      .attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arrowhead)')
      .attr('opacity', d => d.isNewLine ? 0 : 1)

    linePaths.filter(d => d.isNewLine)
      .transition().duration(300).delay(d => d.delay).attr('opacity', 1)

    wordGroups.filter(d => d.isRelation).raise()
    
    // 更新位置（中文注释）
    simulation.on('tick.update', () => {
      wordGroups.attr('transform', d => `translate(${d.x}, ${d.y})`)
      
      linePaths.attr('d', d => {
        const { relNode, argNode, isFirst } = d
        const r1 = compactMode ? nodeRadius + 3 : 0
        const r2 = compactMode ? nodeRadius + 3 : 0
        
        let sx, sy, ex, ey
        if (isFirst) {
          sx = argNode.x; sy = argNode.y; ex = relNode.x; ey = relNode.y
        } else {
          sx = relNode.x; sy = relNode.y; ex = argNode.x; ey = argNode.y
        }
        
        // 计算边界点（中文注释）
        const dx = ex - sx, dy = ey - sy
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 1) return ''
        
        let startX, startY, endX, endY
        if (compactMode) {
          startX = sx + (dx / dist) * r1
          startY = sy + (dy / dist) * r1
          endX = ex - (dx / dist) * r2
          endY = ey - (dy / dist) * r2
        } else {
          // 矩形边界交点（中文注释）
          const sNode = isFirst ? argNode : relNode
          const eNode = isFirst ? relNode : argNode
          const sW = sNode.actualTextWidth + 12, sH = sNode.actualTextHeight + 10
          const eW = eNode.actualTextWidth + 12, eH = eNode.actualTextHeight + 10
          
          const getEdge = (cx, cy, w, h, tx, ty) => {
            const ddx = tx - cx, ddy = ty - cy
            if (Math.abs(ddx) < 0.001 && Math.abs(ddy) < 0.001) return { x: cx, y: cy }
            const halfW = w / 2, halfH = h / 2
            let t = Infinity
            if (ddx > 0) t = Math.min(t, halfW / ddx)
            if (ddx < 0) t = Math.min(t, -halfW / ddx)
            if (ddy > 0) t = Math.min(t, halfH / ddy)
            if (ddy < 0) t = Math.min(t, -halfH / ddy)
            return { x: cx + ddx * t, y: cy + ddy * t }
          }
          
          const startEdge = getEdge(sx, sy, sW, sH, ex, ey)
          const endEdge = getEdge(ex, ey, eW, eH, sx, sy)
          startX = startEdge.x; startY = startEdge.y
          endX = endEdge.x; endY = endEdge.y
        }
        
        // 曲线控制点（中文注释）
        const midX = (startX + endX) / 2
        const midY = (startY + endY) / 2
        const curveDist = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2)
        const offset = Math.max(10, Math.min(25, curveDist * 0.15))
        const bendDirection = isFirst ? 1 : -1
        const perpX = -(endY - startY) / Math.max(1, curveDist) * offset * bendDirection
        const perpY = (endX - startX) / Math.max(1, curveDist) * offset * bendDirection
        
        return `M ${startX} ${startY} Q ${midX + perpX} ${midY + perpY}, ${endX} ${endY}`
      })
    })

    // 悬停效果（中文注释）
    wordGroups
      .on('mouseenter', function(event, d) {
        const el = d3.select(this)
        if (compactMode) {
          el.select('circle').transition().duration(150)
            .attr('r', nodeRadius + 4)
            .attr('fill-opacity', 1)
        } else {
          el.select('rect').transition().duration(150)
            .attr('fill-opacity', d.isRelation ? 1 : 0.35)
            .attr('stroke-width', 2.5)
        }
        
        const reachableNodes = findReachableNodes(d.id)
        wordGroups.transition().duration(150)
          .attr('opacity', node => reachableNodes.has(node.id) ? 1 : 0.2)
        
        linePaths.transition().duration(150)
          .attr('stroke', line => {
            if (reachableNodes.has(line.relNodeId) && reachableNodes.has(line.argNodeId)) {
              return 'rgba(139, 92, 246, 0.8)'
            }
            return 'rgba(139, 92, 246, 0.08)'
          })
          .attr('stroke-width', line => {
            return (reachableNodes.has(line.relNodeId) && reachableNodes.has(line.argNodeId)) ? 2.5 : 1
          })
          .attr('marker-end', line => {
            if (reachableNodes.has(line.relNodeId) && reachableNodes.has(line.argNodeId)) {
              return 'url(#arrowhead-hl)'
            }
            return 'url(#arrowhead)'
          })
        
        // 显示 tooltip（中文注释）
        if (!compact && d.infons && d.infons[0]) {
          const containerRect = containerRef.current?.getBoundingClientRect()
          if (containerRect) {
            const tooltipWidth = 260, tooltipHeight = 140, margin = 12
            let x = event.clientX - containerRect.left + margin
            let y = event.clientY - containerRect.top + margin
            if (x + tooltipWidth > containerRect.width) x = event.clientX - containerRect.left - tooltipWidth - margin
            if (y + tooltipHeight > containerRect.height) y = event.clientY - containerRect.top - tooltipHeight - margin
            x = Math.max(margin, x); y = Math.max(margin, y)
            setTooltip({ visible: true, infon: d.infons[0], x, y })
          }
        }
      })
      .on('mousemove', function(event, d) {
        if (!compact && d.infons && d.infons[0]) {
          const containerRect = containerRef.current?.getBoundingClientRect()
          if (containerRect) {
            const tooltipWidth = 260, tooltipHeight = 140, margin = 12
            let x = event.clientX - containerRect.left + margin
            let y = event.clientY - containerRect.top + margin
            if (x + tooltipWidth > containerRect.width) x = event.clientX - containerRect.left - tooltipWidth - margin
            if (y + tooltipHeight > containerRect.height) y = event.clientY - containerRect.top - tooltipHeight - margin
            x = Math.max(margin, x); y = Math.max(margin, y)
            setTooltip(prev => ({ ...prev, x, y }))
          }
        }
      })
      .on('mouseleave', function(event, d) {
        const el = d3.select(this)
        if (compactMode) {
          el.select('circle').transition().duration(150)
            .attr('r', nodeRadius + (d.count > 1 ? Math.min(d.count, 4) : 0))
            .attr('fill-opacity', d.isExpiring ? 0.2 : 0.8)
        } else {
          el.select('rect').transition().duration(150)
            .attr('fill-opacity', d.isExpiring ? 0.05 : (d.isRelation ? 1 : 0.15))
            .attr('stroke-width', d.isRelation ? 1.5 : 1)
        }
        
        wordGroups.transition().duration(150).attr('opacity', 1)
        linePaths.transition().duration(150)
          .attr('stroke', 'rgba(139, 92, 246, 0.25)')
          .attr('stroke-width', 1.5)
          .attr('marker-end', 'url(#arrowhead)')
        
        setTooltip({ visible: false, infon: null, x: 0, y: 0 })
      })

    return () => simulation.stop()
  }, [wordData, relations, containerWidth, containerHeight, compact, useCompactNodes])

  // 格式化 infon 详情（中文注释）
  const formatInfonDetails = (infon) => {
    if (!infon) return []
    const preferred = ['infon_type', 'entity', 'attribute', 'temporal', 'spatial', 'relation_name', 'description', 'confidence']
    return preferred
      .filter(key => key in infon && infon[key] !== undefined && infon[key] !== null && infon[key] !== '')
      .map(key => ({ key, value: infon[key] }))
  }

  return (
    <div ref={containerRef} style={{ marginBottom: compact ? 0 : '12px', position: 'relative' }}>
      {/* 标题栏（中文注释） */}
      {!compact && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <div className={styles.wordCloudTitle} style={{ marginBottom: 0 }}>Infons Cloud</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>
              {wordData.length} infons
            </span>
            {/* 显示模式切换（中文注释） */}
            {wordData.length > 15 && (
              <select
                value={displayMode}
                onChange={(e) => setDisplayMode(e.target.value)}
                style={{
                  padding: '2px 6px',
                  border: '1px solid var(--color-border)',
                  borderRadius: 4,
                  backgroundColor: 'var(--color-bg-secondary)',
                  color: 'var(--color-text-secondary)',
                  fontSize: 10,
                  cursor: 'pointer'
                }}
              >
                <option value="auto">Auto</option>
                <option value="full">Text</option>
                <option value="compact">Dots</option>
              </select>
            )}
            <span style={{ fontSize: 9, color: 'var(--color-text-tertiary)', opacity: 0.7 }}>
              scroll zoom • drag
            </span>
          </div>
        </div>
      )}
      
      {/* SVG 区域（中文注释） */}
      <div className={`${styles.wordCloudRoot} ${compact ? styles.wordCloudRootCompact : ''}`}>
        {wordData.length === 0 ? (
          <div className={styles.infonEmpty} style={{ padding: '20px', textAlign: 'center' }}>
            No inference yet
          </div>
        ) : (
          <svg
            ref={svgRef}
            className={styles.wordCloudSvg}
            width="100%"
            height={containerHeight}
            viewBox={`0 0 ${containerWidth} ${containerHeight}`}
            preserveAspectRatio="xMidYMid meet"
          />
        )}
      </div>
      
      {/* Tooltip（中文注释） */}
      {!compact && tooltip.visible && tooltip.infon && (
        <div
          style={{
            position: 'absolute',
            left: tooltip.x,
            top: tooltip.y,
            backgroundColor: 'var(--color-bg-primary)',
            border: '1px solid var(--color-border)',
            borderRadius: 6,
            padding: '8px 12px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            zIndex: 1000,
            maxWidth: 260,
            fontSize: 11,
            pointerEvents: 'none'
          }}
        >
          {formatInfonDetails(tooltip.infon).map(({ key, value }) => (
            <div key={key} style={{ marginBottom: 3, display: 'flex', gap: 8 }}>
              <span style={{ color: 'var(--color-text-tertiary)', fontWeight: 600, minWidth: 65, fontSize: 10 }}>
                {key}:
              </span>
              <span style={{ color: 'var(--color-text-primary)', wordBreak: 'break-word' }}>
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
