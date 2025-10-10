import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../store'
import styles from './AgentPage.module.css'
import * as d3 from 'd3'

// 信息元词云可视化（中文注释）：使用 D3 力导向布局 + 词云算法
export default function WordCloud() {
  const { getCurrentSession, infonSessions } = useStore()
  const session = getCurrentSession()
  const runs = useMemo(() => (session ? (infonSessions?.[session.id]?.runs || []) : []), [session, infonSessions])
  
  // 选中信息元（中文注释）：点击词后显示详情
  const [selectedInfon, setSelectedInfon] = useState(null)
  
  // SVG 容器引用（中文注释）
  const svgRef = useRef(null)
  const containerRef = useRef(null)
  const [containerWidth, setContainerWidth] = useState(800)
  const [containerHeight, setContainerHeight] = useState(500)
  
  // 保存节点位置，用于流式加载时保持已有节点位置（中文注释）
  const nodePositionsRef = useRef(new Map()) // key: keyword-type, value: {x, y, vx, vy}
  // 保存已渲染连线 key，防止旧连线重复淡入（中文注释）
  const lineKeysRef = useRef(new Set()) // key: `${relIid}:${argIid}`
  
  // 容器尺寸观测（中文注释）：固定高度避免无限拉长
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
    // 固定高度为 280px
    setContainerHeight(280)
    return () => ro.disconnect()
  }, [])
  
  // 会话切换时清空节点位置缓存（中文注释）
  useEffect(() => {
    nodePositionsRef.current.clear()
    lineKeysRef.current.clear()
  }, [session?.id])

  // 信息元类型颜色映射（中文注释）
  const getInfonColor = (infonType) => {
    const colors = {
      IND: '#3b82f6',   // 个体：蓝色
      PAR: '#10b981',   // 参数：绿色
      TIM: '#8b5cf6',   // 时间：紫色
      LOC: '#f59e0b',   // 位置：橙色
      REL: '#0ea5e9',   // 关系：天蓝色
      TYP: '#06b6d4',   // 类型：青色
      SIT: '#f97316',   // 情景：橙红色
    }
    return colors[String(infonType).toUpperCase()] || '#64748b'
  }

  // 提取信息元关键词（中文注释）
  const getInfonKeyword = (infon) => {
    if (!infon || typeof infon !== 'object') return 'Unknown'
    const t = String(infon.infon_type || '').toUpperCase()
    if (t === 'IND') {
      if (Array.isArray(infon.names) && infon.names.length) return String(infon.names[0])
      return 'Individual'
    }
    if (t === 'PAR') return String(infon.value ?? 'Parameter')
    if (t === 'TIM') return String(infon.temporal_value ?? 'Time')
    if (t === 'LOC') return String(infon.spatial_value ?? 'Location')
    if (t === 'REL') return String(infon.relation_name ?? 'Relation')
    if (t === 'TYP') return String(infon.type_name ?? 'Type')
    if (t === 'SIT') return String(infon.description ?? 'Situation')
    return t || 'Unknown'
  }

  // 从所有 runs 中提取信息元数据（中文注释）：过滤 SIT，保留 REL 作为节点
  const { wordData, relations, infonMap } = useMemo(() => {
    const allInfons = []
    const relationInfons = []
    const infonById = new Map()
    
    for (const run of runs) {
      // 允许 running 状态参与可视化（中文注释）：实现流式显示
      if (!run || (run.status !== 'done' && run.status !== 'running')) continue
      const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
      infons.forEach((infon) => {
        const type = String(infon.infon_type || '').toUpperCase()
        const iid = infon.iid
        
        // 保存到 iid 映射表（中文注释）
        if (iid) {
          infonById.set(iid, infon)
        }
        
        // 过滤掉 SIT（中文注释）
        if (type === 'SIT') {
          return
        }
        
        // REL 作为关系节点保留（中文注释）
        if (type === 'REL') {
          relationInfons.push({
            infon,
            keyword: getInfonKeyword(infon),
            type: type,
            color: getInfonColor(infon.infon_type),
            confidence: Math.max(0.3, Math.min(1, Number(infon?.confidence ?? 0.7))),
            runId: run.id,
            targetType: run.targetType,
            isPending: run.targetType === 'pending',
            iid: iid,
            isRelation: true,
            count: 1,
            infons: [infon],
            iids: [iid]
          })
          return
        }
        
        allInfons.push({
          infon,
          keyword: getInfonKeyword(infon),
          type: type,
          color: getInfonColor(infon.infon_type),
          confidence: Math.max(0.3, Math.min(1, Number(infon?.confidence ?? 0.7))),
          runId: run.id,
          targetType: run.targetType,
          isPending: run.targetType === 'pending',
          iid: iid,
          isRelation: false
        })
      })
    }

    // 按关键词分组，聚合置信度和计数（中文注释）
    const grouped = new Map()
    allInfons.forEach((item) => {
      const key = `${item.keyword}-${item.type}` // 同关键词不同类型也分开
      if (!grouped.has(key)) {
        grouped.set(key, {
          keyword: item.keyword,
          type: item.type,
          color: item.color,
          confidence: item.confidence,
          count: 1,
          infons: [item.infon],
          isPending: item.isPending,
          iids: item.iid ? [item.iid] : [],
          isRelation: false
        })
      } else {
        const existing = grouped.get(key)
        existing.count += 1
        existing.confidence = Math.max(existing.confidence, item.confidence)
        existing.infons.push(item.infon)
        existing.isPending = existing.isPending || item.isPending
        if (item.iid) existing.iids.push(item.iid)
      }
    })

    return {
      wordData: [...Array.from(grouped.values()), ...relationInfons], // 合并普通词和关系节点
      relations: relationInfons,
      infonMap: infonById
    }
  }, [runs])

  // D3 词云布局与渲染（中文注释）
  useEffect(() => {
    if (!svgRef.current || !wordData.length) return
    
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove() // 清空

    const width = containerWidth
    const height = containerHeight
    const padding = 10 // 边界留白（中文注释）
    const centerX = width / 2
    const centerY = height / 2

    // 计算字体大小：基于置信度和出现次数（中文注释）：缩小字号
    const maxCount = Math.max(...wordData.map(d => d.count))
    const minFontSize = 12
    const maxFontSize = 16
    
    const fontSizeScale = d3.scaleSqrt()
      .domain([1, maxCount])
      .range([minFontSize, maxFontSize])

    // 先创建临时文本元素测量实际尺寸（中文注释）
    const tempGroup = svg.append('g').style('visibility', 'hidden')
    
    // 为每个词准备节点数据（中文注释）：关系节点和普通节点分开初始化
    const nodes = wordData.map((d, i) => {
      const baseFontSize = fontSizeScale(d.count)
      const fontSize = baseFontSize * (0.7 + d.confidence * 0.3) // 置信度影响字体大小
      
      // 创建临时文本测量实际尺寸（中文注释）
      const tempText = tempGroup.append('text')
        .attr('font-size', d.isRelation ? Math.max(9, fontSize * 0.8) : fontSize)
        .attr('font-weight', d.isRelation ? 700 : 600)
        .text(d.keyword)
      
      const bbox = tempText.node().getBBox()
      const actualTextWidth = bbox.width
      const actualTextHeight = bbox.height
      tempText.remove()
      
      // 初始位置：优先使用保存的位置（流式加载），否则使用默认初始位置（中文注释）
      const nodeKey = `${d.keyword}-${d.type}`
      const savedPos = nodePositionsRef.current.get(nodeKey)
      
      let initX, initY, initVx, initVy
      if (savedPos) {
        // 使用保存的位置（已存在的节点）（中文注释）
        initX = savedPos.x
        initY = savedPos.y
        initVx = savedPos.vx
        initVy = savedPos.vy
      } else {
        // 新节点：计算初始位置（中文注释）
        if (d.isRelation) {
          // 关系节点：中心较小范围，按索引均匀分布（中文注释）
          const relCount = wordData.filter(item => item.isRelation).length
          const relIndex = wordData.filter((item, idx) => idx < i && item.isRelation).length
          const angle = (relIndex / Math.max(1, relCount)) * Math.PI * 2
          const radius = Math.random() * Math.min(width, height) * 0.1
          initX = centerX + Math.cos(angle) * radius
          initY = centerY + Math.sin(angle) * radius
        } else {
          // 普通节点：外围均匀圆形分布（中文注释）
          const normalCount = wordData.filter(item => !item.isRelation).length
          const normalIndex = wordData.filter((item, idx) => idx < i && !item.isRelation).length
          const angle = (normalIndex / Math.max(1, normalCount)) * Math.PI * 2 + Math.random() * 0.3
          const radius = Math.min(width, height) * 0.28 + Math.random() * Math.min(width, height) * 0.1
          initX = centerX + Math.cos(angle) * radius
          initY = centerY + Math.sin(angle) * radius
        }
        initVx = 0
        initVy = 0
      }
      
      return {
        ...d,
        fontSize,
        actualTextWidth,
        actualTextHeight,
        x: initX,
        y: initY,
        vx: initVx,
        vy: initVy,
        id: i,
        nodeKey: nodeKey,
        isNew: !savedPos // 标记是否是新节点（中文注释）
      }
    })
    
    // 移除临时组（中文注释）
    tempGroup.remove()

    // 构建关系连线数据（中文注释）：用于link force
    const links = []
    const iidToNodeIndex = new Map()
    nodes.forEach((node, idx) => {
      if (node.iid) {
        iidToNodeIndex.set(node.iid, idx)
      }
      if (node.iids) {
        node.iids.forEach(iid => {
          iidToNodeIndex.set(iid, idx)
        })
      }
    })
    
    relations.forEach(rel => {
      const argRefs = rel.infon?.arg_refs || []
      if (argRefs.length < 2) return
      
      const relNodeIdx = iidToNodeIndex.get(rel.iid)
      if (relNodeIdx === undefined) return
      
      // 为每个参数创建到关系节点的连线（中文注释）
      argRefs.forEach(argRef => {
        const argNodeIdx = iidToNodeIndex.get(argRef)
        if (argNodeIdx !== undefined) {
          links.push({
            source: argNodeIdx,
            target: relNodeIdx,
            distance: 150 // 最小距离：增加到150
          })
        }
      })
    })

    // D3 力导向模拟：结合连线约束和边界限制（中文注释）
    // 检查是否有新节点，用于调整 simulation 的初始 alpha（中文注释）
    const hasNewNodes = nodes.some(n => n.isNew)
    const initialAlpha = hasNewNodes ? 1 : 0.3 // 有新节点时用完整 alpha，否则用较低值减少抖动
    
    const simulation = d3.forceSimulation(nodes)
      .force('charge', d3.forceManyBody().strength(d => d.isRelation ? -60 : -100))
      .force('collide', d3.forceCollide().radius(d => Math.max(d.actualTextWidth + 12, d.actualTextHeight + 12) / 2 + 10).iterations(4))
      .force('link', d3.forceLink(links).distance(d => d.distance).strength(0.7))
      .force('center', d3.forceCenter(centerX, centerY).strength(0.03))
      .force('x', d3.forceX(centerX).strength(d => d.isRelation ? 0.05 : 0.02))
      .force('y', d3.forceY(centerY).strength(d => d.isRelation ? 0.05 : 0.02))
      .alpha(initialAlpha)
      .alphaDecay(hasNewNodes ? 0.01 : 0.05) // 新节点时慢衰减，否则快速稳定
      .velocityDecay(0.4)
      .on('tick', () => {
        // 边界约束（中文注释）：限制节点在容器内
        nodes.forEach(d => {
          const halfW = (d.actualTextWidth + 12) / 2
          const halfH = (d.actualTextHeight + 12) / 2
          d.x = Math.max(padding + halfW, Math.min(width - padding - halfW, d.x))
          d.y = Math.max(padding + halfH, Math.min(height - padding - halfH, d.y))
          
          // 保存节点位置（流式加载用）（中文注释）
          if (d.nodeKey) {
            nodePositionsRef.current.set(d.nodeKey, {
              x: d.x,
              y: d.y,
              vx: d.vx || 0,
              vy: d.vy || 0
            })
          }
        })
      })

    // 拖拽行为（中文注释）
    const drag = d3.drag()
      .on('start', function(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
        d3.select(this).style('cursor', 'grabbing')
      })
      .on('drag', function(event, d) {
        d.fx = event.x
        d.fy = event.y
      })
      .on('end', function(event, d) {
        if (!event.active) simulation.alphaTarget(0)
        d.fx = null
        d.fy = null
        d3.select(this).style('cursor', 'grab')
      })

    // 创建词元素组（中文注释）：流式进入动画
    const wordGroups = svg.selectAll('.word-group')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'word-group')
      .style('cursor', 'grab')
      .style('opacity', d => d.isNew ? 0 : 1) // 新节点从透明开始，已存在节点直接显示
      .call(drag)
      .on('click', (event, d) => {
        setSelectedInfon(d.infons[0]) // 显示第一个关联的信息元
      })
    
    // 流式淡入效果（中文注释）
    wordGroups.filter(d => d.isNew)
      .transition()
      .duration(400)
      .style('opacity', 1)

    // 背景：所有节点用矩形（中文注释）
    wordGroups.append('rect')
      .attr('class', styles.wordCloudBg)
      .attr('rx', d => d.isRelation ? 8 : 4)
      .attr('ry', d => d.isRelation ? 8 : 4)
      .attr('fill', d => d.isRelation ? 'rgba(255, 255, 255, 0.95)' : d.color)
      .attr('fill-opacity', d => d.isRelation ? 1 : 0.15)
      .attr('stroke', d => d.color)
      .attr('stroke-width', d => d.isRelation ? 1.2 : 1)
      .attr('stroke-dasharray', d => d.isRelation ? '3,3' : '0')
      .attr('width', d => d.actualTextWidth + 12)
      .attr('height', d => d.actualTextHeight + 12)
      .attr('x', d => -(d.actualTextWidth + 12) / 2)
      .attr('y', d => -(d.actualTextHeight + 12) / 2)

    // 文本（中文注释）
    wordGroups.append('text')
      .attr('class', styles.wordCloudText)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .attr('fill', d => d.color)
      .attr('font-size', d => d.isRelation ? Math.max(9, d.fontSize * 0.8) : d.fontSize)
      .attr('font-weight', d => d.isRelation ? 700 : 600)
      .attr('opacity', d => d.isRelation ? 1 : (0.7 + d.confidence * 0.3))
      .text(d => d.keyword)

    // 计数标记（右上角小徽章）（中文注释）：仅用于普通节点
    wordGroups.filter(d => !d.isRelation && d.count > 1)
      .append('circle')
      .attr('class', styles.wordCloudBadge)
      .attr('cx', d => (d.actualTextWidth + 12) / 2)
      .attr('cy', d => -(d.actualTextHeight + 12) / 2)
      .attr('r', 8)
      .attr('fill', d => d.color)
      .attr('opacity', 0.9)

    wordGroups.filter(d => !d.isRelation && d.count > 1)
      .append('text')
      .attr('class', styles.wordCloudBadgeText)
      .attr('x', d => (d.actualTextWidth + 12) / 2)
      .attr('y', d => -(d.actualTextHeight + 12) / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .attr('fill', '#fff')
      .attr('font-size', 10)
      .attr('font-weight', 700)
      .text(d => d.count)

    // 创建关系连线组（中文注释）：在词元素之前绘制（作为底层）
    const relationGroup = svg.insert('g', ':first-child')
      .attr('class', 'relation-lines')
    
    // 定义箭头标记（中文注释）：降低透明度
    const defs = svg.append('defs')
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('refX', 5)
      .attr('refY', 2.5)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 5 2.5, 0 5')
      .attr('fill', 'rgba(14, 165, 233, 0.3)')
    
    // 高亮箭头标记（中文注释）
    defs.append('marker')
      .attr('id', 'arrowhead-highlight')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('refX', 5)
      .attr('refY', 2.5)
      .attr('orient', 'auto')
      .append('polygon')
      .attr('points', '0 0, 5 2.5, 0 5')
      .attr('fill', 'rgba(14, 165, 233, 0.9)')

    // 构建 iid 到节点的映射（中文注释）
    const iidToNode = new Map()
    nodes.forEach(node => {
      if (node.iid) {
        iidToNode.set(node.iid, node)
      }
      if (node.iids) {
        node.iids.forEach(iid => {
          iidToNode.set(iid, node)
        })
      }
    })

    // 计算射线与矩形边界的交点（中文注释）
    const getRectEdgePoint = (centerX, centerY, width, height, targetX, targetY) => {
      const dx = targetX - centerX
      const dy = targetY - centerY
      
      if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) {
        return { x: centerX, y: centerY }
      }
      
      const halfW = width / 2
      const halfH = height / 2
      
      // 计算射线与四条边的交点（中文注释）
      let t = Infinity
      
      // 右边
      if (dx > 0) {
        t = Math.min(t, halfW / dx)
      }
      // 左边
      if (dx < 0) {
        t = Math.min(t, -halfW / dx)
      }
      // 下边
      if (dy > 0) {
        t = Math.min(t, halfH / dy)
      }
      // 上边
      if (dy < 0) {
        t = Math.min(t, -halfH / dy)
      }
      
      return {
        x: centerX + dx * t,
        y: centerY + dy * t
      }
    }

    // 预先创建所有连线（中文注释）
    const lineData = []
    const adjacencyMap = new Map() // 邻接表：nodeId -> Set of connected nodeIds
    
    relations.forEach((rel, idx) => {
      const argRefs = rel.infon?.arg_refs || []
      if (argRefs.length === 0) return

      const relNode = iidToNode.get(rel.iid)
      if (!relNode) return

      let availableIndex = 0
      argRefs.forEach((ref) => {
        const argNode = iidToNode.get(ref)
        if (!argNode) return

        const isFirstAvailable = availableIndex === 0
        const lineKey = `${rel.iid}:${ref}`
        const isNewLine = !lineKeysRef.current.has(lineKey)

        lineData.push({
          relNode,
          argNode,
          isFirst: isFirstAvailable,
          index: availableIndex,
          delay: idx * 100 + availableIndex * 50,
          relNodeId: relNode.id,
          argNodeId: argNode.id,
          lineKey,
          isNewLine
        })

        // 构建邻接表（无向图）（中文注释）
        if (!adjacencyMap.has(relNode.id)) adjacencyMap.set(relNode.id, new Set())
        if (!adjacencyMap.has(argNode.id)) adjacencyMap.set(argNode.id, new Set())
        adjacencyMap.get(relNode.id).add(argNode.id)
        adjacencyMap.get(argNode.id).add(relNode.id)

        availableIndex += 1
      })
    })
    
    // 将本轮新增的连线 key 记录到缓存集合（中文注释）
    lineData.forEach(d => { if (d.isNewLine) lineKeysRef.current.add(d.lineKey) })
    
    // BFS 找出从指定节点可达的节点（限制深度为2）（中文注释）
    const findReachableNodes = (startNodeId, maxDepth = 2) => {
      const reachable = new Set([startNodeId])
      const queue = [{ nodeId: startNodeId, depth: 0 }]
      
      while (queue.length > 0) {
        const { nodeId: current, depth } = queue.shift()
        
        // 如果已达到最大深度，不再扩展（中文注释）
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

    // 创建连线路径元素（中文注释）：降低透明度和线宽
    const linePaths = relationGroup.selectAll('.relation-line')
      .data(lineData)
      .enter()
      .append('path')
      .attr('class', 'relation-line')
      .attr('fill', 'none')
      .attr('stroke', 'rgba(14, 165, 233, 0.15)')
      .attr('stroke-width', 1)
      .attr('marker-end', 'url(#arrowhead)')
      .attr('opacity', d => d.isNewLine ? 0 : 1)

    // 仅对新连线做淡入动画（中文注释）
    linePaths
      .filter(d => d.isNewLine)
      .transition()
      .duration(400)
      .delay(d => d.delay)
      .attr('opacity', 1)

    // 将关系节点提升到最上层（中文注释）：确保显示在连线上方
    wordGroups.filter(d => d.isRelation).raise()
    
    // 每次tick更新节点和连线位置（中文注释）
    simulation.on('tick.update', () => {
      wordGroups.attr('transform', d => `translate(${d.x}, ${d.y})`)
      
      // 更新所有连线（中文注释）
      linePaths.attr('d', d => {
        const relNode = d.relNode
        const argNode = d.argNode
        
        const relX = relNode.x
        const relY = relNode.y
        const relWidth = relNode.actualTextWidth + 12
        const relHeight = relNode.actualTextHeight + 12
        
        const argX = argNode.x
        const argY = argNode.y
        const argWidth = argNode.actualTextWidth + 12
        const argHeight = argNode.actualTextHeight + 12
        
        // 确定方向（中文注释）
        let startX, startY, startW, startH, endX, endY, endW, endH
        if (d.isFirst) {
          // arg1 → REL（中文注释）
          startX = argX; startY = argY; startW = argWidth; startH = argHeight
          endX = relX; endY = relY; endW = relWidth; endH = relHeight
        } else {
          // REL → arg2, arg3, ...（中文注释）
          startX = relX; startY = relY; startW = relWidth; startH = relHeight
          endX = argX; endY = argY; endW = argWidth; endH = argHeight
        }
        
        // 计算边界交点（中文注释）
        const startEdge = getRectEdgePoint(startX, startY, startW, startH, endX, endY)
        const endEdge = getRectEdgePoint(endX, endY, endW, endH, startX, startY)
        
        // 计算曲线控制点（中文注释）：避免交错
        const midX = (startEdge.x + endEdge.x) / 2
        const midY = (startEdge.y + endEdge.y) / 2
        const dx = endEdge.x - startEdge.x
        const dy = endEdge.y - startEdge.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        
        // 动态调整弯曲程度（中文注释）：距离越远弯曲越大，减少交错
        const offset = Math.max(20, Math.min(40, dist * 0.2))
        
        // 根据连线索引和相对位置决定弯曲方向（中文注释）
        const bendDirection = d.isFirst ? 1 : -1 // 第一条向一侧，其他向另一侧
        const perpX = -dy / Math.max(1, dist) * offset * bendDirection
        const perpY = dx / Math.max(1, dist) * offset * bendDirection
        const ctrlX = midX + perpX
        const ctrlY = midY + perpY
        
        return `M ${startEdge.x} ${startEdge.y} Q ${ctrlX} ${ctrlY}, ${endEdge.x} ${endEdge.y}`
      })
    })


    // 添加悬停效果（中文注释）：高亮所有可达节点及其连线
    wordGroups
      .on('mouseenter', function(event, d) {
        d3.select(this).select('rect')
          .transition()
          .duration(200)
          .attr('fill-opacity', d.isRelation ? 1 : 0.3)
          .attr('stroke-width', 2)
        
        // 找出所有可达的节点（中文注释）
        const reachableNodes = findReachableNodes(d.id)
        
        // 高亮可达节点，淡化其他节点（中文注释）
        wordGroups
          .transition()
          .duration(200)
          .attr('opacity', node => reachableNodes.has(node.id) ? 1 : 0.3)
        
        // 高亮连接到可达节点的所有连线（中文注释）
        linePaths
          .transition()
          .duration(200)
          .attr('stroke', line => {
            // 如果连线的两端都在可达集合中，高亮显示
            if (reachableNodes.has(line.relNodeId) && reachableNodes.has(line.argNodeId)) {
              return 'rgba(14, 165, 233, 0.8)'
            }
            return 'rgba(14, 165, 233, 0.05)'
          })
          .attr('stroke-width', line => {
            if (reachableNodes.has(line.relNodeId) && reachableNodes.has(line.argNodeId)) {
              return 2
            }
            return 1
          })
          .attr('marker-end', line => {
            if (reachableNodes.has(line.relNodeId) && reachableNodes.has(line.argNodeId)) {
              return 'url(#arrowhead-highlight)'
            }
            return 'url(#arrowhead)'
          })
      })
      .on('mouseleave', function(event, d) {
        d3.select(this).select('rect')
          .transition()
          .duration(200)
          .attr('fill-opacity', d.isRelation ? 1 : 0.15)
          .attr('stroke-width', d.isRelation ? 1.2 : 1)
        
        // 恢复所有节点（中文注释）
        wordGroups
          .transition()
          .duration(200)
          .attr('opacity', 1)
        
        // 恢复所有连线（中文注释）
        linePaths
          .transition()
          .duration(200)
          .attr('stroke', 'rgba(14, 165, 233, 0.15)')
          .attr('stroke-width', 1)
          .attr('marker-end', 'url(#arrowhead)')
      })

    return () => {
      simulation.stop()
    }
  }, [wordData, relations, infonMap, containerWidth, containerHeight])

  return (
    <div ref={containerRef} className={styles.wordCloudRoot}>
      <div className={styles.wordCloudTitle}>Information Element Word Cloud</div>
      {wordData.length === 0 ? (
        <div className={styles.infonEmpty} style={{ padding: '20px', textAlign: 'center' }}>No inference yet</div>
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
      
      {selectedInfon ? (
        <details className={styles.wordCloudDetails} open>
          <summary className={styles.wordCloudDetailsSummary}>
            Infon details
            <button 
              className={styles.closeDetailsBtn}
              onClick={(e) => { e.preventDefault(); setSelectedInfon(null); }}
              style={{ marginLeft: '8px' }}
            >
              ✕
            </button>
          </summary>
          <div className={styles.wordCloudDetailsContent}>
            <table className={styles.wordCloudTable}>
              <tbody>
                {(() => {
                  const preferred = [
                    'iid','infon_type','confidence',
                    'names','value','temporal_value','spatial_value',
                    'relation_name','type_name','description'
                  ]
                  const keys = Array.from(new Set([...preferred, ...Object.keys(selectedInfon || {})]))
                  return keys.map((k) => {
                    if (!(k in (selectedInfon || {}))) return null
                    const v = selectedInfon[k]
                    const isString = typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
                    const str = isString ? String(v) : JSON.stringify(v, null, 2)
                    return (
                      <tr key={k}>
                        <td className={styles.wordCloudKey}>{k}</td>
                        <td className={styles.wordCloudValue}>{isString ? str : (<pre className={styles.wordCloudPre}>{str}</pre>)}</td>
                      </tr>
                    )
                  })
                })()}
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </div>
  )
}

