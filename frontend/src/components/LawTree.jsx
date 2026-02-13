import React, { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { useStore } from '../store'

const LAWS = [
  { key: 'PIPL', label: 'PIPL', file: './law/PIPL.json' },
  { key: 'GDPR', label: 'GDPR', file: './law/GDPR.json' },
  { key: 'CCPA_CPRA', label: 'CCPA/CPRA', file: './law/CCPA_CPRA.json' },
  { key: 'CUSTOM', label: 'Custom', file: null }, 
]

async function fetchLawData(file) {
  const res = await fetch(file)
  return await res.json()
}

// 自定义隐私项列表
const PRIVACY_ITEMS = [
  { id: 'name', label: 'Name', category: 'Identity' },
  { id: 'email', label: 'Email Address', category: 'Contact' },
  { id: 'phone', label: 'Phone Number', category: 'Contact' },
  { id: 'address', label: 'Home Address', category: 'Contact' },
  { id: 'dob', label: 'Date of Birth', category: 'Identity' },
  { id: 'gender', label: 'Gender', category: 'Identity' },
  { id: 'photo', label: 'Photos/Images', category: 'Identity' },
  { id: 'ip', label: 'IP Address', category: 'Digital' },
  { id: 'device', label: 'Device ID', category: 'Digital' },
  { id: 'cookies', label: 'Cookies', category: 'Digital' },
  { id: 'location', label: 'Location/GPS', category: 'Location' },
  { id: 'browsing', label: 'Browsing History', category: 'Behavioral' },
  { id: 'search', label: 'Search History', category: 'Behavioral' },
  { id: 'biometric', label: 'Biometric Data', category: 'Sensitive' },
  { id: 'financial', label: 'Financial Information', category: 'Sensitive' },
  { id: 'health', label: 'Health Data', category: 'Sensitive' },
  { id: 'social', label: 'Social Media Activity', category: 'Behavioral' },
  { id: 'purchases', label: 'Purchase History', category: 'Behavioral' },
]

export default function PrivacyExposureTree() {
  const [lawData, setLawData] = useState([null, null, null, null])
  const [newItemInput, setNewItemInput] = useState('') // 新隐私项输入框
  const [holdingLawIdx, setHoldingLawIdx] = useState(null) // 正在长按的法律索引
  const [holdProgress, setHoldProgress] = useState(0) // 长按进度 0-100
  const containerRef = useRef(null)
  const svgRef = useRef(null)
  const holdTimerRef = useRef(null)
  const holdStartRef = useRef(null)
  const [size, setSize] = useState({ width: 928, height: 600 })
  const clipId = useMemo(() => `clip-${Math.random().toString(36).slice(2, 9)}`, [])
  
  // 鱼眼镜头：当前悬停的节点（中文注释）
  const [hoveredNode, setHoveredNode] = useState(null)
  
  // 鱼眼效果：鼠标位置（相对于SVG中心）
  const [mousePos, setMousePos] = useState(null)

  // 仅用于展示：提取路径字符串中的最后一段，避免显示完整路径
  const getLeafDisplayName = (name) => {
    if (!name) return ''
    return name
      .split(/[>\/／]/)
      .map(part => part.trim())
      .filter(Boolean)
      .pop() || name
  }
  
  // 从 store 获取推理结果和相关方法（中文注释）
  const { 
    getCurrentSession, 
    privacyInferences, 
    setSelectedLaw, 
    startPrivacyInference,
    abortPrivacyInference,
    clearPrivacyInference,
    selectedLawIdx,
    setSelectedLawIdx,
    customPrivacyItems,
    addCustomPrivacyItem,
    removeCustomPrivacyItem,
    selectedPrivacyItems: selectedPrivacyItemsArray,
    setSelectedPrivacyItems,
    togglePrivacyItem
  } = useStore()
  
  // 使用store中的lawIdx
  const lawIdx = selectedLawIdx
  const setLawIdx = setSelectedLawIdx
  
  // 将数组转换为Set以便使用
  const selectedPrivacyItems = useMemo(() => new Set(selectedPrivacyItemsArray), [selectedPrivacyItemsArray])
  
  const session = getCurrentSession()
  const inference = useMemo(() => (session ? privacyInferences?.[session.id] : null), [session, privacyInferences])

  // 预加载法律数据（跳过Custom）
  useEffect(() => {
    LAWS.forEach((law, idx) => {
      if (!lawData[idx] && law.file) {
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
  
  // 初始化时设置默认法律（中文注释）
  useEffect(() => {
    if (useStore.getState().selectedLaw) return

    // 如果第一个是 Custom（无文件），直接设置
    if (LAWS[0].key === 'CUSTOM') {
      const allItems = [...PRIVACY_ITEMS, ...customPrivacyItems]
      const selectedDetails = allItems.filter(item => selectedPrivacyItems.has(item.id))
      setSelectedLaw('CUSTOM', {
        customItems: selectedDetails,
        isCustom: true
      })
    } else if (lawData[0]) {
      setSelectedLaw(LAWS[0].key, lawData[0])
    }
    // eslint-disable-next-line
  }, [lawData, customPrivacyItems, selectedPrivacyItems])

  // 容器自适应
  useEffect(() => {
    const update = () => {
      if (!containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const w = Math.max(320, Math.floor(rect.width))
      const h = Math.max(400, Math.floor(rect.width * 0.78))
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
    
    // 匹配推理结果到法律节点（仅使用law_node_name，无需law_path）
    inference.risks.forEach(risk => {
      const nodeName = risk.law_node_name || ''
      
      // 跳过还没有law_node_name的部分风险对象
      if (!nodeName || nodeName === 'Loading...') {
        return
      }
      
      let matchedNode = null
      
      // 策略：使用节点名称匹配，优先匹配叶子节点
      if (nodeName) {
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
        const levelPriority = { HIGH: 3, MEDIUM: 2, LOW: 1, UNKNOWN: 0 }
        
        // 支持部分风险对象（可能缺少某些字段）
        const riskLevel = risk.risk_level || 'UNKNOWN'
        const confidence = risk.confidence ?? 0
        
        if (!map.has(key)) {
          map.set(key, {
            level: riskLevel,
            confidence: confidence,
            risks: [risk],
            isLeaf: matchedNode.isLeaf,
            path: matchedNode.path,
            node: matchedNode.node
          })
        } else {
          const existing = map.get(key)
          if ((levelPriority[riskLevel] || 0) > (levelPriority[existing.level] || 0)) {
            existing.level = riskLevel
          }
          existing.confidence = Math.max(existing.confidence, confidence)
          // 使用_objIndex去重，避免部分更新导致重复
          const existingIndices = existing.risks.map(r => r._objIndex).filter(i => i !== undefined)
          if (risk._objIndex === undefined || !existingIndices.includes(risk._objIndex)) {
            existing.risks.push(risk)
          } else {
            // 更新现有的risk对象
            const idx = existing.risks.findIndex(r => r._objIndex === risk._objIndex)
            if (idx >= 0) {
              existing.risks[idx] = risk
            }
          }
        }
      }
    })
    
    // 向上传播高亮到父节点（中文注释）：第二级及以上的节点也要高亮
    const propagatedMap = new Map(map)
    const levelPriority = { HIGH: 3, MEDIUM: 2, LOW: 1, UNKNOWN: 0 }
    
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

  // 绘制放射状树 + 鱼眼效果（中文注释：radial tree layout with fisheye）
  useEffect(() => {
    const data = lawData[lawIdx]
    if (!data || !svgRef.current) return

    const width = size.width
    const height = size.height
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // 计算中心点和半径（考虑四角信息的空间）
    const cx = width / 2
    const cy = height / 2
    const radius = Math.min(width, height) / 2 - 30  // 适当边距

    // 鱼眼效果参数
    const fisheyeRadius = 120  // 鱼眼影响半径
    const fisheyeDistortion = 3  // 失真强度

    // 鱼眼失真函数（中文注释：根据距离计算放大倍数和位移）
    const fisheye = (point, focus) => {
      if (!focus) return { x: point[0], y: point[1], scale: 1 }
      
      const dx = point[0] - focus[0]
      const dy = point[1] - focus[1]
      const distance = Math.sqrt(dx * dx + dy * dy)
      
      if (distance >= fisheyeRadius) {
        return { x: point[0], y: point[1], scale: 1 }
      }
      
      // 鱼眼变换公式
      const k = fisheyeDistortion
      const normalizedDist = distance / fisheyeRadius
      const distortedDist = (1 - Math.exp(-k * normalizedDist)) / (1 - Math.exp(-k))
      const scale = 1 + (2.5 - 1) * (1 - normalizedDist)  // 最大放大2.5倍
      
      if (distance === 0) {
        return { x: point[0], y: point[1], scale: 2.5 }
      }
      
      const newDist = distortedDist * fisheyeRadius
      const ratio = newDist / distance
      
      return {
        x: focus[0] + dx * ratio,
        y: focus[1] + dy * ratio,
        scale: scale
      }
    }
    
    // 风险颜色映射（黄-橙-红三级）
    const getRiskColor = (level) => {
      switch (level) {
        case 'HIGH': return '#ef4444'
        case 'MEDIUM': return '#f97316'
        case 'LOW': return '#eab308'
        default: return null
      }
    }

    // 构建层级数据
    const root = d3.hierarchy(data)
    
    // 放射状树布局（更紧凑地利用空间）
    const treeLayout = d3.tree()
      .size([2 * Math.PI, radius * 0.92])
      .separation((a, b) => (a.parent === b.parent ? 1 : 1.2) / a.depth)
    
    treeLayout(root)

    // 压缩内层径向距离，使内层两个层级更紧凑
    const allNodesRaw = root.descendants()
    const maxDepthRaw = d3.max(allNodesRaw, d => d.depth)
    if (maxDepthRaw > 0) {
      const maxR = radius * 0.92
      allNodesRaw.forEach(d => {
        if (d.depth > 0) {
          const t = d.depth / maxDepthRaw
          d.y = maxR * Math.pow(t, 1.4)
        }
      })
    }

    // 径向坐标转换函数
    const radialPoint = (x, y) => {
      const angle = x - Math.PI / 2
      return [y * Math.cos(angle), y * Math.sin(angle)]
    }

    // 预计算所有节点的基础位置和信息
    const nodes = root.descendants()
    nodes.forEach(d => {
      const [px, py] = radialPoint(d.x, d.y)
      d.px = px
      d.py = py
      
      // 节点大小：根据子树叶子节点数量编码（越多越大）
      const leafCount = d.leaves().length
      if (d.depth === 0) {
        d.baseSize = 7
      } else if (d.children) {
        // 分支节点：大小根据叶子数量，区间 3.5 ~ 6
        d.baseSize = Math.min(6, Math.max(3.5, 2 + Math.sqrt(leafCount) * 0.8))
      } else {
        // 叶子节点
        d.baseSize = 3
      }
      
      const risk = riskMap.get(d.data.name)
      if (risk && !risk.inherited) d.baseSize += 1.5
      
      // 分支节点：计算风险覆盖率（子树中有多少比例的叶子有风险）
      d.riskRatio = 0
      if (d.children) {
        const leaves = d.leaves()
        const riskyLeaves = leaves.filter(l => {
          const r = riskMap.get(l.data.name)
          return r && !r.inherited
        })
        d.riskRatio = leaves.length > 0 ? riskyLeaves.length / leaves.length : 0
      }
      
      // 标记是否是叶子
      d.isLeaf = !d.children || d.children.length === 0
    })
    
    const links = root.links()

    // SVG 设置
    svg
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('width', '100%')
      .attr('height', height)
      .attr('style', 'display:block; max-width:100%; height:auto; cursor:crosshair;')

    // 统计风险节点数（只统计叶子节点/直接风险，不统计继承的）
    const directRiskNodes = nodes.filter(d => {
      const risk = riskMap.get(d.data.name)
      return risk && !risk.inherited  // 只计算直接风险，排除继承的
    })
    const highRiskCount = directRiskNodes.filter(d => riskMap.get(d.data.name)?.level === 'HIGH').length
    const mediumRiskCount = directRiskNodes.filter(d => riskMap.get(d.data.name)?.level === 'MEDIUM').length
    const lowRiskCount = directRiskNodes.filter(d => riskMap.get(d.data.name)?.level === 'LOW').length

    // ========== 四角信息（统一边距 16px）==========
    const maxDepth = d3.max(nodes, d => d.depth)
    const padding = 12
    
    // 左上角：整体安全状态 / 悬停节点名称
    const topLeft = svg.append('g')
      .attr('transform', `translate(${padding}, ${padding + 6})`)
    const topLeftText = topLeft.append('text')
      .attr('font-size', 13)
      .attr('font-weight', 600)
      .attr('fill', directRiskNodes.length > 0 ? '#ef4444' : '#10b981')
      .text(directRiskNodes.length > 0 
        ? `${directRiskNodes.length} exposure${directRiskNodes.length > 1 ? 's' : ''} found` 
        : 'No exposure')
    const topLeftSub = topLeft.append('text')
      .attr('y', 18)
      .attr('font-size', 10)
      .attr('fill', 'var(--color-text-tertiary)')
      .text(directRiskNodes.length > 0 
        ? 'Hover on highlighted nodes for details'
        : 'Your conversation looks safe under this law')
    
    // 记录默认文本用于恢复
    const topLeftDefault = {
      text: directRiskNodes.length > 0 
        ? `${directRiskNodes.length} exposure${directRiskNodes.length > 1 ? 's' : ''} found`
        : 'No exposure',
      sub: directRiskNodes.length > 0 
        ? 'Hover on highlighted nodes for details'
        : 'Your conversation looks safe under this law',
      color: directRiskNodes.length > 0 ? '#ef4444' : '#10b981'
    }

    // 右上角：按级别的风险统计
    const topRight = svg.append('g')
      .attr('transform', `translate(${width - padding}, ${padding + 6})`)
      .attr('text-anchor', 'end')
    
    topRight.append('text')
      .attr('font-size', 9)
      .attr('font-weight', 500)
      .attr('fill', 'var(--color-text-tertiary)')
      .attr('letter-spacing', '0.5px')
      .text('RISK SUMMARY')
    
    if (directRiskNodes.length > 0) {
      const riskLine = []
      if (highRiskCount > 0) riskLine.push(`${highRiskCount} High`)
      if (mediumRiskCount > 0) riskLine.push(`${mediumRiskCount} Med`)
      if (lowRiskCount > 0) riskLine.push(`${lowRiskCount} Low`)
      
      topRight.append('text')
        .attr('y', 16)
        .attr('font-size', 11)
        .attr('fill', 'var(--color-text-secondary)')
        .text(riskLine.join(' · '))
    } else {
      topRight.append('text')
        .attr('y', 16)
        .attr('font-size', 11)
        .attr('fill', '#10b981')
        .text('All clear')
    }
    
    // 右下角：悬停时显示"为什么是风险"
    const bottomRight = svg.append('g')
      .attr('transform', `translate(${width - padding}, ${height - padding - 34})`)
      .attr('text-anchor', 'end')
    
    const riskDetailTitle = bottomRight.append('text')
      .attr('font-size', 9)
      .attr('font-weight', 500)
      .attr('fill', 'var(--color-text-tertiary)')
      .attr('letter-spacing', '0.5px')
      .text('')
    const riskDetailLine1 = bottomRight.append('text')
      .attr('y', 16)
      .attr('font-size', 10)
      .attr('fill', 'var(--color-text-secondary)')
      .text('')
    const riskDetailLine2 = bottomRight.append('text')
      .attr('y', 30)
      .attr('font-size', 10)
      .attr('fill', 'var(--color-text-tertiary)')
      .text('')

    // 左下角：图例
    const bottomLeft = svg.append('g')
      .attr('transform', `translate(${padding}, ${height - padding - 78})`)
    
    const rowHeight = 14
    const levelLegendData = [
      { color: '#ef4444', label: 'High Risk' },
      { color: '#f97316', label: 'Medium' },
      { color: '#eab308', label: 'Low' }
    ]
    levelLegendData.forEach((item, i) => {
      const row = bottomLeft.append('g')
        .attr('transform', `translate(0, ${i * rowHeight})`)
      row.append('circle')
        .attr('r', 4).attr('cx', 4).attr('cy', 0)
        .attr('fill', item.color)
      row.append('text')
        .attr('x', 14).attr('y', 4)
        .attr('font-size', 10)
        .attr('fill', 'var(--color-text-tertiary)')
        .text(item.label)
    })
    
    const typeLegend = bottomLeft.append('g')
      .attr('transform', `translate(0, ${levelLegendData.length * rowHeight + 6})`)
    // 分支节点：圆形
    typeLegend.append('circle')
      .attr('r', 4).attr('cx', 4).attr('cy', 0)
      .attr('fill', '#64748b')
    typeLegend.append('text')
      .attr('x', 14).attr('y', 4)
      .attr('font-size', 10)
      .attr('fill', 'var(--color-text-tertiary)')
      .text('Category')
    // 叶子节点：方形
    const leafRow = typeLegend.append('g')
      .attr('transform', `translate(0, ${rowHeight})`)
    leafRow.append('rect')
      .attr('x', 0.5).attr('y', -3.5)
      .attr('width', 7).attr('height', 7)
      .attr('rx', 1.5)
      .attr('fill', '#94a3b8')
    leafRow.append('text')
      .attr('x', 14).attr('y', 4)
      .attr('font-size', 10)
      .attr('fill', 'var(--color-text-tertiary)')
      .text('Specific item')
    // 继承风险：空心
    const inheritedRow = typeLegend.append('g')
      .attr('transform', `translate(0, ${rowHeight * 2})`)
    inheritedRow.append('circle')
      .attr('r', 4).attr('cx', 4).attr('cy', 0)
      .attr('fill', '#fff')
      .attr('stroke', '#94a3b8')
      .attr('stroke-width', 2)
    inheritedRow.append('text')
      .attr('x', 14).attr('y', 4)
      .attr('font-size', 10)
      .attr('fill', 'var(--color-text-tertiary)')
      .text('Inherited')

    // 主容器，移到中心
    const g = svg.append('g')
      .attr('transform', `translate(${cx},${cy})`)

    // 深度同心环背景（装饰性）
    const depthRings = g.append('g')
      .attr('class', 'depth-rings')
      .style('pointer-events', 'none')
    
    for (let depth = 1; depth <= maxDepth; depth++) {
      const ringRadius = (depth / maxDepth) * radius * 0.85
      depthRings.append('circle')
        .attr('r', ringRadius)
        .attr('fill', 'none')
        .attr('stroke', 'var(--color-border-light)')
        .attr('stroke-width', depth === maxDepth ? 1.5 : 0.5)
        .attr('stroke-dasharray', depth === maxDepth ? 'none' : '2,4')
        .attr('opacity', 0.15 + (depth / maxDepth) * 0.15)
    }

    // 背景圆（用于追踪鼠标和清除悬停状态）
    g.append('circle')
      .attr('r', radius + 50)
      .attr('fill', 'transparent')
      .style('pointer-events', 'all')

    // 鱼眼边框圆（跟随鼠标位置）
    const fisheyeBorder = g.append('circle')
      .attr('class', 'fisheye-border')
      .attr('r', fisheyeRadius)
      .attr('fill', 'none')
      .attr('stroke', 'var(--color-accent-primary)')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4')
      .attr('opacity', 0)
      .style('pointer-events', 'none')

    // 绘制连线（粗细随层级，曲线连接，颜色根据风险状态）
    const linkGroup = g.append('g')
      .attr('class', 'links')
      .attr('fill', 'none')

    const linkPaths = linkGroup.selectAll('path')
      .data(links)
      .join('path')
      .attr('d', d => {
        // 使用贝塞尔曲线让连线更柔和
        const sx = d.source.px, sy = d.source.py
        const tx = d.target.px, ty = d.target.py
        const mx = (sx + tx) / 2, my = (sy + ty) / 2
        // 控制点向中心偏移，让曲线有弧度
        const cpx = mx * 0.85, cpy = my * 0.85
        return `M${sx},${sy}Q${cpx},${cpy} ${tx},${ty}`
      })
      .attr('stroke', d => {
        // 目标节点有直接风险时，连线带风险色
        const targetRisk = riskMap.get(d.target.data.name)
        if (targetRisk && !targetRisk.inherited) {
          return getRiskColor(targetRisk.level) || 'var(--color-border-medium)'
        }
        return 'var(--color-border-medium)'
      })
      .attr('stroke-width', d => {
        // 浅层粗，深层细
        const depth = d.target.depth
        return Math.max(0.8, 2.5 - depth * 0.4)
      })
      .attr('stroke-opacity', d => {
        // 有风险的连线更醒目
        const targetRisk = riskMap.get(d.target.data.name)
        if (targetRisk && !targetRisk.inherited) return 0.7
        return 0.35
      })
      .style('transition', 'stroke 0.15s ease, stroke-width 0.15s ease, stroke-opacity 0.15s ease')

    // 绘制节点
    const nodeGroup = g.append('g')
      .attr('class', 'nodes')

    const nodeGs = nodeGroup.selectAll('g')
      .data(nodes)
      .join('g')
      .attr('transform', d => `translate(${d.px},${d.py})`)

    // 分支节点：风险覆盖弧线（背景圆弧，显示子节点中风险占比）
    const riskArcGen = d3.arc()
      .startAngle(-Math.PI / 2)
    
    nodeGs.filter(d => d.children && d.riskRatio > 0)
      .append('path')
      .attr('class', 'risk-arc')
      .attr('d', d => {
        const endAngle = -Math.PI / 2 + d.riskRatio * Math.PI * 2
        return riskArcGen({
          innerRadius: d.baseSize + 1.5,
          outerRadius: d.baseSize + 3.5,
          endAngle: endAngle
        })
      })
      .attr('fill', d => {
        const risk = riskMap.get(d.data.name)
        if (risk) return getRiskColor(risk.level) || '#f59e0b'
        return '#f59e0b'
      })
      .attr('opacity', 0.7)
      .style('pointer-events', 'none')
    
    // 分支节点：弧线背景底环
    nodeGs.filter(d => d.children && d.riskRatio > 0)
      .append('circle')
      .attr('r', d => d.baseSize + 2.5)
      .attr('fill', 'none')
      .attr('stroke', 'var(--color-border-light)')
      .attr('stroke-width', 1)
      .attr('opacity', 0.3)
      .style('pointer-events', 'none')

    // 高风险节点脉冲光环（动画效果）
    const pulseRings = nodeGs.filter(d => {
      const risk = riskMap.get(d.data.name)
      return risk && risk.level === 'HIGH' && !risk.inherited
    })
    .append('circle')
      .attr('class', 'pulse-ring')
      .attr('r', d => d.baseSize)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 2)
      .attr('opacity', 0)
      .style('pointer-events', 'none')
    
    // 脉冲动画
    function animatePulse() {
      pulseRings
        .attr('r', d => d.baseSize)
        .attr('opacity', 0.8)
        .attr('stroke-width', 2)
        .transition()
        .duration(1500)
        .ease(d3.easeQuadOut)
        .attr('r', d => d.baseSize + 12)
        .attr('opacity', 0)
        .attr('stroke-width', 0.5)
        .on('end', animatePulse)
    }
    animatePulse()

    // 获取节点颜色函数
    const getNodeFill = (d) => {
      const risk = riskMap.get(d.data.name)
        if (risk) {
        if (!risk.inherited) return getRiskColor(risk.level) || 'var(--color-accent-primary)'
        return '#fff'
      }
      if (d.depth === 0) return 'var(--color-accent-primary)'
      // 分支节点颜色根据深度渐变
      if (d.children) {
        const t = d.depth / maxDepth
        return d3.interpolate('#475569', '#94a3b8')(t)
      }
      return '#b0bec5'
    }
    
    const getNodeStroke = (d) => {
      const risk = riskMap.get(d.data.name)
      if (risk && risk.inherited) return getRiskColor(risk.level) || '#64748b'
      return '#fff'
    }
    
    const getNodeStrokeWidth = (d) => {
      const risk = riskMap.get(d.data.name)
      return (risk && risk.inherited) ? 2 : 1.5
    }

    // 风险节点光晕效果（底层阴影）
    nodeGs.filter(d => {
      const risk = riskMap.get(d.data.name)
      return risk && !risk.inherited
    })
    .append('circle')
      .attr('r', d => d.baseSize + 5)
      .attr('fill', d => {
        const risk = riskMap.get(d.data.name)
        return getRiskColor(risk.level) || '#f59e0b'
      })
      .attr('opacity', 0.15)
      .style('pointer-events', 'none')
      .style('filter', 'blur(3px)')

    // 叶子节点用小方形，分支节点用圆形
    const nodeCircles = nodeGs.append('path')
      .attr('d', d => {
        const s = d.baseSize
        if (d.isLeaf) {
          // 叶子：圆角方形
          const r = 1.5
          return `M${-s + r},${-s} L${s - r},${-s} Q${s},${-s} ${s},${-s + r} L${s},${s - r} Q${s},${s} ${s - r},${s} L${-s + r},${s} Q${-s},${s} ${-s},${s - r} L${-s},${-s + r} Q${-s},${-s} ${-s + r},${-s}Z`
        }
        // 分支/根：圆形
        return d3.symbol().type(d3.symbolCircle).size(s * s * Math.PI)()
      })
      .attr('fill', getNodeFill)
      .attr('stroke', getNodeStroke)
      .attr('stroke-width', getNodeStrokeWidth)
      .attr('cursor', 'pointer')
      .style('pointer-events', 'all')
      .style('transition', 'stroke 0.15s ease, stroke-width 0.15s ease')
    

    // 顶层标签组（在所有节点之上，不会被遮盖）
    const labelGroup = g.append('g')
      .attr('class', 'labels')
      .style('pointer-events', 'none')

    const labelGs = labelGroup.selectAll('g')
      .data(nodes)
      .join('g')
      .attr('transform', d => `translate(${d.px},${d.py})`)

    // 节点文本标签背景（提高可读性）
    const nodeLabelBgs = labelGs.append('rect')
      .attr('class', 'node-label-bg')
      .attr('fill', 'rgba(255,255,255,0.95)')
      .attr('rx', 4)
      .attr('ry', 4)
      .attr('opacity', 0)
      .attr('stroke', 'var(--color-border-light)')
      .attr('stroke-width', 1)

    // 节点文本标签（初始隐藏，鱼眼区域内显示）
    const nodeLabels = labelGs.append('text')
      .attr('class', 'node-label')
      .attr('dy', d => -d.baseSize - 12)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-weight', 600)
      .attr('fill', 'var(--color-text-primary)')
      .attr('opacity', 0)
      .text(d => d.data.name)
      .each(function(d) {
        // 计算文本边界框用于背景
        const bbox = this.getBBox()
        d.labelWidth = bbox.width
        d.labelHeight = bbox.height
      })

    // 更新背景矩形尺寸
    nodeLabelBgs
      .attr('x', d => -(d.labelWidth || 0) / 2 - 6)
      .attr('y', d => -d.baseSize - 12 - (d.labelHeight || 14) + 2)
      .attr('width', d => (d.labelWidth || 0) + 12)
      .attr('height', d => (d.labelHeight || 14) + 6)

    // 鼠标移动时应用鱼眼效果
    const applyFisheye = (focusX, focusY) => {
      // 更新鱼眼边框位置
      fisheyeBorder
        .attr('cx', focusX)
        .attr('cy', focusY)
        .attr('opacity', 0.6)
      
      // 更新节点位置和大小
      nodeGs.each(function(d) {
        const result = fisheye([d.px, d.py], [focusX, focusY])
        d.fx = result.x
        d.fy = result.y
        d.fscale = result.scale
        
        // 计算到鼠标的距离
        const dx = d.px - focusX
        const dy = d.py - focusY
        d.distToMouse = Math.sqrt(dx * dx + dy * dy)
      })
      
      // 找出离鼠标最近的1个节点
      const sortedByDist = [...nodes].sort((a, b) => a.distToMouse - b.distToMouse)
      const closestNode = sortedByDist[0]
      
      // 移动节点并缩放整个节点组（让弧线、光晕等子元素一起缩放）
      nodeGs.attr('transform', d => `translate(${d.fx},${d.fy}) scale(${d.fscale})`)
      
      // 只显示最近的1个节点的标签
      const getOpacity = d => {
        if (d !== closestNode) return 0
        if (d.distToMouse > fisheyeRadius) return 0
        return 1
      }
      
      // 智能调整标签位置，避免超出边界
      labelGs.attr('transform', d => {
        if (d !== closestNode) return `translate(${d.fx},${d.fy})`
        
        let lx = d.fx
        let ly = d.fy
        const labelW = (d.labelWidth || 100) / 2 + 10
        const labelH = (d.labelHeight || 14) + (d.baseSize * d.fscale) + 20
        
        // 检查边界并调整
        const maxX = cx - 10
        const maxY = cy - 10
        
        // 水平边界
        if (lx - labelW < -maxX) lx = -maxX + labelW
        if (lx + labelW > maxX) lx = maxX - labelW
        
        // 垂直边界（标签在上方）
        if (ly - labelH < -maxY) {
          // 如果上方超出，把标签放到下方
          d.labelBelow = true
        } else {
          d.labelBelow = false
        }
        
        return `translate(${lx},${ly})`
      })
      
      nodeLabels
        .attr('opacity', getOpacity)
        .attr('font-size', 13)
        .attr('dy', d => d.labelBelow ? (d.baseSize * d.fscale) + 20 : -(d.baseSize * d.fscale) - 14)
      
      nodeLabelBgs
        .attr('opacity', d => getOpacity(d))
        .attr('y', d => d.labelBelow 
          ? (d.baseSize * d.fscale) + 20 - (d.labelHeight || 14) + 2
          : -(d.baseSize * d.fscale) - 14 - (d.labelHeight || 14) + 2)
      
      // 更新连线（保持曲线）
      linkPaths.attr('d', d => {
        const sourceResult = fisheye([d.source.px, d.source.py], [focusX, focusY])
        const targetResult = fisheye([d.target.px, d.target.py], [focusX, focusY])
        const sx = sourceResult.x, sy = sourceResult.y
        const tx = targetResult.x, ty = targetResult.y
        const cpx = (sx + tx) / 2 * 0.85, cpy = (sy + ty) / 2 * 0.85
        return `M${sx},${sy}Q${cpx},${cpy} ${tx},${ty}`
      })
    }

    // 重置鱼眼效果
    const resetFisheye = () => {
      fisheyeBorder.attr('opacity', 0)
      nodeGs.attr('transform', d => `translate(${d.px},${d.py}) scale(1)`)
      labelGs.attr('transform', d => `translate(${d.px},${d.py})`)
      nodeLabels.attr('opacity', 0)
      nodeLabelBgs.attr('opacity', 0)
      linkPaths.attr('d', d => {
        const sx = d.source.px, sy = d.source.py
        const tx = d.target.px, ty = d.target.py
        const cpx = (sx + tx) / 2 * 0.85, cpy = (sy + ty) / 2 * 0.85
        return `M${sx},${sy}Q${cpx},${cpy} ${tx},${ty}`
      })
    }

    // SVG 鼠标事件
    svg.on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event, g.node())
      setMousePos({ x: mx, y: my })
      applyFisheye(mx, my)
    })
    
    svg.on('mouseleave', () => {
      setMousePos(null)
      resetFisheye()
      setHoveredNode(null)
    })

    // 节点交互
    nodeCircles
      .on('mouseenter', (event, d) => {
        event.stopPropagation()
        setHoveredNode(d)
        
        // 更新左上角：显示当前节点信息
        const risk = riskMap.get(d.data.name)
        topLeftText.text(getLeafDisplayName(d.data.name))
        if (risk && !risk.inherited) {
          topLeftText.attr('fill', getRiskColor(risk.level))
          topLeftSub.text(`Your data may be exposed here`)
            .attr('fill', getRiskColor(risk.level))
        } else if (risk && risk.inherited) {
          topLeftText.attr('fill', 'var(--color-text-primary)')
          topLeftSub.text('Contains risk items below')
            .attr('fill', 'var(--color-text-tertiary)')
        } else {
          topLeftText.attr('fill', 'var(--color-text-primary)')
          topLeftSub.text('No risk detected here')
            .attr('fill', 'var(--color-text-tertiary)')
        }
        
        // 更新右下角：风险的具体原因
        if (risk && !risk.inherited && risk.risks && risk.risks.length > 0) {
          riskDetailTitle.text('WHY THIS IS A RISK')
          
          const firstRisk = risk.risks[0]
          const reason = firstRisk.reason || firstRisk.inference_chain || ''
          riskDetailLine1.text(reason.length > 36 ? reason.slice(0, 34) + '...' : reason)
          
          // 显示涉及多少条你的信息
          const usedInfons = firstRisk.used_infons || firstRisk.infon_ids || []
          const infonCount = Array.isArray(usedInfons) ? usedInfons.length : 0
          riskDetailLine2.text(infonCount > 0 
            ? `Based on ${infonCount} piece${infonCount > 1 ? 's' : ''} of your info`
            : '')
        } else {
          riskDetailTitle.text('')
          riskDetailLine1.text('')
          riskDetailLine2.text('')
        }
        
        // 高亮当前节点
        d3.select(event.currentTarget)
          .attr('stroke', 'var(--color-accent-primary)')
          .attr('stroke-width', 2.5)
        
        // 高亮路径到根
        const ancestors = d.ancestors()
        const ancestorNames = new Set(ancestors.map(a => a.data.name))
        
        linkPaths
          .attr('stroke', link => {
            if (ancestorNames.has(link.source.data.name) && ancestorNames.has(link.target.data.name)) {
              return 'var(--color-accent-primary)'
            }
            return 'var(--color-border-light)'
          })
          .attr('stroke-width', link => {
            if (ancestorNames.has(link.source.data.name) && ancestorNames.has(link.target.data.name)) {
              return 2.5
            }
            return 1
          })
          .attr('stroke-opacity', link => {
            if (ancestorNames.has(link.source.data.name) && ancestorNames.has(link.target.data.name)) {
              return 1
            }
            return 0.2
          })
      })
      .on('mouseleave', (event, d) => {
        // 恢复左上角默认安全状态
        topLeftText.text(topLeftDefault.text)
          .attr('fill', topLeftDefault.color)
        topLeftSub.text(topLeftDefault.sub)
          .attr('fill', 'var(--color-text-tertiary)')
        
        // 清空右下角
        riskDetailTitle.text('')
        riskDetailLine1.text('')
        riskDetailLine2.text('')
        
        // 恢复节点边框
        d3.select(event.currentTarget)
          .attr('stroke', getNodeStroke(d))
          .attr('stroke-width', getNodeStrokeWidth(d))
        
        // 恢复连线
        linkPaths
          .attr('stroke', 'var(--color-border-light)')
          .attr('stroke-width', 1)
          .attr('stroke-opacity', 0.5)
      })

  }, [lawData, lawIdx, size, riskMap])

  // 处理复选框变化
  const handleCheckboxChange = (itemId) => {
    togglePrivacyItem(itemId)
  }

  // 添加自定义隐私项
  const handleAddCustomItem = () => {
    const trimmed = newItemInput.trim()
    if (!trimmed) return
    
    const newItem = {
      id: `custom_${Date.now()}`,
      label: trimmed,
      category: 'Custom'
    }
    
    addCustomPrivacyItem(newItem)
    setNewItemInput('')
    
    // 自动选中新添加的项
    if (!selectedPrivacyItems.has(newItem.id)) {
      togglePrivacyItem(newItem.id)
    }
  }
  
  // 删除自定义隐私项
  const handleRemoveCustomItem = (itemId) => {
    removeCustomPrivacyItem(itemId)
    // 同时从选中项中移除
    if (selectedPrivacyItems.has(itemId)) {
      togglePrivacyItem(itemId)
    }
  }

  // 长按法律选项卡处理
  const handleLawMouseDown = (idx) => {
    setHoldingLawIdx(idx)
    holdStartRef.current = Date.now()
    
    const updateProgress = () => {
      if (!holdStartRef.current) return
      
      const elapsed = Date.now() - holdStartRef.current
      const progress = Math.min((elapsed / 1000) * 100, 100) // 1秒
      setHoldProgress(progress)
      
      if (progress >= 100) {
        // 触发推理
        handleTriggerInference(idx)
        setHoldingLawIdx(null)
        setHoldProgress(0)
        holdStartRef.current = null
      } else {
        holdTimerRef.current = requestAnimationFrame(updateProgress)
      }
    }
    
    holdTimerRef.current = requestAnimationFrame(updateProgress)
  }

  const handleLawMouseUp = (idx) => {
    const wasHolding = holdingLawIdx === idx
    const progress = holdProgress
    
    setHoldingLawIdx(null)
    setHoldProgress(0)
    holdStartRef.current = null
    if (holdTimerRef.current) {
      cancelAnimationFrame(holdTimerRef.current)
      holdTimerRef.current = null
    }
    
    // 如果长按未完成（进度<100%），则视为普通点击，只切换法律不推理
    if (wasHolding && progress < 100) {
      setLawIdx(idx)
    }
  }

  const handleTriggerInference = async (idx) => {
    if (!session) {
      console.warn('[LawTree] 没有会话，无法触发推理')
      return
    }
    
    console.log('[LawTree] 触发推理:', LAWS[idx].key, '当前推理状态:', inference?.status)
    
    try {
      // 1. 如果正在推理，先中断
      if (inference?.status === 'running') {
        console.log('[LawTree] 中断正在进行的推理')
        abortPrivacyInference(session)
        // 等待中断完成
        await new Promise(resolve => setTimeout(resolve, 150))
      }
      
      // 2. 清除当前推理结果（无论是否完成）
      if (inference) {
        console.log('[LawTree] 清除旧的推理结果')
        clearPrivacyInference()
        // 等待清除完成
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      
      // 3. 切换法律显示
      setLawIdx(idx)
      
      // 4. 准备法律数据
      let lawKey = LAWS[idx].key
      let lawDataToSet = null
      
      if (lawKey === 'CUSTOM') {
        console.log('[LawTree] 设置 Custom 法律，选中项:', selectedPrivacyItems.size)
        
        // 构建选中的隐私项详细信息列表
        const allItems = [...PRIVACY_ITEMS, ...customPrivacyItems]
        const selectedItemsDetails = allItems.filter(item => selectedPrivacyItems.has(item.id))
        
        lawDataToSet = { 
          customItems: selectedItemsDetails, // 传递完整的项信息，而不仅仅是ID
          isCustom: true 
        }
      } else {
        if (!lawData[idx]) {
          console.warn('[LawTree] 法律数据未加载:', lawKey)
          return
        }
        console.log('[LawTree] 设置法律:', lawKey)
        lawDataToSet = lawData[idx]
      }
      
      // 5. 更新 selectedLaw
      setSelectedLaw(lawKey, lawDataToSet)
      
      // 6. 等待一下确保状态更新完成，然后触发推理
      await new Promise(resolve => setTimeout(resolve, 250))
      
      if (startPrivacyInference) {
        console.log('[LawTree] 启动新的推理')
        const result = await startPrivacyInference(null) // 手动触发，不排除任何消息
        console.log('[LawTree] 推理已启动:', result)
      } else {
        console.error('[LawTree] startPrivacyInference 不可用')
      }
    } catch (error) {
      console.error('[LawTree] 触发推理失败:', error)
    }
  }

  // 清理定时器
  useEffect(() => {
    return () => {
      if (holdTimerRef.current) {
        cancelAnimationFrame(holdTimerRef.current)
      }
    }
  }, [])

  // 按类别分组隐私项（包含自定义项）
  const groupedItems = useMemo(() => {
    const groups = {}
    const allItems = [...PRIVACY_ITEMS, ...customPrivacyItems]
    allItems.forEach(item => {
      if (!groups[item.category]) {
        groups[item.category] = []
      }
      groups[item.category].push(item)
    })
    return groups
  }, [customPrivacyItems])

  // 判断是否显示自定义选项界面
  const isCustomMode = LAWS[lawIdx].key === 'CUSTOM'

  // Custom模式的风险映射：匹配推理结果到隐私项（中文注释）
  const customRiskMap = useMemo(() => {
    const map = new Map() // key: 隐私项ID, value: { level, confidence, risks }
    
    if (!isCustomMode || !inference || !inference.risks) return map
    
    // 检查推理结果是否与Custom法律匹配
    if (inference.lawKey && inference.lawKey !== 'CUSTOM') {
      return map
    }
    
    console.log('[LawTree] Custom模式风险映射，推理风险数:', inference.risks.length)
    
    // 只使用选中的隐私项（预设+自定义）
    const allItems = [...PRIVACY_ITEMS, ...customPrivacyItems]
    const selectedItems = allItems.filter(item => selectedPrivacyItems.has(item.id))
    
    console.log('[LawTree] 选中的隐私项数:', selectedItems.length, selectedItems.map(i => i.label))
    
    // 为每个推理风险找到匹配的隐私项
    inference.risks.forEach(risk => {
      const lawNodeName = (risk.law_node_name || '').trim()
      
      // 跳过还没有law_node_name的部分风险对象
      if (!lawNodeName || lawNodeName === 'Loading...') {
        return
      }
      
      console.log('[LawTree] 处理风险:', lawNodeName)
      
      // 只匹配选中的项
      selectedItems.forEach(item => {
        const itemLabel = item.label.trim()
        
        // 精确匹配策略：law_node_name必须与隐私项标签匹配
        let isMatch = false
        
        // 1. 精确匹配（忽略大小写）
        if (lawNodeName.toLowerCase() === itemLabel.toLowerCase()) {
          isMatch = true
          console.log('[LawTree] 精确匹配:', itemLabel)
        }
        
        // 2. 部分匹配：law_node_name包含完整的项标签（作为独立词）
        if (!isMatch) {
          // 使用单词边界匹配，避免部分词匹配
          const regex = new RegExp(`\\b${itemLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i')
          if (regex.test(lawNodeName)) {
            isMatch = true
            console.log('[LawTree] 词边界匹配:', itemLabel)
          }
        }
        
        if (isMatch) {
          const levelPriority = { HIGH: 3, MEDIUM: 2, LOW: 1, UNKNOWN: 0 }
          
          // 支持部分风险对象（可能缺少某些字段）
          const riskLevel = risk.risk_level || 'UNKNOWN'
          const confidence = risk.confidence ?? 0
          
          if (!map.has(item.id)) {
            map.set(item.id, {
              level: riskLevel,
              confidence: confidence,
              risks: [risk]
            })
            console.log('[LawTree] 添加风险映射:', item.label, riskLevel)
          } else {
            const existing = map.get(item.id)
            if ((levelPriority[riskLevel] || 0) > (levelPriority[existing.level] || 0)) {
              existing.level = riskLevel
            }
            existing.confidence = Math.max(existing.confidence, confidence)
            // 使用_objIndex去重，避免部分更新导致重复
            const existingIndices = existing.risks.map(r => r._objIndex).filter(i => i !== undefined)
            if (risk._objIndex === undefined || !existingIndices.includes(risk._objIndex)) {
              existing.risks.push(risk)
            } else {
              // 更新现有的risk对象
              const idx = existing.risks.findIndex(r => r._objIndex === risk._objIndex)
              if (idx >= 0) {
                existing.risks[idx] = risk
              }
            }
          }
        }
      })
    })
    
    console.log('[LawTree] 最终风险映射数:', map.size)
    
    return map
  }, [isCustomMode, inference, customPrivacyItems, selectedPrivacyItems])

  // 渲染自定义隐私项复选框界面
  const renderCustomPrivacyOptions = () => (
    <div style={{ 
      height: size.height,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>
      {/* 顶部：添加自定义项的输入框 */}
      <div style={{ 
        padding: '8px 10px',
        borderBottom: '1px solid var(--color-border-light)',
        flexShrink: 0
      }}>
        <div style={{ 
          display: 'flex',
          gap: '6px',
          alignItems: 'center'
        }}>
          <input
            type="text"
            value={newItemInput}
            onChange={(e) => setNewItemInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAddCustomItem()}
            placeholder="Add custom privacy item..."
            style={{
              flex: 1,
              padding: '6px 10px',
              fontSize: 12,
              border: '1px solid var(--color-border-light)',
              borderRadius: 6,
              background: 'var(--color-bg-primary)',
              color: 'var(--color-text-primary)',
              outline: 'none'
            }}
          />
          <button
            onClick={handleAddCustomItem}
            style={{
              padding: '6px 12px',
              fontSize: 12,
              fontWeight: 600,
              background: 'var(--color-bg-tertiary)',
              color: 'var(--color-text-primary)',
              border: '1px solid var(--color-border-light)',
              borderRadius: 6,
              cursor: 'pointer',
              whiteSpace: 'nowrap'
            }}
          >
            Add
          </button>
        </div>
        <div style={{
          marginTop: 4,
          fontSize: 11,
          color: 'var(--color-text-tertiary)',
          textAlign: 'center',
          display: 'flex',
          gap: '8px',
          justifyContent: 'center',
          alignItems: 'center',
          flexWrap: 'wrap'
        }}>
          <span>{selectedPrivacyItems.size} item{selectedPrivacyItems.size !== 1 ? 's' : ''} selected</span>
          {customRiskMap.size > 0 && (
            <>
              <span style={{ color: 'var(--color-border-light)' }}>|</span>
              <span style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '4px',
                fontWeight: 600,
                color: '#ef4444'
              }}>
                ⚠️ {customRiskMap.size} risk{customRiskMap.size !== 1 ? 's' : ''} detected
              </span>
            </>
          )}
        </div>
      </div>

      {/* 中部：滚动区域，显示所有隐私项 */}
      <div style={{ 
        flex: 1,
        overflowY: 'auto',
        padding: '8px 10px'
      }}>
        {Object.entries(groupedItems).map(([category, items]) => (
          <div key={category} style={{ marginBottom: 10 }}>
            <div style={{ 
              fontSize: 11, 
              fontWeight: 600, 
              color: 'var(--color-text-secondary)',
              marginBottom: 4,
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              {category}
            </div>
            <div style={{ 
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
              gap: '4px'
            }}>
              {items.map(item => {
                const isCustomItem = item.id.startsWith('custom_')
                const isSelected = selectedPrivacyItems.has(item.id)
                // 只有选中的项才检查风险
                const risk = isSelected ? customRiskMap.get(item.id) : null
                
                // 根据风险等级获取颜色
                const getRiskColor = (level) => {
                  switch (level) {
                    case 'HIGH': return { bg: '#fef2f2', border: '#ef4444', dot: '#ef4444' }    // 红色
                    case 'MEDIUM': return { bg: '#fff7ed', border: '#f97316', dot: '#f97316' }  // 橙色
                    case 'LOW': return { bg: '#fefce8', border: '#eab308', dot: '#eab308' }     // 黄色
                    case 'UNKNOWN': return { bg: '#f1f5f9', border: '#94a3b8', dot: '#94a3b8' } // 灰色（部分数据）
                    default: return null
                  }
                }
                
                const riskColors = risk ? getRiskColor(risk.level) : null
                
                return (
                  <div
                    key={item.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      background: riskColors 
                        ? riskColors.bg
                        : selectedPrivacyItems.has(item.id) 
                          ? 'var(--color-bg-tertiary)' 
                          : 'transparent',
                      borderRadius: 6,
                      border: riskColors 
                        ? `1.5px solid ${riskColors.border}`
                        : '1px solid var(--color-border-light)',
                      transition: 'all 0.15s',
                      fontSize: 12,
                      padding: '0',
                      overflow: 'hidden',
                      position: 'relative'
                    }}
                    title={risk ? `Risk: ${risk.level} (${(risk.confidence * 100).toFixed(0)}% confidence)\n${risk.risks.length} risk(s) detected` : ''}
                  >
                    {risk && riskColors && (
                      <div style={{
                        position: 'absolute',
                        top: '2px',
                        right: isCustomItem ? '30px' : '2px',
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: riskColors.dot,
                        boxShadow: `0 0 4px ${riskColors.dot}`
                      }} />
                    )}
                    <label
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        padding: '5px 8px',
                        flex: 1,
                        cursor: 'pointer',
                        color: 'var(--color-text-primary)',
                        minWidth: 0
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={selectedPrivacyItems.has(item.id)}
                        onChange={() => handleCheckboxChange(item.id)}
                        style={{ 
                          cursor: 'pointer',
                          width: '14px',
                          height: '14px',
                          flexShrink: 0
                        }}
                      />
                      <span style={{ 
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        fontWeight: risk ? 600 : 400
                      }}>
                        {item.label}
                      </span>
                    </label>
                    {isCustomItem && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleRemoveCustomItem(item.id)
                        }}
                        style={{
                          padding: '4px 6px',
                          background: 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          color: 'var(--color-text-tertiary)',
                          fontSize: '14px',
                          display: 'flex',
                          alignItems: 'center',
                          transition: 'color 0.15s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.color = '#ef4444'}
                        onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-tertiary)'}
                        title="Delete custom item"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text-primary)', marginBottom: 6, paddingLeft: 4 }}>
        Privacy Exposure Tree
      </div>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          minHeight: 200,
          background: 'var(--color-bg-secondary)',
          borderRadius: 14,
          boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
          border: '1px solid var(--color-border-light)',
          padding: '4px 4px 0 4px',
          overflow: 'hidden'
        }}
      >
        <div style={{ display: 'flex', gap: 6, marginBottom: 4, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'center' }}>
          {LAWS.map((law, idx) => {
            const isHolding = holdingLawIdx === idx
            const isActive = lawIdx === idx
            
            return (
              <div key={law.key} style={{ position: 'relative' }}>
                <div
                  onMouseDown={() => handleLawMouseDown(idx)}
                  onMouseUp={() => handleLawMouseUp(idx)}
                  onMouseLeave={() => handleLawMouseUp(idx)}
                  onTouchStart={() => handleLawMouseDown(idx)}
                  onTouchEnd={() => handleLawMouseUp(idx)}
                  style={{
                    cursor: 'pointer',
                    padding: '5px 10px',
                    borderRadius: 8,
                    fontWeight: 600,
                    fontSize: 11,
                    color: isActive ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                    background: isHolding
                      ? `linear-gradient(to right, #3b82f6 ${holdProgress}%, var(--color-bg-tertiary) ${holdProgress}%)`
                      : isActive 
                        ? 'var(--color-bg-tertiary)' 
                        : 'transparent',
                    borderWidth: 0.5,
                    borderStyle: 'solid',
                    borderColor: 'var(--color-border-light)',
                    boxShadow: isActive ? '0 0 0 1px #334155' : 'none',
                    transition: isHolding ? 'none' : 'background-color 0.18s, color 0.18s',
                    transform: isHolding ? 'scale(0.98)' : 'scale(1)',
                    userSelect: 'none',
                  }}
                >
                  {law.label}
                </div>
                {isHolding && (
                  <div style={{
                    position: 'absolute',
                    top: '100%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    marginTop: 4,
                    padding: '4px 8px',
                    background: 'rgba(0,0,0,0.8)',
                    color: 'white',
                    fontSize: 10,
                    borderRadius: 4,
                    whiteSpace: 'nowrap',
                    pointerEvents: 'none',
                    zIndex: 10
                  }}>
                    Hold for {((1000 - holdProgress * 10) / 1000).toFixed(1)}s to analyze...
                  </div>
                )}
              </div>
            )
          })}
          
          {inference && inference.lawKey && inference.lawKey !== LAWS[lawIdx].key && !isCustomMode && (
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
        {isCustomMode ? (
          renderCustomPrivacyOptions()
        ) : (
          <svg ref={svgRef} style={{ 
            width: '100%', 
            height: size.height, 
            display: 'block'
          }} />
        )}
      </div>
    </div>
  )
}
