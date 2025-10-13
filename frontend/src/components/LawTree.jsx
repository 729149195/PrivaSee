import React, { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { useStore } from '../store'

const LAWS = [
  { key: 'PIPL', label: 'PIPL', file: './law/PIPL.json' },
  { key: 'GDPR', label: 'GDPR', file: './law/GDPR.json' },
  { key: 'CCPA_CPRA', label: 'CCPA/CPRA', file: './law/CCPA_CPRA.json' },
  { key: 'CUSTOM', label: 'Custom', file: null }, // Custom mode uses checkboxes instead of tree
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

export default function LawTree() {
  const [lawData, setLawData] = useState([null, null, null, null])
  const [newItemInput, setNewItemInput] = useState('') // 新隐私项输入框
  const [holdingLawIdx, setHoldingLawIdx] = useState(null) // 正在长按的法律索引
  const [holdProgress, setHoldProgress] = useState(0) // 长按进度 0-100
  const containerRef = useRef(null)
  const svgRef = useRef(null)
  const holdTimerRef = useRef(null)
  const holdStartRef = useRef(null)
  const [size, setSize] = useState({ width: 928, height: 600 })
  
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
    if (lawData[0] && !useStore.getState().selectedLaw) {
      setSelectedLaw(LAWS[0].key, lawData[0])
    }
    // eslint-disable-next-line
  }, [lawData])

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
        const result = await startPrivacyInference()
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
                    case 'MEDIUM': return { bg: '#fffbeb', border: '#f59e0b', dot: '#f59e0b' }  // 橙色
                    case 'LOW': return { bg: '#f0fdf4', border: '#10b981', dot: '#10b981' }     // 绿色
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
                    padding: '6px 12px',
                    borderRadius: 10,
                    fontWeight: 600,
                    fontSize: 12,
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
          <svg ref={svgRef} style={{ width: '100%', height: size.height, display: 'block' }} />
        )}
      </div>
    </div>
  )
}
