import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { Layout, Input, Button, Upload, Splitter } from 'antd'
import { SearchOutlined, UploadOutlined, SendOutlined } from '@ant-design/icons'
import styles from './ConversationViz.module.css'

/**
 * ConversationViz: MLLM multimodal conversation visualization (timeline + bubbles + references)
 * UI text in English; comments may be bilingual.
 */
export default function ConversationViz({ data }) {
  // In-component two-pane layout: left=messages, right=gitflow-like flow (single mode)
  // Demo data when no external data provided
  const demoData = useMemo(() => ({
    messages: [
      {
        id: 'msg_001',
        type: 'user',
        timestamp: '2024-01-01T10:00:00Z',
        content: {
          text: 'Hi, can you describe these images and extract any visible text? Also, please compute 17 * 23.',
          images: ['https://picsum.photos/seed/a/300/180', 'https://picsum.photos/seed/b/220/160'],
          files: []
        },
        references: [],
        tools_used: [],
        metadata: { tokens: 32, processing_time: 0.03 }
      },
      {
        id: 'msg_002',
        type: 'assistant',
        timestamp: '2024-01-01T10:00:02Z',
        content: {
          text: 'I will first run OCR and object detection on the images, then compute the result.'
        },
        references: ['msg_001'],
        tools_used: ['ocr', 'detector', 'calculator'],
        metadata: { tokens: 41, processing_time: 0.25 }
      },
      {
        id: 'msg_003',
        type: 'assistant',
        timestamp: '2024-01-01T10:00:04Z',
        content: {
          text: 'OCR result: "Cafe Roma". Objects: cup, book. 17 * 23 = 391.'
        },
        references: ['msg_001', 'msg_002'],
        tools_used: ['ocr', 'detector', 'calculator'],
        metadata: { tokens: 28, processing_time: 0.62 }
      },
      {
        id: 'msg_004',
        type: 'user',
        timestamp: '2024-01-01T10:00:08Z',
        content: {
          text: 'Great. Summarize in one sentence and attach a small preview.'
        },
        references: ['msg_003'],
        tools_used: [],
        metadata: { tokens: 15, processing_time: 0.02 }
      },
      {
        id: 'msg_005',
        type: 'assistant',
        timestamp: '2024-01-01T10:00:10Z',
        content: {
          text: 'Summary: The images show a cup and a book; OCR reads "Cafe Roma".',
          images: ['https://picsum.photos/seed/c/200/120']
        },
        references: ['msg_003', 'msg_004'],
        tools_used: [],
        metadata: { tokens: 24, processing_time: 0.12 }
      }
    ],
    topics: [
      { name: 'Vision + OCR', message_ids: ['msg_001', 'msg_002', 'msg_003', 'msg_005'], color: '#1677ff' },
      { name: 'Math', message_ids: ['msg_001', 'msg_002', 'msg_003'], color: '#fa8c16' },
    ]
  }), [])

  const conv = data && data.messages && data.messages.length ? data : demoData

  // UI states
  const [expanded, setExpanded] = useState(() => new Set())
  const [hoveredMsgId, setHoveredMsgId] = useState('')
  const [search, setSearch] = useState('')
  const [activeTopic, setActiveTopic] = useState('')
  
  // Chat input states
  const [inputText, setInputText] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState([])
  
  // Context window details state
  const [showContextDetails, setShowContextDetails] = useState(false)

  // Refs for layout + lines rendering (messages tab)
  const containerRef = useRef(null)
  const overlayRef = useRef(null)
  const msgRefs = useRef(new Map())
  const bubbleRefs = useRef(new Map())

  // Flow refs
  const flowSvgRef = useRef(null)

  // Filter utilities
  const searchLower = (search || '').trim().toLowerCase()
  const matchMessage = (m) => {
    if (!searchLower) return true
    const t = (m?.content?.text || '').toLowerCase()
    const tool = (m?.tools_used || []).join(',').toLowerCase()
    return t.includes(searchLower) || tool.includes(searchLower)
  }
  const inActiveTopic = (m) => {
    if (!activeTopic) return true
    const topic = conv.topics.find((t) => t.name === activeTopic)
    if (!topic) return true
    return topic.message_ids.includes(m.id)
  }

  // First filter messages, then merge consecutive assistant messages for display
  const filteredMessages = useMemo(() => conv.messages.filter((m) => matchMessage(m) && inActiveTopic(m)), [conv, searchLower, activeTopic])
  
  const visibleMessages = useMemo(() => {
    const result = []
    let i = 0
    
    while (i < filteredMessages.length) {
      const msg = filteredMessages[i]
      
      if (msg.type === 'assistant') {
        // Merge consecutive assistant messages
        const assistantMsgs = []
        let j = i
        
        while (j < filteredMessages.length && filteredMessages[j].type === 'assistant') {
          assistantMsgs.push(filteredMessages[j])
          j++
        }
        
        // Create merged assistant message
        const allTexts = assistantMsgs.map(m => m.content?.text).filter(Boolean)
        const allImages = assistantMsgs.flatMap(m => m.content?.images || [])
        const allFiles = assistantMsgs.flatMap(m => m.content?.files || [])
        const allTools = [...new Set(assistantMsgs.flatMap(m => m.tools_used || []))]
        
        const mergedMsg = {
          ...assistantMsgs[0], // Use first message as base
          id: `merged_${assistantMsgs[0].id}`,
          content: {
            text: allTexts.join('\n\n'),
            images: allImages,
            files: allFiles
          },
          tools_used: allTools,
          originalMessages: assistantMsgs // Keep reference to original messages
        }
        
        result.push(mergedMsg)
        i = j
      } else {
        result.push(msg)
        i++
      }
    }
    
    return result
  }, [filteredMessages])
  const idToIndex = useMemo(() => Object.fromEntries(visibleMessages.map((m, i) => [m.id, i])), [visibleMessages])

  // Flow graph: build Q&A pairs and vertical main flow
  const flowData = useMemo(() => {
    const nodes = []
    const links = []

    // Build Q&A pairs: visibleMessages already has merged assistant messages
    const pairs = []
    let i = 0
    while (i < visibleMessages.length) {
      const msg = visibleMessages[i]
      if (msg.type === 'user') {
        // This is a user question
        const userMsg = msg
        
        // Find the next assistant message (already merged)
        let assistantMsg = null
        if (i + 1 < visibleMessages.length && visibleMessages[i + 1].type === 'assistant') {
          assistantMsg = visibleMessages[i + 1]
          i += 2 // Skip both user and assistant
        } else {
          i += 1 // Only user, no response
        }
        
        pairs.push({ user: userMsg, assistant: assistantMsg })
      } else {
        i++
      }
    }

    // Create main flow: Q -> Context -> A for each pair
    pairs.forEach((pair, pairIndex) => {
      const userMsg = pair.user
      const assistantMsg = pair.assistant
      
      // User input node - count each type based on actual content
      const userMicro = []
      if (userMsg.content?.text) userMicro.push('text')
      // Add one 'image' for each image
      if (userMsg.content?.images?.length) {
        for (let i = 0; i < userMsg.content.images.length; i++) {
          userMicro.push('image')
        }
      }
      // Add one 'file' for each file
      if (userMsg.content?.files?.length) {
        for (let i = 0; i < userMsg.content.files.length; i++) {
          userMicro.push('file')
        }
      }
      
      nodes.push({
        id: `user_${pairIndex}`,
        label: `User Q${pairIndex + 1}`,
        type: 'user',
        pairIndex,
        micro: userMicro.length ? userMicro : ['text']
      })

      // Context node for this pair
      nodes.push({
        id: `ctx_${pairIndex}`,
        label: `Context ${pairIndex + 1}`,
        type: 'context',
        pairIndex,
        micro: ['tokens']
      })

      // Assistant response node (already merged) - only show content, not tools
      if (assistantMsg) {
        const assistantMicro = []
        if (assistantMsg.content?.text) assistantMicro.push('text')
        // Add one 'image' for each image
        if (assistantMsg.content?.images?.length) {
          for (let i = 0; i < assistantMsg.content.images.length; i++) {
            assistantMicro.push('image')
          }
        }
        // Add one 'file' for each file
        if (assistantMsg.content?.files?.length) {
          for (let i = 0; i < assistantMsg.content.files.length; i++) {
            assistantMicro.push('file')
          }
        }
        // Tools are displayed in external system nodes, not in assistant content
        
        nodes.push({
          id: `assistant_${pairIndex}`,
          label: `Assistant A${pairIndex + 1}`,
          type: 'assistant',
          pairIndex,
          micro: assistantMicro.length ? assistantMicro : ['text']
        })

        // Main flow links
        links.push({ source: `user_${pairIndex}`, target: `ctx_${pairIndex}`, value: 3 })
        links.push({ source: `ctx_${pairIndex}`, target: `assistant_${pairIndex}`, value: 3 })

        // External system nodes for this pair
        const allTools = new Set(assistantMsg.tools_used || [])
        
        // For demo data, all tools (ocr, detector, calculator) should go to MCP node
        if (allTools.size > 0) {
          const mcpTools = Array.from(allTools)
          nodes.push({
            id: `mcp_${pairIndex}`,
            label: 'MCP Tools',
            type: 'mcp',
            pairIndex,
            micro: mcpTools
          })
          links.push({ source: `mcp_${pairIndex}`, target: `ctx_${pairIndex}`, value: 2 })
        }
        
        if (['retriever', 'ranker', 'vector'].some(tool => allTools.has(tool))) {
          nodes.push({
            id: `rag_${pairIndex}`,
            label: 'RAG',
            type: 'rag',
            pairIndex,
            micro: ['retriever', 'ranker', 'vector']
          })
          links.push({ source: `rag_${pairIndex}`, target: `ctx_${pairIndex}`, value: 2 })
        }
      } else {
        // No assistant response, just add the main flow connection to context
        links.push({ source: `user_${pairIndex}`, target: `ctx_${pairIndex}`, value: 3 })
      }
    })

    // Add global context node (independent, accumulative)
    nodes.push({ id: 'global_context', label: 'Global Context', type: 'context', micro: ['tokens'] })

    // No direct assistant -> user arrows as this would be reverse data flow
    // Conversation continuity is represented through the global context flows

    // Global context aggregation: ctx_i -> global_context; and feed-forward: global_context -> ctx_{i} (i>0)
    for (let i = 0; i < pairs.length; i++) {
      const ctxId = `ctx_${i}`
      if (nodes.find(n => n.id === ctxId)) {
        links.push({ source: ctxId, target: 'global_context', value: 1, type: 'context_in' })
        if (i > 0) {
          links.push({ source: 'global_context', target: ctxId, value: 1, type: 'context_out' })
        }
      }
    }

    // Compute global context usage from message metadata
    const totalTokens = visibleMessages.reduce((sum, m) => sum + (m?.metadata?.tokens || 0), 0)
    const capacity = 200000
    const used = Math.min(totalTokens, capacity)
    const evicted = Math.max(0, totalTokens - capacity)

    // Count global context connections for badge display
    const contextInCount = links.filter(l => l.type === 'context_in').length
    const contextOutCount = links.filter(l => l.type === 'context_out').length

    return { nodes, links, ctx: { capacity, used, evicted, contextInCount, contextOutCount } }
  }, [visibleMessages])

  // Toggle expand
  const toggleExpand = (id) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  // Layout constants (messages)
  const spineX = 36 // left rail center
  const rowGap = 14

  // Draw lines after layout (messages) — no timeline on left, so clear overlay
  useEffect(() => {
    const svg = d3.select(overlayRef.current)
    const container = containerRef.current
    if (!svg.node() || !container) return

    const rect = container.getBoundingClientRect()
    const height = container.scrollHeight
    const width = rect.width
    svg.attr('width', width).attr('height', height)
    svg.selectAll('*').remove()

  }, [visibleMessages, expanded])

  // Re-render on resize to keep lines aligned
  useEffect(() => {
    const ro = new ResizeObserver(() => {
      // Trigger re-render by toggling state lightly
      setHoveredMsgId((prev) => prev)
    })
    if (containerRef.current) ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  // Tooltip state (messages tab)
  const tooltipRef = useRef(null)
  const [tooltip, setTooltip] = useState({ show: false, x: 0, y: 0, text: '' })

  const onBubbleEnter = (e, m) => {
    const rect = containerRef.current?.getBoundingClientRect()
    const x = e.clientX - (rect?.left || 0)
    const y = e.clientY - (rect?.top || 0)
    const meta = m?.metadata || {}
    const txt = `${new Date(m.timestamp).toLocaleString()} · tokens=${meta.tokens ?? '-'} · time=${meta.processing_time ?? '-'}s`
    setTooltip({ show: true, x, y, text: txt })
    setHoveredMsgId(m.id)
  }
  const onBubbleLeave = () => {
    setTooltip((t) => ({ ...t, show: false }))
    setHoveredMsgId('')
  }

  const isUser = (m) => m.type === 'user'
  const bubbleClass = (m) => [styles.bubble, isUser(m) ? styles.bubbleUser : styles.bubbleAssistant].join(' ')

  // Export current overlay SVG
  const exportSvg = () => {
    const svgEl = overlayRef.current
    if (!svgEl) return
    const serializer = new XMLSerializer()
    const src = serializer.serializeToString(svgEl)
    const blob = new Blob([src], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'conversation_links.svg'
    a.click()
    URL.revokeObjectURL(url)
  }

  const onSearchKey = (e) => {
    if (e.key === 'Enter') {
      // Scroll to first match
      for (const m of visibleMessages) {
        if (matchMessage(m)) {
          const el = msgRefs.current.get(m.id)
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
          break
        }
      }
    }
  }

  // Chat input handlers
  const handleSendMessage = () => {
    if (inputText.trim() || uploadedFiles.length > 0) {
      console.log('Sending message:', { text: inputText, files: uploadedFiles })
      setInputText('')
      setUploadedFiles([])
    }
  }

  const handleFileUpload = (info) => {
    const { fileList } = info
    setUploadedFiles(fileList)
  }

  // Draw right-side vertical flow with drag support and different node shapes
  useEffect(() => {
    const svg = d3.select(flowSvgRef.current)
    const width = svg.node() ? svg.node().clientWidth || 360 : 360
    const height = 560
    svg.attr('viewBox', `0 0 ${width} ${height}`)
    svg.selectAll('*').remove()

    // Vertical main flow layout with boundary checking
    const margin = 40 // Minimum margin from edges (reduced to bring columns closer)
    const centerX = Math.round(width / 2)
    const qaColumnX = margin + 100 // Left-aligned Q/A column (slightly closer to center)
    const rightSideX = Math.min(width - margin - 50, centerX + 60)  // External systems on right (further reduced gap)
    const rectW = 100, rectH = 50
    
    const nodePositions = new Map()
    
    // Boundary check helper
    const clampPosition = (x, y) => ({
      x: Math.max(margin + rectW/2, Math.min(width - margin - rectW/2, x)),
      y: Math.max(margin + rectH/2, Math.min(height - margin - rectH/2, y))
    })
    
    // Get all pairs
    const pairs = new Set(flowData.nodes.map(n => n.pairIndex).filter(p => p !== undefined))
    const sortedPairs = Array.from(pairs).sort((a, b) => a - b)
    
    // Calculate dynamic spacing based on number of pairs to fit in viewport
    const pairSpacing = Math.min(240, Math.max(280, (height - 160) / Math.max(1, sortedPairs.length)))
    
    sortedPairs.forEach((pairIndex, idx) => {
      const baseY = 80 + idx * pairSpacing
      
      // Main flow: User -> Context -> Assistant (left aligned column)
      const userNode = flowData.nodes.find(n => n.id === `user_${pairIndex}`)
      const ctxNode = flowData.nodes.find(n => n.id === `ctx_${pairIndex}`)
      const assistantNode = flowData.nodes.find(n => n.id === `assistant_${pairIndex}`)
      
      if (userNode) {
        const pos = clampPosition(qaColumnX, baseY)
        nodePositions.set(userNode.id, pos)
      }
      if (ctxNode) {
        const pos = clampPosition(qaColumnX, baseY + 80)
        nodePositions.set(ctxNode.id, pos)
      }
      if (assistantNode) {
        const pos = clampPosition(qaColumnX, baseY + 160)
        nodePositions.set(assistantNode.id, pos)
      }
      
      // External systems on right — anchor around context Y with fixed lane spacing to avoid overlap
      const visionNode = flowData.nodes.find(n => n.id === `vision_${pairIndex}`)
      const mcpNode = flowData.nodes.find(n => n.id === `mcp_${pairIndex}`)
      const ragNode = flowData.nodes.find(n => n.id === `rag_${pairIndex}`)

      const ctxPos = nodePositions.get(`ctx_${pairIndex}`) || clampPosition(qaColumnX, baseY + 80)
      const laneDelta = 80 // greater than rectH to avoid within-pair overlap
      if (visionNode) {
        const pos = clampPosition(rightSideX, ctxPos.y - laneDelta)
        nodePositions.set(visionNode.id, pos)
      }
      if (mcpNode) {
        const pos = clampPosition(rightSideX, ctxPos.y)
        nodePositions.set(mcpNode.id, pos)
      }
      if (ragNode) {
        const pos = clampPosition(rightSideX, ctxPos.y + laneDelta)
        nodePositions.set(ragNode.id, pos)
      }
    })

    // Position global context with boundary checking
    const globalContext = flowData.nodes.find(n => n.id === 'global_context')
    if (globalContext) {
      const globalX = Math.min(width - margin - rectW/2, rightSideX + 60)
      const globalY = Math.min(height - margin - rectH/2, Math.max(margin + rectH/2, height / 2))
      nodePositions.set('global_context', { x: globalX, y: globalY })
    }

    const computeNodeWidth = (n) => {
      const r = 8, pad = 12, spacing = r * 2.5
      const cnt = Math.max(1, Array.isArray(n.micro) ? n.micro.length : 1)
      const minWidth = pad * 2 + (2 * r) + spacing * (cnt - 1)
      
      if (n.type === 'mcp') return Math.max(60, minWidth)  // At least 60, but expand if needed
      if (n.type === 'rag') return Math.max(50, minWidth) // At least 50, but expand if needed  
      return minWidth
    }

    const nodes = flowData.nodes.map((n) => {
      const pos = nodePositions.get(n.id) || { x: centerX, y: 100 }
      const w = computeNodeWidth(n)
      return { ...n, x: pos.x, y: pos.y, w, h: rectH }
    })

    const idToNode = Object.fromEntries(nodes.map((n) => [n.id, n]))
    const maxV = Math.max(1, d3.max(flowData.links, (d) => d.value) || 1)
    const thick = d3.scaleLinear().domain([0, maxV]).range([6, 24])

    const linkPath = (s, t) => {
      // Calculate connection points on node edges
      const dx = t.x - s.x
      const dy = t.y - s.y
      const distance = Math.sqrt(dx * dx + dy * dy)
      
      if (distance === 0) return `M${s.x},${s.y} L${t.x},${t.y}`
      
      const unitX = dx / distance
      const unitY = dy / distance
      
      // Node dimensions for edge calculation
      const getNodeRadius = (node) => {
        if (node.type === 'mcp') return 25  // Triangle
        if (node.type === 'rag') return 25  // Circle
        return Math.max(node.w, node.h) / 2  // Rectangle
      }
      
      const sRadius = getNodeRadius(s)
      const tRadius = getNodeRadius(t)
      
      const x1 = s.x + unitX * sRadius
      const y1 = s.y + unitY * sRadius
      const x2 = t.x - unitX * tRadius
      const y2 = t.y - unitY * tRadius
      
      // Main flow (vertical): straight lines
      if (Math.abs(s.x - t.x) < 20) {
        return `M${x1},${y1} L${x2},${y2}`
      }
      
      // External flow: curved lines
      const mx = (x1 + x2) / 2
      const my = (y1 + y2) / 2
      return `M${x1},${y1} Q${mx},${my} ${x2},${y2}`
    }

    // Add animation styles for flowing triangles
    svg.append('defs').append('style').text(`
      @keyframes triangleFlow { 
        0% { transform: translateX(0); opacity: 0; } 
        10% { opacity: 1; } 
        90% { opacity: 1; } 
        100% { transform: translateX(100px); opacity: 0; } 
      }
      .flowingTriangle { 
        animation: triangleFlow 2s linear infinite; 
        transform-origin: center; 
      }
    `)

    // Group multi-edges and handle bidirectional links separately
    const processedPairs = new Set()
    const enhancedLinks = []
    
    flowData.links.forEach((link, linkIdx) => {
      
      // Check if this is a bidirectional pair
      const hasReverse = flowData.links.some(l => l.source === link.target && l.target === link.source)
      
      if (hasReverse) {
        // Skip if we've already processed this bidirectional pair
        const pairKey = [link.source, link.target].sort().join('<->')
        if (processedPairs.has(pairKey)) return
        processedPairs.add(pairKey)
        
        // Create two separate lines for bidirectional connection
        enhancedLinks.push({
          ...link,
          pathType: 'curved',
          bidir: false,
          bidirDirection: 'forward',
          bidirOffset: 50,
          multiCount: 1,
          multiIndex: 0
        })
        
        // Create reverse direction link
        const reverseLink = flowData.links.find(l => l.source === link.target && l.target === link.source)
        if (reverseLink) {
          enhancedLinks.push({
            ...reverseLink,
            pathType: 'curved',
            bidir: false,
            bidirDirection: 'reverse',
            bidirOffset: 50,
            multiCount: 1,
            multiIndex: 0
          })
        }
      } else {
        // Single direction link
        enhancedLinks.push({
          ...link,
          pathType: 'curved',
          bidir: false,
          bidirDirection: 'single',
          bidirOffset: 0,
          multiCount: 1,
          multiIndex: 0
        })
      }
    })

    // Helper to compute edge-to-edge positions for any two nodes
    const edgePoints = (s, t) => {
      const dx = t.x - s.x
      const dy = t.y - s.y
      const dist = Math.max(1e-6, Math.sqrt(dx * dx + dy * dy))
      const ux = dx / dist
      const uy = dy / dist
      const rectHalf = (n) => {
        if (n.type === 'mcp') return { hw: 30, hh: 26 } // triangle approx bounds
        if (n.type === 'rag') return { hw: 25, hh: 25 } // circle radius
        return { hw: (n.w || 100) / 2, hh: (n.h || 50) / 2 }
      }
      const eps = 4
      const { hw: shw, hh: shh } = rectHalf(s)
      const { hw: thw, hh: thh } = rectHalf(t)
      const tx = Math.abs(ux) < 1e-6 ? Number.POSITIVE_INFINITY : shw / Math.abs(ux)
      const ty = Math.abs(uy) < 1e-6 ? Number.POSITIVE_INFINITY : shh / Math.abs(uy)
      const ts = Math.min(tx, ty)
      const x1 = s.x + ux * ts
      const y1 = s.y + uy * ts
      const tx2 = Math.abs(ux) < 1e-6 ? Number.POSITIVE_INFINITY : thw / Math.abs(ux)
      const ty2 = Math.abs(uy) < 1e-6 ? Number.POSITIVE_INFINITY : thh / Math.abs(uy)
      const tt = Math.min(tx2, ty2)
      let x2 = t.x - ux * tt
      let y2 = t.y - uy * tt
      if (Math.hypot(x2 - x1, y2 - y1) < eps) {
        x2 = x1 + ux * eps
        y2 = y1 + uy * eps
      }
      return { x1, y1, x2, y2, ux, uy, dist }
    }

    // Draw links - all use curved paths now
    const linksLayer = svg.append('g')
    const links = linksLayer.selectAll('path')
      .data(enhancedLinks)
      .enter().append('path')
      .attr('d', (d) => {
        // Curved path with bidirectional offset support
        const s = idToNode[d.source]
        const t = idToNode[d.target]
        const { x1, y1, x2, y2, ux, uy, dist } = edgePoints(s, t)
        const safeDist = Math.max(1e-3, dist)
        
        // Use bidirOffset for bidirectional links, otherwise use smaller default offset
        const offsetMag = d.bidirOffset !== undefined ? Math.abs(d.bidirOffset) : 10
        
        // perpendicular unit
        const px = -uy
        const py = ux
        
        // Use bidirOffset sign for bidirectional links
        const sign = d.bidirOffset !== undefined ? Math.sign(d.bidirOffset) : 1
        const seq = (k) => (k % 2 === 0 ? -(k/2 + 1) : (k+1)/2)
        const multiOff = (d.multiCount > 1) ? seq(d.multiIndex) * 6 : 0
        
        const mx = (x1 + x2) / 2 + px * (offsetMag * sign + multiOff)
        const my = (y1 + y2) / 2 + py * (offsetMag * sign + multiOff)
        
        // if nodes are extremely close, keep a tiny curvature to maintain direction
        if (safeDist < 6) {
          return `M${x1},${y1} Q${x1 + px * 6 * sign},${y1 + py * 6 * sign} ${x2},${y2}`
        }
        return `M${x1},${y1} Q${mx},${my} ${x2},${y2}`
      })
      .attr('fill', 'none')
      .attr('stroke', (d) => d.type === 'conversation_flow' ? '#52c41a' : (d.type === 'context_in' || d.type === 'context_out') ? '#91caff' : '#bfbfbf')
      .attr('stroke-opacity', 0.85)
      .attr('stroke-width', 2) // Fixed width, no Sankey flow thickness
      .attr('stroke-linecap', 'round')
      .attr('stroke-linejoin', 'round')
      .attr('id', (d, i) => `path_${i}_${Date.now()}`)

    // Helper: link color
    const linkColor = (d) => (d.type === 'conversation_flow' ? '#52c41a' : (d.type === 'context_in' || d.type === 'context_out') ? '#91caff' : '#bfbfbf')

    // Add flowing dot markers (dense, direction-true)
    const markersLayer = svg.append('g').attr('class', 'flowMarkers')

    const renderFlowingMarkers = () => {
      markersLayer.selectAll('*').remove()

      // Render flowing dots for all flows (excluding conversation continuity if needed)
      links.each(function(d) {
        if (d.type === 'conversation_flow') return
        const path = this
        const len = Math.max(1, path.getTotalLength())
        const numDots = Math.max(6, Math.floor(len / 30))
        const r = 3
        const dur = Math.max(1.2, len / 120) // keep constant speed across lengths
        for (let i = 0; i < numDots; i++) {
          const grp = markersLayer.append('g')
          const dot = grp.append('circle')
            .attr('r', r)
            .attr('fill', linkColor(d))
            .attr('fill-opacity', 0.95)
            .attr('stroke', 'none')
          const am = grp.append('animateMotion')
            .attr('dur', `${dur}s`)
            .attr('repeatCount', 'indefinite')
            .attr('begin', `${(i * dur) / numDots}s`)
          am.append('mpath').attr('href', `#${d3.select(path).attr('id')}`)
          // subtle pulsation for visibility
          dot.append('animate')
            .attr('attributeName', 'r')
            .attr('values', `${r};${r * 0.8};${r}`)
            .attr('dur', `${dur}s`)
            .attr('repeatCount', 'indefinite')
            .attr('begin', `${(i * dur) / numDots}s`)
        }
      })
    }

    renderFlowingMarkers()

    // Re-render markers on window resize/zoom changes
    const ro = new ResizeObserver(() => renderFlowingMarkers())
    if (flowSvgRef.current) ro.observe(flowSvgRef.current)
    // Also re-render after a small timeout to ensure paths are laid out
    setTimeout(renderFlowingMarkers, 0)

    // Node colors based on type
    const getNodeBorderColor = (type) => {
      switch (type) {
        case 'user': return '#52c41a' // Green for user input
        case 'assistant': return '#1677ff' // Blue for assistant output
        case 'mcp': return '#fa8c16'
        case 'rag': return '#2f54eb'
        case 'vision': return '#13c2c2'
        case 'context': return '#91caff'
        default: return '#d9d9d9'
      }
    }

    // Draw nodes with drag behavior
    const nodeGroups = svg.append('g').selectAll('g')
      .data(nodes)
      .enter().append('g')
      .attr('transform', (d) => `translate(${d.x},${d.y})`)
      .attr('class', styles.nodeGroup)
      .call(d3.drag()
        .on('start', function(event, d) {
          d3.select(this).raise()
        })
        .on('drag', function(event, d) {
          d.x = event.x
          d.y = event.y
          d3.select(this).attr('transform', `translate(${d.x},${d.y})`)
          
          // Update links with new path calculations - all use curved paths now
          links.attr('d', (linkData) => {
            const s = idToNode[linkData.source]
            const t = idToNode[linkData.target]
            const pts = edgePoints(s, t)
            const dx = pts.x2 - pts.x1
            const dy = pts.y2 - pts.y1
            const len = Math.max(1e-3, Math.hypot(dx, dy))
            const px = -dy / len
            const py = dx / len
            
            // Use bidirOffset for bidirectional links, otherwise use smaller default offset
            const offsetMag = linkData.bidirOffset !== undefined ? Math.abs(linkData.bidirOffset) : 10
            const sign = linkData.bidirOffset !== undefined ? Math.sign(linkData.bidirOffset) : 1
            const seq = (k) => (k % 2 === 0 ? -(k/2 + 1) : (k+1)/2)
            const multiOff = (linkData.multiCount > 1) ? seq(linkData.multiIndex) * 6 : 0
            
            const mx = (pts.x1 + pts.x2) / 2 + px * (offsetMag * sign + multiOff)
            const my = (pts.y1 + pts.y2) / 2 + py * (offsetMag * sign + multiOff)
            return `M${pts.x1},${pts.y1} Q${mx},${my} ${pts.x2},${pts.y2}`
          })
          
          // Re-render flowing markers to follow the updated paths
          renderFlowingMarkers()
        })
        .on('end', function() {
          renderFlowingMarkers()
        })
      )

    // Draw node shapes based on type
    nodeGroups.each(function(d) {
      const group = d3.select(this)
      const borderColor = getNodeBorderColor(d.type)
      
      if (d.type === 'mcp') {
        // Triangle for MCP
        const triangleSize = 25
        group.append('polygon')
          .attr('points', `0,${-triangleSize} ${triangleSize * 0.866},${triangleSize/2} ${-triangleSize * 0.866},${triangleSize/2}`)
          .attr('fill', '#fafafa')
          .attr('stroke', borderColor)
          .attr('stroke-width', 2)
        
        // Add M label
        group.append('text')
          .attr('x', -18)
          .attr('y', -18)
          .attr('font-size', '12px')
          .attr('font-weight', 'bold')
          .attr('fill', '#333')
          .text('M')
          
      } else if (d.type === 'rag') {
        // Hollow circle for RAG  
        const radius = 25
        group.append('circle')
          .attr('r', radius)
          .attr('fill', 'none')  // Hollow - no fill
          .attr('stroke', borderColor)
          .attr('stroke-width', 2)
        
        // Add R label
        group.append('text')
          .attr('x', 0)
          .attr('y', 4)
          .attr('font-size', '12px')
          .attr('font-weight', 'bold')
          .attr('fill', '#333')
          .attr('text-anchor', 'middle')
          .text('R')
          
      } else {
        // Rectangle for others
        group.append('rect')
          .attr('x', -d.w/2)
          .attr('y', -d.h/2)
          .attr('width', d.w)
          .attr('height', d.h)
          .attr('rx', 8)
          .attr('ry', 8)
          .attr('fill', '#fafafa')
          .attr('stroke', borderColor)
          .attr('stroke-width', 2)
        
        // Add appropriate label
        const label = d.type === 'user' ? 'U' : d.type === 'assistant' ? 'A' : 'C'
        group.append('text')
          .attr('x', -d.w/2 + 8)
          .attr('y', -d.h/2 + 15)
          .attr('font-size', '12px')
          .attr('font-weight', 'bold')
          .attr('fill', '#333')
          .text(label)
        
        // Add connection badges for global context node
        if (d.id === 'global_context') {
          // Input badge (top-left)
          if (flowData.ctx.contextInCount > 0) {
            const badgeGroup = group.append('g')
              .attr('transform', `translate(${-d.w/2 + 10}, ${-d.h/2 - 5})`)
            
            badgeGroup.append('circle')
              .attr('r', 8)
              .attr('fill', '#91caff')
              .attr('stroke', '#fff')
              .attr('stroke-width', 1)
            
            badgeGroup.append('text')
              .attr('x', 0)
              .attr('y', 3)
              .attr('font-size', '10px')
              .attr('font-weight', 'bold')
              .attr('fill', '#fff')
              .attr('text-anchor', 'middle')
              .text(flowData.ctx.contextInCount)
          }
          
          // Output badge (top-right)  
          if (flowData.ctx.contextOutCount > 0) {
            const badgeGroup = group.append('g')
              .attr('transform', `translate(${d.w/2 - 10}, ${-d.h/2 - 5})`)
            
            badgeGroup.append('circle')
              .attr('r', 8)
              .attr('fill', '#52c41a')
              .attr('stroke', '#fff')
              .attr('stroke-width', 1)
            
            badgeGroup.append('text')
              .attr('x', 0)
              .attr('y', 3)
              .attr('font-size', '10px')
              .attr('font-weight', 'bold')
              .attr('fill', '#fff')
              .attr('text-anchor', 'middle')
              .text(flowData.ctx.contextOutCount)
          }
        }
      }
    })

    const microColor = (t) => {
      switch (t) {
        case 'text': return '#1677ff'
        case 'image': return '#722ed1'
        case 'file': return '#595959'
        case 'ocr': return '#13c2c2'
        case 'detector': return '#52c41a'
        case 'search': return '#fa8c16'
        case 'calculator': return '#fa541c'
        case 'fileio': return '#8c8c8c'
        case 'retriever': return '#2f54eb'
        case 'ranker': return '#1890ff'
        case 'vector': return '#5cdbd3'
        case 'tokens': return '#91caff'
        default: return '#bfbfbf'
      }
    }

    // Larger micro dots inside nodes
    nodeGroups.each(function (d) {
      const group = d3.select(this)
      const r = 8 // Larger radius
      let bounds, pad = 12
      
      if (d.type === 'mcp') {
        // Triangle bounds
        bounds = { w: 40, h: 40 }
      } else if (d.type === 'rag') {
        // Circle bounds
        bounds = { w: 50, h: 50 }
      } else {
        // Rectangle bounds
        bounds = { w: d.w, h: d.h }
      }
      
      const base = Array.isArray(d.micro) ? d.micro : []
      const need = Math.max(1, base.length || 1)  // Always show all micro nodes
      
      const spacing = r * 2.5
      // Calculate optimal layout for the actual number of nodes needed
      const cols = Math.max(1, Math.ceil(Math.sqrt(need)))
      const rows = Math.max(1, Math.ceil(need / cols))
      
      // Center the dots based on actual layout
      const totalWidth = (cols - 1) * spacing
      const totalHeight = (rows - 1) * spacing
      const startX = -totalWidth / 2
      const startY = -totalHeight / 2
      
      for (let i = 0; i < need; i++) {
        const col = i % cols
        const row = Math.floor(i / cols)
        const cx = startX + col * spacing
        const cy = startY + row * spacing
        const type = base[i % base.length] || 'tokens'
        
        group.append('circle')
          .attr('class', styles.microDot)
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', r)
          .attr('fill', microColor(type))
          .attr('stroke', '#fff')
          .attr('stroke-width', 1)
      }
    })
  }, [flowData])

  return (
    <Splitter style={{ height: '100vh', width: '100%' }}>
      {/* Left Panel: conversation + input */}
      <Splitter.Panel defaultSize="60%" min="30%" max="80%">
        <div className={styles.leftPane} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Conversation messages */}
          <div style={{ flex: 1, overflow: 'auto' }} ref={containerRef}>
          <svg ref={overlayRef} className={styles.overlaySvg} />
          <div className={styles.msgList}>
            {visibleMessages.map((m) => {
              const topic = (conv.topics || []).find((t) => t.message_ids.includes(m.id))
              const dotColor = topic?.color || '#d9d9d9'
              const isExpanded = expanded.has(m.id)
              const hasText = !!(m?.content?.text)
              const imgList = Array.isArray(m?.content?.images) ? m.content.images : []
              const files = Array.isArray(m?.content?.files) ? m.content.files : []
              const rowClass = isUser(m) ? styles.rowRight : styles.rowLeft
              const bubbleCn = bubbleClass(m)
                const borderColor = isUser(m) ? '#52c41a' : '#1677ff' // Green for user, blue for assistant
              return (
                <div key={m.id} ref={(el) => el && msgRefs.current.set(m.id, el)} className={`${styles.msgRow} ${rowClass}`}>
                  <div className={styles.spineSlot} aria-hidden />
                  <div ref={(el) => el && bubbleRefs.current.set(m.id, el)} className={`${styles.bubbleWrap} ${hoveredMsgId === m.id ? styles.highlight : ''}`}>
                    <div className={styles.metaBar}>
                      <span className={styles.topicDot} style={{ background: dotColor }} />
                      <span>{m.type === 'user' ? 'User' : 'Assistant'}</span>
                      <span>·</span>
                      <span>{new Date(m.timestamp).toLocaleTimeString()}</span>
                      {Array.isArray(m.tools_used) && m.tools_used.map((t) => (
                        <span key={t} className={styles.toolBadge}>{t}</span>
                      ))}
                    </div>
                      <div className={bubbleCn} style={{ borderColor }} onClick={() => toggleExpand(m.id)} onMouseEnter={(e) => onBubbleEnter(e, m)} onMouseLeave={onBubbleLeave} role="button" aria-label="message bubble" tabIndex={0}>
                      {hasText ? (
                        <div className={`${styles.contentText} ${isExpanded ? '' : styles.contentTextCollapsed}`}>{m.content.text}</div>
                      ) : null}
                      {imgList.length ? (
                        <div className={styles.mediaGrid}>
                          {imgList.map((src, i) => (
                            <img key={i} src={src} alt="preview" className={styles.thumb} />
                          ))}
                        </div>
                      ) : null}
                      {files.length ? (
                        <div style={{ marginTop: 8, display: 'grid', gap: 4 }}>
                          {files.map((f) => (
                            <div key={f} className={styles.fileItem}>File: {f}</div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
          <div ref={tooltipRef} className={`${styles.tooltip} ${tooltip.show ? styles.tooltipShow : ''}`} style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.text}</div>
        </div>
          
          {/* Chat input interface */}
          <div style={{ padding: '16px', borderTop: '1px solid #f0f0f0', background: '#fafafa' }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
              <Input.TextArea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type your message here..."
                autoSize={{ minRows: 1, maxRows: 4 }}
                style={{ flex: 1 }}
                onPressEnter={(e) => {
                  if (!e.shiftKey) {
                    e.preventDefault()
                    handleSendMessage()
                  }
                }}
              />
              <div style={{ display: 'flex', gap: '4px' }}>
                <Upload
                  accept="image/*"
                  showUploadList={false}
                  beforeUpload={() => false}
                  onChange={handleFileUpload}
                >
                  <Button icon={<UploadOutlined />}/>
                </Upload>
                <Button 
                  type="primary" 
                  icon={<SendOutlined />} 
                  onClick={handleSendMessage}
                  disabled={!inputText.trim() && uploadedFiles.length === 0}
                />
              </div>
            </div>
            {uploadedFiles.length > 0 && (
              <div style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                {uploadedFiles.length} file(s) selected
              </div>
            )}
          </div>
        </div>
      </Splitter.Panel>

      {/* Right Panel: flow visualization */}
      <Splitter.Panel>
        <div className={styles.rightPane} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Search bar moved to top right */}
          <div className={styles.searchBar}>
            <Input 
              prefix={<SearchOutlined />}
              placeholder="Search text or tool..." 
              value={search} 
              onChange={(e) => setSearch(e.target.value)} 
              onPressEnter={onSearchKey}
              allowClear
            />
      </div>

          <div className={styles.flowRoot} style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
          <div className={styles.legend}>
            <div className={styles.legendSection}>
              <div className={styles.legendTitle}>Input Types</div>
              <span className={styles.legendSwatch} style={{ background: '#1677ff' }}></span><span>Text</span>
              <span className={styles.legendSwatch} style={{ background: '#722ed1' }}></span><span>Image</span>
              <span className={styles.legendSwatch} style={{ background: '#595959' }}></span><span>File</span>
            </div>
            <div className={styles.legendSection}>
              <div className={styles.legendTitle}>Tools</div>
              <span className={styles.legendSwatch} style={{ background: '#13c2c2' }}></span><span>OCR</span>
              <span className={styles.legendSwatch} style={{ background: '#52c41a' }}></span><span>Detector</span>
              <span className={styles.legendSwatch} style={{ background: '#fa541c' }}></span><span>Calculator</span>
            </div>
            <div className={styles.legendSection}>
              <div className={styles.legendTitle}>Systems</div>
              <span className={styles.legendSwatch} style={{ background: '#fa8c16' }}></span><span>MCP</span>
              <span className={styles.legendSwatch} style={{ background: '#2f54eb' }}></span><span>RAG</span>
              <span className={styles.legendSwatch} style={{ background: '#91caff' }}></span><span>Context Window</span>
            </div>
          </div>
            <div style={{ flex: 1 }}>
          <svg ref={flowSvgRef} className={styles.flowSvg} />
            </div>
            <div style={{ marginTop: 'auto' }}>
          <div className={styles.panelTitle}>Context Window</div>
              <div 
                className={styles.kvRow} 
                style={{ cursor: 'pointer' }}
                onClick={() => setShowContextDetails(!showContextDetails)}
              >
                <span>Capacity</span><span>{flowData.ctx.capacity.toLocaleString()} tokens</span>
              </div>
              <div 
                className={styles.kvRow}
                style={{ cursor: 'pointer' }}
                onClick={() => setShowContextDetails(!showContextDetails)}
              >
                <span>Used</span><span>{flowData.ctx.used.toLocaleString()} tokens</span>
              </div>
          <div className={styles.meter}>
            <div className={styles.meterFill} style={{ width: `${Math.min(100, (flowData.ctx.used / flowData.ctx.capacity) * 100)}%` }}></div>
            <div className={styles.meterEvict} style={{ width: `${Math.min(100, (flowData.ctx.evicted / flowData.ctx.capacity) * 100)}%` }} title="evicted"></div>
          </div>
              {showContextDetails && (
                <div style={{ marginTop: '16px', padding: '12px', background: '#f9f9f9', borderRadius: '8px', fontSize: '12px' }}>
                  <div>Evicted: {flowData.ctx.evicted.toLocaleString()} tokens</div>
                  <div>Available: {(flowData.ctx.capacity - flowData.ctx.used).toLocaleString()} tokens</div>
                  <div>Usage: {((flowData.ctx.used / flowData.ctx.capacity) * 100).toFixed(1)}%</div>
                </div>
              )}
        </div>
      </div>
    </div>
      </Splitter.Panel>
    </Splitter>
  )
}


