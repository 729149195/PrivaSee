import { create } from 'zustand'
import { buildSystemPrompt } from './templates/infons.js'
import { loadUserSessions, saveUserSessions } from './users/historyStorage'

// 说明：
// 1) 本 store 管理 ChatGPT 风格的多会话、消息流与流式生成状态；
// 2) 采用 OpenAI Chat Completions 协议与本地 Ollama(OpenAI 兼容)接口对接；
// 3) 所有 UI 文本在代码中使用英文，注释使用中文；
// 4) 每次对话上下文从当前会话的全部消息构建，以实现连续对话记忆。

// 生成唯一 ID：优先使用浏览器 crypto.randomUUID，降级到时间戳
const generateId = () => {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  } catch (_) { }
  return 'id_' + Date.now() + '_' + Math.random().toString(16).slice(2)
}

// 安全 JSON 解析
function tryParseJSON(text) {
  try {
    return { ok: true, value: JSON.parse(text) }
  } catch (_) {
    return { ok: false, value: null }
  }
}

// 从文本中提取首个完整 JSON 对象：忽略 JSON 内部字符串中的大括号
function extractFirstJSONObject(text) {
  if (typeof text !== 'string' || !text) return null
  const start = text.indexOf('{')
  if (start < 0) return null
  let depth = 0
  let inString = false
  let escape = false
  for (let i = start; i < text.length; i++) {
    const ch = text[i]
    if (inString) {
      if (escape) { escape = false; continue }
      if (ch === '\\') { escape = true; continue }
      if (ch === '"') { inString = false; continue }
    } else {
      if (ch === '"') { inString = true; continue }
      if (ch === '{') depth++
      if (ch === '}') {
        depth--
        if (depth === 0) {
          return text.slice(start, i + 1)
        }
      }
    }
  }
  return null
}

// 简单稳定哈希：用于文本/图片去重
function computeHashId(input) {
  const s = String(input || '')
  let h = 5381
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) + h) + s.charCodeAt(i)
    h = h >>> 0
  }
  return 'h' + h.toString(16)
}

// 规范化模型输出：确保符合 OUTPUT_FORMAT，并填充 record_time
function normalizeInfonOutput(obj, { recordTimeISO, defaultModality, sessionId, messageRound, infonIndex, infonType }) {
  const out = (obj && typeof obj === 'object') ? obj : {}
  const now = recordTimeISO || new Date().toISOString()
  out.run_metadata = out.run_metadata && typeof out.run_metadata === 'object' ? out.run_metadata : {}
  if (!out.run_metadata.record_time) out.run_metadata.record_time = now
  if (!out.run_metadata.source_id) out.run_metadata.source_id = 'unknown'
  if (!out.run_metadata.generator) out.run_metadata.generator = 'infon_extractor'
  out.situations = Array.isArray(out.situations) ? out.situations : []
  out.entities = Array.isArray(out.entities) ? out.entities : []
  out.infons = Array.isArray(out.infons) ? out.infons : []
  out.quality_report = out.quality_report && typeof out.quality_report === 'object' ? out.quality_report : { stats: {} }
  // 填充每个 situation/infons 的 record_time 与 modality 缺省
  out.situations = out.situations.map((s) => {
    const t = { ...(typeof s === 'object' ? s : {}) }
    if (!t.record_time) t.record_time = now
    if (!t.modality && defaultModality) t.modality = defaultModality
    return t
  })
  out.infons = out.infons.map((i, index) => {
    const t = { ...(typeof i === 'object' ? i : {}) }
    if (!t.record_time) t.record_time = now
    // 生成基于对话轮次和信息元次序的iid
    if (!t.iid && infonType) {
      const typePrefix = infonType.toLowerCase().slice(0, 3) // 取前三个字母作为前缀
      const round = messageRound || 1
      const idx = (infonIndex || 0) + index + 1
      t.iid = `${typePrefix}:r${round}_${idx}`
    }
    return t
  })
  return out
}

function buildInfonSystemPrompt(modalities, nowISO, options = {}) {
  const { currentRound = 1, existingInfons = [] } = options;
  return buildSystemPrompt({
    modalities,
    includeExamples: false,
    currentRound,
    existingInfons,
    extraInstructions: `System time (ISO8601) = ${nowISO}. Set run_metadata.record_time to this value. For each situation and infon, if record_time is missing, set it to this value. Only set occur_time when it is explicitly expressed; otherwise omit.`
  })
}

// 增量更新逻辑（中文注释）：检测和处理重复/冲突的信息元
// 策略调整：
// 1. 每轮的信息元都保留（因为它们有独立的上下文和 iid）
// 2. 只标记明确的语义冲突（如同一主体的不同属性值）
// 3. 返回标记了冲突关系的新信息元，由上层决定如何展示
function deduplicateAndMergeInfons(newInfons, existingInfons) {
  if (!Array.isArray(newInfons) || newInfons.length === 0) return newInfons;
  if (!Array.isArray(existingInfons) || existingInfons.length === 0) return newInfons;
  
  const result = [];
  const conflictInfons = []; // 记录被新信息元替换的旧信息元
  
  // 合并新旧信息元列表用于查找引用（中文注释）
  const allInfonsForLookup = [...existingInfons, ...newInfons];
  
  // 处理每个新信息元（中文注释）
  newInfons.forEach(newInfon => {
    const conflicts = findConflictingInfons(newInfon, existingInfons, allInfonsForLookup);
    
    if (conflicts.length > 0) {
      // 发现冲突：标记此信息元替换了哪些旧信息元（中文注释）
      result.push({ 
        ...newInfon, 
        _supersedes: conflicts.map(c => c.iid) // 记录此信息元取代了哪些旧的
      });
      conflictInfons.push(...conflicts);
    } else {
      // 无冲突：直接添加（中文注释）
      result.push(newInfon);
    }
  });
  
  return result;
}

// 查找与新信息元冲突的已有信息元（中文注释）
// 冲突定义：表达同一主体的不同属性值，需要用新值替换旧值
function findConflictingInfons(newInfon, existingInfons, allInfonsForLookup) {
  if (!newInfon || !Array.isArray(existingInfons)) return [];
  
  const type = String(newInfon.infon_type || '').toUpperCase();
  const conflicts = [];
  
  if (type === 'DESC') {
    // DESC冲突检测：查找同一主体实体的不同属性值（中文注释）
    const newEntity = String(newInfon.entity || '').trim().toLowerCase();
    const newAttr = String(newInfon.attribute || '').trim().toLowerCase();
    
    // 特殊处理：姓名类属性冲突（中文注释）
    const isNameEntity = ['姓名', '名字', 'name', '名称'].includes(newEntity);
    
    existingInfons.forEach(existing => {
      if (String(existing.infon_type || '').toUpperCase() !== 'DESC') return;
      
      const existEntity = String(existing.entity || '').trim().toLowerCase();
      const existAttr = String(existing.attribute || '').trim().toLowerCase();
      
      // 同一实体类别，但属性值不同（中文注释）
      if (newEntity === existEntity && newAttr !== existAttr) {
        // 对于姓名类属性，只有当两者都是姓名时才判定为冲突
        if (isNameEntity) {
          conflicts.push(existing);
        }
      }
    });
  } else if (type === 'REL') {
    // REL冲突检测：查找同一关系名称连接同一第一参数的关系（中文注释）
    const newRelName = String(newInfon.relation_name || '').trim().toLowerCase();
    const newArgRefs = Array.isArray(newInfon.arg_refs) ? newInfon.arg_refs : [];
    
    // 特殊处理：名称关系冲突（中文注释）
    const isNameRelation = ['名字', '姓名', 'name', '名称', '名称关系', '名字关系'].includes(newRelName);
    
    if (isNameRelation && newArgRefs.length >= 2) {
      const newSubject = newArgRefs[0]; // 第一个参数是主体（如"我"）
      
      existingInfons.forEach(existing => {
        if (String(existing.infon_type || '').toUpperCase() !== 'REL') return;
        
        const existRelName = String(existing.relation_name || '').trim().toLowerCase();
        const existArgRefs = Array.isArray(existing.arg_refs) ? existing.arg_refs : [];
        
        // 同一类型关系，且主体相同（中文注释）
        if (isNameRelation && existArgRefs.length >= 2) {
          const existSubject = existArgRefs[0];
          
          // 检查是否指向同一主体（通过查找主体信息元的实际内容）（中文注释）
          // 使用合并后的列表查找，以支持跨新旧信息元的引用
          if (isSameSubject(newSubject, existSubject, allInfonsForLookup || existingInfons)) {
            conflicts.push(existing);
          }
        }
      });
    }
  }
  
  return conflicts;
}

// 检查两个信息元引用是否指向同一主体（中文注释）
function isSameSubject(iid1, iid2, allInfons) {
  if (iid1 === iid2) return true;
  
  // 查找两个iid对应的信息元内容
  const infon1 = allInfons.find(i => i.iid === iid1);
  const infon2 = allInfons.find(i => i.iid === iid2);
  
  if (!infon1 || !infon2) return false;
  
  // 如果都是DESC类型且实体相同，认为是同一主体
  if (String(infon1.infon_type || '').toUpperCase() === 'DESC' &&
      String(infon2.infon_type || '').toUpperCase() === 'DESC') {
    const entity1 = String(infon1.entity || '').trim().toLowerCase();
    const entity2 = String(infon2.entity || '').trim().toLowerCase();
    const attr1 = String(infon1.attribute || '').trim().toLowerCase();
    const attr2 = String(infon2.attribute || '').trim().toLowerCase();
    
    // 同一实体和属性，或都是"我"、"用户"等主体代词
    const subjectPronouns = ['我', 'i', 'me', '用户', 'user'];
    return (entity1 === entity2 && attr1 === attr2) || 
           (subjectPronouns.includes(attr1) && subjectPronouns.includes(attr2));
  }
  
  return false;
}

// 在流中增量解析 infons 数组，逐个对象产出
function incrementalExtractInfons(streamText, parser) {
  const state = parser || { 
    foundArray: false, 
    arrayStart: -1, 
    scanPos: 0, 
    inString: false, 
    escape: false, 
    objStart: -1, 
    braceDepth: 0, 
    closed: false, 
    objectStates: new Map(), // Map<objIndex, {lastParsedHash, data}>
    currentObjIndex: 0
  }
  const yielded = []
  const text = String(streamText || '')

  // 若尚未定位到 infons 数组，先查找
  if (!state.foundArray) {
    const m = /"infons"\s*:\s*\[/.exec(text)
    if (!m) {
      state.scanPos = text.length
      return { state, yielded }
    }
    state.foundArray = true
    state.arrayStart = m.index + m[0].lastIndexOf('[')
    state.scanPos = state.arrayStart + 1
  }

  let i = state.scanPos
  let inString = state.inString
  let escape = state.escape
  let objStart = state.objStart
  let braceDepth = state.braceDepth

  // 当数组已经关闭则不再扫描
  if (state.closed) return { state, yielded }

  for (; i < text.length; i++) {
    const ch = text[i]
    if (inString) {
      if (escape) { escape = false; continue }
      if (ch === '\\') { escape = true; continue }
      if (ch === '"') { inString = false; continue }
      continue
    }
    if (ch === '"') { inString = true; continue }
    if (ch === '{') {
      if (objStart < 0) { 
        objStart = i
        braceDepth = 1
        // 新对象开始
        if (!state.objectStates.has(state.currentObjIndex)) {
          state.objectStates.set(state.currentObjIndex, { lastParsedHash: null, data: {} })
        }
      } else { 
        braceDepth++ 
      }
      continue
    }
    if (ch === '}') {
      if (objStart >= 0) {
        braceDepth--
        if (braceDepth === 0) {
          // 对象完整闭合
          let objText = text.slice(objStart, i + 1)
          objText = objText.trim()
          
          if (!objText.endsWith('}')) {
            const lastBrace = objText.lastIndexOf('}')
            if (lastBrace >= 0) {
              objText = objText.slice(0, lastBrace + 1)
            }
          }
          
          const hash = computeHashId(objText)
          if (!state.objectStates.get(state.currentObjIndex)?.lastParsedHash || 
              state.objectStates.get(state.currentObjIndex).lastParsedHash !== hash) {
            const { ok, value } = tryParseJSON(objText)
            if (ok) {
              yielded.push({ ...value, _objIndex: state.currentObjIndex, _isComplete: true })
              const objState = state.objectStates.get(state.currentObjIndex)
              if (objState) {
                objState.data = value
                objState.lastParsedHash = hash
              }
            } else {
              // 解析失败时使用部分数据
              const objState = state.objectStates.get(state.currentObjIndex)
              if (objState && Object.keys(objState.data).length > 0) {
                yielded.push({ ...objState.data, _objIndex: state.currentObjIndex, _isComplete: true })
                objState.lastParsedHash = hash
              }
            }
          }
          objStart = -1
          state.currentObjIndex++
        }
      }
      continue
    }
    if (ch === ']') {
      // 数组关闭（仅当当前不在对象中）
      if (objStart < 0) { state.closed = true; i++ ; break }
    }
    
    // 尝试部分解析（每隔一定字符数或遇到特定标记时）
    if (objStart >= 0 && braceDepth > 0) {
      const objText = text.slice(objStart, i + 1)
      // 当累积了足够多的内容时，尝试部分解析
      if ((ch === ',' || ch === '\n') && (i - objStart) > 20) {
        const objState = state.objectStates.get(state.currentObjIndex)
        const currentHash = computeHashId(objText)
        
        // 只有当内容有变化时才解析
        if (objState && objState.lastParsedHash !== currentHash) {
          const partialData = parsePartialInfon(objText)
          if (partialData && Object.keys(partialData).length > 0) {
            // 检查是否有新字段
            const hasNewData = Object.keys(partialData).some(
              key => partialData[key] !== objState.data[key]
            )
            
            if (hasNewData) {
              objState.data = { ...objState.data, ...partialData }
              objState.lastParsedHash = currentHash
              yielded.push({ ...objState.data, _objIndex: state.currentObjIndex, _isComplete: false })
            }
          }
        }
      }
    }
  }

  state.inString = inString
  state.escape = escape
  state.objStart = objStart
  state.braceDepth = braceDepth
  state.scanPos = i
  return { state, yielded }
}

// 解析部分infon对象
function parsePartialInfon(objText) {
  const result = {}
  
  // 关键字段优先提取
  const criticalFields = ['iid', 'infon_type', 'entity', 'attribute', 'temporal', 'spatial']
  const otherFields = ['data_type', 'relation_name', 'arity', 'arg_refs', 'description', 'confidence', 'bbox']
  
  for (const field of [...criticalFields, ...otherFields]) {
    const value = extractInfonFieldValue(objText, field)
    if (value !== null) {
      result[field] = value
    }
  }
  
  return result
}

// 从部分JSON文本中提取字段值
function extractInfonFieldValue(text, fieldName) {
  const patterns = [
    // 字符串值
    new RegExp(`"${fieldName}"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"`, 's'),
    // 数字值
    new RegExp(`"${fieldName}"\\s*:\\s*(\\d+\\.?\\d*)`, 's'),
    // 布尔值
    new RegExp(`"${fieldName}"\\s*:\\s*(true|false)`, 's'),
    // 数组值（简单处理）
    new RegExp(`"${fieldName}"\\s*:\\s*(\\[[^\\]]*\\])`, 's'),
  ]
  
  for (const pattern of patterns) {
    const match = text.match(pattern)
    if (match) {
      try {
        // 字符串
        if (pattern.source.includes('"([^"]*')) {
          return match[1].replace(/\\"/g, '"').replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
        }
        // 数字、布尔、数组
        return JSON.parse(match[1])
      } catch (err) {
        continue
      }
    }
  }
  
  return null
}

// 新建会话：创建一个空消息会话，标题默认 "New chat"
const createEmptySession = () => ({
  id: generateId(),
  title: 'New chat',
  createdAt: Date.now(),
  updatedAt: Date.now(),
  messages: [], // {id, role: 'user'|'assistant'|'system', content, createdAt, streaming?, error?}
})

// 解析 OpenAI SSE 流：将 response.body 按行解析 data: 片段
async function streamOpenAIResponse(reader, onDelta) {
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop() || ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      if (trimmed.startsWith('data:')) {
        const payload = trimmed.slice('data:'.length).trim()
        if (payload === '[DONE]') return
        try {
          const json = JSON.parse(payload)
          // OpenAI Chat 模式：choices[0].delta.content 为增量
          const choice = json?.choices?.[0]
          const contentDelta = choice?.delta?.content ?? ''
          // 兼容多种字段名：reasoning_content / reasoning / thoughts / inner_thoughts
          const reasoningDelta = (
            choice?.delta?.reasoning_content ??
            choice?.delta?.reasoning ??
            choice?.delta?.thoughts ??
            choice?.delta?.inner_thoughts ??
            ''
          )
          const finish = choice?.finish_reason || null
          if (contentDelta || reasoningDelta) onDelta({ content: contentDelta, reasoning: reasoningDelta, finish: null })
          if (finish) onDelta({ content: '', reasoning: '', finish })
        } catch (_) {
          // 忽略不可解析的行
        }
      }
    }
  }
}

// 解析 Ollama /api/chat 流：逐行解析 JSON，并处理“全量快照”或“增量 token”两种格式
async function streamOllamaChatResponse(reader, onDelta) {
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let accumulated = '' // 用于处理返回全量快照时的去重
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split(/\r?\n/)
    buffer = lines.pop() || ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        const json = JSON.parse(trimmed)
        // chat 流常见字段：message.content；fallback 到 generate 的 response
        const nextFull = (
          (typeof json?.message?.content === 'string' ? json.message.content : '') ||
          (typeof json?.response === 'string' ? json.response : '')
        )
        const finish = json?.done ? 'stop' : null
        if (nextFull) {
          let delta = nextFull
          if (nextFull.startsWith(accumulated)) {
            delta = nextFull.slice(accumulated.length)
          }
          accumulated = nextFull
          if (delta) onDelta({ content: delta, reasoning: '', finish: null })
        }
        if (finish) onDelta({ content: '', reasoning: '', finish })
      } catch (_) {
        // 忽略不可解析的行
      }
    }
  }
}

export const useStore = create((set, get) => ({
  // 基础配置：指向本地 Ollama OpenAI 兼容接口
  baseUrl: '/v1',
  model: 'gemma3:12b',
  models: [], // 可选模型列表
  customModels: [], // 通过 API key 添加的自定义模型
  customProviders: {}, // { [modelId]: { baseUrl, apiKey } }

  // 用户状态标识（中文注释）：用于判断是否启用历史数据持久化
  currentUserId: null,
  
  // 设置当前用户（登录时调用）
  setCurrentUser: (userId) => {
    set({ currentUserId: userId })
    // 登录时加载用户的历史数据
    if (userId) {
      get()._loadUserHistory(userId)
    }
  },
  
  // 清除当前用户（退出登录时调用）
  clearCurrentUser: () => {
    const { currentUserId } = get()
    // 退出前保存当前数据
    if (currentUserId) {
      get()._saveUserHistory(currentUserId)
    }
    // 清空会话，重置为一个空会话（无痕模式）
    const emptySession = createEmptySession()
    set({ 
      currentUserId: null,
      sessions: [emptySession],
      currentSessionId: emptySession.id,
      infonSessions: {},
      privacyInferences: {}
    })
  },

  // 多会话与状态：初始化一个空会话
  sessions: (() => {
    const s = createEmptySession()
    return [s]
  })(),
  currentSessionId: null,
  isGenerating: false,
  abortController: null,

  // 信息元提取：按会话维护运行列表
  // infonSessions: { [sessionId]: { runs: Array<Run> } }
  // Run: { id, targetType: 'pending'|'message', targetKey: string, modality: 'text'|'image', imageIndex?, status: 'running'|'done'|'aborted'|'error', progress: number, buffer: string, resultJson: any|null, error?: string, createdAt }
  infonSessions: {},
  // 缓存：上次提取的文本与每张图片的哈希
  lastPendingTextHash: null,
  lastPendingImageHashes: [],
  // 流式增量解析器状态：按 runId 维护
  infonParsers: {},

  // 高亮信息元：用于在聊天界面中高亮显示选中的信息元
  // { infon: object, run: object } 或 null
  highlightedInfon: null,

  // 设置高亮信息元
  setHighlightedInfon(infon, run) {
    set({ highlightedInfon: infon ? { infon, run } : null })
  },

  // 隐私推理：按会话维护推理结果
  // privacyInferences: { [sessionId]: { status: 'idle'|'running'|'done'|'error', risks: Array, buffer: string, abortController: AbortController|null, createdAt, updatedAt } }
  privacyInferences: {},
  
  // 隐私推理增量解析器状态：按会话维护
  privacyParsers: {}, // { [sessionId]: parserState }
  
  // 选中的法律（用于推理）
  selectedLaw: null, // { key: 'PIPL', data: {...} }
  
  // 选中的法律索引（用于UI显示）
  selectedLawIdx: 0,
  
  // 用户自定义的隐私项
  customPrivacyItems: [],
  
  // 选中的隐私项（Set转为Array存储）
  selectedPrivacyItems: [],
  
  // 设置选中的法律
  setSelectedLaw(lawKey, lawData) {
    set({ selectedLaw: { key: lawKey, data: lawData } })
  },
  
  // 设置选中的法律索引
  setSelectedLawIdx(idx) {
    set({ selectedLawIdx: idx })
  },
  
  // 添加自定义隐私项
  addCustomPrivacyItem(item) {
    set(state => ({
      customPrivacyItems: [...state.customPrivacyItems, item]
    }))
  },
  
  // 删除自定义隐私项
  removeCustomPrivacyItem(itemId) {
    set(state => ({
      customPrivacyItems: state.customPrivacyItems.filter(item => item.id !== itemId)
    }))
  },
  
  // 设置自定义隐私项列表
  setCustomPrivacyItems(items) {
    set({ customPrivacyItems: items })
  },
  
  // 设置选中的隐私项
  setSelectedPrivacyItems(items) {
    set({ selectedPrivacyItems: Array.isArray(items) ? items : Array.from(items) })
  },
  
  // 切换隐私项选中状态
  togglePrivacyItem(itemId) {
    set(state => {
      const selected = new Set(state.selectedPrivacyItems)
      if (selected.has(itemId)) {
        selected.delete(itemId)
      } else {
        selected.add(itemId)
      }
      return { selectedPrivacyItems: Array.from(selected) }
    })
  },

  // 初始化当前会话：第一次使用时指向首个会话
  _ensureCurrentSession() {
    const { sessions, currentSessionId } = get()
    if (!currentSessionId && sessions.length > 0) {
      set({ currentSessionId: sessions[0].id })
    }
  },

  // 读取当前会话：找不到则返回 null
  getCurrentSession() {
    const { sessions, currentSessionId } = get()
    return sessions.find(s => s.id === currentSessionId) || null
  },

  // 内部：获取或创建当前会话的信息元会话容器
  _getOrCreateInfonSession(sessionId) {
    let box = get().infonSessions?.[sessionId]
    if (!box) {
      box = { runs: [] }
      set((state) => ({ infonSessions: { ...(state.infonSessions || {}), [sessionId]: box } }))
    }
    return box
  },

  // 内部：追加信息元运行
  _appendInfonRun(sessionId, run) {
    set((state) => {
      const current = state.infonSessions?.[sessionId] || { runs: [] }
      const next = { runs: [...current.runs, run] }
      return { infonSessions: { ...(state.infonSessions || {}), [sessionId]: next } }
    })
  },

  // 内部：更新信息元运行
  _updateInfonRun(sessionId, runId, updater) {
    set((state) => {
      const box = state.infonSessions?.[sessionId]
      if (!box) return {}
      const runs = box.runs.map(r => r.id === runId ? updater(r) : r)
      return { infonSessions: { ...state.infonSessions, [sessionId]: { runs } } }
    })
  },

  // 读取当前会话的所有信息元运行
  getCurrentInfonRuns() {
    const session = get().getCurrentSession()
    if (!session) return []
    return (get().infonSessions?.[session.id]?.runs) || []
  },

  // 清空所有 pending 信息元：供组件发送前调用
  clearAllPendingInfons() {
    const session = get().getCurrentSession()
    if (!session) return
    try {
      const runs = (get().infonSessions?.[session.id]?.runs) || []
      runs.forEach((r) => {
        if (r.targetType === 'pending' && r.status === 'running') {
          try { r.controller?.abort?.() } catch (_) {}
        }
      })
    } catch (_) {}
    set((state) => {
      const box = state.infonSessions?.[session.id] || { runs: [] }
      const nextRuns = box.runs.filter(r => r.targetType !== 'pending')
      return { infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
  },

  // 采纳当前会话的所有 pending 结果为指定 message 的结果
  // 将 targetType: 'pending' 的 run 改写为 targetType: 'message'，targetKey=messageId
  // 返回被采纳的数量
  adoptPendingInfonsToMessage(messageId) {
    const session = get().getCurrentSession()
    if (!session || !messageId) return 0
    let adopted = 0
    set((state) => {
      const box = state.infonSessions?.[session.id]
      if (!box) return {}
      const runs = box.runs.map((r) => {
        if (r.targetType === 'pending') {
          adopted++
          return { ...r, targetType: 'message', targetKey: messageId }
        }
        return r
      })
      return { infonSessions: { ...state.infonSessions, [session.id]: { runs } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
    return adopted
  },

  // 创建会话：并切换为当前
  createSession() {
    const newSession = createEmptySession()
    set((state) => ({
      sessions: [newSession, ...state.sessions],
      currentSessionId: newSession.id,
    }))
  },

  // 设置模型
  setModel(modelId) {
    set({ model: modelId })
  },

  // 拉取模型列表：兼容 OpenAI / Ollama 响应结构
  async fetchModels() {
    try {
      const res = await fetch(`${get().baseUrl}/models`, { method: 'GET' })
      const json = await res.json().catch(() => ({}))
      let list = []
      // OpenAI: { data: [{id}...] }
      if (Array.isArray(json?.data)) {
        list = json.data.map((m) => m?.id).filter(Boolean)
      }
      // 某些实现: { models: [...] }
      if (!list.length && Array.isArray(json?.models)) {
        list = json.models.map((m) => m?.id || m?.name || m).filter(Boolean)
      }
      // 兜底：如果直接是数组
      if (!list.length && Array.isArray(json)) {
        list = json.map((m) => m?.id || m?.name || m).filter(Boolean)
      }
      if (list.length) set((state) => ({ models: Array.from(new Set([...(state.models || []), ...list])) }))
    } catch (_) {
      // 忽略错误
    }
  },

  // 添加自定义 API 模型：提供 modelId/baseUrl/apiKey，合并到选择列表
  addApiModel({ id, baseUrl, apiKey }) {
    if (!id || !baseUrl || !apiKey) return
    set((state) => ({
      customProviders: { ...(state.customProviders || {}), [id]: { baseUrl, apiKey } },
      customModels: Array.from(new Set([...(state.customModels || []), id])),
      models: Array.from(new Set([...(state.models || []), id]))
    }))
  },

  // 切换会话
  switchSession(id) {
    set({ currentSessionId: id })
  },

  // 删除会话：如果删除当前会话，则自动切换到剩余第一个
  deleteSession(id) {
    set((state) => {
      const nextSessions = state.sessions.filter(s => s.id !== id)
      let nextCurrent = state.currentSessionId
      if (state.currentSessionId === id) {
        nextCurrent = nextSessions[0]?.id || null
      }
      return { sessions: nextSessions, currentSessionId: nextCurrent }
    })
  },

  // 重命名会话
  renameSession(id, title) {
    set((state) => ({
      sessions: state.sessions.map(s => s.id === id ? { ...s, title, updatedAt: Date.now() } : s),
    }))
  },

  // 追加消息：用于用户或助手消息写入
  _appendMessage(sessionId, message) {
    set((state) => ({
      sessions: state.sessions.map(s => {
        if (s.id !== sessionId) return s
        return {
          ...s,
          updatedAt: Date.now(),
          messages: [...s.messages, message],
        }
      })
    }))
  },

  // 更新某条消息：按消息 id 定位并更新（用于流式增量）
  _updateMessage(sessionId, messageId, updater) {
    set((state) => ({
      sessions: state.sessions.map(s => {
        if (s.id !== sessionId) return s
        const messages = s.messages.map(m => m.id === messageId ? updater(m) : m)
        return { ...s, messages, updatedAt: Date.now() }
      })
    }))
  },

  // ---------- 信息元提取：启动/中止 ----------
  // 停止所有 pending 目标的提取；clear=true 时同时清除结果
  abortPendingInfons(clear = false) {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = (get().infonSessions?.[session.id]?.runs) || []
    for (const r of runs) {
      if (r.targetType === 'pending' && r.status === 'running') {
        try { r.controller?.abort?.() } catch (_) {}
      }
    }
    if (clear) {
      set((state) => {
        const box = state.infonSessions?.[session.id]
        if (!box) return {}
        const nextRuns = box.runs.filter(r => r.targetType !== 'pending')
        return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } } }
      })
    } else {
      // 直接移除被中止的 pending 运行
      const toAbort = new Set(runs.filter(r => r.targetType === 'pending' && r.status === 'running').map(r => r.id))
      set((state) => {
        const box = state.infonSessions?.[session.id]
        if (!box) return {}
        const nextRuns = box.runs.filter(r => !(r.targetType === 'pending' && toAbort.has(r.id)))
        return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } } }
      })
    }
  },

  // 单独中止某个 run
  abortInfonRun(runId) {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = (get().infonSessions?.[session.id]?.runs) || []
    const r = runs.find(x => x.id === runId)
    if (!r) return
    try { r.controller?.abort?.() } catch (_) {}
    // 移除该 run
    set((state) => {
      const box = state.infonSessions?.[session.id]
      if (!box) return {}
      const nextRuns = box.runs.filter(x => x.id !== runId)
      return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } } }
    })
  },

  // 发送消息时处理 pending 信息元：清除所有 pending 任务，因为 message 任务将替代它们
  clearAllPendingInfons() {
    const session = get().getCurrentSession()
    if (!session) return
    const runs = (get().infonSessions?.[session.id]?.runs) || []
    // 先中止所有 pending 运行
    runs.forEach((r) => {
      if (r.targetType === 'pending' && r.status === 'running') {
        try { r.controller?.abort?.() } catch (_) {}
      }
    })
    // 再移除
    const toRemove = new Set(runs.filter(r => r.targetType === 'pending').map(r => r.id))
    if (toRemove.size > 0) {
      set((state) => {
        const box = state.infonSessions?.[session.id]
        if (!box) return {}
        const nextRuns = box.runs.filter(r => !toRemove.has(r.id))
        return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } } }
      })
    }
  },

  // 启动基于 pending 输入的信息元提取
  startPendingInfons(text, imageDataUrls) {
    const session = get().getCurrentSession()
    if (!session) return
    // 输入为空不启动
    const t = (text || '').trim()
    const imgs = Array.isArray(imageDataUrls) ? imageDataUrls.filter(Boolean) : []
    if (!t && imgs.length === 0) return

    // 计算哈希
    const textHash = t ? computeHashId(t) : null
    const imageHashes = imgs.map((u) => computeHashId(u))

    // 文本：只有 hash 改变才重提；若改变则移除旧文本 pending 结果
    if (t) {
      if (textHash !== get().lastPendingTextHash) {
        // 先中止旧的 pending 文本 run，再移除
        try {
          const currentRuns = (get().infonSessions?.[session.id]?.runs) || []
          currentRuns.forEach((r) => {
            if (r.targetType === 'pending' && r.modality === 'text' && r.status === 'running') {
              try { r.controller?.abort?.() } catch (_) {}
            }
          })
        } catch (_) {}
        set((state) => {
          const box = state.infonSessions?.[session.id] || { runs: [] }
          const nextRuns = box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'text'))
          return { infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } }, lastPendingTextHash: textHash }
        })
        get()._startTextInfonRun({ targetType: 'pending', targetKey: 'pending', text: t })
      }
    } else {
      // 没有文本则中止并清理所有 pending 文本 run
      try {
        const currentRuns = (get().infonSessions?.[session.id]?.runs) || []
        currentRuns.forEach((r) => {
          if (r.targetType === 'pending' && r.modality === 'text' && r.status === 'running') {
            try { r.controller?.abort?.() } catch (_) {}
          }
        })
      } catch (_) {}
      set((state) => {
        const box = state.infonSessions?.[session.id]
        if (!box) return {}
        const nextRuns = box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'text'))
        return { infonSessions: { ...state.infonSessions, [session.id]: { runs: nextRuns } }, lastPendingTextHash: null }
      })
    }

    // 图片：新增只为新 hash 启动；移除消失的 hash 的 run
    // 中止并移除不再存在的图片 pending runs
    try {
      const box = get().infonSessions?.[session.id] || { runs: [] }
      const currentHashes = new Set(imageHashes)
      box.runs.forEach((r) => {
        if (r.targetType === 'pending' && r.modality === 'image' && !currentHashes.has(r._hash) && r.status === 'running') {
          try { r.controller?.abort?.() } catch (_) {}
        }
      })
    } catch (_) {}
    set((state) => {
      const box = state.infonSessions?.[session.id] || { runs: [] }
      const currentHashes = new Set(imageHashes)
      let nextRuns = box.runs.filter(r => !(r.targetType === 'pending' && r.modality === 'image' && !currentHashes.has(r._hash)))
      // 再为新增的 hash 启动 run
      const existing = new Set(nextRuns.filter(r => r.targetType === 'pending' && r.modality === 'image').map(r => r._hash))
      const toStart = []
      imageHashes.forEach((h, idx) => { if (!existing.has(h)) toStart.push(idx) })
      // 写回 runs（暂不加入新 run，这里只清理；启动在 set 之后执行）
      return { infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } }, lastPendingImageHashes: imageHashes }
    })

    // 启动新增图片的 run
    const existingHashes = new Set(((get().infonSessions?.[session.id]?.runs) || []).filter(r => r.targetType === 'pending' && r.modality === 'image').map(r => r._hash))
    imageHashes.forEach((h, idx) => {
      if (!existingHashes.has(h)) {
        get()._startImageInfonRun({ targetType: 'pending', targetKey: 'pending', dataUrl: imgs[idx], imageIndex: idx, _hash: h })
      }
    })
  },

  // 发送后基于消息 ID 启动信息元提取
  startMessageInfons(messageId) {
    const session = get().getCurrentSession()
    if (!session) return
    const m = (session.messages || []).find(x => x.id === messageId)
    if (!m) return
    const t = (m.content || '').trim()
    const imgs = Array.isArray(m.images) ? m.images.filter(Boolean) : []
    // 先中止旧的该 message 的运行，再清理
    try {
      const runs = (get().infonSessions?.[session.id]?.runs) || []
      runs.forEach((r) => {
        if (r.targetType === 'message' && r.targetKey === messageId && r.status === 'running') {
          try { r.controller?.abort?.() } catch (_) {}
        }
      })
    } catch (_) {}
    // 清理旧的该 message 的 runs
    set((state) => {
      const box = state.infonSessions?.[session.id] || { runs: [] }
      const nextRuns = box.runs.filter(r => r.targetType !== 'message' || r.targetKey !== messageId)
      return { infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } } }
    })
    if (t) get()._startTextInfonRun({ targetType: 'message', targetKey: messageId, text: t })
    if (imgs.length) imgs.forEach((dataUrl, idx) => get()._startImageInfonRun({ targetType: 'message', targetKey: messageId, dataUrl, imageIndex: idx }))
  },

  // 内部：文本信息元提取（/v1/chat/completions）
  async _startTextInfonRun({ targetType, targetKey, text }) {
    const session = get().getCurrentSession()
    if (!session) return

    // 计算当前对话轮次（中文注释）：基于消息数量
    const messageCount = (session.messages || []).length
    const currentRound = Math.floor(messageCount / 2) + 1
    
    // 获取已有的所有信息元（中文注释）：用于模型参考，避免重复和建立跨轮关系
    const currentRuns = get().getCurrentInfonRuns()
    const completedRuns = currentRuns.filter(r => r.status === 'done')
    const existingInfons = []
    completedRuns.forEach(r => {
      if (r.resultJson?.infons) {
        existingInfons.push(...r.resultJson.infons)
      }
    })

    const runId = generateId()
    const run = {
      id: runId,
      targetType,
      targetKey,
      modality: 'text',
      status: 'running',
      progress: 0,
      buffer: '',
      resultJson: null,
      createdAt: Date.now(),
      controller: null,
    }
    get()._appendInfonRun(session.id, run)

    // 文本信息元提取强制使用 DeepSeek 提供商（中文注释）：与 AgentPage 示例一致
    const deepseekId = 'deepseek-chat'
    let provider = get().customProviders?.[deepseekId]
    if (!provider) {
      try {
        get().addApiModel?.({ id: deepseekId, baseUrl: 'https://api.deepseek.com/v1', apiKey: 'sk-8c2ee9474f2f44f5969dcd5de280e634' })
      } catch (_) { }
      provider = get().customProviders?.[deepseekId]
    }
    const baseUrl = provider ? provider.baseUrl : get().baseUrl
    const headers = { 'Content-Type': 'application/json' }
    if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`

    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['text'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons as a strict single JSON object. Input text:\n\n${text}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, (r) => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ model: deepseekId, messages, temperature: 0, stream: true }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: errText || 'Request failed' }))
        return
      }
      const reader = res.body?.getReader()
      if (!reader) {
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'No stream' }))
        return
      }

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          // 流式增量解析（中文注释）：逐步提取infons
          const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (currentRun?.buffer || '') + content
          
          // 更新buffer
          get()._updateInfonRun(session.id, runId, (r) => ({ ...r, buffer }))
          
          // 使用增量解析器逐个提取infons
          const parserState = get().infonParsers?.[runId] || null
          const { state: newState, yielded } = incrementalExtractInfons(buffer, parserState)
          
          // 更新解析器状态
          set(state => ({
            infonParsers: {
              ...state.infonParsers,
              [runId]: newState
            }
          }))
          
          // 如果有新的infons被解析出来，立即添加到结果中（流式显示）
          if (yielded && yielded.length > 0) {
            get()._updateInfonRun(session.id, runId, (r) => {
              const currentInfons = r.resultJson?.infons || []
              
              // 智能合并：根据_objIndex更新现有对象或添加新对象
              const updatedInfons = [...currentInfons]
              
              yielded.forEach(newInfon => {
                const objIndex = newInfon._objIndex
                
                if (objIndex !== undefined) {
                  const existingIndex = updatedInfons.findIndex(inf => inf._objIndex === objIndex)
                  
                  if (existingIndex >= 0) {
                    // 更新现有对象
                    updatedInfons[existingIndex] = {
                      ...updatedInfons[existingIndex],
                      ...newInfon
                    }
                  } else {
                    // 添加新对象
                    updatedInfons.push(newInfon)
                  }
                } else {
                  // 没有objIndex，直接添加
                  updatedInfons.push(newInfon)
                }
              })
              
              return {
                ...r,
                status: 'running',
                resultJson: {
                  ...r.resultJson,
                  infons: updatedInfons
                }
              }
            })
          }
        }
        if (finish) {
          const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const sliced = extractFirstJSONObject(raw) || raw
          const { ok, value } = tryParseJSON(sliced)
          if (ok) {
            // 计算当前对话轮次和信息元次序
            const sessionObj = get().getCurrentSession()
            const messageCount = (sessionObj?.messages || []).length
            const messageRound = Math.floor(messageCount / 2) + 1 // 每轮对话包含用户和助手消息
            const currentRuns = get().getCurrentInfonRuns()
            const completedRuns = currentRuns.filter(r => r.status === 'done')
            const infonIndex = completedRuns.reduce((sum, r) => sum + (r.resultJson?.infons?.length || 0), 0)

            const normalized = normalizeInfonOutput(value, {
              recordTimeISO: nowISO,
              defaultModality: 'text',
              sessionId: session.id,
              messageRound,
              infonIndex,
              infonType: 'desc'
            })
            
            // 应用增量更新逻辑：去重和冲突解决（中文注释）
            const deduplicated = deduplicateAndMergeInfons(normalized.infons || [], existingInfons)
            const finalResult = { ...normalized, infons: deduplicated }
            
            get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'done', progress: 100, resultJson: finalResult }))
          } else {
            get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
          }
        }
      })
    } catch (err) {
      const aborted = err && err.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 内部：图像信息元提取（/api/chat）
  async _startImageInfonRun({ targetType, targetKey, dataUrl, imageIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    // 计算当前对话轮次（中文注释）：基于消息数量
    const messageCount = (session.messages || []).length
    const currentRound = Math.floor(messageCount / 2) + 1
    
    // 获取已有的所有信息元（中文注释）：用于模型参考，避免重复和建立跨轮关系
    const currentRuns = get().getCurrentInfonRuns()
    const completedRuns = currentRuns.filter(r => r.status === 'done')
    const existingInfons = []
    completedRuns.forEach(r => {
      if (r.resultJson?.infons) {
        existingInfons.push(...r.resultJson.infons)
      }
    })

    const runId = generateId()
    const run = {
      id: runId,
      targetType,
      targetKey,
      modality: 'image',
      imageIndex,
      _hash,
      status: 'running',
      progress: 0,
      buffer: '',
      resultJson: null,
      createdAt: Date.now(),
      controller: null,
    }
    get()._appendInfonRun(session.id, run)

    // 自定义提供商通常不支持图片
    const provider = get().customProviders?.[get().model]
    if (provider) {
      get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'Image messages are not supported for this model' }))
      return
    }

    const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')
    const stripDataUrl = (s) => {
      if (typeof s !== 'string') return s
      const i = s.indexOf(',')
      if (i >= 0 && s.slice(0, i).includes('base64')) return s.slice(i + 1)
      return s
    }

    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['image'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: 'Extract Situation Theory infons as a strict single JSON object.', images: [stripDataUrl(dataUrl)] },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, (r) => ({ ...r, controller }))

    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: get().model, messages, stream: true, options: { temperature: 0 } }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: errText || 'Request failed' }))
        return
      }
      const reader = res.body?.getReader()
      if (!reader) {
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'No stream' }))
        return
      }

      await streamOllamaChatResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          // 流式增量解析（中文注释）：逐步提取infons
          const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (currentRun?.buffer || '') + content
          
          // 更新buffer
          get()._updateInfonRun(session.id, runId, (r) => ({ ...r, buffer }))
          
          // 使用增量解析器逐个提取infons
          const parserState = get().infonParsers?.[runId] || null
          const { state: newState, yielded } = incrementalExtractInfons(buffer, parserState)
          
          // 更新解析器状态
          set(state => ({
            infonParsers: {
              ...state.infonParsers,
              [runId]: newState
            }
          }))
          
          // 如果有新的infons被解析出来，立即添加到结果中（流式显示）
          if (yielded && yielded.length > 0) {
            get()._updateInfonRun(session.id, runId, (r) => {
              const currentInfons = r.resultJson?.infons || []
              
              // 智能合并：根据_objIndex更新现有对象或添加新对象
              const updatedInfons = [...currentInfons]
              
              yielded.forEach(newInfon => {
                const objIndex = newInfon._objIndex
                
                if (objIndex !== undefined) {
                  const existingIndex = updatedInfons.findIndex(inf => inf._objIndex === objIndex)
                  
                  if (existingIndex >= 0) {
                    // 更新现有对象
                    updatedInfons[existingIndex] = {
                      ...updatedInfons[existingIndex],
                      ...newInfon
                    }
                  } else {
                    // 添加新对象
                    updatedInfons.push(newInfon)
                  }
                } else {
                  // 没有objIndex，直接添加
                  updatedInfons.push(newInfon)
                }
              })
              
              return {
                ...r,
                status: 'running',
                resultJson: {
                  ...r.resultJson,
                  infons: updatedInfons
                }
              }
            })
          }
        }
        if (finish) {
          const raw = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)?.buffer || ''
          const sliced = extractFirstJSONObject(raw) || raw
          const { ok, value } = tryParseJSON(sliced)
          if (ok) {
            // 计算当前对话轮次和信息元次序
            const sessionObj = get().getCurrentSession()
            const messageCount = (sessionObj?.messages || []).length
            const messageRound = Math.floor(messageCount / 2) + 1 // 每轮对话包含用户和助手消息
            const currentRuns = get().getCurrentInfonRuns()
            const completedRuns = currentRuns.filter(r => r.status === 'done')
            const infonIndex = completedRuns.reduce((sum, r) => sum + (r.resultJson?.infons?.length || 0), 0)

            const normalized = normalizeInfonOutput(value, {
              recordTimeISO: nowISO,
              defaultModality: 'image',
              sessionId: session.id,
              messageRound,
              infonIndex,
              infonType: 'desc'
            })
            
            // 应用增量更新逻辑：去重和冲突解决（中文注释）
            const deduplicated = deduplicateAndMergeInfons(normalized.infons || [], existingInfons)
            const finalResult = { ...normalized, infons: deduplicated }
            
            get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'done', progress: 100, resultJson: finalResult }))
          } else {
            get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'Invalid JSON output' }))
          }
        }
      })
    } catch (err) {
      const aborted = err && err.name === 'AbortError'
      get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: aborted ? 'aborted' : 'error', error: aborted ? undefined : 'Network error' }))
    }
  },

  // 发送消息：立即返回用户消息 ID，流式请求在后台进行
  async sendMessage(text) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 如历史对话包含图片，则改走多模态 /api/chat，并将图片并入上下文
    const hasHistoricalImages = (session.messages || []).some(m => Array.isArray(m.images) && m.images.length > 0)
    const providerForModel = get().customProviders?.[get().model]
    if (hasHistoricalImages && !providerForModel) {
      // 委托到多模态路径（其本身也会“立即返回”）
      return await get().sendMessageWithImages(text, [])
    }

    // 写入用户消息
    const userMsgId = generateId()
    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: text,
      createdAt: Date.now(),
    })

    // 预创建助手空消息用于流式写入
    const assistantMsgId = generateId()
    get()._appendMessage(session.id, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      reasoning: '',
      phase: 'thinking',
      streaming: true,
      createdAt: Date.now(),
    })

    // 组装上下文：将当前会话全部消息转换为 OpenAI Chat messages
    const payloadMessages = get().getCurrentSession().messages.map(m => ({
      role: m.role,
      content: m.content,
    }))

    // 发起请求：使用 AbortController 以支持停止
    const controller = new AbortController()
    set({ isGenerating: true, abortController: controller })

    // 后台执行网络与流式处理，不阻塞返回
    ;(async () => {
      try {
        const provider = get().customProviders?.[get().model]
        const baseUrl = provider ? provider.baseUrl : get().baseUrl
        const headers = { 'Content-Type': 'application/json' }
        if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`

        const res = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            model: get().model,
            messages: payloadMessages,
            temperature: 0.7,
            stream: true,
          }),
          signal: controller.signal,
        })

        if (!res.ok) {
          const textErr = await res.text().catch(() => '')
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: textErr || 'Request failed', content: m.content }))
          set({ isGenerating: false, abortController: null })
          return
        }

        const reader = res.body?.getReader()
        if (!reader) {
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: 'No stream', content: m.content }))
          set({ isGenerating: false, abortController: null })
          return
        }

        let inThink = false
        await streamOpenAIResponse(reader, ({ content, reasoning, finish }) => {
          if (reasoning) {
            get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, reasoning: (m.reasoning || '') + reasoning }))
          }

          if (typeof content === 'string' && content.length) {
            let rest = content
            while (rest && rest.length) {
              if (inThink) {
                const endIdx = rest.indexOf('</think>')
                if (endIdx >= 0) {
                  const head = rest.slice(0, endIdx)
                  const tail = rest.slice(endIdx + 8)
                  if (head) {
                    get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, reasoning: (m.reasoning || '') + head }))
                  }
                  inThink = false
                  rest = tail
                  continue
                } else {
                  get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, reasoning: (m.reasoning || '') + rest }))
                  rest = ''
                  break
                }
              } else {
                const startIdx = rest.indexOf('<think>')
                if (startIdx >= 0) {
                  const before = rest.slice(0, startIdx)
                  const tail = rest.slice(startIdx + 7)
                  if (before) {
                    get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, content: (m.content || '') + before, phase: 'answering' }))
                  }
                  inThink = true
                  rest = tail
                  continue
                } else {
                  get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, content: (m.content || '') + rest, phase: 'answering' }))
                  rest = ''
                  break
                }
              }
            }
          }

          if (finish) {
            get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, phase: 'done' }))
            // 已禁用对模型回复的信息元提取
          }
        })
      } catch (err) {
        const msg = (err && (err.name === 'AbortError')) ? 'Aborted' : 'Network error'
        get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: msg }))
      } finally {
        set({ isGenerating: false, abortController: null })
      }
    })()

    // 立即返回用户消息 ID
    return userMsgId
  },

  // 发送带图片的多模态消息：立即返回用户消息 ID，流式在后台执行
  async sendMessageWithImages(text, imageDataUrls) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 写入用户消息：包含图片预览（data URL）
    const userMsgId = generateId()
    const imgs = Array.isArray(imageDataUrls) ? imageDataUrls.filter(Boolean) : []
    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: text,
      images: imgs,
      createdAt: Date.now(),
    })

    // 如果是自定义提供商，暂不支持图片：直接写入错误并返回
    const provider = get().customProviders?.[get().model]
    if (provider) {
      const assistantMsgId = generateId()
      get()._appendMessage(session.id, {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        reasoning: '',
        phase: 'done',
        streaming: false,
        error: 'Image messages are not supported for this model',
        createdAt: Date.now(),
      })
      return userMsgId
    }

    // 预创建助手空消息用于流式写入
    const assistantMsgId = generateId()
    get()._appendMessage(session.id, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      reasoning: '',
      phase: 'thinking',
      streaming: true,
      createdAt: Date.now(),
    })

    // 将历史消息转换为 Ollama /api/chat 的 messages：携带 images 时去掉 data: 前缀
    const stripDataUrl = (s) => {
      if (typeof s !== 'string') return s
      const i = s.indexOf(',')
      if (i >= 0 && s.slice(0, i).includes('base64')) return s.slice(i + 1)
      return s
    }

    const sessionMsgs = get().getCurrentSession().messages
    let lastImageIdx = -1
    for (let i = sessionMsgs.length - 1; i >= 0; i--) {
      const m = sessionMsgs[i]
      if (Array.isArray(m.images) && m.images.length > 0 && m.role === 'user') { lastImageIdx = i; break }
    }
    const history = sessionMsgs.map((m, idx) => {
      const o = { role: m.role, content: m.content }
      if (idx === lastImageIdx && Array.isArray(m.images) && m.images.length) {
        o.images = m.images.map(stripDataUrl)
      }
      return o
    })

    // 计算 /api 基址：将 baseUrl 的 /v1 替换为 /api
    const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')

    const controller = new AbortController()
    set({ isGenerating: true, abortController: controller })

    ;(async () => {
      try {
        const res = await fetch(`${apiBase}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: get().model,
            messages: history,
            stream: true,
            options: { temperature: 0.2 }
          }),
          signal: controller.signal,
        })

        if (!res.ok) {
          const textErr = await res.text().catch(() => '')
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: textErr || 'Request failed', content: m.content }))
          set({ isGenerating: false, abortController: null })
          return
        }

        const reader = res.body?.getReader()
        if (!reader) {
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: 'No stream', content: m.content }))
          set({ isGenerating: false, abortController: null })
          return
        }

        await streamOllamaChatResponse(reader, ({ content, finish }) => {
          if (typeof content === 'string' && content.length) {
            get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, content: (m.content || '') + content, phase: 'answering' }))
          }
          if (finish) {
            get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, phase: 'done' }))
          }
        })
      } catch (err) {
        const msg = (err && (err.name === 'AbortError')) ? 'Aborted' : 'Network error'
        get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: msg }))
      } finally {
        set({ isGenerating: false, abortController: null })
      }
    })()

    // 立即返回用户消息 ID
    return userMsgId
  },

  // 停止生成：调用 AbortController 取消流
  stopGenerating() {
    const { abortController } = get()
    try { abortController?.abort() } catch (_) { }
    set({ isGenerating: false, abortController: null })
  },

  // 重新生成：删除最后一条助手消息并复用最后一条用户消息再次生成
  async regenerateLast() {
    const session = get().getCurrentSession()
    if (!session) return
    // 找到最后的用户消息
    const lastUserIndex = [...session.messages].reverse().findIndex(m => m.role === 'user')
    if (lastUserIndex === -1) return
    const idxFromEnd = lastUserIndex
    const userIdx = session.messages.length - 1 - idxFromEnd
    const lastUser = session.messages[userIdx]

    // 如果最后一条是助手，先移除：保持一问一答结构
    set((state) => ({
      sessions: state.sessions.map(s => {
        if (s.id !== session.id) return s
        const msgs = [...s.messages]
        if (msgs.length > userIdx + 1 && msgs[userIdx + 1].role === 'assistant') {
          msgs.splice(userIdx + 1, 1)
        }
        return { ...s, messages: msgs, updatedAt: Date.now() }
      })
    }))

    await get().sendMessage(lastUser.content)
  },

  // ========== 隐私推理相关方法 ==========
  
  // 启动隐私推理：基于当前会话的信息元和选中的法律
  async startPrivacyInference() {
    const session = get().getCurrentSession()
    if (!session) return
    
    const { selectedLaw, infonSessions } = get()
    if (!selectedLaw || !selectedLaw.data) {
      console.warn('No law selected for inference')
      return
    }
    
    // 获取当前会话的所有信息元（中文注释）：只使用增量更新后保留的信息元
    const runs = infonSessions?.[session.id]?.runs || []
    const allRawInfons = []
    const supersededIids = new Set() // 收集所有被取代的信息元iid
    
    // 第一遍：收集所有信息元和被取代的iid
    runs.forEach(run => {
      if (run.status === 'done' || run.status === 'running') {
        const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
        allRawInfons.push(...infons)
        // 收集被取代的iid
        infons.forEach(infon => {
          if (Array.isArray(infon._supersedes)) {
            infon._supersedes.forEach(oldIid => supersededIids.add(oldIid))
          }
        })
      }
    })
    
    // 第二遍：过滤掉被取代的信息元
    const allInfons = allRawInfons.filter(infon => {
      return infon.iid && !supersededIids.has(infon.iid)
    })
    
    if (allInfons.length === 0) {
      console.warn('No infons available for inference')
      return
    }
    
    // 初始化推理状态（中文注释）：记录当前选中的法律key，用于匹配高亮
    const abortController = new AbortController()
    set(state => ({
      privacyInferences: {
        ...state.privacyInferences,
        [session.id]: {
          status: 'running',
          risks: [],
          buffer: '',
          abortController,
          lawKey: selectedLaw.key, // 记录推理时使用的法律
          createdAt: Date.now(),
          updatedAt: Date.now()
        }
      }
    }))
    
    try {
      console.log(`[Privacy Inference] 使用 ${allInfons.length} 个信息元进行推理`)
      
      // 构建推理提示词
      const { fillPromptTemplate } = await import('./templates/inference.js')
      const prompt = fillPromptTemplate(allInfons, selectedLaw.data)
      
      console.log(`[Privacy Inference] Prompt 长度: ${prompt.length} 字符`)
      
      // 隐私推理强制使用 DeepSeek API（中文注释）：与信息元提取保持一致
      const deepseekId = 'deepseek-chat'
      let provider = get().customProviders?.[deepseekId]
      if (!provider) {
        try {
          get().addApiModel?.({ id: deepseekId, baseUrl: 'https://api.deepseek.com/v1', apiKey: 'sk-8c2ee9474f2f44f5969dcd5de280e634' })
        } catch (_) { }
        provider = get().customProviders?.[deepseekId]
      }
      const apiUrl = provider?.baseUrl || 'https://api.deepseek.com/v1'
      const apiKey = provider?.apiKey || ''
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {})
        },
        body: JSON.stringify({
          model: deepseekId,
          messages: [{ role: 'user', content: prompt }],
          stream: true,
          temperature: 0.5, // 适中温度以平衡创造性和准确性
        }),
        signal: abortController.signal
      })
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      
      // 流式接收并逐个解析风险项（中文注释）
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')
        
        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') continue
          
          try {
            const parsed = JSON.parse(data)
            const delta = parsed?.choices?.[0]?.delta || {}
            // 兼容多种字段：content / reasoning_content / reasoning / thoughts / inner_thoughts
            const contentDelta = (
              delta?.content ||
              delta?.reasoning_content ||
              delta?.reasoning ||
              delta?.thoughts ||
              delta?.inner_thoughts ||
              ''
            )
            if (contentDelta) {
              buffer += contentDelta
              
              // 使用增量解析器逐个提取风险项（中文注释）
              const { incrementalExtractRisks } = await import('./templates/inference.js')
              const parserState = get().privacyParsers?.[session.id] || null
              const { state: newState, yielded } = incrementalExtractRisks(buffer, parserState)
              
              // 更新解析器状态
              set(state => ({
                privacyParsers: {
                  ...state.privacyParsers,
                  [session.id]: newState
                }
              }))
              
              // 如果有新的风险项被解析出来，立即添加到结果中（流式显示）
              if (yielded && yielded.length > 0) {
                set(state => {
                  const currentRisks = state.privacyInferences?.[session.id]?.risks || []
                  
                  // 智能合并：根据_objIndex更新现有对象或添加新对象
                  const updatedRisks = [...currentRisks]
                  
                  yielded.forEach(newRisk => {
                    const objIndex = newRisk._objIndex
                    
                    if (objIndex !== undefined) {
                      // 查找是否已存在相同objIndex的risk
                      const existingIndex = updatedRisks.findIndex(r => r._objIndex === objIndex)
                      
                      if (existingIndex >= 0) {
                        // 更新现有对象（合并字段，保留新数据）
                        updatedRisks[existingIndex] = {
                          ...updatedRisks[existingIndex],
                          ...newRisk
                        }
                      } else {
                        // 添加新对象
                        updatedRisks.push(newRisk)
                      }
                    } else {
                      // 没有objIndex，直接添加（兜底逻辑）
                      updatedRisks.push(newRisk)
                    }
                  })
                  
                  return {
                    privacyInferences: {
                      ...state.privacyInferences,
                      [session.id]: {
                        ...state.privacyInferences[session.id],
                        status: 'running',
                        risks: updatedRisks,
                        buffer: buffer,
                        updatedAt: Date.now()
                      }
                    }
                  }
                })
              }
            }
          } catch (err) {
            // 忽略解析错误
          }
        }
      }
      
      console.log(`[Privacy Inference] 流式接收完成`)
      
      // 完成推理
      set(state => ({
        privacyInferences: {
          ...state.privacyInferences,
          [session.id]: {
            ...state.privacyInferences[session.id],
            status: 'done',
            abortController: null,
            updatedAt: Date.now()
          }
        }
      }))
      
    } catch (err) {
      if (err.name === 'AbortError') {
        set(state => ({
          privacyInferences: {
            ...state.privacyInferences,
            [session.id]: {
              ...state.privacyInferences[session.id],
              status: 'aborted',
              abortController: null,
              updatedAt: Date.now()
            }
          }
        }))
      } else {
        set(state => ({
          privacyInferences: {
            ...state.privacyInferences,
            [session.id]: {
              ...state.privacyInferences[session.id],
              status: 'error',
              error: err.message,
              abortController: null,
              updatedAt: Date.now()
            }
          }
        }))
      }
    }
  },
  
  // 停止隐私推理
  abortPrivacyInference() {
    const session = get().getCurrentSession()
    if (!session) return
    
    const inference = get().privacyInferences?.[session.id]
    if (inference?.abortController) {
      try {
        inference.abortController.abort()
      } catch (_) {}
    }
  },
  
  // 清除推理结果
  clearPrivacyInference() {
    const session = get().getCurrentSession()
    if (!session) return
    
    set(state => {
      const newInferences = { ...state.privacyInferences }
      const newParsers = { ...state.privacyParsers }
      delete newInferences[session.id]
      delete newParsers[session.id]
      return { 
        privacyInferences: newInferences,
        privacyParsers: newParsers
      }
    })
  },

  // ========== 用户历史数据持久化方法 ==========
  
  // 内部：加载用户历史数据
  _loadUserHistory(userId) {
    try {
      const data = loadUserSessions(userId)
      
      if (data && data.sessions && data.sessions.length > 0) {
        set({
          sessions: data.sessions,
          infonSessions: data.infonSessions || {},
          privacyInferences: data.privacyInferences || {},
          currentSessionId: data.sessions[0]?.id || null,
          customPrivacyItems: data.customPrivacyItems || [],
          selectedLawIdx: data.selectedLawIdx ?? 0,
          selectedPrivacyItems: data.selectedPrivacyItems || []
        })
        console.log('[PrivaSee] 用户历史数据已加载')
      } else {
        // 如果没有历史数据，初始化一个新会话
        const newSession = createEmptySession()
        set({
          sessions: [newSession],
          currentSessionId: newSession.id,
          infonSessions: {},
          privacyInferences: {},
          customPrivacyItems: [],
          selectedLawIdx: 0,
          selectedPrivacyItems: []
        })
      }
    } catch (error) {
      console.error('[PrivaSee] 加载用户历史失败:', error)
    }
  },
  
  // 内部：保存用户历史数据
  _saveUserHistory(userId) {
    try {
      const { sessions, infonSessions, privacyInferences, customPrivacyItems, selectedLawIdx, selectedPrivacyItems } = get()
      
      // 清理不可序列化的字段（中文注释）：移除 abortController
      const serializableInferences = {}
      Object.keys(privacyInferences).forEach(sessionId => {
        const inference = privacyInferences[sessionId]
        if (inference) {
          const { abortController, ...rest } = inference
          serializableInferences[sessionId] = rest
        }
      })
      
      saveUserSessions(userId, sessions, infonSessions, serializableInferences, customPrivacyItems, selectedLawIdx, selectedPrivacyItems)
    } catch (error) {
      console.error('[PrivaSee] 保存用户历史失败:', error)
    }
  },
  
  // 手动保存当前用户的数据
  saveCurrentUserHistory() {
    const { currentUserId } = get()
    if (currentUserId) {
      get()._saveUserHistory(currentUserId)
    }
  },

}))

// 自动保存：当用户登录时，定时保存历史数据（中文注释）
if (typeof window !== 'undefined') {
  let autoSaveTimer = null
  
  useStore.subscribe((state) => {
    // 清除旧的定时器
    if (autoSaveTimer) {
      clearTimeout(autoSaveTimer)
    }
    
    // 如果用户已登录，设置定时保存（30秒后）
    if (state.currentUserId) {
      autoSaveTimer = setTimeout(() => {
        useStore.getState().saveCurrentUserHistory()
      }, 30000) // 30秒延迟保存
    }
  })
  
  // 页面卸载前保存（中文注释）
  window.addEventListener('beforeunload', () => {
    const state = useStore.getState()
    if (state.currentUserId) {
      state._saveUserHistory(state.currentUserId)
    }
  })
}


