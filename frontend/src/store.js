import { create } from 'zustand'
import { buildSystemPrompt } from './templates/infons.js'
import { loadUserSessions, saveUserSessions } from './users/historyStorage'
import { getDefaultModelsConfig } from './config/defaultModelsConfig'
import { getDefaultApiModels, getDefaultApiModelIds } from './config/defaultApiModelsConfig'
import { getModelModalities } from './utils/modelUtils'
import { callDeepseekOcr, callDeepseekOcrStream } from './utils/deepseekOcrApi'
import { saveFiles, loadFiles, deleteMessageFiles, deleteSessionFiles } from './utils/fileStorage'

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

// 解析 Ollama /api/chat 流：逐行解析 JSON，并处理"全量快照"或"增量 token"两种格式
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
  model: getDefaultModelsConfig().conversationModel,
  models: [...getDefaultApiModelIds()], // 可选模型列表，初始化时包含内置 API 模型
  customModels: [...getDefaultApiModelIds()], // 通过 API key 添加的自定义模型，初始化时包含内置 API 模型
  customProviders: getDefaultApiModels(), // { [modelId]: { baseUrl, apiKey } }，初始化时加载内置 API 模型

  // 用户状态标识（中文注释）：用于判断是否启用历史数据持久化
  currentUserId: null,
  
  // 模型配置（中文注释）：用户可自定义的模型选择
  // 直接推理模式的模型配置
  directInferenceModel: getDefaultModelsConfig().directInferenceModel, // 直接推理模式：隐私推理模型
  
  // 提取信息元模式的模型配置
  infonExtractionModel: getDefaultModelsConfig().infonExtractionModel, // 提取信息元模式：信息元提取模型
  infonPrivacyInferenceModel: getDefaultModelsConfig().infonPrivacyInferenceModel, // 提取信息元模式：隐私推理模型
  
  // 共用的模型配置
  imageParsingModel: getDefaultModelsConfig().imageParsingModel, // 图片解析模型（共用）
  protectionSuggestionModel: getDefaultModelsConfig().protectionSuggestionModel, // Privacy Protection Suggestions模型（共用，仅限API key模型）
  
  // 推断模式（中文注释）：extract（提取信息元）或 direct（直接推断）
  inferenceMode: getDefaultModelsConfig().inferenceMode, // 默认为提取信息元模式
  
  // Pending用户输入（中文注释）：用于直接推断模式下获取未发送的输入
  pendingUserInput: '',
  
  // Pending音频（中文注释）：用于直接推断模式下获取未发送的音频转写
  pendingAudios: [],
  
  // OCR 文件对象映射（中文注释）：内存中保存 File 对象用于预览，不持久化
  // 格式：{ [sessionId]: { [messageId]: { [fileId]: File } } }
  ocrFileObjects: {},
  
  // Pending图片（中文注释）：用于直接推断模式下获取未发送的图片分析
  pendingImages: [],
  
  // 设置模型配置
  setDirectInferenceModel: (modelId) => {
    set({ directInferenceModel: modelId })
  },
  
  setInfonExtractionModel: (modelId) => {
    set({ infonExtractionModel: modelId })
  },
  
  setInfonPrivacyInferenceModel: (modelId) => {
    set({ infonPrivacyInferenceModel: modelId })
  },
  
  setImageParsingModel: (modelId) => {
    set({ imageParsingModel: modelId })
  },
  
  setProtectionSuggestionModel: (modelId) => {
    set({ protectionSuggestionModel: modelId })
  },
  
  // 恢复默认模型配置
  resetToDefaultModels: () => {
    const defaultConfig = getDefaultModelsConfig()
    set({
      directInferenceModel: defaultConfig.directInferenceModel,
      infonExtractionModel: defaultConfig.infonExtractionModel,
      infonPrivacyInferenceModel: defaultConfig.infonPrivacyInferenceModel,
      imageParsingModel: defaultConfig.imageParsingModel,
      protectionSuggestionModel: defaultConfig.protectionSuggestionModel,
      inferenceMode: defaultConfig.inferenceMode,
    })
  },
  
  // 设置推断模式
  setInferenceMode: (mode) => {
    // 模式切换时清除当前编辑中的 pending 数据（但保留历史数据）
    const session = get().getCurrentSession()
    if (session?.id) {
      const currentInfonSession = get().infonSessions?.[session.id]
      if (currentInfonSession) {
        // 移除所有 targetType 为 'pending' 的 runs，保留 'message' 类型的历史数据
        const filteredRuns = (currentInfonSession.runs || []).filter(run => run.targetType !== 'pending')
        
        set(state => ({
          infonSessions: {
            ...state.infonSessions,
            [session.id]: {
              ...currentInfonSession,
              runs: filteredRuns
            }
          }
        }))
      }
    }
    
    set({ inferenceMode: mode })
  },
  
  // 设置pending用户输入
  setPendingUserInput: (input) => {
    set({ pendingUserInput: input })
  },
  
  // 设置pending音频
  setPendingAudios: (audios) => {
    set({ pendingAudios: audios })
  },
  
  // 设置pending图片
  setPendingImages: (images) => {
    set({ pendingImages: images })
  },
  
  // 分析图片内容（用于直接推断模式）
  async analyzeImage(imageDataUrl) {
    try {
      // 使用配置的图片解析模型
      const configuredModel = get().imageParsingModel || 'gemma3:12b'
      const provider = get().customProviders?.[configuredModel]
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const model = configuredModel
      
      // 精心设计的提示词：提取图片中的所有细节和可能关联到个人隐私的内容
      const systemPrompt = `You are a privacy-focused AI assistant. Analyze the provided image and extract ALL details that could potentially be used to infer personal information or be associated with other data to deduce privacy-related information.

Your task:
1. Identify and describe ALL visible elements in the image (people, objects, locations, text, symbols, etc.)
2. Extract any text visible in the image
3. Identify any identifiable information (faces, license plates, addresses, names, etc.)
4. Note contextual clues (time, place, activity, relationships, etc.)
5. Identify any metadata or background details that could reveal personal information
6. Provide a brief summary of the image content

Output format:
- Be thorough and detailed
- List all observations systematically
- Highlight privacy-sensitive elements
- Keep your response concise but comprehensive
- Use clear and structured language`

      const userPrompt = 'Analyze this image and extract all details as instructed.'
      
      const headers = { 'Content-Type': 'application/json' }
      if (provider?.apiKey) {
        headers['Authorization'] = `Bearer ${provider.apiKey}`
      }
      
      // 根据模型类型确定 max_tokens（omni 系列限制为 2048）
      const modelName = model.toLowerCase()
      const isOmni = modelName.includes('omni')
      const maxTokens = isOmni ? 1000 : 2000
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: model,
          messages: [
            { role: 'system', content: systemPrompt },
            {
              role: 'user',
              content: [
                { type: 'text', text: userPrompt },
                { type: 'image_url', image_url: { url: imageDataUrl } }
              ]
            }
          ],
          temperature: 0.3,
          max_tokens: maxTokens,
        })
      })
      
      if (!response.ok) {
        throw new Error(`Image analysis failed: ${response.statusText}`)
      }
      
      const result = await response.json()
      const analysisText = result.choices?.[0]?.message?.content?.trim() || ''
      
      if (!analysisText) {
        throw new Error('No analysis result returned')
      }
      
      return analysisText
    } catch (error) {
      console.error('[Image Analysis] Error:', error)
      throw error
    }
  },
  
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
      model: getDefaultModelsConfig().conversationModel, // 重置为默认对话模型
      infonSessions: {},
      privacyInferences: {},
      sessionKeywords: {} // 清空关键词
    })
  },
  
  // 清除全部记录
  clearAllData: () => {
    const emptySession = createEmptySession()
    set({
      sessions: [emptySession],
      currentSessionId: emptySession.id,
      model: getDefaultModelsConfig().conversationModel, // 重置为默认对话模型
      infonSessions: {},
      privacyInferences: {},
      sessionKeywords: {},
      protectionSuggestions: {},
      customPrivacyItems: [],
      selectedPrivacyItems: []
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
  
  // 隐私推理关键词：按会话维护关键词列表（持久化，用于高亮和上下文）
  // sessionKeywords: { [sessionId]: Set<string> } - 累积的关键词集合
  sessionKeywords: {},
  
  // 隐私推理增量解析器状态：按会话维护
  privacyParsers: {}, // { [sessionId]: parserState }
  
  // 隐私保护修改建议：按会话维护
  // protectionSuggestions: { [sessionId]: { status: 'idle'|'running'|'done'|'error', suggestions: Array, error: string, abortController: AbortController|null } }
  protectionSuggestions: {},
  
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
    get()._ensureCurrentSession()
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
    if (!session || !messageId) return { adopted: 0, runIds: [] }
    let adopted = 0
    const adoptedRunIds = []
    set((state) => {
      const box = state.infonSessions?.[session.id]
      if (!box) return {}
      const runs = box.runs.map((r) => {
        if (r.targetType === 'pending') {
          adopted++
          adoptedRunIds.push(r.id)
          return { ...r, targetType: 'message', targetKey: messageId }
        }
        return r
      })
      return { infonSessions: { ...state.infonSessions, [session.id]: { runs } }, lastPendingTextHash: null, lastPendingImageHashes: [] }
    })
    return { adopted, runIds: adoptedRunIds }
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
  
  // 删除自定义API模型
  removeApiModel(id) {
    if (!id) return
    set((state) => {
      const newProviders = { ...state.customProviders }
      delete newProviders[id]
      
      return {
        customProviders: newProviders,
        customModels: (state.customModels || []).filter(m => m !== id),
        models: (state.models || []).filter(m => m !== id),
        // 如果删除的是当前选中的模型，切换到默认模型
        model: state.model === id ? getDefaultModelsConfig().conversationModel : state.model,
        // 如果删除的是配置的模型，重置为默认值
        directInferenceModel: state.directInferenceModel === id ? 'deepseek-chat' : state.directInferenceModel,
        infonExtractionModel: state.infonExtractionModel === id ? 'deepseek-chat' : state.infonExtractionModel,
        infonPrivacyInferenceModel: state.infonPrivacyInferenceModel === id ? 'deepseek-chat' : state.infonPrivacyInferenceModel,
        imageParsingModel: state.imageParsingModel === id ? 'gemma3:12b' : state.imageParsingModel,
        protectionSuggestionModel: state.protectionSuggestionModel === id ? 'deepseek-chat' : state.protectionSuggestionModel,
      }
    })
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
      
      // 清理该会话的关键词
      const updatedKeywords = { ...state.sessionKeywords }
      delete updatedKeywords[id]
      
      return { 
        sessions: nextSessions, 
        currentSessionId: nextCurrent,
        sessionKeywords: updatedKeywords
      }
    })
    
    // 异步清理 IndexedDB 中的文件
    deleteSessionFiles(id).catch(err => 
      console.error('[deleteSession] 清理 IndexedDB 文件失败:', err)
    )
  },

  // 重命名会话
  renameSession(id, title) {
    set((state) => ({
      sessions: state.sessions.map(s => s.id === id ? { ...s, title, updatedAt: Date.now() } : s),
    }))
  },

  // 自动生成会话标题（基于第一条消息）
  async generateSessionTitle(sessionId) {
    console.log('[Session Title] 开始生成标题，会话ID:', sessionId)
    const session = get().sessions.find(s => s.id === sessionId)
    if (!session) {
      console.log('[Session Title] 未找到会话')
      return
    }
    
    // 只有在标题为默认值时才生成
    if (!session.title.startsWith('New chat')) {
      console.log('[Session Title] 标题不是默认值，跳过生成:', session.title)
      return
    }
    
    // 获取第一条用户消息
    const firstUserMessage = session.messages.find(msg => msg.role === 'user')
    if (!firstUserMessage) {
      console.log('[Session Title] 未找到用户消息')
      return
    }
    
    console.log('[Session Title] 第一条消息:', {
      hasContent: !!firstUserMessage.content,
      contentType: typeof firstUserMessage.content,
      hasImages: !!(firstUserMessage.images && firstUserMessage.images.length > 0),
      hasImageAnalysis: !!(firstUserMessage.imageAnalysis && Object.keys(firstUserMessage.imageAnalysis).length > 0),
      hasAudios: !!(firstUserMessage.audios && firstUserMessage.audios.length > 0)
    })
    
    // 提取消息内容：包含纯文本 + 音频转录 + 图片分析
    const contentParts = []
    
    // 1. 提取文本内容（已包含音频转录，带 <audio> 标签）
    if (typeof firstUserMessage.content === 'string') {
      // 移除音频标签，只保留转录文本
      const contentWithoutTags = firstUserMessage.content.replace(/<audio>([\s\S]*?)<\/audio>/gi, '$1')
      if (contentWithoutTags.trim()) {
        contentParts.push(contentWithoutTags.trim())
        console.log('[Session Title] 提取文本内容:', contentWithoutTags.trim().slice(0, 50) + '...')
      }
    } else if (Array.isArray(firstUserMessage.content)) {
      // 多模态消息（旧格式），提取文本部分
      const textParts = firstUserMessage.content
        .filter(part => part.type === 'text')
        .map(part => part.text)
      if (textParts.length > 0) {
        contentParts.push(textParts.join(' '))
        console.log('[Session Title] 提取文本内容（数组格式）:', textParts.join(' ').slice(0, 50) + '...')
      }
    }
    
    // 2. 提取图片分析内容
    const imageAnalysisMap = firstUserMessage.imageAnalysis || {}
    const imageUrls = firstUserMessage.images || []
    console.log('[Session Title] 图片分析数据:', {
      imageCount: imageUrls.length,
      analysisCount: Object.keys(imageAnalysisMap).length
    })
    imageUrls.forEach(url => {
      const analysis = imageAnalysisMap[url]
      if (analysis && analysis.trim()) {
        // 取图片分析的前200字符作为摘要
        const summary = analysis.trim().slice(0, 200)
        contentParts.push(summary)
        console.log('[Session Title] 提取图片分析:', summary.slice(0, 50) + '...')
      }
    })
    
    const content = contentParts.join(' ')
    console.log('[Session Title] 合并后的内容长度:', content.length)
    if (!content || content.trim().length === 0) {
      console.log('[Session Title] 内容为空，无法生成标题')
      return
    }
    
    // 限制内容长度
    const truncatedContent = content.slice(0, 500)
    
    try {
      // 使用 DeepSeek API 生成标题
      const configuredModel = get().infonExtractionModel || 'deepseek-chat'
      const provider = get().customProviders?.[configuredModel]
      
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const apiKey = provider ? provider.apiKey : null
      
      const requestBody = {
        model: configuredModel,
        messages: [
          {
            role: 'system',
            content: '你是一个对话标题生成助手。请根据用户的第一条消息，生成一个简短的对话标题（5-10个字）。只输出标题本身，不要有任何解释或标点符号。'
          },
          {
            role: 'user',
            content: `请为以下消息生成一个简短的对话标题：\n\n${truncatedContent}`
          }
        ],
        temperature: 0.7,
        max_tokens: 20
      }
      
      const headers = {
        'Content-Type': 'application/json'
      }
      
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`
      }
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(requestBody)
      })
      
      if (!response.ok) {
        console.error('[Session Title] API错误:', response.statusText)
        return
      }
      
      const data = await response.json()
      const generatedTitle = data.choices?.[0]?.message?.content?.trim()
      
      if (generatedTitle && generatedTitle.length > 0) {
        // 清理标题：移除引号、换行符等
        const cleanTitle = generatedTitle
          .replace(/^["'「『]+|["'」』]+$/g, '')
          .replace(/\n/g, ' ')
          .slice(0, 50) // 限制最大长度
        
        if (cleanTitle.length > 0) {
          get().renameSession(sessionId, cleanTitle)
          console.log('[Session Title] 自动生成标题:', cleanTitle)
        }
      }
    } catch (error) {
      console.error('[Session Title] 生成失败:', error)
    }
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
    get()._ensureCurrentSession()
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
  startPendingInfons(text, imageDataUrls, audioData) {
    get()._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return
    // 输入为空不启动
    const t = (text || '').trim()
    const imgs = Array.isArray(imageDataUrls) ? imageDataUrls.filter(Boolean) : []
    const audios = Array.isArray(audioData) ? audioData.filter(Boolean) : []
    if (!t && imgs.length === 0 && audios.length === 0) return

    // 计算哈希
    const textHash = t ? computeHashId(t) : null
    const imageHashes = imgs.map((u) => computeHashId(u))
    const audioHashes = audios.map((a) => computeHashId(a.id + (a.transcript || '')))

    // 检查哪些需要更新
    const textNeedsUpdate = t && textHash !== get().lastPendingTextHash
    const textNeedsRemove = !t && get().lastPendingTextHash !== null
    
    // 一次性中止需要更新的 runs
    try {
      const currentRuns = (get().infonSessions?.[session.id]?.runs) || []
      const imageHashSet = new Set(imageHashes)
      const audioHashSet = new Set(audioHashes)
      
      currentRuns.forEach((r) => {
        if (r.targetType !== 'pending' || r.status !== 'running') return
        
        // 中止需要更新的文本 run
        if ((textNeedsUpdate || textNeedsRemove) && r.modality === 'text') {
          try { r.controller?.abort?.() } catch (_) {}
        }
        // 中止不再存在的图片 run
        else if (r.modality === 'image' && !imageHashSet.has(r._hash)) {
          try { r.controller?.abort?.() } catch (_) {}
        }
        // 中止不再存在的音频 run
        else if (r.modality === 'audio' && !audioHashSet.has(r._hash)) {
          try { r.controller?.abort?.() } catch (_) {}
        }
      })
    } catch (_) {}

    // 一次性更新所有模态的 runs
    set((state) => {
      const box = state.infonSessions?.[session.id] || { runs: [] }
      let nextRuns = [...box.runs]
      
      // 移除需要更新的文本 run
      if (textNeedsUpdate || textNeedsRemove) {
        nextRuns = nextRuns.filter(r => !(r.targetType === 'pending' && r.modality === 'text'))
      }
      
      // 移除不再存在的图片 run
      const imageHashSet = new Set(imageHashes)
      nextRuns = nextRuns.filter(r => !(r.targetType === 'pending' && r.modality === 'image' && !imageHashSet.has(r._hash)))
      
      // 移除不再存在的音频 run
      const audioHashSet = new Set(audioHashes)
      nextRuns = nextRuns.filter(r => !(r.targetType === 'pending' && r.modality === 'audio' && !audioHashSet.has(r._hash)))
      
      return {
        infonSessions: { ...(state.infonSessions || {}), [session.id]: { runs: nextRuns } },
        lastPendingTextHash: t ? textHash : null,
        lastPendingImageHashes: imageHashes,
        lastPendingAudioHashes: audioHashes
      }
    })

    // 启动新的文本提取
    if (textNeedsUpdate) {
      get()._startTextInfonRun({ targetType: 'pending', targetKey: 'pending', text: t })
    }

    // 启动新增图片的 run
    const existingImageHashes = new Set(((get().infonSessions?.[session.id]?.runs) || []).filter(r => r.targetType === 'pending' && r.modality === 'image').map(r => r._hash))
    imageHashes.forEach((h, idx) => {
      if (!existingImageHashes.has(h)) {
        get()._startImageInfonRun({ targetType: 'pending', targetKey: 'pending', dataUrl: imgs[idx], imageIndex: idx, _hash: h })
      }
    })

    // 启动新增音频的 run
    const existingAudioHashes = new Set(((get().infonSessions?.[session.id]?.runs) || []).filter(r => r.targetType === 'pending' && r.modality === 'audio').map(r => r._hash))
    audioHashes.forEach((h, idx) => {
      if (!existingAudioHashes.has(h)) {
        get()._startAudioInfonRun({ targetType: 'pending', targetKey: 'pending', audio: audios[idx], audioIndex: idx, _hash: h })
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
    const audios = Array.isArray(m.audios) ? m.audios.filter(Boolean) : []
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
    if (audios.length) audios.forEach((audio, idx) => {
      const audioHash = computeHashId(audio.id + (audio.transcript || ''))
      get()._startAudioInfonRun({ targetType: 'message', targetKey: messageId, audio, audioIndex: idx, _hash: audioHash })
    })
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

    // 使用用户配置的信息元提取模型（中文注释）：优先使用配置，回退到 DeepSeek
    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const provider = get().customProviders?.[configuredModel]
    
    // 如果有provider（自定义API模型），使用provider配置；否则使用本地baseUrl（Ollama）
    const baseUrl = provider ? provider.baseUrl : get().baseUrl
    const headers = { 'Content-Type': 'application/json' }
    if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`
    
    // 根据模型类型确定 max_tokens（omni 系列限制为 2048）
    const isOmniModel = configuredModel.toLowerCase().includes('omni')
    const maxTokens = isOmniModel ? 2000 : 4096
    
    console.log(`[Infon Extraction] 使用模型: ${configuredModel}, API: ${baseUrl}`)

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
        headers: {
          ...headers,
          'Connection': 'keep-alive'
        },
        body: JSON.stringify({ 
          model: configuredModel, 
          messages, 
          temperature: 0,
          stream: true,
          max_tokens: maxTokens, // 根据模型限制输出长度
          top_p: 0.95, // 核采样
          frequency_penalty: 0.0,
          presence_penalty: 0.0,
        }),
        signal: controller.signal,
        keepalive: true // 启用连接复用
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

      // Debounce配置：减少解析频率
      let parseTimer = null
      let lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 200

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          // 更新buffer
          const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (currentRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, (r) => ({ ...r, buffer }))
          
          // Debounce解析逻辑
          const performParsing = () => {
            const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            if (!currentRun) return
            
            const buffer = currentRun.buffer || ''
            const parserState = get().infonParsers?.[runId] || null
            const { state: newState, yielded } = incrementalExtractInfons(buffer, parserState)
            
            set(state => ({
              infonParsers: {
                ...state.infonParsers,
                [runId]: newState
              }
            }))
            
            if (yielded && yielded.length > 0) {
              get()._updateInfonRun(session.id, runId, (r) => {
                const currentInfons = r.resultJson?.infons || []
                const updatedInfons = [...currentInfons]
                
                yielded.forEach(newInfon => {
                  const objIndex = newInfon._objIndex
                  
                  if (objIndex !== undefined) {
                    const existingIndex = updatedInfons.findIndex(inf => inf._objIndex === objIndex)
                    if (existingIndex >= 0) {
                      updatedInfons[existingIndex] = {
                        ...updatedInfons[existingIndex],
                        ...newInfon
                      }
                    } else {
                      updatedInfons.push(newInfon)
                    }
                  } else {
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
            
            lastParseTime = Date.now()
          }
          
          // Debounce策略
          const now = Date.now()
          const timeSinceLastParse = now - lastParseTime
          
          if (parseTimer) clearTimeout(parseTimer)
          
          if (timeSinceLastParse >= PARSE_DEBOUNCE_MS) {
            performParsing()
          } else {
            parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
          }
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
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

    // 使用用户配置的图片解析模型（中文注释）
    const configuredModel = get().imageParsingModel || 'gemma3:12b'
    const customProviders = get().customProviders
    
    // 检查模型是否支持图片
    const modalities = getModelModalities(configuredModel, customProviders)
    if (!modalities.image) {
      get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'Image messages are not supported for this model' }))
      return
    }
    
    const provider = customProviders?.[configuredModel]
    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['image'], nowISO, { currentRound, existingInfons })
    
    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, (r) => ({ ...r, controller }))

    try {
      let res, reader
      
      // 如果是 API 模型，使用 OpenAI Vision API 格式
      if (provider) {
        const messages = [
          { role: 'system', content: systemPrompt },
          { 
            role: 'user', 
            content: [
              {
                type: 'text',
                text: 'Extract Situation Theory infons as a strict single JSON object.'
              },
              {
                type: 'image_url',
                image_url: {
                  url: dataUrl // 保持 data:image/... 格式
                }
              }
            ]
          },
        ]
        
        const baseUrl = provider.baseUrl
        const headers = { 'Content-Type': 'application/json' }
        if (provider.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`
        
        // 根据模型类型确定 max_tokens（omni 系列限制为 2048）
        const isOmniModel = configuredModel.toLowerCase().includes('omni')
        const maxTokens = isOmniModel ? 2000 : 4096
        
        res = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: {
            ...headers,
            'Connection': 'keep-alive'
          },
          body: JSON.stringify({
            model: configuredModel,
            messages,
            temperature: 0,
            stream: true,
            max_tokens: maxTokens,
            top_p: 0.95,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
          }),
          signal: controller.signal,
          keepalive: true
        })
      } else {
        // 本地 Ollama 模型：使用 /api/chat 格式
        const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')
        const stripDataUrl = (s) => {
          if (typeof s !== 'string') return s
          const i = s.indexOf(',')
          if (i >= 0 && s.slice(0, i).includes('base64')) return s.slice(i + 1)
          return s
        }
        
        const messages = [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: 'Extract Situation Theory infons as a strict single JSON object.', images: [stripDataUrl(dataUrl)] },
        ]
        
        res = await fetch(`${apiBase}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: configuredModel, messages, stream: true, options: { temperature: 0 } }),
          signal: controller.signal,
        })
      }
      
      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: errText || 'Request failed' }))
        return
      }
      
      reader = res.body?.getReader()
      if (!reader) {
        get()._updateInfonRun(session.id, runId, (r) => ({ ...r, status: 'error', error: 'No stream' }))
        return
      }

      // 使用对应的流式响应处理器
      const streamHandler = provider ? streamOpenAIResponse : streamOllamaChatResponse
      await streamHandler(reader, async ({ content, finish }) => {
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

  // 内部：音频信息元提取（基于转录文本）
  async _startAudioInfonRun({ targetType, targetKey, audio, audioIndex, _hash }) {
    const session = get().getCurrentSession()
    if (!session) return

    const transcript = (audio.transcript || '').trim()
    if (!transcript) {
      // 如果没有转录文本，标记为error
      const runId = generateId()
      const run = {
        id: runId,
        targetType,
        targetKey,
        modality: 'audio',
        audioIndex,
        _hash,
        status: 'error',
        error: 'No transcript available',
        progress: 0,
        buffer: '',
        resultJson: null,
        createdAt: Date.now(),
        controller: null,
      }
      get()._appendInfonRun(session.id, run)
      return
    }

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
      modality: 'audio',
      audioIndex,
      _hash,
      status: 'running',
      progress: 0,
      buffer: '',
      resultJson: null,
      createdAt: Date.now(),
      controller: null,
    }
    get()._appendInfonRun(session.id, run)

    // 使用用户配置的信息元提取模型（中文注释）：优先使用配置，回退到 DeepSeek
    const configuredModel = get().infonExtractionModel || 'deepseek-chat'
    const provider = get().customProviders?.[configuredModel]
    
    // 如果有provider（自定义API模型），使用provider配置；否则使用本地baseUrl（Ollama）
    const baseUrl = provider ? provider.baseUrl : get().baseUrl
    const headers = { 'Content-Type': 'application/json' }
    if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`

    // 根据模型类型确定 max_tokens（omni 系列限制为 2048）
    const isOmniModel = configuredModel.toLowerCase().includes('omni')
    const maxTokens = isOmniModel ? 2000 : 4096

    const nowISO = new Date().toISOString()
    const systemPrompt = buildInfonSystemPrompt(['audio'], nowISO, { currentRound, existingInfons })
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Extract Situation Theory infons as a strict single JSON object. Input audio transcript:\n\n${transcript}` },
    ]

    const controller = new AbortController()
    get()._updateInfonRun(session.id, runId, (r) => ({ ...r, controller }))

    try {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          ...headers,
          'Connection': 'keep-alive'
        },
        body: JSON.stringify({ 
          model: configuredModel, 
          messages, 
          temperature: 0,
          stream: true,
          max_tokens: maxTokens, // 根据模型限制输出长度
          top_p: 0.95, // 核采样
          frequency_penalty: 0.0,
          presence_penalty: 0.0,
        }),
        signal: controller.signal,
        keepalive: true // 启用连接复用
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

      // Debounce配置：减少解析频率
      let parseTimer = null
      let lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 200

      await streamOpenAIResponse(reader, async ({ content, finish }) => {
        if (typeof content === 'string' && content.length) {
          // 更新buffer
          const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
          const buffer = (currentRun?.buffer || '') + content
          get()._updateInfonRun(session.id, runId, (r) => ({ ...r, buffer }))
          
          // Debounce解析逻辑
          const performParsing = () => {
            const currentRun = get().infonSessions?.[session.id]?.runs.find(x => x.id === runId)
            if (!currentRun) return
            
            const buffer = currentRun.buffer || ''
            const parserState = get().infonParsers?.[runId] || null
            const { state: newState, yielded } = incrementalExtractInfons(buffer, parserState)
            
            set(state => ({
              infonParsers: {
                ...state.infonParsers,
                [runId]: newState
              }
            }))
            
            if (yielded && yielded.length > 0) {
              get()._updateInfonRun(session.id, runId, (r) => {
                const currentInfons = r.resultJson?.infons || []
                const updatedInfons = [...currentInfons]
                
                yielded.forEach(newInfon => {
                  const objIndex = newInfon._objIndex
                  
                  if (objIndex !== undefined) {
                    const existingIndex = updatedInfons.findIndex(inf => inf._objIndex === objIndex)
                    if (existingIndex >= 0) {
                      updatedInfons[existingIndex] = {
                        ...updatedInfons[existingIndex],
                        ...newInfon
                      }
                    } else {
                      updatedInfons.push(newInfon)
                    }
                  } else {
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
            
            lastParseTime = Date.now()
          }
          
          // Debounce策略
          const now = Date.now()
          const timeSinceLastParse = now - lastParseTime
          
          if (parseTimer) clearTimeout(parseTimer)
          
          if (timeSinceLastParse >= PARSE_DEBOUNCE_MS) {
            performParsing()
          } else {
            parseTimer = setTimeout(performParsing, PARSE_DEBOUNCE_MS)
          }
        }
        if (finish) {
          if (parseTimer) clearTimeout(parseTimer)
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
              defaultModality: 'audio',
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
  async sendMessage(text, audioDataArray = []) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 如历史对话包含图片或音频，则改走多模态路径
    const hasHistoricalImages = (session.messages || []).some(m => Array.isArray(m.images) && m.images.length > 0)
    const hasHistoricalAudios = (session.messages || []).some(m => Array.isArray(m.audios) && m.audios.length > 0)
    const hasAudios = Array.isArray(audioDataArray) && audioDataArray.length > 0
    if (hasHistoricalImages || hasHistoricalAudios || hasAudios) {
      // 委托到多模态路径（支持本地模型和 API 模型）
      return await get().sendMessageWithImages(text, [], audioDataArray)
    }

    // 写入用户消息
    const userMsgId = generateId()
    const audios = Array.isArray(audioDataArray) ? audioDataArray.filter(Boolean) : []
    
    // 构建消息内容：文本 + 带标签的音频转写
    let messageContent = text
    if (audios.length > 0) {
      const audioTranscripts = audios
        .filter(audio => audio.transcript && audio.transcript.trim())
        .map(audio => `<audio>${audio.transcript.trim()}</audio>`)
        .join('\n')
      if (audioTranscripts) {
        messageContent = [text, audioTranscripts].filter(Boolean).join('\n\n')
      }
    }
    
    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: messageContent,
      audios: audios, // 保留原始音频数据用于UI显示
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

        // 根据模型类型确定 max_tokens（omni 系列限制为 2048）
        const currentModelName = get().model.toLowerCase()
        const isOmniModel = currentModelName.includes('omni')
        const maxTokens = isOmniModel ? 2000 : 4096

        const res = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: {
            ...headers,
            'Connection': 'keep-alive'
          },
          body: JSON.stringify({
            model: get().model,
            messages: payloadMessages,
            temperature: 0.7,
            stream: true,
            max_tokens: maxTokens, // 根据模型限制输出长度
            top_p: 0.9, // 核采样
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
          }),
          signal: controller.signal,
          keepalive: true // 启用连接复用
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
  async sendMessageWithDeepSeekOCR(text, selectedCommands, selectedFiles, resolution = 'gundam') {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 写入用户消息：包含命令标签和文件
    const userMsgId = generateId()

    // 只保存文件元数据，不保存文件内容（避免 localStorage 超限）
    const fileMetadata = selectedFiles.map((fileData) => ({
      id: fileData.id,
      name: fileData.name,
      size: fileData.size,
      type: fileData.type
      // 不再保存 dataUrl，避免存储空间超限
    }))

    // 在内存中保存 File 对象的引用，用于预览（不持久化）
    const fileObjectsMap = {}
    selectedFiles.forEach((fileData) => {
      if (fileData.file) {
        fileObjectsMap[fileData.id] = fileData.file
      }
    })

    // 将 File 对象映射存储到 store 的非持久化字段中
    set(state => ({
      ocrFileObjects: {
        ...state.ocrFileObjects,
        [session.id]: {
          ...state.ocrFileObjects?.[session.id],
          [userMsgId]: fileObjectsMap
        }
      }
    }))

    // 异步保存文件到 IndexedDB（持久化存储）
    ;(async () => {
      try {
        const filesToSave = selectedFiles
          .filter(f => f.file)
          .map(f => ({ id: f.id, file: f.file }))
        
        if (filesToSave.length > 0) {
          await saveFiles(session.id, userMsgId, filesToSave)
          console.log('[sendMessageWithDeepSeekOCR] 文件已保存到 IndexedDB', { count: filesToSave.length })
        }
      } catch (error) {
        console.error('[sendMessageWithDeepSeekOCR] 保存文件到 IndexedDB 失败:', error)
      }
    })()

    // 消息内容只保留文本，不包含标签文本
    const messageContent = text || ''

    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: messageContent,
      files: fileMetadata, // 只保存文件元数据用于UI显示
      commands: selectedCommands, // 保存命令数据
      createdAt: Date.now(),
    })

    // 检查模型配置
    const currentModel = get().model
    const customProviders = get().customProviders
    const provider = customProviders?.[currentModel]

    if (!provider) {
      throw new Error(`模型 ${currentModel} 的配置不存在`)
    }

    console.log('[sendMessageWithDeepSeekOCR] 开始处理 OCR 请求', {
      commands: selectedCommands.length,
      files: selectedFiles.length,
      text: text,
      resolution: resolution
    })

    // 立即创建助手消息（显示加载状态）
    const assistantMsgId = generateId()
    get()._appendMessage(session.id, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      streaming: true,
      phase: 'thinking',
      createdAt: Date.now(),
    })

    // 设置消息状态为处理中
    get()._updateMessage(session.id, userMsgId, (m) => ({
      ...m,
      ocrStatus: 'processing',
      ocrProgress: 0
    }))

    try {
      // 使用流式 API 处理每个文件
      const results = []
      let currentContent = '' // 累积的内容
      
      for (let i = 0; i < selectedFiles.length; i++) {
        const fileData = selectedFiles[i]
        const file = fileData.file // 获取原始 File 对象
        const uploadedFilename = fileData.serverFilename // 已上传的文件名
        const command = selectedCommands[i] || selectedCommands[0] // 如果命令数量少于文件，使用第一个命令

        console.log(`[sendMessageWithDeepSeekOCR] 流式处理文件 ${i + 1}/${selectedFiles.length}: ${fileData.name}`)
        
        // 如果文件还没有上传完成，等待上传
        if (fileData.uploadStatus === 'uploading') {
          console.log(`[sendMessageWithDeepSeekOCR] 文件还在上传中，这不应该发生`)
          throw new Error('文件还在上传中')
        }
        
        if (fileData.uploadStatus === 'error') {
          throw new Error(`文件上传失败: ${fileData.uploadError}`)
        }

        // 如果是多文件，添加文件分隔符
        if (selectedFiles.length > 1 && i > 0) {
          currentContent += '\n\n---\n\n'
        }
        
        if (selectedFiles.length > 1) {
          const fileHeader = `文件 ${i + 1} (${fileData.name}) - ${command.label}:\n`
          currentContent += fileHeader
          // 立即显示文件头
          get()._updateMessage(session.id, assistantMsgId, (m) => ({
            ...m,
            content: currentContent
          }))
        }

        try {
          // 使用流式 API，传递已上传的文件名
          const result = await callDeepseekOcrStream({
            file: uploadedFilename ? null : file,  // 如果有上传的文件名就不传file
            uploadedFilename: uploadedFilename,  // 传递已上传的文件名
            commandId: command.id,
            provider: provider,
            resolution: resolution,
            question: text || undefined,
            onProgress: ({ value, stage }) => {
              const progress = Math.round(((i + (value / 100)) / selectedFiles.length) * 100)
              get()._updateMessage(session.id, userMsgId, (m) => ({
                ...m,
                ocrStatus: 'processing',
                ocrProgress: progress,
                ocrStage: stage
              }))
            },
            onContent: (chunk) => {
              // 流式接收内容块，实时更新助手消息
              currentContent += chunk
              get()._updateMessage(session.id, assistantMsgId, (m) => ({
                ...m,
                content: currentContent,
                streaming: true
              }))
            }
          })

          results.push({
            fileName: file.name,
            command: command.label,
            result: result
          })

          console.log(`[sendMessageWithDeepSeekOCR] 文件 ${file.name} 流式处理完成`)
        } catch (error) {
          console.error(`[sendMessageWithDeepSeekOCR] 文件 ${file.name} 处理失败:`, error)
          const errorMsg = `处理出错：${error.message}`
          currentContent += errorMsg
          
          // 更新显示错误
          get()._updateMessage(session.id, assistantMsgId, (m) => ({
            ...m,
            content: currentContent
          }))
          
          results.push({
            fileName: file.name,
            command: command.label,
            error: error.message
          })
        }
      }

      // 标记流式传输完成
      get()._updateMessage(session.id, assistantMsgId, (m) => ({
        ...m,
        streaming: false,
        phase: 'done',
      }))

      // 更新用户消息状态为完成
      get()._updateMessage(session.id, userMsgId, (m) => ({
        ...m,
        ocrStatus: 'completed',
        ocrProgress: 100
      }))

      console.log('[sendMessageWithDeepSeekOCR] OCR 处理完成')
      return userMsgId

    } catch (error) {
      console.error('[sendMessageWithDeepSeekOCR] OCR 处理失败:', error)

      // 更新助手消息显示错误
      get()._updateMessage(session.id, assistantMsgId, (m) => ({
        ...m,
        content: `OCR 处理失败：${error.message}`,
        streaming: false,
        phase: 'done',
        error: error.message,
      }))

      // 更新用户消息状态为失败
      get()._updateMessage(session.id, userMsgId, (m) => ({
        ...m,
        ocrStatus: 'error',
        ocrError: error.message
      }))

      throw error
    }
  },

  async sendMessageWithImages(text, imageDataUrls, audioDataArray = [], imageAnalysisMap = {}) {
    const state = get()
    state._ensureCurrentSession()
    const session = get().getCurrentSession()
    if (!session) return

    // 写入用户消息：包含图片预览（data URL）和音频数据
    const userMsgId = generateId()
    const imgs = Array.isArray(imageDataUrls) ? imageDataUrls.filter(Boolean) : []
    const audios = Array.isArray(audioDataArray) ? audioDataArray.filter(Boolean) : []
    
    // 构建消息内容：文本 + 带标签的音频转写
    let messageContent = text
    if (audios.length > 0) {
      const audioTranscripts = audios
        .filter(audio => audio.transcript && audio.transcript.trim())
        .map(audio => `<audio>${audio.transcript.trim()}</audio>`)
        .join('\n')
      if (audioTranscripts) {
        messageContent = [text, audioTranscripts].filter(Boolean).join('\n\n')
      }
    }
    
    get()._appendMessage(session.id, {
      id: userMsgId,
      role: 'user',
      content: messageContent,
      images: imgs,
      audios: audios, // 保留原始音频数据用于UI显示
      imageAnalysis: imageAnalysisMap, // 保存图片分析数据（直接推理模式）
      createdAt: Date.now(),
    })

    // 检查模型是否支持图片
    const currentModel = get().model
    const customProviders = get().customProviders
    const modalities = getModelModalities(currentModel, customProviders)
    
    console.log('[sendMessageWithImages] 当前模型:', currentModel, '支持图片:', modalities.image, 'API模型:', !!customProviders?.[currentModel])
    
    if (!modalities.image) {
      // 模型不支持图片，返回错误
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
    
    const provider = get().customProviders?.[get().model]
    console.log('[sendMessageWithImages] 使用', provider ? 'API 模型' : '本地 Ollama 模型', '处理图片')

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

    const controller = new AbortController()
    set({ isGenerating: true, abortController: controller })

    // 如果是 API 模型，使用 OpenAI Vision API 格式
    if (provider) {
      ;(async () => {
        try {
          const sessionMsgs = get().getCurrentSession().messages
          
          // 过滤掉刚创建的空 assistant 消息（避免发送空消息到 API）
          const filteredMsgs = sessionMsgs.filter(m => {
            // 排除当前正在生成的助手消息
            if (m.id === assistantMsgId) return false
            // 排除空的 assistant 消息
            if (m.role === 'assistant' && (!m.content || m.content.trim() === '')) return false
            return true
          })
          
          // 转换消息为 OpenAI Vision API 格式
          const payloadMessages = filteredMsgs.map((m) => {
            // 如果消息有图片，使用 content 数组格式
            if (m.role === 'user' && Array.isArray(m.images) && m.images.length > 0) {
              const contentArray = []
              
              // 添加文本内容
              if (m.content && m.content.trim()) {
                contentArray.push({
                  type: 'text',
                  text: m.content
                })
              }
              
              // 添加图片
              m.images.forEach(img => {
                contentArray.push({
                  type: 'image_url',
                  image_url: {
                    url: img // 保持 data:image/... 格式
                  }
                })
              })
              
              return { role: m.role, content: contentArray }
            }
            
            // 普通消息
            return { role: m.role, content: m.content }
          })

          const baseUrl = provider.baseUrl
          const headers = { 'Content-Type': 'application/json' }
          if (provider.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`

          // 检查是否有图片消息
          const hasImages = filteredMsgs.some(m => Array.isArray(m.images) && m.images.length > 0)
          
          // 判断是否使用流式响应（根据模型特性）
          const currentModelName = get().model.toLowerCase()
          const isOmniModel = currentModelName.includes('omni') // omni 系列必须使用流式
          const isVLModel = currentModelName.includes('vl') && !isOmniModel // vl 系列不能使用流式（除了 omni）
          
          // 决定是否使用流式
          let useStreaming = true // 默认使用流式
          if (hasImages && isVLModel) {
            useStreaming = false // vl 模型处理图片时不能使用流式
          } else if (hasImages && isOmniModel) {
            useStreaming = true // omni 模型必须使用流式
          } else if (!hasImages) {
            useStreaming = true // 纯文本消息使用流式
          }
          
          // 根据模型类型确定 max_tokens
          const maxTokens = isOmniModel ? 2000 : 4096 // omni 系列限制为 2048，使用 2000 安全值
          
          // 构建请求体
          const requestBody = (hasImages && isVLModel) ? {
            // vl 模型处理图片：使用简化配置（无 stream）
            model: get().model,
            messages: payloadMessages,
            temperature: 0.3,
            max_tokens: 2000,
          } : {
            // 其他情况：使用流式配置
            model: get().model,
            messages: payloadMessages,
            temperature: 0.7,
            stream: true,
            max_tokens: maxTokens,
            top_p: 0.9,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
          }
          
          console.log('[sendMessageWithImages API] 请求地址:', `${baseUrl}/chat/completions`)
          console.log('[sendMessageWithImages API] 模型类型:', { 
            name: get().model, 
            isVL: isVLModel, 
            isOmni: isOmniModel,
            hasImages,
            useStreaming,
            maxTokens
          })
          console.log('[sendMessageWithImages API] 原始消息数:', sessionMsgs.length, '过滤后:', filteredMsgs.length, '→ payload:', payloadMessages.length)
          console.log('[sendMessageWithImages API] 请求配置:', (hasImages && isVLModel) ? 'vl模型简化配置（无stream）' : `omni/文本流式配置 (max_tokens: ${maxTokens})`)
          console.log('[sendMessageWithImages API] 最后一条消息:', JSON.stringify(payloadMessages[payloadMessages.length - 1], null, 2).substring(0, 500))

          const res = await fetch(`${baseUrl}/chat/completions`, {
            method: 'POST',
            headers,  // 使用简单的 headers，不添加额外的
            body: JSON.stringify(requestBody),
            signal: controller.signal,
          })

          if (!res.ok) {
            const textErr = await res.text().catch(() => '')
            console.error('[sendMessageWithImages API] 请求失败:', res.status, res.statusText, textErr)
            get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: textErr || 'Request failed', content: m.content }))
            set({ isGenerating: false, abortController: null })
            return
          }

          // 如果使用非流式响应（vl 模型处理图片时）
          if (!useStreaming && hasImages && isVLModel) {
            const result = await res.json()
            const content = result.choices?.[0]?.message?.content || ''
            
            console.log('[sendMessageWithImages API] vl模型非流式响应接收完成，内容长度:', content.length)
            
            // 更新消息内容
            get()._updateMessage(session.id, assistantMsgId, (m) => ({ 
              ...m, 
              content: content,
              streaming: false, 
              phase: 'done' 
            }))
            set({ isGenerating: false, abortController: null })
            return
          }

          // 流式响应处理（omni 模型或纯文本消息）
          console.log('[sendMessageWithImages API] 使用流式响应处理')
          
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
            }
          })
        } catch (err) {
          console.error('[sendMessageWithImages API] 捕获到错误:', err)
          console.error('[sendMessageWithImages API] 错误详情:', {
            name: err?.name,
            message: err?.message,
            stack: err?.stack
          })
          const msg = (err && (err.name === 'AbortError')) ? 'Aborted' : `Network error: ${err?.message || '未知错误'}`
          get()._updateMessage(session.id, assistantMsgId, (m) => ({ ...m, streaming: false, error: msg }))
        } finally {
          set({ isGenerating: false, abortController: null })
        }
      })()
      
      return userMsgId
    }

    // 本地 Ollama 模型：使用 /api/chat 格式
    // 将历史消息转换为 Ollama /api/chat 的 messages：携带 images 时去掉 data: 前缀
    const stripDataUrl = (s) => {
      if (typeof s !== 'string') return s
      const i = s.indexOf(',')
      if (i >= 0 && s.slice(0, i).includes('base64')) return s.slice(i + 1)
      return s
    }

    const sessionMsgs = get().getCurrentSession().messages
    
    // 过滤掉刚创建的空 assistant 消息
    const filteredMsgs = sessionMsgs.filter(m => {
      if (m.id === assistantMsgId) return false
      if (m.role === 'assistant' && (!m.content || m.content.trim() === '')) return false
      return true
    })
    
    let lastImageIdx = -1
    for (let i = filteredMsgs.length - 1; i >= 0; i--) {
      const m = filteredMsgs[i]
      if (Array.isArray(m.images) && m.images.length > 0 && m.role === 'user') { lastImageIdx = i; break }
    }
    const history = filteredMsgs.map((m, idx) => {
      const o = { role: m.role, content: m.content }
      if (idx === lastImageIdx && Array.isArray(m.images) && m.images.length) {
        o.images = m.images.map(stripDataUrl)
      }
      return o
    })

    // 计算 /api 基址：将 baseUrl 的 /v1 替换为 /api
    const apiBase = (get().baseUrl || '').replace(/\/?v1\/?$/, '/api')

    console.log('[sendMessageWithImages Ollama] 请求地址:', `${apiBase}/chat`)
    console.log('[sendMessageWithImages Ollama] 原始消息数:', sessionMsgs.length, '过滤后:', filteredMsgs.length, '→ history:', history.length)

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
        console.error('[sendMessageWithImages Ollama] 捕获到错误:', err)
        console.error('[sendMessageWithImages Ollama] 错误详情:', {
          name: err?.name,
          message: err?.message,
          stack: err?.stack
        })
        const msg = (err && (err.name === 'AbortError')) ? 'Aborted' : `Network error: ${err?.message || '未知错误'}`
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
  async startPrivacyInference(editingMessageId = null) {
    const session = get().getCurrentSession()
    if (!session) return
    
    const { selectedLaw, infonSessions, inferenceMode } = get()
    if (!selectedLaw || !selectedLaw.data) {
      console.warn('No law selected for inference')
      return
    }
    
    // 如果正在编辑消息（editingMessageId != null），清空sessionKeywords
    // 这样可以确保修改后的推理不会包含修改前的旧关键词
    if (inferenceMode === 'direct' && editingMessageId) {
      const sessionKeywords = get().sessionKeywords || {}
      if (sessionKeywords[session.id]) {
        const updatedKeywords = { ...sessionKeywords }
        delete updatedKeywords[session.id]
        set({ sessionKeywords: updatedKeywords })
        console.log('[Privacy Inference] 编辑模式：清空旧关键词')
      }
    }
    
    let allInfons = []
    let directInput = null
    
    // 检查推断模式（中文注释）：extract（提取信息元）或 direct（直接推断）
    if (inferenceMode === 'direct') {
      // 直接推断模式：收集用户输入内容（包括pending和已发送的消息，以及音频转写文本）
      const textParts = []
      
      // 辅助函数：从消息内容中提取文本
      const extractTextFromContent = (content) => {
        if (!content) return ''
        if (typeof content === 'string') return content
        if (Array.isArray(content)) {
          // 多模态内容：提取所有text类型的部分
          return content
            .filter(part => part && part.type === 'text')
            .map(part => part.text || '')
            .join('\n')
        }
        return ''
      }
      
      // 辅助函数：从消息中提取音频转写文本，并加上<audio></audio>标签
      const extractAudioTranscripts = (message) => {
        const audios = message.audios || []
        return audios
          .filter(audio => audio && audio.transcript && audio.transcript.trim())
          .map(audio => `<audio>${audio.transcript.trim()}</audio>`)
          .join('\n')
      }
      
      // 辅助函数：从消息中提取图片分析文本，并加上<img></img>标签
      const extractImageAnalysis = (message) => {
        const imageAnalysisMap = message.imageAnalysis || {}
        const imageUrls = message.images || []
        return imageUrls
          .map(url => imageAnalysisMap[url])
          .filter(analysis => analysis && analysis.trim())
          .map(analysis => `<img>${analysis.trim()}</img>`)
          .join('\n')
      }
      
      // 1. 获取所有已发送的用户消息（排除正在编辑的消息）
      const userMessages = (session.messages || [])
        .filter(msg => msg.role === 'user' && msg.id !== editingMessageId)
      userMessages.forEach(msg => {
        const text = extractTextFromContent(msg.content)
        const audioText = extractAudioTranscripts(msg)
        const imageText = extractImageAnalysis(msg)
        
        if (text) textParts.push(text)
        if (audioText) textParts.push(audioText)
        if (imageText) textParts.push(imageText)
      })
      
      // 2. 获取pending输入（如果有）
      const pendingInput = get().pendingUserInput || ''
      if (pendingInput && pendingInput.trim()) {
        textParts.push(pendingInput.trim())
      }
      
      // 3. 获取pending音频（如果有）
      const pendingAudios = get().pendingAudios || []
      pendingAudios.forEach(audio => {
        if (audio && audio.transcript && audio.transcript.trim()) {
          textParts.push(`<audio>${audio.transcript.trim()}</audio>`)
        }
      })
      
      // 4. 获取pending图片（如果有）
      const pendingImages = get().pendingImages || []
      pendingImages.forEach(image => {
        if (image && image.analysis && image.analysis.trim()) {
          textParts.push(`<img>${image.analysis.trim()}</img>`)
        }
      })
      
      // 合并所有文本
      directInput = textParts.filter(Boolean).join('\n\n')
      
      if (!directInput || directInput.trim().length === 0) {
        console.warn('[Privacy Inference] 直接推断模式：无用户输入内容')
        return
      }
      
      const audioCount = userMessages.reduce((sum, msg) => sum + (msg.audios?.length || 0), 0) + pendingAudios.length
      // 统计图片分析数量：已发送消息中有 imageAnalysis 的图片 + pending 图片
      const sentImagesWithAnalysis = userMessages.reduce((sum, msg) => {
        const imageAnalysisMap = msg.imageAnalysis || {}
        const images = msg.images || []
        const analysisCount = images.filter(url => imageAnalysisMap[url] && imageAnalysisMap[url].trim()).length
        return sum + analysisCount
      }, 0)
      const imageCount = sentImagesWithAnalysis + pendingImages.length
      console.log(`[Privacy Inference] 直接推断模式，用户输入长度: ${directInput.length} 字符，来源：${userMessages.length}条已发送消息 + ${pendingInput ? '1条pending输入' : '0条pending输入'} + ${audioCount}条音频转写 + ${imageCount}条图片分析`)
    } else {
      // 提取信息元模式：获取当前会话的所有信息元
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
      allInfons = allRawInfons.filter(infon => {
        return infon.iid && !supersededIids.has(infon.iid)
      })
      
      if (allInfons.length === 0) {
        console.warn('No infons available for inference')
        return
      }
      
      console.log(`[Privacy Inference] 提取信息元模式，使用 ${allInfons.length} 个信息元进行推理`)
    }
    
    // 初始化推理状态（中文注释）：记录当前选中的法律key，用于匹配高亮
    // 保存之前的推理结果（如果存在），用于中止后恢复
    const previousInference = get().privacyInferences?.[session.id]
    const previousRisks = previousInference?.status === 'done' ? (previousInference.risks || []) : []
    
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
          previousRisks: previousRisks, // 保存之前的结果，用于中止时恢复
          createdAt: Date.now(),
          updatedAt: Date.now(),
          // 直接推理模式：创建临时关键词集合，用于累积新关键词
          tempKeywords: inferenceMode === 'direct' ? new Set() : undefined
        }
      },
      // 重置隐私推理解析器状态，确保流式增量从头开始
      privacyParsers: {
        ...(state.privacyParsers || {}),
        [session.id]: null
      }
    }))
    
    try {
      // 使用用户配置的隐私推理模型（中文注释）：根据推断模式选择不同的模型
      const inferenceMode = get().inferenceMode
      let configuredModel
      if (inferenceMode === 'direct') {
        configuredModel = get().directInferenceModel || 'deepseek-chat'
      } else {
        configuredModel = get().infonPrivacyInferenceModel || 'deepseek-chat'
      }
      const provider = get().customProviders?.[configuredModel]
      
      // 如果有provider（自定义API模型），使用provider配置；否则使用本地baseUrl（Ollama）
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const apiKey = provider?.apiKey || ''
      
      // 根据模型类型确定 max_tokens（omni 系列限制为 2048）
      const isOmniModel = configuredModel.toLowerCase().includes('omni')
      const maxTokens = isOmniModel ? 2000 : 4096
      
      // 构建推理提示词（中文注释）：根据模式传递不同的参数
      const { fillPromptTemplate } = await import('./templates/inference.js')
      
      // 获取历史关键词（用于上下文补充）
      const historicalKeywords = get().sessionKeywords?.[session.id]
      const keywordsArray = historicalKeywords instanceof Set ? Array.from(historicalKeywords) : []
      
      const prompt = inferenceMode === 'direct' 
        ? fillPromptTemplate([], selectedLaw.data, directInput, keywordsArray)  // 直接推断模式：传递用户输入和历史关键词
        : fillPromptTemplate(allInfons, selectedLaw.data, null, [])  // 提取信息元模式：传递信息元列表
      
      console.log(`[Privacy Inference] 发起推理请求到 ${apiUrl}，使用模型: ${configuredModel}`)

      console.log(`[Privacy Inference] Prompt 预览:`, prompt)
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Connection': 'keep-alive',
          ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {})
        },
        body: JSON.stringify({
          model: configuredModel,
          messages: [{ role: 'user', content: prompt }],
          stream: true,
          temperature: 0.5, // 适中温度以平衡创造性和准确性
          max_tokens: maxTokens, // 根据模型限制最大输出tokens
          top_p: 0.9, // 核采样，提升生成速度和质量
          frequency_penalty: 0.0,
          presence_penalty: 0.0,
        }),
        signal: abortController.signal,
        keepalive: true // 启用连接复用
      })
      
      console.log(`[Privacy Inference] API响应状态: ${response.status}`)
      
      if (!response.ok) {
        const errorText = await response.text().catch(() => '')
        console.error(`[Privacy Inference] API错误: ${response.status} - ${errorText}`)
        throw new Error(`API error: ${response.status} - ${errorText}`)
      }
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      
      // Debounce配置：减少解析频率，提升性能
      let parseTimer = null
      let lastParseTime = 0
      const PARSE_DEBOUNCE_MS = 200 // 100ms debounce间隔
      
      // 定义解析函数（复用逻辑）
      const performParsing = async () => {
        // 清理 buffer：移除 <think> 标签后再解析
        let cleanedBuffer = buffer
        cleanedBuffer = cleanedBuffer.replace(/<think>[\s\S]*?<\/think>/gi, '')
        
        // 使用增量解析器逐个提取风险项
        const { incrementalExtractRisks } = await import('./templates/inference.js')
        const parserState = get().privacyParsers?.[session.id] || null
        const { state: newState, yielded } = incrementalExtractRisks(cleanedBuffer, parserState)
        
        // 更新解析器状态
        set(state => ({
          privacyParsers: {
            ...state.privacyParsers,
            [session.id]: newState
          }
        }))
        
        // 如果有新的风险项被解析出来，立即添加到结果中
        if (yielded && yielded.length > 0) {
          // 在直接推理模式下，立即累积新解析出来的关键词（流式高亮）
          if (inferenceMode === 'direct') {
            const currentInference = get().privacyInferences?.[session.id]
            const tempKeywords = currentInference?.tempKeywords || new Set()
            
            yielded.forEach(newRisk => {
              const usedInfons = newRisk.used_infons || []
              usedInfons.forEach(keyword => {
                if (typeof keyword === 'string' && keyword.trim()) {
                  tempKeywords.add(keyword.trim())
                }
              })
            })
            
            // 更新临时关键词集合（不影响当前显示的关键词）
            set(state => ({
              privacyInferences: {
                ...state.privacyInferences,
                [session.id]: {
                  ...state.privacyInferences[session.id],
                  tempKeywords: tempKeywords
                }
              }
            }))
            
            console.log(`[Privacy Inference] 流式累积临时关键词: ${tempKeywords.size} 个`)
          }
          
          set(state => {
            const currentRisks = state.privacyInferences?.[session.id]?.risks || []
            const updatedRisks = [...currentRisks]
            
            yielded.forEach(newRisk => {
              const objIndex = newRisk._objIndex
              
              if (objIndex !== undefined) {
                const existingIndex = updatedRisks.findIndex(r => r._objIndex === objIndex)
                
                if (existingIndex >= 0) {
                  updatedRisks[existingIndex] = {
                    ...updatedRisks[existingIndex],
                    ...newRisk
                  }
                } else {
                  updatedRisks.push(newRisk)
                }
              } else {
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
        
        lastParseTime = Date.now()
      }
      
      // 流式接收并逐个解析风险项（使用debounce优化）
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
            const contentDelta = (
              delta?.content ||
              delta?.reasoning_content ||
              delta?.reasoning ||
              delta?.thoughts ||
              delta?.inner_thoughts ||
              ''
            )
            
            // 首次收到内容时记录日志
            if (contentDelta && buffer.length === 0) {
              console.log('[Privacy Inference] 开始接收流式内容')
            }
            
            if (contentDelta) {
              buffer += contentDelta
              
              // 使用debounce策略：只在间隔足够长或首次接收时解析
              const now = Date.now()
              const timeSinceLastParse = now - lastParseTime
              
              if (parseTimer) clearTimeout(parseTimer)
              
              // 如果距离上次解析超过debounce间隔，立即解析
              if (timeSinceLastParse >= PARSE_DEBOUNCE_MS) {
                await performParsing()
              } else {
                // 否则设置定时器，延迟解析
                parseTimer = setTimeout(async () => {
                  await performParsing()
                }, PARSE_DEBOUNCE_MS)
              }
            }
          } catch (err) {
            // 忽略解析错误
          }
        }
      }
      
      // 流结束后，清除定时器并执行最后一次解析
      if (parseTimer) clearTimeout(parseTimer)
      await performParsing()
      
      console.log(`[Privacy Inference] 流式接收完成，buffer长度: ${buffer.length}`)
      console.log(`[Privacy Inference] Buffer内容预览:`, buffer)
      
      // 尝试清理buffer：移除模型的思考过程和markdown标记
      let cleanBuffer = buffer
      
      // 1. 移除 <think>...</think> 标签及其内容（某些模型如 DeepSeek 会输出思考过程）
      cleanBuffer = cleanBuffer.replace(/<think>[\s\S]*?<\/think>/gi, '')
      
      // 2. 移除 markdown 代码块标记
      cleanBuffer = cleanBuffer.replace(/^```json\s*/i, '').replace(/```\s*$/i, '')
      cleanBuffer = cleanBuffer.replace(/^```\s*/i, '').replace(/```\s*$/i, '')
      
      // 3. 查找第一个 { 和最后一个 }，提取JSON部分
      const firstBrace = cleanBuffer.indexOf('{')
      const lastBrace = cleanBuffer.lastIndexOf('}')
      if (firstBrace >= 0 && lastBrace > firstBrace) {
        cleanBuffer = cleanBuffer.slice(firstBrace, lastBrace + 1)
        console.log(`[Privacy Inference] 清理后buffer长度: ${cleanBuffer.length}`)
        
        // 尝试直接解析完整JSON（如果已经接收完成）
        if (cleanBuffer.length > 0) {
          try {
            const parsed = JSON.parse(cleanBuffer)
            if (parsed.risks && Array.isArray(parsed.risks)) {
              console.log(`[Privacy Inference] 成功解析完整JSON，风险数: ${parsed.risks.length}`)
              
              // 累积关键词到 sessionKeywords（直接推理模式下）
              if (inferenceMode === 'direct' && parsed.risks.length > 0) {
                const existingKeywords = get().sessionKeywords?.[session.id] || new Set()
                const newKeywords = new Set(existingKeywords)
                
                parsed.risks.forEach(risk => {
                  const usedInfons = risk.used_infons || []
                  usedInfons.forEach(keyword => {
                    if (typeof keyword === 'string' && keyword.trim()) {
                      newKeywords.add(keyword.trim())
                    }
                  })
                })
                
                console.log(`[Privacy Inference] 累积关键词: ${existingKeywords.size} -> ${newKeywords.size} (新增 ${newKeywords.size - existingKeywords.size})`)
                
                set(state => ({
                  sessionKeywords: {
                    ...state.sessionKeywords,
                    [session.id]: newKeywords
                  }
                }))
              }
              
              // 直接推理模式：将临时关键词移动到正式关键词
              const currentInference = get().privacyInferences?.[session.id]
              const tempKeywords = currentInference?.tempKeywords
              if (inferenceMode === 'direct' && tempKeywords && tempKeywords.size > 0) {
                console.log(`[Privacy Inference] 推理完成，应用 ${tempKeywords.size} 个新关键词（替换旧关键词）`)
                set(s => ({
                  sessionKeywords: {
                    ...s.sessionKeywords,
                    [session.id]: tempKeywords
                  }
                }))
              }
              
              // 直接设置结果
              set(state => ({
                privacyInferences: {
                  ...state.privacyInferences,
                  [session.id]: {
                    ...state.privacyInferences[session.id],
                    status: 'done',
                    risks: parsed.risks,
                    buffer: cleanBuffer,
                    abortController: null,
                    tempKeywords: undefined, // 清除临时关键词
                    updatedAt: Date.now()
                  }
                }
              }))
              return
            }
          } catch (parseErr) {
            console.log(`[Privacy Inference] 完整JSON解析失败:`, parseErr.message)
          }
        }
      }
      
      // 检查是否真的有内容
      const currentState = get().privacyInferences?.[session.id]
      const hasRisks = currentState?.risks && currentState.risks.length > 0
      console.log(`[Privacy Inference] 解析器状态:`, get().privacyParsers?.[session.id])
      
      if (!hasRisks && buffer.length === 0) {
        console.warn('[Privacy Inference] 未收到任何内容，可能API调用失败')
        set(state => ({
          privacyInferences: {
            ...state.privacyInferences,
            [session.id]: {
              ...state.privacyInferences[session.id],
              status: 'error',
              error: 'No response received from API',
              abortController: null,
              updatedAt: Date.now()
            }
          }
        }))
        return
      }
      
      // 完成推理
      console.log(`[Privacy Inference] 推理完成，风险数: ${currentState?.risks?.length || 0}`)
      
      // 直接推理模式：应用临时关键词（替换旧关键词）
      if (inferenceMode === 'direct') {
        const currentInference = get().privacyInferences?.[session.id]
        const tempKeywords = currentInference?.tempKeywords
        
        if (tempKeywords && tempKeywords.size > 0) {
          console.log(`[Privacy Inference] 推理完成，应用 ${tempKeywords.size} 个新关键词（替换旧关键词）`)
          set(state => ({
            sessionKeywords: {
              ...state.sessionKeywords,
              [session.id]: tempKeywords
            }
          }))
        } else {
          console.log(`[Privacy Inference] 推理完成，但没有提取到关键词`)
        }
      }
      
      // 在直接推理模式下，保留 previousRisks 以便在输入清空时恢复
      const shouldKeepPreviousRisks = get().inferenceMode === 'direct'
      
      set(state => ({
        privacyInferences: {
          ...state.privacyInferences,
          [session.id]: {
            ...state.privacyInferences[session.id],
            status: 'done',
            abortController: null,
            previousRisks: shouldKeepPreviousRisks ? state.privacyInferences[session.id].previousRisks : undefined,
            tempKeywords: undefined, // 清除临时关键词
            updatedAt: Date.now()
          }
        }
      }))
      
    } catch (err) {
      if (err.name === 'AbortError') {
        // 推理被中止，恢复之前的结果
        const currentState = get().privacyInferences?.[session.id]
        const previousRisks = currentState?.previousRisks || []
        
        console.log(`[Privacy Inference] 推理被中止，恢复之前的 ${previousRisks.length} 个风险项`)
        
        // 在直接推理模式下，将恢复的结果作为新的previousRisks保留，以便下次输入清空时再次恢复
        const shouldKeepPreviousRisks = get().inferenceMode === 'direct'
        
        set(state => ({
          privacyInferences: {
            ...state.privacyInferences,
            [session.id]: {
              ...state.privacyInferences[session.id],
              status: previousRisks.length > 0 ? 'done' : 'aborted',
              risks: previousRisks, // 恢复之前的结果
              abortController: null,
              previousRisks: shouldKeepPreviousRisks ? previousRisks : undefined,
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
  
  // 清除当前推理结果并恢复到上一次推理结果（用于直接推理模式中输入被清空的情况）
  clearCurrentInferenceAndRestore() {
    const session = get().getCurrentSession()
    if (!session) return
    
    const currentInference = get().privacyInferences?.[session.id]
    if (!currentInference) return
    
    // 如果推理正在运行，中止它（会自动恢复到previousRisks）
    if (currentInference.status === 'running') {
      console.log('[Privacy Inference] 清除当前推理：中止正在运行的推理')
      if (currentInference.abortController) {
        try {
          currentInference.abortController.abort()
        } catch (_) {}
      }
      // abortController.abort() 会触发 AbortError，在 catch 块中会恢复 previousRisks
    } else if (currentInference.status === 'done') {
      // 如果推理已完成，检查是否有previousRisks可以恢复
      const previousRisks = currentInference.previousRisks || []
      
      console.log(`[Privacy Inference] 清除当前推理：恢复到上一次的 ${previousRisks.length} 个风险项`)
      
      set(state => ({
        privacyInferences: {
          ...state.privacyInferences,
          [session.id]: {
            ...state.privacyInferences[session.id],
            status: previousRisks.length > 0 ? 'done' : 'idle',
            risks: previousRisks,
            // 保留previousRisks，以便下次输入清空时再次恢复
            previousRisks: previousRisks.length > 0 ? previousRisks : undefined,
            buffer: '',
            updatedAt: Date.now()
          }
        }
      }))
      
      // 同时清除解析器状态
      set(state => ({
        privacyParsers: {
          ...state.privacyParsers,
          [session.id]: null
        }
      }))
    }
  },

  // ========== 隐私保护修改建议方法 ==========
  
  // 生成隐私保护修改建议
  async generateProtectionSuggestions(text, editingMessageId = null) {
    const session = get().getCurrentSession()
    if (!session) {
      console.warn('[Protection] 没有当前会话')
      return
    }
    
    if (!text || !text.trim()) {
      console.warn('[Protection] 文本为空')
      return
    }
    
    // 获取当前的隐私推理结果和信息元
    const currentInference = get().privacyInferences?.[session.id]
    if (!currentInference || currentInference.status !== 'done') {
      console.warn('[Protection] 隐私推理未完成')
      set(state => ({
        protectionSuggestions: {
          ...state.protectionSuggestions,
          [session.id]: {
            status: 'error',
            error: '请先完成隐私推理分析',
            suggestions: []
          }
        }
      }))
      return
    }
    
    const privacyRisks = currentInference.risks || []
    
    // 获取当前的信息元
    const runs = get().infonSessions?.[session.id]?.runs || []
    const allInfons = []
    const supersededIids = new Set()
    
    // 收集所有有效的信息元
    runs.forEach(run => {
      if (run.status === 'done' || run.status === 'running') {
        const infons = Array.isArray(run?.resultJson?.infons) ? run.resultJson.infons : []
        allInfons.push(...infons)
        infons.forEach(infon => {
          if (Array.isArray(infon._supersedes)) {
            infon._supersedes.forEach(oldIid => supersededIids.add(oldIid))
          }
        })
      }
    })
    
    // 过滤掉被取代的信息元
    const validInfons = allInfons.filter(infon => infon.iid && !supersededIids.has(infon.iid))
    
    console.log(`[Protection] 开始生成建议，风险数: ${privacyRisks.length}，信息元数: ${validInfons.length}`)
    
    // 初始化建议状态
    const abortController = new AbortController()
    set(state => ({
      protectionSuggestions: {
        ...state.protectionSuggestions,
        [session.id]: {
          status: 'running',
          suggestions: [],
          error: null,
          abortController
        }
      }
    }))
    
    try {
      // 使用配置的Privacy Protection Suggestions模型（必须是API key模型）
      const configuredModel = get().protectionSuggestionModel || 'deepseek-chat'
      const provider = get().customProviders?.[configuredModel]
      
      // 验证是否是API key模型
      if (!provider) {
        console.warn('[Protection] Privacy Protection Suggestions只能使用API key模型')
        set(state => ({
          protectionSuggestions: {
            ...state.protectionSuggestions,
            [session.id]: {
              status: 'error',
              error: 'Privacy Protection Suggestions requires an API key model. Please configure one in Settings.',
              suggestions: []
            }
          }
        }))
        return
      }
      
      const apiUrl = provider.baseUrl
      const apiKey = provider?.apiKey || ''
      
      // 确保 apiUrl 格式正确
      if (!apiUrl) {
        throw new Error('未配置API地址')
      }
      
      const normalizedUrl = new URL(apiUrl.replace(/\/$/, '') + '/chat/completions')
      const fullUrl = normalizedUrl.toString()
      
      // 构建提示词
      const { fillProtectionPrompt } = await import('./templates/protection.js')
      const prompt = fillProtectionPrompt(text, privacyRisks, validInfons)
      
      // 根据模型类型确定 max_tokens（omni 系列限制为 2048）
      const isOmniModel = configuredModel.toLowerCase().includes('omni')
      const maxTokens = isOmniModel ? 2000 : 4096
      
      console.log(`[Protection] 发起建议生成请求到 ${fullUrl}`)
      console.log(`[Protection] 使用模型: ${configuredModel}`)
      console.log(`[Protection] Prompt 长度: ${prompt.length} 字符`)
      
      try {
        const response = await fetch(fullUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {})
          },
          body: JSON.stringify({
            model: configuredModel,
            messages: [{ role: 'user', content: prompt }],
            stream: true, // 使用流式生成
            temperature: 0.7,
            max_tokens: maxTokens,
          }),
          signal: abortController.signal
        })
        
        console.log(`[Protection] API响应状态: ${response.status}`)
        
        if (!response.ok) {
          const errorText = await response.text().catch(() => '')
          console.error(`[Protection] API错误响应: ${response.status} - ${errorText}`)
          throw new Error(`API error: ${response.status} - ${errorText}`)
        }
        
        // 流式处理响应
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let parser = null
        const allSuggestions = new Map() // 使用Map来管理建议，key为_objIndex
        
        const { incrementalExtractSuggestions } = await import('./templates/protection.js')
        
        while (true) {
          const { value, done } = await reader.read()
          if (done) {
            console.log('[Protection] 流式生成完成')
            break
          }
          
          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split('\n')
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)
              if (data === '[DONE]') continue
              
              try {
                const parsed = JSON.parse(data)
                const content = parsed?.choices?.[0]?.delta?.content || ''
                if (content) {
                  buffer += content
                  
                  // 使用增量解析器
                  const result = incrementalExtractSuggestions(buffer, parser)
                  parser = result.state
                  
                  // 更新建议列表
                  for (const suggestion of result.yielded) {
                    const objIndex = suggestion._objIndex ?? 0
                    allSuggestions.set(objIndex, suggestion)
                  }
                  
                  // 实时更新状态
                  if (result.yielded.length > 0) {
                    const suggestionsArray = Array.from(allSuggestions.values()).sort((a, b) => {
                      const orderMap = { 'high_privacy': 0, 'balanced': 1, 'low_privacy': 2 }
                      return (orderMap[a.level] || 999) - (orderMap[b.level] || 999)
                    })
                    
                    set(state => ({
                      protectionSuggestions: {
                        ...state.protectionSuggestions,
                        [session.id]: {
                          ...state.protectionSuggestions?.[session.id],
                          status: 'running',
                          suggestions: suggestionsArray,
                          error: null,
                          abortController
                        }
                      }
                    }))
                  }
                }
              } catch (parseErr) {
                // 忽略解析错误，继续处理
                if (process.env.NODE_ENV === 'development') {
                  console.debug('[Protection] JSON解析失败:', parseErr)
                }
              }
            }
          }
        }
        
        // 流式完成，标记所有建议为完成状态
        const finalSuggestions = Array.from(allSuggestions.values()).map(s => ({
          ...s,
          _isComplete: true
        })).sort((a, b) => {
          const orderMap = { 'high_privacy': 0, 'balanced': 1, 'low_privacy': 2 }
          return (orderMap[a.level] || 999) - (orderMap[b.level] || 999)
        })
        
        console.log(`[Protection] 成功生成 ${finalSuggestions.length} 个建议`)
        
        // 更新最终状态
        set(state => ({
          protectionSuggestions: {
            ...state.protectionSuggestions,
            [session.id]: {
              status: 'done',
              suggestions: finalSuggestions,
              error: null,
              abortController: null
            }
          }
        }))
        
      } catch (fetchErr) {
        console.error('[Protection] 网络请求或响应处理失败:', fetchErr)
        
        // 区分不同的错误类型
        let errorMsg = fetchErr.message
        if (fetchErr.name === 'AbortError') {
          errorMsg = '请求已中止'
        } else if (fetchErr instanceof TypeError) {
          errorMsg = '网络连接失败，请检查API地址是否正确'
        }
        
        throw new Error(errorMsg)
      }
      
    } catch (err) {
      console.error('[Protection] 生成建议失败:', err)
      
      if (err.name === 'AbortError') {
        set(state => ({
          protectionSuggestions: {
            ...state.protectionSuggestions,
            [session.id]: {
              status: 'error',
              error: '建议生成已中止',
              suggestions: [],
              abortController: null
            }
          }
        }))
      } else {
        set(state => ({
          protectionSuggestions: {
            ...state.protectionSuggestions,
            [session.id]: {
              status: 'error',
              error: err.message || '建议生成失败',
              suggestions: [],
              abortController: null
            }
          }
        }))
      }
    }
  },
  
  // 停止生成建议
  abortProtectionSuggestions() {
    const session = get().getCurrentSession()
    if (!session) return
    
    const suggestions = get().protectionSuggestions?.[session.id]
    if (suggestions?.abortController) {
      try {
        suggestions.abortController.abort()
      } catch (_) {}
    }
  },
  
  // 清除建议
  clearProtectionSuggestions() {
    const session = get().getCurrentSession()
    if (!session) return
    
    set(state => {
      const newSuggestions = { ...state.protectionSuggestions }
      delete newSuggestions[session.id]
      return { 
        protectionSuggestions: newSuggestions
      }
    })
  },

  // ========== 用户历史数据持久化方法 ==========
  
  // 内部：加载用户历史数据
  _loadUserHistory(userId) {
    try {
      const data = loadUserSessions(userId, getDefaultModelsConfig())
      
      if (data && data.sessions && data.sessions.length > 0) {
        set({
          sessions: data.sessions,
          infonSessions: data.infonSessions || {},
          privacyInferences: data.privacyInferences || {},
          sessionKeywords: data.sessionKeywords || {}, // 加载关键词
          currentSessionId: data.sessions[0]?.id || null,
          customPrivacyItems: data.customPrivacyItems || [],
          selectedLawIdx: data.selectedLawIdx ?? 0,
          selectedPrivacyItems: data.selectedPrivacyItems || [],
          // 模型配置
          model: data.conversationModel || getDefaultModelsConfig().conversationModel,
          directInferenceModel: data.directInferenceModel || getDefaultModelsConfig().directInferenceModel,
          infonExtractionModel: data.infonExtractionModel || getDefaultModelsConfig().infonExtractionModel,
          infonPrivacyInferenceModel: data.infonPrivacyInferenceModel || getDefaultModelsConfig().infonPrivacyInferenceModel,
          imageParsingModel: data.imageParsingModel || getDefaultModelsConfig().imageParsingModel,
          protectionSuggestionModel: data.protectionSuggestionModel || getDefaultModelsConfig().protectionSuggestionModel,
          inferenceMode: data.inferenceMode || getDefaultModelsConfig().inferenceMode // 加载推断模式
        })
        console.log('[PrivaSee] 用户历史数据已加载（包含关键词）')
      } else {
        // 如果没有历史数据，初始化一个新会话
        const newSession = createEmptySession()
        set({
          sessions: [newSession],
          currentSessionId: newSession.id,
          infonSessions: {},
          privacyInferences: {},
          sessionKeywords: {}, // 初始化关键词
          customPrivacyItems: [],
          selectedLawIdx: 0,
          selectedPrivacyItems: [],
          inferenceMode: 'extract' // 默认为提取信息元模式
        })
      }
    } catch (error) {
      console.error('[PrivaSee] 加载用户历史失败:', error)
    }
  },
  
  // 内部：保存用户历史数据
  _saveUserHistory(userId) {
    try {
      const {
        sessions,
        infonSessions,
        privacyInferences,
        customPrivacyItems,
        selectedLawIdx,
        selectedPrivacyItems,
        model,
        directInferenceModel,
        infonExtractionModel,
        infonPrivacyInferenceModel,
        imageParsingModel,
        protectionSuggestionModel,
        inferenceMode,
        sessionKeywords
      } = get()
      
      // 清理不可序列化的字段（中文注释）：移除 abortController
      const serializableInferences = {}
      Object.keys(privacyInferences).forEach(sessionId => {
        const inference = privacyInferences[sessionId]
        if (inference) {
          const { abortController, ...rest } = inference
          serializableInferences[sessionId] = rest
        }
      })
      
      saveUserSessions(userId, sessions, infonSessions, serializableInferences, customPrivacyItems, selectedLawIdx, selectedPrivacyItems, model, directInferenceModel, infonExtractionModel, infonPrivacyInferenceModel, imageParsingModel, protectionSuggestionModel, inferenceMode, sessionKeywords)
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


