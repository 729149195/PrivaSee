/**
 * 主记忆流 Slice (MemoryStream + Association Backtracking)
 * 
 * 功能：
 * 1. 信息元写入主记忆流 (含 Top-K 关联绑定)
 * 2. 风险触发式可控检索 (准标识符组合 / 细化线索 / 敏感域命中)
 * 3. 关联回溯查询 (证据指针 + 关联信息元列表)
 * 4. 一键清空
 * 
 * 依赖后端 /api/memory/* 端点
 */

// 后端 Memory Stream 服务基础 URL
// 使用 Vite 代理路径: /memory-api -> http://127.0.0.1:5000/api/memory
// 与 OCR (/ocr-api) 和 Whisper (/whisper-api) 保持一致的代理模式
const MEMORY_API_BASE = import.meta.env.VITE_MEMORY_URL || '/memory-api'

// 匿名临时 ID：仅存在于内存中，刷新浏览器即消失
// 未登录用户的记忆流功能照常工作，但数据不会跨页面会话保留
const _anonymousMemoryId = `_anon_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`

function _sanitizeIdPart(value, fallback = 'x') {
  const text = String(value ?? '').trim()
  if (!text) return fallback
  const normalized = text.replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '')
  return normalized || fallback
}

function _scopeInfonsForSession(infons, userId, sessionId, roundNum) {
  if (!Array.isArray(infons) || infons.length === 0) return []
  const safeUser = _sanitizeIdPart(userId, 'user')
  const safeSession = _sanitizeIdPart(sessionId, 'session')
  const safeRound = Number.isFinite(Number(roundNum)) ? Number(roundNum) : 1
  const scope = `u${safeUser}_s${safeSession}_r${safeRound}`

  const iidMap = new Map()
  const used = new Set()
  const pickUnique = (base) => {
    let candidate = `${base}__${scope}`
    let idx = 1
    while (used.has(candidate)) {
      candidate = `${base}__${scope}_${idx}`
      idx += 1
    }
    used.add(candidate)
    return candidate
  }

  infons.forEach((infon, index) => {
    const src = String(infon?.iid || '').trim() || `auto_${index + 1}`
    const scoped = pickUnique(_sanitizeIdPart(src, `auto_${index + 1}`))
    iidMap.set(src, scoped)
  })

  return infons.map((infon, index) => {
    const src = String(infon?.iid || '').trim() || `auto_${index + 1}`
    const scopedIid = iidMap.get(src) || pickUnique(_sanitizeIdPart(src, `auto_${index + 1}`))
    const nextArgRefs = Array.isArray(infon?.arg_refs)
      ? infon.arg_refs.map(ref => iidMap.get(ref) || ref)
      : infon?.arg_refs
    return {
      ...infon,
      _source_iid: src,
      iid: scopedIid,
      arg_refs: nextArgRefs,
    }
  })
}

export const createMemoryStreamSlice = (set, get) => {
  const pushAssociationEvent = (event) => {
    const now = Date.now()
    const normalized = {
      id: `assoc_evt_${now}_${Math.random().toString(36).slice(2, 7)}`,
      ts: now,
      type: event?.type || 'unknown',
      title: event?.title || '',
      detail: event?.detail || '',
      payload: event?.payload || {},
    }
    set(s => ({
      memoryAssociationEvents: [normalized, ...(s.memoryAssociationEvents || [])].slice(0, 120),
    }))
  }

  return ({
  // ==================== 状态 ====================
  
  // 记忆流健康/统计状态
  memoryStreamStatus: null,  // { status, total_infons, index_size, embedding_dim }
  
  // 上次写入结果
  memoryStreamLastIngest: null,  // { ingested_count, skipped_count, ingested, ... }
  
  // 检索到的历史信息元 (用于隐私推理前置上下文)
  memoryRetrievedInfons: [],
  
  // 触发检测结果
  memoryTriggerResult: null,  // { triggered, triggers, retrieved_infons }
  
  // 回溯查询结果缓存
  memoryBacktraceCache: {},  // { [iid]: { evidence_pointer, associations, ... } }
  // 关联线索调取/回溯事件时间线（用于 Debug 可视化）
  memoryAssociationEvents: [],
  
  // 加载状态
  memoryStreamLoading: false,
  memoryStreamError: null,
  
  // ==================== 信息元写入 (含关联绑定) ====================
  
  /**
   * 将信息元批量写入主记忆流
   * 在 infonHelpers.js 的 handleInfonFinish 之后调用
   * 
   * @param {Array} infons - 信息元列表
   * @param {string} sessionId - 会话标识
   * @param {number} roundNum - 轮次编号
   */
  /**
   * 获取当前记忆流用户标识
   * - 已登录: 使用 currentUserId (数据持久化)
   * - 未登录: 使用 _anonymousMemoryId (刷新浏览器即消失)
   */
  _getMemoryUserId() {
    return get().currentUserId || _anonymousMemoryId
  },

  async ingestInfonsToMemory(infons, sessionId, roundNum) {
    if (!Array.isArray(infons) || infons.length === 0) return
    
    try {
      const userId = get()._getMemoryUserId()
      const scopedInfons = _scopeInfonsForSession(infons, userId, sessionId, roundNum)
      const response = await fetch(`${MEMORY_API_BASE}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          infons: scopedInfons,
          session_id: sessionId,
          round_num: roundNum,
        }),
      })
      
      if (!response.ok) {
        const errText = await response.text().catch(() => '')
        console.warn('[MemoryStream] 写入失败:', errText)
        return
      }
      
      const result = await response.json()
      
      set({ memoryStreamLastIngest: result })
      
      // 将关联信息回写到 infon 对象 (更新前端状态)
      if (result.ingested && result.ingested.length > 0) {
        const linked = result.ingested.filter(item => Array.isArray(item.associations) && item.associations.length > 0)
        pushAssociationEvent({
          type: 'ingest_association_bind',
          title: 'Ingest association binding',
          detail: `${result.ingested.length} 条写入，${linked.length} 条产生关联`,
          payload: {
            ingested_count: result.ingested.length,
            linked_count: linked.length,
            linked_iids: linked.slice(0, 8).map(item => item.iid),
          },
        })

        const infonSession = get().infonSessions?.[sessionId]
        if (infonSession) {
          const ingestedMap = new Map()
          const iidRemap = new Map()
          result.ingested.forEach(item => {
            const sourceIid = item.source_iid || item.iid
            const resolvedIid = item.iid || sourceIid
            ingestedMap.set(sourceIid, {
              iid: resolvedIid,
              evidence_pointer: item.evidence_pointer,
              associations: item.associations,
            })
            if (sourceIid && resolvedIid && sourceIid !== resolvedIid) {
              iidRemap.set(sourceIid, resolvedIid)
            }
          })
          
          // 更新 runs 中对应信息元的 associations 和 evidence_pointer
          set(s => {
            const box = s.infonSessions?.[sessionId]
            if (!box) return {}
            const updatedRuns = box.runs.map(run => {
              if (!run.resultJson?.infons) return run
              const updatedInfons = run.resultJson.infons.map(infon => {
                const sourceIid = infon.iid
                const ingestData = ingestedMap.get(sourceIid)
                const nextIid = ingestData?.iid || sourceIid
                const nextArgRefs = Array.isArray(infon.arg_refs)
                  ? infon.arg_refs.map(ref => iidRemap.get(ref) || ref)
                  : infon.arg_refs

                if (ingestData) {
                  return {
                    ...infon,
                    iid: nextIid,
                    arg_refs: nextArgRefs,
                    evidence_pointer: ingestData.evidence_pointer,
                    associations: ingestData.associations,
                  }
                }
                if (nextIid !== sourceIid || nextArgRefs !== infon.arg_refs) {
                  return {
                    ...infon,
                    iid: nextIid,
                    arg_refs: nextArgRefs,
                  }
                }
                return infon
              })
              return {
                ...run,
                resultJson: { ...run.resultJson, infons: updatedInfons },
              }
            })
            return {
              infonSessions: {
                ...s.infonSessions,
                [sessionId]: { runs: updatedRuns },
              },
            }
          })
        }
      }
      
      console.log(`[MemoryStream] 写入完成: ${result.ingested_count} 条, 跳过 ${result.skipped_count} 条`)
    } catch (err) {
      console.warn('[MemoryStream] 写入请求异常:', err.message)
    }
  },

  /**
   * 删除指定会话下已失效的生命周期信息元（主要用于 pending 更新/撤销）
   */
  async removeMemoryInfonsByRunIds(sessionId, runIds = [], targetType = 'pending') {
    if (!sessionId || !Array.isArray(runIds) || runIds.length === 0) return null
    try {
      const response = await fetch(`${MEMORY_API_BASE}/remove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: get()._getMemoryUserId(),
          session_id: sessionId,
          target_type: targetType,
          run_ids: runIds,
        }),
      })
      if (!response.ok) return null
      return await response.json()
    } catch (_) {
      return null
    }
  },

  /**
   * 删除一个会话窗口下的所有信息元（会话被删除时调用）
   */
  async removeMemoryBySession(sessionId) {
    if (!sessionId) return null
    try {
      const response = await fetch(`${MEMORY_API_BASE}/remove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: get()._getMemoryUserId(),
          session_id: sessionId,
        }),
      })
      if (!response.ok) return null
      return await response.json()
    } catch (_) {
      return null
    }
  },

  /**
   * 发送成功后：将 pending 信息元升级为 message 信息元
   */
  async promotePendingMemoryInfons(sessionId, runIds = [], messageId = '') {
    if (!sessionId || !Array.isArray(runIds) || runIds.length === 0) return null
    try {
      const response = await fetch(`${MEMORY_API_BASE}/promote-pending`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: get()._getMemoryUserId(),
          session_id: sessionId,
          run_ids: runIds,
          message_id: messageId,
        }),
      })
      if (!response.ok) return null
      return await response.json()
    } catch (_) {
      return null
    }
  },
  
  // ==================== 风险触发检索 ====================
  
  /**
   * 风险触发检测 + 可控检索
   * 在 privacySlice.js 的 startPrivacyInference 之前调用
   * 
   * @param {Array} infons - 当前消息的信息元列表
   * @returns {Array} 检索到的历史信息元列表 (可能为空)
   */
  async triggerCheckAndRetrieve(infons) {
    if (!Array.isArray(infons) || infons.length === 0) {
      set({ memoryRetrievedInfons: [], memoryTriggerResult: null })
      return []
    }
    
    try {
      const response = await fetch(`${MEMORY_API_BASE}/trigger-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: get()._getMemoryUserId(), infons }),
      })
      
      if (!response.ok) {
        console.warn('[MemoryStream] 触发检测失败')
        set({ memoryRetrievedInfons: [], memoryTriggerResult: null })
        return []
      }
      
      const result = await response.json()
      
      const retrievedInfons = result.retrieved_infons || []
      const triggerTypes = Array.isArray(result.triggers)
        ? result.triggers.map(t => t.trigger_type).filter(Boolean)
        : []
      const queryIids = Array.isArray(infons) ? infons.map(i => i?.iid).filter(Boolean).slice(0, 10) : []
      const retrievedIids = retrievedInfons.map(i => i?.iid).filter(Boolean).slice(0, 12)
      set({
        memoryRetrievedInfons: retrievedInfons,
        memoryTriggerResult: result,
      })
      pushAssociationEvent({
        type: 'trigger_check',
        title: result.triggered ? 'Trigger check: retrieval triggered' : 'Trigger check: not triggered',
        detail: `触发器: ${triggerTypes.length > 0 ? triggerTypes.join(', ') : 'none'}; 检索 ${retrievedInfons.length} 条`,
        payload: {
          triggered: !!result.triggered,
          trigger_types: triggerTypes,
          query_iids: queryIids,
          retrieved_iids: retrievedIids,
          retrieved_count: retrievedInfons.length,
        },
      })
      
      if (result.triggered) {
        console.log(
          `[MemoryStream] 触发检索: ${result.triggers.map(t => t.trigger_type).join(', ')}, ` +
          `检索到 ${retrievedInfons.length} 条历史信息元`
        )
      }
      
      return retrievedInfons
    } catch (err) {
      console.warn('[MemoryStream] 触发检测请求异常:', err.message)
      set({ memoryRetrievedInfons: [], memoryTriggerResult: null })
      return []
    }
  },
  
  // ==================== 向量搜索 ====================
  
  /**
   * 直接向量相似度搜索
   * 
   * @param {string} queryText - 查询文本
   * @param {number} k - 返回数量
   * @returns {Array} 搜索结果
   */
  async searchMemoryStream(queryText, k = 5) {
    try {
      set({ memoryStreamLoading: true, memoryStreamError: null })
      
      const response = await fetch(`${MEMORY_API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: get()._getMemoryUserId(), query: queryText, k }),
      })
      
      if (!response.ok) throw new Error('搜索失败')
      
      const result = await response.json()
      set({ memoryStreamLoading: false })
      return result.results || []
    } catch (err) {
      set({ memoryStreamLoading: false, memoryStreamError: err.message })
      return []
    }
  },
  
  // ==================== 关联回溯 ====================
  
  /**
   * 关联回溯查询
   * 给定信息元 iid，获取证据指针和关联信息元列表
   * 
   * @param {string} iid - 信息元唯一标识
   * @returns {Object|null} 回溯结果
   */
  async queryBacktrace(iid) {
    if (!iid) return null
    
    // 检查缓存
    const cached = get().memoryBacktraceCache?.[iid]
    if (cached) {
      pushAssociationEvent({
        type: 'backtrace_cache_hit',
        title: 'Backtrace cache hit',
        detail: `${iid} 命中缓存`,
        payload: {
          iid,
          evidence_pointer: cached?.evidence_pointer || null,
          association_iids: Array.isArray(cached?.associations)
            ? cached.associations.map(a => a?.iid).filter(Boolean).slice(0, 12)
            : [],
        },
      })
      return cached
    }
    
    try {
      const params = new URLSearchParams({ user_id: get()._getMemoryUserId() })
      const response = await fetch(
        `${MEMORY_API_BASE}/backtrace/${encodeURIComponent(iid)}?${params}`
      )
      
      if (!response.ok) {
        if (response.status === 404) {
          pushAssociationEvent({
            type: 'backtrace_miss',
            title: 'Backtrace miss',
            detail: `${iid} 未找到可回溯记录`,
            payload: { iid },
          })
          return null
        }
        throw new Error('回溯查询失败')
      }
      
      const result = await response.json()
      pushAssociationEvent({
        type: 'backtrace_query',
        title: 'Backtrace queried',
        detail: `${iid} -> ${(result?.associations || []).length} 条关联`,
        payload: {
          iid,
          evidence_pointer: result?.evidence_pointer || null,
          association_iids: Array.isArray(result?.associations)
            ? result.associations.map(a => a?.iid).filter(Boolean).slice(0, 20)
            : [],
        },
      })
      
      // 缓存结果
      set(s => ({
        memoryBacktraceCache: {
          ...s.memoryBacktraceCache,
          [iid]: result,
        },
      }))
      
      return result
    } catch (err) {
      console.warn('[MemoryStream] 回溯查询异常:', err.message)
      return null
    }
  },
  
  // ==================== 健康检查 / 统计 ====================
  
  /**
   * 获取记忆流状态
   */
  async fetchMemoryStreamStatus() {
    try {
      const params = new URLSearchParams({ user_id: get()._getMemoryUserId() })
      const response = await fetch(`${MEMORY_API_BASE}/health?${params}`)
      if (!response.ok) throw new Error('健康检查失败')
      
      const result = await response.json()
      set({ memoryStreamStatus: result })
      return result
    } catch (err) {
      set({ memoryStreamStatus: { status: 'error', error: err.message } })
      return null
    }
  },
  
  // ==================== 可视化数据 ====================
  
  // 可视化数据缓存
  memoryVisualizationData: null,  // { points, edges, total, method }
  memoryVisualizationLoading: false,
  
  /**
   * 获取信息元可视化数据 (自动降维到 2D)
   * 
   * @param {string} method - 降维方法: 'auto' (默认), 'tsne' 或 'pca'
   * @returns {Object|null} 可视化数据
   */
  async fetchVisualizationData(method = 'auto') {
    try {
      set({ memoryVisualizationLoading: true, memoryStreamError: null })
      
      const params = new URLSearchParams({ method, user_id: get()._getMemoryUserId() })
      const response = await fetch(`${MEMORY_API_BASE}/visualization?${params}`)
      
      if (!response.ok) throw new Error('获取可视化数据失败')
      
      const result = await response.json()
      set({ memoryVisualizationData: result, memoryVisualizationLoading: false })
      return result
    } catch (err) {
      set({ memoryVisualizationLoading: false, memoryStreamError: err.message })
      console.warn('[MemoryStream] 可视化数据获取失败:', err.message)
      return null
    }
  },
  
  // ==================== 一键清空 ====================
  
  /**
   * 清空所有信息元记录和向量索引
   * 用于测试、调参和保证实验可复现性
   */
  async clearMemoryStream() {
    try {
      set({ memoryStreamLoading: true, memoryStreamError: null })
      
      const response = await fetch(`${MEMORY_API_BASE}/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: get()._getMemoryUserId() }),
      })
      
      if (!response.ok) throw new Error('清空失败')
      
      set({
        memoryStreamLoading: false,
        memoryStreamLastIngest: null,
        memoryRetrievedInfons: [],
        memoryTriggerResult: null,
        memoryBacktraceCache: {},
        memoryAssociationEvents: [],
        memoryStreamStatus: null,
        memoryVisualizationData: null,
      })
      
      console.log('[MemoryStream] 所有数据已清空')
      return true
    } catch (err) {
      set({ memoryStreamLoading: false, memoryStreamError: err.message })
      console.error('[MemoryStream] 清空失败:', err)
      return false
    }
  },
  })
}

