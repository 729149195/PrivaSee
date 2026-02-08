/**
 * 信息元合并与去重模块
 * 处理信息元的冲突检测和增量更新
 */

/**
 * 增量更新逻辑：检测和处理重复/冲突的信息元
 * 策略：
 * 1. 先对同一批次内的完全重复信息元进行去重（同 entity+attribute / 同 temporal+spatial / 同 relation_name+arg_refs）
 * 2. 再检测与已有信息元的语义冲突（如同一主体的不同属性值）
 * 3. 返回标记了冲突关系的新信息元，由上层决定如何展示
 * 
 * @param {Array} newInfons - 新的信息元列表
 * @param {Array} existingInfons - 已存在的信息元列表
 * @returns {Array} 处理后的新信息元列表
 */
export function deduplicateAndMergeInfons(newInfons, existingInfons) {
  if (!Array.isArray(newInfons) || newInfons.length === 0) return newInfons
  
  // === 第一步：同批次内去重（同 entity+attribute 等完全重复的信息元只保留置信度最高的） ===
  const deduped = deduplicateWithinBatch(newInfons)
  
  if (!Array.isArray(existingInfons) || existingInfons.length === 0) return deduped
  
  const result = []
  const conflictInfons = [] // 记录被新信息元替换的旧信息元
  
  // 合并新旧信息元列表用于查找引用
  const allInfonsForLookup = [...existingInfons, ...deduped]
  
  // === 第二步：跨批次冲突检测 ===
  deduped.forEach(newInfon => {
    const conflicts = findConflictingInfons(newInfon, existingInfons, allInfonsForLookup)
    
    if (conflicts.length > 0) {
      // 发现冲突：标记此信息元替换了哪些旧信息元
      result.push({ 
        ...newInfon, 
        _supersedes: conflicts.map(c => c.iid) // 记录此信息元取代了哪些旧的
      })
      conflictInfons.push(...conflicts)
    } else {
      // 无冲突：直接添加
      result.push(newInfon)
    }
  })
  
  return result
}

/**
 * 同批次内去重：合并语义完全相同的信息元，保留置信度最高的
 * 完全重复定义：
 * - DESC: entity + attribute 完全相同（不区分大小写）
 * - SCEN: temporal + spatial 完全相同
 * - REL: relation_name + arg_refs 完全相同
 */
function deduplicateWithinBatch(infons) {
  if (!Array.isArray(infons) || infons.length <= 1) return infons
  
  const seen = new Map() // fingerprint → { bestInfon, firstIndex }
  const noFpIndices = new Set() // 无指纹的信息元的索引
  
  // 第一遍：找出每个指纹的最佳信息元
  for (let i = 0; i < infons.length; i++) {
    const infon = infons[i]
    const fp = getInfonFingerprint(infon)
    if (!fp) {
      noFpIndices.add(i)
      continue
    }
    
    if (seen.has(fp)) {
      const entry = seen.get(fp)
      const newConf = Number(infon.confidence ?? 0)
      const existConf = Number(entry.bestInfon.confidence ?? 0)
      if (newConf > existConf) {
        entry.bestInfon = infon
        // 保留首次出现的位置不变，以维持原始顺序
      }
    } else {
      seen.set(fp, { bestInfon: infon, firstIndex: i })
    }
  }
  
  // 第二遍：按原始顺序输出，只在首次出现位置输出最佳版本
  const firstIndexSet = new Set(Array.from(seen.values()).map(e => e.firstIndex))
  const fpByIndex = new Map(Array.from(seen.entries()).map(([fp, e]) => [e.firstIndex, fp]))
  
  const result = []
  for (let i = 0; i < infons.length; i++) {
    if (noFpIndices.has(i)) {
      // 无指纹的信息元：直接保留
      result.push(infons[i])
    } else if (firstIndexSet.has(i)) {
      // 该指纹首次出现的位置：输出最佳版本
      const fp = fpByIndex.get(i)
      result.push(seen.get(fp).bestInfon)
    }
    // 其他位置（重复的后续出现）：跳过
  }
  
  return result
}

/**
 * 计算信息元的语义指纹（用于判断完全重复）
 */
function getInfonFingerprint(infon) {
  if (!infon) return null
  const type = String(infon.infon_type || '').toUpperCase()
  
  if (type === 'DESC') {
    const entity = String(infon.entity || '').trim().toLowerCase()
    const attribute = String(infon.attribute || '').trim().toLowerCase()
    return `DESC:${entity}:${attribute}`
  }
  if (type === 'SCEN') {
    const temporal = String(infon.temporal || '').trim().toLowerCase()
    const spatial = String(infon.spatial || '').trim().toLowerCase()
    return `SCEN:${temporal}:${spatial}`
  }
  if (type === 'REL') {
    const relName = String(infon.relation_name || '').trim().toLowerCase()
    const argRefs = Array.isArray(infon.arg_refs) ? [...infon.arg_refs].sort().join('|') : ''
    return `REL:${relName}:${argRefs}`
  }
  return null
}

/**
 * 查找与新信息元冲突的已有信息元
 * 冲突定义：表达同一主体的不同属性值，需要用新值替换旧值
 * 
 * @param {object} newInfon - 新信息元
 * @param {Array} existingInfons - 已存在的信息元列表
 * @param {Array} allInfonsForLookup - 用于查找引用的完整列表
 * @returns {Array} 冲突的信息元列表
 */
export function findConflictingInfons(newInfon, existingInfons, allInfonsForLookup) {
  if (!newInfon || !Array.isArray(existingInfons)) return []
  
  const type = String(newInfon.infon_type || '').toUpperCase()
  const conflicts = []
  
  if (type === 'DESC') {
    // DESC冲突检测：查找同一主体实体的不同属性值
    const newEntity = String(newInfon.entity || '').trim().toLowerCase()
    const newAttr = String(newInfon.attribute || '').trim().toLowerCase()
    
    // 特殊处理：姓名类属性冲突
    const isNameEntity = ['姓名', '名字', 'name', '名称'].includes(newEntity)
    
    existingInfons.forEach(existing => {
      if (String(existing.infon_type || '').toUpperCase() !== 'DESC') return
      
      const existEntity = String(existing.entity || '').trim().toLowerCase()
      const existAttr = String(existing.attribute || '').trim().toLowerCase()
      
      // 同一实体类别，但属性值不同
      if (newEntity === existEntity && newAttr !== existAttr) {
        // 对于姓名类属性，只有当两者都是姓名时才判定为冲突
        if (isNameEntity) {
          conflicts.push(existing)
        }
      }
    })
  } else if (type === 'REL') {
    // REL冲突检测：查找同一关系名称连接同一第一参数的关系
    const newRelName = String(newInfon.relation_name || '').trim().toLowerCase()
    const newArgRefs = Array.isArray(newInfon.arg_refs) ? newInfon.arg_refs : []
    
    // 特殊处理：名称关系冲突
    const isNameRelation = ['名字', '姓名', 'name', '名称', '名称关系', '名字关系'].includes(newRelName)
    
    if (isNameRelation && newArgRefs.length >= 2) {
      const newSubject = newArgRefs[0] // 第一个参数是主体（如"我"）
      
      existingInfons.forEach(existing => {
        if (String(existing.infon_type || '').toUpperCase() !== 'REL') return
        
        const existRelName = String(existing.relation_name || '').trim().toLowerCase()
        const existArgRefs = Array.isArray(existing.arg_refs) ? existing.arg_refs : []
        
        // 同一类型关系，且主体相同
        if (isNameRelation && existArgRefs.length >= 2) {
          const existSubject = existArgRefs[0]
          
          // 检查是否指向同一主体（通过查找主体信息元的实际内容）
          // 使用合并后的列表查找，以支持跨新旧信息元的引用
          if (isSameSubject(newSubject, existSubject, allInfonsForLookup || existingInfons)) {
            conflicts.push(existing)
          }
        }
      })
    }
  }
  
  return conflicts
}

/**
 * 检查两个信息元引用是否指向同一主体
 * 
 * @param {string} iid1 - 第一个信息元 ID
 * @param {string} iid2 - 第二个信息元 ID
 * @param {Array} allInfons - 所有信息元列表
 * @returns {boolean} 是否指向同一主体
 */
export function isSameSubject(iid1, iid2, allInfons) {
  if (iid1 === iid2) return true
  
  // 查找两个iid对应的信息元内容
  const infon1 = allInfons.find(i => i.iid === iid1)
  const infon2 = allInfons.find(i => i.iid === iid2)
  
  if (!infon1 || !infon2) return false
  
  // 如果都是DESC类型且实体相同，认为是同一主体
  if (String(infon1.infon_type || '').toUpperCase() === 'DESC' &&
      String(infon2.infon_type || '').toUpperCase() === 'DESC') {
    const entity1 = String(infon1.entity || '').trim().toLowerCase()
    const entity2 = String(infon2.entity || '').trim().toLowerCase()
    const attr1 = String(infon1.attribute || '').trim().toLowerCase()
    const attr2 = String(infon2.attribute || '').trim().toLowerCase()
    
    // 同一实体和属性，或都是"我"、"用户"等主体代词
    const subjectPronouns = ['我', 'i', 'me', '用户', 'user']
    return (entity1 === entity2 && attr1 === attr2) || 
           (subjectPronouns.includes(attr1) && subjectPronouns.includes(attr2))
  }
  
  return false
}
