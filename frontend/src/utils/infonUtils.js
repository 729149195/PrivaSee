/**
 * 信息元工具函数集合
 * 用于处理信息元相关的颜色、关键词提取、索引构建等操作
 */

/**
 * 获取信息元类型对应的高亮颜色
 * @param {string} infonType - 信息元类型 (DESC, SCEN, REL, SIT)
 * @returns {string} 颜色的十六进制值
 */
export function getInfonColor(infonType) {
  const colors = {
    DESC: '#3b82f6',  // 描述（实体+属性）：蓝色
    SCEN: '#10b981',  // 场景（时间+位置）：翠绿色
    REL: '#8b5cf6',   // 关系：紫色
    SIT: '#f59e0b',   // 情景：琥珀色
  }
  return colors[String(infonType).toUpperCase()] || '#64748b'
}

/**
 * 从信息元中提取用于匹配的关键词
 * @param {object} infon - 信息元对象
 * @returns {Array<string>} 关键词数组
 */
export function getMatchKeywords(infon) {
  if (!infon || typeof infon !== 'object') return []
  const keywords = []
  const t = String(infon.infon_type || '').toUpperCase()
  
  if (t === 'DESC') {
    // 描述：提取实体和属性作为关键词（优先属性，因为属性是实际值）
    if (infon.attribute) keywords.push(String(infon.attribute))
    if (infon.entity) keywords.push(String(infon.entity))
  } else if (t === 'SCEN') {
    // 场景：提取时间和空间作为关键词（优先时间）
    if (infon.temporal) keywords.push(String(infon.temporal))
    if (infon.spatial) keywords.push(String(infon.spatial))
  } else if (t === 'REL' && infon.relation_name) {
    keywords.push(String(infon.relation_name))
  }
  // SIT 不提取关键词用于高亮
  
  return keywords.filter(k => k && k.trim())
}

/**
 * 构建信息元索引，用于快速查找 iid 对应的信息元
 * @param {Array} infonList - 信息元列表，每项包含 { infon, run }
 * @returns {Object} 索引对象，键为 iid，值为 infon
 */
export function buildInfonIndex(infonList) {
  const index = {}
  infonList.forEach(({ infon }) => {
    if (infon.iid) index[infon.iid] = infon
  })
  return index
}

/**
 * 收集关系信息元关联的所有信息元
 * @param {object} infon - 关系类型的信息元
 * @param {object} infonIndex - 信息元索引对象
 * @returns {Array} 关联的信息元数组
 */
export function getRelatedInfons(infon, infonIndex) {
  const related = []
  if (String(infon.infon_type || '').toUpperCase() === 'REL' && Array.isArray(infon.arg_refs)) {
    infon.arg_refs.forEach(ref => {
      if (infonIndex[ref]) related.push(infonIndex[ref])
    })
  }
  return related
}

