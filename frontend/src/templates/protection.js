// Privacy Protection Suggestions Prompt Template
// 隐私保护修改建议提示词模板

export const PROTECTION_SUGGESTIONS_PROMPT = `
您是一位隐私保护专家，专门为用户提供文本修改建议，帮助在隐私保护和模型效用之间找到平衡。

## 任务
基于以下输入，提供3种不同级别的隐私保护修改建议：
1. **高隐私保护（低效用）**：最大程度保护隐私，可能显著降低模型理解和响应质量
2. **平衡方案（中等效用）**：在隐私保护和模型效用之间取得平衡
3. **低隐私保护（高效用）**：最小程度修改，保持模型效用，仅移除最敏感信息

## 输入数据

### 1. 原始文本
{{ORIGINAL_TEXT}}

### 2. 检测到的隐私风险
{{PRIVACY_RISKS}}

### 3. 检测到的信息元
{{INFONS}}

## 修改原则

### 高隐私保护策略
- 移除或替换所有可识别个人的信息
- 泛化所有具体细节（时间、地点、数字等）
- 使用抽象描述替代具体内容
- 优先考虑匿名性，即使损失信息完整性

### 平衡策略
- 保留必要的上下文信息
- 对敏感信息进行适度泛化
- 保持查询的核心意图
- 在隐私和可用性间权衡

### 低隐私保护策略
- 仅移除最敏感的个人信息（如身份证号、真实姓名等）
- 保留大部分上下文细节
- 最小化修改，保持原意
- 优先保证模型理解准确性

## 输出格式

输出**仅包含有效JSON**（无markdown代码块，无额外说明）：

{
  "suggestions": [
    {
      "level": "high_privacy",
      "label": "高隐私保护",
      "privacy_score": 95,
      "utility_score": 40,
      "modified_text": "修改后的文本内容",
      "changes_summary": "简要说明做了哪些修改，为什么这样修改",
      "removed_risks": ["列出移除了哪些隐私风险"]
    },
    {
      "level": "balanced",
      "label": "平衡方案",
      "privacy_score": 75,
      "utility_score": 70,
      "modified_text": "修改后的文本内容",
      "changes_summary": "简要说明做了哪些修改",
      "removed_risks": ["列出移除了哪些隐私风险"]
    },
    {
      "level": "low_privacy",
      "label": "低隐私保护",
      "privacy_score": 50,
      "utility_score": 90,
      "modified_text": "修改后的文本内容",
      "changes_summary": "简要说明做了哪些修改",
      "removed_risks": ["列出移除了哪些隐私风险"]
    }
  ]
}

## 关键要求
1. **仅输出JSON** - 不要使用markdown代码块，不要添加任何解释文字
2. **保持语言一致** - 如果输入是中文，修改后的文本也必须是中文；英文同理
3. **修改要具体** - modified_text必须是完整的、可直接使用的文本
4. **评分要合理** - privacy_score和utility_score范围0-100，要符合修改程度
5. **说明要清晰** - changes_summary要简洁说明修改了什么，为什么
6. **按顺序输出** - 必须按high_privacy → balanced → low_privacy顺序
7. **保持完整性** - 即使高隐私保护方案也要确保文本可读、有意义

## 示例

输入：
"我叫张三，身份证号123456789012345678，住在北京市朝阳区某小区，想咨询一下我的糖尿病饮食建议"

输出应该包含三个级别的建议，例如：
- 高隐私：移除所有个人信息，泛化健康状况
- 平衡：保留健康状况但泛化位置和身份
- 低隐私：仅移除身份证号和详细地址

现在请分析输入并输出完整的JSON建议。
`

/**
 * 填充保护建议提示词模板
 * @param {string} originalText - 原始文本
 * @param {Array} privacyRisks - 隐私风险列表
 * @param {Array} infons - 信息元列表
 * @returns {string} 填充后的提示词
 */
export function fillProtectionPrompt(originalText, privacyRisks, infons) {
  // 格式化隐私风险
  let risksText = '未检测到隐私风险'
  if (Array.isArray(privacyRisks) && privacyRisks.length > 0) {
    risksText = privacyRisks.map((risk, idx) => {
      const level = risk.risk_level || 'UNKNOWN'
      const lawNode = risk.law_node_name || '未知'
      const exposure = risk.privacy_exposure || '未知'
      return `${idx + 1}. [${level}] ${lawNode}: ${exposure}`
    }).join('\n')
  }

  // 格式化信息元
  let infonsText = '未检测到信息元'
  if (Array.isArray(infons) && infons.length > 0) {
    infonsText = infons.map((infon, idx) => {
      const type = String(infon.infon_type || '').toUpperCase()
      const iid = infon.iid || `infon_${idx}`
      
      let detail = ''
      if (type === 'DESC') {
        const entity = infon.entity || ''
        const attribute = infon.attribute || ''
        detail = `${entity}: ${attribute}`
      } else if (type === 'SCEN') {
        const temporal = infon.temporal || ''
        const spatial = infon.spatial || ''
        detail = `${temporal} @ ${spatial}`
      } else if (type === 'REL') {
        detail = infon.relation_name || '关系'
      } else if (type === 'SIT') {
        detail = infon.description || '情境'
      }
      
      return `- [${iid}] ${type}: ${detail}`
    }).join('\n')
  }

  // 填充模板
  const prompt = PROTECTION_SUGGESTIONS_PROMPT
    .replace('{{ORIGINAL_TEXT}}', originalText || '')
    .replace('{{PRIVACY_RISKS}}', risksText)
    .replace('{{INFONS}}', infonsText)

  return prompt
}

/**
 * 解析保护建议响应
 * @param {string} responseText - API响应文本
 * @returns {Object|null} 解析后的建议对象
 */
export function parseProtectionResponse(responseText) {
  try {
    // 尝试直接解析
    const parsed = JSON.parse(responseText)
    if (parsed && parsed.suggestions && Array.isArray(parsed.suggestions)) {
      return parsed
    }
  } catch (e) {
    // 如果直接解析失败，尝试提取JSON对象
    const jsonMatch = responseText.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[0])
        if (parsed && parsed.suggestions && Array.isArray(parsed.suggestions)) {
          return parsed
        }
      } catch (e2) {
        console.warn('[Protection] 无法解析JSON响应', e2)
      }
    }
  }
  
  return null
}

/**
 * 验证建议数据的完整性
 * @param {Object} suggestion - 单个建议对象
 * @returns {boolean} 是否有效
 */
export function validateSuggestion(suggestion) {
  if (!suggestion || typeof suggestion !== 'object') return false
  
  const required = ['level', 'label', 'privacy_score', 'utility_score', 'modified_text']
  return required.every(field => suggestion[field] !== undefined && suggestion[field] !== null)
}

