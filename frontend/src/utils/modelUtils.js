/**
 * 模型相关工具函数集合
 * 用于检测模型能力和特性
 */

/**
 * 多模态能力检测：基于模型 ID 关键词 + 自定义提供商回退
 * @param {string} id - 模型 ID
 * @param {object} customProviders - 自定义提供商配置
 * @returns {boolean} 是否支持多模态
 */
export function isModelMultimodal(id, customProviders = {}) {
  try {
    if (!id) return false
    // 先基于模型 ID 关键词判断：优先识别已知多模态家族
    const s = String(id).toLowerCase()
    if (/(vision|vl|multi.?modal|llava|idefics|qwen[-_]?vl|gpt-?4o|xcomposer|internvl|minicpm-?v|pixtral|gemma[-_]?3)/.test(s)) return true
    // 自定义提供商（OpenAI 兼容 API）通常不支持图片；若上面未命中则认为是文本
    if (customProviders?.[id]) return false
    return false
  } catch (_) {
    return false
  }
}

