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
    if (/(vision|vl|multi.?modal|llava|idefics|qwen[-_]?vl|qwen.*vl|gpt-?4o|xcomposer|internvl|minicpm-?v|pixtral|gemma[-_]?3|omni)/.test(s)) return true
    // 自定义提供商（OpenAI 兼容 API）通常不支持图片；若上面未命中则认为是文本
    if (customProviders?.[id]) return false
    return false
  } catch (_) {
    return false
  }
}

/**
 * 检测模型支持的模态类型
 * @param {string} id - 模型 ID
 * @param {object} customProviders - 自定义提供商配置
 * @returns {object} { text: boolean, image: boolean, audio: boolean }
 */
export function getModelModalities(id, customProviders = {}) {
  try {
    if (!id) return { text: false, image: false, audio: false }
    
    const s = String(id).toLowerCase()
    const modalities = {
      text: true, // 默认所有模型支持文本
      image: false,
      audio: false
    }
    
    // 图像模态检测
    if (/(vision|vl|multi.?modal|llava|idefics|qwen[-_]?vl|qwen.*vl|gpt-?4o|xcomposer|internvl|minicpm-?v|pixtral|gemma[-_]?3|omni)/.test(s)) {
      modalities.image = true
    }
    
    // 音频模态检测
    if (/(omni|audio|speech|whisper)/.test(s)) {
      modalities.audio = true
    }
    
    return modalities
  } catch (_) {
    return { text: false, image: false, audio: false }
  }
}

/**
 * 检测模型是否支持思维链（Chain of Thought）
 * @param {string} id - 模型 ID
 * @param {object} customProviders - 自定义提供商配置
 * @returns {boolean} 是否支持思维链
 */
export function supportsChainOfThought(id, customProviders = {}) {
  try {
    if (!id) return false
    
    const s = String(id).toLowerCase()
    
    // 已知支持思维链的模型家族
    // DeepSeek: 只有 r1 系列支持思维链，不包括 deepseek-chat
    // Qwen: qwen3, qwen-plus, qwen-turbo, qwen-max 系列
    // GPT: o1 系列和 gpt-4 系列，gpt-oss 系列
    // Claude: claude-3 系列
    if (/(deepseek[-_]?r1|^o1|qwen3|qwen[-_]?plus|qwen[-_]?turbo|qwen[-_]?max|gpt[-_]?oss|claude[-_]?3|gpt[-_]?4)/.test(s)) {
      return true
    }
    
    return false
  } catch (_) {
    return false
  }
}

