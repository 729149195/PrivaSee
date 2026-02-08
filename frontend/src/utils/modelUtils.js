/**
 * 模型相关工具函数集合
 * 用于检测模型能力和特性
 */

/**
 * 将原始模型 ID 转换为简短友好的显示名称
 * 例如 "niels32167/qwen3-4b-instruct:latest" → "Qwen3 4B Instruct"
 * @param {string} id - 模型原始 ID
 * @returns {string} 简短显示名称
 */
export function prettifyModelName(id) {
  if (!id) return ''
  let s = String(id)

  // 1. 去掉命名空间前缀 (e.g. "niels32167/")
  s = s.replace(/^[^/]+\//, '')

  // 2. 处理标签后缀 — 保留有意义的标签，去掉 "latest" 等冗余标签
  const colonIdx = s.indexOf(':')
  if (colonIdx !== -1) {
    const name = s.slice(0, colonIdx)
    const tag = s.slice(colonIdx + 1)
    // 无意义标签直接去掉
    if (/^(latest)$/i.test(tag)) {
      s = name
    } else {
      // 有意义的标签（如 7b, q4_k_m）用空格拼接
      s = name + ' ' + tag
    }
  }

  // 3. 将分隔符统一为空格
  s = s.replace(/[-_]+/g, ' ')

  // 4. 处理参数量标识：把 "4b" → "4B", "7b" → "7B" 等
  s = s.replace(/\b(\d+(?:\.\d+)?)\s*b\b/gi, (_, n) => `${n}B`)

  // 5. 首字母大写，但保留全大写缩写（VL, OCR 等）和量化标记
  s = s.replace(/\b\w+/g, (word) => {
    // 已全大写或是大写+数字组合的（如 4B, VL, Q4, K, M）保持原样
    if (/^[A-Z0-9]+$/.test(word)) return word
    // 量化标记保持小写风格也行，统一大写
    if (/^(q\d+|k|m|fp\d+)$/i.test(word)) return word.toUpperCase()
    // 否则首字母大写
    return word.charAt(0).toUpperCase() + word.slice(1)
  })

  return s.trim()
}

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
    if (/(vision|vl|multi.?modal|llava|idefics|qwen[-_]?vl|qwen.*vl|gpt-?4o|xcomposer|internvl|minicpm-?v|pixtral|gemma[-_]?3|omni|ocr)/.test(s)) return true
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
    if (/(vision|vl|multi.?modal|llava|idefics|qwen[-_]?vl|qwen.*vl|gpt-?4o|xcomposer|internvl|minicpm-?v|pixtral|gemma[-_]?3|omni|ocr)/.test(s)) {
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
    // Qwen: qwen3 官方版（非 instruct 微调），qwen-plus, qwen-turbo, qwen-max 系列
    // GPT: o1 系列和 gpt-4 系列，gpt-oss 系列
    // Claude: claude-3 系列
    // 注意：instruct 微调版本不输出 <think> 标签，排除掉
    if (/instruct/i.test(s)) return false
    if (/(deepseek[-_]?r1|^o1|qwen3|qwen[-_]?plus|qwen[-_]?turbo|qwen[-_]?max|gpt[-_]?oss|claude[-_]?3|gpt[-_]?4)/.test(s)) {
      return true
    }
    
    return false
  } catch (_) {
    return false
  }
}

