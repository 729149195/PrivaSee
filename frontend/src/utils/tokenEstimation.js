/**
 * Token 估算工具函数集合
 * 用于估算文本、图片和消息上下文的 token 数量
 */

/**
 * 估算文本的 token 数量
 * CJK≈1/字，拉丁≈1/4 字符，Emoji≈2/个
 * @param {string} text - 需要估算的文本
 * @returns {number} 估算的 token 数量
 */
export function estimateTextTokens(text) {
  try {
    const s = String(text || '')
    if (!s) return 0
    const cjkRegex = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu
    const emojiRegex = /\p{Extended_Pictographic}/gu
    const cjkCount = (s.match(cjkRegex) || []).length
    const emojiCount = (s.match(emojiRegex) || []).length
    const rest = s.replace(cjkRegex, '').replace(emojiRegex, '')
    const latinCount = rest.length
    const tokens = cjkCount + Math.ceil(latinCount / 4) + emojiCount * 2
    return Math.max(0, tokens)
  } catch (_) {
    // 回退：按拉丁字符近似
    return Math.ceil(String(text || '').length / 4)
  }
}

/**
 * 根据模型估算单张图片的 token 数量
 * 按常见多模态模型族粗略估算
 * @param {string} modelId - 模型 ID
 * @returns {number} 估算的图片 token 数量
 */
export function getImageTokenEstimate(modelId) {
  try {
    const s = String(modelId || '').toLowerCase()
    if (/gpt-?4o|gpt-?4-?vision|4o-mini/.test(s)) return 120
    if (/qwen[-_]?vl/.test(s)) return 256
    if (/pixtral/.test(s)) return 256
    if (/llava|minicpm-?v|internvl|idefics|xcomposer/.test(s)) return 384
    if (/gemma3|gemma[-_]?3/.test(s)) return 256
    return 512
  } catch (_) {
    return 512
  }
}

/**
 * 估算消息上下文的 token 数量
 * 消息包装开销 + 文本 + 图片
 * @param {Array} messages - 消息数组
 * @param {string} modelId - 模型 ID
 * @returns {number} 估算的总 token 数量
 */
export function estimateTokens(messages, modelId) {
  if (!Array.isArray(messages)) return 0
  let sum = 0
  const perImage = getImageTokenEstimate(modelId)
  for (const m of messages) {
    if (!m) continue
    // Chat message 包装开销（经验值）
    sum += 4
    if (typeof m.content === 'string') sum += estimateTextTokens(m.content)
    if (Array.isArray(m.images)) sum += m.images.length * perImage
  }
  // 结尾 priming（经验值）
  sum += 2
  return sum
}

