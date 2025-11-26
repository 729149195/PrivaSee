/**
 * 流式响应处理工具模块
 * 处理 OpenAI 和 Ollama 的 SSE 流式响应
 */

/**
 * 解析 OpenAI SSE 流：将 response.body 按行解析 data: 片段
 * @param {ReadableStreamDefaultReader} reader - 流读取器
 * @param {Function} onDelta - 回调函数，接收 {content, reasoning, finish}
 */
export async function streamOpenAIResponse(reader, onDelta) {
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

/**
 * 解析 Ollama /api/chat 流：逐行解析 JSON，并处理"全量快照"或"增量 token"两种格式
 * @param {ReadableStreamDefaultReader} reader - 流读取器
 * @param {Function} onDelta - 回调函数，接收 {content, reasoning, finish}
 */
export async function streamOllamaChatResponse(reader, onDelta) {
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
