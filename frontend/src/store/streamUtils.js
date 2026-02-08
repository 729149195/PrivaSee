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
  let finishEmitted = false
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
        if (payload === '[DONE]') { finishEmitted = true; await onDelta({ content: '', reasoning: '', finish: 'stop' }); return }
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
          // 必须 await 异步回调，确保错误能正确传播
          if (contentDelta || reasoningDelta) await onDelta({ content: contentDelta, reasoning: reasoningDelta, finish: null })
          if (finish) { finishEmitted = true; await onDelta({ content: '', reasoning: '', finish }) }
        } catch (_) {
          // 忽略不可解析的行
        }
      }
    }
  }
  // 流结束后处理残留 buffer
  if (buffer.trim()) {
    const trimmed = buffer.trim()
    if (trimmed.startsWith('data:')) {
      const payload = trimmed.slice('data:'.length).trim()
      if (payload === '[DONE]') { finishEmitted = true; await onDelta({ content: '', reasoning: '', finish: 'stop' }); return }
      try {
        const json = JSON.parse(payload)
        const choice = json?.choices?.[0]
        const contentDelta = choice?.delta?.content ?? ''
        const reasoningDelta = choice?.delta?.reasoning_content ?? choice?.delta?.reasoning ?? ''
        const finish = choice?.finish_reason || null
        if (contentDelta || reasoningDelta) await onDelta({ content: contentDelta, reasoning: reasoningDelta, finish: null })
        if (finish) { finishEmitted = true; await onDelta({ content: '', reasoning: '', finish }) }
      } catch (_) { /* 忽略 */ }
    }
  }
  // 兜底：如果流结束但从未发出 finish 信号，补发一个
  if (!finishEmitted) {
    await onDelta({ content: '', reasoning: '', finish: 'stop' })
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
  let finishEmitted = false

  const processJsonLine = async (jsonStr) => {
    const json = JSON.parse(jsonStr)
    // 检测 Ollama 错误消息（如模型崩溃、OOM 等）
    if (json?.error) {
      console.error('[streamOllamaChatResponse] Ollama error:', json.error)
      throw new Error(json.error)
    }
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
      // 必须 await 异步回调，确保错误能正确传播
      if (delta) await onDelta({ content: delta, reasoning: '', finish: null })
    }
    if (finish) { finishEmitted = true; await onDelta({ content: '', reasoning: '', finish }) }
  }

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
        await processJsonLine(trimmed)
      } catch (e) {
        // 如果是 Ollama 错误，向上抛出让调用方处理
        if (e?.message && !e.message.startsWith('Unexpected') && !e.message.startsWith('JSON')) throw e
        // 忽略 JSON 解析错误
      }
    }
  }
  // 流结束后处理残留 buffer（最后一条消息可能没有尾部换行符）
  if (buffer.trim()) {
    try {
      await processJsonLine(buffer.trim())
    } catch (e) {
      if (e?.message && !e.message.startsWith('Unexpected') && !e.message.startsWith('JSON')) throw e
    }
  }
  // 兜底：如果流结束但从未发出 finish 信号，补发一个
  if (!finishEmitted) {
    await onDelta({ content: '', reasoning: '', finish: 'stop' })
  }
}
