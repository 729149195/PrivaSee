// 会话标题生成提示词模板

export const SESSION_TITLE_SYSTEM_PROMPT = '你是一个对话标题生成助手。请根据用户的第一条消息，生成一个简短的对话标题（5-10个字）。只输出标题本身，不要有任何解释或标点符号。'

export function buildSessionTitleUserPrompt(content) {
  return `请为以下消息生成一个简短的对话标题：\n\n${content}`
}

// 清理生成的标题
export function cleanGeneratedTitle(title, maxLength = 50) {
  if (!title) return null
  return title
    .replace(/^["'「『]+|["'」』]+$/g, '')
    .replace(/\n/g, ' ')
    .slice(0, maxLength)
}

// 为 DeepSeek OCR 消息生成标题
export function generateOcrTitle(commands, files) {
  const commandLabel = commands?.[0]?.label || 'OCR'
  let docName = ''
  
  if (files.length === 1) {
    docName = files[0].name.replace(/\.[^/.]+$/, '')
  } else {
    docName = files[0].name.replace(/\.[^/.]+$/, '') + '等'
  }
  
  return `${commandLabel}-${docName}`.slice(0, 50)
}
