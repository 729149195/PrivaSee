const OCR_SYSTEM_PROMPT = `You are DeepSeek-OCR, an expert vision-language assistant. Follow the user's instruction precisely. When extracting structured data, use clear markdown formatting. When extracting plain text, preserve the original reading order and spacing as much as possible. Reply in Chinese when the original content is Chinese, otherwise keep the original language.`

export const OCR_COMMANDS = [
  {
    id: 'free_ocr',
    label: '自由OCR识别',
    description: '提取图像中的文本内容',
    shortcut: '/ocr',
    instruction: 'Free OCR. Extract every piece of text from the document image and output plain text in natural reading order.'
  },
  {
    id: 'markdown',
    label: '转换为Markdown',
    description: '转换为结构化 Markdown',
    shortcut: '/markdown',
    instruction: 'Convert the document to markdown. Preserve headings, lists, tables and emphasis when appropriate.'
  },
  {
    id: 'table_extract',
    label: '表格提取',
    description: '识别并提取表格数据',
    shortcut: '/table',
    instruction: 'Extract all tables from this document and convert to clean markdown table format.'
  },
  {
    id: 'formula_extract',
    label: '公式识别',
    description: '转换为 LaTeX 格式',
    shortcut: '/formula',
    instruction: 'Extract every mathematical formula from the document and output each one in LaTeX format. Preserve symbols accurately.'
  },
  {
    id: 'visual_qa',
    label: '视觉问答',
    description: '回答图像相关问题',
    shortcut: '/qa',
    instruction: 'Provide a thorough visual description of the image and answer likely questions about its content, text, and layout.'
  },
  {
    id: 'layout_analysis',
    label: '布局分析',
    description: '分析文档布局结构',
    shortcut: '/layout',
    instruction: 'Analyze the layout of this document. Describe sections, hierarchy, columns, and any notable design elements.'
  },
  {
    id: 'key_value_extract',
    label: '键值对提取',
    description: '提取结构化键值信息',
    shortcut: '/extract',
    instruction: 'Extract all key-value pairs from this document and output them as a markdown list of "key: value" items.'
  }
]

export const OCR_COMMAND_MAP = OCR_COMMANDS.reduce((acc, item) => {
  acc[item.id] = item
  return acc
}, {})

export const getOcrCommandById = (id) => OCR_COMMAND_MAP[id] || null

export const getOcrSystemPrompt = () => OCR_SYSTEM_PROMPT

