/**
 * 内置 API 模型配置文件
 * 用于统一管理所有自定义 API 模型
 * 
 * 配置格式：
 * {
 *   'model-id': {
 *     url: 'API基础地址（不包含 /chat/completions）',
 *     model: '实际调用 API 时使用的模型 ID',
 *     apikey: 'API密钥',
 *     description: '模型描述'
 *   }
 * }
 */

export const DEFAULT_API_MODELS = {
  // DeepSeek Chat 模型
  'deepseek-chat': {
    url: 'https://api.deepseek.com/v1',
    model: 'deepseek-chat',
    apikey: 'sk-8c2ee9474f2f44f5969dcd5de280e634',
    description: 'DeepSeek Chat - 通用对话和推理',
    contextLength: 32768
  },
  
  // DeepSeek OCR 模型（本地部署）
  // 注意：需要先启动本地 DeepSeek-OCR 服务器（端口5001）
  // 运行: bash backend/start_deepseek_ocr_new.sh
  'deepseek-ocr-local': {
    url: 'http://localhost:5001/api',
    model: 'deepseek-ai/DeepSeek-OCR',
    apikey: '',
    description: 'DeepSeek OCR 本地部署 （支持多种文档处理功能）',
    contextLength: 32768,
    capabilities: ['ocr', 'markdown', 'table', 'formula', 'visual_qa']
  },

  // DeepSeek OCR 模型（使用 SiliconFlow - 尝试不同的模型名称）
  'deepseek-ocr': {
    url: 'https://api.siliconflow.cn/v1',
    model: 'Qwen/Qwen2-VL-7B-Instruct',
    apikey: 'sk-tjsfubvyogeavgnopvuupghnpdanakzxxsrnqfyxkchadcpc',
    description: 'DeepSeek OCR （基于 Qwen2-VL，支持文档处理）',
    contextLength: 32768,
    capabilities: ['ocr', 'markdown', 'table', 'formula', 'visual_qa']
  },
  
  // Qwen Flash 模型
  'qwen-flash': {
    url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'qwen-turbo-latest',
    apikey: 'sk-050b8f5117124731a5c962e5890500aa',
    description: 'Qwen Flash - 快速响应模型',
    contextLength: 131072
  },
  
  // Qwen3 VL 8B 模型
  'qwen3-vl-8b-instruct': {
    url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'qwen-vl-max-latest',
    apikey: 'sk-050b8f5117124731a5c962e5890500aa',
    description: 'Qwen3 VL 8B - 视觉语言模型',
    contextLength: 32768
  },
  
  // Qwen 2.5 Omni 7B 模型
  'qwen2.5-omni-7b': {
    url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'qwen2.5-omni-7b-instruct',
    apikey: 'sk-050b8f5117124731a5c962e5890500aa',
    description: 'Qwen 2.5 Omni 7B - 多模态模型',
    contextLength: 32768
  },
}

/**
 * 获取所有内置 API 模型配置（转换为 customProviders 格式）
 * @returns {Object} API 模型配置对象，格式：{ modelId: { baseUrl, apiKey, modelId, description, contextLength } }
 */
export const getDefaultApiModels = () => {
  const result = {}

  Object.entries(DEFAULT_API_MODELS).forEach(([id, config]) => {
    result[id] = {
      baseUrl: config.url,
      apiKey: config.apikey,
      modelId: config.model,
      description: config.description,
      contextLength: config.contextLength
    }
  })

  return result
}

/**
 * 获取特定模型的配置
 * @param {string} modelId - 模型 ID
 * @returns {Object|null} 模型配置或 null
 */
export const getApiModelConfig = (modelId) => {
  const config = DEFAULT_API_MODELS[modelId]
  if (!config) return null

  return {
    baseUrl: config.url,
    apiKey: config.apikey,
    modelId: config.model,
    description: config.description,
    contextLength: config.contextLength
  }
}

/**
 * 获取所有内置 API 模型的 ID 列表
 * @returns {string[]} 模型 ID 数组
 */
export const getDefaultApiModelIds = () => {
  return Object.keys(DEFAULT_API_MODELS)
}

