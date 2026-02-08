/**
 * 默认模型配置文件
 * 用于统一管理各个模式和共用模型的默认配置
 * 修改此文件可以更改系统的默认模型设置
 */

export const DEFAULT_MODELS_CONFIG = {
  // 对话模型的默认配置
  conversation: {
    model: 'niels32167/qwen3-4b-instruct:latest', // 默认对话模型
  },

  // 提取信息元模式的默认模型配置
  infonExtraction: {
    extractionModel: 'niels32167/qwen3-4b-instruct:latest', // 提取信息元模式：信息元提取模型
    privacyInferenceModel: 'niels32167/qwen3-4b-instruct:latest', // 提取信息元模式：隐私推理模型
  },

  // 共用模型的默认配置
  shared: {
    imageParsingModel: 'qwen2.5vl:7b', // 图片解析模型（共用）- Qwen 2.5 VL 7B，适配4070(12GB显存)，仅占约6GB
    protectionSuggestionModel: 'niels32167/qwen3-4b-instruct:latest', // Privacy Protection Suggestions模型（共用）
  },
}

/**
 * 获取默认模型配置（用于重置）
 * @returns {Object} 默认配置对象
 */
export const getDefaultModelsConfig = () => {
  return {
    conversationModel: DEFAULT_MODELS_CONFIG.conversation.model,
    infonExtractionModel: DEFAULT_MODELS_CONFIG.infonExtraction.extractionModel,
    infonPrivacyInferenceModel: DEFAULT_MODELS_CONFIG.infonExtraction.privacyInferenceModel,
    imageParsingModel: DEFAULT_MODELS_CONFIG.shared.imageParsingModel,
    protectionSuggestionModel: DEFAULT_MODELS_CONFIG.shared.protectionSuggestionModel,
  }
}

