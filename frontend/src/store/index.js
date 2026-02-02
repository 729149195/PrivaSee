/**
 * Store 模块统一导出入口
 */

// 基础工具函数
export {
  generateId,
  tryParseJSON,
  extractFirstJSONObject,
  computeHashId,
  normalizeInfonOutput,
  createEmptySession
} from './utils.js'

// 流处理工具
export {
  streamOpenAIResponse,
  streamOllamaChatResponse
} from './streamUtils.js'

// 信息元解析器
export {
  incrementalExtractInfons,
  incrementalExtractInfonsJSON,
  parsePartialInfon,
  extractInfonFieldValue
} from './infonParser.js'

// 信息元合并/去重
export {
  deduplicateAndMergeInfons,
  findConflictingInfons,
  isSameSubject
} from './infonMerge.js'

// 信息元提取辅助函数
export {
  getExistingInfons,
  createInfonRun,
  createStreamHandler,
  executeInfonRequest,
  getModelApiConfig
} from './slices/infonHelpers.js'

// 消息发送辅助函数
export {
  createUserMessage,
  createAssistantMessage,
  createThinkTagHandler,
  getModelConfig,
  executeStreamingChat,
  stripDataUrl,
  buildOllamaHistory,
  createOllamaStreamHandler
} from './slices/messageHelpers.js'

// 隐私推理辅助函数
export {
  createPrivacyRiskParser,
  incrementalExtractRisks,
  extractKeywordsFromRisks,
  parsePrivacyBuffer,
  buildPrivacyInferencePrompt,
  collectInfonsForInference,
  // 状态更新辅助函数
  mergeRisks,
  cleanAndParseBuffer,
  parseSSELine
} from './slices/privacyHelpers.js'
