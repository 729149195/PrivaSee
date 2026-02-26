// 配置管理 Slice
import { getDefaultModelsConfig } from '../../config/defaultModelsConfig'
import { getDefaultApiModels, getDefaultApiModelIds } from '../../config/defaultApiModelsConfig'
import { getImageAnalysisSystemPrompt, IMAGE_ANALYSIS_USER_PROMPT, getImageAnalysisMaxTokens } from '../../templates/imageAnalysis.js'

export const createConfigSlice = (set, get) => ({
  // 基础配置
  baseUrl: '/v1',
  model: getDefaultModelsConfig().conversationModel,
  conversationThinkMode: getDefaultModelsConfig().conversationThinkMode,
  models: [...getDefaultApiModelIds()],
  customModels: [...getDefaultApiModelIds()],
  customProviders: getDefaultApiModels(),
  
  // 模型配置
  infonExtractionModel: getDefaultModelsConfig().infonExtractionModel,
  infonExtractionThinkMode: getDefaultModelsConfig().infonExtractionThinkMode,
  infonPrivacyInferenceModel: getDefaultModelsConfig().infonPrivacyInferenceModel,
  infonPrivacyInferenceThinkMode: getDefaultModelsConfig().infonPrivacyInferenceThinkMode,
  imageParsingModel: getDefaultModelsConfig().imageParsingModel,
  imageParsingThinkMode: getDefaultModelsConfig().imageParsingThinkMode,
  protectionSuggestionModel: getDefaultModelsConfig().protectionSuggestionModel,
  protectionSuggestionThinkMode: getDefaultModelsConfig().protectionSuggestionThinkMode,
  autoPrivacyInference: true,
  
  // OCR 文件对象
  ocrFileObjects: {},

  // Setters
  setInfonExtractionModel: (modelId) => set({ infonExtractionModel: modelId }),
  setInfonExtractionThinkMode: (enabled) => set({ infonExtractionThinkMode: !!enabled }),
  setInfonPrivacyInferenceModel: (modelId) => set({ infonPrivacyInferenceModel: modelId }),
  setInfonPrivacyInferenceThinkMode: (enabled) => set({ infonPrivacyInferenceThinkMode: !!enabled }),
  setImageParsingModel: (modelId) => set({ imageParsingModel: modelId }),
  setImageParsingThinkMode: (enabled) => set({ imageParsingThinkMode: !!enabled }),
  setProtectionSuggestionModel: (modelId) => set({ protectionSuggestionModel: modelId }),
  setProtectionSuggestionThinkMode: (enabled) => set({ protectionSuggestionThinkMode: !!enabled }),
  
  resetToDefaultModels: () => {
    const d = getDefaultModelsConfig()
    set({
      conversationThinkMode: d.conversationThinkMode,
      infonExtractionModel: d.infonExtractionModel,
      infonExtractionThinkMode: d.infonExtractionThinkMode,
      infonPrivacyInferenceModel: d.infonPrivacyInferenceModel,
      infonPrivacyInferenceThinkMode: d.infonPrivacyInferenceThinkMode,
      imageParsingModel: d.imageParsingModel,
      imageParsingThinkMode: d.imageParsingThinkMode,
      protectionSuggestionModel: d.protectionSuggestionModel,
      protectionSuggestionThinkMode: d.protectionSuggestionThinkMode,
    })
  },
  
  setAutoPrivacyInference: (enabled) => set({ autoPrivacyInference: enabled }),
  
  setModel(modelId) { set({ model: modelId }) },
  setConversationThinkMode: (enabled) => set({ conversationThinkMode: !!enabled }),

  async fetchModels() {
    try {
      const res = await fetch(`${get().baseUrl}/models`, { method: 'GET' })
      const json = await res.json().catch(() => ({}))
      let list = []
      if (Array.isArray(json?.data)) list = json.data.map(m => m?.id).filter(Boolean)
      else if (Array.isArray(json?.models)) list = json.models.map(m => m?.id || m?.name || m).filter(Boolean)
      else if (Array.isArray(json)) list = json.map(m => m?.id || m?.name || m).filter(Boolean)
      if (list.length) set(s => ({ models: Array.from(new Set([...(s.models || []), ...list])) }))
    } catch (_) {}
  },

  addApiModel({ id, baseUrl, apiKey }) {
    if (!id || !baseUrl || !apiKey) return
    set(s => ({
      customProviders: { ...(s.customProviders || {}), [id]: { baseUrl, apiKey } },
      customModels: Array.from(new Set([...(s.customModels || []), id])),
      models: Array.from(new Set([...(s.models || []), id]))
    }))
  },
  
  removeApiModel(id) {
    if (!id) return
    set(s => {
      const newProviders = { ...s.customProviders }
      delete newProviders[id]
      return {
        customProviders: newProviders,
        customModels: (s.customModels || []).filter(m => m !== id),
        models: (s.models || []).filter(m => m !== id),
        model: s.model === id ? getDefaultModelsConfig().conversationModel : s.model,
        conversationThinkMode: s.model === id ? false : s.conversationThinkMode,
        infonExtractionModel: s.infonExtractionModel === id ? 'deepseek-chat' : s.infonExtractionModel,
        infonExtractionThinkMode: s.infonExtractionModel === id ? false : s.infonExtractionThinkMode,
        infonPrivacyInferenceModel: s.infonPrivacyInferenceModel === id ? 'deepseek-chat' : s.infonPrivacyInferenceModel,
        infonPrivacyInferenceThinkMode: s.infonPrivacyInferenceModel === id ? false : s.infonPrivacyInferenceThinkMode,
        imageParsingModel: s.imageParsingModel === id ? 'gemma3:12b' : s.imageParsingModel,
        imageParsingThinkMode: s.imageParsingModel === id ? false : s.imageParsingThinkMode,
        protectionSuggestionModel: s.protectionSuggestionModel === id ? 'deepseek-chat' : s.protectionSuggestionModel,
        protectionSuggestionThinkMode: s.protectionSuggestionModel === id ? false : s.protectionSuggestionThinkMode,
      }
    })
  },

  // 图片分析
  async analyzeImage(imageDataUrl) {
    try {
      const configuredModel = get().imageParsingModel || 'gemma3:12b'
      const think = !!get().imageParsingThinkMode
      const provider = get().customProviders?.[configuredModel]
      const apiUrl = provider ? provider.baseUrl : get().baseUrl
      const headers = { 'Content-Type': 'application/json' }
      if (provider?.apiKey) headers['Authorization'] = `Bearer ${provider.apiKey}`
      
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: configuredModel,
          messages: [
            { role: 'system', content: getImageAnalysisSystemPrompt() },
            { role: 'user', content: [
              { type: 'text', text: IMAGE_ANALYSIS_USER_PROMPT },
              { type: 'image_url', image_url: { url: imageDataUrl } }
            ]}
          ],
          temperature: 0.3,
          max_tokens: getImageAnalysisMaxTokens(configuredModel),
          think,
        })
      })
      
      if (!response.ok) throw new Error(`Image analysis failed: ${response.statusText}`)
      const result = await response.json()
      const analysisText = result.choices?.[0]?.message?.content?.trim() || ''
      if (!analysisText) throw new Error('No analysis result returned')
      return analysisText
    } catch (error) {
      console.error('[Image Analysis] Error:', error)
      throw error
    }
  },

  // 法律与隐私项
  selectedLaw: null,
  selectedLawIdx: 0,
  customPrivacyItems: [],
  selectedPrivacyItems: [],
  
  setSelectedLaw(lawKey, lawData) { set({ selectedLaw: { key: lawKey, data: lawData } }) },
  setSelectedLawIdx(idx) { set({ selectedLawIdx: idx }) },
  addCustomPrivacyItem(item) { set(s => ({ customPrivacyItems: [...s.customPrivacyItems, item] })) },
  removeCustomPrivacyItem(itemId) { set(s => ({ customPrivacyItems: s.customPrivacyItems.filter(i => i.id !== itemId) })) },
  setCustomPrivacyItems(items) { set({ customPrivacyItems: items }) },
  setSelectedPrivacyItems(items) { set({ selectedPrivacyItems: Array.isArray(items) ? items : Array.from(items) }) },
  togglePrivacyItem(itemId) {
    set(s => {
      const sel = new Set(s.selectedPrivacyItems)
      sel.has(itemId) ? sel.delete(itemId) : sel.add(itemId)
      return { selectedPrivacyItems: Array.from(sel) }
    })
  },
})
