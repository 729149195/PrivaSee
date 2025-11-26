// 配置管理 Slice
import { getDefaultModelsConfig } from '../../config/defaultModelsConfig'
import { getDefaultApiModels, getDefaultApiModelIds } from '../../config/defaultApiModelsConfig'
import { IMAGE_ANALYSIS_SYSTEM_PROMPT, IMAGE_ANALYSIS_USER_PROMPT, getImageAnalysisMaxTokens } from '../../templates/imageAnalysis.js'

export const createConfigSlice = (set, get) => ({
  // 基础配置
  baseUrl: '/v1',
  model: getDefaultModelsConfig().conversationModel,
  models: [...getDefaultApiModelIds()],
  customModels: [...getDefaultApiModelIds()],
  customProviders: getDefaultApiModels(),
  
  // 模型配置
  directInferenceModel: getDefaultModelsConfig().directInferenceModel,
  infonExtractionModel: getDefaultModelsConfig().infonExtractionModel,
  infonPrivacyInferenceModel: getDefaultModelsConfig().infonPrivacyInferenceModel,
  imageParsingModel: getDefaultModelsConfig().imageParsingModel,
  protectionSuggestionModel: getDefaultModelsConfig().protectionSuggestionModel,
  inferenceMode: getDefaultModelsConfig().inferenceMode,
  autoPrivacyInference: true,
  
  // Pending 状态
  pendingUserInput: '',
  pendingAudios: [],
  pendingImages: [],
  ocrFileObjects: {},

  // Setters
  setDirectInferenceModel: (modelId) => set({ directInferenceModel: modelId }),
  setInfonExtractionModel: (modelId) => set({ infonExtractionModel: modelId }),
  setInfonPrivacyInferenceModel: (modelId) => set({ infonPrivacyInferenceModel: modelId }),
  setImageParsingModel: (modelId) => set({ imageParsingModel: modelId }),
  setProtectionSuggestionModel: (modelId) => set({ protectionSuggestionModel: modelId }),
  
  resetToDefaultModels: () => {
    const d = getDefaultModelsConfig()
    set({
      directInferenceModel: d.directInferenceModel,
      infonExtractionModel: d.infonExtractionModel,
      infonPrivacyInferenceModel: d.infonPrivacyInferenceModel,
      imageParsingModel: d.imageParsingModel,
      protectionSuggestionModel: d.protectionSuggestionModel,
      inferenceMode: d.inferenceMode,
    })
  },
  
  setInferenceMode: (mode) => {
    const session = get().getCurrentSession()
    if (session?.id) {
      const box = get().infonSessions?.[session.id]
      if (box) {
        const runs = (box.runs || []).filter(r => r.targetType !== 'pending')
        set(s => ({ infonSessions: { ...s.infonSessions, [session.id]: { ...box, runs } } }))
      }
    }
    set({ inferenceMode: mode })
  },
  
  setAutoPrivacyInference: (enabled) => set({ autoPrivacyInference: enabled }),
  setPendingUserInput: (input) => set({ pendingUserInput: input }),
  setPendingAudios: (audios) => set({ pendingAudios: audios }),
  setPendingImages: (images) => set({ pendingImages: images }),
  
  setModel(modelId) { set({ model: modelId }) },

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
        directInferenceModel: s.directInferenceModel === id ? 'deepseek-chat' : s.directInferenceModel,
        infonExtractionModel: s.infonExtractionModel === id ? 'deepseek-chat' : s.infonExtractionModel,
        infonPrivacyInferenceModel: s.infonPrivacyInferenceModel === id ? 'deepseek-chat' : s.infonPrivacyInferenceModel,
        imageParsingModel: s.imageParsingModel === id ? 'gemma3:12b' : s.imageParsingModel,
        protectionSuggestionModel: s.protectionSuggestionModel === id ? 'deepseek-chat' : s.protectionSuggestionModel,
      }
    })
  },

  // 图片分析
  async analyzeImage(imageDataUrl) {
    try {
      const configuredModel = get().imageParsingModel || 'gemma3:12b'
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
            { role: 'system', content: IMAGE_ANALYSIS_SYSTEM_PROMPT },
            { role: 'user', content: [
              { type: 'text', text: IMAGE_ANALYSIS_USER_PROMPT },
              { type: 'image_url', image_url: { url: imageDataUrl } }
            ]}
          ],
          temperature: 0.3,
          max_tokens: getImageAnalysisMaxTokens(configuredModel),
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
