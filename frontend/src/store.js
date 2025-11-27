// PrivaSee Store - 主文件
// 使用 Zustand slice 模式组合各功能模块
import { create } from 'zustand'

// 导入 Slices
import { createConfigSlice } from './store/slices/configSlice'
import { createSessionSlice } from './store/slices/sessionSlice'
import { createInfonSlice } from './store/slices/infonSlice'
import { createPrivacySlice } from './store/slices/privacySlice'
import { createProtectionSlice } from './store/slices/protectionSlice'
import { createUserSlice } from './store/slices/userSlice'
import { createMessageSlice } from './store/slices/messageSlice'

export const useStore = create((set, get) => ({
  // 组合所有 Slices
  ...createConfigSlice(set, get),
  ...createSessionSlice(set, get),
  ...createInfonSlice(set, get),
  ...createPrivacySlice(set, get),
  ...createProtectionSlice(set, get),
  ...createUserSlice(set, get),
  ...createMessageSlice(set, get),
}))

// 自动保存
if (typeof window !== 'undefined') {
  let autoSaveTimer = null
  useStore.subscribe(state => {
    if (autoSaveTimer) clearTimeout(autoSaveTimer)
    if (state.currentUserId) {
      autoSaveTimer = setTimeout(() => useStore.getState().saveCurrentUserHistory(), 30000)
    }
  })
  window.addEventListener('beforeunload', () => {
    const state = useStore.getState()
    if (state.currentUserId) state._saveUserHistory(state.currentUserId)
  })
}
