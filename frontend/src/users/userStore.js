import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

// 用户状态管理
export const useUserStore = create(
  persist(
    (set, get) => ({
  // 用户信息
  currentUser: null, // { id, username, avatar?, createdAt }
  isLoggedIn: false,

      // 登录
      login: (user) => {
        set({
          currentUser: {
            ...user,
            avatar: user.avatar || generateAvatar(user.username),
            loginAt: Date.now()
          },
          isLoggedIn: true
        })
      },

  // 注册（简化版本，实际应该调用后端API）
  register: (username, password) => {
    const newUser = {
      id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      username,
      createdAt: Date.now(),
      avatar: generateAvatar(username)
    }
    
    // 保存到本地用户数据库
    const users = JSON.parse(localStorage.getItem('privasee_users') || '{}')
    users[username] = {
      ...newUser,
      password: hashPassword(password) // 简单哈希（生产环境应使用后端加密）
    }
    localStorage.setItem('privasee_users', JSON.stringify(users))
    
    set({
      currentUser: newUser,
      isLoggedIn: true
    })
  },

      // 退出
      logout: () => {
        set({
          currentUser: null,
          isLoggedIn: false
        })
      },

      // 更新用户信息
      updateUser: (updates) => {
        set((state) => ({
          currentUser: state.currentUser ? { ...state.currentUser, ...updates } : null
        }))
      }
    }),
    {
      name: 'privasee-user-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        currentUser: state.currentUser,
        isLoggedIn: state.isLoggedIn
      })
    }
  )
)

// 简单的密码哈希函数（生产环境应使用更安全的方式）
function hashPassword(password) {
  let hash = 0
  for (let i = 0; i < password.length; i++) {
    const char = password.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash
  }
  return hash.toString(36)
}

// 生成用户头像（使用 DiceBear API 或纯色头像）
function generateAvatar(username) {
  // 使用 DiceBear Avatars API（免费的头像生成服务）
  const styles = ['adventurer', 'avataaars', 'bottts', 'fun-emoji', 'identicon', 'lorelei', 'micah', 'miniavs', 'notionists', 'open-peeps', 'personas', 'pixel-art']
  const style = styles[Math.abs(simpleHash(username)) % styles.length]
  return `https://api.dicebear.com/7.x/${style}/svg?seed=${encodeURIComponent(username)}`
}

// 简单哈希函数
function simpleHash(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i)
    hash = hash & hash
  }
  return hash
}

// 验证用户登录
export function validateLogin(username, password) {
  const users = JSON.parse(localStorage.getItem('privasee_users') || '{}')
  const user = users[username]
  
  if (!user) {
    return { success: false, error: '用户不存在' }
  }
  
  if (user.password !== hashPassword(password)) {
    return { success: false, error: '密码错误' }
  }
  
  return {
    success: true,
    user: {
      id: user.id,
      username: user.username,
      avatar: user.avatar,
      createdAt: user.createdAt
    }
  }
}

