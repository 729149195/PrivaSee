import React, { useState } from 'react'
import { EyeInvisibleOutlined, CloseOutlined } from '@ant-design/icons'
import { useUserStore } from './userStore'

// 无痕模式指示器组件（中文注释）：在未登录时显示提示
export default function PrivacyModeIndicator() {
  const { isLoggedIn } = useUserStore()
  const [dismissed, setDismissed] = useState(false)

  // 只在未登录时显示
  if (isLoggedIn || dismissed) return null

  return (
    <div style={{
      margin: '8px 12px',
      padding: '6px 10px',
      display: 'flex',
      alignItems: 'center',
      gap: 6,
      fontSize: 12,
      color: '#64748b',
      background: '#f1f5f9',
      borderRadius: 8,
      lineHeight: 1.4,
    }}>
      <EyeInvisibleOutlined style={{ fontSize: 13, color: '#94a3b8', flexShrink: 0 }} />
      <span style={{ flex: 1 }}>无痕模式 · 对话不会保存</span>
      <CloseOutlined
        onClick={() => setDismissed(true)}
        style={{
          fontSize: 10,
          color: '#94a3b8',
          cursor: 'pointer',
          flexShrink: 0,
          padding: 2,
        }}
      />
    </div>
  )
}

