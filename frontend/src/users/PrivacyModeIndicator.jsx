import React from 'react'
import { Alert } from 'antd'
import { EyeInvisibleOutlined } from '@ant-design/icons'
import { useUserStore } from './userStore'

// 无痕模式指示器组件（中文注释）：在未登录时显示提示
export default function PrivacyModeIndicator() {
  const { isLoggedIn } = useUserStore()
  
  // 只在未登录时显示
  if (isLoggedIn) return null
  
  return (
    <div style={{ padding: '8px 16px' }}>
      <Alert
        message="无痕模式"
        description="当前未登录，对话历史不会保存。刷新页面后对话将消失。"
        type="info"
        icon={<EyeInvisibleOutlined />}
        showIcon
        closable
        style={{
          fontSize: '12px',
          borderRadius: '8px'
        }}
      />
    </div>
  )
}

