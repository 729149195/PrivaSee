import React from 'react'
import { FileTextOutlined } from '@ant-design/icons'

/**
 * 文档标签组件
 * @param {object} file - 文件数据 {id, name, size, type, dataUrl}
 * @param {function} onClick - 点击回调（打开预览）
 */
const DocumentTag = ({ file, onClick }) => {
  const getFileSizeText = (fileData) => {
    const bytes = fileData.size || 0
    if (bytes < 1024) return `${bytes}B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
    return `${(bytes / (1024 * 1024)).toFixed(2)}MB`
  }

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '6px 12px',
        borderRadius: '6px',
        backgroundColor: 'rgba(24, 144, 255, 0.08)',
        border: '1px solid rgba(24, 144, 255, 0.2)',
        cursor: onClick ? 'pointer' : 'default',
        marginRight: '8px',
        marginBottom: '6px',
        transition: 'all 0.2s ease',
      }}
      onClick={onClick}
      onMouseEnter={(e) => {
        if (onClick) {
          e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.12)'
          e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.3)'
        }
      }}
      onMouseLeave={(e) => {
        if (onClick) {
          e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.08)'
          e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.2)'
        }
      }}
    >
      <FileTextOutlined style={{ marginRight: '6px', color: '#1890ff', fontSize: '14px' }} />
      <span
        style={{
          fontSize: '12px',
          color: '#262626',
          fontWeight: 500,
          userSelect: 'none'
        }}
      >
        {file.name}
      </span>
      <span style={{
        fontSize: '11px',
        color: '#8c8c8c',
        marginLeft: '4px',
        userSelect: 'none',
      }}>
        ({getFileSizeText(file)})
      </span>
    </div>
  )
}

export default DocumentTag

