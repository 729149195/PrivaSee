import React from 'react'
import { FileTextOutlined } from '@ant-design/icons'

/**
 * 文档标签组件
 * @param {object} file - 文件数据 {id, name, size, type, dataUrl}
 * @param {function} onClick - 点击回调（打开预览）
 * @param {object} command - 命令数据（可选，会嵌入在文档标签内）
 */
const DocumentTag = ({ file, onClick, command }) => {
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
        padding: '8px 14px',
        borderRadius: '8px',
        backgroundColor: 'rgba(24, 144, 255, 0.06)',
        border: '1px solid rgba(24, 144, 255, 0.15)',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.03)',
        gap: '10px',
      }}
      onClick={onClick}
      onMouseEnter={(e) => {
        if (onClick) {
          e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.1)'
          e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.25)'
          e.currentTarget.style.boxShadow = '0 2px 8px rgba(24, 144, 255, 0.12)'
          e.currentTarget.style.transform = 'translateY(-1px)'
        }
      }}
      onMouseLeave={(e) => {
        if (onClick) {
          e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.06)'
          e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.15)'
          e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.03)'
          e.currentTarget.style.transform = 'translateY(0)'
        }
      }}
    >
      {/* 文档图标 */}
      <FileTextOutlined style={{ color: '#1890ff', fontSize: '16px', flexShrink: 0 }} />
      
      {/* 文档信息 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 }}>
        <span
          style={{
            fontSize: '13px',
            color: '#262626',
            fontWeight: 500,
            userSelect: 'none',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {file.name}
        </span>
        <span style={{
          fontSize: '11px',
          color: '#8c8c8c',
          userSelect: 'none',
        }}>
          {getFileSizeText(file)}
        </span>
      </div>
      
      {/* 嵌入的命令标签 */}
      {command && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            padding: '4px 10px',
            borderRadius: '12px',
            backgroundColor: 'rgba(24, 144, 255, 0.12)',
            border: '1px solid rgba(24, 144, 255, 0.25)',
            marginLeft: 'auto',
            flexShrink: 0,
          }}
        >
          <span style={{
            fontSize: '12px',
            color: '#1890ff',
            fontWeight: 600,
            userSelect: 'none',
            whiteSpace: 'nowrap',
          }}>
            {command.label}
          </span>
        </div>
      )}
    </div>
  )
}

export default DocumentTag

