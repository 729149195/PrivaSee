import React from 'react'

/**
 * 功能命令标签组件
 * @param {object} command - 命令数据 {id, label}
 */
const CommandTag = ({ command }) => {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '6px 12px',
        borderRadius: '16px',
        backgroundColor: 'rgba(24, 144, 255, 0.08)',
        border: '1px solid rgba(24, 144, 255, 0.2)',
        marginRight: '8px',
        marginBottom: '6px',
      }}
    >
      <span style={{
        fontSize: '13px',
        color: '#1890ff',
        fontWeight: 500,
        userSelect: 'none'
      }}>
        {command.label}
      </span>
    </div>
  )
}

export default CommandTag

