import React, { useState, useEffect, useRef } from 'react'
import { List } from 'antd'
import { OCR_COMMANDS } from '../../utils/ocrCommands'

/**
 * 斜杠命令组件
 * 在输入框中输入 / 时显示可用的 OCR 功能菜单
 */
const SlashCommands = ({ onSelectCommand, position, visible }) => {
  const [selectedIndex, setSelectedIndex] = useState(0)
  const menuRef = useRef(null)
  const overlayRef = useRef(null)

  const commands = OCR_COMMANDS

  // 键盘导航
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!visible) return

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          setSelectedIndex((prev) => (prev + 1) % commands.length)
          break
        case 'ArrowUp':
          e.preventDefault()
          setSelectedIndex((prev) => (prev - 1 + commands.length) % commands.length)
          break
        case 'Enter':
          e.preventDefault()
          onSelectCommand(commands[selectedIndex])
          break
        case 'Escape':
          onSelectCommand(null)
          break
        default:
          break
      }
    }

    if (visible) {
      window.addEventListener('keydown', handleKeyDown)
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [visible, selectedIndex, onSelectCommand])

  // 确保选中项可见（移除滚动逻辑，因为现在不滚动）
  useEffect(() => {
    // 菜单现在不滚动，所有项目都可见，不需要额外逻辑
  }, [selectedIndex, visible])

  // 点击遮罩层关闭
  const handleOverlayClick = (e) => {
    if (e.target === overlayRef.current) {
      onSelectCommand(null)
    }
  }

  if (!visible) {
    return null
  }

  return (
    <>
      {/* 遮罩层 */}
      <div
        ref={overlayRef}
        onClick={handleOverlayClick}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 9998,
          background: 'transparent'
        }}
      />
      
      {/* 命令菜单 */}
      <div
        ref={menuRef}
        style={{
          position: 'fixed',
          top: `${position?.top || 0}px`,
          left: `${position?.left || 0}px`,
          zIndex: 9999,
          background: '#fff',
          border: '1px solid #d9d9d9',
          borderRadius: '6px',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
          minWidth: '100px',
          maxWidth: '280px'
        }}
      >
        <List
          size="small"
          dataSource={commands}
          style={{
            margin: 0,
            padding: '4px 0'
          }}
          renderItem={(item, index) => (
            <List.Item
              data-index={index}
              onClick={() => onSelectCommand(item)}
              onMouseEnter={() => setSelectedIndex(index)}
              style={{
                cursor: 'pointer',
                padding: '0',
                margin: '0 4px',
                borderRadius: '4px',
                background: index === selectedIndex ? '#f5f5f5' : 'transparent',
                transition: 'background 0.2s',
                border: 'none'
              }}
            >
              <div style={{
                padding: '8px 12px',
                whiteSpace: 'nowrap',
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: '50px'
              }}>
                <div style={{
                  fontSize: '13px',
                  fontWeight: 500,
                  color: '#262626',
                  marginBottom: '2px',
                  textAlign: 'center'
                }}>
                  {item.label}
                </div>
                <div style={{
                  fontSize: '12px',
                  color: '#8c8c8c',
                  lineHeight: '1.4',
                  textAlign: 'center'
                }}>
                  {item.description}
                </div>
              </div>
            </List.Item>
          )}
        />
      </div>
    </>
  )
}

export default SlashCommands
