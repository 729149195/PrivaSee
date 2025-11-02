import React, { useState } from 'react'
import { Button, Dropdown } from 'antd'
import { DownOutlined, CloseOutlined } from '@ant-design/icons'

/**
 * 分辨率模式选项
 */
const RESOLUTION_MODES = [
  { key: 'tiny', label: '快速预览 (512px)', description: '最快速度' },
  { key: 'small', label: '标准处理 (640px)', description: '平衡速度和质量' },
  { key: 'base', label: '高质量 (1024px)', description: '高质量输出' },
  { key: 'large', label: '超高质量 (1280px)', description: '最佳质量' },
  { key: 'gundam', label: '智能裁剪 (推荐)', description: '自适应处理' },
]

/**
 * 功能命令标签组件
 * @param {object} command - 命令数据 {id, label}
 * @param {string} resolution - 当前选中的分辨率模式
 * @param {function} onResolutionChange - 分辨率变更回调
 * @param {boolean} removable - 是否可删除
 * @param {function} onRemove - 删除回调
 * @param {boolean} showResolutionSelector - 是否显示分辨率选择器（输入框中的标签才显示）
 * @param {function} onTagClick - 点击标签本身的回调（用于打开命令菜单）
 * @param {boolean} commandMenuOpen - 命令菜单是否打开（用于保持展开状态）
 */
const CommandTag = ({ 
  command, 
  resolution = 'gundam',
  onResolutionChange,
  removable = false, 
  onRemove,
  showResolutionSelector = false,
  onTagClick,
  commandMenuOpen = false
}) => {
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [hover, setHover] = useState(false)

  // 当分辨率菜单或命令菜单打开时，保持展开状态
  const isExpanded = hover || dropdownOpen || commandMenuOpen

  const currentResolution = RESOLUTION_MODES.find(m => m.key === resolution) || RESOLUTION_MODES[4]

  const menuItems = RESOLUTION_MODES.map(mode => ({
    key: mode.key,
    label: (
      <div style={{ padding: '4px 0' }}>
        <div style={{ 
          fontWeight: mode.key === resolution ? 600 : 400,
          color: mode.key === resolution ? '#1890ff' : '#262626'
        }}>
          {mode.label}
        </div>
        <div style={{ fontSize: '12px', color: '#8c8c8c', marginTop: '2px' }}>
          {mode.description}
        </div>
      </div>
    ),
    onClick: (e) => {
      // 阻止事件冒泡，防止触发标签的 onClick
      e?.domEvent?.stopPropagation()
      if (onResolutionChange) {
        onResolutionChange(mode.key)
      }
      setDropdownOpen(false)
    }
  }))

  return (
    <div
      style={{
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        padding: '6px 12px',
        paddingLeft: (showResolutionSelector && isExpanded) ? '32px' : '12px',
        paddingRight: (removable && isExpanded) ? '28px' : '12px',
        marginLeft: '8px',
        borderRadius: '16px',
        backgroundColor: isExpanded ? 'rgba(24, 144, 255, 0.12)' : 'rgba(24, 144, 255, 0.08)',
        border: isExpanded ? '1px solid rgba(24, 144, 255, 0.3)' : '1px solid rgba(24, 144, 255, 0.2)',
        transition: 'all 0.2s ease',
        cursor: onTagClick ? 'pointer' : 'default',
        whiteSpace: 'nowrap'
      }}
      onClick={(e) => {
        // 点击标签本身（不是删除按钮或分辨率按钮）时触发回调
        if (!e.target.closest('.command-delete-btn') && !e.target.closest('.ant-btn')) {
          onTagClick?.(e)
        }
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {/* 分辨率选择器箭头（仅在输入框中的标签显示） */}
      {showResolutionSelector && (
        <Dropdown
          menu={{ items: menuItems }}
          trigger={['click']}
          placement="bottomLeft"
          open={dropdownOpen}
          onOpenChange={setDropdownOpen}
        >
          <Button
            type="text"
            size="small"
            icon={<DownOutlined style={{ fontSize: '10px' }} />}
            onClick={(e) => {
              e.stopPropagation()
              setDropdownOpen(!dropdownOpen)
            }}
            style={{
              position: 'absolute',
              left: '4px',
              top: '50%',
              transform: 'translateY(-50%)',
              color: '#1890ff',
              opacity: isExpanded ? 1 : 0,
              transition: 'opacity 0.2s ease',
              padding: '2px',
              width: '20px',
              height: '20px',
              minWidth: '20px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onMouseEnter={(e) => {
              e.stopPropagation()
              e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.1)'
            }}
            onMouseLeave={(e) => {
              e.stopPropagation()
              e.currentTarget.style.backgroundColor = 'transparent'
            }}
          />
        </Dropdown>
      )}
      
      {/* 命令标签文本 */}
      <span style={{
        fontSize: '13px',
        color: '#1890ff',
        fontWeight: 500,
        userSelect: 'none'
      }}>
        {command.label}
      </span>
      
      {/* 删除按钮（仅在可删除模式下显示） */}
      {removable && (
        <Button
          type="text"
          size="small"
          icon={<CloseOutlined />}
          onClick={(e) => { 
            e.stopPropagation()
            onRemove?.()
          }}
          className="command-delete-btn"
          style={{
            position: 'absolute',
            right: '2px',
            top: '50%',
            transform: 'translateY(-50%)',
            color: '#666',
            opacity: isExpanded ? 1 : 0,
            transition: 'opacity 0.2s ease',
            padding: '4px',
            width: '24px',
            height: '24px',
            minWidth: '24px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onMouseEnter={(e) => {
            e.stopPropagation()
            e.currentTarget.style.backgroundColor = 'rgba(0, 0, 0, 0.04)'
          }}
          onMouseLeave={(e) => {
            e.stopPropagation()
            e.currentTarget.style.backgroundColor = 'transparent'
          }}
        />
      )}
    </div>
  )
}

export default CommandTag

