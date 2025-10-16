import React from 'react'
import { Input, Popconfirm } from 'antd'
import { EditOutlined, DeleteOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'

/**
 * 会话列表项组件
 * @param {object} session - 会话对象
 * @param {string} currentSessionId - 当前活跃会话 ID
 * @param {string} editingSessionId - 正在编辑的会话 ID
 * @param {string} editingTitle - 编辑中的标题
 * @param {string} draggingSessionId - 正在拖拽的会话 ID
 * @param {function} onSwitch - 切换会话的回调
 * @param {function} onRename - 重命名会话的回调
 * @param {function} onDelete - 删除会话的回调
 * @param {function} onEditStart - 开始编辑的回调
 * @param {function} onEditEnd - 结束编辑的回调
 * @param {function} setEditingTitle - 设置编辑标题的回调
 * @param {function} onDragStart - 拖拽开始的回调
 * @param {function} onDragOver - 拖拽中的回调
 * @param {function} onDrop - 放置的回调
 * @param {function} onDragEnd - 拖拽结束的回调
 * @param {function} setRef - 设置 ref 的回调
 */
const ChatSessionItem = ({
  session,
  currentSessionId,
  editingSessionId,
  editingTitle,
  draggingSessionId,
  onSwitch,
  onRename,
  onDelete,
  onEditStart,
  onEditEnd,
  setEditingTitle,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
  setRef
}) => {
  const isActive = session.id === currentSessionId
  const isEditing = editingSessionId === session.id
  const isDragging = draggingSessionId === session.id

  const handleEditComplete = () => {
    const newTitle = editingTitle.trim()
    if (newTitle && newTitle !== session.title) {
      onRename(session.id, newTitle)
    }
    onEditEnd()
  }

  return (
    <div
      className={`${styles.chatItem} ${isActive ? styles.chatItemActive : ''}`}
      onClick={() => {
        if (!isEditing) {
          onSwitch(session.id)
        }
      }}
      title={isEditing ? '' : session.title}
      draggable
      onDragStart={onDragStart(session.id)}
      onDragOver={onDragOver(session.id)}
      onDrop={onDrop(session.id)}
      onDragEnd={onDragEnd}
      ref={(el) => setRef(session.id, el)}
      style={{ opacity: isDragging ? 0.8 : 1 }}
    >
      <div className={styles.chatItemHeader}>
        <div className={styles.chatItemInfo}>
          {isEditing ? (
            <Input
              className={styles.chatNameInput}
              value={editingTitle}
              onChange={(e) => setEditingTitle(e.target.value)}
              onPressEnter={(e) => {
                e.stopPropagation()
                handleEditComplete()
              }}
              onBlur={handleEditComplete}
              onClick={(e) => e.stopPropagation()}
              autoFocus
              size="small"
            />
          ) : (
            <div className={styles.chatName}>{session.title}</div>
          )}
          {!isEditing && (
            <div className={styles.chatMeta}>{new Date(session.updatedAt).toLocaleString()}</div>
          )}
        </div>
        <div className={styles.chatActions}>
          <button 
            className={styles.iconBtn} 
            onClick={(e) => { 
              e.stopPropagation()
              onEditStart(session.id, session.title)
            }} 
            title="Rename"
          >
            <EditOutlined />
          </button>
          <Popconfirm
            title="删除对话"
            description="确定要删除这个对话吗？"
            onConfirm={(e) => {
              e?.stopPropagation()
              onDelete(session.id)
            }}
            onCancel={(e) => e?.stopPropagation()}
            okText="删除"
            cancelText="取消"
            placement="right"
          >
            <button 
              className={styles.iconBtn} 
              onClick={(e) => e.stopPropagation()} 
              title="Delete"
            >
              <DeleteOutlined />
            </button>
          </Popconfirm>
        </div>
      </div>
    </div>
  )
}

export default ChatSessionItem

