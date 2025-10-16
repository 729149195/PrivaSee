import React from 'react'
import { Button, Tooltip } from 'antd'
import { CopyOutlined, EditOutlined, RedoOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import MarkdownMessage from '../MarkdownMessage'
import RelationConnections from './RelationConnections'
import RelationTags from './RelationTags'
import MessageEditor from './MessageEditor'

/**
 * 消息气泡组件
 * @param {object} message - 消息对象
 * @param {boolean} isUser - 是否为用户消息
 * @param {string} editingMessageId - 正在编辑的消息 ID
 * @param {string} editingContent - 编辑中的内容
 * @param {function} setEditingContent - 设置编辑内容
 * @param {Array} editingImages - 编辑中的图片
 * @param {function} setEditingImages - 设置编辑图片
 * @param {function} onCopy - 复制的回调
 * @param {function} onEdit - 编辑的回调
 * @param {function} onSaveEdit - 保存编辑的回调
 * @param {function} onCancelEdit - 取消编辑的回调
 * @param {function} onRetry - 重试的回调
 * @param {boolean} isGenerating - 是否正在生成
 * @param {function} renderHighlightedText - 渲染高亮文本的函数
 * @param {Array} messageRelations - 消息的关系信息元
 * @param {object} infonIndex - 信息元索引
 * @param {Array} pendingHighlights - 待处理的高亮
 * @param {Array} pendingRelations - 待处理的关系
 * @param {object} pendingInfonIndex - 待处理的信息元索引
 */
const MessageBubble = ({
  message,
  isUser,
  editingMessageId,
  editingContent,
  setEditingContent,
  editingImages,
  setEditingImages,
  originalEditingContent,
  originalEditingImages,
  onCopy,
  onEdit,
  onSaveEdit,
  onCancelEdit,
  onRetry,
  isGenerating,
  renderHighlightedText,
  messageRelations,
  infonIndex,
  pendingHighlights,
  pendingRelations,
  pendingInfonIndex,
  sendLockState
}) => {
  const isEditing = editingMessageId === message.id

  if (isUser) {
    return (
      <div className={`${styles.msgRow} ${styles.rowUser}`}>
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
          {isEditing ? (
            <MessageEditor
              editingContent={editingContent}
              setEditingContent={setEditingContent}
              editingImages={editingImages}
              setEditingImages={setEditingImages}
              originalContent={originalEditingContent}
              originalImages={originalEditingImages}
              onSave={onSaveEdit}
              onCancel={onCancelEdit}
              pendingHighlights={pendingHighlights}
              pendingRelations={pendingRelations}
              pendingInfonIndex={pendingInfonIndex}
              sendLockState={sendLockState}
            />
          ) : (
            <>
              <div className={`${styles.msgBubble} ${styles.msgBubbleUser}`} style={{ position: 'relative' }}>
                {messageRelations.length > 0 && (
                  <RelationConnections messageId={message.id} relations={messageRelations} infonIndex={infonIndex} />
                )}
                <div className={styles.msgContent} style={{ position: 'relative', zIndex: 2 }}>
                  {renderHighlightedText(message.content, message.id)}
                </div>
                {Array.isArray(message.images) && message.images.length > 0 && (
                  <div className={styles.msgImages}>
                    {message.images.map((src, imgIdx) => (
                      <img key={imgIdx} src={src} alt={`img-${imgIdx}`} className={styles.msgImage} />
                    ))}
                  </div>
                )}
                {/* 关系标签 */}
                {messageRelations.length > 0 && (
                  <RelationTags relations={messageRelations} infonIndex={infonIndex} />
                )}
              </div>
              {/* 用户消息操作按钮 */}
              <div className={styles.messageActions} style={{ justifyContent: 'flex-end' }}>
                <Tooltip title="复制">
                  <Button 
                    type="text" 
                    size="small" 
                    icon={<CopyOutlined />}
                    onClick={() => onCopy(message.content)}
                    className={styles.messageActionBtn}
                  />
                </Tooltip>
                <Tooltip title="编辑">
                  <Button 
                    type="text" 
                    size="small" 
                    icon={<EditOutlined />}
                    onClick={() => onEdit(message.id, message.content, message.images)}
                    className={styles.messageActionBtn}
                    disabled={isGenerating}
                  />
                </Tooltip>
              </div>
            </>
          )}
        </div>
        <div className={styles.avatar}>U</div>
      </div>
    )
  }

  // 助手消息
  return (
    <div className={`${styles.msgRow} ${styles.rowAssistant}`}>
      <div className={styles.avatar}>A</div>
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div className={`${styles.msgBubble} ${styles.msgBubbleAssistant}`} style={{ position: 'relative' }}>
          {messageRelations.length > 0 && (
            <RelationConnections messageId={message.id} relations={messageRelations} infonIndex={infonIndex} />
          )}
          {message.reasoning && (
            <div className={styles.reasoningBox}>
              <div className={styles.reasoningTitle}>Thinking</div>
              <div className={styles.reasoningBody}>
                <MarkdownMessage content={message.reasoning} />
              </div>
            </div>
          )}
          <div className={styles.msgContent}>
            <div className={styles.assistantTextHighlight} style={{ position: 'relative', zIndex: 2 }}>
              {renderHighlightedText(message.content, message.id)}
            </div>
          </div>
          {/* 关系标签 */}
          {messageRelations.length > 0 && (
            <RelationTags relations={messageRelations} infonIndex={infonIndex} />
          )}
          {message.streaming ? <div className={styles.cursor}>▍</div> : null}
          {message.error ? <div className={styles.error}>Error: {message.error}</div> : null}
        </div>
        {/* 助手消息操作按钮 */}
        {!message.streaming && (
          <div className={styles.messageActions}>
            <Tooltip title="复制">
              <Button 
                type="text" 
                size="small" 
                icon={<CopyOutlined />}
                onClick={() => onCopy(message.content)}
                className={styles.messageActionBtn}
              />
            </Tooltip>
            <Tooltip title="重新生成">
              <Button 
                type="text" 
                size="small" 
                icon={<RedoOutlined />}
                onClick={() => onRetry(message.id)}
                className={styles.messageActionBtn}
                disabled={isGenerating}
              />
            </Tooltip>
          </div>
        )}
      </div>
    </div>
  )
}

export default MessageBubble

