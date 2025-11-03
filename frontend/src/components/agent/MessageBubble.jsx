import React, { useState, useEffect } from 'react'
import { Button, Tooltip } from 'antd'
import { CopyOutlined, EditOutlined, RedoOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import MarkdownMessage from '../MarkdownMessage'
import RelationConnections from './RelationConnections'
import RelationTags from './RelationTags'
import MessageEditor from './MessageEditor'
import AudioTag from './AudioTag'
import DocumentTag from './DocumentTag'
import CommandTag from './CommandTag'
import DocumentPreviewModal from './DocumentPreviewModal'
import { useStore } from '../../store'
import { loadFiles } from '../../utils/fileStorage'

// 导入 vite.svg 图标
const ViteIcon = '/vite.svg'

/**
 * 消息气泡组件
 * @param {object} message - 消息对象
 * @param {boolean} isUser - 是否为用户消息
 * @param {string} editingMessageId - 正在编辑的消息 ID
 * @param {string} editingContent - 编辑中的内容
 * @param {function} setEditingContent - 设置编辑内容
 * @param {Array} editingImages - 编辑中的图片
 * @param {function} setEditingImages - 设置编辑图片
 * @param {Array} editingAudios - 编辑中的音频
 * @param {function} setEditingAudios - 设置编辑音频
 * @param {function} onCopy - 复制的回调
 * @param {function} onEdit - 编辑的回调
 * @param {function} onSaveEdit - 保存编辑的回调
 * @param {function} onCancelEdit - 取消编辑的回调
 * @param {function} onRetry - 重试的回调
 * @param {function} onImageClick - 图片点击的回调
 * @param {boolean} isGenerating - 是否正在生成
 * @param {function} renderHighlightedText - 渲染高亮文本的函数
 * @param {Array} messageRelations - 消息的关系信息元
 * @param {object} infonIndex - 信息元索引
 * @param {Array} pendingHighlights - 待处理的高亮
 * @param {Array} pendingRelations - 待处理的关系
 * @param {object} pendingInfonIndex - 待处理的信息元索引
 * @param {boolean} currentModelIsMultimodal - 当前模型是否支持多模态
 * @param {string} inferenceMode - 推断模式 ('extract' | 'direct')
 * @param {function} processImageUpload - 处理图片上传的函数（用于直接推断模式）
 */
const MessageBubble = ({
  message,
  isUser,
  editingMessageId,
  editingContent,
  setEditingContent,
  editingImages,
  setEditingImages,
  editingAudios,
  setEditingAudios,
  editingFiles,
  setEditingFiles,
  editingCommands,
  setEditingCommands,
  originalEditingContent,
  originalEditingImages,
  originalEditingAudios,
  originalEditingFiles,
  originalEditingCommands,
  onEditingTranscriptChange,
  onCopy,
  onEdit,
  onSaveEdit,
  onCancelEdit,
  onRetry,
  onImageClick,
  isGenerating,
  renderHighlightedText,
  messageRelations,
  infonIndex,
  pendingHighlights,
  pendingRelations,
  pendingInfonIndex,
  sendLockState,
  currentModelIsMultimodal,
  inferenceMode,
  processImageUpload
}) => {
  const isEditing = editingMessageId === message.id
  const [previewFile, setPreviewFile] = useState(null)
  const [restorationAttempted, setRestorationAttempted] = useState(false)
  
  // 从 store 中获取当前会话和 File 对象映射
  const currentSessionId = useStore((state) => state.currentSessionId)
  const ocrFileObjects = useStore((state) => state.ocrFileObjects)
  
  // 获取当前消息的 File 对象映射
  const messageFileObjects = ocrFileObjects?.[currentSessionId]?.[message.id] || {}
  
  // 从 IndexedDB 恢复文件对象（如果需要）
  useEffect(() => {
    const restoreFiles = async () => {
      // 检查是否有files但没有File对象，且还未尝试过恢复
      if (message.files && message.files.length > 0 && 
          Object.keys(messageFileObjects).length === 0 && 
          !restorationAttempted) {
        setRestorationAttempted(true)
        
        try {
          console.log('[MessageBubble] 从 IndexedDB 恢复文件...', message.id)
          const fileIds = message.files.map(f => f.id)
          const restoredFiles = await loadFiles(currentSessionId, message.id, fileIds)
          
          if (Object.keys(restoredFiles).length > 0) {
            // 直接使用 useStore.setState 更新，避免闭包问题
            useStore.setState((state) => ({
              ocrFileObjects: {
                ...state.ocrFileObjects,
                [currentSessionId]: {
                  ...(state.ocrFileObjects?.[currentSessionId] || {}),
                  [message.id]: restoredFiles
                }
              }
            }))
            console.log('[MessageBubble] 恢复了 ' + Object.keys(restoredFiles).length + ' 个文件')
          }
        } catch (error) {
          console.error('[MessageBubble] 从 IndexedDB 恢复文件失败:', error)
        }
      }
    }
    
    restoreFiles()
  }, [message.id, message.files, currentSessionId, messageFileObjects, restorationAttempted])

  // 辅助函数：从消息内容中移除 <audio>...</audio> 标签
  // 音频转写文本完全通过 AudioTag 组件显示
  const removeAudioTags = (content) => {
    if (typeof content !== 'string') return content
    // 移除所有 <audio>...</audio> 标签及其内容
    return content.replace(/<audio>[\s\S]*?<\/audio>/gi, '').trim()
  }
  
  // 获取显示用的内容（移除了音频标签）
  const displayContent = removeAudioTags(message.content)

  if (isUser) {
    return (
      <>
        <div className={`${styles.msgRow} ${styles.rowUser}`}>
          <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
            {isEditing ? (
            <MessageEditor
              editingContent={editingContent}
              setEditingContent={setEditingContent}
              editingImages={editingImages}
              setEditingImages={setEditingImages}
              editingAudios={editingAudios}
              setEditingAudios={setEditingAudios}
              originalContent={originalEditingContent}
              originalImages={originalEditingImages}
              originalAudios={originalEditingAudios}
              onEditingTranscriptChange={onEditingTranscriptChange}
              onSave={onSaveEdit}
              onCancel={onCancelEdit}
              inferenceMode={inferenceMode}
              processImageUpload={processImageUpload}
              pendingHighlights={pendingHighlights}
              pendingRelations={pendingRelations}
              pendingInfonIndex={pendingInfonIndex}
              sendLockState={sendLockState}
              currentModelIsMultimodal={currentModelIsMultimodal}
              renderHighlightedText={renderHighlightedText}
            />
          ) : (
            <>
              <div className={`${styles.msgBubble} ${styles.msgBubbleUser}`} style={{ position: 'relative', padding: '14px 16px' }}>
                {messageRelations.length > 0 && (
                  <RelationConnections messageId={message.id} relations={messageRelations} infonIndex={infonIndex} />
                )}
                {/* 文档标签（包含嵌入的命令标签） */}
                {Array.isArray(message.files) && message.files.length > 0 && (
                  <div style={{ 
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '8px',
                    marginBottom: displayContent ? '12px' : '0' 
                  }}>
                    {message.files.map((file, idx) => (
                      <DocumentTag 
                        key={file.id || idx} 
                        file={file} 
                        onClick={() => setPreviewFile(file)}
                        command={message.commands?.[idx]}
                      />
                    ))}
                  </div>
                )}
                {displayContent && (
                  <div className={styles.msgContent} style={{ position: 'relative', zIndex: 2 }}>
                    {renderHighlightedText(displayContent, message.id)}
                  </div>
                )}
                {Array.isArray(message.images) && message.images.length > 0 && (
                  <div className={styles.msgImages}>
                    {message.images.map((src, imgIdx) => (
                      <img 
                        key={imgIdx} 
                        src={src} 
                        alt={`img-${imgIdx}`} 
                        className={styles.msgImage}
                        onClick={() => onImageClick?.(src)}
                        style={{ cursor: 'pointer' }}
                      />
                    ))}
                  </div>
                )}
                {/* 音频标签 */}
                {Array.isArray(message.audios) && message.audios.length > 0 && (
                  <div className={styles.msgAudios}>
                    {message.audios.map((audio, audioIdx) => (
                      <AudioTag 
                        key={audio.id || audioIdx} 
                        audioData={audio} 
                        removable={false} 
                        variant="message"
                        editable={false}
                        renderHighlightedText={(text) => renderHighlightedText(text, message.id)}
                      />
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
                    onClick={() => onEdit(message.id, message.content, message.images, message.audios, message.imageAnalysis, message.files, message.commands)}
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
      {/* 文档预览 Modal */}
      <DocumentPreviewModal 
        file={previewFile}
        fileObject={previewFile ? messageFileObjects[previewFile.id] : null}
        onClose={() => setPreviewFile(null)} 
      />
    </>
    )
  }

  // 助手消息
  return (
    <div className={`${styles.msgRow} ${styles.rowAssistant}`}>
      <div className={styles.avatar} style={{ 
        backgroundImage: `url(${ViteIcon})`, 
        backgroundSize: 'contain', 
        backgroundPosition: 'center', 
        backgroundRepeat: 'no-repeat',
        backgroundColor: 'transparent'
      }}></div>
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
              <MarkdownMessage content={message.content} />
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

