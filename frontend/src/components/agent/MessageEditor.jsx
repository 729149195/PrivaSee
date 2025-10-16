import React from 'react'
import { Button, Upload } from 'antd'
import { SendOutlined, CloseOutlined, CameraOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import HighlightInput from '../HighlightInput'
import RelationTags from './RelationTags'

/**
 * 消息编辑器组件
 * @param {string} editingContent - 编辑中的内容
 * @param {function} setEditingContent - 设置编辑内容
 * @param {Array} editingImages - 编辑中的图片
 * @param {function} setEditingImages - 设置编辑图片
 * @param {function} onSave - 保存的回调
 * @param {function} onCancel - 取消的回调
 * @param {Array} pendingHighlights - 待处理的高亮
 * @param {Array} pendingRelations - 待处理的关系
 * @param {object} pendingInfonIndex - 待处理的信息元索引
 * @param {object} sendLockState - 发送锁定状态
 * @param {string} originalContent - 原始内容
 * @param {Array} originalImages - 原始图片
 * @param {boolean} currentModelIsMultimodal - 当前模型是否支持多模态
 */
const MessageEditor = ({
  editingContent,
  setEditingContent,
  editingImages,
  setEditingImages,
  onSave,
  onCancel,
  pendingHighlights,
  pendingRelations,
  pendingInfonIndex,
  sendLockState,
  originalContent,
  originalImages,
  currentModelIsMultimodal
}) => {
  // 检查内容是否发生变化
  const hasContentChanged = 
    editingContent !== originalContent || 
    JSON.stringify(editingImages) !== JSON.stringify(originalImages)
  
  // 按钮是否应该禁用
  const isSaveDisabled = 
    !hasContentChanged || // 内容未修改
    (!editingContent.trim() && editingImages.length === 0) || // 内容为空
    sendLockState.locked // 正在处理中
  return (
    <div className={styles.editingComposer}>
      {/* 图片预览 */}
      {editingImages.length > 0 && (
        <div className={styles.composerPreviews}>
          {editingImages.map((src, imgIdx) => (
            <div key={imgIdx} className={styles.composerPreviewItem}>
              <img src={src} alt={`img-${imgIdx}`} className={styles.composerPreviewImg} />
              <button
                className={styles.composerPreviewRemove}
                onClick={() => setEditingImages(editingImages.filter((_, i) => i !== imgIdx))}
              >✕</button>
            </div>
          ))}
        </div>
      )}
      {/* 输入框和按钮 */}
      <div className={styles.composerRow}>
        <HighlightInput
          className={styles.composerInput}
          value={editingContent}
          onChange={setEditingContent}
          placeholder="编辑消息..."
          highlights={pendingHighlights}
          autoSize={{ minRows: 2, maxRows: 10 }}
        />
        <div className={styles.composerButtons}>
          <Upload
            disabled={!currentModelIsMultimodal}
            multiple
            accept="image/*"
            showUploadList={false}
            beforeUpload={(file) => {
              const reader = new FileReader()
              reader.onload = () => setEditingImages((prev) => [...prev, reader.result])
              reader.readAsDataURL(file)
              return Upload.LIST_IGNORE
            }}
          >
            <Button 
              icon={<CameraOutlined />} 
              disabled={!currentModelIsMultimodal} 
              title={currentModelIsMultimodal ? '上传图片' : '当前模型不支持图片'} 
            />
          </Upload>
          <Button 
            icon={<CloseOutlined />} 
            onClick={onCancel}
            title="取消编辑"
          />
          <Button 
            type={sendLockState.locked ? "default" : "primary"}
            icon={sendLockState.stage === 'ready' && !sendLockState.locked ? <SendOutlined /> : null}
            onClick={onSave}
            disabled={isSaveDisabled}
            loading={sendLockState.locked}
            className={sendLockState.locked ? styles.sendButtonLocked : ''}
          >
            {sendLockState.locked ? sendLockState.label : ''}
          </Button>
        </div>
      </div>
      {/* Pending关系标签显示 */}
      {pendingRelations.length > 0 && (
        <RelationTags 
          relations={pendingRelations} 
          infonIndex={pendingInfonIndex} 
          style={{ marginTop: 8 }} 
        />
      )}
    </div>
  )
}

export default MessageEditor

