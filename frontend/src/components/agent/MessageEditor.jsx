import React from 'react'
import { Button } from 'antd'
import { CheckOutlined, CloseOutlined } from '@ant-design/icons'
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
  pendingInfonIndex
}) => {
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
      {/* 输入框 */}
      <div className={styles.composerRow}>
        <HighlightInput
          className={styles.composerInput}
          value={editingContent}
          onChange={setEditingContent}
          placeholder="编辑消息..."
          highlights={pendingHighlights}
          autoSize={{ minRows: 2, maxRows: 10 }}
        />
      </div>
      {/* Pending关系标签显示 */}
      {pendingRelations.length > 0 && (
        <RelationTags 
          relations={pendingRelations} 
          infonIndex={pendingInfonIndex} 
          style={{ marginTop: 8 }} 
        />
      )}
      {/* 操作按钮 */}
      <div style={{ display: 'flex', gap: 8, marginTop: 12, justifyContent: 'flex-end' }}>
        <Button size="small" icon={<CheckOutlined />} onClick={onSave} type="primary">保存并重新生成</Button>
        <Button size="small" icon={<CloseOutlined />} onClick={onCancel}>取消</Button>
      </div>
    </div>
  )
}

export default MessageEditor

