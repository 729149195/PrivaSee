import React from 'react'
import { Button, Upload } from 'antd'
import { SendOutlined, StopOutlined, CameraOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import HighlightInput from '../HighlightInput'
import RelationTags from './RelationTags'

/**
 * 消息输入组合器组件（底部固定输入框）
 * @param {string} input - 输入内容
 * @param {function} setInput - 设置输入内容
 * @param {function} onSend - 发送消息的回调
 * @param {Array} selectedImages - 已选择的图片
 * @param {function} setSelectedImages - 设置已选择图片
 * @param {function} onRemoveImage - 移除图片的回调
 * @param {function} onImageClick - 点击图片的回调
 * @param {boolean} isGenerating - 是否正在生成
 * @param {function} onStop - 停止生成的回调
 * @param {object} sendLockState - 发送锁定状态
 * @param {Array} pendingHighlights - 待处理的高亮
 * @param {Array} pendingRelations - 待处理的关系
 * @param {object} pendingInfonIndex - 待处理的信息元索引
 * @param {boolean} currentModelIsMultimodal - 当前模型是否支持多模态
 */
const MessageComposer = ({
  input,
  setInput,
  onSend,
  selectedImages,
  setSelectedImages,
  onRemoveImage,
  onImageClick,
  isGenerating,
  onStop,
  sendLockState,
  pendingHighlights,
  pendingRelations,
  pendingInfonIndex,
  currentModelIsMultimodal
}) => {
  return (
    <div className={styles.composerDock}>
      <div className={styles.composer}>
        {/* 预览总在输入框上方 */}
        {selectedImages.length > 0 && (
          <div className={styles.composerPreviews}>
            {selectedImages.map((src, i) => (
              <div key={i} className={styles.composerPreviewItem}>
                <img 
                  src={src} 
                  alt={`preview-${i}`} 
                  className={styles.composerPreviewImg} 
                  onClick={() => onImageClick?.(src)}
                  style={{ cursor: 'pointer' }}
                />
                <button 
                  className={styles.composerPreviewRemove} 
                  onClick={(e) => { 
                    e.stopPropagation(); 
                    onRemoveImage?.(i); 
                  }}
                >✕</button>
              </div>
            ))}
          </div>
        )}
        <div className={styles.composerRow}>
          <HighlightInput
            className={styles.composerInput}
            placeholder="Message ChatGPT"
            value={input}
            onChange={setInput}
            onPressEnter={onSend}
            highlights={pendingHighlights}
            autoSize={{ minRows: 1, maxRows: 6 }}
          />
          <div className={styles.composerButtons}>
            <Upload
              disabled={!currentModelIsMultimodal}
              multiple
              accept="image/*"
              showUploadList={false}
              beforeUpload={(file) => {
                const reader = new FileReader()
                reader.onload = () => setSelectedImages((prev) => [...prev, reader.result])
                reader.readAsDataURL(file)
                return Upload.LIST_IGNORE
              }}
            >
              <Button 
                icon={<CameraOutlined />} 
                disabled={!currentModelIsMultimodal} 
                title={currentModelIsMultimodal ? '' : 'Current model does not support images'} 
              />
            </Upload>
            {!isGenerating ? (
              <Button 
                type={sendLockState.locked ? "default" : "primary"}
                icon={sendLockState.stage === 'ready' ? <SendOutlined /> : null}
                disabled={(!input.trim() && selectedImages.length === 0) || sendLockState.locked}
                onClick={onSend}
                loading={sendLockState.locked}
                className={sendLockState.locked ? styles.sendButtonLocked : ''}
              >
                {sendLockState.locked ? sendLockState.label : ''}
              </Button>
            ) : (
              <Button danger icon={<StopOutlined />} onClick={onStop}>Stop</Button>
            )}
          </div>
        </div>
        {/* Pending 关系标签 */}
        {pendingRelations.length > 0 && (
          <RelationTags 
            relations={pendingRelations} 
            infonIndex={pendingInfonIndex} 
            style={{ marginTop: '8px' }} 
          />
        )}
      </div>
      <div className={styles.disclaimer}>Model streams responses. Context comes from this chat history.</div>
    </div>
  )
}

export default MessageComposer

