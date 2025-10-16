import React from 'react'
import { Button, Upload, message } from 'antd'
import { SendOutlined, CameraOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import HighlightInput from '../HighlightInput'
import RelationTags from './RelationTags'
import { compressImage, checkFileSize, getFileSizeText } from '../../utils/imageCompression'

/**
 * 着陆页组件（首次访问时的欢迎界面）
 * @param {string} landingInput - 着陆页输入内容
 * @param {function} setLandingInput - 设置着陆页输入内容
 * @param {function} onSend - 发送消息的回调
 * @param {Array} selectedImages - 已选择的图片
 * @param {function} setSelectedImages - 设置已选择图片
 * @param {function} onRemoveImage - 移除图片的回调
 * @param {function} onImageClick - 点击图片的回调
 * @param {object} sendLockState - 发送锁定状态
 * @param {Array} pendingHighlights - 待处理的高亮
 * @param {Array} pendingRelations - 待处理的关系
 * @param {object} pendingInfonIndex - 待处理的信息元索引
 * @param {boolean} currentModelIsMultimodal - 当前模型是否支持多模态
 */
const LandingView = ({
  landingInput,
  setLandingInput,
  onSend,
  selectedImages,
  setSelectedImages,
  onRemoveImage,
  onImageClick,
  sendLockState,
  pendingHighlights,
  pendingRelations,
  pendingInfonIndex,
  currentModelIsMultimodal
}) => {
  return (
    <div className={styles.landing}>
      <div className={styles.landingTitle}>How can I help you today?</div>
      <div className={styles.landingSearch}>
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
        <div className={styles.landingInputArea}>
          <div className={styles.landingControls}>
            <Upload
              disabled={!currentModelIsMultimodal}
              multiple
              accept="image/*"
              showUploadList={false}
              beforeUpload={async (file) => {
                // 检查文件大小
                if (!checkFileSize(file, 20)) {
                  message.error(`图片 "${file.name}" 过大 (${getFileSizeText(file)})，最大支持20MB`)
                  return Upload.LIST_IGNORE
                }
                
                try {
                  // 显示加载提示
                  const hideLoading = message.loading(`压缩图片中... (${getFileSizeText(file)})`, 0)
                  
                  // 压缩图片
                  const compressed = await compressImage(file, {
                    maxWidth: 1920,
                    maxHeight: 1080,
                    quality: 0.8,
                    maxSizeMB: 2
                  })
                  
                  hideLoading()
                  setSelectedImages((prev) => [...prev, compressed])
                  message.success('图片上传成功')
                } catch (error) {
                  message.error(`图片处理失败: ${error.message}`)
                  console.error('[ImageUpload]', error)
                }
                
                return Upload.LIST_IGNORE
              }}
            >
              <Button 
                icon={<CameraOutlined />} 
                disabled={!currentModelIsMultimodal} 
                title={currentModelIsMultimodal ? '' : 'Current model does not support images'} 
              />
            </Upload>
            <HighlightInput
              className={styles.landingInput}
              placeholder="Type your question..."
              value={landingInput}
              onChange={setLandingInput}
              onPressEnter={onSend}
              highlights={pendingHighlights}
              autoSize={{ minRows: 1, maxRows: 6 }}
            />
            <Button 
              type={sendLockState.locked ? "default" : "primary"}
              icon={sendLockState.stage === 'ready' ? <SendOutlined /> : null}
              onClick={onSend} 
              disabled={!landingInput.trim() && selectedImages.length === 0}
              loading={sendLockState.locked}
              className={sendLockState.locked ? styles.sendButtonLocked : ''}
            >
              {sendLockState.locked ? sendLockState.label : ''}
            </Button>
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
      </div>
    </div>
  )
}

export default LandingView

