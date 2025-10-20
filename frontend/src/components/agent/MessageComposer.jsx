import React from 'react'
import { Button, Upload, message } from 'antd'
import { SendOutlined, StopOutlined, CameraOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import HighlightInput from '../HighlightInput'
import RelationTags from './RelationTags'
import AudioRecorder from './AudioRecorder'
import AudioTag from './AudioTag'
import ImagePreview from './ImagePreview'
import { compressImage, checkFileSize, getFileSizeText } from '../../utils/imageCompression'

/**
 * 消息输入组合器组件（底部固定输入框）
 * @param {string} input - 输入内容
 * @param {function} setInput - 设置输入内容
 * @param {function} onSend - 发送消息的回调
 * @param {Array} selectedImages - 已选择的图片
 * @param {function} setSelectedImages - 设置已选择图片
 * @param {function} onRemoveImage - 移除图片的回调
 * @param {function} onImageClick - 点击图片的回调
 * @param {Array} selectedAudios - 已选择的音频
 * @param {function} setSelectedAudios - 设置已选择音频
 * @param {function} onRemoveAudio - 移除音频的回调
 * @param {function} onTranscriptChange - 修改音频转录的回调
 * @param {boolean} isGenerating - 是否正在生成
 * @param {function} onStop - 停止生成的回调
 * @param {object} sendLockState - 发送锁定状态
 * @param {Array} pendingHighlights - 待处理的高亮
 * @param {Array} pendingRelations - 待处理的关系
 * @param {object} pendingInfonIndex - 待处理的信息元索引
 * @param {boolean} currentModelIsMultimodal - 当前模型是否支持多模态
 * @param {boolean} isEditingMessage - 是否正在编辑消息
 * @param {function} renderHighlightedText - 渲染高亮文本的函数
 * @param {string} inferenceMode - 推断模式 ('extract' | 'direct')
 * @param {function} processImageUpload - 处理图片上传的函数（用于直接推断模式）
 */
const MessageComposer = ({
  input,
  setInput,
  onSend,
  selectedImages,
  setSelectedImages,
  onRemoveImage,
  onImageClick,
  selectedAudios = [],
  setSelectedAudios,
  onRemoveAudio,
  onTranscriptChange,
  isGenerating,
  onStop,
  sendLockState,
  pendingHighlights,
  pendingRelations,
  pendingInfonIndex,
  currentModelIsMultimodal,
  isEditingMessage,
  renderHighlightedText,
  inferenceMode,
  processImageUpload
}) => {
  const handleAudioAdded = (audioData) => {
    setSelectedAudios?.((prev) => [...prev, audioData])
  }
  
  // 获取图片URL（兼容字符串和对象格式）
  const getImageUrl = (img) => {
    return typeof img === 'string' ? img : img?.url
  }
  
  // 获取图片对象（兼容字符串和对象格式）
  const getImageData = (img) => {
    return typeof img === 'string' ? { url: img, status: 'done' } : img
  }

  return (
    <div className={styles.composerDock}>
      <div className={styles.composer}>
        {/* 预览总在输入框上方 */}
        {selectedImages.length > 0 && (
          <div className={styles.composerPreviews}>
            {selectedImages.map((img, i) => {
              const imageData = getImageData(img)
              const imageUrl = getImageUrl(img)
              
              return (
                <div key={imageData.id || i} className={styles.composerPreviewItem}>
                  <ImagePreview
                    imageData={imageData}
                    onRemove={() => onRemoveImage?.(i)}
                    onClick={() => onImageClick?.(imageUrl)}
                    removable={true}
                  />
                </div>
              )
            })}
          </div>
        )}
        {/* 音频预览在输入框上方 */}
        {selectedAudios.length > 0 && (
          <div className={styles.composerAudios}>
            {selectedAudios.map((audio, i) => (
              <AudioTag
                key={audio.id}
                audioData={audio}
                onRemove={() => onRemoveAudio?.(i)}
                onTranscriptChange={onTranscriptChange}
                removable={true}
                variant="input"
                editable={true}
                renderHighlightedText={renderHighlightedText}
              />
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
            highlights={isEditingMessage ? [] : pendingHighlights}
            autoSize={{ minRows: 1, maxRows: 6 }}
            disabled={isEditingMessage}
          />
          <div className={styles.composerButtons}>
            <Upload
              disabled={!currentModelIsMultimodal || isEditingMessage}
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
                  
                  // 直接推断模式：使用图片分析功能
                  if (inferenceMode === 'direct' && processImageUpload) {
                    await processImageUpload(compressed, setSelectedImages)
                    message.success('图片上传成功，正在分析...')
                  } else {
                    // 提取信息元模式：直接添加图片（保持向后兼容）
                    setSelectedImages((prev) => [...prev, compressed])
                    message.success('图片上传成功')
                  }
                } catch (error) {
                  message.error(`图片处理失败: ${error.message}`)
                  console.error('[ImageUpload]', error)
                }
                
                return Upload.LIST_IGNORE
              }}
            >
              <Button 
                icon={<CameraOutlined />} 
                disabled={!currentModelIsMultimodal || isEditingMessage} 
                title={currentModelIsMultimodal ? '' : 'Current model does not support images'} 
              />
            </Upload>
            <AudioRecorder 
              onAudioAdded={handleAudioAdded}
              disabled={isEditingMessage}
            />
            {!isGenerating ? (
              <Button 
                type={sendLockState.locked ? "default" : "primary"}
                icon={sendLockState.stage === 'ready' ? <SendOutlined /> : null}
                disabled={(!input.trim() && selectedImages.length === 0 && selectedAudios.length === 0) || sendLockState.locked}
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
        {/* Pending 关系标签（编辑模式下不显示） */}
        {!isEditingMessage && pendingRelations.length > 0 && (
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

