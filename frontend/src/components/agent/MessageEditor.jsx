import React from 'react'
import { Button, Upload, message } from 'antd'
import { SendOutlined, CloseOutlined, CameraOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import HighlightInput from '../HighlightInput'
import RelationTags from './RelationTags'
import AudioRecorder from './AudioRecorder'
import AudioTag from './AudioTag'
import ImagePreview from './ImagePreview'
import { compressImage, checkFileSize, getFileSizeText } from '../../utils/imageCompression'

/**
 * 消息编辑器组件
 * @param {string} editingContent - 编辑中的内容
 * @param {function} setEditingContent - 设置编辑内容
 * @param {Array} editingImages - 编辑中的图片
 * @param {function} setEditingImages - 设置编辑图片
 * @param {Array} editingAudios - 编辑中的音频
 * @param {function} setEditingAudios - 设置编辑音频
 * @param {function} onSave - 保存的回调
 * @param {function} onCancel - 取消的回调
 * @param {Array} pendingHighlights - 待处理的高亮
 * @param {Array} pendingRelations - 待处理的关系
 * @param {object} pendingInfonIndex - 待处理的信息元索引
 * @param {object} sendLockState - 发送锁定状态
 * @param {string} originalContent - 原始内容
 * @param {Array} originalImages - 原始图片
 * @param {Array} originalAudios - 原始音频
 * @param {boolean} currentModelIsMultimodal - 当前模型是否支持多模态
 * @param {function} renderHighlightedText - 渲染高亮文本的函数
 * @param {string} inferenceMode - 推断模式 ('extract' | 'direct')
 * @param {function} processImageUpload - 处理图片上传的函数（用于直接推断模式）
 */
const MessageEditor = ({
  editingContent,
  setEditingContent,
  editingImages,
  setEditingImages,
  editingAudios = [],
  setEditingAudios,
  onEditingTranscriptChange,
  onSave,
  onCancel,
  pendingHighlights,
  pendingRelations,
  pendingInfonIndex,
  sendLockState,
  originalContent,
  originalImages,
  originalAudios = [],
  currentModelIsMultimodal,
  renderHighlightedText,
  inferenceMode,
  processImageUpload
}) => {
  const handleAudioAdded = (audioData) => {
    setEditingAudios?.((prev) => [...prev, audioData])
  }

  const removeEditingAudio = (index) => {
    setEditingAudios?.((prev) => prev.filter((_, i) => i !== index))
  }
  
  // 获取图片URL（兼容字符串和对象格式）
  const getImageUrl = (img) => {
    return typeof img === 'string' ? img : img?.url
  }
  
  // 获取图片对象（兼容字符串和对象格式）
  const getImageData = (img) => {
    return typeof img === 'string' ? { url: img, status: 'done' } : img
  }

  // 检查内容是否发生变化
  const hasContentChanged = 
    editingContent !== originalContent || 
    JSON.stringify(editingImages) !== JSON.stringify(originalImages) ||
    JSON.stringify(editingAudios) !== JSON.stringify(originalAudios)
  
  // 按钮是否应该禁用
  const isSaveDisabled = 
    !hasContentChanged || // 内容未修改
    (!editingContent.trim() && editingImages.length === 0 && editingAudios.length === 0) || // 内容为空
    sendLockState.locked // 正在处理中
  return (
    <div className={styles.editingComposer}>
      {/* 图片预览 */}
      {editingImages.length > 0 && (
        <div className={styles.composerPreviews}>
          {editingImages.map((img, imgIdx) => {
            const imageData = getImageData(img)
            return (
              <div key={imageData.id || imgIdx} className={styles.composerPreviewItem}>
                <ImagePreview
                  imageData={imageData}
                  onRemove={() => setEditingImages(editingImages.filter((_, i) => i !== imgIdx))}
                  removable={true}
                />
              </div>
            )
          })}
        </div>
      )}
      {/* 音频预览 */}
      {editingAudios.length > 0 && (
        <div className={styles.composerAudios}>
          {editingAudios.map((audio, i) => (
            <AudioTag
              key={audio.id}
              audioData={audio}
              onRemove={() => removeEditingAudio(i)}
              onTranscriptChange={onEditingTranscriptChange}
              removable={true}
              variant="input"
              editable={true}
              renderHighlightedText={renderHighlightedText}
            />
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
                  await processImageUpload(compressed, setEditingImages)
                  message.success('图片上传成功，正在分析...')
                } else {
                  // 提取信息元模式：直接添加图片（保持向后兼容）
                  setEditingImages((prev) => [...prev, compressed])
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
              disabled={!currentModelIsMultimodal} 
              title={currentModelIsMultimodal ? '上传图片' : '当前模型不支持图片'} 
            />
          </Upload>
          <AudioRecorder 
            onAudioAdded={handleAudioAdded}
            disabled={false}
          />
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

