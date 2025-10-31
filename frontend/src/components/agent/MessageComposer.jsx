import React, { useState } from 'react'
import { Button, Upload, message } from 'antd'
import { SendOutlined, StopOutlined, CameraOutlined, PlusOutlined, DeleteOutlined, CloseOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import HighlightInput from '../HighlightInput'
import RelationTags from './RelationTags'
import AudioRecorder from './AudioRecorder'
import AudioTag from './AudioTag'
import ImagePreview from './ImagePreview'
import SlashCommands from './SlashCommands'
import DocumentUploader from './DocumentUploader'
import { compressImage, checkFileSize, getFileSizeText } from '../../utils/imageCompression'
import { useStore } from '../../store'
import { callDeepseekOcr } from '../../utils/deepseekOcrApi'

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
 * @param {string} model - 当前选中的模型ID
 * @param {Array} selectedFiles - 已选择的文件（deepseek-ocr模式）
 * @param {function} setSelectedFiles - 设置已选择的文件
 * @param {function} onRemoveFile - 移除文件的回调
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
  processImageUpload,
  model,
  selectedFiles = [],
  setSelectedFiles,
  onRemoveFile,
  selectedCommand,
  setSelectedCommand
}) => {
  // 斜杠命令状态
  const [showSlashCommands, setShowSlashCommands] = useState(false)
  const [slashCommandPosition, setSlashCommandPosition] = useState({ top: 0, left: 0 })
  const [selectedOCRCommand, setSelectedOCRCommand] = useState(null)
  const [showDocumentUploader, setShowDocumentUploader] = useState(false)

  const customProviders = useStore((state) => state.customProviders)
  const deepseekProvider = customProviders?.['deepseek-ocr']

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

  // 监听输入变化，检测斜杠命令
  const handleInputChange = (newValue) => {
    // 检测是否输入了单个 "/"（所有模式都显示菜单）
    if (newValue === '/') {
      // 计算命令菜单位置
      setTimeout(() => {
        // 尝试多种方式找到输入框（HighlightInput 使用 contentEditable div）
        let inputElement = null

        // 方法1: 通过当前焦点元素
        const activeElement = document.activeElement
        if (activeElement && activeElement.getAttribute('contenteditable') === 'true') {
          inputElement = activeElement
        }

        // 方法2: 查找所有 contentEditable 元素，找到可见且在视口中的
        if (!inputElement) {
          const allEditables = document.querySelectorAll('[contenteditable="true"]')

          for (const editable of allEditables) {
            const rect = editable.getBoundingClientRect()
            // 检查是否在视口中且可见
            if (rect.height > 0 && rect.width > 0 &&
                rect.top >= 0 && rect.top < window.innerHeight) {
              inputElement = editable
              break
            }
          }
        }

        // 方法3: 通过 data-placeholder 查找
        if (!inputElement) {
          inputElement = document.querySelector('[contenteditable="true"][data-placeholder]')
        }

        if (inputElement) {
          const rect = inputElement.getBoundingClientRect()

          // 估算菜单高度（7个项目，每个约50px，加上padding）
          const estimatedMenuHeight = 7 * 50 + 20 // 约370px
          const menuLeft = rect.left

          // 优先在输入框上方显示，如果空间不够则在下方显示
          let menuTop
          if (rect.top > estimatedMenuHeight + 10) {
            // 上方有足够空间
            menuTop = rect.top - estimatedMenuHeight - 10
          } else {
            // 上方空间不够，在下方显示
            menuTop = rect.bottom + 10
          }

          setSlashCommandPosition({
            top: menuTop,
            left: menuLeft
          })
          setShowSlashCommands(true)
        } else {
          // 即使找不到输入框，也显示菜单（在默认位置）
          setSlashCommandPosition({
            top: 200,
            left: 300
          })
          setShowSlashCommands(true)
        }
      }, 50) // 增加延迟到50ms，确保DOM已更新
      
      // 不要立即修改input状态，等待用户选择命令
      return
    }

    setInput(newValue)
  }

  // 处理斜杠命令选择
  const handleCommandSelect = (command) => {
    if (!command) {
      setShowSlashCommands(false)
      // 取消时也要删除"/"
      setTimeout(() => {
        const inputElement = document.activeElement
        if (inputElement && inputElement.getAttribute('contenteditable') === 'true') {
          const text = inputElement.textContent || ''
          if (text === '/') {
            inputElement.textContent = ''
            setInput('')
          }
        }
      }, 0)
      return
    }

    // 设置选中的命令（只能选择一个）
    setSelectedCommand(command)
    setShowSlashCommands(false)

    // 立即删除输入框中的"/" - 直接操作DOM + 更新状态
    setTimeout(() => {
      // 查找输入框元素
      const allEditables = document.querySelectorAll('[contenteditable="true"]')
      for (const editable of allEditables) {
        const text = editable.textContent || ''
        if (text === '/') {
          editable.textContent = ''
          setInput('')
          break
        }
      }
    }, 0)
  }

  // 处理文档上传
  const handleDocumentUpload = async (files) => {
    if (!files || files.length === 0) {
      setShowDocumentUploader(false)
      setSelectedOCRCommand(null)
      return
    }

    setShowDocumentUploader(false)

    if (model === 'deepseek-ocr') {
      // deepseek-ocr 模式：将文件添加到 selectedFiles
      const file = files[0]
      const fileData = {
        id: Date.now() + '_' + Math.random(),
        name: file.name,
        size: file.size,
        type: file.type,
        file: file
      }
      setSelectedFiles?.((prev) => [...prev, fileData])
      message.success('文件上传成功')
    } else {
      // 其他模式：直接进行 OCR 处理
      try {
        if (!deepseekProvider) {
          throw new Error('未配置 DeepSeek OCR API，请先在设置中添加')
        }

        const hide = message.loading(`正在处理: ${selectedOCRCommand.label}`, 0)

        try {
          const result = await callDeepseekOcr({
            file: files[0],
            commandId: selectedOCRCommand.id,
            provider: deepseekProvider
          })

          setInput(result.text || '')
          message.success(`✅ ${selectedOCRCommand.label} 完成`)
        } finally {
          hide()
        }

      } catch (error) {
        message.error(`OCR 处理失败: ${error.message}`)
        console.error('OCR Error:', error)
      }
    }

    setSelectedOCRCommand(null)
  }

  return (
    <div className={styles.composerDock}>
      {/* 斜杠命令菜单 */}
      <SlashCommands
        visible={showSlashCommands}
        position={slashCommandPosition}
        onSelectCommand={handleCommandSelect}
      />

      {/* 文档上传器 */}
      <DocumentUploader
        visible={showDocumentUploader}
        onFileSelect={handleDocumentUpload}
        onClose={() => {
          setShowDocumentUploader(false)
          setSelectedOCRCommand(null)
        }}
      />

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
        {/* 文件预览在输入框上方（deepseek-ocr模式） */}
        {model === 'deepseek-ocr' && selectedFiles.length > 0 && (
          <div
            className={styles.composerFiles}
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '6px 8px',
              marginBottom: '8px'
            }}
          >
            {selectedFiles.map((file, i) => (
              <div
                key={file.id || i}
                className={styles.composerFileItem}
                style={{
                  position: 'relative',
                  display: 'inline-flex',
                  alignItems: 'center',
                  padding: '6px 28px 6px 8px',
                  borderRadius: '6px',
                  backgroundColor: 'rgba(24, 144, 255, 0.06)',
                  border: '1px solid rgba(24, 144, 255, 0.15)',
                  transition: 'all 0.2s ease',
                  cursor: 'default',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.1)'
                  e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.25)'
                  const deleteBtn = e.currentTarget.querySelector('.file-delete-btn')
                  if (deleteBtn) deleteBtn.style.opacity = '1'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.06)'
                  e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.15)'
                  const deleteBtn = e.currentTarget.querySelector('.file-delete-btn')
                  if (deleteBtn) deleteBtn.style.opacity = '0'
                }}
              >
                <FileTextOutlined style={{ marginRight: '6px', color: '#1890ff', fontSize: '14px' }} />
                <span
                  style={{
                    fontSize: '12px',
                    color: '#262626',
                    fontWeight: 500,
                    userSelect: 'none'
                  }}
                >
                  {file.name}
                </span>
                <span style={{
                  fontSize: '11px',
                  color: '#8c8c8c',
                  marginLeft: '4px',
                  userSelect: 'none',
                  marginRight: '20px'
                }}>
                  ({getFileSizeText(file)})
                </span>
                <Button
                  type="text"
                  size="small"
                  icon={<CloseOutlined />}
                  onClick={() => onRemoveFile?.(i)}
                  className="file-delete-btn"
                  style={{
                    position: 'absolute',
                    right: '2px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    color: '#666',
                    opacity: 0,
                    transition: 'opacity 0.2s ease',
                    padding: '3px',
                    width: '20px',
                    height: '20px',
                    minWidth: '20px',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
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
              </div>
            ))}
          </div>
        )}
        <div className={styles.composerRow}>
          <HighlightInput
            className={styles.composerInput}
            placeholder={
              model === 'deepseek-ocr' && input.trim() === ''
                ? "Message ChatGPT (输入 / 使用 OCR 功能)"
                : "Message ChatGPT"
            }
            value={input}
            onChange={handleInputChange}
            onPressEnter={onSend}
            highlights={isEditingMessage ? [] : pendingHighlights}
            autoSize={{ minRows: 1, maxRows: 6 }}
            disabled={isEditingMessage}
          />
          <div className={styles.composerButtons}>
            {model === 'deepseek-ocr' ? (
              // deepseek-ocr 模式：显示文件上传按钮 + 功能标签
              <>
                <Button
                  icon={<PlusOutlined />}
                  disabled={isEditingMessage}
                  onClick={() => setShowDocumentUploader(true)}
                  title="上传文档进行 OCR 处理"
                />
                {selectedCommand && (
                  <div
                    style={{
                      position: 'relative',
                      display: 'inline-flex',
                      alignItems: 'center',
                      marginLeft: '8px',
                      padding: '6px 12px 6px 12px',
                      borderRadius: '16px',
                      backgroundColor: 'rgba(24, 144, 255, 0.08)',
                      border: '1px solid rgba(24, 144, 255, 0.2)',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer',
                      whiteSpace: 'nowrap'
                    }}
                    onClick={(e) => {
                      // 点击标签本身（不是删除按钮）时打开命令菜单
                      if (!e.target.closest('.command-delete-btn')) {
                        // 计算命令菜单位置
                        const rect = e.currentTarget.getBoundingClientRect()
                        const estimatedMenuHeight = 7 * 50 + 20
                        let menuTop
                        if (rect.top > estimatedMenuHeight + 10) {
                          menuTop = rect.top - estimatedMenuHeight - 10
                        } else {
                          menuTop = rect.bottom + 10
                        }
                        setSlashCommandPosition({
                          top: menuTop,
                          left: rect.left
                        })
                        setShowSlashCommands(true)
                      }
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.12)'
                      e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.3)'
                      e.currentTarget.style.padding = '6px 28px 6px 12px'
                      const deleteBtn = e.currentTarget.querySelector('.command-delete-btn')
                      if (deleteBtn) deleteBtn.style.opacity = '1'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.08)'
                      e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.2)'
                      e.currentTarget.style.padding = '6px 12px 6px 12px'
                      const deleteBtn = e.currentTarget.querySelector('.command-delete-btn')
                      if (deleteBtn) deleteBtn.style.opacity = '0'
                    }}
                  >
                    <span style={{
                      fontSize: '13px',
                      color: '#1890ff',
                      fontWeight: 500,
                      userSelect: 'none'
                    }}>
                      {selectedCommand.label}
                    </span>
                    <Button
                      type="text"
                      size="small"
                      icon={<CloseOutlined />}
                      onClick={() => setSelectedCommand(null)}
                      className="command-delete-btn"
                      style={{
                        position: 'absolute',
                        right: '2px',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        color: '#666',
                        opacity: 0,
                        transition: 'opacity 0.2s ease',
                        padding: '4px',
                        width: '24px',
                        height: '24px',
                        minWidth: '24px',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
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
                  </div>
                )}
              </>
            ) : (
              // 其他模式：显示图片和音频按钮
              <>
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
              </>
            )}
            {!isGenerating ? (
              <Button
                type={sendLockState.locked ? "default" : "primary"}
                icon={sendLockState.stage === 'ready' ? <SendOutlined /> : null}
                disabled={
                  (model === 'deepseek-ocr'
                    ? (!input.trim() && selectedFiles.length === 0)
                    : (!input.trim() && selectedImages.length === 0 && selectedAudios.length === 0)
                  ) || sendLockState.locked
                }
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

