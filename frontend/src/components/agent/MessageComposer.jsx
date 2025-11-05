import React, { useState, useRef } from 'react'
import { Button, Upload, message, Progress } from 'antd'
import { SendOutlined, StopOutlined, CameraOutlined, PlusOutlined, DeleteOutlined, CloseOutlined, FileTextOutlined, LoadingOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import HighlightInput from '../HighlightInput'
import RelationTags from './RelationTags'
import AudioRecorder from './AudioRecorder'
import AudioTag from './AudioTag'
import ImagePreview from './ImagePreview'
import SlashCommands from './SlashCommands'
import DocumentUploader from './DocumentUploader'
import CommandTag from './CommandTag'
import { compressImage, checkFileSize, getFileSizeText } from '../../utils/imageCompression'
import { useStore } from '../../store'
import { callDeepseekOcr } from '../../utils/deepseekOcrApi'
import { uploadFile } from '../../utils/fileUpload'
import { saveFile } from '../../utils/fileStorage'

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
  setSelectedCommand,
  selectedResolution = 'gundam',
  setSelectedResolution
}) => {
  // 斜杠命令状态
  const [showSlashCommands, setShowSlashCommands] = useState(false)
  const [slashCommandPosition, setSlashCommandPosition] = useState({ top: 0, left: 0 })
  const [selectedOCRCommand, setSelectedOCRCommand] = useState(null)
  const [showDocumentUploader, setShowDocumentUploader] = useState(false)

  const customProviders = useStore((state) => state.customProviders)
  const deepseekProvider = customProviders?.['deepseek-ocr']
  const currentSessionId = useStore((state) => state.currentSessionId)

  // 文件选择器引用（OCR模式直接选择文件）
  const fileInputRef = useRef(null)

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

  // 处理文件选择（OCR模式：直接触发文件选择器）
  const handleFileSelectClick = () => {
    const isOcrMode = model === 'deepseek-ocr' || model === 'deepseek-ocr-local'
    if (isOcrMode) {
      // OCR 模式：直接打开文件选择器
      fileInputRef.current?.click()
    } else {
      // 其他模式：打开文档上传对话框
      setShowDocumentUploader(true)
    }
  }

  // 处理文件选择变化（OCR模式：立即上传）
  const handleFileChange = async (e) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return

    if (!deepseekProvider) {
      message.error('未配置 DeepSeek OCR API，请先在设置中添加')
      e.target.value = ''
      return
    }

    // 定义支持的格式
    const acceptedTypes = [
      'image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/bmp', 'image/tiff', 
      'application/pdf',
      // Office 文档
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
      'application/msword', // .doc
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
      'application/vnd.ms-excel', // .xls
      'application/vnd.openxmlformats-officedocument.presentationml.presentation', // .pptx
      'application/vnd.ms-powerpoint', // .ppt
      // LibreOffice
      'application/vnd.oasis.opendocument.text', // .odt
      'application/vnd.oasis.opendocument.spreadsheet', // .ods
      'application/vnd.oasis.opendocument.presentation' // .odp
    ]
    
    const acceptedExtensions = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'tif', 
                                'pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 
                                'odt', 'ods', 'odp']

    // 验证所有文件
    const invalidFiles = []
    const oversizedFiles = []
    
    for (const file of files) {
      const fileExt = file.name.toLowerCase().split('.').pop()
      if (!acceptedTypes.includes(file.type) && !acceptedExtensions.includes(fileExt)) {
        invalidFiles.push(file.name)
      }
      
      const isLt40M = file.size / 1024 / 1024 < 40
      if (!isLt40M) {
        oversizedFiles.push(file.name)
      }
    }

    if (invalidFiles.length > 0) {
      message.error(`不支持的文件格式: ${invalidFiles.join(', ')}`)
    }
    
    if (oversizedFiles.length > 0) {
      message.error(`文件过大（超过40MB）: ${oversizedFiles.join(', ')}`)
    }

    // 过滤出有效的文件
    const validFiles = files.filter(file => {
      const fileExt = file.name.toLowerCase().split('.').pop()
      const isValidType = acceptedTypes.includes(file.type) || acceptedExtensions.includes(fileExt)
      const isValidSize = file.size / 1024 / 1024 < 40
      return isValidType && isValidSize
    })

    if (validFiles.length === 0) {
      e.target.value = ''
      return
    }

    // 处理每个有效文件
    for (const file of validFiles) {
      const fileId = Date.now() + '_' + Math.random()
      const fileData = {
        id: fileId,
        name: file.name,
        size: file.size,
        type: file.type,
        file: file,
        uploadStatus: 'uploading',
        uploadProgress: 0,
        uploadError: null,
        serverFilename: null
      }

      // 添加到文件列表
      setSelectedFiles?.((prev) => [...prev, fileData])

      // 立即存储到 IndexedDB
      try {
        await saveFile(currentSessionId, 'temp_' + fileId, fileId, file)
        console.log('[MessageComposer] 文件已保存到 IndexedDB:', file.name)
      } catch (error) {
        console.error('[MessageComposer] 保存文件到 IndexedDB 失败:', error)
      }

      // 立即开始上传到服务器
      try {
        const result = await uploadFile(file, deepseekProvider, (progress) => {
          setSelectedFiles?.((prev) => 
            prev.map(f => 
              f.id === fileId 
                ? { ...f, uploadProgress: progress }
                : f
            )
          )
        })

        // 上传成功
        setSelectedFiles?.((prev) => 
          prev.map(f => 
            f.id === fileId 
              ? { 
                  ...f, 
                  uploadStatus: 'completed', 
                  uploadProgress: 100,
                  serverFilename: result.filename
                }
              : f
          )
        )
        
        console.log('[MessageComposer] 文件上传成功:', file.name)
      } catch (error) {
        // 上传失败
        setSelectedFiles?.((prev) => 
          prev.map(f => 
            f.id === fileId 
              ? { 
                  ...f, 
                  uploadStatus: 'error', 
                  uploadError: error.message 
                }
              : f
          )
        )
        
        message.error(`${file.name} 上传失败: ${error.message}`)
        console.error('[MessageComposer] 文件上传失败:', error)
      }
    }

    if (validFiles.length > 0) {
      message.success(`成功添加 ${validFiles.length} 个文件`)
    }

    // 清空input，允许重复选择同一文件
    e.target.value = ''
  }

  // 处理文档上传（非OCR模式使用）
  const handleDocumentUpload = async (files) => {
    if (!files || files.length === 0) {
      setShowDocumentUploader(false)
      setSelectedOCRCommand(null)
      return
    }

    setShowDocumentUploader(false)

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
        {/* 文件预览在输入框上方（OCR模式） */}
        {(model === 'deepseek-ocr' || model === 'deepseek-ocr-local') && selectedFiles.length > 0 && (
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
                  backgroundColor: file.uploadStatus === 'error' ? 'rgba(255, 77, 79, 0.06)' : 'rgba(24, 144, 255, 0.06)',
                  border: `1px solid ${file.uploadStatus === 'error' ? 'rgba(255, 77, 79, 0.15)' : 'rgba(24, 144, 255, 0.15)'}`,
                  transition: 'all 0.2s ease',
                  cursor: 'default',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => {
                  if (file.uploadStatus !== 'error') {
                    e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.1)'
                    e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.25)'
                  }
                  const deleteBtn = e.currentTarget.querySelector('.file-delete-btn')
                  if (deleteBtn) deleteBtn.style.opacity = '1'
                }}
                onMouseLeave={(e) => {
                  if (file.uploadStatus !== 'error') {
                    e.currentTarget.style.backgroundColor = 'rgba(24, 144, 255, 0.06)'
                    e.currentTarget.style.borderColor = 'rgba(24, 144, 255, 0.15)'
                  }
                  const deleteBtn = e.currentTarget.querySelector('.file-delete-btn')
                  if (deleteBtn) deleteBtn.style.opacity = '0'
                }}
              >
                {file.uploadStatus === 'uploading' ? (
                  <LoadingOutlined style={{ marginRight: '6px', color: '#1890ff', fontSize: '14px' }} />
                ) : file.uploadStatus === 'error' ? (
                  <CloseOutlined style={{ marginRight: '6px', color: '#ff4d4f', fontSize: '14px' }} />
                ) : (
                  <FileTextOutlined style={{ marginRight: '6px', color: '#1890ff', fontSize: '14px' }} />
                )}
                <span
                  style={{
                    fontSize: '12px',
                    color: file.uploadStatus === 'error' ? '#ff4d4f' : '#262626',
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
                
                {/* 上传进度指示器 */}
                {file.uploadStatus === 'uploading' && (
                  <div style={{
                    position: 'absolute',
                    right: '28px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    width: '20px',
                    height: '20px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <Progress
                      type="circle"
                      percent={file.uploadProgress || 0}
                      size={20}
                      strokeWidth={6}
                      strokeColor={{
                        '0%': '#4285f4',
                        '50%': '#34a853',
                        '100%': '#fbbc04'
                      }}
                      trailColor="rgba(0, 0, 0, 0.12)"
                      format={() => ''}
                      strokeLinecap="round"
                    />
                  </div>
                )}
                
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
              (model === 'deepseek-ocr' || model === 'deepseek-ocr-local') && input.trim() === ''
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
            {(model === 'deepseek-ocr' || model === 'deepseek-ocr-local') ? (
              // OCR 模式：显示文件上传按钮 + 功能标签
              <>
                {/* 隐藏的文件输入框 */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".png,.jpg,.jpeg,.gif,.webp,.bmp,.tiff,.tif,.pdf,.docx,.doc,.xlsx,.xls,.pptx,.ppt,.odt,.ods,.odp"
                  multiple
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
                <Button
                  icon={<PlusOutlined />}
                  disabled={isEditingMessage}
                  onClick={handleFileSelectClick}
                  title="上传文档进行 OCR 处理（可多选）"
                />
                {selectedCommand && (
                  <CommandTag
                    command={selectedCommand}
                    resolution={selectedResolution}
                    onResolutionChange={setSelectedResolution}
                    removable={true}
                    onRemove={() => setSelectedCommand(null)}
                    showResolutionSelector={true}
                    commandMenuOpen={showSlashCommands}
                    onTagClick={(e) => {
                      // 点击标签打开命令菜单
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
                    }}
                  />
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
                    if (!checkFileSize(file, 40)) {
                      message.error(`图片 "${file.name}" 过大 (${getFileSizeText(file)})，最大支持40MB`)
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
                  ((model === 'deepseek-ocr' || model === 'deepseek-ocr-local')
                    ? (
                        (!input.trim() && selectedFiles.length === 0 && !selectedCommand) ||
                        // 有文件正在上传时禁用发送按钮
                        selectedFiles.some(f => f.uploadStatus === 'uploading')
                      )
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

