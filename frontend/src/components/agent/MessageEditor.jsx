import React, { useRef } from 'react'
import { Button, Upload, message, Progress } from 'antd'
import { SendOutlined, CloseOutlined, CameraOutlined, PlusOutlined, FileTextOutlined, LoadingOutlined } from '@ant-design/icons'
import styles from '../AgentPage.module.css'
import HighlightInput from '../HighlightInput'
import RelationTags from './RelationTags'
import AudioRecorder from './AudioRecorder'
import AudioTag from './AudioTag'
import ImagePreview from './ImagePreview'
import SlashCommands from './SlashCommands'
import CommandTag from './CommandTag'
import DocumentUploader from './DocumentUploader'
import { compressImage, checkFileSize, getFileSizeText } from '../../utils/imageCompression'
import { useStore } from '../../store'
import { uploadFile } from '../../utils/fileUpload'
import { saveFile } from '../../utils/fileStorage'

/**
 * 消息编辑器组件
 * @param {string} editingContent - 编辑中的内容
 * @param {function} setEditingContent - 设置编辑内容
 * @param {Array} editingImages - 编辑中的图片
 * @param {function} setEditingImages - 设置编辑图片
 * @param {Array} editingAudios - 编辑中的音频
 * @param {function} setEditingAudios - 设置编辑音频
 * @param {Array} editingFiles - 编辑中的文件（deepseek-ocr模式）
 * @param {function} setEditingFiles - 设置编辑文件
 * @param {Array} editingCommands - 编辑中的命令（deepseek-ocr模式）
 * @param {function} setEditingCommands - 设置编辑命令
 * @param {function} onSave - 保存的回调
 * @param {function} onCancel - 取消的回调
 * @param {Array} pendingHighlights - 待处理的高亮
 * @param {Array} pendingRelations - 待处理的关系
 * @param {object} pendingInfonIndex - 待处理的信息元索引
 * @param {object} sendLockState - 发送锁定状态
 * @param {string} originalContent - 原始内容
 * @param {Array} originalImages - 原始图片
 * @param {Array} originalAudios - 原始音频
 * @param {Array} originalFiles - 原始文件
 * @param {Array} originalCommands - 原始命令
 * @param {boolean} currentModelIsMultimodal - 当前模型是否支持多模态
 * @param {function} renderHighlightedText - 渲染高亮文本的函数
 * @param {string} inferenceMode - 推断模式 ('extract' | 'direct')
 * @param {function} processImageUpload - 处理图片上传的函数（用于直接推断模式）
 * @param {string} model - 当前选中的模型ID
 * @param {string} selectedResolution - 已选择的分辨率模式（deepseek-ocr模式）
 * @param {function} setSelectedResolution - 设置已选择的分辨率模式
 */
const MessageEditor = ({
  editingContent,
  setEditingContent,
  editingImages,
  setEditingImages,
  editingAudios = [],
  setEditingAudios,
  editingFiles = [],
  setEditingFiles,
  editingCommands = [],
  setEditingCommands,
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
  originalFiles = [],
  originalCommands = [],
  currentModelIsMultimodal,
  renderHighlightedText,
  inferenceMode,
  processImageUpload,
  model,
  selectedResolution = 'gundam',
  setSelectedResolution
}) => {
  // 斜杠命令状态
  const [showSlashCommands, setShowSlashCommands] = React.useState(false)
  const [slashCommandPosition, setSlashCommandPosition] = React.useState({ top: 0, left: 0 })
  const [showDocumentUploader, setShowDocumentUploader] = React.useState(false)

  const customProviders = useStore((state) => state.customProviders)
  const deepseekProvider = customProviders?.['deepseek-ocr']
  const currentSessionId = useStore((state) => state.currentSessionId)

  // 文件选择器引用（OCR模式直接选择文件）
  const fileInputRef = useRef(null)

  const handleAudioAdded = (audioData) => {
    setEditingAudios?.((prev) => [...prev, audioData])
  }

  const removeEditingAudio = (index) => {
    setEditingAudios?.((prev) => prev.filter((_, i) => i !== index))
  }
  
  const removeEditingFile = (index) => {
    setEditingFiles?.((prev) => prev.filter((_, i) => i !== index))
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
    // 检测是否输入了单个 "/"（OCR 模式显示菜单）
    if (newValue === '/' && (model === 'deepseek-ocr' || model === 'deepseek-ocr-local')) {
      // 计算命令菜单位置
      setTimeout(() => {
        let inputElement = null

        // 方法1: 通过当前焦点元素
        const activeElement = document.activeElement
        if (activeElement && activeElement.getAttribute('contenteditable') === 'true') {
          inputElement = activeElement
        }

        // 方法2: 查找所有 contentEditable 元素
        if (!inputElement) {
          const allEditables = document.querySelectorAll('[contenteditable="true"]')
          for (const editable of allEditables) {
            const rect = editable.getBoundingClientRect()
            if (rect.height > 0 && rect.width > 0 &&
                rect.top >= 0 && rect.top < window.innerHeight) {
              inputElement = editable
              break
            }
          }
        }

        if (inputElement) {
          const rect = inputElement.getBoundingClientRect()
          const estimatedMenuHeight = 7 * 50 + 20
          const menuLeft = rect.left

          let menuTop
          if (rect.top > estimatedMenuHeight + 10) {
            menuTop = rect.top - estimatedMenuHeight - 10
          } else {
            menuTop = rect.bottom + 10
          }

          setSlashCommandPosition({
            top: menuTop,
            left: menuLeft
          })
          setShowSlashCommands(true)
        } else {
          setSlashCommandPosition({
            top: 200,
            left: 300
          })
          setShowSlashCommands(true)
        }
      }, 50)
      
      return
    }

    setEditingContent(newValue)
  }

  // 处理斜杠命令选择
  const handleCommandSelect = (command) => {
    if (!command) {
      setShowSlashCommands(false)
      setTimeout(() => {
        const inputElement = document.activeElement
        if (inputElement && inputElement.getAttribute('contenteditable') === 'true') {
          const text = inputElement.textContent || ''
          if (text === '/') {
            inputElement.textContent = ''
            setEditingContent('')
          }
        }
      }, 0)
      return
    }

    // 设置选中的命令（只能选择一个）
    setEditingCommands([command])
    setShowSlashCommands(false)

    // 立即删除输入框中的"/"
    setTimeout(() => {
      const allEditables = document.querySelectorAll('[contenteditable="true"]')
      for (const editable of allEditables) {
        const text = editable.textContent || ''
        if (text === '/') {
          editable.textContent = ''
          setEditingContent('')
          break
        }
      }
    }, 0)
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
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      'application/vnd.ms-powerpoint',
      'application/vnd.oasis.opendocument.text',
      'application/vnd.oasis.opendocument.spreadsheet',
      'application/vnd.oasis.opendocument.presentation'
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
      setEditingFiles?.((prev) => [...prev, fileData])

      // 立即存储到 IndexedDB
      try {
        await saveFile(currentSessionId, 'temp_edit_' + fileId, fileId, file)
        console.log('[MessageEditor] 文件已保存到 IndexedDB:', file.name)
      } catch (error) {
        console.error('[MessageEditor] 保存文件到 IndexedDB 失败:', error)
      }

      // 立即开始上传到服务器
      try {
        const result = await uploadFile(file, deepseekProvider, (progress) => {
          setEditingFiles?.((prev) => 
            prev.map(f => 
              f.id === fileId 
                ? { ...f, uploadProgress: progress }
                : f
            )
          )
        })

        // 上传成功
        setEditingFiles?.((prev) => 
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
        
        console.log('[MessageEditor] 文件上传成功:', file.name)
      } catch (error) {
        // 上传失败
        setEditingFiles?.((prev) => 
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
        console.error('[MessageEditor] 文件上传失败:', error)
      }
    }

    if (validFiles.length > 0) {
      message.success(`成功添加 ${validFiles.length} 个文件`)
    }

    e.target.value = ''
  }

  // 检查内容是否发生变化
  const hasContentChanged = 
    editingContent !== originalContent || 
    JSON.stringify(editingImages) !== JSON.stringify(originalImages) ||
    JSON.stringify(editingAudios) !== JSON.stringify(originalAudios) ||
    JSON.stringify(editingFiles) !== JSON.stringify(originalFiles) ||
    JSON.stringify(editingCommands) !== JSON.stringify(originalCommands)
  
  // 按钮是否应该禁用
  const isOcrMode = model === 'deepseek-ocr' || model === 'deepseek-ocr-local'
  const isSaveDisabled = 
    !hasContentChanged || // 内容未修改
    (isOcrMode 
      ? ((!editingContent.trim() && editingFiles.length === 0 && editingCommands.length === 0) || 
         editingFiles.some(f => f.uploadStatus === 'uploading'))
      : (!editingContent.trim() && editingImages.length === 0 && editingAudios.length === 0)
    ) ||
    sendLockState.locked // 正在处理中
  return (
    <div className={styles.editingComposer}>
      {/* 斜杠命令菜单 */}
      <SlashCommands
        visible={showSlashCommands}
        position={slashCommandPosition}
        onSelectCommand={handleCommandSelect}
      />

      {/* 文档上传器 */}
      <DocumentUploader
        visible={showDocumentUploader}
        onFileSelect={() => setShowDocumentUploader(false)}
        onClose={() => setShowDocumentUploader(false)}
      />

      {/* 图片预览 */}
      {!isOcrMode && editingImages.length > 0 && (
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
      {!isOcrMode && editingAudios.length > 0 && (
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
      {/* 文件预览在输入框上方（OCR模式） */}
      {isOcrMode && editingFiles.length > 0 && (
        <div
          className={styles.composerFiles}
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '6px 8px',
            marginBottom: '8px'
          }}
        >
          {editingFiles.map((file, i) => (
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
                onClick={() => removeEditingFile(i)}
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
      {/* 输入框和按钮 */}
      <div className={styles.composerRow}>
        <HighlightInput
          className={styles.composerInput}
          value={editingContent}
          onChange={isOcrMode ? handleInputChange : setEditingContent}
          placeholder={
            isOcrMode && editingContent.trim() === ''
              ? "编辑消息... (输入 / 使用 OCR 功能)"
              : "编辑消息..."
          }
          highlights={pendingHighlights}
          autoSize={{ minRows: 2, maxRows: 10 }}
        />
        <div className={styles.composerButtons}>
          {isOcrMode ? (
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
                onClick={() => fileInputRef.current?.click()}
                title="上传文档进行 OCR 处理（可多选）"
              />
              {editingCommands.length > 0 && editingCommands[0] && (
                <CommandTag
                  command={editingCommands[0]}
                  resolution={selectedResolution}
                  onResolutionChange={setSelectedResolution}
                  removable={true}
                  onRemove={() => setEditingCommands([])}
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
            disabled={!currentModelIsMultimodal}
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
            </>
          )}
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

