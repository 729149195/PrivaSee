import React, { useState, useEffect } from 'react'
import { Modal } from 'antd'
import { FileTextOutlined } from '@ant-design/icons'

/**
 * 文档预览 Modal
 * @param {object} file - 文件数据 {id, name, size, type}
 * @param {File} fileObject - File 对象（用于生成预览）
 * @param {function} onClose - 关闭回调
 */
const DocumentPreviewModal = ({ file, fileObject, onClose }) => {
  const [previewUrl, setPreviewUrl] = useState(null)

  useEffect(() => {
    // 如果有 File 对象，创建临时的 dataUrl 用于预览
    if (fileObject && fileObject instanceof File) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreviewUrl(e.target.result)
      }
      reader.onerror = () => {
        console.error('[DocumentPreview] 读取文件失败')
        setPreviewUrl(null)
      }
      reader.readAsDataURL(fileObject)
      
      // 清理函数：组件卸载时释放 URL（如果是通过 URL.createObjectURL 创建的）
      return () => {
        // FileReader 不需要释放，仅在使用 createObjectURL 时需要
      }
    } else {
      setPreviewUrl(null)
    }
  }, [fileObject])

  if (!file) return null

  const getFileSizeText = (fileData) => {
    const bytes = fileData.size || 0
    if (bytes < 1024) return `${bytes}B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
    return `${(bytes / (1024 * 1024)).toFixed(2)}MB`
  }

  return (
    <Modal
      open={!!file}
      onCancel={onClose}
      footer={null}
      width="80%"
      style={{ maxWidth: '1200px', top: 20 }}
      styles={{ body: { padding: 0, maxHeight: 'calc(100vh - 120px)', overflow: 'hidden' } }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)' }}>
        {/* 文件预览内容 - 直接显示，不带文件名栏 */}
        <div style={{ flex: 1, overflow: 'auto', padding: '0', backgroundColor: '#fff' }}>
          {previewUrl ? (
            file.type === 'application/pdf' ? (
              <embed
                src={previewUrl}
                type="application/pdf"
                width="100%"
                height="100%"
                style={{ minHeight: '100%', border: 'none' }}
              />
            ) : file.type.startsWith('image/') ? (
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                minHeight: '100%',
                padding: '20px'
              }}>
                <img 
                  src={previewUrl} 
                  alt={file.name} 
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '100%',
                    height: 'auto',
                    objectFit: 'contain'
                  }} 
                />
              </div>
            ) : (
              <div style={{ 
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '100%',
                textAlign: 'center', 
                padding: '60px 20px', 
                color: '#8c8c8c' 
              }}>
                <FileTextOutlined style={{ fontSize: '64px', marginBottom: '16px' }} />
                <p>暂不支持此文件类型的预览</p>
                <p style={{ fontSize: '12px', marginTop: '8px' }}>
                  文件类型: {file.type}
                </p>
              </div>
            )
          ) : (
            <div style={{ 
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '100%',
              textAlign: 'center', 
              padding: '60px 20px', 
              color: '#8c8c8c' 
            }}>
              <FileTextOutlined style={{ fontSize: '64px', marginBottom: '16px' }} />
              <p>文件预览不可用</p>
              <p style={{ fontSize: '12px', marginTop: '8px' }}>
                {!fileObject ? '文件内容仅在会话期间可用，刷新后无法预览' : '正在加载...'}
              </p>
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}

export default DocumentPreviewModal

