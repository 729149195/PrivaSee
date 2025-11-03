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
    // Prefer blob URL for better compatibility (especially for PDF viewers)
    if (fileObject && fileObject instanceof File) {
      const objectUrl = URL.createObjectURL(fileObject)
      setPreviewUrl(objectUrl)
      return () => {
        URL.revokeObjectURL(objectUrl)
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
      title={file?.name || 'Document preview'}
      width="80%"
      style={{ maxWidth: '1200px', top: 20 }}
      styles={{
        content: { borderRadius: 12, overflow: 'hidden' },
        header: { padding: '10px 16px', margin: 0, borderBottom: '1px solid #f0f0f0' },
        body: { padding: 0, height: 'calc(100vh - 180px)', overflow: 'hidden' }
      }}
      maskClosable
    >
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* 文件预览内容 - 固定高度、无额外滚动 */}
        <div style={{ flex: 1, overflow: 'hidden', padding: '0', backgroundColor: '#fff' }}>
          {previewUrl ? (
            file.type === 'application/pdf' ? (
              <embed
                src={previewUrl}
                type="application/pdf"
                width="100%"
                height="100%"
                style={{ minHeight: '100%', border: 'none', display: 'block' }}
              />
            ) : file.type.startsWith('image/') ? (
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                height: '100%',
                padding: '20px',
                overflow: 'auto'
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
                <p>Preview for this file type is not supported</p>
                <p style={{ fontSize: '12px', marginTop: '8px' }}>
                  File type: {file.type}
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
              <p>Preview is unavailable</p>
              <p style={{ fontSize: '12px', marginTop: '8px' }}>
                {!fileObject ? 'File content is only available during the session. Refresh makes it unavailable.' : 'Loading...'}
              </p>
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}

export default DocumentPreviewModal

