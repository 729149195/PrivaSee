import React from 'react'
import { Modal } from 'antd'
import { FileTextOutlined } from '@ant-design/icons'

/**
 * 文档预览 Modal
 * @param {object} file - 文件数据 {id, name, size, type, dataUrl}
 * @param {function} onClose - 关闭回调
 */
const DocumentPreviewModal = ({ file, onClose }) => {
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
      bodyStyle={{ padding: 0, maxHeight: 'calc(100vh - 120px)', overflow: 'hidden' }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)' }}>
        {/* 文件信息头部 */}
        <div style={{ 
          padding: '16px 24px', 
          borderBottom: '1px solid #f0f0f0',
          backgroundColor: '#fafafa'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <FileTextOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
            <div>
              <div style={{ fontSize: '16px', fontWeight: 500, color: '#262626' }}>
                {file.name}
              </div>
              <div style={{ fontSize: '12px', color: '#8c8c8c', marginTop: '4px' }}>
                {file.type} • {getFileSizeText(file)}
              </div>
            </div>
          </div>
        </div>
        
        {/* 文件预览内容 */}
        <div style={{ flex: 1, overflow: 'auto', padding: '24px', backgroundColor: '#fff' }}>
          {file.dataUrl ? (
            file.type === 'application/pdf' ? (
              <embed
                src={file.dataUrl}
                type="application/pdf"
                width="100%"
                height="100%"
                style={{ minHeight: '600px', border: 'none' }}
              />
            ) : file.type.startsWith('image/') ? (
              <img 
                src={file.dataUrl} 
                alt={file.name} 
                style={{ 
                  maxWidth: '100%', 
                  height: 'auto',
                  display: 'block',
                  margin: '0 auto'
                }} 
              />
            ) : (
              <div style={{ 
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
              textAlign: 'center', 
              padding: '60px 20px', 
              color: '#8c8c8c' 
            }}>
              <FileTextOutlined style={{ fontSize: '64px', marginBottom: '16px' }} />
              <p>文件内容不可用</p>
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}

export default DocumentPreviewModal

