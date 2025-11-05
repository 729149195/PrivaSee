import React, { useState } from 'react'
import { Modal, Upload, Button, message as antdMessage } from 'antd'
import { InboxOutlined, DeleteOutlined } from '@ant-design/icons'

const { Dragger } = Upload

/**
 * 文档上传组件
 * 支持图片、PDF 等文档格式上传
 */
const DocumentUploader = ({ onFileSelect, onClose, visible = true }) => {
  const [fileList, setFileList] = useState([])

  // 支持的文件类型
  const ACCEPTED_TYPES = {
    'image/png': ['.png'],
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/gif': ['.gif'],
    'image/webp': ['.webp'],
    'image/bmp': ['.bmp'],
    'image/tiff': ['.tiff', '.tif'],
    'application/pdf': ['.pdf'],
    // Office 文档格式
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    'application/msword': ['.doc'],
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    'application/vnd.ms-excel': ['.xls'],
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
    'application/vnd.ms-powerpoint': ['.ppt'],
    // LibreOffice 格式
    'application/vnd.oasis.opendocument.text': ['.odt'],
    'application/vnd.oasis.opendocument.spreadsheet': ['.ods'],
    'application/vnd.oasis.opendocument.presentation': ['.odp'],
  }

  const acceptString = Object.values(ACCEPTED_TYPES).flat().join(',')

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: acceptString,
    fileList: fileList,
    beforeUpload: (file) => {
      // 检查文件类型
      if (!Object.keys(ACCEPTED_TYPES).includes(file.type)) {
        antdMessage.error('不支持的文件格式')
        return Upload.LIST_IGNORE
      }
      
      // 检查文件大小（最大 40MB）
      const isLt40M = file.size / 1024 / 1024 < 40
      if (!isLt40M) {
        antdMessage.error('文件大小不能超过 40MB')
        return Upload.LIST_IGNORE
      }

      setFileList([file])
      return false // 阻止自动上传
    },
    onRemove: () => {
      setFileList([])
    },
    showUploadList: {
      showRemoveIcon: true,
      removeIcon: <DeleteOutlined />
    }
  }

  const handleUpload = () => {
    if (fileList.length === 0) {
      antdMessage.warning('请先选择文件')
      return
    }
    onFileSelect(fileList)
    setFileList([])
  }

  const handleCancel = () => {
    setFileList([])
    onClose()
  }

  return (
    <Modal
      title="上传文档"
      open={visible}
      onCancel={handleCancel}
      footer={[
        <Button key="cancel" onClick={handleCancel}>
          取消
        </Button>,
        <Button
          key="upload"
          type="primary"
          onClick={handleUpload}
          disabled={fileList.length === 0}
        >
          上传
        </Button>
      ]}
      width={500}
      centered
      maskClosable={true}
    >
      <Dragger {...uploadProps}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
        <p className="ant-upload-hint">
          支持: 图片 (PNG, JPG, GIF, WebP, BMP, TIFF)<br/>
          文档 (PDF, Word, Excel, PowerPoint) (最大 40MB)
        </p>
      </Dragger>
    </Modal>
  )
}

export default DocumentUploader
