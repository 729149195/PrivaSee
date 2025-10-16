import React from 'react'
import { Modal } from 'antd'
import styles from '../AgentPage.module.css'

/**
 * 图片预览 Modal 组件
 * @param {string} previewImage - 预览图片的 URL
 * @param {function} onClose - 关闭回调函数
 */
const ImagePreviewModal = ({ previewImage, onClose }) => {
  return (
    <Modal
      open={!!previewImage}
      onCancel={onClose}
      footer={null}
      width="90vw"
      centered
      className={styles.imagePreviewModal}
    >
      {previewImage && (
        <div className={styles.imagePreviewContainer}>
          <img src={previewImage} alt="Preview" className={styles.imagePreviewImg} />
        </div>
      )}
    </Modal>
  )
}

export default ImagePreviewModal

