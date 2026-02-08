import React from 'react'
import styles from '../AgentPage.module.css'

/**
 * 图片预览 Modal 组件（ChatGPT 风格）
 * @param {string} previewImage - 预览图片的 URL
 * @param {function} onClose - 关闭回调函数
 */
const ImagePreviewModal = ({ previewImage, onClose }) => {
  // 按 ESC 键关闭（hooks 必须在条件返回之前调用）
  React.useEffect(() => {
    if (!previewImage) return

    const handleEsc = (e) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    
    document.addEventListener('keydown', handleEsc)
    // 禁止页面滚动
    document.body.style.overflow = 'hidden'
    
    return () => {
      document.removeEventListener('keydown', handleEsc)
      document.body.style.overflow = ''
    }
  }, [previewImage, onClose])

  if (!previewImage) return null

  // 点击遮罩层关闭
  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div className={styles.imagePreviewOverlay} onClick={handleBackdropClick}>
      <button 
        className={styles.imagePreviewClose}
        onClick={onClose}
        aria-label="Close preview"
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
      <div className={styles.imagePreviewContent}>
        <img 
          src={previewImage} 
          alt="Preview" 
          className={styles.imagePreviewImg}
          onClick={(e) => e.stopPropagation()}
        />
      </div>
    </div>
  )
}

export default ImagePreviewModal

