import React from 'react'
import { Spin, Tooltip } from 'antd'
import { LoadingOutlined, ExclamationCircleOutlined } from '@ant-design/icons'
import styles from './ImagePreview.module.css'

/**
 * 图片预览组件，支持状态显示
 * @param {object} imageData - 图片数据对象 { id, url, status, analysis, error }
 * @param {function} onRemove - 移除回调
 * @param {function} onClick - 点击回调
 * @param {boolean} removable - 是否可移除
 */
const ImagePreview = ({ imageData, onRemove, onClick, removable = true }) => {
  const { url, status = 'done', error } = imageData || {}
  
  // 判断是否显示遮罩
  const showOverlay = status === 'uploading' || status === 'analyzing' || status === 'error'
  
  // 遮罩文本
  let overlayText = ''
  let overlayIcon = null
  
  if (status === 'uploading') {
    overlayText = '上传中...'
    overlayIcon = <LoadingOutlined style={{ fontSize: 16 }} />
  } else if (status === 'analyzing') {
    overlayText = '解析中...'
    overlayIcon = <LoadingOutlined style={{ fontSize: 16 }} />
  } else if (status === 'error') {
    overlayText = '分析失败'
    overlayIcon = <ExclamationCircleOutlined style={{ fontSize: 16 }} />
  }
  
  return (
    <div className={styles.imagePreviewContainer}>
      <img 
        src={url} 
        alt="preview" 
        className={styles.previewImg}
        onClick={() => onClick?.(url)}
        style={{ cursor: onClick ? 'pointer' : 'default' }}
      />
      
      {showOverlay && (
        <div className={styles.overlay}>
          <div className={styles.overlayContent}>
            {overlayIcon}
            <span className={styles.overlayText}>{overlayText}</span>
            {status === 'error' && error && (
              <Tooltip title={error}>
                <span className={styles.errorDetail}>查看详情</span>
              </Tooltip>
            )}
          </div>
        </div>
      )}
      
      {removable && (
        <button 
          className={styles.removeButton}
          onClick={(e) => { 
            e.stopPropagation()
            onRemove?.()
          }}
        >
          ✕
        </button>
      )}
    </div>
  )
}

export default ImagePreview

