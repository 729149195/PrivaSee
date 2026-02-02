import { useCallback } from 'react'

/**
 * 图片分析管理 Hook
 * 管理图片上传状态
 * 
 * 图片数据结构：
 * {
 *   id: string,              // 唯一标识
 *   url: string,             // base64 data URL
 *   status: 'uploading' | 'done' | 'error',
 *   timestamp: number        // 时间戳
 * }
 */
export function useImageAnalysis() {
  /**
   * 处理图片上传
   * @param {string} imageDataUrl - 图片 base64 URL
   * @param {Function} setImages - 更新图片列表的函数
   * @returns {Promise<object>} 图片对象
   */
  const processImageUpload = useCallback(async (imageDataUrl, setImages) => {
    const imageId = `img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    // 创建初始图片对象
    const imageObj = {
      id: imageId,
      url: imageDataUrl,
      status: 'uploading',
      timestamp: Date.now()
    }
    
    // 添加到列表（显示上传中状态）
    setImages(prev => [...prev, imageObj])
    
    // 短暂延迟后标记为完成
    setTimeout(() => {
      setImages(prev => prev.map(img => 
        img.id === imageId ? { ...img, status: 'done' } : img
      ))
    }, 500)
    
    return imageObj
  }, [])
  
  /**
   * 取消图片（用户删除图片时）
   * @param {string} imageId - 图片ID
   * @param {Function} setImages - 更新图片列表的函数
   */
  const cancelImageAnalysis = useCallback((imageId, setImages) => {
    setImages(prev => prev.filter(img => img.id !== imageId))
  }, [])
  
  return {
    processImageUpload,
    cancelImageAnalysis
  }
}
