import { useState } from 'react'

/**
 * 图片选择管理 Hook
 * 管理选择的图片和预览状态
 */
export function useImageSelection() {
  const [selectedImages, setSelectedImages] = useState([])
  const [previewImage, setPreviewImage] = useState(null)

  /**
   * 处理图片选择
   * 将文件读取为 data URL 后加入队列
   */
  const handlePickImages = async (e) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    
    const toDataUrl = (file) => new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
    
    try {
      const urls = await Promise.all(files.map(toDataUrl))
      setSelectedImages((prev) => [...prev, ...urls])
    } catch (_) { }
    e.target.value = ''
  }

  /**
   * 移除选中的图片
   */
  const removeSelectedImage = (idx) => {
    setSelectedImages((prev) => prev.filter((_, i) => i !== idx))
  }

  /**
   * 清空所有选中的图片
   */
  const clearSelectedImages = () => {
    setSelectedImages([])
  }

  return {
    selectedImages,
    setSelectedImages,
    previewImage,
    setPreviewImage,
    handlePickImages,
    removeSelectedImage,
    clearSelectedImages
  }
}

