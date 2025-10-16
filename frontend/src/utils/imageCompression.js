/**
 * 图片压缩工具
 * 用于在上传前压缩图片，避免超出localStorage存储限制
 */

/**
 * 压缩图片
 * @param {File} file - 原始图片文件
 * @param {Object} options - 压缩选项
 * @param {number} options.maxWidth - 最大宽度，默认1920
 * @param {number} options.maxHeight - 最大高度，默认1080
 * @param {number} options.quality - 压缩质量 0-1，默认0.8
 * @param {number} options.maxSizeMB - 最大文件大小(MB)，默认2MB
 * @returns {Promise<string>} - 返回压缩后的base64字符串
 */
export const compressImage = (file, options = {}) => {
  const {
    maxWidth = 1920,
    maxHeight = 1080,
    quality = 0.8,
    maxSizeMB = 2
  } = options

  return new Promise((resolve, reject) => {
    // 检查文件类型
    if (!file.type.startsWith('image/')) {
      reject(new Error('文件不是图片类型'))
      return
    }

    const reader = new FileReader()
    
    reader.onerror = () => reject(new Error('读取文件失败'))
    
    reader.onload = (e) => {
      const img = new Image()
      
      img.onerror = () => reject(new Error('加载图片失败'))
      
      img.onload = () => {
        try {
          // 计算缩放后的尺寸
          let { width, height } = img
          
          if (width > maxWidth || height > maxHeight) {
            const ratio = Math.min(maxWidth / width, maxHeight / height)
            width = Math.floor(width * ratio)
            height = Math.floor(height * ratio)
          }
          
          // 创建canvas进行压缩
          const canvas = document.createElement('canvas')
          canvas.width = width
          canvas.height = height
          
          const ctx = canvas.getContext('2d')
          ctx.drawImage(img, 0, 0, width, height)
          
          // 尝试不同的质量级别，确保文件大小在限制内
          let currentQuality = quality
          let compressed = canvas.toDataURL('image/jpeg', currentQuality)
          
          // 估算base64大小（字节）
          const getBase64Size = (base64) => {
            const stringLength = base64.length - 'data:image/jpeg;base64,'.length
            const sizeInBytes = 4 * Math.ceil(stringLength / 3) * 0.5624896334383812
            return sizeInBytes
          }
          
          // 如果文件仍然太大，逐步降低质量
          let attempts = 0
          const maxAttempts = 5
          const targetSizeBytes = maxSizeMB * 1024 * 1024
          
          while (getBase64Size(compressed) > targetSizeBytes && attempts < maxAttempts) {
            currentQuality -= 0.1
            if (currentQuality < 0.1) {
              currentQuality = 0.1
              break
            }
            compressed = canvas.toDataURL('image/jpeg', currentQuality)
            attempts++
          }
          
          // 如果还是太大，进一步缩小尺寸
          if (getBase64Size(compressed) > targetSizeBytes && width > 800) {
            const scaleFactor = 0.7
            canvas.width = Math.floor(width * scaleFactor)
            canvas.height = Math.floor(height * scaleFactor)
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
            compressed = canvas.toDataURL('image/jpeg', 0.7)
          }
          
          const finalSizeMB = (getBase64Size(compressed) / 1024 / 1024).toFixed(2)
          console.log(`[ImageCompression] 原始大小: ${(file.size / 1024 / 1024).toFixed(2)}MB, 压缩后: ${finalSizeMB}MB, 质量: ${currentQuality.toFixed(2)}`)
          
          resolve(compressed)
        } catch (error) {
          reject(new Error(`压缩图片失败: ${error.message}`))
        }
      }
      
      img.src = e.target.result
    }
    
    reader.readAsDataURL(file)
  })
}

/**
 * 检查文件大小是否在限制内
 * @param {File} file - 文件对象
 * @param {number} maxSizeMB - 最大大小(MB)，默认10MB
 * @returns {boolean} - 是否在限制内
 */
export const checkFileSize = (file, maxSizeMB = 10) => {
  const sizeMB = file.size / 1024 / 1024
  return sizeMB <= maxSizeMB
}

/**
 * 获取文件大小的友好显示
 * @param {File} file - 文件对象
 * @returns {string} - 格式化的大小字符串
 */
export const getFileSizeText = (file) => {
  const sizeMB = file.size / 1024 / 1024
  if (sizeMB >= 1) {
    return `${sizeMB.toFixed(2)}MB`
  }
  const sizeKB = file.size / 1024
  return `${sizeKB.toFixed(2)}KB`
}

