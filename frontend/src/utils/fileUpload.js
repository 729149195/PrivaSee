/**
 * 文件上传工具
 * 用于在选中文件后立即上传到服务器
 */

const normalizeBaseUrl = (baseUrl) => {
  if (!baseUrl || typeof baseUrl !== 'string') return ''
  return baseUrl.replace(/\/$/, '')
}

/**
 * 上传文件到服务器
 * @param {File} file - 要上传的文件
 * @param {Object} provider - API provider配置
 * @param {Function} onProgress - 上传进度回调 (percent: 0-100)
 * @returns {Promise<Object>} - 返回上传结果 {success, filename, path, originalName, size}
 */
export async function uploadFile(file, provider, onProgress) {
  if (!file) {
    throw new Error('未提供需要上传的文件')
  }

  if (!provider) {
    throw new Error('未找到 DeepSeek OCR 的 API 配置')
  }

  const baseUrl = normalizeBaseUrl(provider.baseUrl)
  if (!baseUrl) {
    throw new Error('DeepSeek OCR API 基础地址无效')
  }

  const formData = new FormData()
  formData.append('file', file)

  const headers = {}
  if (provider.apiKey) {
    headers['Authorization'] = `Bearer ${provider.apiKey}`
  }

  // 使用 XMLHttpRequest 来跟踪上传进度
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()

    // 监听上传进度
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        const percent = Math.round((e.loaded / e.total) * 100)
        onProgress(percent)
      }
    })

    // 监听完成
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText)
          if (data.success) {
            resolve(data)
          } else {
            reject(new Error(data.error || '上传失败'))
          }
        } catch (error) {
          reject(new Error('解析响应失败'))
        }
      } else {
        let errorMessage = `HTTP ${xhr.status}`
        try {
          const errorData = JSON.parse(xhr.responseText)
          errorMessage = errorData.error || errorMessage
        } catch (e) {
          // 使用默认错误信息
        }
        reject(new Error(errorMessage))
      }
    })

    // 监听错误
    xhr.addEventListener('error', () => {
      reject(new Error('网络请求失败'))
    })

    // 监听中断
    xhr.addEventListener('abort', () => {
      reject(new Error('上传被中断'))
    })

    // 发起请求
    xhr.open('POST', `${baseUrl}/upload`)
    
    // 设置请求头
    Object.keys(headers).forEach(key => {
      xhr.setRequestHeader(key, headers[key])
    })

    xhr.send(formData)
  })
}

/**
 * 批量上传文件
 * @param {Array<File>} files - 要上传的文件列表
 * @param {Object} provider - API provider配置
 * @param {Function} onProgress - 整体进度回调 (current, total, percent)
 * @param {Function} onFileProgress - 单个文件进度回调 (fileIndex, percent)
 * @returns {Promise<Array<Object>>} - 返回上传结果数组
 */
export async function uploadFiles(files, provider, onProgress, onFileProgress) {
  const results = []
  const total = files.length

  for (let i = 0; i < total; i++) {
    const file = files[i]
    
    try {
      const result = await uploadFile(file, provider, (percent) => {
        if (onFileProgress) {
          onFileProgress(i, percent)
        }
      })
      
      results.push({ success: true, file, result })
      
      if (onProgress) {
        onProgress(i + 1, total, Math.round(((i + 1) / total) * 100))
      }
    } catch (error) {
      results.push({ success: false, file, error: error.message })
      
      if (onProgress) {
        onProgress(i + 1, total, Math.round(((i + 1) / total) * 100))
      }
    }
  }

  return results
}

