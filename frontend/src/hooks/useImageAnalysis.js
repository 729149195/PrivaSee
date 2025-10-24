import { useState, useCallback, useEffect } from 'react'
import { message as antdMessage } from 'antd'
import { useStore } from '../store'

/**
 * 图片分析管理 Hook
 * 自动分析图片并管理状态
 * 
 * 图片数据结构：
 * {
 *   id: string,              // 唯一标识
 *   url: string,             // base64 data URL
 *   status: 'uploading' | 'analyzing' | 'done' | 'error',
 *   analysis: string,        // 分析结果文本
 *   error: string,           // 错误信息
 *   timestamp: number        // 时间戳
 * }
 */
export function useImageAnalysis(inferenceMode) {
  const analyzeImage = useStore(state => state.analyzeImage)
  
  /**
   * 处理图片上传并自动分析（仅在直接推断模式下）
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
      analysis: '',
      error: '',
      timestamp: Date.now()
    }
    
    // 添加到列表（显示上传中状态）
    setImages(prev => [...prev, imageObj])
    
    // 如果不是直接推断模式，直接标记为完成（不需要分析）
    if (inferenceMode !== 'direct') {
      setTimeout(() => {
        setImages(prev => prev.map(img => 
          img.id === imageId ? { ...img, status: 'done' } : img
        ))
      }, 500)
      return imageObj
    }
    
    // 3秒后切换到解析状态
    setTimeout(() => {
      setImages(prev => prev.map(img => 
        img.id === imageId ? { ...img, status: 'analyzing' } : img
      ))
      
      // 开始分析
      analyzeImage(imageDataUrl)
        .then(analysisText => {
          // 分析成功
          setImages(prev => prev.map(img => 
            img.id === imageId 
              ? { ...img, status: 'done', analysis: analysisText }
              : img
          ))
          console.log('[Image Analysis] 图片分析完成:', imageId, analysisText)
        })
        .catch(error => {
          // 分析失败
          setImages(prev => prev.map(img => 
            img.id === imageId 
              ? { ...img, status: 'error', error: error.message }
              : img
          ))
          antdMessage.error(`图片分析失败: ${error.message}`)
          console.error('[Image Analysis] 图片分析失败:', imageId, error)
        })
    }, 3000)
    
    return imageObj
  }, [analyzeImage, inferenceMode])
  
  /**
   * 处理图片修改后的重新分析
   * @param {string} imageId - 图片ID
   * @param {string} imageDataUrl - 新的图片 URL
   * @param {Function} setImages - 更新图片列表的函数
   */
  const reanalyzeImage = useCallback(async (imageId, imageDataUrl, setImages) => {
    if (inferenceMode !== 'direct') return
    
    // 重置为上传中状态
    setImages(prev => prev.map(img => 
      img.id === imageId 
        ? { ...img, url: imageDataUrl, status: 'uploading', analysis: '', error: '' }
        : img
    ))
    
    // 3秒后切换到解析状态
    setTimeout(() => {
      setImages(prev => prev.map(img => 
        img.id === imageId ? { ...img, status: 'analyzing' } : img
      ))
      
      // 开始分析
      analyzeImage(imageDataUrl)
        .then(analysisText => {
          setImages(prev => prev.map(img => 
            img.id === imageId 
              ? { ...img, status: 'done', analysis: analysisText }
              : img
          ))
        })
        .catch(error => {
          setImages(prev => prev.map(img => 
            img.id === imageId 
              ? { ...img, status: 'error', error: error.message }
              : img
          ))
          antdMessage.error(`图片分析失败: ${error.message}`)
        })
    }, 3000)
  }, [analyzeImage, inferenceMode])
  
  /**
   * 取消图片分析（用户删除图片时）
   * @param {string} imageId - 图片ID
   * @param {Function} setImages - 更新图片列表的函数
   */
  const cancelImageAnalysis = useCallback((imageId, setImages) => {
    setImages(prev => prev.filter(img => img.id !== imageId))
  }, [])
  
  return {
    processImageUpload,
    reanalyzeImage,
    cancelImageAnalysis
  }
}

