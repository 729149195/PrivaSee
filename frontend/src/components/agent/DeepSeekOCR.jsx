import React, { useState } from 'react'
import { Progress, message } from 'antd'
import SlashCommands from './SlashCommands'
import DocumentUploader from './DocumentUploader'
import './DeepSeekOCR.css'
import { useStore } from '../../store'
import { callDeepseekOcr, callDeepseekOcrBatch } from '../../utils/deepseekOcrApi'

/**
 * DeepSeek-OCR 集成组件
 * 提供完整的 OCR 功能集成到聊天界面
 */
const DeepSeekOCR = ({ onOCRResult }) => {
  const [showCommands, setShowCommands] = useState(false)
  const [showUploader, setShowUploader] = useState(false)
  const [selectedCommand, setSelectedCommand] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [processingStage, setProcessingStage] = useState('')
  const [commandMenuPosition, setCommandMenuPosition] = useState({ top: 0, left: 0 })

  const customProviders = useStore((state) => state.customProviders)
  const deepseekProvider = customProviders?.['deepseek-ocr']

  /**
   * 处理斜杠命令选择
   */
  const handleCommandSelect = (command) => {
    if (!command) {
      setShowCommands(false)
      return
    }

    setSelectedCommand(command)
    setShowCommands(false)
    setShowUploader(true)
  }

  /**
   * 处理文件上传
   */
  const handleFileSelect = async (files) => {
    if (!files || files.length === 0) {
      setShowUploader(false)
      return
    }

    setShowUploader(false)
    setProcessing(true)
    setUploadProgress(0)
    setProcessingStage('准备上传...')

    try {
      if (!deepseekProvider) {
        throw new Error('未配置 DeepSeek OCR API，请先在设置中添加')
      }

      // 单文件处理
      if (files.length === 1) {
        const result = await processOCR(files[0], selectedCommand.id, deepseekProvider)
        if (onOCRResult) {
          onOCRResult({
            command: selectedCommand,
            result: result,
            file: files[0]
          })
        }
        message.success('OCR 处理完成！')
      } 
      // 批量处理
      else {
        const results = await processBatchOCR(files, selectedCommand.id, deepseekProvider)
        if (onOCRResult) {
          onOCRResult({
            command: selectedCommand,
            results: results,
            files: files,
            batch: true
          })
        }
        message.success('批量 OCR 处理完成！')
      }
    } catch (error) {
      console.error('OCR 处理失败:', error)
      message.error(`OCR 处理失败: ${error.message}`)
    } finally {
      setProcessing(false)
      setUploadProgress(0)
      setProcessingStage('')
      setSelectedCommand(null)
    }
  }

  /**
   * 单个文件 OCR 处理
   */
  const processOCR = async (file, functionType, provider) => {
    const result = await callDeepseekOcr({
      file,
      commandId: functionType,
      provider,
      onProgress: ({ value, stage }) => {
        if (typeof value === 'number') {
          setUploadProgress(Math.max(0, Math.min(100, value)))
        }
        if (stage) {
          setProcessingStage(stage)
        }
      }
    })

    return {
      text: result.text,
      function: functionType,
      function_name: result.command?.label,
      metadata: {
        filename: file.name,
        timestamp: new Date().toISOString()
      },
      raw: result.raw
    }
  }

  /**
   * 批量文件 OCR 处理
   */
  const processBatchOCR = async (files, functionType, provider) => {
    const results = await callDeepseekOcrBatch({
      files,
      commandId: functionType,
      provider,
      onProgress: ({ value, stage }) => {
        if (typeof value === 'number') {
          setUploadProgress(Math.max(0, Math.min(100, value)))
        }
        if (stage) {
          setProcessingStage(stage)
        }
      }
    })

    return results.map(({ file, text, command, raw }) => ({
      text,
      function: functionType,
      function_name: command?.label,
      metadata: {
        filename: file.name,
        timestamp: new Date().toISOString()
      },
      raw
    }))
  }

  /**
   * 显示命令菜单
   */
  const showCommandMenu = (position) => {
    setCommandMenuPosition(position)
    setShowCommands(true)
  }

  return (
    <>
      {/* 斜杠命令菜单 */}
      <SlashCommands
        visible={showCommands}
        position={commandMenuPosition}
        onSelectCommand={handleCommandSelect}
      />

      {/* 文档上传器 */}
      {showUploader && (
        <DocumentUploader
          onFileSelect={handleFileSelect}
          onClose={() => {
            setShowUploader(false)
            setSelectedCommand(null)
          }}
        />
      )}

      {/* 处理中提示 */}
      {processing && (
        <div className="ocr-processing-overlay">
          <div className="ocr-processing-modal">
            <div className="spinner"></div>
            <p style={{ marginTop: '20px', marginBottom: '10px' }}>
              {selectedCommand?.label}
            </p>
            <Progress
              percent={uploadProgress}
              status={uploadProgress === 100 ? 'success' : 'active'}
              strokeColor={{
                from: '#108ee9',
                to: '#87d068',
              }}
              style={{ width: '80%', margin: '0 auto' }}
            />
            <p className="processing-hint" style={{ marginTop: '10px', fontSize: '12px', color: '#8c8c8c' }}>
              {processingStage}
            </p>
          </div>
        </div>
      )}
    </>
  )
}

/**
 * OCR 按钮组件
 * 用于在聊天输入框附近显示 OCR 功能入口
 */
export const OCRButton = ({ onClick }) => {
  return (
    <button 
      className="ocr-button" 
      onClick={onClick}
      title="OCR 功能 (输入 / 查看所有功能)"
    >
      📄 OCR
    </button>
  )
}

/**
 * OCR 结果显示组件
 */
export const OCRResultDisplay = ({ result, command }) => {
  const [expanded, setExpanded] = useState(true)

  if (!result) return null

  return (
    <div className="ocr-result-container">
      <div 
        className="ocr-result-header"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="ocr-result-title">
          {command?.label || '📝 OCR 识别结果'}
        </span>
        <span className="ocr-result-toggle">
          {expanded ? '▼' : '▶'}
        </span>
      </div>
      
      {expanded && (
        <div className="ocr-result-content">
          <pre className="ocr-result-text">{result.text}</pre>
          {result.metadata && (
            <div className="ocr-result-metadata">
              <small>
                文件: {result.metadata.filename} | 
                模式: {result.metadata.resolution_mode} | 
                时间: {new Date(result.metadata.timestamp).toLocaleString()}
              </small>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Hook: 使用 OCR 功能
 * 在任何组件中使用 OCR 功能
 */
export const useOCR = () => {
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const customProviders = useStore((state) => state.customProviders)

  const processDocument = async (file, functionType = 'free_ocr', options = {}) => {
    setIsProcessing(true)
    setError(null)

    try {
      const provider = customProviders?.['deepseek-ocr']
      if (!provider) {
        throw new Error('未配置 DeepSeek OCR API，请先在设置中添加')
      }

      const result = await callDeepseekOcr({
        file,
        commandId: functionType,
        provider,
        question: options.question,
        onProgress: options.onProgress
      })

      return {
        text: result.text,
        function: functionType,
        function_name: result.command?.label,
        metadata: {
          filename: file?.name,
          timestamp: new Date().toISOString()
        },
        raw: result.raw
      }
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setIsProcessing(false)
    }
  }

  return {
    processDocument,
    isProcessing,
    error
  }
}

export default DeepSeekOCR

