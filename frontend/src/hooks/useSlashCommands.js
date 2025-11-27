import { useState, useRef, useCallback } from 'react'

/**
 * 斜杠命令处理 Hook
 * 统一 LandingView 和 MessageComposer 的斜杠命令逻辑
 * 
 * @param {object} params - 配置参数
 * @param {function} params.setInput - 设置输入内容的函数
 * @param {function} params.setSelectedCommand - 设置选中命令的函数
 * @param {string} params.model - 当前模型
 * @param {boolean} params.checkOcrModeOnly - 是否只在 OCR 模式下显示菜单
 */
export function useSlashCommands({
  setInput,
  setSelectedCommand,
  model = '',
  checkOcrModeOnly = false,
}) {
  const [showSlashCommands, setShowSlashCommands] = useState(false)
  const [slashCommandPosition, setSlashCommandPosition] = useState({ top: 0, left: 0 })
  const [showDocumentUploader, setShowDocumentUploader] = useState(false)
  const fileInputRef = useRef(null)

  /**
   * 计算斜杠命令菜单位置
   */
  const calculateMenuPosition = useCallback(() => {
    // 尝试多种方式找到输入框（HighlightInput 使用 contentEditable div）
    let inputElement = null

    // 方法1: 通过当前焦点元素
    const activeElement = document.activeElement
    if (activeElement && activeElement.getAttribute('contenteditable') === 'true') {
      inputElement = activeElement
    }

    // 方法2: 查找所有 contentEditable 元素，找到可见且在视口中的
    if (!inputElement) {
      const allEditables = document.querySelectorAll('[contenteditable="true"]')
      for (const editable of allEditables) {
        const rect = editable.getBoundingClientRect()
        if (rect.height > 0 && rect.width > 0 && rect.top >= 0 && rect.top < window.innerHeight) {
          inputElement = editable
          break
        }
      }
    }

    // 方法3: 通过 data-placeholder 查找
    if (!inputElement) {
      inputElement = document.querySelector('[contenteditable="true"][data-placeholder]')
    }

    if (inputElement) {
      const rect = inputElement.getBoundingClientRect()
      const estimatedMenuHeight = 7 * 50 + 20

      let menuTop
      if (rect.top > estimatedMenuHeight + 10) {
        menuTop = rect.top - estimatedMenuHeight - 10
      } else {
        menuTop = rect.bottom + 10
      }

      return { top: menuTop, left: rect.left }
    }

    // 默认位置
    return { top: 200, left: 300 }
  }, [])

  /**
   * 处理输入变化，检测斜杠命令
   */
  const handleInputChange = useCallback((newValue, originalSetInput) => {
    const isOcrMode = model === 'deepseek-ocr' || model === 'deepseek-ocr-local'
    const shouldShowMenu = checkOcrModeOnly ? (newValue === '/' && isOcrMode) : (newValue === '/')

    if (shouldShowMenu) {
      setTimeout(() => {
        const position = calculateMenuPosition()
        setSlashCommandPosition(position)
        setShowSlashCommands(true)
      }, 50)
      return
    }

    originalSetInput(newValue)
  }, [model, checkOcrModeOnly, calculateMenuPosition])

  /**
   * 处理斜杠命令选择
   */
  const handleCommandSelect = useCallback((command, clearInputCallback) => {
    if (!command) {
      setShowSlashCommands(false)
      // 取消时删除 "/"
      setTimeout(() => {
        const inputElement = document.activeElement
        if (inputElement && inputElement.getAttribute('contenteditable') === 'true') {
          const text = inputElement.textContent || ''
          if (text === '/') {
            inputElement.textContent = ''
            clearInputCallback('')
          }
        }
      }, 0)
      return
    }

    // 设置选中的命令
    setSelectedCommand(command)

    // 删除输入框中的 "/"
    setTimeout(() => {
      const inputElement = document.activeElement
      if (inputElement && inputElement.getAttribute('contenteditable') === 'true') {
        const text = inputElement.textContent || ''
        if (text === '/') {
          inputElement.textContent = ''
          clearInputCallback('')
        }
      }
    }, 0)

    setShowSlashCommands(false)

    // 如果需要打开文件选择器
    if (command.id && fileInputRef.current) {
      setTimeout(() => {
        fileInputRef.current?.click()
      }, 100)
    }
  }, [setSelectedCommand])

  /**
   * 关闭斜杠命令菜单
   */
  const closeSlashCommands = useCallback(() => {
    setShowSlashCommands(false)
  }, [])

  return {
    showSlashCommands,
    setShowSlashCommands,
    slashCommandPosition,
    showDocumentUploader,
    setShowDocumentUploader,
    fileInputRef,
    handleInputChange,
    handleCommandSelect,
    closeSlashCommands,
    calculateMenuPosition,
  }
}

/**
 * 图片工具函数
 */
export const imageUtils = {
  /**
   * 获取图片 URL（兼容字符串和对象格式）
   */
  getImageUrl: (img) => {
    return typeof img === 'string' ? img : img?.url
  },

  /**
   * 获取图片对象（兼容字符串和对象格式）
   */
  getImageData: (img) => {
    return typeof img === 'string' ? { url: img, status: 'done' } : img
  },
}
