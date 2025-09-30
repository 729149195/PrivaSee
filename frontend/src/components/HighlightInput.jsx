import React, { useRef, useEffect, useState, useCallback } from 'react'
import styles from './HighlightInput.module.css'

// 高亮输入框组件（中文注释）：支持实时高亮的 contentEditable 输入框
export default function HighlightInput({ 
  value, 
  onChange, 
  onPressEnter, 
  placeholder,
  className,
  highlights = [], // [{ keyword, color }]
  autoSize = { minRows: 1, maxRows: 6 }
}) {
  const editorRef = useRef(null)
  const [isFocused, setIsFocused] = useState(false)
  const isComposingRef = useRef(false)

  // 高亮文本（中文注释）：将纯文本转换为带高亮的 HTML
  const highlightText = useCallback((text) => {
    if (!text || !highlights.length) {
      return text || ''
    }

    // 按关键词长度降序排序（中文注释）
    const sortedHighlights = [...highlights].sort((a, b) => b.keyword.length - a.keyword.length)
    
    // 构建正则表达式（中文注释）
    const uniqueKeywords = [...new Set(sortedHighlights.map(h => h.keyword))]
    const pattern = uniqueKeywords.map(kw => kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
    if (!pattern) return text

    const regex = new RegExp(`(${pattern})`, 'gi')
    const parts = text.split(regex)
    
    return parts.map((part, i) => {
      if (i % 2 === 1) {
        const match = sortedHighlights.find(h => h.keyword.toLowerCase() === part.toLowerCase())
        if (match) {
          return `<mark class="${styles.highlight}" style="background-color: ${match.color}20; color: ${match.color}; font-weight: 600; padding: 0px 2px; border-radius: 2px;">${part}</mark>`
        }
      }
      return part
    }).join('')
  }, [highlights])

  // 保存光标位置（中文注释）
  const saveCursorPosition = useCallback(() => {
    const selection = window.getSelection()
    if (!selection.rangeCount) return 0
    
    const range = selection.getRangeAt(0)
    const preCaretRange = range.cloneRange()
    preCaretRange.selectNodeContents(editorRef.current)
    preCaretRange.setEnd(range.endContainer, range.endOffset)
    
    return preCaretRange.toString().length
  }, [])

  // 恢复光标位置（中文注释）
  const restoreCursorPosition = useCallback((pos) => {
    if (!editorRef.current) return

    const selection = window.getSelection()
    const range = document.createRange()
    
    let charCount = 0
    let nodeStack = [editorRef.current]
    let node, foundNode, foundOffset

    while (node = nodeStack.pop()) {
      if (node.nodeType === Node.TEXT_NODE) {
        const nextCharCount = charCount + node.length
        if (!foundNode && pos >= charCount && pos <= nextCharCount) {
          foundNode = node
          foundOffset = pos - charCount
          break
        }
        charCount = nextCharCount
      } else {
        for (let i = node.childNodes.length - 1; i >= 0; i--) {
          nodeStack.push(node.childNodes[i])
        }
      }
    }

    if (foundNode) {
      range.setStart(foundNode, foundOffset)
      range.collapse(true)
      selection.removeAllRanges()
      selection.addRange(range)
    }
  }, [])

  // 更新编辑器内容（中文注释）：当 highlights 变化时更新（流式高亮）
  useEffect(() => {
    if (!editorRef.current) return
    
    const plainText = editorRef.current.textContent || ''
    const cursorPos = saveCursorPosition()
    const highlightedHTML = highlightText(plainText)
    
    if (editorRef.current.innerHTML !== highlightedHTML) {
      editorRef.current.innerHTML = highlightedHTML
      // 只在焦点时恢复光标（中文注释）
      if (isFocused) {
        restoreCursorPosition(cursorPos)
      }
    }
  }, [highlights, highlightText, isFocused, saveCursorPosition, restoreCursorPosition])

  // 处理输入（中文注释）：只更新文本内容，不立即高亮（避免卡顿）
  const handleInput = useCallback(() => {
    if (!editorRef.current) return
    const plainText = editorRef.current.textContent || ''
    onChange?.(plainText)
  }, [onChange])

  // 处理键盘事件（中文注释）
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposingRef.current) {
      e.preventDefault()
      onPressEnter?.({ preventDefault: () => {}, shiftKey: false })
    }
  }, [onPressEnter])

  // 处理输入法（中文注释）
  const handleCompositionStart = useCallback(() => {
    isComposingRef.current = true
  }, [])

  const handleCompositionEnd = useCallback(() => {
    isComposingRef.current = false
  }, [])

  // 处理焦点（中文注释）
  const handleFocus = useCallback(() => {
    setIsFocused(true)
  }, [])

  const handleBlur = useCallback(() => {
    setIsFocused(false)
  }, [])

  // 同步 value 变化（中文注释）：当 value prop 变化时更新编辑器内容
  useEffect(() => {
    if (!editorRef.current) return
    
    const currentText = editorRef.current.textContent || ''
    const newText = value || ''
    
    // 只有当 value 与编辑器内容不同时才更新（中文注释）
    if (currentText !== newText) {
      const cursorPos = saveCursorPosition()
      const highlightedHTML = highlightText(newText)
      editorRef.current.innerHTML = highlightedHTML
      
      // 如果有焦点且不是清空操作，恢复光标（中文注释）
      if (isFocused && newText) {
        restoreCursorPosition(cursorPos)
      }
    }
  }, [value, highlightText, isFocused, saveCursorPosition, restoreCursorPosition])

  return (
    <div className={`${styles.container} ${className || ''}`}>
      <div
        ref={editorRef}
        contentEditable
        className={styles.editor}
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        onCompositionStart={handleCompositionStart}
        onCompositionEnd={handleCompositionEnd}
        onFocus={handleFocus}
        onBlur={handleBlur}
        data-placeholder={placeholder}
        suppressContentEditableWarning
      />
    </div>
  )
}
