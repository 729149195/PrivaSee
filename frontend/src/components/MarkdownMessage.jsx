import React, { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import rehypeRaw from 'rehype-raw'
import copyToClipboard from 'copy-to-clipboard'
import clsx from 'clsx'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github-dark-dimmed.css'
import styles from './MarkdownMessage.module.css'

// 说明（中文注释）：
// 高保真 Markdown 渲染，接近 ChatGPT：
// - 支持 GFM 表格/任务列表
// - 支持 KaTeX 数学公式
// - 支持代码高亮与语言标题、复制按钮
// - 链接新窗口打开
// - 正确处理粗体、斜体等格式

export default function MarkdownMessage({ content = '' }) {
  const [copiedId, setCopiedId] = useState('')
  
  // 轻量预处理：去除开头的空行，避免在包含表格等块级元素时被父级样式拉出大空白
  const normalizedContent = useMemo(() => {
    const raw = String(content ?? '')
    // 移除最前面的连续空行（包含可能的空格/制表符）
    return raw.replace(/^(?:[ \t]*\r?\n)+/, '')
  }, [content])

  const components = useMemo(() => ({
    // 段落渲染：检查是否包含块级元素，避免嵌套错误
    p({ children, node, ...props }) {
      // 检查是否为空段落或只包含换行符
      const isEmptyParagraph = () => {
        if (!children) return true
        const childArray = React.Children.toArray(children)
        if (childArray.length === 0) return true
        
        // 检查是否只包含空白字符
        const textContent = childArray
          .map(child => typeof child === 'string' ? child : '')
          .join('')
          .trim()
        
        return textContent === ''
      }
      
      // 如果是空段落，不渲染
      if (isEmptyParagraph()) {
        return null
      }
      
      // 递归检查所有子元素，确保没有块级元素
      const hasBlockChild = (child) => {
        if (!React.isValidElement(child)) {
          return false
        }
        
        const type = child.type
        const className = child.props?.className || ''
        
        // 检查是否是块级元素
        const blockTypes = ['div', 'pre', 'blockquote', 'ul', 'ol', 'table']
        if (blockTypes.includes(type)) {
          return true
        }
        
        // 检查是否是代码块
        if (type === 'code') {
          if (typeof className === 'string' && className.includes('language-')) {
            return true
          }
        }
        
        // 检查自定义类名
        if (typeof className === 'string' && (
          className.includes('blockWrap') || 
          className.includes('tableWrap')
        )) {
          return true
        }
        
        // 递归检查子元素
        if (child.props && child.props.children) {
          const nestedChildren = React.Children.toArray(child.props.children)
          return nestedChildren.some(hasBlockChild)
        }
        
        return false
      }
      
      const childArray = React.Children.toArray(children)
      const hasBlockElement = childArray.some(hasBlockChild)
      
      // 如果包含块级元素，使用 div 替代 p
      if (hasBlockElement) {
        return <div {...props}>{children}</div>
      }
      
      return <p {...props}>{children}</p>
    },
    // 代码块渲染：添加标题栏与复制
    code({ inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '')
      const lang = match ? match[1] : ''
      
      // 正确提取文本内容，处理 children 可能是数组或对象的情况
      const extractText = (node) => {
        if (typeof node === 'string') return node
        if (typeof node === 'number') return String(node)
        if (Array.isArray(node)) return node.map(extractText).join('')
        if (node && typeof node === 'object' && node.props && node.props.children) {
          return extractText(node.props.children)
        }
        return ''
      }
      
      const codeText = extractText(children) || ''
      
      const handleCopy = async () => {
        try {
          await navigator.clipboard.writeText(codeText)
        } catch (_) {
          try { copyToClipboard(codeText) } catch (_) {}
        }
      }
      if (inline) {
        return <code className={styles.inlineCode} {...props}>{children}</code>
      }
      return (
        <div className={styles.blockWrap}>
          <div className={styles.blockHeader}>
            <span className={styles.lang}>{lang || 'text'}</span>
            <button className={styles.copyBtn} onClick={handleCopy}>Copy</button>
          </div>
          <pre className={clsx('hljs', styles.pre)}>
            <code className={className} {...props}>{children}</code>
          </pre>
        </div>
      )
    },
    a({ node, ...props }) {
      return <a target="_blank" rel="noopener noreferrer" {...props} />
    },
    table({ children }) {
      return <div className={styles.tableWrap}><table>{children}</table></div>
    },
  }), [])

  return (
    <div className={styles.root}>
      <div className={styles.markdown}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
          rehypePlugins={[rehypeRaw, rehypeKatex, [rehypeHighlight, { detect: true }]]}
          components={components}
          skipHtml={false}
        >
          {normalizedContent}
        </ReactMarkdown>
      </div>
    </div>
  )
}


