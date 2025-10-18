import React, { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkBreaks from 'remark-breaks'
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
  
  // react-markdown 已经可以正确处理所有 Markdown 格式，无需额外预处理
  const normalizedContent = useMemo(() => {
    return String(content ?? '')
  }, [content])

  const components = useMemo(() => ({
    // 段落渲染：检查是否包含块级元素，避免嵌套错误
    p({ children, node, ...props }) {
      // 简化策略：如果子元素中有任何非文本节点，就使用 div
      // 这样可以避免所有可能的嵌套问题
      const childArray = React.Children.toArray(children)
      const hasNonTextChild = childArray.some(child => {
        // 如果是 React 元素（不是纯文本或数字）
        if (React.isValidElement(child)) {
          // 检查是否是代码块或其他块级元素
          const type = child.type
          const className = child.props?.className || ''
          
          // code 元素（可能是内联代码或代码块）
          if (type === 'code') {
            // 如果有 language- 前缀，说明是代码块
            if (typeof className === 'string' && className.includes('language-')) {
              return true
            }
          }
          
          // 检查是否是 div 或 pre（代码块组件返回的）
          if (type === 'div' || type === 'pre') {
            return true
          }
          
          // 检查自定义类名
          if (typeof className === 'string' && (
            className.includes('blockWrap') || 
            className.includes('tableWrap')
          )) {
            return true
          }
        }
        return false
      })
      
      // 如果包含非文本子元素，使用 div 替代 p
      if (hasNonTextChild) {
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


