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
    p({ children, ...props }) {
      const textContent = React.Children.toArray(children)
        .map(child => (typeof child === 'string' ? child : ''))
        .join('')
        .trim()
      if (!textContent && React.Children.count(children) === 0) return null
      return <p {...props}>{children}</p>
    },
    // 行内 code 仅保留行内样式；代码块由 pre 统一渲染，避免 DOM 嵌套告警
    code({ className, children, ...props }) {
      return <code className={clsx(styles.inlineCode, className)} {...props}>{children}</code>
    },
    pre({ children }) {
      const codeEl = React.Children.toArray(children)[0]
      if (!React.isValidElement(codeEl)) return <pre>{children}</pre>

      const className = codeEl.props?.className || ''
      const match = /language-(\w+)/.exec(className)
      const lang = match ? match[1] : 'text'
      const codeChildren = codeEl.props?.children

      const extractText = (node) => {
        if (typeof node === 'string') return node
        if (typeof node === 'number') return String(node)
        if (Array.isArray(node)) return node.map(extractText).join('')
        if (node && typeof node === 'object' && node.props && node.props.children) {
          return extractText(node.props.children)
        }
        return ''
      }

      const codeText = extractText(codeChildren) || ''
      const handleCopy = async () => {
        try {
          await navigator.clipboard.writeText(codeText)
        } catch (_) {
          try { copyToClipboard(codeText) } catch (_) {}
        }
      }

      return (
        <div className={styles.blockWrap}>
          <div className={styles.blockHeader}>
            <span className={styles.lang}>{lang}</span>
            <button className={styles.copyBtn} onClick={handleCopy}>Copy</button>
          </div>
          <pre className={clsx('hljs', styles.pre)}>
            <code className={className}>{codeChildren}</code>
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


