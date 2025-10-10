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
  const components = useMemo(() => ({
    // 代码块渲染：添加标题栏与复制
    code({ inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '')
      const lang = match ? match[1] : ''
      const codeText = String(children || '')
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
            <code className={className} {...props}>{codeText}</code>
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
    // 确保粗体正确渲染
    strong({ children }) {
      return <strong>{children}</strong>
    },
    // 确保斜体正确渲染
    em({ children }) {
      return <em>{children}</em>
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
          {content}
        </ReactMarkdown>
      </div>
    </div>
  )
}


