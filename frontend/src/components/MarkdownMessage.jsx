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
  
  // 预处理内容，确保 markdown 格式正确解析5028
  const normalizedContent = useMemo(() => {
    const input = String(content ?? '')

    // 保护代码块 ```...```
    const fencePlaceholders = []
    let tmp = input.replace(/```[\s\S]*?```/g, (m) => {
      const i = fencePlaceholders.push(m) - 1
      return `@@FENCE_${i}@@`
    })

    // 保护行内代码 `...`
    const inlineCodePlaceholders = []
    tmp = tmp.replace(/`[^`\n]+?`/g, (m) => {
      const i = inlineCodePlaceholders.push(m) - 1
      return `@@CODE_${i}@@`
    })

    // 保护数学块 $$...$$
    const mathBlockPlaceholders = []
    tmp = tmp.replace(/\$\$[\s\S]*?\$\$/g, (m) => {
      const i = mathBlockPlaceholders.push(m) - 1
      return `@@MATHBLOCK_${i}@@`
    })

    // 保护行内数学 $...$
    const inlineMathPlaceholders = []
    tmp = tmp.replace(/\$(?:[^$\n]|\\\$)+\$/g, (m) => {
      const i = inlineMathPlaceholders.push(m) - 1
      return `@@MATH_${i}@@`
    })

    // 处理强调格式（特别是中文字符周围的情况）
    // 使用迭代方法，从长到短处理，确保优先匹配更复杂的格式
    
    // 1. 先处理粗体+斜体 ***...***
    while (tmp.includes('***')) {
      const replaced = tmp.replace(/\*\*\*(.+?)\*\*\*/, '<strong><em>$1</em></strong>')
      if (replaced === tmp) break
      tmp = replaced
    }
    while (tmp.includes('___')) {
      const replaced = tmp.replace(/___(.+?)___/, '<strong><em>$1</em></strong>')
      if (replaced === tmp) break
      tmp = replaced
    }
    
    // 2. 处理粗体 **...**
    while (tmp.includes('**')) {
      const replaced = tmp.replace(/\*\*(.+?)\*\*/, '<strong>$1</strong>')
      if (replaced === tmp) break
      tmp = replaced
    }
    while (tmp.includes('__')) {
      const replaced = tmp.replace(/__(.+?)__/, '<strong>$1</strong>')
      if (replaced === tmp) break
      tmp = replaced
    }
    
    // 3. 处理斜体 *...* (单个星号，且前后不是星号)
    while (/(?<!\*)\*(?!\*)/.test(tmp)) {
      const replaced = tmp.replace(/(?<!\*)\*(.+?)\*(?!\*)/, '<em>$1</em>')
      if (replaced === tmp) break
      tmp = replaced
    }
    while (/(?<!_)_(?!_)/.test(tmp)) {
      const replaced = tmp.replace(/(?<!_)_(.+?)_(?!_)/, '<em>$1</em>')
      if (replaced === tmp) break
      tmp = replaced
    }

    // 恢复占位符
    tmp = tmp.replace(/@@MATHBLOCK_(\d+)@@/g, (_, i) => mathBlockPlaceholders[Number(i)])
    tmp = tmp.replace(/@@MATH_(\d+)@@/g, (_, i) => inlineMathPlaceholders[Number(i)])
    tmp = tmp.replace(/@@FENCE_(\d+)@@/g, (_, i) => fencePlaceholders[Number(i)])
    tmp = tmp.replace(/@@CODE_(\d+)@@/g, (_, i) => inlineCodePlaceholders[Number(i)])

    return tmp
  }, [content])

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
          {normalizedContent}
        </ReactMarkdown>
      </div>
    </div>
  )
}


