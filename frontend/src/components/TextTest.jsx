import React, { useEffect, useRef, useState } from 'react'
import styles from './TextTest.module.css'
import { useStore } from '../store'

export default function TextTest() {
    const text = useStore((s) => s.text)
    const setText = useStore((s) => s.setText)
    const [highlightedHtml, setHighlightedHtml] = useState('')
    const [isAnalyzingPII, setIsAnalyzingPII] = useState(false)
    const [errorMessage, setErrorMessage] = useState('')
    const abortRef = useRef(null)
    const inputRef = useRef(null)
    const [language, setLanguage] = useState('en')
    const [model, setModel] = useState('spacy/en_core_web_lg')
    const [allEntities, setAllEntities] = useState([])
    const [selectedEntities, setSelectedEntities] = useState([])
    const [uncertHtml, setUncertHtml] = useState('')
    const [uncertModel, setUncertModel] = useState('distilgpt2')
    const [uncertSamples, setUncertSamples] = useState(3)
    const [uncertSteps, setUncertSteps] = useState(4)

    const escaped = (value) =>
        value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')

    const buildHighlightedHtml = (rawText, entities) => {
        if (!rawText) return ''
        if (!entities || entities.length === 0) return escaped(rawText)

        // Filter overlapping by preferring earlier, longer spans
        const accepted = []
        let lastEnd = -1
        for (const ent of entities) {
            if (ent.start >= lastEnd) {
                accepted.push(ent)
                lastEnd = ent.end
            }
        }

        let html = ''
        let cursor = 0
        for (const ent of accepted) {
            const before = rawText.slice(cursor, ent.start)
            const chunk = rawText.slice(ent.start, ent.end)
            html += escaped(before)
            const conf = (ent.score ?? 0).toFixed(2)
            const title = `${ent.entity_type} (${conf})`
            const typeClass = styles[`type-${ent.entity_type}`] || ''
            const className = `${styles.pii} ${typeClass}`.trim()
            const label = `${ent.entity_type} ${conf}`
            html += `<span class="${className}" data-entity="${escaped(label)}" title="${escaped(title)}">${escaped(chunk)}</span>`
            cursor = ent.end
        }
        html += escaped(rawText.slice(cursor))
        return html
    }

    const runPIIAnalyze = async () => {
        // Cancel previous in-flight
        if (abortRef.current) {
            abortRef.current.abort()
            abortRef.current = null
        }
        setErrorMessage('')
        const trimmed = text ?? ''
        if (!trimmed) {
            setHighlightedHtml('')
            setIsAnalyzingPII(false)
            return
        }
        const controller = new AbortController()
        abortRef.current = controller
        try {
            setIsAnalyzingPII(true)
            const resp = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: trimmed, language, model, entities: selectedEntities }),
                signal: controller.signal,
            })
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}))
                throw new Error(err?.detail || `Request failed: ${resp.status}`)
            }
            const data = await resp.json()
            const html = buildHighlightedHtml(trimmed, (data && data.entities) || [])
            setHighlightedHtml(html)
        } catch (e) {
            if (e?.name === 'AbortError') return
            setErrorMessage(e?.message || 'Analyze error')
            setHighlightedHtml(escaped(text))
        } finally {
            setIsAnalyzingPII(false)
        }
    }

    // Fetch supported entities when model/language changes
    useEffect(() => {
        let cancelled = false
            ; (async () => {
                try {
                    const resp = await fetch('http://localhost:8000/entities', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ language, model }),
                    })
                    if (!resp.ok) throw new Error(`Failed to get entities: ${resp.status}`)
                    const data = await resp.json()
                    if (!cancelled) {
                        setAllEntities(data.entities || [])
                        // 默认全选：若之前没有选择，自动全选；若已有选择，仅保留存在的项
                        setSelectedEntities((prev) => {
                            const list = data.entities || []
                            if (!prev || prev.length === 0) return list
                            return prev.filter((e) => list.includes(e))
                        })
                    }
                } catch (e) {
                    if (!cancelled) {
                        setAllEntities([])
                    }
                }
            })()
        return () => {
            cancelled = true
        }
    }, [language, model])

    // Build uncertainty html from token spans
    const buildUncertaintyHtml = (rawText, tokens) => {
        if (!rawText) return ''
        if (!tokens || tokens.length === 0) return escaped(rawText)
        // Map per char the max(mean, std) for heat intensity
        const scores = new Array(rawText.length).fill(0)
        for (const t of tokens) {
            const s = Math.max(t.score_mean ?? 0, t.score_std ?? 0)
            for (let i = t.start; i < t.end && i < scores.length; i++) scores[i] = Math.max(scores[i], s)
        }
        let html = ''
        for (let i = 0; i < rawText.length;) {
            const base = scores[i]
            if (base <= 0) {
                html += escaped(rawText[i])
                i += 1
                continue
            }
            let j = i + 1
            while (j < rawText.length && scores[j] === base) j += 1
            const chunk = rawText.slice(i, j)
            const alpha = Math.min(0.85, Math.max(0.1, base))
            const bg = `rgba(255, 77, 79, ${alpha})`
            html += `<span class="${styles.heatSpan}" style="background:${bg}">${escaped(chunk)}</span>`
            i = j
        }
        return html
    }

    // Request uncertainty whenever text changes
    const [isAnalyzingUncert, setIsAnalyzingUncert] = useState(false)
    const runUncertainty = async () => {
        if (!text) {
            setUncertHtml('')
            return
        }
        const controller = new AbortController()
        try {
            setIsAnalyzingUncert(true)
            const resp = await fetch('http://localhost:8000/uncertainty', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, model: uncertModel, samples: uncertSamples, ig_steps: uncertSteps, max_input_tokens: 256 }),
                signal: controller.signal,
            })
            if (!resp.ok) throw new Error(`Uncertainty failed: ${resp.status}`)
            const data = await resp.json()
            setUncertHtml(buildUncertaintyHtml(text, data.tokens || []))
        } catch (e) {
            setUncertHtml(escaped(text))
        } finally {
            setIsAnalyzingUncert(false)
        }
    }

    const runAnalyzeAll = () => {
        runPIIAnalyze()
        runUncertainty()
    }

    // Auto-grow textarea height to fit content (no max height cap)
    useEffect(() => {
        const el = inputRef.current
        if (!el) return
        el.style.height = 'auto'
        el.style.height = `${el.scrollHeight}px`
    }, [text])

    return (
        <div className={styles.container}>
            <h1 className={styles.title}>TextTest</h1>
            <div className={styles.panel}>
                <div className={styles.panelSection}>
                    <div className={styles.panelTitle}>PII Named Entity Recognition</div>
                    <div className={styles.panelGrid}>
                        <div className={styles.field}>
                            <label className={styles.label}>NER model</label>
                            <select className={styles.select} value={model} onChange={(e) => setModel(e.target.value)}>
                                <option value="spacy/en_core_web_lg">spaCy/en_core_web_lg</option>
                                <option value="spacy/en_core_web_sm">spaCy/en_core_web_sm</option>
                            </select>
                        </div>
                        <div className={styles.field}>
                            <label className={styles.label}>Language</label>
                            <select className={styles.select} value={language} onChange={() => setLanguage('en')}>
                                <option value="en">English (en)</option>
                            </select>
                        </div>
                        <div className={styles.help}>按实体过滤仅影响高亮展示。</div>
                        <div className={styles.toolbar}>
                            <div className={styles.toolbarLeft}>
                                <span className={styles.count}>已选 {selectedEntities.length}/{allEntities.length}</span>
                                <div className={styles.filterLegend}>
                                    <span className={`${styles.chip} ${styles.chipSample}`}>未选</span>
                                    <span className={`${styles.chip} ${styles.chipActive} ${styles.chipSample}`}>已选</span>
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: 8 }}>
                                <button type="button" className={styles.btnSecondary}
                                    onClick={() => setSelectedEntities(allEntities)}>全选</button>
                                <button type="button" className={styles.btnSecondary}
                                    onClick={() => setSelectedEntities([])}>全不选</button>
                            </div>
                        </div>
                        <div className={styles.chipGroup}>
                            {allEntities.map((ent) => {
                                const active = selectedEntities.includes(ent)
                                return (
                                    <label key={ent} className={`${styles.chip} ${active ? styles.chipActive : ''}`}>
                                        <input
                                            type="checkbox"
                                            checked={active}
                                            onChange={(e) => {
                                                setSelectedEntities((prev) => {
                                                    if (e.target.checked) return [...new Set([...prev, ent])]
                                                    return prev.filter((x) => x !== ent)
                                                })
                                            }}
                                        />
                                        <span>{ent}</span>
                                    </label>
                                )
                            })}
                        </div>
                    </div>
                </div>

                <div className={styles.panelSection}>
                    <div className={styles.panelTitle}>Uncertainty Recognition Configuration</div>
                    <div className={styles.panelGrid}>
                        <div className={styles.field}>
                            <label className={styles.label}>Model</label>
                            <select className={styles.select} value={uncertModel} onChange={(e) => setUncertModel(e.target.value)}>
                                <option value="distilgpt2">distilgpt2 (fast)</option>
                                <option value="gpt2">gpt2</option>
                            </select>
                        </div>
                        <div className={styles.field}>
                            <label className={styles.label}>MC samples</label>
                            <input className={styles.number} inputMode="numeric" value={uncertSamples}
                                onChange={(e) => setUncertSamples(parseInt(e.target.value || '1'))} />
                        </div>
                        <div className={styles.field}>
                            <label className={styles.label}>IG steps</label>
                            <input className={styles.number} inputMode="numeric" value={uncertSteps}
                                onChange={(e) => setUncertSteps(parseInt(e.target.value || '1'))} />
                        </div>
                        <div className={styles.help}>MC samples 越大越稳定但更慢；IG steps 越大越平滑但更慢。颜色深度基于 token 的影响力均值与不确定性标准差的最大值。</div>
                    </div>
                </div>
            </div>
            <div className={styles.row}>
                <div className={styles.left}>
                    <div className={styles.colHeader}>
                        <span className={styles.colTitle}>PII highlight</span>
                        <div className={styles.legend}>
                            <span>Type Color</span>
                            <span className={`${styles.pii} ${styles['type-EMAIL_ADDRESS']}`} data-entity="Entity 0.99(Confidence)">Sample</span>
                        </div>
                    </div>
                    <div className={styles.textBox}>
                        {isAnalyzingPII ? (
                            <div className={styles.loadingPlaceholder}>
                                <span className={styles.loading}>Analyzing…</span>
                                <div className={styles.loadingLine} style={{ width: '85%' }}></div>
                                <div className={styles.loadingLine} style={{ width: '70%' }}></div>
                                <div className={styles.loadingLine} style={{ width: '90%' }}></div>
                            </div>
                        ) : text ? (
                            <div className={styles.rich} dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
                        ) : (
                            'No text yet'
                        )}
                    </div>
                </div>
                <div className={styles.middle}>
                    <div className={styles.colHeader_uncert}>
                        <span className={styles.colTitle}>Uncertainty heatmap</span>
                        <div className={styles.legend}>
                            <span>Low</span>
                            <div className={styles.legendBar}></div>
                            <span>High</span>
                        </div>
                    </div>
                    <div className={styles.textBox}>
                        {isAnalyzingUncert ? (
                            <div className={styles.loadingPlaceholder}>
                                <span className={styles.loading}>Computing…</span>
                                <div className={styles.loadingLine} style={{ width: '90%' }}></div>
                                <div className={styles.loadingLine} style={{ width: '60%' }}></div>
                                <div className={styles.loadingLine} style={{ width: '75%' }}></div>
                                <div className={styles.loadingLine} style={{ width: '80%' }}></div>
                            </div>
                        ) : text ? (
                            <div className={styles.rich} dangerouslySetInnerHTML={{ __html: uncertHtml }} />
                        ) : (
                            'No text yet'
                        )}
                    </div>
                </div>
                <div className={styles.right}>
                    <div className={styles.inputTitle}>Input</div>
                    <textarea
                        className={styles.textarea}
                        ref={inputRef}
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Type something..."
                    />
                    <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                        <button type="button" className={styles.button} onClick={runAnalyzeAll} disabled={isAnalyzingPII || isAnalyzingUncert || !text}>Analyze</button>
                    </div>
                </div>
            </div>
            <div className={styles.controls}>
                {!!errorMessage && <span className={styles.error} title={errorMessage}>Analyzer unavailable</span>}
            </div>
        </div>
    )
}