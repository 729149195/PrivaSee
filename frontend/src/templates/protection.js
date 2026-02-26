// Privacy Protection Suggestions — optimized for 4B models
import { buildCurrentTimeInstruction } from './timeContext.js'

// fillProtectionPrompt generates the full prompt (no separate template needed)
export function fillProtectionPrompt(originalText, privacyRisks, infons) {
  // Format detected risks
  let risksText = 'No privacy risks detected'
  if (Array.isArray(privacyRisks) && privacyRisks.length > 0) {
    risksText = privacyRisks.map((risk, idx) => {
      const level = risk.risk_level || '?'
      const name = risk.law_node_name || '?'
      const desc = risk.inference_chain || risk.reason || risk.privacy_exposure || ''
      return `${idx + 1}. [${level}] ${name}${desc ? ' — ' + desc : ''}`
    }).join('\n')
  }

  // Format infons as context
  let infonsSection = ''
  if (Array.isArray(infons) && infons.length > 0) {
    const items = infons.map(inf => {
      const t = String(inf.infon_type || '').toUpperCase()
      if (t === 'DESC') return `${inf.entity || ''}:${inf.attribute || ''}`
      if (t === 'SCEN') return `${inf.temporal || ''}@${inf.spatial || ''}`
      if (t === 'REL') return inf.relation_name || ''
      return null
    }).filter(Boolean)
    if (items.length) infonsSection = `\nExtracted data points: ${items.join(', ')}`
  }

  return `${buildCurrentTimeInstruction()}

Rewrite the user's text to protect privacy. Provide 3 versions with different protection levels.

## Original Text
${originalText || ''}

## Detected Privacy Risks
${risksText}${infonsSection}

## Output: exactly 3 lines, one per level
Format per line: level,label,modified_text,changes_summary,removed_risks

Fields:
- level: high_privacy | balanced | low_privacy
- label: short label in the SAME LANGUAGE as original text
- modified_text: full rewritten text (escape commas as \\,)
- changes_summary: brief description of what changed (same language as original)
- removed_risks: items removed, separated by |

## Protection strategies
- high_privacy: remove ALL identifiable info, generalize everything, maximize anonymity
- balanced: keep core intent, remove direct identifiers, moderate generalization
- low_privacy: minimal changes, only remove the most sensitive items (ID numbers, full addresses)

## Rules
- Keep the SAME LANGUAGE as the original text
- Each version must be a complete, usable text (not a template)
- Output exactly 3 lines, no headers, no markdown, no explanation
- high_privacy first, then balanced, then low_privacy

Output:`
}

// ============================================================================
// COMPACT FORMAT PARSERS
// ============================================================================

function unescapeValue(value) {
  if (typeof value !== 'string') return value
  return value.replace(/\\,/g, ',').replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
}

function splitArrayField(value) {
  if (!value || typeof value !== 'string') return []
  return value.split(/(?<!\\)\|/).map(p => p.trim()).filter(Boolean)
}

const VALID_LEVELS = new Set(['high_privacy', 'balanced', 'low_privacy'])
const SUGGESTION_FIELDS = ['level', 'label', 'modified_text', 'changes_summary', 'removed_risks']

// 按逗号分割（尊重转义）
function splitEscapedComma(line) {
  const values = []
  let cur = '', escaped = false
  for (const ch of line) {
    if (escaped) { cur += ch; escaped = false; continue }
    if (ch === '\\') { cur += ch; escaped = true; continue }
    if (ch === ',') { values.push(cur); cur = ''; continue }
    cur += ch
  }
  if (cur || values.length > 0) values.push(cur)
  return values
}

// 解析一行紧凑格式为建议对象
function parseCompactSuggestionLine(line) {
  if (!line?.trim()) return null
  const trimmed = line.trim()
  // 必须以有效 level 开头
  if (!VALID_LEVELS.has(trimmed.split(',')[0]?.trim())) return null

  const values = splitEscapedComma(trimmed)
  const suggestion = {}
  for (let i = 0; i < SUGGESTION_FIELDS.length && i < values.length; i++) {
    const field = SUGGESTION_FIELDS[i]
    const val = values[i].trim()
    suggestion[field] = field === 'removed_risks' ? splitArrayField(val) : unescapeValue(val)
  }
  return suggestion.level ? suggestion : null
}

// 完整解析紧凑格式文本
export function parseCompactProtectionFormat(text) {
  if (!text || typeof text !== 'string') return null
  const clean = text.replace(/```[\w]*\n?/g, '').replace(/\n?```$/g, '').trim()
  const suggestions = []
  for (const line of clean.split('\n')) {
    const s = parseCompactSuggestionLine(line.trim())
    if (s) suggestions.push(s)
  }
  return suggestions.length > 0 ? suggestions : null
}

// 解析保护建议响应（紧凑格式优先，JSON 后备）
export function parseProtectionResponse(responseText) {
  const compact = parseCompactProtectionFormat(responseText)
  if (compact) return compact
  try {
    const parsed = JSON.parse(responseText)
    if (parsed?.suggestions && Array.isArray(parsed.suggestions)) return parsed
  } catch (_) {
    const m = responseText.match(/\{[\s\S]*\}/)
    if (m) try { const p = JSON.parse(m[0]); if (p?.suggestions) return p } catch (_) {}
  }
  return null
}

// 验证建议数据完整性
export function validateSuggestion(suggestion) {
  if (!suggestion || typeof suggestion !== 'object') return false
  return ['level', 'label', 'modified_text'].every(f => suggestion[f])
}

// 流式增量解析保护建议
export function incrementalExtractSuggestions(streamText, parser) {
  const text = String(streamText || '').replace(/```[\w]*\n?/g, '').replace(/\n?```$/g, '')

  const state = {
    parsedLines: 0,
    suggestionIndex: 0,
    lastPartialHash: null,
    formatDetected: true,
    ...(parser || {})
  }

  const yielded = []
  const lines = text.split('\n')

  // parsedLines 防溢出（<think> 移除后 buffer 可能变短）
  if (state.parsedLines > lines.length) state.parsedLines = 0

  for (let i = state.parsedLines; i < lines.length; i++) {
    const trimmed = lines[i].trim()
    const isLast = i === lines.length - 1
    const isStreaming = isLast && !text.endsWith('\n')

    if (!trimmed || trimmed.startsWith('```') || trimmed === '---') {
      if (!isStreaming) state.parsedLines++
      continue
    }

    if (isStreaming) {
      // 部分解析：内容变化时才更新
      const hash = simpleHash(trimmed)
      if (hash !== state.lastPartialHash) {
        state.lastPartialHash = hash
        const partial = parseCompactSuggestionLine(trimmed)
        if (partial) {
          partial._objIndex = state.suggestionIndex
          partial._isComplete = false
          yielded.push(partial)
        }
      }
    } else {
      // 完整行解析
      const s = parseCompactSuggestionLine(trimmed)
      if (s) {
        s._objIndex = state.suggestionIndex++
        s._isComplete = true
        yielded.push(s)
        state.lastPartialHash = null
      }
      state.parsedLines++
    }
  }

  return { state, yielded }
}

function simpleHash(str) {
  let h = 0
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h) + str.charCodeAt(i)
    h = h & h
  }
  return h.toString(36)
}
