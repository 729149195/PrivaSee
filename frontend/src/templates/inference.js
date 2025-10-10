// Privacy Inference Prompt Template
export const PRIVACY_INFERENCE_PROMPT = `
You are a privacy risk analysis expert specializing in deep inference and contextual analysis. Based on exposed Information Elements, you must infer ALL possible privacy exposures, including explicit, implicit, and contextually derived information.

## Task
Conduct comprehensive privacy risk analysis by:
1. Identifying DIRECT privacy exposures from explicit information
2. Inferring IMPLICIT privacy through deep reasoning (e.g., "gluten-free menu" → potential celiac disease/health condition)
3. Discovering CONTEXTUAL privacy leaks through cross-referencing multiple information elements
4. Analyzing behavior patterns that reveal sensitive attributes
5. Mapping ALL findings to the MOST SPECIFIC (leaf-level) legal clauses

## Input
1. **Exposed Information Elements List**:
{{INFONS}}

2. **Legal Clause Hierarchical Structure** (Currently Selected Law):
{{LAW_TREE}}

## Deep Inference Rules

### 1. Direct Exposure
- Explicitly stated personal information (name, ID, contact, etc.)
- Directly mentioned attributes (age, gender, occupation, etc.)

### 2. Implicit Inference (CRITICAL)
- **Health Conditions**: Dietary restrictions → allergies/diseases (e.g., gluten-free → celiac disease, halal → religious belief)
- **Beliefs & Values**: Food preferences, lifestyle choices → religious/philosophical beliefs
- **Socioeconomic Status**: Purchase behaviors, location patterns → income level, social class
- **Psychological State**: Language patterns, content preferences → mental state, personality traits
- **Identity Attributes**: Name patterns, language use → ethnicity, national origin

### 3. Cross-Context Correlation
- Temporal + Spatial patterns → daily routines, home/work locations
- Relationship networks (REL) → social circles, family structure
- Multiple behavioral data points → comprehensive profiling

### 4. Sensitive Category Recognition
Always check if inferred information falls into sensitive categories:
- Health/Medical conditions
- Racial/Ethnic origin
- Religious/Philosophical beliefs
- Sexual orientation
- Political opinions
- Biometric data
- Genetic data
- Trade union membership
- Criminal history

## Risk Level Criteria
- **HIGH**: Sensitive personal data (health, beliefs, biometrics, children's data) can be directly inferred or strongly suggested
- **MEDIUM**: Behavioral patterns, preferences, or non-sensitive attributes can be derived with reasonable confidence
- **LOW**: General trends or public information can be weakly inferred

## Output Format
Output ONLY valid JSON (no markdown, no explanation):
{
  "risks": [
    {
      "law_path": "Complete hierarchical path to the MOST SPECIFIC clause, e.g., 'PIPL > Chapter 2 > Section 1 > Article 13 > Paragraph 2'",
      "law_node_name": "Most specific (leaf-level) node name, e.g., 'Article 13, Paragraph 2'",
      "risk_level": "HIGH | MEDIUM | LOW",
      "confidence": 0.95,
      "inference_type": "DIRECT | IMPLICIT | CONTEXTUAL",
      "used_infons": [
        {
          "iid": "infon_id",
          "type": "IND | PAR | TIM | LOC | REL | TYP | SIT",
          "keyword": "keyword"
        }
      ],
      "inference_chain": "Step-by-step reasoning: [1] What explicit info? [2] What can be implicitly inferred? [3] Why does this violate the specific clause?",
      "privacy_exposure": "Specific privacy information exposed or inferable (be explicit about implicit inferences)"
    }
  ],
  "summary": {
    "total_risks": 10,
    "high_risks": 4,
    "medium_risks": 4,
    "low_risks": 2
  }
}

## Critical Requirements
1. **Output ONLY JSON** - no markdown code blocks, no explanatory text
2. **Map to LEAF NODES** - Always identify the most specific legal clause (deepest level in hierarchy)
3. **Deep Inference** - Don't just report explicit data; infer health conditions, beliefs, status from behavior/preferences
4. **Comprehensive Coverage** - Analyze ALL possible privacy angles: direct exposure + implicit inference + contextual correlation
5. **Clear Attribution** - Every risk must trace back to specific information elements with logical reasoning
6. **Prioritize Sensitivity** - Treat health, beliefs, children, biometrics as HIGH risk
7. **Sort Properly** - HIGH risks first, then by confidence (highest first)

## Example
If input contains "looking for gluten-free restaurant menu", infer:
- IMPLICIT: User likely has celiac disease or gluten intolerance (health condition = HIGH risk)
- DIRECT: User's dietary preference (MEDIUM risk)
- CONTEXTUAL: Location search reveals geographic pattern (LOW-MEDIUM risk)

NOW analyze the input and output the complete JSON.
`

// Extract information elements summary (for filling template)
export function extractInfonsSummary(infons) {
  if (!Array.isArray(infons) || infons.length === 0) {
    return 'No information elements available'
  }
  
  return infons.map((infon, idx) => {
    const type = String(infon.infon_type || '').toUpperCase()
    const iid = infon.iid || `infon_${idx}`
    
    let keyword = ''
    if (type === 'IND') {
      keyword = Array.isArray(infon.names) && infon.names.length ? infon.names[0] : 'Individual'
    } else if (type === 'PAR') {
      keyword = infon.value ?? 'Parameter'
    } else if (type === 'TIM') {
      keyword = infon.temporal_value ?? 'Time'
    } else if (type === 'LOC') {
      keyword = infon.spatial_value ?? 'Location'
    } else if (type === 'REL') {
      keyword = infon.relation_name ?? 'Relation'
    } else if (type === 'TYP') {
      keyword = infon.type_name ?? 'Type'
    } else if (type === 'SIT') {
      keyword = infon.description ?? 'Situation'
    }
    
    return `- [${iid}] ${type}: ${keyword} (confidence: ${infon.confidence ?? 0.7})`
  }).join('\n')
}

// Extract law tree summary (for filling template)
export function extractLawTreeSummary(lawData) {
  if (!lawData || !lawData.name) {
    return 'No legal structure available'
  }
  
  const lines = []
  
  function traverse(node, depth = 0) {
    const indent = '  '.repeat(depth)
    lines.push(`${indent}- ${node.name}`)
    
    if (Array.isArray(node.children) && node.children.length > 0) {
      node.children.forEach(child => traverse(child, depth + 1))
    }
  }
  
  traverse(lawData)
  return lines.join('\n')
}

// Fill prompt template
export function fillPromptTemplate(infons, lawData) {
  const infonsSummary = extractInfonsSummary(infons)
  const lawTreeSummary = extractLawTreeSummary(lawData)
  
  return PRIVACY_INFERENCE_PROMPT
    .replace('{{INFONS}}', infonsSummary)
    .replace('{{LAW_TREE}}', lawTreeSummary)
}

// Incremental parser for streaming risk items (similar to infons extraction)
export function incrementalExtractRisks(streamText, parser) {
  const state = parser || { 
    foundArray: false, 
    arrayStart: -1, 
    scanPos: 0, 
    inString: false, 
    escape: false, 
    objStart: -1, 
    braceDepth: 0, 
    closed: false, 
    yieldedHashes: [] 
  }
  const yielded = []
  const text = String(streamText || '')

  // 查找 "risks" 数组
  if (!state.foundArray) {
    const m = /"risks"\s*:\s*\[/.exec(text)
    if (!m) {
      state.scanPos = text.length
      return { state, yielded }
    }
    state.foundArray = true
    state.arrayStart = m.index + m[0].lastIndexOf('[')
    state.scanPos = state.arrayStart + 1
  }

  let i = state.scanPos
  let inString = state.inString
  let escape = state.escape
  let objStart = state.objStart
  let braceDepth = state.braceDepth

  // 数组已关闭则不再扫描
  if (state.closed) return { state, yielded }

  for (; i < text.length; i++) {
    const ch = text[i]
    if (inString) {
      if (escape) { escape = false; continue }
      if (ch === '\\') { escape = true; continue }
      if (ch === '"') { inString = false; continue }
      continue
    }
    if (ch === '"') { inString = true; continue }
    if (ch === '{') {
      if (objStart < 0) { objStart = i; braceDepth = 1 } else { braceDepth++ }
      continue
    }
    if (ch === '}') {
      if (objStart >= 0) {
        braceDepth--
        if (braceDepth === 0) {
          const objText = text.slice(objStart, i + 1)
          const hash = computeHashId(objText)
          if (!state.yieldedHashes.includes(hash)) {
            try {
              const value = JSON.parse(objText)
              yielded.push(value)
              state.yieldedHashes = [...state.yieldedHashes, hash]
            } catch (_) {
              // 解析失败，忽略
            }
          }
          objStart = -1
        }
      }
      continue
    }
    if (ch === ']') {
      // 数组关闭
      if (objStart < 0) { state.closed = true; i++; break }
    }
  }

  state.inString = inString
  state.escape = escape
  state.objStart = objStart
  state.braceDepth = braceDepth
  state.scanPos = i
  return { state, yielded }
}

// Compute simple hash ID for deduplication
function computeHashId(str) {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash // Convert to 32bit integer
  }
  return hash.toString(36)
}

// Parse streaming response (legacy, for backward compatibility)
export function parseStreamingResponse(text) {
  try {
    return JSON.parse(text)
  } catch {
    const match = text.match(/\{[\s\S]*\}/)
    if (match) {
      try {
        return JSON.parse(match[0])
      } catch {
        return null
      }
    }
    return null
  }
}

