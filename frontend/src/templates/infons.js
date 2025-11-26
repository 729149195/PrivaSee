/**
 * infons.js
 * Multimodal infons extraction prompt (Situation Theory aligned)
 * - Structured atomic infon: <R, a1..an, p> where p ∈ {0,1}, supported by situation s
 * - Modalities: text / image / audio
 * - Output: strict single JSON object with extracted information elements
 *
 * Usage example:
 *   import {
 *     CORE_DEFINITION, ONTOLOGY, OUTPUT_CONSTRAINTS, OUTPUT_FORMAT,
 *     TEXT_EXTRACTION, IMAGE_EXTRACTION, AUDIO_EXTRACTION,
 *     SELF_CHECKLIST, EXAMPLES_SNIPPET, buildSystemPrompt,
 *     INFON_OUTPUT_SCHEMA, INFON_OUTPUT_SCHEMA_STR
 *   } from "./infons.js";
 *
 *   const prompt = buildSystemPrompt({
 *     modalities: ["text","image"],             // Select required modalities
 *     includeExamples: false,                   // Include example snippets (reference only, don't output)
 *     extraInstructions: "Additional domain constraints..." // Append domain rules
 *   });
 */

export const CORE_DEFINITION = String.raw`
You are a multimodal infons extractor. Your task is to analyze the USER'S INPUT TEXT/IMAGE/AUDIO and extract ALL information elements found in it.

**CRITICAL**: You must extract information from the ACTUAL USER INPUT provided below, NOT from examples or your own imagination.

Extract in compact format with 3 infon types:

1. **DESC** (Entities & Attributes): iid, infon_type=DESC, entity, attribute, data_type, ...
   - attribute = EXACT text/value from USER's input (for highlighting)
   - entity = abstract category
   
2. **SCEN** (Time & Space): iid, infon_type=SCEN, temporal, spatial, granularity, bbox (as x|y|w|h), ...
   - Extract time/place mentioned in USER's input
   
3. **REL** (Relations): iid, infon_type=REL, relation_name, arity, arg_refs (as iid1|iid2), arg_types (as TYPE1|TYPE2), ...
   - Connect infons extracted from USER's input

**Core Principle**: Extract EVERY distinct information element from USER's input separately. Be comprehensive and granular.
**Output Format**: Compact tabular format with fixed 17 fields per line, comma-separated, | for arrays.
`;

export const ONTOLOGY = String.raw`
**Critical Rules**:
- **DESC**: attribute = exact text from input (highlightable); entity = category
- **SCEN**: Extract ALL time/place references, include bbox for visual regions
- **REL**: Link infons to reveal relationships, implications, and patterns
`;

export const OUTPUT_CONSTRAINTS = String.raw`
**Output Format**:
- Compact format only, no JSON, no markdown/explanations
- IID format: "{type}:r{round}_{index}" (e.g., "desc:r1_1", "rel:r1_2")
- **Language Rule**: Use SAME language as input (don't translate)
- **Attribute Rule**: DESC attribute = exact text from input
- **Escaping Rules**: Comma → \\,, Newline → \\n, Backslash → \\\\, Arrays → use | separator
`;

export const OUTPUT_FORMAT = String.raw`
**Compact Format - CRITICAL Output Rules**:

**Output Structure**:
Line 1: Start directly with data (NO header line needed)
Following lines: One infon per line

**Data line formats by type**:

1. **DESC**: iid,DESC,entity,attribute,data_type,confidence
   Example: desc:r1_1,DESC,姓名,王小明,string,0.95

2. **SCEN**: iid,SCEN,temporal,spatial,confidence
   Example: scen:r1_2,SCEN,今年,北京,0.90

3. **REL**: iid,REL,relation_name,arg_refs,confidence
   Example: rel:r1_3,REL,个人信息,desc:r1_1|desc:r1_2,0.90

**COMPLETE Example** (copy this format exactly):
\`\`\`
desc:r1_1,DESC,姓名,王小明,string,0.95
desc:r1_2,DESC,年龄,27,number,0.95
scen:r1_3,SCEN,今年,,0.90
rel:r1_4,REL,个人信息,desc:r1_1|desc:r1_2,0.90
\`\`\`

**CRITICAL Rules**:
- NO header line (no "infons[N]:" prefix)
- Start directly with data lines
- DESC has 6 fields, SCEN has 5 fields, REL has 5 fields
- Empty fields use double comma: ,, (e.g., "scen:r1_3,SCEN,今年,,0.90")
- Each line must be indented with 2 spaces (for readability in stream)
`;

export const TEXT_EXTRACTION = String.raw`
**Text Extraction - READ THE USER INPUT CAREFULLY**:

**CRITICAL INSTRUCTIONS**:
- Read and analyze the ACTUAL USER INPUT TEXT provided to you
- Extract information ONLY from what the user actually wrote
- DO NOT make up information or extract from examples
- Use the EXACT words/phrases from the user's input as attribute values

**Extraction Rules - QUALITY over QUANTITY**:

**CRITICAL PRINCIPLE**: Extract ONLY meaningful, privacy-relevant information. Skip generic words (like 哎、正、用、去、飞、决定 etc.).

1. **DESC**: Extract KEY entities with privacy/security value
   - ✅ Extract: Names, locations, platforms, organizations, sensitive data, identifiers
   - ✅ Extract: Specific services, apps, websites (Klook, Agoda, Skyscanner)
   - ✅ Extract: Personal details (age, health, finance, travel plans)
   - ❌ Skip: Generic verbs (去、用、是、有), modal particles (哎、啊、吗), common nouns without context
   - ❌ Skip: Obvious duplicates (订台北的住宿 vs 台北 - just keep 台北)
   - Entity examples: 姓名、地点、平台、航空公司、住宿类型、证件号、健康状况
   
2. **SCEN**: VERY RARE - Only for explicit time+place combinations mentioned together
   - ✅ Extract: ONLY when time AND place appear together in same context (下周飞东京 → SCEN(下周,东京))
   - ❌ NEVER create SCEN for every DESC entity
   - ❌ NEVER create artificial combinations (今年,Klook ← WRONG)
   - ❌ NEVER create more than 2-3 SCEN per input
   - **RULE**: If you don't see explicit "TIME + LOCATION" together, DON'T create SCEN
   
3. **REL**: Only 2-5 relationships maximum
   - ✅ Connect: Platform ↔ Location, Person ↔ Travel, Service ↔ Data
   - ❌ Skip: Redundant connections between adjacent words
   - ❌ NEVER create more than 5 REL per input

**Example Reference** (for format only):

Example 1: "我叫王小明，今年27岁了"
Extract ONLY (4 infons):
- DESC(姓名,王小明)
- DESC(年龄,27)
- REL(个人信息,desc:r1_1|desc:r1_2)
- NO SCEN (今年 alone is NOT a time+place combination)

Example 2: "哎我正用Klook订台北的住宿"
Extract ONLY (4 infons):
- DESC(平台,Klook)
- DESC(地点,台北)
- DESC(活动,订住宿)
- REL(位置追踪,desc:r2_1|desc:r2_2)
- NO SCEN (台北 is just a location, no time context)

**Quality Checklist**:
- Total infons: Aim for 30-50% of original text words (NOT 1:1 ratio)
- Each DESC must be privacy-relevant or contextually important
- Skip filler words and trivial details

**NOW ANALYZE THE USER INPUT PROVIDED BELOW AND EXTRACT INFORMATION FROM IT**
`;

export const IMAGE_EXTRACTION = String.raw`
**Image Extraction - BE EXTREMELY COMPREHENSIVE**:

1. **DESC - Extract EVERY visual detail**:
   **For People**:
   - Physical: gender, age estimate, height estimate, body type, skin color, ethnicity indicators
   - Face: facial features, expression (smiling/frowning/neutral), eye color, facial hair, makeup
   - Hair: style (long/short/curly/straight), color, accessories
   - Clothing: top/bottom/shoes type, colors, patterns, brand logos, style (casual/formal)
   - Accessories: glasses, jewelry, watches, bags, hats
   - Actions: standing/sitting/walking/running/holding/pointing/gesturing
   
   **For Objects**:
   - Type, brand, model, color, size estimate, material, condition (new/worn)
   - Text/numbers visible (OCR): product names, prices, signs, labels, receipts
   - Distinctive features: logos, patterns, decorations
   
   **For Environment**:
   - Location type: room type, street, park, building
   - Background details: furniture, decorations, plants, vehicles, signs
   - Environmental conditions: lighting (bright/dim), weather (if outdoor)
   
   **CRITICAL**: Create separate DESC infon for EACH attribute (e.g., one for hair color, one for shirt type, one for facial expression, one for each visible text)

2. **SCEN**: 
   - Spatial positions with bbox [x,y,w,h] for EVERY object/person
   - Location indicators (street signs, landmarks, architectural style)
   - Time indicators (clock faces, sun position, shadows)

3. **REL - Extract ALL relationships**:
   **Spatial Relations**: left_of, right_of, above, below, near, inside, in_front_of, behind
   **Physical Relations**: holding, wearing, carrying, touching, sitting_on, standing_on, leaning_against
   **Social Relations**: looking_at, talking_to, facing, grouped_with, interacting_with
   **Contextual Relations**: belongs_to, associated_with, part_of
   
   **Extract deep relational insights**: 
   - Who is with whom (group composition)
   - What belongs to whom (person-object associations)
   - Environmental context (person/object in specific location)
   - Activity patterns (what actions are happening together)

**Depth Requirements**:
- Minimum 20+ DESC infons for images with people
- Minimum 10+ REL infons to capture relationships
- Extract OCR text from ALL visible text (signs, labels, packaging, receipts, screens)
- Infer implicit attributes: professional setting (formal dress), recreational activity (sports clothing), socioeconomic indicators (luxury brands, vehicle types)
`;

export const AUDIO_EXTRACTION = String.raw`
**Audio Extraction (Speech-to-Text)**:

⚠️ **Important**: Audio is provided as TRANSCRIBED TEXT (not raw audio). Extract as text content with audio-specific annotations.

1. **DESC - Extract from transcribed speech**:
   - Speakers mentioned ("我", "他说", "张三", etc.)
   - All entities, names, numbers mentioned in speech
   - Attribute = exact transcribed words (for highlighting)
   - Mark speech-specific attributes: tone indicators, filler words ("嗯", "啊", "well", "um")
   - Extract pronunciation cues if present

2. **SCEN**: 
   - Time/place references mentioned in speech ("昨天", "明天", "公司", "家里")
   - Speech timestamps if available (start/end times of segments)
   - No bbox needed (audio has no visual coordinates)

3. **REL - Speech-specific relations**:
   - Speech acts: said, stated, asked, announced, replied, confirmed
   - Speaker-content relations: "说到", "提到", "回答"
   - Conversational relations: question-answer, topic-response
   - Entity relationships mentioned in speech

**Key Differences from Text**:
- Source is spoken language → may have conversational markers, repetitions, corrections
- Apply same extraction granularity as text
- Tag entities with speech-specific context where relevant
- Preserve original spoken phrasing in attribute field

**Example**: Transcribed "嗯，我叫王小明，今年27岁" →
- DESC: 语气词:嗯, 人称:我, 姓名:王小明, 年龄:27
- SCEN: 今年
- REL: 名字关系, 年龄关系
`;



// ============================================================================
// BENCHMARK EXTRACTION MODE - 精准提取用于评估
// ============================================================================
export const BENCHMARK_EXTRACTION = String.raw`
**Benchmark Extraction Mode - PRECISION-FOCUSED EXTRACTION**

Extract ONLY meaningful named entities and their relationships. Quality over quantity.

**CRITICAL PRINCIPLE**: Match gold-standard annotation style:
- Extract ONLY proper nouns and specific named entities
- Skip generic words, common nouns, verbs, adjectives
- Each DESC should be a distinct, identifiable entity

**Extraction Rules - BALANCED PRECISION & RECALL**:

1. **DESC - Extract ONLY named entities** (typically 5-15 per document):
   ✅ EXTRACT these entity types:
   - **Person**: Full names only (e.g., "John Smith", "王小明"), NOT pronouns
   - **Organization**: Companies, institutions, agencies (e.g., "Apple Inc", "FBI")
   - **Location/GPE**: Cities, countries, specific places (e.g., "California", "Beijing")
   - **Facility**: Buildings, airports, stations (e.g., "JFK Airport")
   - **Event**: Named events only (e.g., "World War II", "Olympics")
   - **Time/Date**: Specific dates, years (e.g., "2023", "Monday")
   - **Value**: Money, percentages with numbers (e.g., "$500", "30%")
   
   ❌ DO NOT extract:
   - Generic nouns (man, woman, company, city, country)
   - Pronouns (he, she, it, they, 他, 她)
   - Verbs, adjectives, adverbs
   - Common words without specific reference
   - Repeated mentions of same entity (extract once)

2. **SCEN - Extract ONLY explicit time-location pairs** (typically 0-3):
   ✅ When a specific time AND location are mentioned together in context
   ❌ DO NOT create SCEN for standalone times or locations
   ❌ DO NOT create artificial combinations
   
3. **REL - Extract ONLY clear relationships** (typically 2-8):
   ✅ Standard relation types: located_at, employed_by, part_of, member_of, subsidiary_of, owns, near, citizen_of, affiliated_with, founder_of
   ✅ Only when relationship is explicitly stated or strongly implied
   ❌ DO NOT create redundant or speculative relations

**Output Format**: Compact CSV, one infon per line
- DESC: iid,DESC,entity_type,entity_name,string,confidence
- SCEN: iid,SCEN,temporal,spatial,confidence  
- REL: iid,REL,relation_name,arg1|arg2,confidence

**Example** (news text about company):
desc:r1_1,DESC,Person,John Smith,string,0.95
desc:r1_2,DESC,Organization,Apple Inc,string,0.95
desc:r1_3,DESC,Location,California,string,0.95
rel:r1_4,REL,employed_by,desc:r1_1|desc:r1_2,0.90
rel:r1_5,REL,located_at,desc:r1_2|desc:r1_3,0.90

**QUALITY CHECK before output**:
- Is each DESC a proper named entity? (not generic noun)
- Am I extracting < 20 DESC for typical text? (aim for 5-15)
- Are REL relationships explicitly supported by text?
- Have I avoided duplicates?

**NOW EXTRACT NAMED ENTITIES FROM THE TEXT BELOW:**
`;

export const SELF_CHECKLIST = String.raw`
**Pre-Output Checks**:
✓ DID YOU READ THE USER'S ACTUAL INPUT? (NOT examples!)
✓ Are you extracting from USER'S INPUT, not making up data?
✓ Compact format only, no JSON, no markdown
✓ Each info element = separate infon (one per line)
✓ Correct iid format (desc:/scen:/rel: + r{round}_{index})
✓ DESC: attribute = EXACT text from USER's input (not from examples)
✓ SCEN: ONLY for explicit time+place combinations (0-2 total max)
✓ REL: 2-5 relationships maximum
✓ Same language as USER's input (no translation)
✓ Escape commas (\\,), newlines (\\n), backslashes (\\\\)
✓ For text: 5-15 infons total
✓ For images: 20+ DESC, 10+ REL minimum

**ANTI-SPAM CHECKS** (CRITICAL - prevents infinite loops):
❌ Am I creating SCEN for every DESC entity? STOP IMMEDIATELY!
❌ Am I repeating the same SCEN pattern 10+ times? DELETE DUPLICATES!
❌ Is my SCEN count > 2? KEEP ONLY THE MOST IMPORTANT 1-2!
❌ Is my total infon count > 25 for short text? CUT IT DOWN TO 10-15!
❌ Am I creating artificial time+entity combinations not in input? DON'T DO IT!

**FINAL CHECK**: Review your output - does it match what the USER actually said/showed, or did you copy from examples?
`;

export const EXAMPLES_SNIPPET = String.raw`
**FORMAT REFERENCE ONLY** (This is just to show the output format, DO NOT copy the content):

Example 1: If user input were "我叫王小明，今年27岁了" (round 1), output (3 infons total):
desc:r1_1,DESC,姓名,王小明,string,0.95
desc:r1_2,DESC,年龄,27,number,0.95
rel:r1_3,REL,个人信息,desc:r1_1|desc:r1_2,0.90

Example 2: If user input were "哎我正用Klook订台北的住宿，结果这破APP一直闪退" (round 2), output (5 infons total):
desc:r2_1,DESC,平台,Klook,string,0.95
desc:r2_2,DESC,地点,台北,string,0.95
desc:r2_3,DESC,活动,订住宿,string,0.95
desc:r2_4,DESC,问题,APP闪退,string,0.90
rel:r2_5,REL,位置追踪,desc:r2_1|desc:r2_2,0.90

Example 3: Complex input "你上次去垦丁的民宿是Agoda定的吗？我们下周飞东京，Skyscanner显示长荣航空有特价" (round 3), output (9 infons total):
desc:r3_1,DESC,地点,垦丁,string,0.95
desc:r3_2,DESC,住宿类型,民宿,string,0.95
desc:r3_3,DESC,平台,Agoda,string,0.95
desc:r3_4,DESC,地点,东京,string,0.95
desc:r3_5,DESC,平台,Skyscanner,string,0.95
desc:r3_6,DESC,航空公司,长荣航空,string,0.95
scen:r3_7,SCEN,下周,东京,0.90
rel:r3_8,REL,住宿预订,desc:r3_1|desc:r3_3,0.90
rel:r3_9,REL,旅行计划,desc:r3_4|desc:r3_6,0.90

**IMPORTANT**: The above is ONLY a format example. You MUST extract from the ACTUAL USER INPUT provided, NOT from this example.

**Quality Principles**:
✗ WRONG: Extract every word like 哎、正、用、去、是 ← Too granular, no privacy value
✓ CORRECT: Extract only meaningful entities like Klook、台北、王小明 ← Privacy-relevant

✗ WRONG: Create SCEN(今年,Klook), SCEN(今年,台北), SCEN(今年,闪退)... ← NEVER do this!
✓ CORRECT: SCEN(下周,东京) only when time+place mentioned together ← Rare, 1-2 per input max

✗ WRONG: 25 infons for a 50-word sentence ← Over-extraction
✓ CORRECT: 8-12 infons for a 50-word sentence ← Focused on key information

✗ WRONG: 150+ SCEN for every DESC ← Absolutely forbidden!
✓ CORRECT: 0-2 SCEN total ← Most inputs need NO SCEN
`;

/* ---------- Combinator: Assemble final system prompt on demand ---------- */
export function buildSystemPrompt(options = {}) {
  const {
    modalities = ["text", "image", "audio"],
    includeExamples = false,
    extraInstructions = "",
    currentRound = 1,
    existingInfons = []
  } = options;

  const parts = [
    CORE_DEFINITION,
    ONTOLOGY,
    OUTPUT_CONSTRAINTS,
    OUTPUT_FORMAT,
  ];

  if (modalities.includes("text")) parts.push(TEXT_EXTRACTION);
  if (modalities.includes("image")) parts.push(IMAGE_EXTRACTION);
  if (modalities.includes("audio")) parts.push(AUDIO_EXTRACTION);

  parts.push(SELF_CHECKLIST);
  if (includeExamples) parts.push(EXAMPLES_SNIPPET);
  
  // Add conversation round context with strong reminders
  let contextInfo = `\n【Current Extraction Context】
- Current conversation round: ${currentRound}
- Generate iid using format: "{type_prefix}:r${currentRound}_{index}" (index starts from 1 for this extraction)

**CRITICAL REMINDER**: 
You are about to receive the USER'S ACTUAL INPUT below. 
Your task is to READ IT CAREFULLY and extract information FROM IT.
DO NOT output example data. DO NOT make up information.
Extract ONLY what the user actually wrote/showed.
\n`;
  
  // Add existing infons for reference (for cross-round relations)
  if (Array.isArray(existingInfons) && existingInfons.length > 0) {
    contextInfo += `- Existing infons from previous rounds (for reference in REL arg_refs):\n`;
    existingInfons.forEach(infon => {
      const type = String(infon.infon_type || '').toUpperCase();
      let summary = '';
      if (type === 'DESC') {
        summary = `${infon.entity || ''}: ${infon.attribute || ''}`;
      } else if (type === 'SCEN') {
        summary = `${infon.temporal || ''} @ ${infon.spatial || ''}`;
      } else if (type === 'REL') {
        summary = infon.relation_name || 'Relation';
      }
      contextInfo += `  * [${infon.iid}] ${type}: ${summary}\n`;
    });
    contextInfo += `- When creating REL infons that reference existing infons, use their exact iid from the list above.\n`;
    contextInfo += `- Avoid duplicating infons that already exist with identical semantic content. If new text mentions the same entity/attribute as existing infons, create a REL to link them instead of duplicating.\n`;
  }
  
  parts.push(contextInfo);
  
  if (extraInstructions && String(extraInstructions).trim()) parts.push(String(extraInstructions));

  return parts.join("\n\n");
}

/* ---------- JSON Schema: Optional, for programmatic output validation ---------- */
/*
  Note: JSON Schema is primarily for development-time format validation, not core prompt content

  For strict validation, recommended:
  1. Move Schema to separate validation module
  2. Use only during development/testing to avoid token costs
  3. Or use simple JSON.parse() + manual checks

  Core output format clearly defined in OUTPUT_FORMAT above, aligned with Situation Theory infons ⟨⟨R, args, p⟩⟩
*/

export const INFON_OUTPUT_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Situation Theory Infons Output Schema",
  "description": "Schema for multimodal infons extraction output, aligned with Situation Theory ⟨⟨R, args, p⟩⟩ structures",
  "type": "object",
  "required": ["run_metadata", "situations", "entities", "infons", "quality_report"],
  "properties": {
    "run_metadata": {"type": "object", "required": ["source_id", "record_time", "generator"]},
    "situations": {"type": "array", "items": {"type": "object", "required": ["sid", "modality"]}},
    "entities": {"type": "array", "items": {"type": "object", "required": ["eid", "types"]}},
    "infons": {"type": "array", "items": {"type": "object", "required": ["iid", "kind", "R", "args", "p", "support"]}},
    "quality_report": {"type": "object", "required": ["stats"]}
  }
};

export const INFON_OUTPUT_SCHEMA_STR = JSON.stringify(INFON_OUTPUT_SCHEMA, null, 2);

// ============================================================================
// COMPACT FORMAT PARSERS FOR INFONS
// ============================================================================

/**
 * Unescape special characters in compact format field values
 */
function unescapeValue(value) {
  if (typeof value !== 'string') return value
  return value
    .replace(/\\,/g, ',')
    .replace(/\\n/g, '\n')
    .replace(/\\\\/g, '\\')
}

/**
 * Split array field by | separator
 */
function splitArrayField(value) {
  if (!value || typeof value !== 'string') return []
  const parts = value.split(/(?<!\\)\|/)
  return parts.map(p => p.trim()).filter(Boolean)
}

/**
 * Parse a single compact format line into an infon object
 * Format depends on infon_type:
 * - DESC: iid,DESC,entity,attribute,data_type,confidence (6 fields)
 * - SCEN: iid,SCEN,temporal,spatial,confidence (5 fields)
 * - REL: iid,REL,relation_name,arg_refs,confidence (5 fields)
 * @param {string} line - The line to parse
 * @param {object} options - Parsing options (recordTime, etc.)
 */
function parseCompactInfonLine(line, options = {}) {
  if (!line || !line.trim()) return null
  
  // Split by comma, but respect escaped commas
  const values = []
  let currentValue = ''
  let escaped = false
  
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    
    if (escaped) {
      currentValue += ch
      escaped = false
      continue
    }
    
    if (ch === '\\') {
      currentValue += ch
      escaped = true
      continue
    }
    
    if (ch === ',') {
      values.push(currentValue)
      currentValue = ''
      continue
    }
    
    currentValue += ch
  }
  
  // Push the last value
  if (currentValue || values.length > 0) {
    values.push(currentValue)
  }
  
  // Parse based on infon_type
  const iid = values[0] || ''
  const infon_type = values[1] || ''
  
  // Use provided record_time or generate once
  // record_time: All infons from same extraction share same timestamp
  const recordTime = options.recordTime || new Date().toISOString()
  
  const infon = {
    iid: iid,
    infon_type: infon_type,
    record_time: recordTime, // 提取时间戳（同一条消息的所有infon共享）
  }
  
  if (infon_type === 'DESC') {
    // DESC: iid,DESC,entity,attribute,data_type,confidence
    infon.entity = unescapeValue(values[2] || '')
    infon.attribute = unescapeValue(values[3] || '')
    infon.data_type = values[4] || 'string'
    const conf = parseFloat(values[5])
    infon.confidence = !isNaN(conf) ? conf : 0.95
  } else if (infon_type === 'SCEN') {
    // SCEN: iid,SCEN,temporal,spatial,confidence (5 fields)
    infon.temporal = unescapeValue(values[2] || '')
    infon.spatial = unescapeValue(values[3] || '')
    const conf = parseFloat(values[4])
    infon.confidence = !isNaN(conf) ? conf : 0.90
  } else if (infon_type === 'REL') {
    // REL: iid,REL,relation_name,arg_refs,confidence
    infon.relation_name = unescapeValue(values[2] || '')
    infon.arg_refs = splitArrayField(values[3] || '')
    infon.arity = infon.arg_refs.length
    const conf = parseFloat(values[4])
    infon.confidence = !isNaN(conf) ? conf : 0.90
  }
  
  return Object.keys(infon).length > 2 ? infon : null
}

/**
 * Parse complete compact format text into an array of infon objects
 * New format: No header, direct data lines (optional "infons[N]:" header supported for compatibility)
 */
export function parseCompactInfonsFormat(text) {
  if (!text || typeof text !== 'string') return null
  
  // Try to match optional header: infons[N]:
  const headerMatch = text.match(/infons\[(\d+)\]:/)
  
  let dataText = text
  if (headerMatch) {
    // Header found, skip it
    const headerEnd = headerMatch.index + headerMatch[0].length
    dataText = text.slice(headerEnd)
  }
  // If no header, treat entire text as data
  
  const lines = dataText.split('\n')
  const infons = []
  
  // Generate single timestamp for all infons in this extraction
  const recordTime = new Date().toISOString()
  
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue
    
    // Check if line starts with valid infon type (desc:, scen:, rel:)
    if (!trimmed.match(/^(desc|scen|rel):/)) continue
    
    const infon = parseCompactInfonLine(trimmed, { recordTime })
    if (infon) {
      infons.push(infon)
    }
  }
  
  return { infons }
}

/**
 * Incremental compact format parser for streaming infons
 * New format: No header, direct data lines (optional "infons[N]:" header supported)
 */
export function incrementalExtractInfonsCompact(streamText, parser) {
  // Merge with defaults to ensure all required fields exist
  const state = {
    foundHeader: false,
    scanPos: 0,
    parsedLines: 0,
    buffer: '',
    infonCount: 0,
    recordTime: null,
    // Preserve any existing fields from parser (like formatDetected, isCompact)
    ...(parser || {}),
    // But ensure critical fields have defaults if missing
    foundHeader: parser?.foundHeader ?? false,
    scanPos: parser?.scanPos ?? 0,
    parsedLines: parser?.parsedLines ?? 0,
    buffer: parser?.buffer ?? '',
    infonCount: parser?.infonCount ?? 0,
    recordTime: parser?.recordTime ?? null
  }
  
  const yielded = []
  const text = String(streamText || '')
  
  // Generate timestamp once for this extraction session
  if (!state.recordTime) {
    state.recordTime = new Date().toISOString()
  }
  
  // Step 1: Check for optional header (for compatibility)
  if (!state.foundHeader) {
    const headerMatch = text.match(/infons\[(\d+)\]:/)
    if (headerMatch) {
      // Header found, skip it
      state.foundHeader = true
      state.scanPos = headerMatch.index + headerMatch[0].length
    } else {
      // No header, treat as headerless format
      state.foundHeader = true
      state.scanPos = 0
    }
  }
  
  // Step 2: Parse data lines incrementally
  const dataText = text.slice(state.scanPos)
  const lines = dataText.split('\n')
  
  // Process each line after the already parsed ones
  for (let i = state.parsedLines; i < lines.length; i++) {
    const line = lines[i]
    
    // Skip if we're at the last line and it might be incomplete (no trailing newline yet)
    if (i === lines.length - 1 && !dataText.endsWith('\n')) {
      break
    }
    
    const trimmed = line.trim()
    if (!trimmed) {
      state.parsedLines++
      continue
    }
    
    // Only process lines that start with valid infon type
    if (!trimmed.match(/^(desc|scen|rel):/)) {
      state.parsedLines++
      continue
    }
    
    // Use shared recordTime for all infons in this stream
    const infon = parseCompactInfonLine(trimmed, { recordTime: state.recordTime })
    if (infon) {
      infon._objIndex = state.infonCount
      infon._isComplete = true
      yielded.push(infon)
      state.infonCount++
    }
    
    state.parsedLines++
  }
  
  return { state, yielded }
}