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
You are a multimodal infons extractor. Extract information as structured JSON with 4 infon types:

1. **DESC** (Entities & Attributes): {"iid": "desc:...", "infon_type": "DESC", "entity": "category", "attribute": "exact_value_from_input", "data_type": "string|number|boolean"}
   - attribute = exact text/value from input (for highlighting)
   - entity = abstract category
   
2. **SCEN** (Time & Space): {"iid": "scen:...", "infon_type": "SCEN", "temporal": "time_expression", "spatial": "location", "granularity": "year|month|day|hour", "bbox": [x,y,w,h]}
   
3. **REL** (Relations): {"iid": "rel:...", "infon_type": "REL", "relation_name": "relation_type", "arity": N, "arg_refs": ["iid1","iid2"], "arg_types": ["DESC","SCEN"]}
   
4. **SIT** (Context/Scene): {"iid": "sit:...", "infon_type": "SIT", "situation_type": "discourse|scene|event", "description": "brief_description"}

**Core Principle**: Extract EVERY distinct information element separately. Be comprehensive and granular.
`;

export const ONTOLOGY = String.raw`
**Critical Rules**:
- **DESC**: attribute = exact text from input (highlightable); entity = category
- **SCEN**: Extract ALL time/place references, include bbox for visual regions
- **REL**: Link infons to reveal relationships, implications, and patterns
- **SIT**: Describe overall context/scene
`;

export const OUTPUT_CONSTRAINTS = String.raw`
**Output Format**:
- Single JSON object only, no markdown/explanations
- IID format: "{type}:r{round}_{index}" (e.g., "desc:r1_1", "rel:r1_2")
- Each infon needs: iid, infon_type, record_time, confidence, support
- Confidence: [0,1], high only for directly observed facts
- **Language Rule**: Use SAME language as input (don't translate)
- **Attribute Rule**: DESC attribute = exact text from input
`;

export const OUTPUT_FORMAT = String.raw`
{"infons": [
  {"iid": "desc:r1_1", "infon_type": "DESC", "entity": "类别", "attribute": "原文值", "data_type": "string", "record_time": "ISO8601", "confidence": 0.95, "support": {"sid":"sit:r1_1","justification":""}},
  {"iid": "scen:r1_2", "infon_type": "SCEN", "temporal": "时间表达", "spatial": "地点", "granularity": "day", "bbox": [x,y,w,h], "record_time": "ISO8601", "confidence": 0.90, "support": {"sid":"sit:r1_1","justification":""}},
  {"iid": "rel:r1_3", "infon_type": "REL", "relation_name": "关系名", "arity": 2, "arg_refs": ["desc:r1_1","scen:r1_2"], "arg_types": ["DESC","SCEN"], "record_time": "ISO8601", "confidence": 0.90, "support": {"sid":"sit:r1_1","justification":""}},
  {"iid": "sit:r1_1", "infon_type": "SIT", "situation_type": "scene", "description": "场景描述", "record_time": "ISO8601", "confidence": 1.0, "support": {"sid":"sit:self","justification":""}}
]}
`;

export const TEXT_EXTRACTION = String.raw`
**Text Extraction**:
1. **SIT**: Overall context/topic
2. **DESC**: Every entity/attribute pair (attribute = exact text)
3. **SCEN**: ALL time/place mentions (use exact words)
4. **REL**: Connections between infons (names, ages, locations, preferences, etc.)

Example "我叫王小明，今年27岁了" → Extract: SIT(自我介绍), DESC(人称:我), DESC(姓名:王小明), SCEN(今年), DESC(年龄:27), REL(名字), REL(年龄关系)
`;

export const IMAGE_EXTRACTION = String.raw`
**Image Extraction - BE EXTREMELY COMPREHENSIVE**:

1. **SIT**: Overall scene description (indoor/outdoor, setting, event type, atmosphere)

2. **DESC - Extract EVERY visual detail**:
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

3. **SCEN**: 
   - Spatial positions with bbox [x,y,w,h] for EVERY object/person
   - Location indicators (street signs, landmarks, architectural style)
   - Time indicators (clock faces, sun position, shadows)

4. **REL - Extract ALL relationships**:
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

1. **SIT**: 
   - Speech context: conversation, announcement, phone call, meeting, interview, etc.
   - Indicate this is voice input: "语音输入", "voice message", etc.

2. **DESC - Extract from transcribed speech**:
   - Speakers mentioned ("我", "他说", "张三", etc.)
   - All entities, names, numbers mentioned in speech
   - Attribute = exact transcribed words (for highlighting)
   - Mark speech-specific attributes: tone indicators, filler words ("嗯", "啊", "well", "um")
   - Extract pronunciation cues if present

3. **SCEN**: 
   - Time/place references mentioned in speech ("昨天", "明天", "公司", "家里")
   - Speech timestamps if available (start/end times of segments)
   - No bbox needed (audio has no visual coordinates)

4. **REL - Speech-specific relations**:
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
- SIT: 语音自我介绍
- DESC: 语气词:嗯, 人称:我, 姓名:王小明, 年龄:27
- SCEN: 今年
- REL: 名字关系, 年龄关系
`;



export const SELF_CHECKLIST = String.raw`
**Pre-Output Checks**:
✓ JSON only, no markdown
✓ Each info element = separate infon
✓ Correct iid format (desc:/scen:/rel:/sit: + r{round}_{index})
✓ DESC: attribute = exact text from input
✓ SCEN: bbox for all visual objects
✓ REL: comprehensive relationships extracted
✓ Same language as input (no translation)
✓ Required fields present (iid, infon_type, record_time, confidence, support)
✓ For images: 20+ DESC, 10+ REL minimum
`;

export const EXAMPLES_SNIPPET = String.raw`
**Example** "我叫王小明，今年27岁了" (round 1):
{"infons": [
  {"iid":"sit:r1_1", "infon_type":"SIT", "description":"自我介绍"},
  {"iid":"desc:r1_2", "infon_type":"DESC", "entity":"人称", "attribute":"我"},
  {"iid":"desc:r1_3", "infon_type":"DESC", "entity":"姓名", "attribute":"王小明"},
  {"iid":"scen:r1_4", "infon_type":"SCEN", "temporal":"今年"},
  {"iid":"desc:r1_5", "infon_type":"DESC", "entity":"年龄", "attribute":"27"},
  {"iid":"rel:r1_6", "infon_type":"REL", "relation_name":"名字", "arg_refs":["desc:r1_2","desc:r1_3"]},
  {"iid":"rel:r1_7", "infon_type":"REL", "relation_name":"年龄关系", "arg_refs":["desc:r1_2","desc:r1_5"]}
]}

✗ WRONG: {"entity":"大麦芽醋", "attribute":"禁止成分"} ← cannot highlight!
✓ CORRECT: {"entity":"成分", "attribute":"大麦芽醋"} ← highlightable!
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
  
  // Add conversation round context
  let contextInfo = `\n【Current Extraction Context】\n- Current conversation round: ${currentRound}\n- Generate iid using format: "{type_prefix}:r${currentRound}_{index}" (index starts from 1 for this extraction)\n`;
  
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
      } else if (type === 'SIT') {
        summary = infon.description || 'Situation';
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