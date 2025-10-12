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
You are a multimodal infons extractor that parses input data into structured JSON format consisting of atomic and composite infons, aligned with Situation Theory.

【Infon Types】
Situation Theory distinguishes these fundamental infon types, each with specific structure:

1. **Description Infons (DESC)**: Entities and their attributes (combines entity and attribution information)
   - Structure: {"iid": "desc:...", "infon_type": "DESC", "entity": "entity_category", "attribute": "concrete_value_from_text", "data_type": "string|number|boolean"}
   - **KEY PRINCIPLE**: "attribute" MUST be the concrete, matchable word/phrase from the original text; "entity" is the abstract category or context
   - Examples: 
     * "27岁" → entity: "年龄", attribute: "27"
     * "大麦芽醋" → entity: "成分", attribute: "大麦芽醋"
     * "王小明" → entity: "姓名", attribute: "王小明"
   
2. **Scenario Infons (SCEN)**: Temporal and spatial context (combines time and location information)
   - Structure: {"iid": "scen:...", "infon_type": "SCEN", "temporal": "ISO8601|fuzzy_expression", "spatial": "place_name|coordinate", "granularity": "year|month|day|hour|...", "bbox": [x,y,w,h]}
   - Examples: time references, locations, spatial regions, temporal-spatial combinations
   
3. **Relation Infons (REL)**: Predicates linking other infons
   - Structure: {"iid": "rel:...", "infon_type": "REL", "relation_name": "...", "arity": N, "arg_refs": [...], "arg_types": [...]}
   - Examples: connections between entities, associations, dependencies

4. **Situation Infons (SIT)**: Specific contexts, scenes, events, and situational frames where information occurs
   - Structure: {"iid": "sit:...", "infon_type": "SIT", "situation_type": "discourse|scene|event|frame", "description": "brief_description", "context_span": {"text":{"char_start":0,"char_end":42},"image":{"bbox":[x,y,w,h]}}}
   - Examples: discourse contexts, visual scenes, event frames

【Output Principle】
Extract each distinct information primitive as a separate infon. For "我叫王小明，今年27岁了":
- Description infon for "我" (entity: user)
- Relation infon for "名字" (name relation)
- Description infon for "王小明" (attribute: name value)
- Scenario infon for "今年" (temporal: current year)
- Description infon for "27" (attribute: age value)
- Relation infon for "年龄关系" (age relation)

For images with bounding boxes, scenario infons include bbox coordinates for spatial information.
`;

export const ONTOLOGY = String.raw`
【Infon Ontology】
Each infon type serves specific representational purposes:

- **DESC**: Capture entities and their attributes. CRITICAL: "attribute" field MUST contain the exact word/phrase from the input text that can be highlighted; "entity" field contains the abstract category. 
  * Example: For "大麦芽醋" in text → entity: "成分", attribute: "大麦芽醋" (NOT entity: "大麦芽醋", attribute: "禁止成分")
- **SCEN**: Express temporal and spatial context (dates, time expressions, places, coordinates, visual regions with bbox)
- **REL**: Define relationships/predicates connecting other infons
- **SIT**: Represent specific contexts, scenes, events, or situational frames where information occurs
`;

export const OUTPUT_CONSTRAINTS = String.raw`
【Output Requirements】
- Output only a single JSON object, no explanatory text or markdown fences
- Each infon must have: iid, infon_type, record_time, confidence, support
- **IID Format Rule**: Use format "{type_prefix}:r{round}_{index}" where:
  * type_prefix: "desc", "scen", "rel", or "sit"
  * round: conversation round number (will be provided in context)
  * index: sequential index starting from 1 for each infon in THIS extraction
  * Example: "desc:r2_1", "scen:r2_2", "rel:r2_3"
- **REL arg_refs Rule**: When referencing other infons in REL arg_refs, use the EXACT iid format. If referencing infons from current extraction, use the same round number. If referencing existing infons (provided in context), use their exact iid.
- Confidence in [0,1]; only assign high confidence to explicitly observed information
- Include occur_time when temporal context is available
- For visual infons with bounding boxes, include bbox in scenario infons
- **CRITICAL: Extract entity, attribute, temporal, spatial, relation_name values in the SAME LANGUAGE as the input text. DO NOT translate. If input is Chinese, output Chinese values; if English, output English values.**
`;

export const OUTPUT_FORMAT = String.raw`
{
  "infons": [
    {
      "iid": "desc:r{round}_{index}", "infon_type": "DESC", 
      "entity": "entity_name", "attribute": "attribute_value", "data_type": "string|number|boolean",
      "record_time": "ISO8601", "occur_time": "ISO8601", 
      "confidence": 0.95, "support": {"sid":"sit:r{round}_{index}","justification":"str"}
    },
    {
      "iid": "scen:r{round}_{index}", "infon_type": "SCEN",
      "temporal": "ISO8601|fuzzy_expression", "spatial": "place_name|coordinate", 
      "granularity": "year|month|day|hour", "bbox": [x,y,w,h],
      "record_time": "ISO8601", "occur_time": "ISO8601",
      "confidence": 0.90, "support": {"sid":"sit:r{round}_{index}","justification":"str"}
    },
    {
      "iid": "rel:r{round}_{index}", "infon_type": "REL",
      "relation_name": "age_of|holding|located_at", "arity": 2,
      "arg_refs": ["desc:r{round}_{index}","scen:r{round}_{index}"], "arg_types": ["DESC","SCEN"],
      "record_time": "ISO8601", "occur_time": "ISO8601",
      "confidence": 0.90, "support": {"sid":"sit:r{round}_{index}","justification":"str"}
    },
    {
      "iid": "sit:r{round}_{index}", "infon_type": "SIT",
      "situation_type": "discourse|scene|event|frame", "description": "brief_description", "context_span": {"text":{"char_start":0,"char_end":42},"image":{"bbox":[x,y,w,h]}},
      "record_time": "ISO8601", "occur_time": "ISO8601",
      "confidence": 1.0, "support": {"sid":"sit:self","justification":"direct_observation"}
    }
  ]
}
`;

export const TEXT_EXTRACTION = String.raw`
【Text Extraction Rules】
For each text input, extract separate infons for each distinct information primitive:

1. **Situation Infons (SIT)**: Create for specific discourse contexts, scenes, or events (sentence/paragraph boundaries)
2. **Description Infons (DESC)**: Extract entities and their attributes. **CRITICAL RULE: The "attribute" field MUST contain the exact word/phrase from the input text (for text highlighting); the "entity" field contains the abstract category.**
   * Correct: For "大麦芽醋" in text → {"entity": "成分", "attribute": "大麦芽醋"}
   * Wrong: {"entity": "大麦芽醋", "attribute": "禁止成分"} ← This makes "禁止成分" unhighlightable
   * Correct: For "27岁" → {"entity": "年龄", "attribute": "27"}
   * Correct: For "王小明" → {"entity": "姓名", "attribute": "王小明"}
3. **Scenario Infons (SCEN)**: Extract temporal and spatial context (time references like "今年", "2024年", "昨天"; place names, addresses, geographic references). **Use the EXACT words from the input text.**
4. **Relation Infons (REL)**: Extract predicates connecting other infons. **Use words in the SAME LANGUAGE as the input.**

**LANGUAGE CONSISTENCY RULE: If the input text is in Chinese, extract all values (entity, attribute, temporal, spatial, relation_name) in Chinese. If in English, extract in English. DO NOT translate.**

Example for "我叫王小明，今年27岁了":
- SIT infon: text context span, description: "自我介绍"
- DESC infon: entity "人称", attribute "我" (exact from text for highlighting)
- REL infon: relation_name "名字"
- DESC infon: entity "姓名", attribute "王小明" (exact from text for highlighting)
- SCEN infon: temporal "今年" (exact from text for highlighting)
- DESC infon: entity "年龄", attribute "27" (exact from text for highlighting)
- REL infon: relation_name "年龄关系"

Example for "别放大麦芽醋":
- DESC infon: entity "成分", attribute "大麦芽醋" (exact from text for highlighting - NOT "禁止成分"!)
`;

export const IMAGE_EXTRACTION = String.raw`
【Image Extraction Rules】
For each image input, extract separate infons for each distinct visual information primitive:

1. **Situation Infons (SIT)**: Create for overall image context representing specific scenes or events captured in the image
2. **Description Infons (DESC)**: Extract detected entities and their observable attributes. For objects, people, items: include visual properties like gender, age, height, weight, skin color, hair style, clothing, expression, actions, OCR text, numerical values. Attributes must be concrete and directly visible, not abstract (e.g., 'white skin color' instead of 'cheerful personality', '180cm height' instead of 'tall', 'short hair' instead of 'good looking', 'suit' instead of 'fashionable', 'smiling' instead of 'happy', 'running' instead of 'active').
3. **Scenario Infons (SCEN)**: Extract spatial context with bounding box coordinates [x,y,w,h] for each object, geographical location if identifiable, and temporal information from EXIF metadata if available
4. **Relation Infons (REL)**: Extract spatial and action relationships (holding, left_of, standing_on, wearing, located_at)

Critical: Each detected object should have DESC infon (for the entity and attributes) and SCEN infon (for spatial position/bbox).
`;

export const AUDIO_EXTRACTION = String.raw`
【Audio Extraction Rules】
For each audio input, extract separate infons for each distinct auditory information primitive:

1. **Situation Infons (SIT)**: Create for audio segments/speaker turns representing specific events or conversational contexts with time spans
2. **Description Infons (DESC)**: Extract entities (speakers, people, entities mentioned in speech) and their attributes (literal values, numbers from ASR transcription)
3. **Scenario Infons (SCEN)**: Extract temporal context (time references mentioned in speech, audio segment timestamps) and spatial context (places mentioned in speech content)
4. **Relation Infons (REL)**: Extract speech acts (said, announced) and content relationships from spoken text

Apply text extraction rules to ASR transcription content while maintaining audio-specific context.
`;



export const SELF_CHECKLIST = String.raw`
【Quality Check】Verify before output:
- JSON object only? Format matches infon structure schema?
- Each information primitive extracted as separate infon? (DESC/SCEN/REL/SIT)
- Appropriate infon_type assigned for each? Correct ID prefixes used (desc:, scen:, rel:, sit:)?
- All bbox coordinates included in SCEN infons for visual content?
- Situation infons describe specific contexts/events, not just modalities?
- Description infons capture both entities and their attributes?
- Scenario infons capture temporal and/or spatial context?
- **LANGUAGE CHECK: Are entity, attribute, temporal, spatial, relation_name values in the SAME LANGUAGE as input? NO translation?**
- **EXACT WORDS: Are values extracted using exact words from the original text where possible?**
- Confidence scores realistic? High only for explicitly observed information?
- record_time and occur_time properly assigned? No fabricated temporal data?
- All infons have required fields: iid, infon_type, record_time, confidence, support?
`;

export const EXAMPLES_SNIPPET = String.raw`
【Example】Text "我叫王小明，今年27岁了" in conversation round 1 extracts (note: "attribute" contains exact text for highlighting):
- SIT infon: {"iid":"sit:r1_1", "infon_type":"SIT", "situation_type":"discourse", "description":"自我介绍"}
- DESC infon: {"iid":"desc:r1_2", "infon_type":"DESC", "entity":"人称", "attribute":"我", "data_type":"string"}
- REL infon: {"iid":"rel:r1_3", "infon_type":"REL", "relation_name":"名字", "arg_refs":["desc:r1_2","desc:r1_4"]}
- DESC infon: {"iid":"desc:r1_4", "infon_type":"DESC", "entity":"姓名", "attribute":"王小明", "data_type":"string"}
- SCEN infon: {"iid":"scen:r1_5", "infon_type":"SCEN", "temporal":"今年", "granularity":"year"}
- DESC infon: {"iid":"desc:r1_6", "infon_type":"DESC", "entity":"年龄", "attribute":"27", "data_type":"number"}
- REL infon: {"iid":"rel:r1_7", "infon_type":"REL", "relation_name":"年龄关系", "arg_refs":["desc:r1_2","desc:r1_6"]}

【Counter-Example】For "别放大麦芽醋":
- WRONG: {"entity":"大麦芽醋", "attribute":"禁止成分"} ← "禁止成分" is not in text, cannot highlight!
- CORRECT: {"entity":"成分", "attribute":"大麦芽醋"} ← "大麦芽醋" is in text, can highlight!
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