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

【Core Concepts】
- Atomic infon: ⟨⟨R, a₁, ..., aₙ, p⟩⟩ where R is an n-ary relation, p ∈ {0,1} indicates polarity (1=supports, 0=does not support)
- Support relation: s ⊨ σ means situation s supports infon σ, with support.sid pointing to SIT
- Minimal granularity: each independent relation produces one atomic infon, logical connections use composite infons

【Output Constraints】
- Output only a single JSON object, no explanatory text
- Unique IDs: namespace prefix + content hash for stable identification
- Bitemporal: occur_time (business/occurrence time) + record_time (recording time)
- Confidence [0,1], do not fabricate evidence or set p=0 for "not detected"
`;

export const ONTOLOGY = String.raw`
【Type System】(Following Situation Theory ontology)
IND: Individual (person/organization/item/visual object) | RELⁿ: n-ary relation type | LOC: Spatial location
TIM: Temporal location (ISO8601 intervals) | SIT: Situation (discourse object) | TYP: Type of individuals
PAR: Parameter (cognitive placeholder) | POL: Polarity (0=false, 1=true) | LIT: Literal values

【Composite Logic】
Supports conjunction (AND), disjunction (OR), implication (IMPLIES), negation (NOT), existential (EXISTS) and universal (FORALL) quantification over situations, using PAR placeholders with infon.bindings for scope annotation
`;

export const OUTPUT_FORMAT = String.raw`
{
  "run_metadata": {"source_id": "str", "record_time": "ISO8601", "generator": "str", "notes": "str"},
  "situations": [{
    "sid": "s:...", "modality": "text|image|audio",
    "span": {"text": {"char_start":0,"char_end":42}, "image": {"bbox":[x,y,w,h]}, "audio": {"t_start":0,"t_end":0}},
    "occur_time": "ISO8601|{\"start\":\"...\",\"end\":\"...\"}", "record_time": "ISO8601",
    "loc": {"type":"LOC","value":"str","geo":{"lat":0,"lon":0}},
    "provenance": {"uri":"str","method":"str","confidence":0.0}
  }],
  "entities": [{
    "eid": "e:...", "names": ["str"], "types": ["IND"],
    "modality_origin": "text|image|audio", "kb_links": [{"kg":"str","node_id":"str"}],
    "visual": {"bbox":[x,y,w,h]}, "text_mention": {"sid":"s:..","char_start":0,"char_end":0},
    "audio_mention": {"sid":"s:..","t_start":0,"t_end":0}
  }],
  "infons": [{
    "iid": "i:...", "kind": "atomic|composite",
    "R": {"name":"str","arity":2,"type_signature":["IND","IND"]},
    "args": [{"ref":"e:alice","type":"IND"}],
    "p": 1, "support": {"sid":"s:...","justification":"str"},
    "occur_time": "ISO8601|区间", "record_time": "ISO8601", "loc": {"type":"LOC","value":"str"},
    "confidence": 0.87, "provenance": {"uri":"str","detector":"str","score":0.87},
    "bindings": [{"var":"x","type":"PAR","scope_sid":"s:..."}],
    "composite": {"op":"AND|OR|NOT|EXISTS","children":["i:child1"]},
    "version": {"n":1,"prev":null,"policy":"append_only"}
  }],
  "quality_report": {
    "stats": {"num_situations":0,"num_entities":0,"num_infons":0},
    "unresolved_parameters": ["x"], "warnings": ["str"]
  }
}
`;

export const TEXT_EXTRACTION = String.raw`
【Text Extraction】Each discourse situation (paragraph/sentence) = SIT → individuals/locations/times/relations → atomic infons
- Individuals: IND/TYP/LIT with coreference resolution → unified eid identification
- Times/Locations: normalized to ISO8601/place names, preserve fuzzy expressions with confidence scores
- Relations: predicate-argument structures → atomic infons ⟨R, args, 1⟩, explicit negation ⟨R, args, 0⟩
- Attributions: said(speaker,content_sit)/authored(author,text) for speech/writing acts
- Quantifiers: PAR parameters, e.g., "an employee" → ⟨EXISTS, x:PAR, ⟨employee, x:IND, 1⟩⟩
`;

export const IMAGE_EXTRACTION = String.raw`
【Image Extraction】Visual situation (image/region) = SIT → visual individuals/spatial relations → infons
- Visual individuals: detected objects → IND, visual attributes → TYP, OCR text → LIT
- Spatial relations: ⟨left_of, obj1:IND, obj2:IND, 1⟩, ⟨holding, person:IND, object:IND, 1⟩, etc.
- Locations: bounding boxes + LOC types, include geographic coordinates if available
- Times: EXIF metadata → occur_time, handle negation of visual absence cautiously
`;

export const AUDIO_EXTRACTION = String.raw`
【Audio Extraction】Auditory situations (speaker/ASR segments) = SIT → speech events/factual claims → infons
- Speech events: ASR transcriptions → LIT entities, ⟨said, speaker:IND, utterance_sit:SIT, 1⟩
- Factual claims: process spoken content using text extraction rules
- Uncertainty: reduce confidence scores + provide justification for speculative interpretations
`;



export const SELF_CHECKLIST = String.raw`
【Quality Check】Verify before output:
- JSON object only? Format matches schema?
- Atomic infons at minimal granularity? No multiple relations combined?
- p=0 only for explicit negation? No "not detected" misclassified?
- R.arity = args count = type_signature?
- Duplicate <R,args,p> merged? Confidence boosted?
- IDs uniquely stable? Time in ISO8601/standard intervals?
`;

export const EXAMPLES_SNIPPET = String.raw`
【Example】Text "Alice joined Acme in Paris on 2024-05-01" + Image "Alice wearing hard hat standing left of car"
- ⟨joined, Alice:IND, Acme:IND, 1⟩, occur_time=2024-05-01, loc=Paris, support.sid=s:S1
- ⟨wearing, Alice:IND, Helmet:TYP, 1⟩, support.sid=s:I1#bbox1
- ⟨left_of, Car:IND, Alice:IND, 1⟩, support.sid=s:I1#bbox1
`;

/* ---------- Combinator: Assemble final system prompt on demand ---------- */
export function buildSystemPrompt(options = {}) {
  const {
    modalities = ["text", "image", "audio"],
    includeExamples = false,
    extraInstructions = ""
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