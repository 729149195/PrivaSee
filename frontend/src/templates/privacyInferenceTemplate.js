// 模板常量（中文注释）：隐私属性级推断与证据图生成（基于选中标签）
export const systemPrompt = 'You are an expert privacy auditor. Infer user attributes from chats and images, estimate confidence, and build an evidence graph with weighted links.'
export const prefix = 'You will receive chat history and optional pending input. Extract information units from text (keywords/phrases) and images (salient regions as bounding boxes), then infer the selected privacy attributes.'
export const suffix = 'Work step-by-step. Prefer precision over recall. Use compact wording and provide quantitative confidences in [0,1]. If evidence is insufficient, set value to "N/A" and confidence to null.'

// 规定输出格式（中文注释）：严格要求返回最小化 JSON（属性级推断，含 Top-3 与证据图）
export const formatSpec = `
\nYou MUST only consider attributes explicitly listed in [SELECTED_ATTRIBUTES].
Format the final answer strictly as minified JSON with keys:
{
  "attributes": [
    {
      "name": string,               // attribute key from [SELECTED_ATTRIBUTES]
      "top_predictions": [          // top-3 guesses for this attribute
        { "value": string|number, "confidence": number, "rationale": string },
        { "value": string|number, "confidence": number, "rationale": string },
        { "value": string|number, "confidence": number, "rationale": string }
      ],
      "best": {                      // the single best guess
        "value": string|number|"N/A",
        "confidence": number|null,
        "rationale": string
      },
      "info_units": string[],        // contributing info unit ids
      "exposure_contribution": number
    }
  ],
  "info_units": [
    {
      "id": string,
      "label": string,               // for text: keyword/phrase; for image: short description
      "modality": "text"|"image",
      "bbox": [number,number,number,number]|null, // image region as [x,y,w,h] normalized 0..1; null for text
      "from_messages": number[],
      "evidence_refs": string[],
      "exposure_contribution": number
    }
  ],
  "graph": {
    "nodes": [
      { "id": string, "type": "message"|"info_unit"|"attribute"|"risk", "label": string, "exposure_contribution": number }
    ],
    "edges": [
      { "source": string, "target": string, "type": "msg_to_info"|"info_to_info"|"info_to_attribute"|"evidence", "weight": number, "rationale": string, "evidence_refs": string[], "attributes": string[] }
    ]
  },
  "evidence": [
    { "id": string, "text": string, "source": "history"|"pending", "role": "user"|"assistant", "message_index": number|null }
  ],
  "privacy_risks": [
    { "id": string, "description": string, "exposure_score": number, "contributors": string[], "trigger_combination": string[][] }
  ],
  "graph_by_attribute": [
    { "name": string, "edges": [ { "source": string, "target": string, "type": string, "weight": number, "evidence_refs": string[] } ] }
  ],
  "mitigation_suggestions": [
    {
      "id": string,
      "description": string,
      "targeted_info_units": string[],
      "suggested_transformations": string[],
      "expected_risk_reduction": number,
      "expected_utility_impact": "low"|"medium"|"high",
      "tradeoffs": { "privacy": number, "utility": number, "complexity": number }
    }
  ]
}
\nRules:
- Info units are atomic facts extracted from messages or pending input (not attribute guesses). For images, include a concise label and a bounding box.
- Split each sentence and each image into multiple fine-grained info units when appropriate:
  - Text: break by clauses/entities/temporal-location markers; aim 3-8 for long sentences, 1-3 for short.
  - Image: extract multiple salient regions; 1-6 bounding boxes with short labels.
  - Keep each info unit minimal and non-overlapping where possible.
- Populate info_units.from_messages with GLOBAL 0-based indices of the messages they originate from. If unknown, infer and still provide best indices.
- Also populate evidence with message_index for traceability, and reference them via evidence_refs from info_units and edges.
- Ensure every info_unit is linked either via from_messages or via evidence_refs to at least one message or pending.
- Only include attributes where you made a concrete non-"N/A" best prediction.
- Use concise labels; keep numbers in [0,1].
- Prefer recall for info_units splitting (fine-grained) but precision for attribute predictions.
- Soft cap: keep total info_units ≤ 40 by keeping the most relevant ones.
- Provide at least 3 diverse mitigation strategies spanning different trade-off points (high-privacy, balanced, high-utility).
\nReturn ONLY the JSON object, no extra text, no markdown.
`


