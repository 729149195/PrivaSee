from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel

# Presidio imports (optional at runtime; endpoint will report clear error if unavailable)
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    _PRESIDIO_AVAILABLE = True
except Exception:  # pragma: no cover - narrow failures treated alike for simplicity
    _PRESIDIO_AVAILABLE = False

# ================= Uncertainty API (token influence + uncertainty) =================
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


router = APIRouter()
APP_REF: FastAPI | None = None


class AnalyzeRequest(BaseModel):
    text: str
    language: str | None = "en"
    model: str | None = None
    entities: list[str] | None = None


class AnalyzeEntity(BaseModel):
    start: int
    end: int
    entity_type: str
    score: float


class AnalyzeResponse(BaseModel):
    entities: list[AnalyzeEntity]


def _create_analyzer_for_model(model_key: str | None, language: str | None):
    if not _PRESIDIO_AVAILABLE:
        return None, (
            "Presidio analyzer is not available. Ensure packages are installed: "
            "pip install presidio-analyzer presidio-anonymizer spacy && python -m spacy download en_core_web_sm"
        )

    key = (model_key or "spacy/en_core_web_sm").strip()
    lang = (language or "en").strip()

    spacy_sm_by_lang = {
        "en": "en_core_web_sm",
        "zh": "zh_core_web_sm",
        "es": "es_core_news_sm",
        "de": "de_core_news_sm",
        "it": "it_core_news_sm",
        "fr": "fr_core_news_sm",
        "pt": "pt_core_news_sm",
    }

    try:
        if key.startswith("spacy/"):
            requested_model = key.split("/", 1)[1]
            if requested_model.endswith("_lg") and lang == "en":
                model_name = requested_model
            else:
                model_name = spacy_sm_by_lang.get(lang, "en_core_web_sm")
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": lang, "model_name": model_name},
                ],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()
            return AnalyzerEngine(nlp_engine=nlp_engine), None

        return None, (
            "Unsupported model. Supported: spacy/en_core_web_sm, spacy/en_core_web_lg"
        )
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


def _normalize_for_patterns(text: str) -> str:
    if not text:
        return text
    dash_variants = {"–": "-", "—": "-", "－": "-", "﹘": "-", "‒": "-", "―": "-"}
    punct_map = {
        "：": ":", "，": ",", "。": ".", "；": ";", "（": "(", "）": ")",
        "［": "[", "］": "]", "｛": "{", "｝": "}", "／": "/", "＠": "@",
        "＆": "&", "＋": "+", "＝": "=", "＿": "_", "＃": "#", "％": "%",
        "＊": "*", "！": "!", "？": "?", "｜": "|", "～": "~",
    }
    normalized_chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xFF10 <= code <= 0xFF19:
            normalized_chars.append(chr(code - 0xFF10 + ord("0")))
            continue
        if ch in dash_variants:
            normalized_chars.append("-")
            continue
        if ch in punct_map:
            normalized_chars.append(punct_map[ch])
            continue
        normalized_chars.append(ch)
    return "".join(normalized_chars)


class UncertaintyRequest(BaseModel):
    text: str
    model: str | None = "distilgpt2"
    samples: int | None = 16
    ig_steps: int | None = 16
    max_input_tokens: int | None = 256
    seed: int | None = 42


class UncertaintyToken(BaseModel):
    start: int
    end: int
    token: str
    score_mean: float
    score_std: float


class UncertaintyResponse(BaseModel):
    tokens: list[UncertaintyToken]


def _get_lm(model_name: str):
    app_state = getattr(APP_REF, "state", None)
    cache = getattr(app_state, "uncert_cache", {}) if app_state is not None else {}
    if model_name in cache:
        return cache[model_name]
    if not _TRANSFORMERS_AVAILABLE:
        return None
    # 兼容 transformers 版本差异：提供缺失的 SequenceSummary（某些版本中被移除或迁移）
    try:
        import types as _types
        from transformers import modeling_utils as _mu  # type: ignore
        if not hasattr(_mu, "SequenceSummary"):
            import torch.nn as _nn
            class _SequenceSummary(_nn.Module):  # type: ignore
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.proj = _nn.Identity()
                def forward(self, hidden_states, *args, **kwargs):  # noqa: D401
                    return hidden_states
            setattr(_mu, "SequenceSummary", _SequenceSummary)
        # 某些版本需要确保模块属性可被 "from ... import SequenceSummary" 找到
        pkg = _mu.__package__
        if pkg:
            import importlib as _importlib
            _module = _importlib.import_module(pkg)
            if not hasattr(_module, "SequenceSummary") and hasattr(_mu, "SequenceSummary"):
                setattr(_module, "SequenceSummary", getattr(_mu, "SequenceSummary"))
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    cache[model_name] = (tokenizer, model)
    if app_state is not None:
        app_state.uncert_cache = cache
    return cache[model_name]


def _integrated_gradients_saliency(tokenizer, model, text: str, samples: int, ig_steps: int, max_input_tokens: int, seed: int | None) -> list[UncertaintyToken]:
    if not text:
        return []
    device = next(model.parameters()).device
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    offsets = enc["offset_mapping"][0].tolist()

    embed_layer = model.get_input_embeddings()
    embed = embed_layer(input_ids)
    seq_len = embed.shape[1]
    scores = []

    def _set_dropout_train(m):
        for module in m.modules():
            if module.__class__.__name__.lower().find("dropout") >= 0:
                module.train()

    for _ in range(max(1, samples)):
        if seed is not None:
            try:
                import torch as _torch
                _torch.manual_seed(seed)
            except Exception:
                pass
        model.zero_grad(set_to_none=True)
        _set_dropout_train(model)
        baseline = torch.zeros_like(embed)
        steps = max(1, ig_steps)
        total_grads = torch.zeros_like(embed)
        for k in range(1, steps + 1):
            alpha = float(k) / steps
            inputs_embeds = (baseline + alpha * (embed - baseline)).clone().detach().requires_grad_(True)
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn)
            logits = outputs.logits[:, -1, :]
            top_id = logits.argmax(dim=-1)
            value = logits[0, top_id[0]]
            model.zero_grad(set_to_none=True)
            value.backward(retain_graph=True)
            total_grads += inputs_embeds.grad
        avg_grads = total_grads / steps
        ig = (embed - baseline) * avg_grads
        sal = ig.abs().sum(dim=-1).detach().cpu().numpy()[0]
        scores.append(sal.tolist())

    import numpy as np
    arr = np.array(scores)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    def _robust_scale(x):
        import numpy as _np
        p95 = float(_np.quantile(x, 0.95)) if x.size > 0 else 1.0
        p95 = p95 if p95 > 1e-8 else 1.0
        out = _np.clip(x / p95, 0.0, 1.0)
        return out

    mean = _robust_scale(mean)
    std = _robust_scale(std)

    tokens: list[UncertaintyToken] = []
    text_bytes = text
    for i in range(seq_len):
        start, end = offsets[i]
        if end <= start:
            continue
        tok = text_bytes[start:end]
        tokens.append(
            UncertaintyToken(start=int(start), end=int(end), token=tok, score_mean=float(mean[i]), score_std=float(std[i]))
        )
    return tokens


class EntitiesRequest(BaseModel):
    language: str | None = "en"
    model: str | None = None


class EntitiesResponse(BaseModel):
    entities: list[str]


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    if not req.text:
        return AnalyzeResponse(entities=[])
    if len(req.text) > 20000:
        raise HTTPException(status_code=413, detail="Text too large (limit 20,000 characters)")
    language = "en"
    model_key = req.model or "spacy/en_core_web_sm"
    app_state = getattr(APP_REF, "state", None)
    cache = getattr(app_state, "analyzer_cache", {}) if app_state is not None else {}
    cache_key = f"{model_key}|{language}"
    analyzer = cache.get(cache_key)
    if analyzer is None:
        analyzer, error = _create_analyzer_for_model(model_key, language)
        if analyzer is None:
            raise HTTPException(status_code=503, detail=error or "Analyzer unavailable")
        cache[cache_key] = analyzer
        if app_state is not None:
            app_state.analyzer_cache = cache
    norm_text = _normalize_for_patterns(req.text)
    results = analyzer.analyze(text=norm_text, language=language)
    allowed = set([e.upper() for e in (req.entities or [])]) if req.entities else None
    seen = set()
    entities = []
    for r in results:
        if allowed and r.entity_type.upper() not in allowed:
            continue
        key = (r.start, r.end, r.entity_type)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            AnalyzeEntity(start=r.start, end=r.end, entity_type=r.entity_type, score=r.score)
        )
    entities.sort(key=lambda x: (x.start, -(x.end - x.start)))
    return AnalyzeResponse(entities=entities)


@router.post("/entities", response_model=EntitiesResponse)
def get_supported_entities(req: EntitiesRequest):
    model_key = req.model or "spacy/en_core_web_sm"
    app_state = getattr(APP_REF, "state", None)
    language = "en"
    cache = getattr(app_state, "analyzer_cache", {}) if app_state is not None else {}
    cache_key = f"{model_key}|{language}"
    analyzer = cache.get(cache_key)
    if analyzer is None:
        analyzer, error = _create_analyzer_for_model(model_key, language)
        if analyzer is None:
            raise HTTPException(status_code=503, detail=error or "Analyzer unavailable")
        cache[cache_key] = analyzer
        if app_state is not None:
            app_state.analyzer_cache = cache
    try:
        entities = sorted(analyzer.get_supported_entities())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))
    return EntitiesResponse(entities=entities)


@router.post("/uncertainty", response_model=UncertaintyResponse)
def compute_uncertainty(req: UncertaintyRequest):
    if not _TRANSFORMERS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Transformers/torch not available. Install: pip install transformers torch --upgrade")
    if not req.text:
        return UncertaintyResponse(tokens=[])
    if len(req.text) > 4000:
        raise HTTPException(status_code=413, detail="Text too large (limit 4,000 characters)")

    model_name = req.model or "distilgpt2"
    samples = int(req.samples or 16)
    ig_steps = int(req.ig_steps or 16)
    max_tokens = int(req.max_input_tokens or 256)
    seed = req.seed

    bundle = _get_lm(model_name)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model could not be loaded. Check model name and installation.")
    tokenizer, model = bundle
    try:
        tokens = _integrated_gradients_saliency(tokenizer, model, req.text, samples=samples, ig_steps=ig_steps, max_input_tokens=max_tokens, seed=seed)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))
    return UncertaintyResponse(tokens=tokens)


def register_text_routes(app: FastAPI):
    global APP_REF
    APP_REF = app
    app.include_router(router)


