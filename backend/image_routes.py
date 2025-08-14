from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter
from pydantic import BaseModel

router = APIRouter()
APP_REF: FastAPI | None = None

import logging
logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    from PIL import Image, ExifTags, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import io
    from transformers import AutoProcessor, CLIPModel
    from ultralytics import YOLO
    try:
        import easyocr  # type: ignore
    except Exception:
        easyocr = None
    try:
        from paddleocr import PaddleOCR as _PaddleOCR  # type: ignore
    except Exception:
        _PaddleOCR = None
    try:
        import faiss  # type: ignore
    except Exception:
        faiss = None
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        mp = None
    _VISION_AVAILABLE = True
except Exception:  # pragma: no cover
    _VISION_AVAILABLE = False


class ImgDetectBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float


class ImgUncertaintyBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class ImgOCRBox(ImgDetectBox):
    text: str
    pii_types: list[str] | None = None
    pii_scores: list[float] | None = None
    uncert: float | None = None


class GeoCandidate(BaseModel):
    name: str
    score: float
    source: str
    path: str | None = None


class ImgAnalysisResponse(BaseModel):
    width: int
    height: int
    exif: dict | None = None
    clip_top: list[str] | None = None
    clip_probs: dict[str, list[float]] | None = None
    clip_labels: dict[str, list[str]] | None = None
    detections: list[ImgDetectBox]
    faces: list[ImgDetectBox]
    plates: list[ImgDetectBox]
    ocr_boxes: list[ImgOCRBox]
    heat_boxes: list[ImgUncertaintyBox]
    geo_candidates: list[GeoCandidate] | None = None

    class GeoUncertainty(BaseModel):
        consistency: float
        entropy_norm: float
        margin: float

    geo_uncertainty: GeoUncertainty | None = None
    geo_prob: float | None = None
    pixel_heat: list[list[float]] | None = None


def _exif_to_dict(pil_img: Image.Image) -> dict:
    try:
        exif_data = pil_img._getexif() or {}
        decoded = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif_data.items()}
        return decoded
    except Exception:
        return {}


def _to_box(xyxy, label: str, score: float) -> ImgDetectBox:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return ImgDetectBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label, score=float(score))


def _iou(a: ImgDetectBox, b: ImgDetectBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a.x2 - a.x1)) * max(0.0, (a.y2 - a.y1))
    area_b = max(0.0, (b.x2 - b.x1)) * max(0.0, (b.y2 - b.y1))
    denom = area_a + area_b - inter + 1e-8
    return float(inter / denom) if denom > 0 else 0.0


def _merge_face_boxes(primary: list[ImgDetectBox], secondary: list[ImgDetectBox], iou_thr: float = 0.35) -> list[ImgDetectBox]:
    merged: list[ImgDetectBox] = list(primary)
    for s in secondary:
        best_i = -1
        best_iou = 0.0
        for i, p in enumerate(merged):
            if p.label != "FACE":
                continue
            iou = _iou(p, s)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= iou_thr:
            p = merged[best_i]
            nx1 = (p.x1 + s.x1) / 2.0
            ny1 = (p.y1 + s.y1) / 2.0
            nx2 = (p.x2 + s.x2) / 2.0
            ny2 = (p.y2 + s.y2) / 2.0
            ns = max(float(p.score or 0.0), float(s.score or 0.0))
            merged[best_i] = ImgDetectBox(x1=nx1, y1=ny1, x2=nx2, y2=ny2, label="FACE", score=ns)
        else:
            merged.append(ImgDetectBox(x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2, label="FACE", score=float(s.score or 0.0)))
    return merged


def _gps_to_latlon(exif: dict) -> tuple[float | None, float | None]:
    try:
        gps = exif.get("GPSInfo") or {}
        lat_ref = gps.get(1)
        lat = gps.get(2)
        lon_ref = gps.get(3)
        lon = gps.get(4)
        def _conv(dms):
            vals = [float(a) / float(b) for a, b in (dms or [])]
            return vals[0] + vals[1]/60.0 + vals[2]/3600.0
        if lat and lon:
            lat_v = _conv(lat) * (1 if (lat_ref in ["N", b"N"]) else -1)
            lon_v = _conv(lon) * (1 if (lon_ref in ["E", b"E"]) else -1)
            return lat_v, lon_v
    except Exception:
        pass
    return None, None


def _get_clip_bundle(app_state):
    bundle = getattr(app_state, "clip_bundle", None)
    if bundle is not None:
        return bundle
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    except Exception:
        logger.warning("CLIP local cache not found; falling back to online download for 'openai/clip-vit-base-patch32'.")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    app_state.clip_bundle = (clip_model, clip_processor)
    return app_state.clip_bundle


def _get_ocr_reader(app_state):
    if easyocr is None:
        return None
    reader = getattr(app_state, "ocr_reader", None)
    if reader is None:
        reader = easyocr.Reader(["en", "ch_sim"], gpu=False)
        app_state.ocr_reader = reader
    return reader


def _get_paddle_ocr(app_state):
    if _PaddleOCR is None:
        return None
    inst = getattr(app_state, "paddle_ocr", None)
    if inst is None:
        try:
            inst = _PaddleOCR(use_gpu=False, lang='ch', use_angle_cls=True, rec=True, det=True)
            app_state.paddle_ocr = inst
        except Exception:
            inst = None
    return inst


def _rect_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (a + b - inter + 1e-8))


def _get_captioner(app_state):
    try:
        bundle = getattr(app_state, "blip_bundle", None)
        if bundle is not None:
            return bundle
        from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
        model_names = [
            "Salesforce/blip-image-captioning-large",
            "Salesforce/blip-image-captioning-base",
        ]
        for name in model_names:
            try:
                proc = BlipProcessor.from_pretrained(name)
                mdl = BlipForConditionalGeneration.from_pretrained(name)
                mdl.eval()
                app_state.blip_bundle = (proc, mdl)
                return app_state.blip_bundle
            except Exception:
                continue
        return None
    except Exception:
        return None


def _get_landmark_index(app_state, landmarks_dir: str = "static/landmarks"):
    cache = getattr(app_state, "landmark_index", None)
    if cache is not None:
        return cache
    import os
    if not os.path.isdir(landmarks_dir):
        app_state.landmark_index = None
        return None
    clip_model, clip_processor = _get_clip_bundle(app_state)
    names: list[str] = []
    vecs: list[np.ndarray] = []
    paths: list[str] = []
    for fname in os.listdir(landmarks_dir):
        path = os.path.join(landmarks_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            vecs.append(feat.cpu().numpy().astype(np.float32))
            name = os.path.splitext(fname)[0]
            names.append(name)
            paths.append(path)
        except Exception:
            continue
    if not vecs or faiss is None:
        app_state.landmark_index = None
        return None
    feats = np.vstack(vecs)
    index = faiss.IndexFlatIP(feats.shape[1])
    index.add(feats)
    app_state.landmark_index = {"index": index, "names": names, "paths": paths}
    return app_state.landmark_index


def _tta_images(img: Image.Image, n: int = 12) -> list[Image.Image]:
    import random
    W, H = img.size
    crops = []
    for _ in range(n):
        scale = random.uniform(0.7, 1.0)
        w = int(W * scale)
        h = int(H * scale)
        x0 = random.randint(0, max(0, W - w))
        y0 = random.randint(0, max(0, H - h))
        patch = img.crop((x0, y0, x0 + w, y0 + h))
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        crops.append(patch)
    return crops


@router.post("/image/analyze", response_model=ImgAnalysisResponse)
def analyze_image(file: UploadFile = File(...)):
    if not _VISION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vision dependencies missing. Install: pip install ultralytics transformers pillow")
    try:
        data = file.file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        width, height = img.size
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    exif = _exif_to_dict(img)

    detections: list[ImgDetectBox] = []
    faces: list[ImgDetectBox] = []
    plates: list[ImgDetectBox] = []
    ocr_boxes: list[ImgOCRBox] = []
    heat_boxes: list[ImgUncertaintyBox] = []

    try:
        yolo = YOLO("yolov8n.pt")
        y_res = yolo.predict(img, verbose=False, classes=[0])[0]
        for b in y_res.boxes:
            cls_id = int(b.cls)
            label = yolo.model.names.get(cls_id, str(cls_id))
            detections.append(_to_box(b.xyxy[0].tolist(), label, float(b.conf)))
    except Exception:
        pass

    try:
        if 'mp' in globals() and mp is not None:
            W, H = img.size
            np_img = np.array(img)
            with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:  # type: ignore
                res = fd.process(np_img)
                mp_boxes: list[ImgDetectBox] = []
                if res and getattr(res, 'detections', None):
                    for det in res.detections:
                        rb = det.location_data.relative_bounding_box
                        if rb is None:
                            continue
                        x1 = float(max(0.0, rb.xmin) * W)
                        y1 = float(max(0.0, rb.ymin) * H)
                        x2 = float(min(1.0, rb.xmin + rb.width) * W)
                        y2 = float(min(1.0, rb.ymin + rb.height) * H)
                        score = float(det.score[0]) if getattr(det, 'score', None) else 0.0
                        mp_boxes.append(ImgDetectBox(x1=x1, y1=y1, x2=x2, y2=y2, label="FACE", score=score))
                if mp_boxes:
                    faces = _merge_face_boxes(faces, mp_boxes, iou_thr=0.35)
    except Exception:
        pass

    try:
        person_boxes = [d for d in detections if str(d.label).lower() == "person"]
        if person_boxes:
            W, H = img.size
            roi_faces: list[ImgDetectBox] = []
            for pb in person_boxes:
                x1 = int(max(0, min(W, pb.x1)))
                y1 = int(max(0, min(H, pb.y1)))
                x2 = int(max(0, min(W, pb.x2)))
                y2 = int(max(0, min(H, pb.y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                try:
                    if 'mp' in globals() and mp is not None:
                        np_crop = np.array(crop)
                        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:  # type: ignore
                            res = fd.process(np_crop)
                            if res and getattr(res, 'detections', None):
                                cw, ch = crop.size
                                for det in res.detections:
                                    rb = det.location_data.relative_bounding_box
                                    if rb is None:
                                        continue
                                    fx1 = float(max(0.0, rb.xmin) * cw)
                                    fy1 = float(max(0.0, rb.ymin) * ch)
                                    fx2 = float(min(1.0, rb.xmin + rb.width) * cw)
                                    fy2 = float(min(1.0, rb.ymin + rb.height) * ch)
                                    sc = float(det.score[0]) if getattr(det, 'score', None) else 0.0
                                    roi_faces.append(ImgDetectBox(x1=x1+fx1, y1=y1+fy1, x2=x1+fx2, y2=y1+fy2, label="FACE", score=sc))
                except Exception:
                    pass
            if roi_faces:
                faces = _merge_face_boxes(faces, roi_faces, iou_thr=0.35)
    except Exception:
        pass

    try:
        y_plate = YOLO("yolov8n-license-plate.pt")
        p_res = y_plate.predict(img, verbose=False)[0]
        for b in p_res.boxes:
            plates.append(_to_box(b.xyxy[0].tolist(), "PLATE", float(b.conf)))
    except Exception:
        pass

    try:
        import numpy as _np
        np_img = _np.array(img)
        def _append_text_box(xs, ys, txt, conf):
            try:
                analyzer = None
                language = "en"
                model_key = "spacy/en_core_web_lg"
                app_state = getattr(APP_REF, "state", None)
                cache = getattr(app_state, "analyzer_cache", {}) if app_state is not None else {}
                cache_key = f"{model_key}|{language}"
                analyzer = cache.get(cache_key)
                if analyzer is None:
                    from backend.text_routes import _create_analyzer_for_model  # lazy import to avoid cycles
                    analyzer, _ = _create_analyzer_for_model(model_key, language)
                    if analyzer is not None and app_state is not None:
                        cache[cache_key] = analyzer
                        app_state.analyzer_cache = cache
                ents = []
                scores = []
                if analyzer is not None:
                    res = analyzer.analyze(text=str(txt), language=language)
                    for r in res:
                        ents.append(r.entity_type)
                        scores.append(r.score)
                # Uncertainty via language-model IG+MC (same as TextTest)
                try:
                    from backend.text_routes import _get_lm, _integrated_gradients_saliency  # type: ignore
                    bundle = _get_lm("distilgpt2")
                    lm_uncert = None
                    if bundle is not None:
                        tok, mdl = bundle
                        tokens = _integrated_gradients_saliency(tok, mdl, str(txt), samples=8, ig_steps=8, max_input_tokens=128, seed=42)
                        if tokens:
                            vals = [max(float(t.score_mean or 0.0), float(t.score_std or 0.0)) for t in tokens]
                            import numpy as _np
                            lm_uncert = float(_np.mean(_np.array(vals, dtype=_np.float32)))
                    rule_uncert = float(1.0 - max(scores) if scores else 1.0)
                    uncert = float(lm_uncert if lm_uncert is not None else rule_uncert)
                except Exception:
                    uncert = float(1.0 - max(scores) if scores else 1.0)
            except Exception:
                ents, scores, uncert = [], [], None
            bx1, by1, bx2, by2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            for ob in ocr_boxes:
                if _rect_iou(bx1, by1, bx2, by2, ob.x1, ob.y1, ob.x2, ob.y2) > 0.6:
                    return
            ocr_boxes.append(ImgOCRBox(
                x1=bx1, y1=by1, x2=bx2, y2=by2,
                label="TEXT", score=float(conf), text=str(txt), pii_types=ents or None, pii_scores=scores or None, uncert=uncert
            ))

        paddle = _get_paddle_ocr(getattr(APP_REF, "state", None))
        if paddle is not None:
            p_res = paddle.predict(np_img)
            for lines in p_res:
                for ln in lines:
                    try:
                        if isinstance(ln, dict):
                            pts = ln.get('points') or ln.get('det_polygons') or ln.get('box') or ln.get('bbox') or []
                            txt = ln.get('text') or ln.get('rec_text') or ""
                            conf = float(ln.get('score') or ln.get('rec_score') or 0.0)
                        else:
                            pts = ln[0]
                            txt = ln[1][0]
                            conf = float(ln[1][1])
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        if txt:
                            _append_text_box(xs, ys, txt, conf)
                    except Exception:
                        continue
        reader = _get_ocr_reader(getattr(APP_REF, "state", None))
        if reader is not None:
            H, W = np_img.shape[0], np_img.shape[1]
            ratio = max(1.0, min(3.0, 1600.0 / float(max(W, H)) ))
            results = reader.readtext(
                np_img,
                detail=1,
                decoder='beamsearch',
                beamWidth=10,
                rotation_info=[90, 180, 270],
                canvas_size=2560,
                mag_ratio=ratio,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.4,
                low_text=0.3,
                link_threshold=0.4,
            )
            for box, txt, conf in results:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                if txt:
                    _append_text_box(xs, ys, txt, conf)

        if plates:
            for pl in plates:
                x1 = int(max(0, min(np_img.shape[1], pl.x1)))
                y1 = int(max(0, min(np_img.shape[0], pl.y1)))
                x2 = int(max(0, min(np_img.shape[1], pl.x2)))
                y2 = int(max(0, min(np_img.shape[0], pl.y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = np_img[y1:y2, x1:x2]
                if paddle is not None:
                    try:
                        p_res = paddle.predict(crop)
                        for lines in p_res:
                            for ln in lines:
                                try:
                                    if isinstance(ln, dict):
                                        pts = ln.get('points') or ln.get('det_polygons') or ln.get('box') or ln.get('bbox') or []
                                        txt = ln.get('text') or ln.get('rec_text') or ""
                                        conf = float(ln.get('score') or ln.get('rec_score') or 0.0)
                                    else:
                                        pts = ln[0]
                                        txt = ln[1][0]
                                        conf = float(ln[1][1])
                                    xs = [x1 + p[0] for p in pts]
                                    ys = [y1 + p[1] for p in pts]
                                    if txt:
                                        _append_text_box(xs, ys, txt, conf)
                                except Exception:
                                    continue
                    except Exception:
                        pass
                if reader is not None:
                    try:
                        r_res = reader.readtext(
                            crop,
                            detail=1,
                            decoder='beamsearch',
                            beamWidth=12,
                            rotation_info=[90, 180, 270],
                            canvas_size=2560,
                            mag_ratio=2.5,
                            contrast_ths=0.05,
                            adjust_contrast=0.7,
                        )
                        for box, txt, conf in r_res:
                            xs = [x1 + p[0] for p in box]
                            ys = [y1 + p[1] for p in box]
                            if txt:
                                _append_text_box(xs, ys, txt, conf)
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        clip_model, clip_processor = _get_clip_bundle(getattr(APP_REF, "state", None))
        clip_top = None  # do not match tags for now

        def _probs(opts: list[str]) -> np.ndarray:
            with torch.no_grad():
                out = clip_model(**clip_processor(text=opts, images=img, return_tensors="pt", padding=True))
            lg = out.logits_per_image[0].cpu().numpy()
            ex = np.exp(lg - lg.max())
            return ex / ex.sum()

        clip_probs: dict[str, list[float]] = {}
        clip_labels: dict[str, list[str]] = {}

        labels_io = ["an indoor scene", "an outdoor scene"]
        io_probs = _probs(labels_io)
        clip_probs["indoor_outdoor"] = io_probs.tolist()
        clip_labels["indoor_outdoor"] = labels_io
        p_outdoor = float(io_probs[1])

        labels_landmark = [
            "a famous landmark", "a generic indoor room", "a close-up object", "a random photo without a distinctive location"
        ]
        lm_probs = _probs(labels_landmark)
        clip_probs["landmark_set"] = lm_probs.tolist()
        clip_labels["landmark_set"] = labels_landmark
        p_landmark = float(lm_probs[0])

        # fill clip_top using top-1 label of each prompt set
        try:
            clip_top_list: list[str] = []
            for k in ["indoor_outdoor", "landmark_set"]:
                if k in clip_labels and k in clip_probs:
                    lbls = clip_labels[k]
                    probs = clip_probs[k]
                    if isinstance(lbls, list) and isinstance(probs, list) and len(lbls) == len(probs) and len(lbls) > 0:
                        import numpy as _np
                        idx = int(_np.argmax(_np.array(probs, dtype=_np.float32)))
                        clip_top_list.append(str(lbls[idx]))
            if clip_top_list:
                clip_top = clip_top_list
        except Exception:
            pass

        # Integrated Gradients with SmoothGrad for top-1 prompt logit
        pixel_heat = None
        try:
            import numpy as _np
            texts_all = labels_io + labels_landmark
            tin = clip_processor(text=texts_all, return_tensors="pt", padding=True)
            pv0 = clip_processor(images=img, return_tensors="pt")["pixel_values"]
            baseline = torch.zeros_like(pv0)
            with torch.no_grad():
                out0 = clip_model(input_ids=tin["input_ids"], attention_mask=tin.get("attention_mask", None), pixel_values=pv0)
                target_i = int(_np.argmax(out0.logits_per_image[0].cpu().numpy()))

            n_steps = 12
            n_samples = 6
            sigma = 0.08
            attr_accum = torch.zeros_like(pv0)
            for _ in range(n_samples):
                noise = sigma * torch.randn_like(pv0)
                pv_noisy = (pv0 + noise).clamp(-3.0, 3.0)
                total_grad = torch.zeros_like(pv0)
                for k in range(1, n_steps + 1):
                    alpha = float(k) / float(n_steps)
                    pv = (baseline + alpha * (pv_noisy - baseline)).detach().requires_grad_(True)
                    out = clip_model(input_ids=tin["input_ids"], attention_mask=tin.get("attention_mask", None), pixel_values=pv)
                    score = out.logits_per_image[0, target_i]
                    clip_model.zero_grad(set_to_none=True)
                    if pv.grad is not None:
                        pv.grad.zero_()
                    score.backward(retain_graph=True)
                    total_grad = total_grad + pv.grad.detach()
                ig = (pv_noisy - baseline) * (total_grad / float(n_steps))
                attr_accum = attr_accum + ig
            attr = attr_accum / float(n_samples)
            sal = attr[0].abs().sum(dim=0)
            sal = sal / (sal.max() + 1e-8)
            pixel_heat = sal.detach().cpu().numpy().tolist()
        except Exception:
            pixel_heat = None
    except Exception:
        logger.exception("CLIP analysis failed")
        clip_top = None
        p_outdoor = None
        p_landmark = None
        pixel_heat = None
        clip_probs = None
        clip_labels = None

    geo_candidates: list[GeoCandidate] | None = None
    geo_uncertainty: ImgAnalysisResponse.GeoUncertainty | None = None
    _idx_used: bool = False
    try:
        idx = _get_landmark_index(getattr(APP_REF, "state", None))
        if idx is not None:
            clip_model, clip_processor = _get_clip_bundle(getattr(APP_REF, "state", None))
            inputs = clip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            q = feat.cpu().numpy().astype(np.float32)
            D, I = idx["index"].search(q, 5)
            geo_candidates = []
            for score, iid in zip(D[0].tolist(), I[0].tolist()):
                if iid < 0 or iid >= len(idx["names"]):
                    continue
                geo_candidates.append(GeoCandidate(name=idx["names"][iid], score=float(score), source="faiss-clip", path=idx["paths"][iid]))

            import numpy as _np
            crops = _tta_images(img, n=12)
            votes: dict[int, int] = {}
            for patch in crops:
                inputs_t = clip_processor(images=patch, return_tensors="pt")
                with torch.no_grad():
                    f = clip_model.get_image_features(**inputs_t)
                f = f / f.norm(dim=-1, keepdim=True)
                q_t = f.cpu().numpy().astype(np.float32)
                d_t, i_t = idx["index"].search(q_t, 1)
                top_i = int(i_t[0][0])
                votes[top_i] = votes.get(top_i, 0) + 1
            counts = _np.array(list(votes.values()), dtype=_np.float32)
            T = float(counts.sum()) if counts.size > 0 else 1.0
            probs = counts / T
            eps = 1e-12
            ent = float(-_np.sum(probs * _np.log(probs + eps)))
            ent_norm = float(ent / (np.log(len(probs)) if len(probs) > 1 else 1.0)) if len(probs) > 1 else 0.0
            consistency = float(probs.max()) if probs.size > 0 else 0.0
            margin = float((D[0][0] - D[0][1]) if len(D[0]) > 1 else D[0][0])
            geo_uncertainty = ImgAnalysisResponse.GeoUncertainty(consistency=consistency, entropy_norm=ent_norm, margin=margin)
            _idx_used = True
    except Exception:
        geo_candidates = None
        geo_uncertainty = None

    try:
        if 'yolo' in locals():
            for d in detections:
                heat_boxes.append(ImgUncertaintyBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2, score=max(0.05, 1.0 - d.score)))
    except Exception:
        pass

    if exif:
        lat, lon = _gps_to_latlon(exif)
        if lat is not None and lon is not None:
            exif = dict(exif)
            exif["GPSLatitudeDecimal"] = lat
            exif["GPSLongitudeDecimal"] = lon

    if geo_uncertainty is None:
        # compute TTA-based stability without using any external index
        try:
            clip_model, clip_processor = _get_clip_bundle(getattr(APP_REF, "state", None))
            crops = _tta_images(img, n=10)
            with torch.no_grad():
                base_f = clip_model.get_image_features(**clip_processor(images=img, return_tensors="pt"))
            base_f = base_f / base_f.norm(dim=-1, keepdim=True)
            sims = []
            for patch in crops:
                with torch.no_grad():
                    f = clip_model.get_image_features(**clip_processor(images=patch, return_tensors="pt"))
                f = f / f.norm(dim=-1, keepdim=True)
                sim = float((base_f @ f.T)[0, 0].cpu().numpy())
                sims.append(sim)
            import numpy as _np
            sims = _np.array(sims, dtype=_np.float32)
            mean_sim = float(_np.mean(sims)) if sims.size else 0.0
            std_sim = float(_np.std(sims)) if sims.size else 0.0
            consistency = max(0.0, min(1.0, (mean_sim + 1.0) / 2.0))
            entropy_norm = max(0.0, min(1.0, std_sim / 0.3))
            geo_uncertainty = ImgAnalysisResponse.GeoUncertainty(consistency=consistency, entropy_norm=entropy_norm, margin=0.0)
        except Exception:
            pass

    geo_prob: float | None = None
    try:
        if geo_uncertainty is not None:
            # database-free geolocatability: combine CLIP cues + TTA stability
            consistency = max(0.0, min(1.0, float(geo_uncertainty.consistency or 0.0)))
            stability = max(0.0, min(1.0, 1.0 - float(geo_uncertainty.entropy_norm or 0.0)))
            po = float(p_outdoor) if p_outdoor is not None else 0.5
            pl = float(p_landmark) if p_landmark is not None else 0.3
            base = (0.35 * po + 0.35 * pl + 0.15 * consistency + 0.15 * stability)
            area = float(img.size[0] * img.size[1])
            face_area = sum((f.x2 - f.x1) * (f.y2 - f.y1) for f in faces)
            face_frac = min(1.0, face_area / area) if area > 0 else 0.0
            plate_bonus = min(0.2, 0.05 * len(plates))
            text_bonus = min(0.2, 0.02 * len(ocr_boxes))
            face_penalty = min(0.5, 1.0 * face_frac)
            geo_prob = max(0.0, min(1.0, base + plate_bonus + text_bonus - face_penalty))
    except Exception:
        geo_prob = None

    return ImgAnalysisResponse(
        width=width,
        height=height,
        exif=exif or None,
        clip_top=clip_top,
        clip_probs=clip_probs,
        clip_labels=clip_labels,
        detections=detections,
        faces=faces,
        plates=plates,
        ocr_boxes=ocr_boxes,
        heat_boxes=heat_boxes,
        geo_candidates=geo_candidates,
        geo_uncertainty=geo_uncertainty,
        geo_prob=geo_prob,
        pixel_heat=pixel_heat,
    )


def register_image_routes(app: FastAPI):
    global APP_REF
    APP_REF = app
    app.include_router(router)


