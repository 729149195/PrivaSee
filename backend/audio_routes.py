from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Form
from fastapi.responses import FileResponse, JSONResponse
import uuid as _uuid
import threading as _threading
import tempfile as _tempfile
import os as _os
import time as _time
from pydantic import BaseModel

router = APIRouter()
APP_REF: FastAPI | None = None

try:
    import numpy as _np_audio
    import io as _io_audio
    import math as _math_audio
    try:
        import soundfile as _sf
    except Exception:
        _sf = None
    try:
        import librosa as _librosa
    except Exception:
        _librosa = None
    _AUDIO_AVAILABLE = True
except Exception:  # pragma: no cover
    _AUDIO_AVAILABLE = False


class AudioInterval(BaseModel):
    start_sec: float
    end_sec: float
    label: str
    score: float | None = None


class AudioAnalysisResponse(BaseModel):
    sample_rate: int
    duration_sec: float
    num_channels: int
    waveform_downsample: list[float]
    onsets_sec: list[float] | None = None
    beats_sec: list[float] | None = None
    segments: list[AudioInterval] | None = None
    frame_uncertainty: list[float] | None = None
    transcript: str | None = None


try:
    from faster_whisper import WhisperModel as _WhisperModel  # type: ignore
    _WHISPER_AVAILABLE = True
except Exception:
    _WHISPER_AVAILABLE = False

# Optional TTS (Coqui) for text-to-speech
try:
    from TTS.api import TTS as _CoquiTTS  # type: ignore
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import soundfile as _sf
except Exception:
    _sf = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptResponse(BaseModel):
    text: str
    segments: list[TranscriptSegment]


def _load_audio_bytes(data: bytes) -> tuple[_np_audio.ndarray, int, int]:
    if _sf is not None:
        try:
            with _sf.SoundFile(_io_audio.BytesIO(data)) as f:
                sr = int(f.samplerate)
                ch = int(f.channels)
                y = f.read(always_2d=True, dtype="float32")
                y = _np_audio.asarray(y, dtype=_np_audio.float32)
                y = y.T
                return y, sr, ch
        except Exception:
            pass
    if _librosa is not None:
        try:
            y, sr = _librosa.load(_io_audio.BytesIO(data), sr=None, mono=False)
            if y.ndim == 1:
                ch = 1
            else:
                ch = int(y.shape[0])
            y = _np_audio.asarray(y, dtype=_np_audio.float32)
            return y, int(sr), ch
        except Exception:
            pass
    try:
        import wave as _wave
        with _wave.open(_io_audio.BytesIO(data)) as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            width = wf.getsampwidth()
            n = wf.getnframes()
            raw = wf.readframes(n)
        import struct as _struct
        if width == 2:
            dtype = _np_audio.int16
            fmt = "<" + "h" * (len(raw) // 2)
        elif width == 1:
            dtype = _np_audio.uint8
            fmt = "<" + "B" * len(raw)
        elif width == 4:
            dtype = _np_audio.int32
            fmt = "<" + "i" * (len(raw) // 4)
        else:
            raise ValueError("Unsupported sample width")
        arr = _np_audio.array(_struct.unpack(fmt, raw), dtype=dtype)
        if ch > 1:
            arr = arr.reshape(-1, ch).T
        else:
            arr = arr.reshape(1, -1)
        if arr.dtype == _np_audio.uint8:
            y = (arr.astype(_np_audio.float32) - 128.0) / 128.0
        elif arr.dtype == _np_audio.int16:
            y = arr.astype(_np_audio.float32) / 32768.0
        elif arr.dtype == _np_audio.int32:
            y = arr.astype(_np_audio.float32) / 2147483648.0
        else:
            y = arr.astype(_np_audio.float32)
        return y, int(sr), int(ch)
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted audio file")


def _to_mono(y: _np_audio.ndarray) -> _np_audio.ndarray:
    return y if y.ndim == 1 else _np_audio.mean(y, axis=0)


def _downsample_waveform(y_mono: _np_audio.ndarray, target_points: int = 1024) -> list[float]:
    if y_mono.size == 0:
        return []
    if y_mono.size <= target_points:
        pad = _np_audio.pad(y_mono, (0, target_points - y_mono.size)) if y_mono.size < target_points else y_mono
        return pad.astype(_np_audio.float32)[:target_points].tolist()
    idx = _np_audio.linspace(0, y_mono.size - 1, num=target_points)
    idx0 = _np_audio.floor(idx).astype(_np_audio.int64)
    idx1 = _np_audio.minimum(idx0 + 1, y_mono.size - 1)
    frac = (idx - idx0).astype(_np_audio.float32)
    vals = (1.0 - frac) * y_mono[idx0] + frac * y_mono[idx1]
    return vals.astype(_np_audio.float32).tolist()


def _compute_features_and_uncertainty(y_mono: _np_audio.ndarray, sr: int) -> tuple[list[float], list[float], list[AudioInterval], list[float]]:
    onsets_sec: list[float] = []
    beats_sec: list[float] = []
    segments: list[AudioInterval] = []
    frame_uncertainty: list[float] = []

    if _librosa is not None:
        try:
            onset_frames = _librosa.onset.onset_detect(y=y_mono, sr=sr, units="frames")
            onsets_sec = _librosa.frames_to_time(onset_frames, sr=sr).astype(float).tolist()
        except Exception:
            onsets_sec = []
        try:
            tempo, beat_frames = _librosa.beat.beat_track(y=y_mono, sr=sr)
            beats_sec = _librosa.frames_to_time(beat_frames, sr=sr).astype(float).tolist()
        except Exception:
            beats_sec = []
        try:
            intervals = _librosa.effects.split(y_mono, top_db=30)
            for s, e in intervals:
                start = float(s) / float(sr)
                end = float(e) / float(sr)
                rms = float(_np_audio.sqrt(_np_audio.mean(_np_audio.square(y_mono[s:e] + 1e-8))))
                segments.append(AudioInterval(start_sec=start, end_sec=end, label="ACTIVE", score=rms))
        except Exception:
            segments = []
        try:
            n_fft = 1024
            hop = 512
            S = _np_audio.abs(_librosa.stft(y=y_mono, n_fft=n_fft, hop_length=hop)) ** 2
            P = S / (S.sum(axis=0, keepdims=True) + 1e-12)
            H = -_np_audio.sum(P * _np_audio.log(P + 1e-12), axis=0)
            H = H / _np_audio.log(P.shape[0])
            frame_uncertainty = _downsample_waveform(H.astype(_np_audio.float32), target_points=1024)
        except Exception:
            frame_uncertainty = []
    else:
        try:
            win = max(1, int(sr * 0.02))
            hop = max(1, int(sr * 0.01))
            frames = []
            for i in range(0, y_mono.size - win + 1, hop):
                f = y_mono[i:i+win]
                energy = float(_np_audio.mean(_np_audio.square(f)))
                frames.append(energy)
            fr = _np_audio.asarray(frames, dtype=_np_audio.float32)
            thr = float(fr.mean() + 0.5 * fr.std()) if fr.size else 0.0
            in_seg = False
            start_i = 0
            for i, v in enumerate(fr.tolist()):
                if not in_seg and v > thr:
                    in_seg = True
                    start_i = i
                elif in_seg and v <= thr:
                    in_seg = False
                    s = start_i * hop / sr
                    e = i * hop / sr
                    segments.append(AudioInterval(start_sec=float(s), end_sec=float(e), label="ACTIVE", score=float(v)))
            if fr.size:
                fr_norm = (fr - fr.min()) / (fr.max() - fr.min() + 1e-8)
                frame_uncertainty = _downsample_waveform(fr_norm, target_points=1024)
        except Exception:
            pass
    return onsets_sec, beats_sec, segments, frame_uncertainty


@router.post("/audio/analyze", response_model=AudioAnalysisResponse)
def analyze_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    try:
        data = file.file.read()
        y, sr, ch = _load_audio_bytes(data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {exc}")

    y_mono = _to_mono(y)
    duration_sec = float(y_mono.size) / float(sr) if sr > 0 else 0.0
    waveform_downsample = _downsample_waveform(y_mono, target_points=1024)

    onsets_sec, beats_sec, segments, frame_uncertainty = _compute_features_and_uncertainty(y_mono, sr)

    return AudioAnalysisResponse(
        sample_rate=int(sr),
        duration_sec=float(duration_sec),
        num_channels=int(ch),
        waveform_downsample=waveform_downsample,
        onsets_sec=onsets_sec or None,
        beats_sec=beats_sec or None,
        segments=segments or None,
        frame_uncertainty=frame_uncertainty or None,
        transcript=None,
    )


def _get_whisper_model() -> "_WhisperModel | None":
    if not _WHISPER_AVAILABLE:
        return None
    app_state = getattr(APP_REF, "state", None)
    bundle = getattr(app_state, "whisper_bundle", None) if app_state is not None else None
    if bundle is not None:
        return bundle
    try:
        model = _WhisperModel("small", device="cpu", compute_type="int8")
        if app_state is not None:
            app_state.whisper_bundle = model
        return model
    except Exception:
        return None


class TranscriptResponse(BaseModel):
    text: str
    segments: list[TranscriptSegment]


@router.post("/audio/transcribe", response_model=TranscriptResponse)
def transcribe_audio(file: UploadFile = File(...)):
    model = _get_whisper_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Whisper not available. Install: pip install faster-whisper")
    try:
        data = file.file.read()
        import tempfile, os
        ext = os.path.splitext(getattr(file, "filename", "") or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            segments, info = model.transcribe(tmp.name, vad_filter=True)
            segs: list[TranscriptSegment] = []
            texts: list[str] = []
            for s in segments:
                segs.append(TranscriptSegment(start=float(s.start), end=float(s.end), text=s.text or ""))
                texts.append(s.text or "")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {exc}")
    return TranscriptResponse(text=(" ").join(texts).strip(), segments=segs)


def register_audio_routes(app: FastAPI):
    global APP_REF
    APP_REF = app
    # runtime state for TTS jobs and model cache
    if not hasattr(app.state, "tts_jobs"):
        app.state.tts_jobs = {}
    if not hasattr(app.state, "tts_model"):
        app.state.tts_model = None
    app.include_router(router)



# ====================== Utilities for TTS ======================
def _resolve_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def _prepare_torch_safe_loading():
    try:
        import torch  # type: ignore
        from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
        try:
            from TTS.tts.models.xtts import XttsAudioConfig  # type: ignore
            _audio_cfg = XttsAudioConfig
        except Exception:
            _audio_cfg = None
        serialization = getattr(torch, "serialization", None)
        if serialization is not None and hasattr(serialization, "add_safe_globals"):
            allow = [XttsConfig]
            if _audio_cfg is not None:
                allow.append(_audio_cfg)
            serialization.add_safe_globals(allow)
        try:
            _orig = torch.load
            def _load(*args, **kwargs):  # type: ignore
                kwargs.setdefault("weights_only", False)
                return _orig(*args, **kwargs)
            torch.load = _load  # type: ignore
        except Exception:
            pass
    except Exception:
        pass


def _get_tts_model(app: FastAPI):
    model = getattr(app.state, "tts_model", None)
    if model is not None:
        return model
    if not _TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="TTS not available. Install: pip install TTS")
    _prepare_torch_safe_loading()
    device = _resolve_device()
    tts = _CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
    try:
        tts = tts.to(device)
    except Exception:
        pass
    app.state.tts_model = tts
    return tts


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    import re as _re
    parts = _re.split(r"([。！？!?.；;]\s*)", text)
    segs: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if any(ch in p for ch in "。！？!?.；;"):
            seg = buf.strip()
            if seg:
                segs.append(seg)
            buf = ""
    tail = buf.strip()
    if tail:
        segs.append(tail)
    return [s for s in segs if s.strip()]


def _write_wav(path: str, data: "_np.ndarray", sr: int):
    if _np is None:
        raise RuntimeError("numpy not available")
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    if _sf is not None:
        _sf.write(path, data, sr)
        return
    # fallback to wave (int16 PCM)
    import wave as _wave
    import struct as _struct
    wav = _wave.open(path, "wb")
    try:
        wav.setnchannels(1 if data.ndim == 1 else data.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sr)
        y = data if data.ndim == 1 else data.mean(axis=1)
        y = _np.clip(y, -1.0, 1.0)
        ints = (y * 32767.0).astype(_np.int16)
        wav.writeframes(_struct.pack('<' + 'h' * ints.size, *ints.tolist()))
    finally:
        wav.close()


def _run_tts_job(app: FastAPI, job_id: str, text: str, language: str, speaker: str | None, speaker_wav_path: str | None, out_path: str):
    jobs = app.state.tts_jobs
    jobs[job_id] = {"status": "running", "progress": 0.0, "message": "starting", "output_path": None}
    try:
        tts = _get_tts_model(app)
        segs = _split_sentences(text)
        if not segs:
            segs = [text.strip()]
        total = len(segs)
        acc: list = []
        sr = 22050
        # try to read sample rate from tts synthesizer
        try:
            sr = int(getattr(getattr(tts, 'synthesizer', None), 'output_sample_rate', 22050))
        except Exception:
            sr = 22050
        silence = None
        if _np is not None:
            silence = _np.zeros(int(0.2 * sr), dtype=_np.float32)
        for i, s in enumerate(segs, start=1):
            wav = tts.tts(text=s, speaker_wav=speaker_wav_path, speaker=speaker, language=language)
            # ensure mono float32 numpy
            if _np is not None:
                w = _np.asarray(wav, dtype=_np.float32)
                w = w.flatten()
                acc.append(w)
                if silence is not None and i < total:
                    acc.append(silence)
            jobs[job_id]["progress"] = float(i) / float(total)
            jobs[job_id]["message"] = f"{i}/{total}"
        if _np is None:
            raise RuntimeError("numpy not available for concatenation")
        full = _np.concatenate(acc) if len(acc) > 1 else acc[0]
        _write_wav(out_path, full, sr)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["output_path"] = out_path
        jobs[job_id]["message"] = "finished"
        jobs[job_id]["progress"] = 1.0
    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(exc)
        jobs[job_id]["progress"] = 1.0


# ====================== API: Text-to-Speech ======================
@router.post("/audio/tts/start")
def start_tts(
    text: str = Form(...),
    language: str = Form("zh"),
    speaker: str | None = Form(None),
    speaker_wav: UploadFile | None = File(None),
):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    if not _TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="TTS not available. Install: pip install TTS")
    app = APP_REF
    assert app is not None
    temp_dir = _tempfile.gettempdir()
    out_path = _os.path.join(temp_dir, f"tts_{_uuid.uuid4().hex}.wav")
    ref_path = None
    if speaker_wav is not None:
        suffix = _os.path.splitext(speaker_wav.filename or "ref.wav")[1] or ".wav"
        tf = _tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        ref_path = tf.name
        tf.write(speaker_wav.file.read())
        tf.flush(); tf.close()
    job_id = _uuid.uuid4().hex
    # 初始化任务状态为 queued，先写入再启动线程，避免竞态覆盖完成态
    app.state.tts_jobs[job_id] = {"status": "queued", "progress": 0.0, "message": "queued", "output_path": None}
    th = _threading.Thread(
        target=_run_tts_job,
        args=(app, job_id, text.strip(), language.strip() or "zh", speaker, ref_path, out_path),
        daemon=True,
    )
    th.start()
    return {"job_id": job_id}


@router.get("/audio/tts/status/{job_id}")
def tts_status(job_id: str):
    app = APP_REF
    assert app is not None
    job = app.state.tts_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(job)


@router.get("/audio/tts/result/{job_id}")
def tts_result(job_id: str):
    app = APP_REF
    assert app is not None
    job = app.state.tts_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="job not finished")
    path = job.get("output_path")
    if not path or not _os.path.exists(path):
        raise HTTPException(status_code=404, detail="result not found")
    return FileResponse(path, media_type="audio/wav", filename=_os.path.basename(path))

