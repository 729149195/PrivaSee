#!/usr/bin/env python3
"""
Standalone Text-to-Speech script using Coqui TTS (XTTS v2).

Features:
- Multilingual synthesis (default: Chinese "zh")
- Optional voice cloning via a reference WAV
- Automatic device selection (CUDA, MPS, or CPU)
- Saves output as a WAV file

Example usages:
  python text_to_wav.py --text "你好，世界" --out ./output.wav --language zh
  python text_to_wav.py --text-file ./input.txt --out ./output.wav --language en
  python text_to_wav.py --text "示例" --out ./output.wav --speaker-wav ./voice.wav --language zh
"""

import argparse
import os
import sys
from typing import Optional


def _resolve_device() -> str:
    """Return the best available device: 'cuda' > 'mps' > 'cpu'."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def _ensure_parent_dir(file_path: str) -> None:
    """Create parent directory for file_path if it does not exist."""
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def synthesize_text_to_wav(
    text: str,
    output_wav_path: str,
    language: str = "zh",
    speaker_wav: Optional[str] = None,
    speaker: Optional[str] = None,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    seed: Optional[int] = None,
) -> None:
    """Synthesize speech from text into a WAV file using Coqui TTS XTTS v2.

    Args:
        text: Input text to synthesize.
        output_wav_path: Destination WAV file path.
        language: Language code (e.g., 'zh', 'en', 'ja', 'ko', 'de', etc.).
        speaker_wav: Optional path to a reference speaker WAV for voice cloning.
        speaker: Optional speaker id/name for multi-speaker models.
        model_name: Coqui TTS model name.
        seed: Optional random seed for reproducibility.
    """
    # 为 macOS/CPU/GPU 自动选择设备
    device = _resolve_device()

    # 延迟导入，便于给出清晰的错误提示
    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Coqui TTS is not installed. Install with: pip install TTS"
        ) from exc

    # 兼容 PyTorch 2.6+ 默认 weights_only=True 导致的反序列化安全限制
    # 1) 允许 XTTS 所需的配置类被反序列化
    # 2) 将 torch.load 的默认 weights_only 设置为 False（仅在未显式传入时）
    try:
        import torch  # type: ignore
        from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
        # 可选：部分 TTS 版本会从模型模块导入 XttsAudioConfig
        xtts_audio_cfg = None
        try:
            from TTS.tts.models.xtts import XttsAudioConfig  # type: ignore

            xtts_audio_cfg = XttsAudioConfig
        except Exception:
            xtts_audio_cfg = None

        serialization = getattr(torch, "serialization", None)
        if serialization is not None and hasattr(serialization, "add_safe_globals"):
            to_allow = [XttsConfig]
            if xtts_audio_cfg is not None:
                to_allow.append(xtts_audio_cfg)
            serialization.add_safe_globals(to_allow)

        # 最小侵入：仅当调用方未传入 weights_only 时，默认改为 False
        try:
            _orig_torch_load = torch.load

            def _torch_load_with_weights_only_false(*args, **kwargs):  # type: ignore
                kwargs.setdefault("weights_only", False)
                return _orig_torch_load(*args, **kwargs)

            torch.load = _torch_load_with_weights_only_false  # type: ignore
        except Exception:
            pass
    except Exception:
        # 若当前 torch 版本无该 API 或导入失败，保持静默并尝试正常加载
        pass

    tts = TTS(model_name)
    try:
        # move model to best device
        tts = tts.to(device)
    except Exception:
        # fallback silently to default device if not supported
        pass

    _ensure_parent_dir(output_wav_path)

    # 若未提供 speaker_wav 且未提供 speaker，尝试使用模型自带的 speakers 列表
    if speaker_wav is None and (speaker is None or str(speaker).strip() == ""):
        try:
            candidate_speakers = getattr(tts, "speakers", None)
            if isinstance(candidate_speakers, (list, tuple)) and len(candidate_speakers) > 0:
                speaker = str(candidate_speakers[0])
        except Exception:
            pass

    # 如果依然无法确定 speaker 且没有 speaker_wav，则给出清晰错误（XTTS v2 通常建议提供 speaker_wav 以进行音色克隆）
    if speaker_wav is None and (speaker is None or str(speaker).strip() == ""):
        raise RuntimeError(
            "Model is multi-speaker. Please provide --speaker-wav or --speaker. "
            "For XTTS v2, --speaker-wav (voice cloning) is recommended."
        )

    # tts_to_file will create/overwrite the file at output_wav_path
    tts.tts_to_file(
        text=text,
        file_path=os.path.abspath(output_wav_path),
        speaker_wav=speaker_wav,
        speaker=speaker,
        language=language,
        split_sentences=True,
        speed=1.0,
        # XTTS v2 supports setting a seed via kwargs in newer TTS versions
        # but if unsupported this will be ignored gracefully
        **({"seed": seed} if seed is not None else {}),
    )


def _read_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert input text to WAV using Coqui TTS XTTS v2.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Inline text to synthesize.",
    )
    input_group.add_argument(
        "--text-file",
        type=str,
        help="Path to a UTF-8 text file to synthesize.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Language code, e.g., zh | en | ja | ko | de | es | fr | it | pt | ru | ar | cs | pl | tr | vi.",
    )
    parser.add_argument(
        "--speaker-wav",
        type=str,
        default=None,
        help="Optional reference speaker WAV for voice cloning.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Optional speaker id/name when using a multi-speaker model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui TTS model name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible synthesis.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.text is not None:
        input_text = args.text
    else:
        if not os.path.exists(args["text_file"] if isinstance(args, dict) else args.text_file):
            print("Text file does not exist.", file=sys.stderr)
            return 2
        input_text = _read_text_from_file(args.text_file)

    if not input_text or not input_text.strip():
        print("Input text is empty.", file=sys.stderr)
        return 2

    try:
        synthesize_text_to_wav(
            text=input_text.strip(),
            output_wav_path=args.out,
            language=args.language,
            speaker_wav=args.speaker_wav,
            speaker=args.speaker,
            model_name=args.model,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Synthesis failed: {exc}", file=sys.stderr)
        return 1

    print(os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


