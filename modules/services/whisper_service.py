from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from faster_whisper import WhisperModel

from modules.state import state


# ============================================================
# SEGMENTATION CONFIG
# ============================================================

VAD_MIN_SILENCE_MS = 280
VAD_SPEECH_PAD_MS = 100


HARD_GAP_S = 0.55


PUNCT_GAP_S = 0.18


MIN_SEGMENT_DURATION_S = 0.35
MIN_WORDS_BEFORE_PUNCT_SPLIT = 2


SOFT_MAX_SEGMENT_S = 5.5


HARD_MAX_SEGMENT_S = 8.0


MAX_SEGMENT_CHARS = 115

STRONG_PUNCT_RE = re.compile(r"[.!?…。！？]+[\"'”’」』】）)]*$")


# ============================================================
# SMALL COMPATIBILITY OBJECTS
# ============================================================

@dataclass
class DialogueWord:
    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass
class DialogueSegment:


    start: float
    end: float
    text: str
    words: list[DialogueWord] = field(default_factory=list)


# ============================================================
# HELPERS
# ============================================================

def _cleanup_torch() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _normalize_word_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ")


def _join_words(words: Iterable[DialogueWord]) -> str:

    parts = [_normalize_word_text(w.word) for w in words if _normalize_word_text(w.word)]
    if not parts:
        return ""

    text = "".join(parts).strip()


    if " " not in text and len(parts) > 1:
        latin_like = sum(bool(re.search(r"[A-Za-zÀ-ž0-9]", p)) for p in parts)
        if latin_like >= max(2, len(parts) // 2):
            text = " ".join(p.strip() for p in parts).strip()

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _ends_sentence(text: str) -> bool:
    return bool(text and STRONG_PUNCT_RE.search(text.strip()))


def _word_from_faster_whisper(raw_word: Any) -> DialogueWord | None:
    start = getattr(raw_word, "start", None)
    end = getattr(raw_word, "end", None)
    text = _normalize_word_text(getattr(raw_word, "word", ""))

    if start is None or end is None or not text.strip():
        return None

    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None

    if end_f < start_f:
        end_f = start_f

    probability = getattr(raw_word, "probability", None)
    try:
        probability = float(probability) if probability is not None else None
    except (TypeError, ValueError):
        probability = None

    return DialogueWord(
        start=start_f,
        end=end_f,
        word=text,
        probability=probability,
    )


def _segment_from_words(words: list[DialogueWord]) -> DialogueSegment | None:
    if not words:
        return None

    text = _join_words(words)
    if not text:
        return None

    return DialogueSegment(
        start=max(0.0, float(words[0].start)),
        end=max(float(words[0].start), float(words[-1].end)),
        text=text,
        words=list(words),
    )


def _should_split_before_word(
    current_words: list[DialogueWord],
    next_word: DialogueWord,
) -> tuple[bool, str]:

    if not current_words:
        return False, ""

    first = current_words[0]
    previous = current_words[-1]

    gap = max(0.0, next_word.start - previous.end)
    current_duration = max(0.0, previous.end - first.start)
    current_text = _join_words(current_words)


    if gap >= HARD_GAP_S and current_duration >= MIN_SEGMENT_DURATION_S:
        return True, f"pause={gap:.2f}s"

    if (
        _ends_sentence(current_text)
        and gap >= PUNCT_GAP_S
        and len(current_words) >= MIN_WORDS_BEFORE_PUNCT_SPLIT
        and current_duration >= MIN_SEGMENT_DURATION_S
    ):
        return True, f"punctuation+pause={gap:.2f}s"


    if (
        current_duration >= SOFT_MAX_SEGMENT_S
        and _ends_sentence(current_text)
        and len(current_words) >= MIN_WORDS_BEFORE_PUNCT_SPLIT
    ):
        return True, f"soft-max={current_duration:.2f}s"


    projected_duration = max(0.0, next_word.end - first.start)
    if projected_duration >= HARD_MAX_SEGMENT_S:
        return True, f"hard-max={projected_duration:.2f}s"


    if len(current_text) >= MAX_SEGMENT_CHARS and (
        _ends_sentence(current_text) or gap >= 0.12
    ):
        return True, f"chars={len(current_text)}"

    return False, ""


def _resegment_words(words: list[DialogueWord]) -> list[DialogueSegment]:
    if not words:
        return []

    words = sorted(words, key=lambda w: (w.start, w.end))

    result: list[DialogueSegment] = []
    current: list[DialogueWord] = []

    for word in words:
        if current:
            split, _reason = _should_split_before_word(current, word)
            if split:
                segment = _segment_from_words(current)
                if segment is not None:
                    result.append(segment)
                current = []

        current.append(word)

    segment = _segment_from_words(current)
    if segment is not None:
        result.append(segment)

    return result


def _fallback_segment(raw_segment: Any) -> DialogueSegment | None:

    text = str(getattr(raw_segment, "text", "") or "").strip()
    if not text:
        return None

    try:
        start = float(getattr(raw_segment, "start", 0.0) or 0.0)
        end = float(getattr(raw_segment, "end", start) or start)
    except (TypeError, ValueError):
        return None

    return DialogueSegment(start=max(0.0, start), end=max(start, end), text=text)


def _collect_and_resegment(raw_segments: Iterable[Any]) -> list[DialogueSegment]:

    output: list[DialogueSegment] = []

    for raw_segment in raw_segments:
        if state.cancel_flags["transcribe"]:
            break

        raw_words = getattr(raw_segment, "words", None) or []
        words = []

        for raw_word in raw_words:
            converted = _word_from_faster_whisper(raw_word)
            if converted is not None:
                words.append(converted)

        if words:
            rebuilt = _resegment_words(words)
            if rebuilt:
                output.extend(rebuilt)
                continue

        fallback = _fallback_segment(raw_segment)
        if fallback is not None:
            output.append(fallback)

    return output


# ============================================================
# PUBLIC API
# ============================================================

def transcribe_video(
    video_path: str,
    model_size: str = "turbo",
    language: str = "auto",
) -> tuple[list[DialogueSegment], str]:
    state.add_log(
        f"🎤 Inicjalizacja transkrypcji Whisper "
        f"(model={model_size}, dialogue-split=ON)..."
    )

    lang_arg = None if language in ("auto", "", None) else language

    state.add_log(
        "⏳ Transkrypcja: word timestamps + VAD + ponowna segmentacja dialogów..."
    )

    preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
    devices = [preferred_device]
    if preferred_device == "cuda":
        devices.append("cpu")

    last_error = None
    for device in devices:
        model = None
        try:
            _cleanup_torch()
            compute_type = "float16" if device == "cuda" else "int8"
            state.add_log(
                f"  🖥️ Whisper device: {device} ({compute_type})."
            )
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            raw_segments, info = model.transcribe(
                video_path,
                word_timestamps=True,
                language=lang_arg,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
                    "speech_pad_ms": VAD_SPEECH_PAD_MS,
                },
                beam_size=5,
                temperature=0.0,
            )
            rebuilt_segments = _collect_and_resegment(raw_segments)
            detected = getattr(info, "language", None) or language
            state.add_log(
                f"  ✅ Whisper finished: language={detected}, "
                f"dialogue segments={len(rebuilt_segments)}."
            )
            if state.cancel_flags["transcribe"]:
                state.add_log("❌ Transcription cancelled by the user.")
            return rebuilt_segments, detected
        except RuntimeError as exc:
            last_error = exc
            is_cuda_oom = device == "cuda" and (
                "out of memory" in str(exc).lower()
                or "cuda failed" in str(exc).lower()
            )
            if not is_cuda_oom:
                raise
            state.add_log(
                "  ⚠️ Whisper ran out of GPU memory. Releasing CUDA "
                "resources and retrying on CPU (slower but safe)."
            )
        finally:
            if model is not None:
                del model
            _cleanup_torch()

    raise last_error or RuntimeError("Whisper transcription failed")
