from __future__ import annotations

import gc
import os
import subprocess
from pathlib import Path

import torch
from pydub import AudioSegment
from pydub.effects import normalize

from modules.state import state


# ============================================================
# PATHS
# ============================================================

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_AUDIO_WAV = AUDIO_DIR / "original_mix_source.wav"
FINAL_AUDIO_WAV = AUDIO_DIR / "final_audio.wav"


# ============================================================
# MIX SETTINGS
# ============================================================
BACKGROUND_GAIN_DB = 1.5
DUB_GAIN_DB = -3.0
RAW_FALLBACK_GAIN_DB = -10.0
DUB_NORMALIZE_HEADROOM_DB = 3.0
FINAL_PEAK_DBFS = -1.0
# ============================================================
# PROCESS HELPERS
# ============================================================
def _run(
    cmd: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
    )
def _probe_duration_seconds(media_path: str) -> float:
    try:
        res = _run([
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path,
        ])

        return max(0.0, float(res.stdout.strip()))

    except Exception:
        return 0.0

def _extract_original_audio(
    video_path: str,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    res = _run([
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i", video_path,
        "-vn",
        "-ac", "2",
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ], check=False)

    if (
        res.returncode != 0
        or not output_path.exists()
    ):
        raise RuntimeError(
            "FFmpeg nie zdołał wyciągnąć oryginalnego audio: "
            + (
                res.stderr[-1000:]
                if res.stderr
                else "unknown error"
            )
        )

    return str(output_path)


def _resolve_separator_output(
    raw_path: str,
) -> str | None:
    if not raw_path:
        return None

    candidates = [
        Path(raw_path),
        Path.cwd() / raw_path,
        AUDIO_DIR / raw_path,
    ]

    for candidate in candidates:
        try:
            candidate = candidate.resolve()

            if (
                candidate.is_file()
                and candidate.stat().st_size > 1000
            ):
                return str(candidate)

        except Exception:
            pass

    return None


def _find_background_file(
    separated: list[str],
) -> str | None:
    resolved = []

    for raw in separated or []:
        resolved_path = _resolve_separator_output(raw)

        if resolved_path:
            resolved.append(resolved_path)

    if not resolved:
        return None

    preferred_keywords = (
        "instrumental",
        "no_vocals",
        "no-vocals",
        "no vocals",
        "accompaniment",
        "karaoke",
    )

    for item in resolved:
        filename = Path(item).name.lower()

        if any(
            keyword in filename
            for keyword in preferred_keywords
        ):
            return item

    # Avoid selecting a vocal stem if possible.
    non_vocal_named = [
        item
        for item in resolved
        if "vocal" not in Path(item).name.lower()
        and "voice" not in Path(item).name.lower()
    ]

    if non_vocal_named:
        return non_vocal_named[0]

    if len(resolved) >= 2:
        return resolved[1]

    return resolved[0]


def _fit_audio_to_duration(
    audio: AudioSegment,
    target_ms: int,
) -> AudioSegment:
    if target_ms <= 0:
        return audio

    if len(audio) < target_ms:
        audio += AudioSegment.silent(
            duration=target_ms - len(audio),
            frame_rate=audio.frame_rate,
        )

    return audio[:target_ms]


def _prepare_dub(
    dub: AudioSegment,
) -> AudioSegment:
    if len(dub) == 0 or dub.rms == 0:
        return dub

    try:
        dub = normalize(
            dub,
            headroom=DUB_NORMALIZE_HEADROOM_DB,
        )
    except Exception:
        pass

    return dub.apply_gain(DUB_GAIN_DB)


def _protect_final_peaks(
    audio: AudioSegment,
) -> AudioSegment:

    if len(audio) == 0:
        return audio

    try:
        peak = audio.max_dBFS

        if peak > FINAL_PEAK_DBFS:
            reduction = FINAL_PEAK_DBFS - peak

            state.add_log(
                f"  🎚️ Peak protection: {reduction:.1f} dB "
                f"(peak {peak:.1f} → {FINAL_PEAK_DBFS:.1f} dBFS)"
            )

            audio = audio.apply_gain(reduction)

    except Exception:
        pass

    return audio

# ============================================================
# MIXING
# ============================================================
def mix_with_background(
    video_path: str,
    dub_path: str,
    keep_bg: bool = True,
) -> str:

    if not os.path.exists(dub_path):
        raise FileNotFoundError(
            f"Brak pliku dubbingu: {dub_path}"
        )

    duration_sec = _probe_duration_seconds(
        video_path
    )

    target_ms = (
        int(round(duration_sec * 1000))
        if duration_sec > 0
        else 0
    )

    dub = AudioSegment.from_file(dub_path)
    dub = _prepare_dub(dub)

    if target_ms > 0:
        dub = _fit_audio_to_duration(
            dub,
            target_ms,
        )

    # --------------------------------------------------------
    # DUB ONLY
    # --------------------------------------------------------

    if not keep_bg:
        state.add_log(
            "🎤 Tło wyłączone — eksportuję sam dubbing."
        )

        final = dub

    # --------------------------------------------------------
    # BACKGROUND + DUB
    # --------------------------------------------------------

    else:
        state.add_log(
            "🎵 Mix filmowy: tło pozostaje aktywne również "
            "podczas wypowiedzi TTS."
        )

        original_audio_path = _extract_original_audio(
            video_path,
            ORIGINAL_AUDIO_WAV,
        )

        original = AudioSegment.from_file(
            original_audio_path
        )

        if target_ms > 0:
            original = _fit_audio_to_duration(
                original,
                target_ms,
            )

        background = None
        separator = None

        try:
            from audio_separator.separator import Separator

            separator = Separator()

            separator.load_model(
                model_filename="UVR-MDX-NET-Inst_HQ_3.onnx"
            )

            separated = separator.separate(
                original_audio_path
            )

            state.add_log(
                "  🔎 UVR outputs: "
                + ", ".join(
                    str(x)
                    for x in (separated or [])
                )
            )

            bg_path = _find_background_file(
                separated
            )

            if not bg_path:
                raise RuntimeError(
                    "Nie znaleziono stema instrumental/no_vocals."
                )

            state.add_log(
                f"  ✅ Używam background stem: {bg_path}"
            )

            background = AudioSegment.from_file(
                bg_path
            )

            if target_ms > 0:
                background = _fit_audio_to_duration(
                    background,
                    target_ms,
                )

            # Keep music/SFX loud enough even while dialogue is present.
            background = background.apply_gain(
                BACKGROUND_GAIN_DB
            )

            state.add_log(
                f"  🎼 Background gain: "
                f"{BACKGROUND_GAIN_DB:+.1f} dB"
            )

        except Exception as e:
            state.add_log(
                "  ⚠️ Separacja UVR nie powiodła się: "
                f"{e}"
            )

            state.add_log(
                "  ↪️ Fallback: pełny oryginalny soundtrack "
                f"{RAW_FALLBACK_GAIN_DB:+.1f} dB pod dubbingiem."
            )

            background = original.apply_gain(
                RAW_FALLBACK_GAIN_DB
            )

        finally:
            if separator is not None:
                try:
                    del separator
                except Exception:
                    pass

            gc.collect()

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        final = background.overlay(
            dub,
            position=0,
        )

        state.add_log(
            f"  🗣️ Dub gain: {DUB_GAIN_DB:+.1f} dB"
        )

    # --------------------------------------------------------
    # FINAL LENGTH + PEAK SAFETY
    # --------------------------------------------------------

    if target_ms > 0:
        final = _fit_audio_to_duration(
            final,
            target_ms,
        )

    final = _protect_final_peaks(final)

    FINAL_AUDIO_WAV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.export(
        FINAL_AUDIO_WAV,
        format="wav",
    )

    state.add_log(
        f"  ✅ Finalny miks audio → "
        f"{FINAL_AUDIO_WAV} "
        f"({len(final) / 1000:.2f}s, "
        f"background={BACKGROUND_GAIN_DB:+.1f} dB, "
        f"dub={DUB_GAIN_DB:+.1f} dB)"
    )

    return str(FINAL_AUDIO_WAV)


# ============================================================
# FINAL VIDEO
# ============================================================

def create_final_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    hardsub: bool = False,
    subtitles_path: str | None = None,
) -> str:
    state.add_log(
        f"🎬 Generowanie końcowego pliku wideo: "
        f"{output_path}"
    )

    out_dir = os.path.dirname(
        os.path.abspath(output_path)
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    duration_sec = _probe_duration_seconds(
        video_path
    )

    if (
        hardsub
        and subtitles_path
        and os.path.exists(subtitles_path)
    ):
        state.add_log(
            "  🔥 Nakładanie napisów na obraz (Hardsub)..."
        )

        escaped_sub = (
            os.path.abspath(subtitles_path)
            .replace("\\", "/")
            .replace(":", "\\:")
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i", video_path,
            "-i", audio_path,
            "-vf", f"subtitles='{escaped_sub}'",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]

    else:
        state.add_log(
            "  🎞️ Kopiowanie obrazu wideo "
            "bez ponownego kodowania..."
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]

    if duration_sec > 0:
        cmd.extend([
            "-t",
            f"{duration_sec:.6f}",
        ])

    cmd.extend([
        "-movflags",
        "+faststart",
        output_path,
    ])

    try:
        _run(cmd)

        state.add_log(
            "  ✅ Końcowy mux FFmpeg zakończony sukcesem."
        )

        return output_path

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr or str(e)

        state.add_log(
            f"  ❌ FFmpeg error: {err_msg[-1500:]}"
        )

        raise RuntimeError(
            f"FFmpeg failed: {err_msg}"
        )
