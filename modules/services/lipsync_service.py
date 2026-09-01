from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from modules.state import state


WAV2LIP_CHECKPOINT = "Wav2Lip/wav2lip_gan.pth"
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def is_lipsync_available() -> bool:
    return os.path.exists(WAV2LIP_CHECKPOINT)


def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
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
        if res.returncode == 0:
            return max(0.0, float(res.stdout.strip()))
    except Exception:
        pass
    return 0.0


def _parse_pads(pads: str | list | tuple) -> list[str]:
    if isinstance(pads, str):
        raw = pads.replace(",", " ").split()
    else:
        raw = list(pads)

    values = []
    for item in raw[:4]:
        try:
            values.append(str(int(item)))
        except (TypeError, ValueError):
            values.append("0")

    while len(values) < 4:
        values.append("0")

    return values


def _prepare_driver_audio(
    audio_path: str,
    video_path: str,
    output_path: str,
) -> str:
    """
    Wav2Lip only needs 16 kHz mono audio for mel generation.
    Pad it to the video duration so inference.py never receives a shorter
    audio timeline than the source picture.
    """
    duration = _probe_duration_seconds(video_path)
    if duration <= 0:
        raise RuntimeError("Nie udało się odczytać długości wideo.")

    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", audio_path,
        "-af", "apad",
        "-t", f"{duration:.6f}",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_path,
    ]

    res = _run(cmd)
    if res.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(
            "Nie udało się przygotować WAV dla Wav2Lip: "
            + (res.stderr[-1000:] if res.stderr else "unknown error")
        )

    return output_path


def _remux_approved_audio(
    lip_video_path: str,
    approved_audio_path: str,
    source_video_path: str,
    output_path: str,
) -> str:
    """
    Replace whatever audio Wav2Lip muxed with the exact approved final mix.
    """
    duration = _probe_duration_seconds(source_video_path)

    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", lip_video_path,
        "-i", approved_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "apad",
    ]

    if duration > 0:
        cmd.extend(["-t", f"{duration:.6f}"])

    cmd.extend([
        "-movflags", "+faststart",
        output_path,
    ])

    res = _run(cmd)

    if (
        res.returncode != 0
        or not os.path.exists(output_path)
        or os.path.getsize(output_path) < 10000
    ):
        raise RuntimeError(
            "Końcowy remux Wav2Lip nie powiódł się: "
            + (res.stderr[-1200:] if res.stderr else "unknown error")
        )

    return output_path


def run_lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    pads: str = "0 10 0 0",
    resize_factor: int = 1,
) -> str:
    if not is_lipsync_available():
        state.add_log(
            f"⚠️ Wav2Lip checkpoint '{WAV2LIP_CHECKPOINT}' nie istnieje. "
            "Pomijam lip-sync."
        )
        return video_path

    if not os.path.exists(video_path):
        state.add_log(f"❌ Wav2Lip: brak wideo: {video_path}")
        return video_path

    if not os.path.exists(audio_path):
        state.add_log(f"❌ Wav2Lip: brak audio: {audio_path}")
        return video_path

    state.add_log(
        "💋 Wav2Lip: synchronizuję usta. Finalny miks audio zostanie "
        "ponownie podpięty po inferencji."
    )

    python_executable = sys.executable or "python3"
    pad_values = _parse_pads(pads)

    driver_audio = str(TEMP_DIR / "wav2lip_driver.wav")
    wav2lip_intermediate = str(TEMP_DIR / "wav2lip_intermediate.mp4")

    try:
        _prepare_driver_audio(
            audio_path=audio_path,
            video_path=video_path,
            output_path=driver_audio,
        )

        cmd = [
            python_executable,
            "Wav2Lip/inference.py",
            "--checkpoint_path", WAV2LIP_CHECKPOINT,
            "--face", video_path,
            "--audio", driver_audio,
            "--outfile", wav2lip_intermediate,
            "--resize_factor", str(max(1, int(resize_factor))),
            "--pads",
            *pad_values,
        ]

        state.add_log(
            "  🚀 Wav2Lip inference: obraz będzie generowany z technicznego "
            "16 kHz WAV o długości filmu."
        )

        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,
        )

        if (
            res.returncode != 0
            or not os.path.exists(wav2lip_intermediate)
            or os.path.getsize(wav2lip_intermediate) < 10000
        ):
            err = (
                res.stderr[-1600:]
                if res.stderr
                else res.stdout[-1600:]
                if res.stdout
                else "Brak komunikatu."
            )
            state.add_log(
                f"  ⚠️ Wav2Lip inference nie powiodło się "
                f"(kod {res.returncode}): {err}"
            )
            return video_path

        state.add_log(
            "  🎚️ Wav2Lip zakończył obraz. Przywracam dokładnie "
            "zaakceptowany final_audio..."
        )

        final_result = _remux_approved_audio(
            lip_video_path=wav2lip_intermediate,
            approved_audio_path=audio_path,
            source_video_path=video_path,
            output_path=output_path,
        )

        state.add_log(
            "  ✅ Wav2Lip gotowy — obraz z lip-sync, audio zachowane z "
            "oryginalnego finalnego miksu."
        )
        return final_result

    except subprocess.TimeoutExpired:
        state.add_log("  ❌ Wav2Lip przekroczył limit czasu.")
        return video_path

    except Exception as e:
        state.add_log(f"  ❌ Wav2Lip wyjątek: {type(e).__name__}: {e}")
        return video_path

    finally:
        for temp_path in (driver_audio, wav2lip_intermediate):
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
