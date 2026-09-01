# -*- coding: utf-8 -*-
"""
Blueprint for video operations (upload, load, transcribe, diarization).

Subtitle import fixes:
- preserves .srt/.ass extension even for Japanese/Chinese/non-Latin filenames,
- avoids file-name collisions,
- robustly sniffs subtitle format,
- returns a clear error when parsing yields zero subtitle events.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename

from modules.config import LANGUAGE_MAPPING, UPLOAD_FOLDER
from modules.services.diarization_service import perform_diarization
from modules.services.whisper_service import transcribe_video
from modules.state import state
from modules.utils.file_utils import parse_subtitle_file
from modules.utils.time_utils import format_time, seconds_from_ass_time


video_bp = Blueprint("video", __name__)


@video_bp.route("/api/video/current", methods=["GET"])
def current_project_video():
    """Serve only the video assigned to the active project."""
    if not state.active_project:
        return jsonify({"error": "No active project"}), 404

    video_path = Path(str(state.active_project.video_path or "")).resolve()
    if not video_path.is_file():
        return jsonify({"error": "Project video was not found"}), 404

    return send_file(video_path, conditional=True)


def _split_text_for_timeline(text: str, rows: list[dict]) -> list[str]:
    """Split replacement dialogue across existing rows without changing their timing."""
    words = re.findall(r"\S+", text or "")
    if not rows:
        return []
    if not words:
        return [""] * len(rows)

    weights = []
    for row in rows:
        original_words = len(re.findall(r"\S+", str(row.get("Original", ""))))
        start = seconds_from_ass_time(str(row.get("Start", "0")))
        end = seconds_from_ass_time(str(row.get("End", "0")))
        duration_weight = max(1.0, (end - start) * 2.5)
        weights.append(max(1.0, original_words, duration_weight))

    total_weight = sum(weights)
    boundaries = [0]
    cumulative = 0.0
    for weight in weights[:-1]:
        cumulative += weight
        boundaries.append(round(len(words) * cumulative / total_weight))
    boundaries.append(len(words))

    result = []
    for index in range(len(rows)):
        start_index = max(boundaries[index], boundaries[index - 1] if index else 0)
        end_index = max(start_index, boundaries[index + 1])
        result.append(" ".join(words[start_index:end_index]).strip())
    return result


def _safe_upload_filename(original_name: str) -> str:
    """
    werkzeug.secure_filename("日本語字幕.srt") may lose the useful stem/extension
    relationship. Preserve a validated extension explicitly.
    """
    original = Path(original_name or "upload")
    ext = original.suffix.lower()

    # Keep common media/subtitle extensions. Unknown extension is still
    # preserved if it is simple/alphanumeric.
    if not (
        1 <= len(ext) <= 10
        and ext.startswith(".")
        and ext[1:].replace("_", "").isalnum()
    ):
        ext = ""

    safe_stem = secure_filename(original.stem).strip("._-")

    if not safe_stem:
        safe_stem = "subtitle" if ext in {".srt", ".ass", ".ssa", ".vtt"} else "media"

    # Unique suffix prevents two uploads with the same localized name
    # from overwriting each other.
    return f"{safe_stem}_{uuid.uuid4().hex[:8]}{ext}"


@video_bp.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "Brak pliku"}), 400

    f = request.files["file"]

    if not f.filename:
        return jsonify({"error": "Pusta nazwa pliku"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    original_filename = f.filename
    filename = _safe_upload_filename(original_filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    f.save(save_path)

    state.add_log(
        f"📁 Przesłano plik: {original_filename} → {filename} "
        f"({os.path.getsize(save_path) // 1024} KB)"
    )

    return jsonify({
        "path": save_path,
        "filename": filename,
        "original_filename": original_filename,
    })


@video_bp.route("/api/subtitles/align", methods=["POST"])
def align_replacement_subtitles():
    """Apply external subtitle text to the existing Whisper timeline."""
    if not state.active_project or not state.active_project.subtitles:
        return jsonify({"error": "Generate or load the original timeline first"}), 400

    data = request.get_json(silent=True) or {}
    subtitle_path = str(data.get("subtitles_path", "") or "").strip()
    if not subtitle_path or not os.path.isfile(subtitle_path):
        return jsonify({"error": "Replacement subtitle file was not found"}), 400

    events = parse_subtitle_file(subtitle_path)
    if not events:
        return jsonify({"error": "No valid subtitle entries were found"}), 400

    rows = [dict(row) for row in state.active_project.subtitles]
    replacement_text = " ".join(str(event.get("text", "")).strip() for event in events)
    aligned = _split_text_for_timeline(replacement_text, rows)
    for row, text in zip(rows, aligned):
        row["Translation"] = text
        row["Edited"] = True
        row["Approved"] = "pending"
        row["AlignmentSource"] = os.path.basename(subtitle_path)

    state.set_df(pd.DataFrame(rows))
    state.add_log(
        f"🧩 Aligned {len(events)} replacement subtitle entries to "
        f"{len(rows)} original timeline segments."
    )
    return jsonify({"ok": True, "rows": rows, "source_count": len(events)})


@video_bp.route("/api/load_video", methods=["POST"])
def load_video():
    if not state.active_project:
        return jsonify({
            "error": "Brak aktywnego projektu. Najpierw utwórz lub wczytaj projekt."
        }), 400

    data = request.get_json() or {}

    video_path = str(data.get("video_path", "") or "").strip()
    subtitles_path = str(data.get("subtitles_path", "") or "").strip()
    yt_url = str(data.get("yt_url", "") or "").strip()
    source_lang = str(data.get("source_lang", "auto") or "auto").strip()
    target_lang = str(data.get("target_lang", "Polish") or "Polish").strip()
    whisper_model_name = str(data.get("whisper_model", "turbo") or "turbo").strip()
    hf_token = str(data.get("hf_token", "") or "").strip()
    try:
        requested_speakers = int(data.get("num_speakers", 0) or 0)
    except (TypeError, ValueError):
        requested_speakers = 0
    requested_speakers = max(0, min(20, requested_speakers))
    num_speakers = requested_speakers or None
    subtitles_role = str(data.get("subtitles_role", "original") or "original").strip()

    state.active_project.video_path = video_path
    state.active_project.subtitles_path = subtitles_path
    state.active_project.source_lang = source_lang
    state.active_project.target_lang = target_lang
    state.active_project.whisper_model = whisper_model_name
    state.active_project.hf_token = hf_token
    state.active_project.num_speakers = requested_speakers
    state.active_project.save()

    state.cancel_flags["transcribe"] = False

    def run_worker():
        try:
            nonlocal video_path

            if yt_url:
                state.add_log(f"📥 Pobieranie wideo z YouTube: {yt_url}")
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                dl_path = os.path.join(UPLOAD_FOLDER, "video_download.mp4")

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "yt_dlp",
                        "-f",
                        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
                        "-o",
                        dl_path,
                        yt_url,
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    error_msg = result.stderr.strip()
                    state.add_log(f"❌ Błąd yt-dlp: {error_msg}")
                    raise RuntimeError(f"yt-dlp failed: {error_msg}")

                video_path = dl_path
                state.active_project.video_path = video_path
                state.active_project.save()

            if not video_path or not os.path.exists(video_path):
                state.add_log("❌ Błąd: Nie odnaleziono pliku wideo.")
                return

            source_code = (
                "auto"
                if source_lang == "auto"
                else LANGUAGE_MAPPING.get(source_lang, "ja")
            )

            source_texts = []
            timestamps = []
            speakers = []
            is_file_subs = False

            if subtitles_path:
                if not os.path.exists(subtitles_path):
                    raise FileNotFoundError(
                        f"Nie odnaleziono pliku napisów: {subtitles_path}"
                    )

                state.add_log(
                    f"📝 Ładowanie gotowych napisów: {subtitles_path}"
                )

                is_file_subs = True
                events = parse_subtitle_file(subtitles_path)

                if not events:
                    raise ValueError(
                        "Plik napisów został odczytany, ale nie znaleziono "
                        "żadnych poprawnych wpisów Dialogue/SRT."
                    )

                source_texts = [str(e["text"]).strip() for e in events]
                timestamps = [
                    (
                        int(seconds_from_ass_time(e["start"]) * 1000),
                        int(seconds_from_ass_time(e["end"]) * 1000),
                    )
                    for e in events
                ]

                state.add_log(
                    f"  ✅ Parser: {len(source_texts)} linii napisów."
                )

                # Diarization is useful for dubbing, but a diarization failure
                # must not make imported subtitles disappear.
                try:
                    speakers = perform_diarization(
                        video_path,
                        timestamps,
                        hf_token,
                        num_speakers,
                    )

                    if len(speakers) != len(source_texts):
                        state.add_log(
                            "  ⚠️ Diarization zwróciła inną liczbę speakerów; "
                            "uzupełniam Unknown."
                        )
                        speakers = (speakers or [])[:len(source_texts)]
                        speakers += ["Unknown"] * (
                            len(source_texts) - len(speakers)
                        )

                except Exception as diar_error:
                    state.add_log(
                        f"  ⚠️ Diarization nie powiodła się: {diar_error}. "
                        "Napisy zostają wczytane z Speaker=Unknown."
                    )
                    speakers = ["Unknown"] * len(source_texts)

            else:
                segments, detected = transcribe_video(
                    video_path,
                    whisper_model_name,
                    source_code,
                )

                if state.cancel_flags["transcribe"]:
                    state.cancel_flags["transcribe"] = False
                    return

                usable_segments = [
                    s for s in segments
                    if getattr(s, "text", "").strip()
                ]

                source_texts = [
                    s.text.strip()
                    for s in usable_segments
                ]

                timestamps = [
                    (
                        int(s.start * 1000),
                        int(s.end * 1000),
                    )
                    for s in usable_segments
                ]

                detected_lang_name = next(
                    (
                        k
                        for k, v in LANGUAGE_MAPPING.items()
                        if v == detected
                    ),
                    detected,
                )

                state.active_project.source_lang = detected_lang_name

                state.add_log(
                    f"  ✅ Whisper: {len(source_texts)} linii."
                )

                try:
                    speakers = perform_diarization(
                        video_path,
                        timestamps,
                        hf_token,
                        num_speakers,
                    )
                except Exception as diar_error:
                    state.add_log(
                        f"  ⚠️ Diarization nie powiodła się: {diar_error}. "
                        "Używam Unknown."
                    )
                    speakers = ["Unknown"] * len(source_texts)

                if len(speakers) != len(source_texts):
                    speakers = (speakers or [])[:len(source_texts)]
                    speakers += ["Unknown"] * (
                        len(source_texts) - len(speakers)
                    )

            is_ready_translation = (
                is_file_subs
                and subtitles_role == "translation"
            )

            df = pd.DataFrame({
                "Lp.": list(range(1, len(source_texts) + 1)),
                "Start": [
                    format_time(ts[0])
                    for ts in timestamps
                ],
                "End": [
                    format_time(ts[1])
                    for ts in timestamps
                ],
                "Original": source_texts,
                "Translation": (
                    source_texts
                    if is_ready_translation
                    else [""] * len(source_texts)
                ),
                "Speaker": speakers,
                "Voice": [""] * len(source_texts),
                "Speed": [1.0] * len(source_texts),
                "TimingMode": ["auto"] * len(source_texts),
                "Confidence": [
                    100 if is_ready_translation else 85
                ] * len(source_texts),
                "Edited": [
                    bool(is_ready_translation)
                ] * len(source_texts),
                "Approved": [
                    "approved" if is_ready_translation else "pending"
                ] * len(source_texts),
                "Ignore": [False] * len(source_texts),
            })

            state.set_df(df)

            if is_ready_translation:
                state.add_log(
                    f"✅ Gotowe tłumaczenie wczytane: {len(df)} wierszy. "
                    "Możesz przejść bezpośrednio do dubbingu."
                )
            else:
                state.add_log(
                    f"✅ Napisy źródłowe wczytane: {len(df)} wierszy."
                )

        except Exception as exc:
            state.add_log(
                f"❌ Błąd krytyczny podczas ładowania wideo/napisów: "
                f"{type(exc).__name__}: {exc}"
            )

    threading.Thread(
        target=run_worker,
        daemon=True,
        name="video-load-worker",
    ).start()

    return jsonify({"status": "started"})


@video_bp.route("/api/load_video/cancel", methods=["POST"])
def cancel_load_video():
    state.cancel_flags["transcribe"] = True
    state.add_log(
        "⏳ Żądanie anulowania transkrypcji zostało wysłane. "
        "Zwalnianie zasobów..."
    )

    import gc
    import torch

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jsonify({"success": True})
