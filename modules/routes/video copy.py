# -*- coding: utf-8 -*-
"""
Blueprint for video operations (upload, load, transcribe, diarization)
"""
import os
import threading
import subprocess
import pandas as pd
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from modules.state import state
from modules.config import UPLOAD_FOLDER, LANGUAGE_MAPPING
from modules.utils.file_utils import parse_srt_file, parse_ass_file
from modules.utils.time_utils import seconds_from_ass_time, format_time
from modules.services.whisper_service import transcribe_video
from modules.services.diarization_service import perform_diarization

video_bp = Blueprint('video', __name__)

@video_bp.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "Brak pliku"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Pusta nazwa pliku"}), 400
    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)
    state.add_log(f"📁 Przesłano plik: {filename} ({os.path.getsize(save_path) // 1024} KB)")
    return jsonify({"path": save_path, "filename": filename})

@video_bp.route("/api/load_video", methods=["POST"])
def load_video():
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu. Najpierw utwórz lub wczytaj projekt."}), 400
    data = request.get_json() or {}
    video_path = data.get("video_path", "").strip()
    subtitles_path = data.get("subtitles_path", "").strip()
    yt_url = data.get("yt_url", "").strip()
    source_lang = data.get("source_lang", "auto").strip()
    target_lang = data.get("target_lang", "Polish").strip()
    whisper_model_name = data.get("whisper_model", "turbo").strip()
    hf_token = data.get("hf_token", "").strip()
    subtitles_role = data.get("subtitles_role", "original").strip()

    state.active_project.video_path = video_path
    state.active_project.subtitles_path = subtitles_path
    state.active_project.source_lang = source_lang
    state.active_project.target_lang = target_lang
    state.active_project.whisper_model = whisper_model_name
    state.active_project.hf_token = hf_token
    state.active_project.save()
    state.cancel_flags["transcribe"] = False

    def run_worker():
        try:
            nonlocal video_path

            # --- POPRAWIONA SEKCJA POBIERANIA YT-DLP ---
            if yt_url:
                state.add_log(f"📥 Pobieranie wideo z YouTube: {yt_url}")
                dl_path = os.path.join(UPLOAD_FOLDER, "video_download.mp4")

                result = subprocess.run(
                    ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best", "-o", dl_path, yt_url],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    error_msg = result.stderr.strip()
                    state.add_log(f"❌ Błąd yt-dlp: {error_msg}")
                    raise Exception(f"yt-dlp failed: {error_msg}")

                video_path = dl_path
                state.active_project.video_path = video_path
                state.active_project.save()
            # ---------------------------------------------

            if not video_path or not os.path.exists(video_path):
                state.add_log("❌ Błąd: Nie odnaleziono pliku wideo.")
                return

            source_code = "auto" if source_lang == "auto" else LANGUAGE_MAPPING.get(source_lang, "ja")
            source_texts = []
            timestamps = []
            speakers = []
            is_file_subs = False

            if subtitles_path and os.path.exists(subtitles_path):
                state.add_log(f"📝 Ładowanie napisów z pliku: {subtitles_path}")
                is_file_subs = True
                if subtitles_path.lower().endswith('.srt'):
                    events = parse_srt_file(subtitles_path)
                else:
                    events = parse_ass_file(subtitles_path)
                source_texts = [e['text'] for e in events]
                timestamps = [
                    (int(seconds_from_ass_time(e['start']) * 1000), int(seconds_from_ass_time(e['end']) * 1000))
                    for e in events
                ]
                speakers = ["Unknown"] * len(source_texts)
                state.add_log(f"  Pomyślnie załadowano {len(source_texts)} napisów z pliku.")

                # Diarization zawsze się uruchamia, jeśli mamy hf_token LUB chcemy fallbacku
                # (twój diarization_service.py obsłuży brak tokenu przez SpeechBrain)
                speakers = perform_diarization(video_path, timestamps, hf_token)
            else:
                segments, detected = transcribe_video(video_path, whisper_model_name, source_code)
                if state.cancel_flags["transcribe"]:
                    state.cancel_flags["transcribe"] = False
                    return
                source_texts = [s.text.strip() for s in segments if s.text.strip()]
                timestamps = [(int(s.start * 1000), int(s.end * 1000)) for s in segments if s.text.strip()]
                speakers = ["Unknown"] * len(source_texts)

                detected_lang_name = [k for k, v in LANGUAGE_MAPPING.items() if v == detected][0] if detected in LANGUAGE_MAPPING.values() else detected
                state.active_project.source_lang = detected_lang_name
                state.add_log(f"  Pomyślnie wygenerowano {len(source_texts)} napisów przez Whisper.")

                # Diarization zawsze się uruchamia
                speakers = perform_diarization(video_path, timestamps, hf_token)

            # Sprawdź czy użytkownik wybrał że wczytany plik to gotowe tłumaczenie
            is_ready_translation = (is_file_subs and subtitles_role == "translation")

            df = pd.DataFrame({
                "Lp.": list(range(1, len(source_texts) + 1)),
                "Start": [format_time(ts[0]) for ts in timestamps],
                "End": [format_time(ts[1]) for ts in timestamps],
                "Original": source_texts,
                "Translation": source_texts if is_ready_translation else [""] * len(source_texts),
                "Speaker": speakers,
                "Confidence": [100 if is_ready_translation else 85] * len(source_texts),
                "Edited": [True if is_ready_translation else False] * len(source_texts),
                "Approved": ["approved" if is_ready_translation else "pending"] * len(source_texts),
            })
            state.set_df(df)

            if is_ready_translation:
                state.add_log(f"✅ Napisy wczytano jako GOTOWE TŁUMACZENIE ({len(df)} wierszy). Możesz od razu generować dubbing.")
            else:
                state.add_log(f"✅ Załadowano i zsynchronizowano napisy projektu ({len(df)} wierszy).")

        except Exception as e:
            state.add_log(f"❌ Błąd krytyczny podczas ładowania wideo: {e}")

    t = threading.Thread(target=run_worker, daemon=True)
    t.start()
    return jsonify({"status": "started"})

@video_bp.route("/api/load_video/cancel", methods=["POST"])
def cancel_load_video():
    state.cancel_flags["transcribe"] = True
    state.add_log("⏳ Żądanie anulowania transkrypcji zostało wysłane. Zwalnianie zasobów...")
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return jsonify({"success": True})
