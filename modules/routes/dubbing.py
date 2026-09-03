import gc
import os
import shutil
import threading
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import torch
from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename

from modules.state import state
from modules.config import LANGUAGE_MAPPING, RESULTS_DIR
from modules.services.tts_service import (
    CHUNKS_DIR,
    build_dubbing_timeline,
    generate_dubbed_audio,
    prepared_chunk_metadata,
)
from modules.services.video_service import mix_with_background, create_final_video
from modules.services.voice_db_service import (
    load_voice_db,
    delete_voice_profile as db_delete_voice_profile,
    import_voice_from_folder,
)
from modules.utils.time_utils import parse_time


dubbing_bp = Blueprint("dubbing", __name__)

_review_state = {
    "running": False,
    "ready": False,
    "error": "",
}

MAX_VOICE_ZIP_FILES = 5000
MAX_VOICE_ZIP_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
MAX_VOICE_ZIP_MEMBER_BYTES = 256 * 1024 * 1024
MAX_VOICE_ZIP_COMPRESSION_RATIO = 250

@dubbing_bp.route("/api/select_output_folder", methods=["POST"])
def select_output_folder():
    """Open a native folder picker for this local desktop-style application."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial = RESULTS_DIR if os.path.isdir(RESULTS_DIR) else os.getcwd()
        folder = filedialog.askdirectory(title="Wybierz katalog zapisu filmu", initialdir=initial)
        root.destroy()
        return jsonify({"folder": folder or ""})
    except Exception as exc:
        state.add_log(f"⚠️ Systemowy wybór katalogu nie jest dostępny: {exc}")
        return jsonify({"error": "Nie udało się otworzyć systemowego wyboru katalogu."}), 500


# ============================================================
# BASIC VOICE ENDPOINTS
# ============================================================

@dubbing_bp.route("/api/voices", methods=["GET"])
def get_voices():
    voices = ["Default"]

    if os.path.exists("speakers_audio"):
        try:
            for filename in sorted(os.listdir("speakers_audio")):
                if filename.lower().endswith(".wav"):
                    voices.append(filename[:-4])
        except Exception as e:
            state.add_log(f"⚠️ Nie udało się odczytać katalogu speakers_audio: {e}")

    return jsonify({"voices": voices})


@dubbing_bp.route("/api/voices/<name>", methods=["GET"])
def play_voice(name):

    safe_name = secure_filename(name)

    if not safe_name:
        return jsonify({"error": "Nieprawidłowa nazwa głosu"}), 400

    db = load_voice_db()
    profile = db.get(safe_name, {}) if isinstance(db, dict) else {}
    preview_path = ""
    if isinstance(profile, dict):
        preview_path = str(profile.get("preview_path", "") or "")

    if preview_path and os.path.isfile(preview_path):
        return send_file(os.path.abspath(preview_path))

    path = os.path.abspath(os.path.join("speakers_audio", f"{safe_name}.wav"))
    speakers_dir = os.path.abspath("speakers_audio")


    if os.path.commonpath([path, speakers_dir]) != speakers_dir:
        return jsonify({"error": "Nieprawidłowa ścieżka głosu"}), 400

    if os.path.exists(path):
        return send_file(path)

    return jsonify({"error": "Głos nie został odnaleziony"}), 404


# ============================================================
# SUBTITLES
# ============================================================

@dubbing_bp.route("/api/subtitles", methods=["GET"])
def get_subtitles():
    if not state.active_project:
        return jsonify({"rows": []})

    return jsonify({"rows": state.active_project.subtitles})


@dubbing_bp.route("/api/subtitles/update", methods=["POST"])
def update_subtitles():
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 400

    data = request.get_json(silent=True) or {}
    rows = data.get("rows", [])

    if not isinstance(rows, list):
        return jsonify({"error": "Pole 'rows' musi być listą"}), 400

    df = pd.DataFrame(rows)
    state.set_df(df)

    return jsonify({"ok": True})


# ============================================================
# RESET PROJECT
# ============================================================

@dubbing_bp.route("/api/reset", methods=["POST"])
def reset_project():
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 400

    state.active_project.video_path = ""
    state.active_project.subtitles_path = ""
    state.active_project.source_lang = "auto"
    state.active_project.target_lang = "Polish"
    state.active_project.context = ""
    state.active_project.subtitles = []
    state.active_project.logs = []
    state.active_project.save()

    state.add_log("🔄 Projekt wyzerowany i zresetowany.")

    return jsonify({"ok": True})


# ============================================================
# GENERATE DUBBING
# ============================================================

@dubbing_bp.route("/api/dubbing/review/prepare", methods=["POST"])
def prepare_dubbing_review():
    """Generate per-dialogue TTS chunks without rendering the final video."""
    if not state.active_project or not state.active_project.subtitles:
        return jsonify({"error": "Brak dialogów do przygotowania"}), 400
    if _review_state["running"]:
        return jsonify({"error": "Przygotowanie dialogów już trwa"}), 409

    data = request.get_json(silent=True) or {}
    target_lang = data.get("target_lang", "Polish")
    voice = data.get("voice", "Default")
    tts_engine = data.get("tts_engine", "edge")
    validation_model = data.get("validation_model", "None")
    try:
        auto_retry_count = max(1, min(10, int(data.get("auto_retry_count", 10))))
    except (TypeError, ValueError):
        auto_retry_count = 10

    subtitles = list(state.active_project.subtitles)
    texts = [
        "" if row.get("Ignore", False) else str(
            row.get("Translation") or row.get("Original") or ""
        ).strip()
        for row in subtitles
    ]
    timestamps = [
        (parse_time(row.get("Start", "")), parse_time(row.get("End", "")))
        for row in subtitles
    ]
    speakers = [row.get("Speaker", "Unknown") for row in subtitles]
    voices = [row.get("Voice", "") for row in subtitles]
    lang_code = LANGUAGE_MAPPING.get(target_lang, "pl")

    _review_state.update(running=True, ready=False, error="")
    state.cancel_flags["dubbing"] = False

    def worker():
        try:
            generate_dubbed_audio(
                video_path=state.active_project.video_path,
                translated_texts=texts,
                timestamps=timestamps,
                speakers=speakers,
                target_lang_code=lang_code,
                voice_name=voice,
                dialogue_voices=voices,
                tts_engine=tts_engine,
                validation_model_size=validation_model,
                auto_retry_count=auto_retry_count,
            )
            if not state.cancel_flags["dubbing"]:
                _review_state["ready"] = True
                state.add_log(
                    "🎬 Dialogi TTS gotowe. Otwieram Timeline Review przed renderem."
                )
        except Exception as exc:
            _review_state["error"] = f"{type(exc).__name__}: {exc}"
            state.add_log(f"❌ Timeline Review: {_review_state['error']}")
        finally:
            _review_state["running"] = False

    threading.Thread(target=worker, daemon=True, name="dubbing-review-prepare").start()
    return jsonify({"status": "started"})


@dubbing_bp.route("/api/dubbing/review/status", methods=["GET"])
def dubbing_review_status():
    metadata = []
    if _review_state["ready"] and state.active_project:
        timestamps = [
            (parse_time(row.get("Start", "")), parse_time(row.get("End", "")))
            for row in state.active_project.subtitles
        ]
        metadata = prepared_chunk_metadata(timestamps)
    return jsonify({**_review_state, "chunks": metadata})


@dubbing_bp.route("/api/dubbing/review/chunk/<int:index>", methods=["GET"])
def preview_dubbing_chunk(index: int):
    if index < 0 or not state.active_project or index >= len(state.active_project.subtitles):
        return jsonify({"error": "Nieprawidłowy numer dialogu"}), 404
    chunk_path = os.path.abspath(os.path.join(CHUNKS_DIR, f"{index}.wav"))
    chunks_root = os.path.abspath(CHUNKS_DIR)
    if os.path.commonpath([chunk_path, chunks_root]) != chunks_root or not os.path.isfile(chunk_path):
        return jsonify({"error": "Próbka dialogu nie istnieje"}), 404
    return send_file(chunk_path)

@dubbing_bp.route("/api/generate_dubbing", methods=["POST"])
def run_generate_dubbing():
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 400

    df = state.get_df()
    if df.empty:
        return jsonify({"error": "Brak załadowanych napisów"}), 400

    video_path = state.active_project.video_path or ""
    if not video_path or not os.path.exists(video_path):
        return jsonify({
            "error": (
                "Brak wczytanego wideo w projekcie. "
                "Najpierw podaj lub wczytaj film wideo!"
            )
        }), 400

    data = request.get_json(silent=True) or {}

    target_lang = data.get("target_lang", "Polish")
    voice = data.get("voice", "Default")
    tts_engine = data.get("tts_engine", "edge")
    keep_bg = bool(data.get("keep_bg", True))
    hardsub = bool(data.get("hardsub", False))
    validation_model = data.get("validation_model", "None")
    output_video_path = str(data.get("output_video_path", "") or "").strip()
    auto_accept = bool(data.get("auto_accept", True))
    reuse_prepared_audio = bool(data.get("reuse_prepared_audio", False))

    try:
        auto_retry_count = max(0, int(data.get("auto_retry_count", 10)))
    except (TypeError, ValueError):
        auto_retry_count = 10

    audio_enhance = bool(data.get("audio_enhance", False))
    enhance_method = data.get("enhance_method", "dsp_denoise")
    lipsync = bool(data.get("lipsync", False))

    if not output_video_path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_video_path = os.path.join(
            RESULTS_DIR,
            f"{state.active_project.name}_output.mp4"
        )
    output_video_path = os.path.abspath(os.path.expanduser(output_video_path))
    if Path(output_video_path).suffix.lower() != ".mp4":
        output_video_path += ".mp4"
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    state.cancel_flags["dubbing"] = False

    state.active_project.dub_lang = target_lang
    state.active_project.voice = voice
    state.active_project.tts_engine = tts_engine
    state.active_project.keep_bg = keep_bg
    state.active_project.hardsub = hardsub
    state.active_project.validation_model = validation_model
    state.active_project.auto_retry_count = auto_retry_count
    state.active_project.audio_enhance = audio_enhance
    state.active_project.enhance_method = enhance_method
    state.active_project.lipsync = lipsync
    state.active_project.output_video_path = output_video_path
    state.active_project.save()

    def run_worker():
        nonlocal output_video_path

        try:
            translations = []
            manual_count = 0
            rejected_count = 0
            ignored_count = 0

            for item in state.active_project.subtitles:
                if item.get("Ignore", False):
                    translations.append("")
                    ignored_count += 1
                    continue

                t_text = str(item.get("Translation") or "").strip()
                o_text = str(item.get("Original") or "").strip()
                approval = item.get("Approved", "pending")

                if not auto_accept and approval != "approved":
                    translations.append("")
                    rejected_count += 1
                elif t_text:
                    translations.append(t_text)
                    manual_count += 1
                else:
                    translations.append(o_text)

            mode_label = "auto" if auto_accept else "ręczne zatwierdzanie"
            state.add_log(
                f"  📝 Linie: {manual_count} z tłum., "
                f"{rejected_count} odrzuconych, "
                f"{ignored_count} ignorowanych, "
                f"tryb={mode_label}"
            )

            lang_code = LANGUAGE_MAPPING.get(target_lang, "pl")

            video_path_local = state.active_project.video_path

            timestamps = []
            speakers = []
            dialogue_voices = []

            for item in state.active_project.subtitles:
                start_ms = parse_time(item.get("Start", ""))
                end_ms = parse_time(item.get("End", ""))

                timestamps.append((start_ms, end_ms))
                speakers.append(item.get("Speaker", "Unknown"))
                dialogue_voices.append(item.get("Voice", ""))

            # ------------------------------------------------
            # 1. TTS
            # ------------------------------------------------

            if reuse_prepared_audio and _review_state["ready"]:
                dub_path = build_dubbing_timeline(
                    timestamps,
                    total=len(translations),
                    active_lines=[bool(text) for text in translations],
                )
            else:
                dub_path = generate_dubbed_audio(
                    video_path=video_path_local,
                    translated_texts=translations,
                    timestamps=timestamps,
                    speakers=speakers,
                    target_lang_code=lang_code,
                    voice_name=voice,
                    dialogue_voices=dialogue_voices,
                    tts_engine=tts_engine,
                    validation_model_size=validation_model,
                    auto_retry_count=auto_retry_count,
                )

            if state.cancel_flags["dubbing"] or not dub_path:
                state.add_log("❌ Generowanie dubbingu zostało anulowane.")
                return

            # ------------------------------------------------
            # 2. Mix with original/background audio
            # ------------------------------------------------

            final_audio = mix_with_background(
                video_path_local,
                dub_path,
                keep_bg,
            )

            if not final_audio:
                raise RuntimeError("Nie udało się utworzyć końcowej ścieżki audio.")

            # ------------------------------------------------
            # 3. Optional audio enhancement
            # ------------------------------------------------

            if audio_enhance:
                from modules.services.audio_enhancer import enhance_audio

                enhanced_audio = enhance_audio(
                    final_audio,
                    method=enhance_method,
                )

                if enhanced_audio:
                    final_audio = enhanced_audio

            if state.cancel_flags["dubbing"]:
                return

            # ------------------------------------------------
            # 4. Generate SRT
            # ------------------------------------------------

            os.makedirs(RESULTS_DIR, exist_ok=True)

            srt_lines = []

            for i, item in enumerate(state.active_project.subtitles):
                srt_lines.append(str(i + 1))
                srt_lines.append(
                    f"{item.get('Start', '')} --> {item.get('End', '')}"
                )

                text = (
                    item.get("Translation")
                    or item.get("Original")
                    or ""
                )

                srt_lines.append(str(text))
                srt_lines.append("")

            subtitles_srt_path = os.path.join(
                RESULTS_DIR,
                f"{state.active_project.name}_subtitles.srt"
            )

            with open(subtitles_srt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_lines))

            # ------------------------------------------------
            # 5. Build final video
            # ------------------------------------------------

            create_final_video(
                video_path=video_path_local,
                audio_path=final_audio,
                output_path=output_video_path,
                hardsub=hardsub,
                subtitles_path=subtitles_srt_path,
            )

            if state.cancel_flags["dubbing"]:
                return

            # ------------------------------------------------
            # 6. Optional lip-sync
            # ------------------------------------------------

            if lipsync:
                from modules.services.lipsync_service import run_lipsync

                output_parent = os.path.dirname(output_video_path)
                output_stem = Path(output_video_path).stem
                lipsync_out = os.path.join(output_parent, f"{output_stem}_lipsynced.mp4")

                res_path = run_lipsync(
                    video_path=output_video_path,
                    audio_path=final_audio,
                    output_path=lipsync_out,
                )

                if (
                    res_path
                    and os.path.exists(res_path)
                    and res_path != output_video_path
                ):
                    output_video_path = res_path
                    state.active_project.output_video_path = output_video_path
                    state.active_project.save()

            # ------------------------------------------------
            # 7. Cleanup
            # ------------------------------------------------

            from modules.utils.cleanup import cleanup_temp_files
            cleanup_temp_files()
            _review_state["ready"] = False

            state.add_log(
                f"🎉 Sukces! Gotowy plik wideo został zapisany w: "
                f"{output_video_path}"
            )

        except Exception as e:
            state.add_log(
                f"❌ Błąd krytyczny podczas generowania filmu: "
                f"{type(e).__name__}: {e}"
            )

        finally:
            state.cancel_flags["dubbing"] = False

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    thread = threading.Thread(
        target=run_worker,
        daemon=True,
        name="dubbing-worker",
    )
    thread.start()

    return jsonify({"status": "started"})


# ============================================================
# CANCEL DUBBING
# ============================================================

@dubbing_bp.route("/api/generate_dubbing/cancel", methods=["POST"])
def cancel_generate_dubbing():
    """Request cancellation of the active dubbing job."""
    state.cancel_flags["dubbing"] = True

    state.add_log(
        "⏳ Żądanie anulowania generowania dubbingu zostało wysłane. "
        "Zwalnianie zasobów GPU..."
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return jsonify({"success": True})


# ============================================================
# EXPORT SRT
# ============================================================

@dubbing_bp.route("/api/export_srt", methods=["GET"])
def export_srt_file():
    """Export SRT subtitles file for active project."""
    if not state.active_project or not state.active_project.subtitles:
        return jsonify({"error": "Brak danych napisów"}), 400

    os.makedirs(RESULTS_DIR, exist_ok=True)

    srt_lines = []

    for i, item in enumerate(state.active_project.subtitles):
        srt_lines.append(str(i + 1))
        srt_lines.append(
            f"{item.get('Start', '')} --> {item.get('End', '')}"
        )

        text = (
            item.get("Translation")
            or item.get("Original")
            or ""
        )

        srt_lines.append(str(text))
        srt_lines.append("")

    subtitles_srt_path = os.path.join(
        RESULTS_DIR,
        f"{state.active_project.name}_subtitles.srt"
    )

    with open(subtitles_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    return send_file(
        subtitles_srt_path,
        as_attachment=True,
    )


# ============================================================
# DOWNLOAD FINAL VIDEO
# ============================================================

@dubbing_bp.route("/api/download_video", methods=["GET"])
def download_output_video():
    """Download compiled final video."""
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 404

    output_path = state.active_project.output_video_path or ""

    if output_path and os.path.exists(output_path):
        return send_file(
            output_path,
            as_attachment=True,
        )

    return jsonify({
        "error": "Plik wynikowy nie istnieje. Wygeneruj film najpierw."
    }), 404


# ============================================================
# VOICE DATABASE API
# ============================================================

@dubbing_bp.route("/api/voices/database", methods=["GET"])
def get_voice_database():
    try:
        db = load_voice_db()
    except Exception as e:
        state.add_log(f"❌ Nie udało się odczytać Voice DB: {e}")
        return jsonify({"error": "Nie udało się odczytać bazy głosów"}), 500

    profiles = []

    for voice_id, data in db.items():

        embeddings = data.get("embeddings", [])
        centroid = data.get("centroid", [])
        legacy_embedding = data.get("embedding", [])

        has_embedding = bool(
            centroid
            or embeddings
            or legacy_embedding
        )

        profiles.append({
            "voice_id": voice_id,
            "name": voice_id,
            "display_name": data.get("display_name", voice_id),
            "source_movie": data.get("source_movie", ""),
            "description": data.get("description", ""),
            "duration_sec": data.get("duration_sec", 0),
            "sample_count": data.get("sample_count", 0),
            "wav_path": data.get("wav_path", ""),
            "preview_path": data.get("preview_path", ""),
            "has_embedding": has_embedding,
            "embedding_count": (
                len(embeddings)
                if isinstance(embeddings, list)
                else 0
            ),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
        })

    profiles.sort(
        key=lambda p: (
            str(p.get("display_name", "")).lower(),
            str(p.get("voice_id", "")).lower(),
        )
    )

    return jsonify({"profiles": profiles})


@dubbing_bp.route("/api/voices/database/import", methods=["POST"])
def import_voice_profile_api():
    """Import all supported samples in a folder-style ZIP as one person."""
    archive = request.files.get("archive")
    display_name = str(request.form.get("display_name", "") or "").strip()
    source_movie = str(request.form.get("source_movie", "") or "").strip()
    if not archive or not archive.filename:
        return jsonify({"error": "Select a ZIP archive"}), 400
    if not display_name:
        return jsonify({"error": "Voice/person name is required"}), 400
    if Path(archive.filename).suffix.lower() != ".zip":
        return jsonify({"error": "Only ZIP archives are supported"}), 400

    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    with tempfile.TemporaryDirectory(prefix="vidubb_voice_zip_") as temp_dir:
        root = Path(temp_dir)
        archive_path = root / "samples.zip"
        archive.save(archive_path)
        samples_dir = root / "samples"
        samples_dir.mkdir()

        try:
            with zipfile.ZipFile(archive_path) as bundle:
                members = [
                    info for info in bundle.infolist()
                    if not info.is_dir() and Path(info.filename).suffix.lower() in supported
                ]
                if not members:
                    return jsonify({"error": "The ZIP contains no supported audio files"}), 400
                if len(members) > MAX_VOICE_ZIP_FILES:
                    return jsonify({"error": "The ZIP contains too many audio files"}), 400
                unpacked_size = sum(max(0, info.file_size) for info in members)
                if unpacked_size > MAX_VOICE_ZIP_UNPACKED_BYTES:
                    return jsonify({"error": "The unpacked ZIP is too large"}), 413
                for index, info in enumerate(members):
                    if info.file_size > MAX_VOICE_ZIP_MEMBER_BYTES:
                        return jsonify({"error": f"Audio file is too large: {Path(info.filename).name}"}), 413
                    ratio = info.file_size / max(1, info.compress_size)
                    if ratio > MAX_VOICE_ZIP_COMPRESSION_RATIO:
                        return jsonify({"error": "The ZIP has a suspicious compression ratio"}), 400
                    # Flatten paths and generate safe names to prevent ZIP path traversal.
                    target = samples_dir / f"{index:04d}_{secure_filename(Path(info.filename).name)}"
                    with bundle.open(info) as source, open(target, "wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
        except zipfile.BadZipFile:
            return jsonify({"error": "The uploaded file is not a valid ZIP archive"}), 400

        voice_id = import_voice_from_folder(
            folder_path=str(samples_dir),
            display_name=display_name,
            source_movie=source_movie,
            description=f"Imported from ZIP ({len(members)} samples)",
        )

    if not voice_id:
        return jsonify({"error": "The samples were too short or unusable"}), 400
    return jsonify({"ok": True, "voice_id": voice_id, "sample_files": len(members)})


@dubbing_bp.route("/api/voices/database/delete", methods=["POST"])
def delete_voice_profile_api():

    data = request.get_json(silent=True) or {}

    voice_id = str(
        data.get("voice_id")
        or data.get("name")
        or ""
    ).strip()

    if not voice_id:
        return jsonify({"error": "Brak identyfikatora głosu"}), 400

    try:
        if db_delete_voice_profile(voice_id):
            return jsonify({
                "ok": True,
                "voice_id": voice_id,
            })

        return jsonify({
            "error": "Profil głosu nie istnieje"
        }), 404

    except Exception as e:
        state.add_log(
            f"❌ Błąd podczas usuwania profilu '{voice_id}': {e}"
        )

        return jsonify({
            "error": "Nie udało się usunąć profilu głosu"
        }), 500
