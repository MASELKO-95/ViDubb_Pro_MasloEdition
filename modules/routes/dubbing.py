# -*- coding: utf-8 -*-
"""
Blueprint for dubbing, voices, subtitle operations, and file downloads
"""
import os
import threading
import pandas as pd
from flask import Blueprint, jsonify, request, send_file
from modules.state import state
from modules.config import LANGUAGE_MAPPING, RESULTS_DIR
from modules.services.tts_service import generate_dubbed_audio
from modules.services.video_service import mix_with_background, create_final_video
from modules.utils.time_utils import parse_time

dubbing_bp = Blueprint('dubbing', __name__)

@dubbing_bp.route("/api/voices", methods=["GET"])
def get_voices():
    """List available voice samples in speakers_audio/ folder"""
    voices = ["Default"]
    if os.path.exists("speakers_audio"):
        for f in os.listdir("speakers_audio"):
            if f.endswith(".wav"):
                voices.append(f[:-4])
    return jsonify({"voices": voices})


@dubbing_bp.route("/api/voices/<name>", methods=["GET"])
def play_voice(name):
    """Play/stream a specific voice sample file"""
    path = os.path.join("speakers_audio", f"{name}.wav")
    if os.path.exists(path):
        return send_file(path)
    return jsonify({"error": "Głos nie został odnaleziony"}), 404


@dubbing_bp.route("/api/subtitles", methods=["GET"])
def get_subtitles():
    """Retrieve subtitle list for active project"""
    if not state.active_project:
        return jsonify({"rows": []})
    return jsonify({"rows": state.active_project.subtitles})


@dubbing_bp.route("/api/subtitles/update", methods=["POST"])
def update_subtitles():
    """Update active project subtitle list from client"""
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 400
        
    rows = request.get_json().get("rows", [])
    df = pd.DataFrame(rows)
    state.set_df(df)
    return jsonify({"ok": True})


@dubbing_bp.route("/api/reset", methods=["POST"])
def reset_project():
    """Reset active project settings and subtitles"""
    if state.active_project:
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
    return jsonify({"error": "Brak aktywnego projektu"}), 400


@dubbing_bp.route("/api/generate_dubbing", methods=["POST"])
def run_generate_dubbing():
    """Generate voice dubbed video asynchronously"""
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 400
        
    df = state.get_df()
    if df.empty:
        return jsonify({"error": "Brak załadowanych napisów"}), 400

    data = request.get_json() or {}
    target_lang = data.get("target_lang", "Polish")
    voice = data.get("voice", "Default")
    tts_engine = data.get("tts_engine", "edge")
    keep_bg = data.get("keep_bg", True)
    hardsub = data.get("hardsub", False)
    validation_model = data.get("validation_model", "None")
    output_video_path = data.get("output_video_path", "").strip()
    auto_accept = data.get("auto_accept", True)  # True = ignoruj zatwierdzenia, False = tylko approved
    auto_retry_count = int(data.get("auto_retry_count", 10))
    audio_enhance = bool(data.get("audio_enhance", False))
    enhance_method = data.get("enhance_method", "dsp_denoise")

    # If no output path specified, default to results/
    if not output_video_path:
        output_video_path = os.path.join(RESULTS_DIR, f"{state.active_project.name}_output.mp4")

    # Update state attributes
    state.cancel_flags["dubbing"] = False

    # Store settings in project
    state.active_project.dub_lang = target_lang
    state.active_project.voice = voice
    state.active_project.tts_engine = tts_engine
    state.active_project.keep_bg = keep_bg
    state.active_project.hardsub = hardsub
    state.active_project.validation_model = validation_model
    state.active_project.auto_retry_count = auto_retry_count
    state.active_project.audio_enhance = audio_enhance
    state.active_project.enhance_method = enhance_method
    state.active_project.output_video_path = output_video_path
    state.active_project.save()

    def run_worker():
        try:
            # Prepare translation text list respecting approval and Ignore flag
            translations = []
            manual_count = 0
            rejected_count = 0
            ignored_count = 0
            for item in state.active_project.subtitles:
                # Fragment marked to be ignored (e.g. intro music) → silence at exact timestamp
                if item.get("Ignore", False):
                    translations.append("")
                    ignored_count += 1
                    continue

                t_text = (item.get("Translation") or "").strip()
                o_text = (item.get("Original") or "").strip()
                approval = item.get("Approved", "pending")

                if not auto_accept and approval == "rejected":
                    # Odrzucona linia — pusta = cisza w TTS
                    translations.append("")
                    rejected_count += 1
                elif t_text:
                    translations.append(t_text)
                    manual_count += 1
                else:
                    translations.append(o_text)

            mode_label = "auto" if auto_accept else "ręczne zatwierdzanie"
            state.add_log(
                f"  📝 Linie: {manual_count} z tłum., {rejected_count} odrzuconych, "
                f"{ignored_count} ignorowanych, tryb={mode_label}"
            )

            # Map target language code
            lang_code = LANGUAGE_MAPPING.get(target_lang, "pl")

            video_path = state.active_project.video_path
            timestamps = []
            for item in state.active_project.subtitles:
                start_ms = parse_time(item.get("Start", ""))
                end_ms = parse_time(item.get("End", ""))
                timestamps.append((start_ms, end_ms))
                
            speakers = [item.get("Speaker", "Unknown") for item in state.active_project.subtitles]

            # 2. Run TTS Generation
            dub_path = generate_dubbed_audio(
                video_path=video_path,
                translated_texts=translations,
                timestamps=timestamps,
                speakers=speakers,
                target_lang_code=lang_code,
                voice_name=voice,
                tts_engine=tts_engine,
                validation_model_size=validation_model,
                auto_retry_count=auto_retry_count
            )

            if state.cancel_flags["dubbing"] or not dub_path:
                state.add_log("❌ Generowanie dubbingu zostało anulowane.")
                return

            # 3. Mix with background instrumental
            final_audio = mix_with_background(video_path, dub_path, keep_bg)

            # 3.5 Optional Audio Enhancement (Denoise / EQ / Clarity)
            if audio_enhance:
                from modules.services.audio_enhancer import enhance_audio
                final_audio = enhance_audio(final_audio, method=enhance_method)


            if state.cancel_flags["dubbing"]:
                return

            # 4. Generate SRT captions file
            srt_lines = []
            for i, item in enumerate(state.active_project.subtitles):
                srt_lines.append(str(i + 1))
                srt_lines.append(f"{item['Start']} --> {item['End']}")
                text = item.get("Translation") or item.get("Original") or ""
                srt_lines.append(text)
                srt_lines.append("")
                
            subtitles_srt_path = os.path.join(RESULTS_DIR, f"{state.active_project.name}_subtitles.srt")
            with open(subtitles_srt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_lines))

            # 5. Compile final video
            create_final_video(
                video_path=video_path,
                audio_path=final_audio,
                output_path=output_video_path,
                hardsub=hardsub,
                subtitles_path=subtitles_srt_path
            )

            from modules.utils.cleanup import cleanup_temp_files
            cleanup_temp_files()
            state.add_log(f"🎉 Sukces! Gotowy plik wideo został zapisany w: {output_video_path}")

        except Exception as e:
            state.add_log(f"❌ Błąd krytyczny podczas generowania filmu: {e}")
        finally:
            state.cancel_flags["dubbing"] = False

    t = threading.Thread(target=run_worker, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@dubbing_bp.route("/api/generate_dubbing/cancel", methods=["POST"])
def cancel_generate_dubbing():
    """Cancel active dubbing synthesis and free GPU memory"""
    state.cancel_flags["dubbing"] = True
    state.add_log("⏳ Żądanie anulowania generowania dubbingu zostało wysłane. Zwalnianie zasobów GPU...")
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return jsonify({"success": True})



@dubbing_bp.route("/api/export_srt", methods=["GET"])
def export_srt_file():
    """Export and download SRT subtitles file for active project"""
    if not state.active_project or not state.active_project.subtitles:
        return jsonify({"error": "Brak danych napisów"}), 400
        
    srt_lines = []
    for i, item in enumerate(state.active_project.subtitles):
        srt_lines.append(str(i + 1))
        srt_lines.append(f"{item['Start']} --> {item['End']}")
        text = item.get("Translation") or item.get("Original") or ""
        srt_lines.append(text)
        srt_lines.append("")
        
    subtitles_srt_path = os.path.join(RESULTS_DIR, f"{state.active_project.name}_subtitles.srt")
    with open(subtitles_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
        
    return send_file(subtitles_srt_path, as_attachment=True)


@dubbing_bp.route("/api/download_video", methods=["GET"])
def download_output_video():
    """Download the compiled video file of the active project"""
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 404
        
    output_path = state.active_project.output_video_path
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    return jsonify({"error": "Plik wynikowy nie istnieje. Wygeneruj film najpierw."}), 404
