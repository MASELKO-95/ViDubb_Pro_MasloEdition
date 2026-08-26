# -*- coding: utf-8 -*-
"""
Video and audio editing service (FFmpeg, separation, mixing)
"""
import os
import gc
import subprocess
import torch
from pydub import AudioSegment
from modules.state import state

def mix_with_background(video_path: str, dub_path: str, keep_bg: bool = True) -> str:
    if keep_bg:
        state.add_log("🎵 Rozpoczęcie separacji audio w tle (wyodrębnianie muzyki/szumów)...")
        try:
            from audio_separator.separator import Separator
            separator = Separator()
            separator.load_model(model_filename="UVR-MDX-NET-Inst_HQ_3.onnx")
            separated = separator.separate(video_path)
            bg_path = None
            for f in separated:
                if "instrumental" in f.lower() or "no_vocals" in f.lower():
                    bg_path = f
                    break
            if not bg_path and len(separated) > 1:
                bg_path = separated[1]
            elif not bg_path:
                bg_path = separated[0]
            state.add_log(f"  Pomyślnie wyodrębniono tło: {bg_path}")
            bg = AudioSegment.from_file(bg_path) - 6
            dub = AudioSegment.from_file(dub_path)
            final = bg.overlay(dub.normalize(), position=0)
            del separator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            state.add_log(f"  ⚠️ Separacja tła nie powiodła się: {e}. Używam surowego dubbingu.")
            final = AudioSegment.from_file(dub_path)
    else:
        state.add_log("🎤 Pomijanie separacji tła. Serwowanie surowego głosu dubbingu.")
        final = AudioSegment.from_file(dub_path)
    final_audio_path = "audio/final_audio.wav"
    os.makedirs("audio", exist_ok=True)
    final.export(final_audio_path, format="wav")
    state.add_log("  ✅ Finalny miks audio został wyeksportowany.")
    return final_audio_path

def create_final_video(video_path: str, audio_path: str, output_path: str, hardsub: bool = False, subtitles_path: str = None) -> str:
    state.add_log(f"🎬 Generowanie końcowego pliku wideo. Ścieżka docelowa: {output_path}")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    if hardsub and subtitles_path and os.path.exists(subtitles_path):
        state.add_log("  🔥 Nakładanie napisów na obraz (Hardsub)...")
        escaped_sub = os.path.abspath(subtitles_path).replace("\\", "/").replace(":", "\\:")
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-i", video_path, "-i", audio_path,
            "-vf", f"subtitles='{escaped_sub}'", "-c:v", "libx264", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path
        ]
    else:
        state.add_log("  Kopiowanie obrazu wideo (Direct stream copy)...")
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path
        ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True)
        state.add_log("  ✅ Proces FFmpeg zakończony sukcesem.")
        return output_path
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode()
        state.add_log(f"  ❌ FFmpeg error: {err_msg}")
        raise RuntimeError(f"FFmpeg failed: {err_msg}")
