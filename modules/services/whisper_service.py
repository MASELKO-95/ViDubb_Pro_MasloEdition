# -*- coding: utf-8 -*-
"""
Whisper transcription service
"""
import gc
import torch
from faster_whisper import WhisperModel
from modules.state import state

def transcribe_video(video_path: str, model_size: str = "turbo", language: str = "auto") -> tuple:
    state.add_log(f"🎤 Inicjalizacja transkrypcji Whisper (model={model_size})...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WhisperModel(model_size, device=device)
    lang_arg = None if language in ('auto', '', None) else language
    state.add_log("⏳ Uruchamianie transkrypcji...")
    segments, info = model.transcribe(
        video_path,
        word_timestamps=True,
        language=lang_arg,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    segments_list = []
    for segment in segments:
        if state.cancel_flags["transcribe"]:
            state.add_log("❌ Transkrypcja anulowana przez użytkownika.")
            break
        segments_list.append(segment)
    detected = info.language if hasattr(info, 'language') else language
    state.add_log(f"  🔍 Whisper zakończył transkrypcję. Wykryty język: {detected}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return segments_list, detected
