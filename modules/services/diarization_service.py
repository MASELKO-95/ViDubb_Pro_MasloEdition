# -*- coding: utf-8 -*-
"""
Speaker diarization service using Pyannote
"""
import os
import gc
import torch
from pydub import AudioSegment
from modules.state import state

def perform_diarization(video_path: str, timestamps: list, hf_token: str) -> list:
    if not hf_token:
        state.add_log("⚠️ Brak tokenu HuggingFace — pomijam diarization.")
        return ["Unknown"] * len(timestamps)
    state.add_log("🗣️ Uruchamianie Speaker Diarization za pomocą Pyannote...")
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        wav_path = "audio/diar_temp.wav"
        os.makedirs("audio", exist_ok=True)
        state.add_log("  Extracting audio stream for diarization model...")
        base_audio = AudioSegment.from_file(video_path)
        base_audio.export(wav_path, format="wav")
        state.add_log("  Analyzing audio diarization tracks...")
        diarization = pipeline(wav_path)
        new_speakers = []
        for start_ms, end_ms in timestamps:
            s_sec, e_sec = start_ms / 1000.0, end_ms / 1000.0
            best_speaker = "Unknown"
            max_overlap = 0
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap = max(0, min(e_sec, turn.end) - max(s_sec, turn.start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker
            new_speakers.append(best_speaker)
        state.add_log("  ✅ Speaker Diarization zakończona pomyślnie.")
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return new_speakers
    except Exception as e:
        state.add_log(f"  ❌ Diarization failed: {str(e)}")
        return ["Unknown"] * len(timestamps)
