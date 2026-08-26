# -*- coding: utf-8 -*-
"""
Utility for cleaning up temporary files after dubbing process
"""
import os
import shutil
import glob
from modules.config import WORKSPACE_DIR

def cleanup_temp_files():
    """Remove temporary audio chunks, separated tracks, and other intermediary files"""
    print("🧹 Czyszczenie plików tymczasowych...")
    
    # 1. Remove audio working directories
    for folder in ["audio", "audio_chunks", "su_audio_chunks"]:
        path = os.path.join(WORKSPACE_DIR, folder)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"  🗑️ Usunięto folder: {folder}/")
    
    # 2. Remove UVR separator output files (Instrumental/Vocals wav files in workspace root)
    uvr_patterns = [
        "*_(Instrumental)_*.wav",
        "*_(Vocals)_*.wav",
        "*_(Instrumental)_*.mp3",
        "*_(Vocals)_*.mp3",
    ]
    for pattern in uvr_patterns:
        for filepath in glob.glob(os.path.join(WORKSPACE_DIR, pattern)):
            try:
                os.remove(filepath)
                print(f"  🗑️ Usunięto plik UVR: {os.path.basename(filepath)}")
            except OSError:
                pass
    
    # 3. Remove temporary diarization/whisper files
    temp_files = [
        os.path.join(WORKSPACE_DIR, "audio", "diar_temp.wav"),
        os.path.join(WORKSPACE_DIR, "output_video.mp3"),
    ]
    for tf in temp_files:
        if os.path.exists(tf):
            try:
                os.remove(tf)
                print(f"  🗑️ Usunięto: {os.path.basename(tf)}")
            except OSError:
                pass

    print("✅ Czyszczenie zakończone.")
