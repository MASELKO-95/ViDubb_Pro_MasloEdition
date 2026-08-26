#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gc
import argparse
import subprocess
import shutil
import requests
import torch
from torch.serialization import add_safe_globals

# ====================== XTTS FIX ======================
print("🔧 Applying PyTorch safe globals fix for XTTS...")

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.tts.utils.speakers import SpeakerManager
    add_safe_globals([XttsConfig, BaseDatasetConfig, XttsAudioConfig, XttsArgs, SpeakerManager])
except Exception:
    pass

if not hasattr(torch.serialization, '_original_load'):
    torch.serialization._original_load = torch.load
    def _trusted_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return torch.serialization._original_load(*args, **kwargs)
    torch.load = _trusted_load
# =====================================================

from faster_whisper import WhisperModel
from TTS.api import TTS
from pydub import AudioSegment

# ========================= DEVICE =========================
GPU_AVAILABLE = torch.cuda.is_available()

def device_for_model(size_gb):
    if not GPU_AVAILABLE:
        return "cpu"
    return "cuda" if size_gb <= 7 else "cpu"

WHISPER_DEVICE = device_for_model(6)
XTTS_DEVICE = device_for_model(2)

print("\n=== DEVICE SETUP ===")
print("Whisper:", WHISPER_DEVICE)
print("XTTS:", XTTS_DEVICE)
print("====================\n")

# ========================= CLI =========================
parser = argparse.ArgumentParser()
parser.add_argument("--yt_url", type=str)
parser.add_argument("--video_url", type=str)
parser.add_argument("--source_language", required=True)
parser.add_argument("--target_language", required=True)
parser.add_argument("--keep_background", action="store_true", help="Keep original background sound")
args = parser.parse_args()

# ========================= HELPERS =========================
def download_video():
    if args.yt_url:
        subprocess.run(["yt-dlp", "-f", "best", "-o", "video.mp4", args.yt_url], check=True)
        return "video.mp4"
    return args.video_url

def extract_audio(video):
    subprocess.run(["ffmpeg", "-y", "-i", video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "audio.wav"], check=True)

# ========================= TRANSCRIPTION =========================
def transcribe():
    print("Loading Whisper...")
    model = WhisperModel("turbo", device=WHISPER_DEVICE, compute_type="int8")
    segments, _ = model.transcribe("audio.wav", vad_filter=True)
    text = [s.text.strip() for s in segments if s.text.strip()]
    del model
    gc.collect()
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
    return text

# ========================= TRANSLATION =========================
def translate_batch(texts, src, tgt):
    print("Translating with Ollama...")
    joined = "\n".join(texts)
    prompt = f"""You are a professional translator. Translate from {src} to {tgt} naturally.
Return exactly the same number of lines.

Text:
{joined}
"""

    try:
        r = requests.post("http://localhost:11434/api/generate",
            json={"model": "translategemma:27b", "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
            timeout=300)
        r.raise_for_status()
        out = [line.strip() for line in r.json()["response"].strip().split("\n") if line.strip()]
        return out if len(out) == len(texts) else texts
    except Exception as e:
        print("Translation failed:", e)
        return texts
    finally:
        print("🧹 Unloading Ollama...")
        try:
            requests.post("http://localhost:11434/api/generate", json={"model": "translategemma:27b", "keep_alive": 0}, timeout=20)
        except:
            pass

# ========================= TTS =========================
def load_tts():
    print("Loading XTTS...")
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(XTTS_DEVICE == "cuda"))

def synthesize(lines, lang, tts):
    os.makedirs("chunks", exist_ok=True)
    files = []
    for i, line in enumerate(lines):
        path = f"chunks/{i}.wav"
        print(f"Synthesizing {i+1}/{len(lines)}...")
        tts.tts_to_file(text=line, speaker_wav="audio.wav", language=lang, file_path=path)
        files.append(path)
    return files

# ========================= AUDIO PROCESSING =========================
def merge_audio(files):
    print("Merging synthesized chunks...")
    combined = AudioSegment.empty()
    for f in files:
        combined += AudioSegment.from_file(f)
    combined.export("dub.wav", format="wav")

def mix_audio(keep_background=False):
    print("Mixing audio...")
    dub = AudioSegment.from_wav("dub.wav")

    if keep_background and os.path.exists("video.mp4"):
        subprocess.run([
            "ffmpeg", "-y", "-i", "video.mp4", "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "original_audio.wav"
        ], check=True)

        orig = AudioSegment.from_wav("original_audio.wav")
        quieter_orig = orig - 8                     # reduce background by 8dB
        mixed = quieter_orig.overlay(dub, position=0)

        mixed.export("final_dub.wav", format="wav")
        print("✅ Background audio preserved (quieted by 8dB)")
    else:
        dub.export("final_dub.wav", format="wav")
        print("✅ Using clean dub audio")

# ========================= LIP SYNC =========================
def apply_lip_sync():
    print("🎙 Applying lip sync with Wav2Lip...")
    if not os.path.exists("Wav2Lip/inference.py"):
        print("⚠️ Wav2Lip not found → simple mux")
        subprocess.run(["ffmpeg", "-y", "-i", "video.mp4", "-i", "final_dub.wav",
                        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "output.mp4"], check=True)
        return

    try:
        subprocess.run([
            "python", "Wav2Lip/inference.py",
            "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip_gan.pth",
            "--face", "video.mp4",
            "--audio", "final_dub.wav",
            "--outfile", "output.mp4",
            "--pads", "0", "20", "0", "0",
            "--wav2lip_batch_size", "8"
        ], check=True)
        print("✅ Lip sync completed!")
    except Exception as e:
        print("Lip sync failed:", e)
        print("Falling back to simple mux...")
        subprocess.run(["ffmpeg", "-y", "-i", "video.mp4", "-i", "final_dub.wav",
                        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "output.mp4"], check=True)

# ========================= CLEANUP =========================
def cleanup():
    shutil.rmtree("chunks", ignore_errors=True)
    for f in ["audio.wav", "dub.wav", "final_dub.wav", "original_audio.wav"]:
        if os.path.exists(f):
            os.remove(f)

# ========================= MAIN =========================
def main():
    video = download_video()
    extract_audio(video)

    print("Transcribing...")
    text = transcribe()

    print("Translating...")
    translated = translate_batch(text, args.source_language, args.target_language)

    tts = load_tts()
    print("Synthesizing with voice cloning...")
    files = synthesize(translated, args.target_language, tts)

    merge_audio(files)
    mix_audio(keep_background=args.keep_background)
    apply_lip_sync()

    cleanup()
    print("\n🎉 DONE! Output saved as → output.mp4")

if __name__ == "__main__":
    main()
