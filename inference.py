import os
import gc
import argparse
import subprocess
import shutil
# Model compatibility setup must run before importing model-backed libraries.
# ruff: noqa: E402
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

from faster_whisper import WhisperModel
from TTS.api import TTS
from pydub import AudioSegment

# ========================= DEVICE =========================
GPU_AVAILABLE = torch.cuda.is_available()

WHISPER_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
XTTS_DEVICE = "cpu"  # Keeps XTTS usable on GPUs with about 8 GB of VRAM.

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
parser.add_argument("--keep_background", action="store_true")
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
    torch.cuda.empty_cache()
    return text

# ========================= TRANSLATION (ELASTYCZNE DOPASOWANIE) =========================
def translate_batch(texts, src, tgt):
    print(f"Translating from {src} → {tgt} with Ollama...")
    joined = "\n".join([f"[{i+1}] {line}" for i, line in enumerate(texts)])

    prompt = f"""You are a professional translator. Translate the following {src} text into {tgt}.

STRICT RULES:
- Output ONLY the translated lines, one per line.
- Do NOT add any numbers, labels, explanations, or extra text.
- Do NOT include the original text.
- Preserve the meaning and style.

{src} text:
{joined}

{tgt} translation:"""

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "microai/suzume-llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 16384,
                    "top_p": 0.95
                }
            },
            timeout=360
        )
        r.raise_for_status()
        response = r.json()["response"].strip()


        print("\n=== SUROWA ODPOWIEDŹ OLLAMY ===")
        print(response)
        print("=== KONIEC ODPOWIEDZI ===\n")

        # =============== CZYSZCZENIE ===============
        lines = response.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue


            if line and line[0].isdigit() and len(line) > 1 and line[1] in ('.', ')', ']', ' '):
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    line = parts[1].strip()
                else:
                    continue


            if line.startswith('(') or line.startswith('['):
                if line.endswith(')') or line.endswith(']'):
                    continue
                line = line.lstrip('([').strip()


            if len(line) < 2 and not any(c.isalpha() for c in line):
                continue

            cleaned.append(line)

        # =============== ELASTYCZNE DOPASOWANIE ===============

        if len(cleaned) < len(texts):
            print(f"⚠️ Model zwrócił tylko {len(cleaned)} linii (oczekiwano {len(texts)}).")
            print("   Uzupełniam brakujące linie oryginalnym tekstem japońskim.")

            final_translation = []
            for i in range(len(texts)):
                if i < len(cleaned):
                    final_translation.append(cleaned[i])
                else:
                    final_translation.append(texts[i])  # oryginalny japoński tekst

            cleaned = final_translation


        elif len(cleaned) > len(texts):
            print(f"⚠️ Model zwrócił za dużo linii ({len(cleaned)}), przycinam do {len(texts)}.")
            cleaned = cleaned[:len(texts)]

        print("✅ Translation completed (missing entries were filled in).")


        for orig, trans in zip(texts[:3], cleaned[:3]):
            print(f"JP: {orig[:80]}...")
            print(f"PL: {trans[:80]}...\n")

        return cleaned

    except Exception as e:
        print(f"Translation failed: {e}")
        return texts
    finally:

        print("🧹 Releasing the Ollama model...")
        try:
            requests.post("http://localhost:11434/api/generate",
                         json={"model": "microai/suzume-llama3", "keep_alive": 0}, timeout=10)
        except requests.RequestException:
            pass
        gc.collect()
        torch.cuda.empty_cache()

# ========================= TTS =========================
def load_tts():
    print("Loading XTTS on CPU...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    return tts

def synthesize(lines, lang, tts):
    os.makedirs("chunks", exist_ok=True)
    files = []
    for i, line in enumerate(lines):
        path = f"chunks/{i}.wav"
        print(f"Synthesizing {i+1}/{len(lines)}...")
        tts.tts_to_file(text=line, speaker_wav="audio.wav", language=lang, file_path=path)
        files.append(path)
    return files

# ========================= AUDIO MIX =========================
def merge_audio(files):
    print("Merging audio...")
    combined = AudioSegment.empty()
    for f in files:
        combined += AudioSegment.from_file(f)
    combined.export("dub.wav", format="wav")

def mix_audio(keep_background=False):
    print("Mixing final audio...")
    dub = AudioSegment.from_wav("dub.wav")
    if keep_background and os.path.exists("video.mp4"):
        subprocess.run(["ffmpeg", "-y", "-i", "video.mp4", "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "original_audio.wav"], check=True)
        bg = AudioSegment.from_wav("original_audio.wav") - 6
        mixed = bg.overlay(dub.normalize())
        mixed.export("final_dub.wav", format="wav")
        print("✅ Background kept")
    else:
        dub.export("final_dub.wav", format="wav")

# ========================= LIP SYNC & CLEANUP =========================
def apply_lip_sync():
    print("Applying lip sync...")
    if not os.path.exists("Wav2Lip/inference.py"):
        subprocess.run(["ffmpeg", "-y", "-i", "video.mp4", "-i", "final_dub.wav", "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "output.mp4"], check=True)
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
        print("✅ Lip sync done")
    except subprocess.CalledProcessError:
        subprocess.run(["ffmpeg", "-y", "-i", "video.mp4", "-i", "final_dub.wav", "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "output.mp4"], check=True)

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

    # Optionally save the translated text to a file.
    with open("translated_lines.txt", "w", encoding="utf-8") as f:
        for line in translated:
            f.write(line + "\n")

    tts = load_tts()
    print("Synthesizing...")
    files = synthesize(translated, args.target_language, tts)

    merge_audio(files)
    mix_audio(keep_background=args.keep_background)
    apply_lip_sync()

    cleanup()
    print("\n🎉 DONE! → output.mp4")

if __name__ == "__main__":
    main()
