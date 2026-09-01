import os
import gc
import torch
import gradio as gr
import subprocess
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import tempfile
# XTTS compatibility setup must run before importing TTS.
# ruff: noqa: E402

# ====================== XTTS SAFE FIX ======================
from torch.serialization import add_safe_globals

print("🔧 Applying the PyTorch safe-globals compatibility fix for XTTS...")

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.tts.utils.speakers import SpeakerManager

    add_safe_globals([XttsConfig, BaseDatasetConfig, XttsAudioConfig, XttsArgs, SpeakerManager])
    print("✅ Added XTTS classes to the safe-globals list.")
except Exception as e:
    print(f"⚠️ Could not add XTTS classes to the safe-globals list: {e}")
    print("   Falling back to a torch.load compatibility wrapper.")
    original_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = safe_load

# Import TTS only after applying the compatibility fix.
from TTS.api import TTS

# Optional separator used to remove background audio.
try:
    from audio_separator.separator import Separator
    separator_available = True
except ImportError:
    separator_available = False
    print("⚠️ audio-separator is not installed; background removal is unavailable.")

# ========================= CONFIG =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if DEVICE == "cuda":
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {free_mem:.2f} GB free / {total_mem:.2f} GB total")
    if free_mem < 4:
        print("⚠️ Low VRAM, consider using CPU mode")

# ========================= PREPROCESSING =========================
def trim_silence(audio_path, silence_thresh=-50, min_silence_len=500, padding=200):
    print("✂️ Wycinanie ciszy...")
    audio = AudioSegment.from_file(audio_path)
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if len(nonsilent_ranges) == 0:
        print("⚠️ Nie wykryto mowy – zwracam oryginał.")
        return audio_path

    start_trim = max(0, nonsilent_ranges[0][0] - padding)
    end_trim = min(len(audio), nonsilent_ranges[-1][1] + padding)
    trimmed = audio[start_trim:end_trim]

    out_path = tempfile.NamedTemporaryFile(suffix="_trimmed.wav", delete=False).name
    trimmed.export(out_path, format="wav")
    print(f"✅ Cisza wycięta, nowy plik: {out_path}")
    return out_path

def remove_background(audio_path, model_name="UVR-MDX-NET-Inst_HQ_3.onnx"):

    if not separator_available:
        print("⚠️ audio-separator niedostępne – pomijam usuwanie tła.")
        return audio_path

    print(f"🎵 Usuwanie tła przy użyciu modelu {model_name}...")
    separator = Separator()
    separator.load_model(model_filename=model_name)

    output_files = separator.separate(audio_path)
    vocal_file = None
    for f in output_files:
        if "vocal" in f.lower() or "no_instrumental" in f.lower():
            vocal_file = f
            break
    if vocal_file is None and len(output_files) > 0:

        vocal_file = output_files[0]

    if vocal_file and os.path.exists(vocal_file):
        print(f"✅ Wokal wyodrębniony: {vocal_file}")
        return vocal_file
    else:
        print("⚠️ Nie znaleziono pliku z wokalem – zwracam oryginał.")
        return audio_path

def preprocess_audio(input_path, remove_bg=True, trim=True):
    if input_path is None:
        return None

    current = input_path
    if remove_bg:
        current = remove_background(current)
    if trim:
        current = trim_silence(current)

    return current

# ========================= CLONING =========================
def clone_voice(reference_audio, text, language="pl", device=DEVICE,
                remove_background=True, trim_silence=True, output_path="cloned.wav"):

    if reference_audio is None:
        return None, "❌ Brak pliku referencyjnego."

    print(f"🎙️ Referencja: {reference_audio}")
    print("🔧 Preprocessing referencji...")
    cleaned_ref = preprocess_audio(reference_audio, remove_bg=remove_background, trim=trim_silence)
    if cleaned_ref is None:
        return None, "❌ Preprocessing nie powiódł się."

    print(f"📝 Tekst: {text}")
    print(f"🌐 Język: {language}")
    print(f"⚙️ Urządzenie: {device}")

    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
        tts.tts_to_file(
            text=text,
            speaker_wav=cleaned_ref,
            language=language,
            file_path=output_path
        )
        del tts
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        return output_path, f"✅ Głos sklonowany pomyślnie do {output_path}"

    except Exception as e:
        return None, f"❌ Błąd: {str(e)}"

def extract_audio_from_video(video_path, output_audio="extracted_audio.wav"):
    """Extract audio from a video file with FFmpeg."""
    if not video_path:
        return None, "No video provided."
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", output_audio
        ], check=True, capture_output=True)
        return output_audio, "Audio extracted successfully."
    except subprocess.CalledProcessError as e:
        return None, f"FFmpeg error: {e.stderr.decode()}"

# ========================= GRADIO UI =========================
css = """
.gradio-container { background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; }
.glass-panel { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; }
.btn-primary { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(99,102,241,0.3); }
"""

with gr.Blocks(css=css, title="🎤 Voice Cloner Pro") as demo:
    gr.Markdown("""
    # Clean the sample before using it as the XTTS voice reference.
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="glass-panel"):
            gr.Markdown("## 1. Źródło głosu")
            audio_input = gr.Audio(label="Nagraj lub wgraj plik audio", type="filepath")
            video_input = gr.Video(label="Lub wideo (wyodrębnij audio)", interactive=True)
            extract_btn = gr.Button("🎬 Wyodrębnij audio z wideo", elem_classes="btn-secondary")
            ref_audio = gr.State()

            gr.Markdown("## 2. Opcje czyszczenia")
            with gr.Row():
                remove_bg_check = gr.Checkbox(label="Usuń tło (separacja wokalu)", value=True,
                                             info="Wymaga audio-separator")
                trim_check = gr.Checkbox(label="Wytnij ciszę", value=True)

            gr.Markdown("## 3. Tekst i ustawienia")
            text_input = gr.TextArea(label="Tekst do wypowiedzenia", lines=4,
                                      placeholder="Wpisz tekst...")
            language = gr.Dropdown(
                choices=["pl", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh-cn"],
                value="pl", label="Język"
            )
            device_choice = gr.Radio(choices=["cuda", "cpu"], value=DEVICE, label="Urządzenie")

            clone_btn = gr.Button("✨ Klonuj głos", variant="primary", elem_classes="btn-primary")

        with gr.Column(scale=1, elem_classes="glass-panel"):
            gr.Markdown("## 4. Wynik")
            output_audio = gr.Audio(label="Sklonowana mowa", type="filepath")
            status = gr.Textbox(label="Status", interactive=False)

    # Logika
    def update_ref_from_audio(audio_file):
        return audio_file

    def extract_and_set(video_file):
        if not video_file:
            return None, "Brak wideo."
        out_path = tempfile.NamedTemporaryFile(suffix="_extracted.wav", delete=False).name
        result, msg = extract_audio_from_video(video_file, out_path)
        return result, msg

    def run_clone(ref, text, lang, device, remove_bg, trim):
        if not ref:
            return None, "❌ Najpierw podaj referencję głosu."
        if not text.strip():
            return None, "❌ Wpisz tekst."

        out = tempfile.NamedTemporaryFile(suffix="_cloned.wav", delete=False).name
        result, msg = clone_voice(ref, text, lang, device, remove_bg, trim, out)
        return result, msg

    audio_input.change(update_ref_from_audio, inputs=[audio_input], outputs=[ref_audio])
    extract_btn.click(extract_and_set, inputs=[video_input], outputs=[ref_audio, status])
    clone_btn.click(run_clone,
                    inputs=[ref_audio, text_input, language, device_choice,
                            remove_bg_check, trim_check],
                    outputs=[output_audio, status])

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0")
