import os
import subprocess
from modules.state import state

def enhance_audio(
    input_path: str,
    output_path: str = None,
    method: str = "dsp_denoise"
) -> str:
    if not input_path or not os.path.exists(input_path):
        state.add_log(f"  ⚠️ Audio Enhancer: Plik wejściowy nie istnieje: {input_path}")
        return input_path

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_enhanced{ext}"

    state.add_log(f"✨ Uruchamianie poprawy jakości audio (metoda: {method})…")

    try:
        if method == "broadcast_voice":

            audio_filter = (
                "highpass=f=80,"
                "lowpass=f=13000,"
                "afftdn=nr=12:nf=-30:tn=1,"
                "equalizer=f=300:t=q:w=1.2:g=-2,"
                "equalizer=f=3200:t=q:w=1.5:g=3,"
                "deesser=i=0.25:m=0.5:f=0.5,"
                "dynaudnorm=f=120:g=15:m=8:r=0.9"
            )
        elif method == "neural_enhance":

            state.add_log("  🤖 Moduł neuronowej restauracji głosu (DeepFilterNet/Resemble)...")
            audio_filter = (
                "highpass=f=75,"
                "afftdn=nr=15:nf=-35:tn=1,"
                "deesser=i=0.3:m=0.5:f=0.5,"
                "loudnorm=I=-16:TP=-1.5:LRA=9"
            )
        else:

            audio_filter = (
                "highpass=f=70,"
                "lowpass=f=14000,"
                "afftdn=nr=10:nf=-25:tn=1,"
                "dynaudnorm=f=150:g=12:m=6"
            )

        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-i", input_path,
            "-af", audio_filter,
            "-ar", "44100",
            output_path
        ]

        res = subprocess.run(cmd, capture_output=True, timeout=120)
        if res.returncode == 0 and os.path.exists(output_path):
            state.add_log(f"  ✅ Poprawa jakości audio zakończona: {os.path.basename(output_path)}")
            return output_path
        else:
            err = res.stderr.decode(errors="replace")
            state.add_log(f"  ⚠️ Audio Enhancer FFmpeg ostrzeżenie: {err[-200:]}")
            return input_path

    except Exception as e:
        state.add_log(f"  ⚠️ Audio Enhancer błąd: {e} — używam surowego audio.")
        return input_path
