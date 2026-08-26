import sys
import importlib

packages = [
    "pyannote.audio",
    "speechbrain",
    "TTS",
    "deepface",
    "audio_separator",
    "onnxruntime",
    "gradio",
    "tensorflow",
    "librosa",
    "whisper",
]

missing = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except ImportError as e:
        print(f"[FAILED] {pkg}: {e}")
        missing.append(pkg)

if not missing:
    print("\nAll key packages are present!")
    sys.exit(0)
else:
    print(f"\nMissing packages: {', '.join(missing)}")
    sys.exit(1)
