🎬 viDubb Pro — Maslo95 Edition

Version 1.0.1a | AI-Powered Video Dubbing & Translation Pipeline



[!WARNING]
This is a test release under active development. Voice recognition,
voice cloning, Timeline Review and Wav2Lip may still require manual review.

Fork of ViDubb by medahmedkrichen, enhanced and customized by Maslo95.

viDubb Pro processes media locally. Uploaded videos, generated audio,
projects and voice profiles are runtime data and are intentionally excluded
from the repository.

🎥 Demo

Example result showing the same source video in three versions:

Original

English Dub

Japanese Dub

🎬 Original audio

🇬🇧 AI English dubbing

🇯🇵 AI Japanese dubbing

docs/demo/original.mp4

docs/demo/dub_english.mp4

docs/demo/dub_japanese.mp4

Demo files

docs/
└── demo/
    ├── original.mp4
    ├── dub_english.mp4
    └── dub_japanese.mp4

GitHub does not always provide a convenient inline player for repository MP4 files.
For larger demo videos, GitHub Releases are recommended.

🖥️ GUI Preview

Launcher



Main interface



Dubbing panel



Recommended screenshot structure:

docs/
└── images/
    ├── launcher.png
    ├── gui-main.png
    └── dubbing-panel.png

✨ Features

🎤 Voice Cloning — XTTS-v2 zero-shot voice cloning for 17+ languages

🗣️ Speaker Diarization — Automatic multi-speaker detection using SpeechBrain ECAPA-TDNN

🌐 AI Translation — Local LLM translation through Ollama

🔊 Edge-TTS Fallback — Microsoft Neural TTS as an alternative TTS engine

🎵 Background Preservation — Vocal/instrumental separation using UVR MDX-Net

📝 Subtitle Editor — Inline editing, approval workflow and ignore markers

🎬 Lip Sync — Optional Wav2Lip integration

🔇 Smart Timing — Timeline-based audio overlay for better synchronization

🎛️ Audio Enhancement — DSP denoising, EQ and voice processing

💻 Local Processing — No cloud API is required for the default local workflow

🚀 Quick Start

Requirements

Python 3.10 recommended

CUDA-compatible GPU recommended, preferably 8 GB+ VRAM

FFmpeg installed and available in PATH

Ollama with a translation model for local AI translation

Installation

git clone https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition.git
cd ViDubb_Pro_MasloEdition

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Running

Option 1 — GUI Launcher

python run_launcher.py

Option 2 — Direct Flask server

python app_new.py

Open:

http://127.0.0.1:7860

The first run may download several AI models.

Model weights, virtual environments and generated media are not included in Git.

🧹 Cache Cleanup

The cleanup script removes:

__pycache__

.pyc

.pyo

temporary tool caches

It does not remove:

.venv

projects

videos

the voice database

Run:

python scripts/clean_cache.py

🎙️ Voice Dataset Builder

The project includes a separate tool for building speaker datasets from long audio or video recordings.

It can:

split recordings into dialogue clips

transcribe speech

compare speaker embeddings using ECAPA

classify samples as accepted, review or rejected

register verified voices in the Voice Database

GUI mode

python tools/voice_dataset_builder.py --gui

Terminal mode

python tools/voice_dataset_builder.py recording1.mp4 recording2.wav \
  --name Yuuki_Takada \
  --reference reference_sample.wav \
  --language ja \
  --register

Generated datasets are stored in:

voice_datasets/<name>/

Additional runs append clips to the existing manifest.

Without a reference file or --voice-id, clips are intentionally placed in
review because the application has no reliable basis for confirming speaker identity.

voice_db/ and speakers_audio/ start empty after cloning.
The application automatically creates the voice database index when required.

👄 Wav2Lip Setup

Wav2Lip is optional and is only required for lip synchronization.

Download:

wav2lip_gan.pth

Place it in:

Wav2Lip/checkpoints/

See the official project:

Wav2Lip

🏗️ Architecture

viDubb Pro/
├── app_new.py
├── run_launcher.py
├── modules/
│   ├── app.py
│   ├── config.py
│   ├── state.py
│   ├── routes/
│   │   ├── dubbing.py
│   │   ├── projects.py
│   │   ├── translate.py
│   │   └── video.py
│   ├── services/
│   │   ├── audio_enhancer.py
│   │   ├── diarization_service.py
│   │   ├── tts_service.py
│   │   ├── video_service.py
│   │   └── whisper_service.py
│   └── utils/
├── Wav2Lip/
├── templates/
│   └── index.html
└── static/

🌍 Supported Languages

English

Polish

Spanish

French

German

Italian

Turkish

Russian

Dutch

Czech

Arabic

Chinese (Simplified)

Japanese

Korean

Hindi

🔧 Configuration

Setting

Default

Description

TTS Engine

Edge-TTS

edge or xtts

Translation

Ollama

Local LLM translation

Whisper Model

turbo

tiny, base, small, medium, large-v3, turbo

Audio Enhancement

Off

DSP denoising and voice processing

🧪 Testing

Install lightweight test dependencies:

pip install -r requirements-test.txt

Run tests:

pytest -q

GitHub Actions automatically runs tests and syntax checks on pushes and pull requests.

The workflow can also be started manually from:

Actions → Tests → Run workflow

The 🧪 TEST button inside viDubb Pro creates a diagnostic report containing recent logs.

Before saving the report, the application masks:

tokens

the local home-directory path

Attach the report to a bug report together with the steps required to reproduce the issue.

🐛 Bug Reports & Community

GitHub Issues

GitHub Discussions

Discord: Lone wolf® (formerly MASELKO-95®)

Username: maselko95

☕ Support

If viDubb Pro is useful to you, you can support development here:

Buy me a coffee

📜 Credits

Original Project: ViDubb by medahmedkrichen

Wav2Lip: Rudrabha/Wav2Lip

Coqui TTS: coqui-ai/TTS

Faster-Whisper: SYSTRAN/faster-whisper

SpeechBrain: speechbrain/speechbrain

📄 License

MIT License — see LICENSE for details.
