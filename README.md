<p align="center">

[![Downloads](https://img.shields.io/github/downloads/MASELKO-95/ViDubb_Pro_MasloEdition/total?style=for-the-badge&logo=github&label=Downloads)](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/releases)

[![Visitors](https://hits.sh/github.com/MASELKO-95/ViDubb_Pro_MasloEdition.svg?style=for-the-badge&label=Visitors&logo=github)](https://hits.sh/github.com/MASELKO-95/ViDubb_Pro_MasloEdition/)

</p>
# 🎬 viDubb Pro — Maslo95 Edition

**Version 1.0.1a** | AI-Powered Video Dubbing, Translation & Voice Cloning Pipeline

[![Tests](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/actions/workflows/tests.yml/badge.svg)](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/actions/workflows/tests.yml)

> [!WARNING]
> **Test release under active development.**
>
> Voice recognition, voice cloning, Timeline Review and Wav2Lip may still
> require manual review. Generated dubbing quality depends on the source audio,
> selected model and voice reference quality.

Fork of [ViDubb](https://github.com/medahmedkrichen/ViDubb) by
[medahmedkrichen](https://github.com/medahmedkrichen), enhanced and customized
by [Maslo95](https://github.com/MASELKO-95).

> viDubb Pro processes media locally by default.
> Uploaded videos, generated audio, projects and voice profiles are runtime data
> and are intentionally excluded from the repository.

---

# 🎥 Demo

See viDubb Pro in action.

The same source video is presented below as:

- 🎬 Original source
- 🇬🇧 English AI dub
- 🇯🇵 Japanese AI dub

---

## 🎬 Original Video

Original source video before translation and dubbing.

[▶ **Watch / Download Original Video**](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/releases/download/Media/original.mp4)

---

## 🇬🇧 English AI Dub

English translation and AI-generated dubbing created with viDubb Pro.

[▶ **Watch / Download English Dub**](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/releases/download/Media/dub_english.mp4)

---

## 🇯🇵 Japanese AI Dub

Japanese translation and AI-generated dubbing created with viDubb Pro.

[▶ **Watch / Download Japanese Dub**](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/releases/download/Media/dub_japanese.mp4)

---

### 📦 Demo Media Release

All demo media is available in the dedicated GitHub Release:

[**Open viDubb Pro Demo Media**](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/releases/tag/Media)

> The demo shows translation, voice synthesis, dialogue timing,
> voice cloning and preservation of the original background audio.

---

# 🖥️ GUI Preview

## Launcher

The launcher verifies dependencies, checks for updates and starts the local
viDubb Pro server.

![viDubb Pro Launcher](https://github.com/user-attachments/assets/c26cb2b3-0dac-4ea7-abe1-b50d0bf7979f)

---

## Main Interface

The main interface provides:

- video and subtitle loading
- Whisper transcription
- speaker diarization
- subtitle editing
- AI translation
- speaker / voice assignment
- timeline controls
- project management
- voice database access
- dubbing generation

![viDubb Pro Main Interface](https://github.com/user-attachments/assets/4d057be4-f45d-4f22-9123-09b6fdce237b)

---

## Dubbing & Voice Generation Panel

The dubbing panel allows you to configure:

- target dubbing language
- dubbing or voiceover mode
- XTTS-v2 voice cloning
- reference voices
- original background audio preservation
- hardcoded subtitles
- optional Wav2Lip lip synchronization
- Whisper validation
- AI / DSP audio enhancement
- output path
- Timeline Review approval

![viDubb Pro Dubbing Panel](https://github.com/user-attachments/assets/c939cbff-5884-4a20-b4e8-e46b6a8969a7)

---

# ✨ Features

### 🎤 Voice Cloning

XTTS-v2 zero-shot voice cloning allows viDubb Pro to generate translated speech
while preserving characteristics of the original speaker's voice.

### 🗣️ Speaker Diarization

Automatic multi-speaker detection and speaker matching using
**SpeechBrain ECAPA-TDNN**.

No Hugging Face access token is required for the default ECAPA-based workflow.

### 🌐 AI Translation

Local AI translation through **Ollama**.

The translation pipeline can use a local LLM, so no cloud API is required for
the default workflow.

### 🔊 Multiple TTS Engines

Supported engines include:

- **XTTS-v2** — voice cloning
- **Edge-TTS** — fallback neural TTS

> Edge-TTS is an optional fallback and may require an internet connection.

### 🎵 Background Audio Preservation

Dialogue can be separated from background audio using UVR / MDX-Net based
processing.

The generated speech can then be mixed back with the original:

- music
- ambience
- sound effects
- background audio

### 📝 Subtitle Editor

Integrated subtitle editor with:

- inline editing
- translation editing
- speaker assignment
- voice assignment
- approval workflow
- confidence information
- ignore markers
- timeline editing

### 🎬 Lip Sync

Optional **Wav2Lip** integration synchronizes mouth movement with newly
generated speech.

### 🔇 Smart Timing

Timeline-based audio generation and overlay helps keep generated dialogue
synchronized with the original video.

### 🎛️ Audio Enhancement

Optional post-processing includes:

- denoising
- de-rumble filtering
- EQ
- voice enhancement
- broadcast-style processing

### 💾 Project System

Projects can preserve work between sessions, including edited subtitles,
translations and speaker assignments.

### 🎙️ Voice Database

Reusable voice profiles can be stored and assigned to detected speakers.

### 🧪 Diagnostic Reports

The built-in `🧪 TEST` button generates a diagnostic report with recent logs.

Sensitive information such as tokens and local home-directory paths is masked
before the report is exported.

---

# 🚀 Quick Start

## Requirements

Recommended environment:

- **Python 3.10**
- **FFmpeg**
- CUDA-compatible NVIDIA GPU
- **8 GB+ VRAM recommended**
- Ollama for local AI translation
- Linux or Windows

Some AI dependencies may not support newer Python versions correctly, so
Python 3.10 is strongly recommended.

---

# 📥 Installation

Clone the repository:

```bash
git clone https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition.git
cd ViDubb_Pro_MasloEdition
```

Create a virtual environment:

```bash
python -m venv .venv
```

## Linux / macOS

```bash
source .venv/bin/activate
```

## Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running viDubb Pro

## Option 1 — GUI Launcher

Recommended:

```bash
python run_launcher.py
```

The launcher can:

- verify dependencies
- install missing dependencies
- check for updates
- launch the local server

---

## Option 2 — Direct Flask Server

```bash
python app_new.py
```

Then open:

```text
http://127.0.0.1:7860
```

in your browser.

> [!NOTE]
> The first run may download several AI models.
> Model weights are not included directly in the Git repository.

---

# 🤖 Ollama Setup

viDubb Pro can use a local Ollama model for translation.

Install Ollama:

https://ollama.com/

Start the Ollama server:

```bash
ollama serve
```

Download or run your preferred translation-capable model, for example:

```bash
ollama run llama3.1
```

The exact model can be configured in viDubb Pro.

---

# 🎙️ Voice Dataset Builder

viDubb Pro includes a separate tool for building reusable speaker datasets from
long audio and video recordings.

The tool can:

- scan long recordings
- detect speech segments
- split recordings into dialogue clips
- transcribe speech
- compare speaker embeddings
- classify voice samples
- build reusable speaker datasets
- register verified voices in the Voice Database

Samples can be classified as:

```text
accepted
review
rejected
```

---

## Voice Dataset Builder — GUI

Run:

```bash
python tools/voice_dataset_builder.py --gui
```

---

## Voice Dataset Builder — Terminal

Example:

```bash
python tools/voice_dataset_builder.py recording1.mp4 recording2.wav \
  --name Yuuki_Takada \
  --reference reference_sample.wav \
  --language ja \
  --register
```

Generated datasets are stored in:

```text
voice_datasets/<name>/
```

For example:

```text
voice_datasets/Yuuki_Takada/
```

Subsequent runs can append new clips to the existing manifest.

Using:

```text
--register
```

also creates a compatible voice profile in the viDubb Pro Voice Database.

> [!IMPORTANT]
> Without a trusted reference file or `--voice-id`, the program intentionally
> sends clips to `review`, because there is not enough evidence to automatically
> confirm speaker identity.

---

# 🗂️ Voice Database

After cloning the repository:

```text
voice_db/
speakers_audio/
```

start empty.

The application creates the required Voice Database index automatically when it
is first needed.

Runtime voice data is intentionally excluded from Git.

---

# 👄 Wav2Lip Setup

Wav2Lip is optional.

It is only required when lip synchronization is enabled.

Download:

```text
wav2lip_gan.pth
```

and place it in:

```text
Wav2Lip/checkpoints/
```

Expected structure:

```text
Wav2Lip/
└── checkpoints/
    └── wav2lip_gan.pth
```

Official Wav2Lip repository:

https://github.com/Rudrabha/Wav2Lip

---

# 🧹 Cache Cleanup

viDubb Pro includes a cleanup script:

```bash
python scripts/clean_cache.py
```

It removes:

```text
__pycache__
*.pyc
*.pyo
temporary tool caches
```

It does **not** delete:

```text
.venv/
projects/
videos/
voice_db/
voice_datasets/
generated media
```

---

# 🏗️ Architecture

```text
viDubb Pro/
│
├── app_new.py
│   └── Flask server entry point
│
├── run_launcher.py
│   └── Tkinter GUI launcher
│
├── modules/
│   │
│   ├── app.py
│   │   └── Flask application factory
│   │
│   ├── config.py
│   │   └── Application configuration
│   │
│   ├── state.py
│   │   └── Global application state
│   │
│   ├── routes/
│   │   │
│   │   ├── dubbing.py
│   │   │   └── Dubbing generation API
│   │   │
│   │   ├── projects.py
│   │   │   └── Project management
│   │   │
│   │   ├── translate.py
│   │   │   └── AI translation
│   │   │
│   │   └── video.py
│   │       └── Video upload, transcription and diarization
│   │
│   ├── services/
│   │   │
│   │   ├── audio_enhancer.py
│   │   │   └── Audio post-processing
│   │   │
│   │   ├── diarization_service.py
│   │   │   └── Speaker identification
│   │   │
│   │   ├── tts_service.py
│   │   │   └── XTTS-v2 and Edge-TTS synthesis
│   │   │
│   │   ├── video_service.py
│   │   │   └── Background separation and video muxing
│   │   │
│   │   └── whisper_service.py
│   │       └── Faster-Whisper transcription
│   │
│   └── utils/
│       └── Shared utilities
│
├── tools/
│   └── voice_dataset_builder.py
│
├── scripts/
│   └── clean_cache.py
│
├── Wav2Lip/
│   └── Lip synchronization engine
│
├── templates/
│   └── index.html
│
├── static/
│   └── CSS, fonts and frontend assets
│
├── voice_db/
│   └── Runtime voice profiles
│
├── voice_datasets/
│   └── Generated speaker datasets
│
└── speakers_audio/
    └── Runtime speaker samples
```

---

# 🌍 Supported Languages

viDubb Pro currently exposes support for:

| Language | Code |
|---|---|
| 🇬🇧 English | `en` |
| 🇵🇱 Polish | `pl` |
| 🇪🇸 Spanish | `es` |
| 🇫🇷 French | `fr` |
| 🇩🇪 German | `de` |
| 🇮🇹 Italian | `it` |
| 🇹🇷 Turkish | `tr` |
| 🇷🇺 Russian | `ru` |
| 🇳🇱 Dutch | `nl` |
| 🇨🇿 Czech | `cs` |
| 🇸🇦 Arabic | `ar` |
| 🇨🇳 Chinese | `zh` |
| 🇯🇵 Japanese | `ja` |
| 🇰🇷 Korean | `ko` |
| 🇮🇳 Hindi | `hi` |

Actual voice cloning support may depend on the selected TTS engine.

---

# 🔧 Configuration

| Setting | Default | Description |
|---|---|---|
| TTS Engine | Edge-TTS | `edge` or `xtts` |
| Voice Cloning | XTTS-v2 | Zero-shot speaker voice cloning |
| Translation | Ollama | Local LLM translation |
| Whisper Model | `turbo` | Speech recognition model |
| Diarization | Auto | Automatic speaker detection |
| Audio Enhancement | Off | DSP voice processing |
| Lip Sync | Off | Optional Wav2Lip processing |

---

# 🎧 Whisper Models

Available Whisper model options include:

```text
tiny
base
small
medium
large-v3
turbo
```

The default configuration uses:

```text
turbo
```

Larger models may improve recognition quality but require more VRAM and
processing time.

---

# 🔄 Dubbing Pipeline

A typical viDubb Pro workflow looks like this:

```text
Video
  │
  ▼
Audio extraction
  │
  ▼
Whisper transcription
  │
  ▼
Speaker diarization
  │
  ▼
Subtitle / transcript editor
  │
  ▼
AI translation
  │
  ▼
Voice assignment
  │
  ▼
XTTS-v2 / TTS generation
  │
  ▼
Audio timing & synchronization
  │
  ├───────────────┐
  │               │
  ▼               ▼
Background     Optional
audio mix      Wav2Lip
  │               │
  └───────┬───────┘
          ▼
     Final video
```

---

# ✅ Timeline Review

Before rendering, viDubb Pro can require manual approval of generated dialogue.

This allows the user to inspect:

- translation
- speaker detection
- assigned voice
- timing
- subtitle text
- generated speech

before final video rendering.

This is especially useful for longer multi-speaker videos where fully automatic
processing may require corrections.

---

# 🧪 Testing

Install lightweight test dependencies:

```bash
pip install -r requirements-test.txt
```

Run tests:

```bash
pytest -q
```

GitHub Actions automatically performs tests and syntax checks after pushes and
for pull requests.

The workflow can also be started manually:

```text
Actions
└── Tests
    └── Run workflow
```

---

# 🐛 Diagnostic Reports

The `🧪 TEST` button inside the application generates a diagnostic report with
recent application logs.

Before saving the report, viDubb Pro masks sensitive information including:

- authentication tokens
- local home-directory paths

When reporting an issue, attach the generated report and describe:

1. What you were trying to do
2. What you expected to happen
3. What actually happened
4. Steps required to reproduce the problem

---

# 🐛 Bug Reports

Report bugs here:

[GitHub Issues](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/issues)

---

# 💬 Discussions

Questions, ideas and development discussions:

[GitHub Discussions](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/discussions)

---

# 📦 Releases

Demo media and future release assets can be found here:

[GitHub Releases](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/releases)

Demo media:

[Media Release](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/releases/tag/Media)

---

# 📤 Publishing Changes

Before pushing changes, clean temporary Python files:

```bash
python scripts/clean_cache.py
```

Check repository status:

```bash
git status --short
```

Stage changes:

```bash
git add -A
```

Create a commit:

```bash
git commit -m "Update viDubb Pro"
```

Push:

```bash
git push origin main
```

---

# 🔒 Runtime Data

The following types of data are intentionally excluded from the repository:

- uploaded videos
- generated videos
- generated speech
- extracted speaker audio
- voice profiles
- speaker datasets
- model weights
- virtual environments
- temporary processing files

This keeps the source repository relatively small and prevents personal media
from being accidentally committed.

Large demonstration videos are distributed through GitHub Releases instead.

---

# 💻 Local-First Processing

The primary viDubb Pro pipeline is designed around local tools:

- Faster-Whisper
- Ollama
- XTTS-v2
- SpeechBrain
- FFmpeg
- UVR / MDX-Net
- Wav2Lip

This makes it possible to perform the main workflow without sending source
videos to a third-party AI API.

> Optional fallback services such as Edge-TTS may require network access.

---

# 📜 Credits

### Original ViDubb Project

[ViDubb](https://github.com/medahmedkrichen/ViDubb)

Created by:

[medahmedkrichen](https://github.com/medahmedkrichen)

---

### Wav2Lip

https://github.com/Rudrabha/Wav2Lip

---

### Coqui TTS / XTTS

https://github.com/coqui-ai/TTS

---

### Faster-Whisper

https://github.com/SYSTRAN/faster-whisper

---

### SpeechBrain

https://github.com/speechbrain/speechbrain

---

### Ollama

https://ollama.com/

---

# ☕ Support

If viDubb Pro is useful to you and you want to support development:

[☕ **Buy me a coffee**](https://buycoffee.to/maslo_github)

---

# 👤 Author / Contact

**Maslo95**

GitHub:

[@MASELKO-95](https://github.com/MASELKO-95)

Discord:

**Lone wolf® (formerly MASELKO-95®)**

Username:

```text
maselko95
```

---

# 📄 License

This project is distributed under the **MIT License**.

See:

[LICENSE](LICENSE)

---

<p align="center">
  <b>viDubb Pro — Maslo95 Edition</b><br>
  Local AI translation • Voice cloning • Dubbing • Speaker diarization
</p>
