# 🎬 viDubb Pro — Maslo95 Edition

**Version 1.0.1a** | Local AI-powered video dubbing, translation and voice-cloning pipeline

[![Tests](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/actions/workflows/tests.yml/badge.svg)](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/actions/workflows/tests.yml)

> [!WARNING]
> This is a test release under active development. Voice recognition, voice cloning, Timeline Review and Wav2Lip may still require manual review.

Fork of [ViDubb](https://github.com/medahmedkrichen/ViDubb) by medahmedkrichen, enhanced and customized by [Maslo95](https://github.com/MASELKO-95).

> viDubb Pro processes media locally. Uploaded videos, generated audio, projects and voice profiles are runtime data and are intentionally excluded from the repository.

## ✨ Features

- 🎤 **Voice Cloning** — XTTS-v2 zero-shot voice cloning for 17+ languages
- 🗣️ **Speaker Diarization** — automatic multi-speaker detection using SpeechBrain ECAPA-TDNN
- 🌐 **AI Translation** — local LLM translation through Ollama
- 🔊 **Edge-TTS Fallback** — Microsoft Neural TTS as a backup engine
- 🎵 **Background Preservation** — vocal/instrumental separation with UVR MDX-Net
- 📝 **Subtitle Editor** — inline editing, approval workflow and ignore markers
- 🎬 **Lip Sync** — optional Wav2Lip integration
- 🔇 **Smart Timing** — timeline-based audio overlay for better dubbing synchronization
- 🎛️ **Audio Enhancement** — denoising, EQ and voice post-processing
- 💻 **Local-first workflow** — no cloud API is required for the default pipeline

## 🎥 Demo

This section is intended to show the same clip before and after processing, making it easy to compare the result.

| Version | Description | Demo file |
|---|---|---|
| 🎞️ **Original** | Original source clip before viDubb Pro processing | `docs/demo/original.mp4` |
| 🇬🇧 **English dub** | Example generated English dubbing | `docs/demo/dub_english.mp4` |
| 🇯🇵 **Japanese dub** | Example generated Japanese dubbing | `docs/demo/dub_japanese.mp4` |

> The README structure is ready for the three demo videos. The MP4 files themselves must be added separately because screenshots of the desktop do not contain the actual video data.

Recommended repository layout:

```text
docs/
├── demo/
│   ├── original.mp4
│   ├── dub_english.mp4
│   └── dub_japanese.mp4
└── images/
    └── launcher.png
```

For larger demo videos, GitHub Releases or an external video host is recommended instead of committing very large MP4 files directly to the repository.

## 🖥️ GUI Preview

### Launcher

![viDubb Pro Launcher](docs/images/launcher.png)

The launcher verifies dependencies, checks for updates and starts the local viDubb Pro server.

### Main application

The main web interface includes:

- video and subtitle loading,
- subtitle timeline editing,
- speaker assignment,
- source and target language selection,
- Whisper model selection,
- translation and approval workflow,
- project management,
- voice database access,
- dubbing generation controls.

### Dubbing panel

The dubbing panel provides controls for:

- target dubbing language,
- dubbing or voice-over mode,
- XTTS-v2 or fallback TTS engines,
- reference voice selection,
- background-audio preservation,
- optional burned-in subtitles,
- optional Wav2Lip synchronization,
- audio enhancement,
- output path,
- Timeline Review before rendering.

## 🚀 Quick Start

### Requirements

- Python 3.10 recommended
- CUDA-compatible GPU recommended, ideally 8 GB+ VRAM
- FFmpeg installed and available on `PATH`
- [Ollama](https://ollama.com/) with a translation model for local AI translation

### Installation

```bash
git clone https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition.git
cd ViDubb_Pro_MasloEdition
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running

**Option 1: GUI Launcher — recommended**

```bash
python run_launcher.py
```

**Option 2: Direct Flask server**

```bash
python app_new.py
```

Open `http://127.0.0.1:7860` in your browser.

The first run may download several AI models. Model weights, virtual environments and generated media are not included in Git.

## 🧹 Cache cleanup

The cleanup script removes `__pycache__`, `.pyc/.pyo` files and tool caches. It does not remove `.venv`, projects, videos or the voice database.

```bash
python scripts/clean_cache.py
```

## 📦 Publishing changes to GitHub

Before pushing changes:

```bash
python scripts/clean_cache.py
git status --short
git add -A
git commit -m "Prepare project for public release"
git push origin main
```

## 🎙️ Voice Dataset Builder

The separate dataset builder can split long audio/video recordings into dialogue clips, transcribe them, compare voices using ECAPA and classify clips into `accepted`, `review` and `rejected` groups.

GUI mode:

```bash
python tools/voice_dataset_builder.py --gui
```

Terminal mode:

```bash
python tools/voice_dataset_builder.py recording1.mp4 recording2.wav \
  --name Yuuki_Takada --reference reference_sample.wav --language ja --register
```

Datasets are stored in `voice_datasets/<name>/`. Re-running the tool appends clips to the existing manifest. `--register` also creates a compatible profile in the current Voice DB. Without a reference file or `--voice-id`, clips are intentionally routed to `review` because the speaker identity cannot be confirmed reliably.

`voice_db/` and `speakers_audio/` start empty after cloning. The application creates the voice database index automatically when first needed.

## 👄 Wav2Lip Setup — optional

Download `wav2lip_gan.pth` and place it in:

```text
Wav2Lip/checkpoints/
```

See the [Wav2Lip project](https://github.com/Rudrabha/Wav2Lip#getting-the-weights) for the model weights.

## 🏗️ Architecture

```text
viDubb Pro/
├── app_new.py                     # Flask server entry point
├── run_launcher.py                # Tkinter GUI launcher
├── modules/
│   ├── app.py                     # Flask app factory
│   ├── config.py                  # Configuration constants
│   ├── state.py                   # Global state manager
│   ├── routes/
│   │   ├── dubbing.py             # Dubbing generation API
│   │   ├── projects.py            # Project CRUD
│   │   ├── translate.py           # AI translation
│   │   └── video.py               # Video upload, transcription, diarization
│   ├── services/
│   │   ├── audio_enhancer.py      # Audio post-processing
│   │   ├── diarization_service.py # Speaker identification
│   │   ├── tts_service.py         # XTTS-v2 and Edge-TTS synthesis
│   │   ├── video_service.py       # Background separation and video muxing
│   │   └── whisper_service.py     # Faster-Whisper transcription
│   └── utils/                     # Time formatting and cleanup utilities
├── Wav2Lip/                       # Lip-sync engine
├── templates/index.html           # Web UI
└── static/                        # CSS, fonts and assets
```

## 🌍 Supported Languages

English, Polish, Spanish, French, German, Italian, Turkish, Russian, Dutch, Czech, Arabic, Chinese (Simplified), Japanese, Korean and Hindi.

## 🔧 Configuration

| Setting | Default | Description |
|---|---|---|
| TTS Engine | Edge-TTS | `edge` or `xtts` for voice cloning |
| Translation | Ollama | Local LLM with configurable endpoint |
| Whisper Model | `turbo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` |
| Audio Enhancement | Off | DSP denoising and voice processing |

## 🧪 Testing and bug reports

Lightweight tests that do not require downloading AI models can be run locally:

```bash
pip install -r requirements-test.txt
pytest -q
```

GitHub Actions runs the tests and syntax checks automatically on each push and pull request. The workflow can also be started manually from **Actions → Tests → Run workflow**.

- [GitHub Issues](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/issues)
- [GitHub Discussions](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/discussions)
- Discord: **Lone wolf® (formerly MASELKO-95®)**, username `maselko95`

The `🧪 TEST` button in the application can export a diagnostic report containing recent logs. Before saving, the report masks tokens and the local home-directory path. Attach the report to a bug report together with the steps performed and the expected result.

## 📜 Credits

- **Original Project**: [ViDubb](https://github.com/medahmedkrichen/ViDubb) by [medahmedkrichen](https://github.com/medahmedkrichen)
- **Wav2Lip**: [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- **Coqui TTS**: [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- **Faster-Whisper**: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **SpeechBrain**: [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)

## ☕ Support

If viDubb Pro is useful to you, you can support its development by [buying me a coffee](https://buycoffee.to/maslo_github).

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
