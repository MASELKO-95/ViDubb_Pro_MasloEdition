# 🎬 viDubb Pro — Maslo95 Edition

**Version 1.0.1a** | AI-Powered Video Dubbing & Translation Pipeline

[![Tests](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/actions/workflows/tests.yml/badge.svg)](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/actions/workflows/tests.yml)

> [!WARNING]
> This is a test release under active development. Voice recognition,
> voice cloning, Timeline Review and Wav2Lip may still require manual review.

Fork of [ViDubb](https://github.com/medahmedkrichen/ViDubb) by medahmedkrichen, enhanced and customized by [Maslo95](https://github.com/MASELKO-95).

> This project processes media locally. Uploaded videos, generated audio,
> projects and voice profiles are runtime data and are intentionally excluded
> from the repository.

## ✨ Features

- 🎤 **Voice Cloning** — XTTS-v2 zero-shot voice cloning for 17+ languages
- 🗣️ **Speaker Diarization** — Automatic multi-speaker detection (SpeechBrain ECAPA-TDNN, local, no tokens)
- 🌐 **AI Translation** — Local LLM translation via Ollama (no cloud APIs required)
- 🔊 **Edge-TTS Fallback** — Microsoft Neural TTS as backup engine
- 🎵 **Background Preservation** — Vocal/instrumental separation (UVR MDX-Net)
- 📝 **Subtitle Editor** — Inline editing, approval workflow, ignore markers for intros/music
- 🎬 **Lip Sync** — Wav2Lip integration for mouth movement synchronization
- 🔇 **Smart Timing** — Timeline-based audio overlay (no more out-of-sync dubbing)
- 🎛️ **Audio Enhancement** — DSP denoising, EQ, broadcast voice processing
- 💻 **100% Local** — Everything runs on your machine, no API keys required

## 🚀 Quick Start

### Requirements

- Python 3.10 (recommended; some AI dependencies may not support newer versions)
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- FFmpeg installed and on PATH
- [Ollama](https://ollama.com/) with a translation model (for local AI translation)

### Installation

```bash
git clone https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition.git
cd ViDubb_Pro_MasloEdition
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running

**Option 1: GUI Launcher (recommended)**
```bash
python run_launcher.py
```

**Option 2: Direct Flask server**
```bash
python app_new.py
```

Open `http://127.0.0.1:7860` in your browser.

The first run may download several AI models. Model weights, virtual
environments and generated media are not included in Git.

### Czyszczenie cache

Skrypt usuwa `__pycache__`, pliki `.pyc/.pyo` oraz cache narzędzi. Nie usuwa
`.venv`, projektów, filmów ani bazy głosów:

```bash
python scripts/clean_cache.py
```

### Publikacja zmian na GitHubie

Przed wysłaniem sprawdź listę zmian, a następnie wykonaj:

```bash
python scripts/clean_cache.py
git status --short
git add -A
git commit -m "Prepare project for public release"
git push origin main
```

### Kreator bazy głosów

Osobny program tnie długie audio/wideo na dialogi, transkrybuje je, porównuje
głos z pewną próbką ECAPA i kataloguje wyniki jako `accepted`, `review` oraz
`rejected`:

```bash
python tools/voice_dataset_builder.py --gui
```

Wersja terminalowa:

```bash
python tools/voice_dataset_builder.py nagranie1.mp4 nagranie2.wav \
  --name Yuuki_Takada --reference pewna_probka.wav --language ja --register
```

Dataset trafia do `voice_datasets/<nazwa>/`. Kolejne uruchomienia dopisują
klipy do istniejącego manifestu. `--register` tworzy także kompatybilny profil
w obecnej Voice DB. Bez pliku referencyjnego lub `--voice-id` program celowo
kieruje klipy do `review`, ponieważ nie ma podstaw do potwierdzenia osoby.

`voice_db/` and `speakers_audio/` start empty after cloning. The application
creates the voice database index automatically when it is first needed.

### Wav2Lip Setup (optional, for lip sync)

Download `wav2lip_gan.pth` and place it in `Wav2Lip/checkpoints/`:
- [Wav2Lip Weights](https://github.com/Rudrabha/Wav2Lip#getting-the-weights)

## 🏗️ Architecture

```
viDubb Pro/
├── app_new.py                    # Flask server entry point
├── run_launcher.py               # Tkinter GUI launcher
├── modules/
│   ├── app.py                    # Flask app factory
│   ├── config.py                 # Configuration constants
│   ├── state.py                  # Global state manager
│   ├── routes/
│   │   ├── dubbing.py            # Dubbing generation API
│   │   ├── projects.py           # Project CRUD
│   │   ├── translate.py          # AI translation (Ollama/OpenAI)
│   │   └── video.py              # Video upload, transcription, diarization
│   ├── services/
│   │   ├── audio_enhancer.py     # Audio post-processing
│   │   ├── diarization_service.py# Speaker identification
│   │   ├── tts_service.py        # XTTS-v2 & Edge-TTS synthesis
│   │   ├── video_service.py      # Background separation & video muxing
│   │   └── whisper_service.py    # Faster-Whisper transcription
│   └── utils/                    # Time formatting, cleanup utilities
├── Wav2Lip/                      # Lip sync engine
├── templates/index.html          # Web UI (single-page app)
└── static/                       # CSS, fonts, assets
```

## 🌍 Supported Languages

English, Polish, Spanish, French, German, Italian, Turkish, Russian, Dutch, Czech, Arabic, Chinese (Simplified), Japanese, Korean, Hindi

## 🔧 Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| TTS Engine | Edge-TTS | `edge` or `xtts` for voice cloning |
| Translation | Ollama | Local LLM, configurable endpoint |
| Whisper Model | `turbo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` |
| Audio Enhancement | Off | DSP denoising, broadcast voice |

## 📜 Credits

- **Original Project**: [ViDubb](https://github.com/medahmedkrichen/ViDubb) by [medahmedkrichen](https://github.com/medahmedkrichen)
- **Wav2Lip**: [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- **Coqui TTS**: [coqui-ai/TTS](https://github.com/coqui-ai/TTS)
- **Faster-Whisper**: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **SpeechBrain**: [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)

## ☕ Support

If viDubb Pro is useful to you, you can support its development by
[buying me a coffee](https://buycoffee.to/maslo_github). Thank you!

## 🐛 Testowanie i zgłaszanie błędów

Lekkie testy, niewymagające pobierania modeli AI, można uruchomić lokalnie:

```bash
pip install -r requirements-test.txt
pytest -q
```

GitHub Actions wykonuje te testy oraz kontrolę składni automatycznie po
każdym pushu i dla każdego pull requestu. Workflow można też uruchomić
ręcznie w zakładce **Actions → Tests → Run workflow**.

- [GitHub Issues](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/issues)
- [GitHub Discussions](https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition/discussions)
- Discord: **Lone wolf® (dawniej MASELKO-95®)**, nazwa użytkownika `maselko95`

Przycisk `🧪 TEST` w aplikacji pozwala pobrać raport diagnostyczny z ostatnimi
logami. Przed zapisaniem raport maskuje tokeny i lokalną ścieżkę katalogu
domowego. Dołącz raport do zgłoszenia, opisując wykonane kroki i oczekiwany
rezultat.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
