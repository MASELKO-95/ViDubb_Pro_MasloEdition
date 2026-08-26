# -*- coding: utf-8 -*-
"""
Text-to-Speech synthesis service (Edge-TTS, XTTS-v2)
===================================================
Features:
- Smart script detection and dynamic voice mapping.
- XTTS-v2 with voice cloning, Japanese fugashi/unidic support, and automatic fallbacks.
- Whisper audio verification with auto-retry loop (up to 10 attempts per chunk).
- Silence and noise detection with graceful fallbacks (Edge-TTS / lector / silence).
"""
import os
import gc
import re
import difflib
import subprocess
import torch
from pydub import AudioSegment
from faster_whisper import WhisperModel
from modules.state import state

# ── Edge-TTS: mapowanie język → głos ────────────────────────────────────────
EDGE_VOICE_MAP = {
    "pl": "pl-PL-MarekNeural",
    "en": "en-US-ChristopherNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-KillianNeural",
    "it": "it-IT-DiegoNeural",
    "ru": "ru-RU-DmitryNeural",
    "tr": "tr-TR-AhmetNeural",
    "nl": "nl-NL-MaartenNeural",
    "cs": "cs-CZ-AntoninNeural",
    "ar": "ar-SA-HamedNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "zh-cn": "zh-CN-YunxiNeural",
    "hi": "hi-IN-MadhurNeural",
}

# Obsługiwane języki XTTS-v2
XTTS_SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu", "hi"
}


def _detect_script(text: str) -> str | None:
    """Heurystyczne wykrycie języka na podstawie znaków unicode."""
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF or 0x4E00 <= cp <= 0x9FFF:
            return "ja"
        if 0xAC00 <= cp <= 0xD7A3:
            return "ko"
        if 0x0400 <= cp <= 0x04FF:
            return "ru"
        if 0x0600 <= cp <= 0x06FF:
            return "ar"
    return None


def _pick_edge_voice(text: str, target_lang: str) -> str:
    """Dobierz głos Edge-TTS bezpiecznie na podstawie znaków."""
    detected = _detect_script(text)
    if detected and detected != target_lang:
        voice = EDGE_VOICE_MAP.get(detected, EDGE_VOICE_MAP.get(target_lang, "en-US-ChristopherNeural"))
        state.add_log(f"    ⚠️ Wykryto pismo '{detected}' w linii '{text[:30]}…' — używam głosu {voice}")
        return voice
    return EDGE_VOICE_MAP.get(target_lang, "en-US-ChristopherNeural")


def _get_short_ref(video_path: str, timestamps: list, speakers: list, target_speaker: str = None) -> str:
    """Wyciągnij krótki klip referencyjny dla klonowania głosu (XTTS)."""
    try:
        base = AudioSegment.from_file(video_path)
        if timestamps:
            best_start, best_end = 0, 0
            max_dur = 0
            for i, (s, e) in enumerate(timestamps):
                if target_speaker and target_speaker != "Unknown":
                    if speakers and i < len(speakers) and speakers[i] != target_speaker:
                        continue
                dur = e - s
                if 3000 <= dur <= 10000 and dur > max_dur:
                    max_dur = dur
                    best_start, best_end = s, e
            if max_dur == 0:
                best_start, best_end = timestamps[0]
            chunk = base[best_start:best_end]
        else:
            start = min(1000, max(0, len(base) - 8000))
            chunk = base[start:start + 8000]
        chunk = chunk.set_frame_rate(22050).set_channels(1)
        safe = (target_speaker or "ref").replace(" ", "_")
        short_path = f"audio/short_ref_{safe}.wav"
        os.makedirs("audio", exist_ok=True)
        chunk.export(short_path, format="wav")
        return short_path
    except Exception as e:
        state.add_log(f"  ⚠️ Nie udało się wyciągnąć referencji audio: {e}")
        return video_path


def _edge_tts_chunk(text: str, voice: str, out_path: str) -> bool:
    """Uruchom edge-tts synchronicznie. Zwraca True jeśli sukces."""
    try:
        res = subprocess.run(
            ["edge-tts", "--voice", voice, "--text", text, "--write-media", out_path],
            capture_output=True, timeout=60
        )
        if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return True
        err = res.stderr.decode(errors="replace")
        state.add_log(f"    ⚠️ edge-tts błąd [{voice}]: {err[:120]}")
        return False
    except Exception as e:
        state.add_log(f"    ⚠️ edge-tts wyjątek: {e}")
        return False


def _clean_text_for_compare(t: str) -> str:
    """Normalizacja tekstu do porównania weryfikacyjnego."""
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r'[^\w\s]', '', t)
    return " ".join(t.split())


def _verify_audio_chunk(
    whisper_model: WhisperModel,
    audio_path: str,
    expected_text: str,
    lang_code: str
) -> tuple[bool, float, str]:
    """
    Weryfikacja jakości wygenerowanego audio przez Whisper.
    Zwraca: (czy_poprawne: bool, stopień_zgodności: float, tekst_rozpoznany: str)
    """
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        return False, 0.0, "[Brak pliku lub pusty plik]"

    try:
        # 1. Sprawdzenie poziomu głośności / ciszy
        seg = AudioSegment.from_file(audio_path)
        if len(seg) < 200:
            return False, 0.0, "[Zbyt krótki fragment]"
        if seg.dBFS < -48.0 or seg.rms == 0:
            return False, 0.0, "[Cisza / brak głosu]"

        # 2. Transkrypcja Whisperem
        lang_arg = lang_code if lang_code in ("pl", "en", "de", "fr", "es", "ja", "ru", "it", "zh-cn") else None
        segments, _ = whisper_model.transcribe(audio_path, language=lang_arg, word_timestamps=False)
        transcribed_text = " ".join([s.text.strip() for s in segments if s.text.strip()]).strip()

        if not transcribed_text:
            return False, 0.0, "[Whisper nic nie usłyszał]"

        # 3. Porównanie tekstów
        c_exp = _clean_text_for_compare(expected_text)
        c_tra = _clean_text_for_compare(transcribed_text)

        if not c_exp:
            return True, 1.0, transcribed_text

        ratio = difflib.SequenceMatcher(None, c_exp, c_tra).ratio()

        # Sprawdzenie słów kluczowych jeśli stringi są krótkie
        exp_words = set(c_exp.split())
        tra_words = set(c_tra.split())
        word_overlap = len(exp_words.intersection(tra_words)) / max(1, len(exp_words))

        score = max(ratio, word_overlap)
        # Próg akceptacji: 45% zgodności (uwzględnia fonetyczne odmiany i interpunkcję)
        is_ok = score >= 0.45 or (len(exp_words) <= 2 and word_overlap >= 0.5)

        return is_ok, score, transcribed_text

    except Exception as e:
        return False, 0.0, f"[Błąd weryfikacji: {e}]"


def generate_dubbed_audio(
    video_path: str,
    translated_texts: list,
    timestamps: list,
    speakers: list,
    target_lang_code: str,
    voice_name: str = "Default",
    tts_engine: str = "edge",
    validation_model_size: str = "None",
    auto_retry_count: int = 10
) -> str:
    state.add_log(f"🗣️ Generowanie dubbingu (engine={tts_engine}, lang={target_lang_code}, voice={voice_name}, validation={validation_model_size})…")
    os.makedirs("audio_chunks", exist_ok=True)
    os.makedirs("audio", exist_ok=True)

    total = len(translated_texts)
    max_retries = max(1, min(10, int(auto_retry_count or 10)))

    # Inicjalizacja modelu Whisper do walidacji jeśli włączony
    validator_model = None
    if validation_model_size and validation_model_size != "None":
        state.add_log(f"🔍 Inicjalizacja modelu Whisper ({validation_model_size}) do walidacji i auto-korekty TTS…")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            validator_model = WhisperModel(validation_model_size, device=device)
            state.add_log("  ✅ Model weryfikacji Whisper gotowy.")
        except Exception as e:
            state.add_log(f"  ⚠️ Nie udało się załadować Whisper do walidacji: {e}. Kontynuuję bez walidacji.")
            validator_model = None

    # ── 1. EDGE-TTS ────────────────────────────────────────────────────────
    if tts_engine == "edge":
        state.add_log(f"  Używam Edge-TTS Microsoftu (domyślny głos: {EDGE_VOICE_MAP.get(target_lang_code, '?')})…")
        for i, text in enumerate(translated_texts):
            if state.cancel_flags["dubbing"]:
                break
            chunk_path = f"audio_chunks/{i}.wav"
            if not text or not text.strip():
                dur = max(500, timestamps[i][1] - timestamps[i][0]) if i < len(timestamps) else 1000
                AudioSegment.silent(duration=dur).export(chunk_path, format="wav")
                continue

            voice = _pick_edge_voice(text, target_lang_code)
            success = False

            for attempt in range(max_retries):
                if state.cancel_flags["dubbing"]:
                    break
                ok = _edge_tts_chunk(text, voice, chunk_path)
                if not ok:
                    ok = _edge_tts_chunk(text, "en-US-ChristopherNeural", chunk_path)

                if ok and validator_model:
                    is_valid, score, heard = _verify_audio_chunk(validator_model, chunk_path, text, target_lang_code)
                    if is_valid:
                        success = True
                        break
                    else:
                        state.add_log(f"    🔁 Linia {i+1} [próba {attempt+1}/{max_retries}] Whisper zgłasza błąd ({score*100:.0f}% zgodności, słyszano: '{heard[:35]}…'). Ponawiam…")
                elif ok:
                    success = True
                    break

            if not success:
                state.add_log(f"    ⚠️ Linia {i+1}: fallback do ciszy po {max_retries} próbach.")
                AudioSegment.silent(duration=1000).export(chunk_path, format="wav")

            if (i + 1) % 20 == 0 or i + 1 == total:
                state.add_log(f"  Edge-TTS: {i + 1}/{total} linii…")

    # ── 2. XTTS-v2 ─────────────────────────────────────────────────────────
    elif tts_engine in ("xtts", "xtts2"):
        state.add_log("  Używam XTTS-v2 z klonowaniem głosu…")

        if voice_name != "Default" and os.path.exists(f"speakers_audio/{voice_name}.wav"):
            ref_path = f"speakers_audio/{voice_name}.wav"
        else:
            ref_path = _get_short_ref(video_path, timestamps, speakers, None)

        # Patch torch.load
        import torch as _torch
        _orig = _torch.load
        def _safe_load(*a, **kw):
            kw["weights_only"] = False
            return _orig(*a, **kw)
        _torch.load = _safe_load

        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from torch.serialization import add_safe_globals
            add_safe_globals([XttsConfig])
        except Exception:
            pass

        try:
            from TTS.api import TTS as CoquiTTS
            tts_model = CoquiTTS(
                "tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=torch.cuda.is_available()
            )
            state.add_log("  ✅ Model XTTS-v2 załadowany.")
        except Exception as e:
            state.add_log(f"  ❌ Nie udało się załadować XTTS-v2: {e}")
            _torch.load = _orig
            raise

        xtts_lang = target_lang_code if target_lang_code in XTTS_SUPPORTED_LANGS else "en"
        if xtts_lang != target_lang_code:
            state.add_log(f"  ⚠️ XTTS nie obsługuje '{target_lang_code}' — używam '{xtts_lang}'")

        for i, text in enumerate(translated_texts):
            if state.cancel_flags["dubbing"]:
                break
            chunk_path = f"audio_chunks/{i}.wav"
            if not text or not text.strip():
                dur = max(500, timestamps[i][1] - timestamps[i][0]) if i < len(timestamps) else 1000
                AudioSegment.silent(duration=dur).export(chunk_path, format="wav")
                continue

            success = False
            for attempt in range(max_retries):
                if state.cancel_flags["dubbing"]:
                    break
                try:
                    tts_model.tts_to_file(
                        text=text,
                        speaker_wav=ref_path,
                        language=xtts_lang,
                        file_path=chunk_path
                    )
                    gen_ok = os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000
                except Exception as e:
                    err_str = str(e)
                    if xtts_lang == "ja" and ("MeCab" in err_str or "fugashi" in err_str or "mecab" in err_str.lower()):
                        state.add_log(f"    ⚠️ XTTS/MeCab błąd → fallback Edge-TTS")
                        gen_ok = _edge_tts_chunk(text, EDGE_VOICE_MAP.get("ja", "ja-JP-KeitaNeural"), chunk_path)
                    else:
                        state.add_log(f"    ⚠️ XTTS błąd linia {i + 1} (próba {attempt+1}): {err_str[:90]}")
                        gen_ok = False

                if gen_ok and validator_model:
                    is_valid, score, heard = _verify_audio_chunk(validator_model, chunk_path, text, target_lang_code)
                    if is_valid:
                        success = True
                        break
                    else:
                        state.add_log(f"    🔁 Linia {i+1} [próba {attempt+1}/{max_retries}] Whisper wykrył zniekształcenie ({score*100:.0f}% zgodności, słyszano: '{heard[:35]}…'). Ponawiam…")
                elif gen_ok:
                    success = True
                    break

            if not success:
                state.add_log(f"    ⚠️ Linia {i+1}: po {max_retries} nieudanych próbach XTTS — bezpieczny fallback do Edge-TTS...")
                fb_voice = _pick_edge_voice(text, target_lang_code)
                ok_fb = _edge_tts_chunk(text, fb_voice, chunk_path)
                if not ok_fb:
                    AudioSegment.silent(duration=1000).export(chunk_path, format="wav")

            if (i + 1) % 10 == 0 or i + 1 == total:
                state.add_log(f"  XTTS: {i + 1}/{total} linii…")

        del tts_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _torch.load = _orig

    else:
        state.add_log(f"  ❌ Nieznany silnik TTS: {tts_engine}")
        raise ValueError(f"Unsupported TTS engine: {tts_engine}")

    if validator_model:
        del validator_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if state.cancel_flags["dubbing"]:
        return ""

    # ── 3. Sklejenie fragmentów na osi czasu (timeline overlay) ───────────
    # Each chunk is placed at its exact subtitle start_ms position,
    # so dubbing stays in sync even if intro music has no TTS audio.
    state.add_log("  Sklejam fragmenty audio na osi czasu…")

    # Build a silent timeline long enough to hold all chunks
    if timestamps:
        total_dur_ms = max(ts[1] for ts in timestamps) + 3000
    else:
        total_dur_ms = 60000
    timeline = AudioSegment.silent(duration=total_dur_ms)

    for i in range(total):
        if i >= len(timestamps):
            break
        start_ms, end_ms = timestamps[i]
        chunk_path = f"audio_chunks/{i}.wav"
        if os.path.exists(chunk_path):
            try:
                chunk = AudioSegment.from_file(chunk_path)
                # Calculate how much room we have before the NEXT subtitle starts.
                # Trim only if the chunk would bleed into the next spoken line.
                if i + 1 < len(timestamps):
                    next_start_ms = timestamps[i + 1][0]
                    max_allowed = max(100, next_start_ms - start_ms)
                else:
                    # Last chunk — no next subtitle, let it play freely
                    max_allowed = len(chunk)
                if len(chunk) > max_allowed:
                    chunk = chunk[:max_allowed]
                timeline = timeline.overlay(chunk, position=int(start_ms))
            except Exception as e:
                state.add_log(f"    ⚠️ Nie mogę wczytać fragmentu {i}: {e}")
        # If no chunk file (Ignore/empty), the timeline stays silent at that position

    dub_path = "audio/dub_raw.wav"
    timeline.export(dub_path, format="wav")
    state.add_log(f"  ✅ Dubbing sklejony (timeline) → {dub_path} ({len(timeline) / 1000:.1f}s)")
    return dub_path
