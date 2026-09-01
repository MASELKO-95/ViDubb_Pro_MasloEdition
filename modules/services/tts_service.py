import os
import gc
import re
import difflib
import subprocess

import torch
from pydub import AudioSegment
from faster_whisper import WhisperModel

from modules.state import state


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_DIR = "audio"
CHUNKS_DIR = "audio_chunks"
SPEAKERS_DIR = "speakers_audio"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(SPEAKERS_DIR, exist_ok=True)


# ============================================================
# EDGE-TTS VOICE MAP
# ============================================================

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


# ============================================================
# XTTS SUPPORTED LANGUAGES
# ============================================================

XTTS_SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu", "hi",
}


# ============================================================
# UTILITY
# ============================================================

def _cleanup_torch():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _safe_filename(value: str) -> str:
    if not value:
        return "Unknown"
    value = str(value)
    value = re.sub(r"[^a-zA-Z0-9_\-]", "_", value)
    return value


# ============================================================
# SCRIPT / LANGUAGE DETECTION
# ============================================================

def _detect_script(text: str) -> str | None:
    if not text:
        return None
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
    detected = _detect_script(text)
    if detected and detected != target_lang:
        voice = EDGE_VOICE_MAP.get(
            detected,
            EDGE_VOICE_MAP.get(target_lang, "en-US-ChristopherNeural")
        )
        state.add_log(f"    ⚠️ Wykryto pismo '{detected}' — używam głosu {voice}")
        return voice
    return EDGE_VOICE_MAP.get(target_lang, "en-US-ChristopherNeural")


# ============================================================
# BUILD SPEAKER REFERENCES (AGGREGATION MODE)
# ============================================================

def _build_speaker_references(
    video_path: str,
    timestamps: list,
    speakers: list
) -> dict:

    speaker_refs = {}
    if not speakers or not timestamps:
        return speaker_refs

    try:
        base_audio = AudioSegment.from_file(video_path)
    except Exception as e:
        state.add_log(f"  ❌ Nie udało się załadować audio z wideo do referencji: {e}")
        return speaker_refs

    unique_speakers = list(set(s for s in speakers if s and s != "Unknown"))
    state.add_log(f"  👥 Znaleziono {len(unique_speakers)} unikalnych speakerów. Agregacja referencji...")

    for speaker in unique_speakers:
        chunks_list = []
        combined_audio = AudioSegment.empty()

        for i, spk_label in enumerate(speakers):
            if spk_label == speaker:
                start_ms, end_ms = timestamps[i]
                duration = end_ms - start_ms

                if duration > 150:
                    ctx = 250
                    safe_start = max(0, start_ms - ctx)
                    safe_end = min(len(base_audio), end_ms + ctx)
                    chunk = base_audio[safe_start:safe_end]
                    chunks_list.append(chunk)
                    combined_audio += chunk
                    combined_audio += AudioSegment.silent(duration=50)

        if len(combined_audio) >= 2000:
            safe_name = _safe_filename(speaker)
            # Update/learn voice in persistent Voice DB
            from modules.services.voice_db_service import update_or_create_voice_profile
            resolved_profile_name = update_or_create_voice_profile(safe_name, chunks_list)

            ref_path = os.path.join(SPEAKERS_DIR, f"{_safe_filename(resolved_profile_name)}.wav")
            if not os.path.exists(ref_path):
                ref_path = os.path.join(SPEAKERS_DIR, f"{safe_name}.wav")
                optimized_audio = (
                    combined_audio
                    .set_frame_rate(22050)
                    .set_channels(1)
                    .set_sample_width(2)
                )
                optimized_audio.export(ref_path, format="wav")

            speaker_refs[speaker] = ref_path
            state.add_log(f"    ✅ {speaker}: przypisano/zaktualizowano referencję głosową '{resolved_profile_name}' ({len(combined_audio)/1000:.1f}s)")
        else:
            state.add_log(f"    ⚠️ {speaker}: za mało materiału ({len(combined_audio)/1000:.1f}s). Pomijam.")

    state.add_log(f"  ✅ Gotowe referencje: {len(speaker_refs)}/{len(unique_speakers)} speakerów.")
    return speaker_refs


# ============================================================
# EDGE TTS
# ============================================================

def _edge_tts_chunk(text: str, voice: str, out_path: str) -> bool:
    try:
        result = subprocess.run(
            ["edge-tts", "--voice", voice, "--text", text, "--write-media", out_path],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return True
        stderr = result.stderr.decode(errors="replace")
        state.add_log(f"    ⚠️ edge-tts błąd [{voice}]: {stderr[:160]}")
        return False
    except Exception as e:
        state.add_log(f"    ⚠️ edge-tts wyjątek: {e}")
        return False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _clean_text_for_compare(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


# ============================================================
# WHISPER AUDIO VALIDATION
# ============================================================

def _verify_audio_chunk(
    whisper_model: WhisperModel,
    audio_path: str,
    expected_text: str,
    lang_code: str
) -> tuple[bool, float, str]:
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        return (False, 0.0, "[Brak pliku lub pusty plik]")

    try:
        audio = AudioSegment.from_file(audio_path)
        if len(audio) < 200:
            return (False, 0.0, "[Zbyt krótki fragment]")
        if audio.dBFS < -48.0 or audio.rms == 0:
            return (False, 0.0, "[Cisza / brak głosu]")

        allowed_languages = {"pl", "en", "de", "fr", "es", "ja", "ru", "it", "zh-cn"}
        lang_arg = lang_code if lang_code in allowed_languages else None

        segments, _ = whisper_model.transcribe(
            audio_path, language=lang_arg, word_timestamps=False
        )
        recognized = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()

        if not recognized:
            return (False, 0.0, "[Whisper nic nie usłyszał]")

        expected_clean = _clean_text_for_compare(expected_text)
        recognized_clean = _clean_text_for_compare(recognized)

        if not expected_clean:
            return (True, 1.0, recognized)

        ratio = difflib.SequenceMatcher(None, expected_clean, recognized_clean).ratio()
        expected_words = set(expected_clean.split())
        recognized_words = set(recognized_clean.split())
        word_overlap = len(expected_words.intersection(recognized_words)) / max(1, len(expected_words))
        score = max(ratio, word_overlap)
        is_ok = score >= 0.45 or (len(expected_words) <= 2 and word_overlap >= 0.5)

        return (is_ok, score, recognized)
    except Exception as e:
        return (False, 0.0, f"[Błąd weryfikacji: {e}]")


# ============================================================
# XTTS ERROR CLASSIFICATION
# ============================================================

def _is_fatal_xtts_configuration_error(error_text: str) -> bool:
    error_lower = error_text.lower()
    fatal_patterns = (
        "multi-speaker model", "speaker_wav", "define either",
        "speaker is not defined", "language", "not supported",
        "unsupported language", "speaker must",
    )
    return any(pattern in error_lower for pattern in fatal_patterns)


# ============================================================
# XTTS SYNTHESIS
# ============================================================

def _xtts_generate_chunk(
    tts_model, text: str, ref_path: str, language: str, out_path: str
) -> tuple[bool, bool, str]:
    try:
        if not ref_path:
            return (False, True, "Brak referencji speaker_wav.")
        if not os.path.exists(ref_path):
            return (False, True, f"Referencja nie istnieje: {ref_path}")
        if os.path.getsize(ref_path) < 1000:
            return (False, True, f"Referencja jest pusta: {ref_path}")

        tts_model.tts_to_file(
            text=text,
            speaker_wav=[ref_path],
            language=language,
            file_path=out_path,
            split_sentences=True
        )

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return (True, False, "")
        return (False, False, "XTTS nie utworzył poprawnego pliku.")
    except Exception as e:
        error_text = str(e)
        fatal = _is_fatal_xtts_configuration_error(error_text)
        return (False, fatal, error_text)


# ============================================================
# MAIN DUBBING
# ============================================================

def build_dubbing_timeline(
    timestamps: list,
    total: int | None = None,
    active_lines: list[bool] | None = None,
) -> str:
    """Build the dub from prepared chunks using the reviewed timeline."""
    state.add_log("  🎬 Sklejam zaakceptowane fragmenty audio na osi czasu…")
    total = len(timestamps) if total is None else min(total, len(timestamps))
    active_lines = active_lines or [True] * total

    total_dur_ms = max((ts[1] for ts in timestamps), default=57_000) + 3000
    timeline = AudioSegment.silent(duration=total_dur_ms)

    for i in range(total):
        if i >= len(active_lines) or not active_lines[i]:
            continue

        start_ms, end_ms = timestamps[i]
        chunk_path = os.path.join(CHUNKS_DIR, f"{i}.wav")
        if not os.path.exists(chunk_path):
            continue

        try:
            with open(chunk_path, "rb") as chunk_file:
                chunk = AudioSegment.from_file(chunk_file)
            # Respect the reviewed End value. Also protect against an overlap
            # when the next clip starts earlier than the chosen end.
            slot_end_ms = max(start_ms + 100, end_ms)
            if i + 1 < len(timestamps):
                slot_end_ms = min(slot_end_ms, timestamps[i + 1][0])
            max_allowed = max(100, slot_end_ms - start_ms)

            if len(chunk) > max_allowed:
                state.add_log(
                    f"    ⚠️ Linia {i + 1}: audio {len(chunk)} ms, slot "
                    f"{max_allowed} ms — końcówka zostanie przycięta."
                )
                chunk = chunk[:max_allowed]

            timeline = timeline.overlay(chunk, position=int(start_ms))
        except Exception as e:
            state.add_log(f"    ⚠️ Nie mogę wczytać fragmentu {i}: {e}")

    dub_path = os.path.join(AUDIO_DIR, "dub_raw.wav")
    export_handle = timeline.export(dub_path, format="wav")
    export_handle.close()
    state.add_log(
        f"  ✅ Dubbing sklejony (reviewed timeline) → {dub_path} "
        f"({len(timeline) / 1000:.1f}s)"
    )
    return dub_path


def prepared_chunk_metadata(timestamps: list) -> list[dict]:
    """Return actual TTS durations for the timeline review UI."""
    result = []
    for index, (start_ms, end_ms) in enumerate(timestamps):
        chunk_path = os.path.join(CHUNKS_DIR, f"{index}.wav")
        actual_ms = 0
        if os.path.isfile(chunk_path):
            try:
                with open(chunk_path, "rb") as chunk_file:
                    actual_ms = len(AudioSegment.from_file(chunk_file))
            except Exception:
                actual_ms = 0
        slot_ms = max(0, int(end_ms) - int(start_ms))
        result.append({
            "index": index,
            "actual_ms": actual_ms,
            "slot_ms": slot_ms,
            "overflow_ms": max(0, actual_ms - slot_ms),
            "ready": actual_ms > 0,
        })
    return result

def generate_dubbed_audio(
    video_path: str,
    translated_texts: list,
    timestamps: list,
    speakers: list,
    target_lang_code: str,
    voice_name: str = "Default",
    dialogue_voices: list | None = None,
    tts_engine: str = "edge",
    validation_model_size: str = "None",
    auto_retry_count: int = 10
) -> str:
    state.add_log(
        f"🗣️ Generowanie dubbingu (engine={tts_engine}, "
        f"lang={target_lang_code}, voice={voice_name}, "
        f"validation={validation_model_size})…"
    )

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    total = len(translated_texts)
    dialogue_voices = dialogue_voices or []
    max_retries = max(1, min(10, int(auto_retry_count or 10)))

    # ========================================================
    # VALIDATION WHISPER
    # ========================================================

    validator_model = None
    if validation_model_size and validation_model_size != "None":
        state.add_log(f"🔍 Inicjalizacja modelu Whisper ({validation_model_size}) do walidacji TTS…")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            validator_model = WhisperModel(validation_model_size, device=device, compute_type=compute_type)
            state.add_log("  ✅ Model weryfikacji Whisper gotowy.")
        except Exception as e:
            state.add_log(f"  ⚠️ Nie udało się załadować Whisper do walidacji: {e}")
            validator_model = None

    # ========================================================
    # ENGINE: EDGE-TTS
    # ========================================================

    if tts_engine == "edge":
        state.add_log("  🌐 Używam Edge-TTS Microsoftu.")

        for i, text in enumerate(translated_texts):
            if state.cancel_flags["dubbing"]:
                break

            chunk_path = os.path.join(CHUNKS_DIR, f"{i}.wav")

            if not text or not text.strip():
                duration = max(500, timestamps[i][1] - timestamps[i][0]) if i < len(timestamps) else 1000
                AudioSegment.silent(duration=duration).export(chunk_path, format="wav")
                continue

            voice = _pick_edge_voice(text, target_lang_code)
            success = False

            for attempt in range(max_retries):
                if state.cancel_flags["dubbing"]:
                    break

                ok = _edge_tts_chunk(text, voice, chunk_path)
                if not ok and voice != "en-US-ChristopherNeural":
                    ok = _edge_tts_chunk(text, "en-US-ChristopherNeural", chunk_path)

                if ok and validator_model:
                    is_valid, score, heard = _verify_audio_chunk(validator_model, chunk_path, text, target_lang_code)
                    if is_valid:
                        success = True
                        break
                    state.add_log(f"    🔁 Linia {i + 1} [próba {attempt + 1}/{max_retries}] Whisper: {score * 100:.0f}% zgodności.")
                elif ok:
                    success = True
                    break

            if not success:
                state.add_log(f"    ⚠️ Linia {i + 1}: fallback do ciszy.")
                duration = max(500, timestamps[i][1] - timestamps[i][0]) if i < len(timestamps) else 1000
                AudioSegment.silent(duration=duration).export(chunk_path, format="wav")

            if (i + 1) % 20 == 0 or i + 1 == total:
                state.add_log(f"  Edge-TTS: {i + 1}/{total} linii…")

    # ========================================================
    # ENGINE: XTTS-v2
    # ========================================================

    elif tts_engine in ("xtts", "xtts2"):
        state.add_log("  🎙️ Używam XTTS-v2 z klonowaniem głosów speakerów…")

        global_ref = None
        custom_voice = (
            voice_name != "Default"
            and os.path.exists(os.path.join(SPEAKERS_DIR, f"{voice_name}.wav"))
        )

        if custom_voice:
            global_ref = os.path.join(SPEAKERS_DIR, f"{voice_name}.wav")
            state.add_log(f"  🎙️ Używam ręcznie wybranego głosu dla wszystkich linii: {global_ref}")
        else:
            speaker_refs = _build_speaker_references(video_path, timestamps, speakers)

            if speaker_refs:
                state.add_log(f"  👥 XTTS otrzymał {len(speaker_refs)} referencji speakerów.")
            else:
                state.add_log("  ⚠️ Nie znaleziono referencji speakerów — buduję globalną referencję ze wszystkich fragmentów.")
                try:
                    base_audio = AudioSegment.from_file(video_path)
                    combined = AudioSegment.empty()
                    chunks_count = 0
                    for start_ms, end_ms in timestamps:
                        duration = end_ms - start_ms
                        if duration > 150:
                            ctx = 250
                            safe_start = max(0, start_ms - ctx)
                            safe_end = min(len(base_audio), end_ms + ctx)
                            combined += base_audio[safe_start:safe_end]
                            combined += AudioSegment.silent(duration=50)
                            chunks_count += 1
                    if len(combined) >= 2500:
                        global_ref = os.path.join(AUDIO_DIR, "short_ref_global.wav")
                        optimized = combined.set_frame_rate(22050).set_channels(1).set_sample_width(2)
                        optimized.export(global_ref, format="wav")
                        state.add_log(f"  ✅ Globalna referencja: {len(optimized)/1000:.1f}s z {chunks_count} fragmentów")
                    else:
                        state.add_log("  ⚠️ Nie udało się zbudować globalnej referencji.")
                except Exception as e:
                    state.add_log(f"  ❌ Błąd budowania globalnej referencji: {e}")

        # Load XTTS
        import torch as _torch
        original_torch_load = _torch.load

        def _safe_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        _torch.load = _safe_load

        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from torch.serialization import add_safe_globals
            add_safe_globals([XttsConfig])
        except Exception:
            pass

        tts_model = None
        try:
            from TTS.api import TTS as CoquiTTS
            state.add_log("  🧠 Ładowanie XTTS-v2...")
            tts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            state.add_log("  ✅ Model XTTS-v2 załadowany.")
        except Exception as e:
            state.add_log(f"  ❌ Nie udało się załadować XTTS-v2: {e}")
            _torch.load = original_torch_load
            if validator_model:
                del validator_model
            _cleanup_torch()
            raise

        xtts_lang = target_lang_code if target_lang_code in XTTS_SUPPORTED_LANGS else "en"
        if xtts_lang != target_lang_code:
            state.add_log(f"  ⚠️ XTTS nie obsługuje '{target_lang_code}' — używam '{xtts_lang}'.")

        for i, text in enumerate(translated_texts):
            if state.cancel_flags["dubbing"]:
                break

            chunk_path = os.path.join(CHUNKS_DIR, f"{i}.wav")

            if not text or not text.strip():
                duration = max(500, timestamps[i][1] - timestamps[i][0]) if i < len(timestamps) else 1000
                AudioSegment.silent(duration=duration).export(chunk_path, format="wav")
                continue

            speaker = speakers[i] if i < len(speakers) else "Unknown"

            selected_voice = str(
                dialogue_voices[i] if i < len(dialogue_voices) else ""
            ).strip()
            selected_ref = (
                os.path.join(SPEAKERS_DIR, f"{selected_voice}.wav")
                if selected_voice else ""
            )

            if selected_ref and os.path.isfile(selected_ref):
                ref_path = selected_ref
            elif global_ref:
                ref_path = global_ref
            else:
                ref_path = speaker_refs.get(speaker) if speaker_refs else None
                if not ref_path and speaker_refs:
                    ref_path = next(iter(speaker_refs.values()))
                    state.add_log(f"    ℹ️ Linia {i + 1}: speaker '{speaker}' nie ma referencji — używam zapasowej.")

            if not ref_path:
                state.add_log(f"    ⚠️ Linia {i + 1}: brak referencji XTTS — fallback Edge-TTS.")
                fallback_voice = _pick_edge_voice(text, target_lang_code)
                if not _edge_tts_chunk(text, fallback_voice, chunk_path):
                    AudioSegment.silent(duration=1000).export(chunk_path, format="wav")
                continue

            success = False
            for attempt in range(max_retries):
                if state.cancel_flags["dubbing"]:
                    break

                gen_ok, fatal_error, error_text = _xtts_generate_chunk(tts_model, text, ref_path, xtts_lang, chunk_path)

                if fatal_error:
                    state.add_log(f"    ❌ XTTS konfiguracja: {error_text[:220]}")
                    break

                if gen_ok:
                    if validator_model:
                        is_valid, score, heard = _verify_audio_chunk(validator_model, chunk_path, text, target_lang_code)
                        if is_valid:
                            success = True
                            break
                        state.add_log(f"    🔁 Linia {i + 1} [próba {attempt + 1}/{max_retries}] Whisper: {score * 100:.0f}% zgodności; słyszano: '{heard[:60]}…'")
                    else:
                        success = True
                        break
                else:
                    state.add_log(f"    ⚠️ XTTS błąd linia {i + 1} (próba {attempt + 1}/{max_retries}): {error_text[:180]}")

            if not success:
                state.add_log(f"    ⚠️ Linia {i + 1}: fallback do Edge-TTS.")
                fallback_voice = _pick_edge_voice(text, target_lang_code)
                fallback_ok = _edge_tts_chunk(text, fallback_voice, chunk_path)
                if not fallback_ok:
                    AudioSegment.silent(duration=1000).export(chunk_path, format="wav")

            if (i + 1) % 10 == 0 or i + 1 == total:
                state.add_log(f"  XTTS: {i + 1}/{total} linii…")

        if tts_model is not None:
            del tts_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _torch.load = original_torch_load

    else:
        state.add_log(f"  ❌ Nieznany silnik TTS: {tts_engine}")
        if validator_model:
            del validator_model
        _cleanup_torch()
        raise ValueError(f"Unsupported TTS engine: {tts_engine}")

    if validator_model:
        del validator_model
        _cleanup_torch()

    if state.cancel_flags["dubbing"]:
        return ""

    return build_dubbing_timeline(timestamps, total=total)
