
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from modules.state import state


# ============================================================
# CONFIG
# ============================================================

VOICE_DB_DIR = Path("voice_db")
VOICE_DB_INDEX = VOICE_DB_DIR / "index.json"
SPEAKERS_AUDIO_DIR = Path("speakers_audio")

# TTS/reference audio
REFERENCE_SAMPLE_RATE = 24000
MAX_REF_DURATION_MS = 30000
MIN_REF_DURATION_MS = 2500

# ECAPA
ECAPA_SAMPLE_RATE = 16000
EMBEDDING_DIM = 192
MAX_EMBEDDINGS_PER_PROFILE = 32
MAX_NEW_EMBEDDINGS_PER_UPDATE = 8
MIN_EMBEDDING_SEGMENT_MS = 1500

# Matching.
# MATCH_THRESHOLD = kandydat podobny, ale jeszcze nie auto-merge.
# STRONG_MATCH_THRESHOLD = dopiero od tego poziomu automatycznie aktualizujemy
# existing profile. A duplicate is safer than mixing two actors.
MATCH_THRESHOLD = 0.78
STRONG_MATCH_THRESHOLD = 0.86

# Czyszczenie mowy
MIN_SPEECH_SEGMENT_MS = 700
MAX_SPEECH_SEGMENT_MS = 8000
SILENCE_MIN_MS = 350
SILENCE_PADDING_MS = 80
MIN_ACCEPTABLE_DBFS = -50.0

# Audio quality
MAX_PEAK_DBFS = -1.0

# Embedding deduplication
DUPLICATE_EMBEDDING_THRESHOLD = 0.995


# ============================================================
# BASIC DB
# ============================================================


def _ensure_db_dir() -> None:
    VOICE_DB_DIR.mkdir(parents=True, exist_ok=True)
    SPEAKERS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if not VOICE_DB_INDEX.exists():
        _atomic_write_json(VOICE_DB_INDEX, {})


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomowy zapis JSON: najpierw .tmp, potem replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    tmp_path.replace(path)


def load_voice_db() -> dict[str, Any]:
    _ensure_db_dir()

    try:
        with VOICE_DB_INDEX.open("r", encoding="utf-8") as f:
            db = json.load(f)

        if not isinstance(db, dict):
            state.add_log("⚠️ Voice DB ma nieprawidłowy format — używam pustej bazy.")
            return {}

        return db

    except Exception as e:
        state.add_log(f"⚠️ Błąd odczytu Voice DB: {e}")
        return {}


def save_voice_db(db: dict[str, Any]) -> None:
    _ensure_db_dir()
    _atomic_write_json(VOICE_DB_INDEX, db)


# ============================================================
# HELPERS
# ============================================================


def _safe_filename(value: str, fallback: str = "voice") -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value[:80] or fallback


def _new_voice_id(display_name: str) -> str:
    """
    Tworzy stabilny identyfikator profilu niezależny od display_name.

    Display name można potem zmienić bez zmiany voice_id.
    """
    base = _safe_filename(display_name, "voice")
    return f"{base}_{uuid.uuid4().hex[:8]}"


def _active_project_name() -> str:
    project = getattr(state, "active_project", None)
    if project is None:
        return ""
    return str(getattr(project, "name", "") or "")


def compute_embedding_hash(embedding: np.ndarray) -> str:
    """Return a short, stable hash for an embedding."""
    if embedding is None:
        return ""

    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return ""

    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _normalize_embedding(
    embedding: np.ndarray | list[float] | None,
) -> np.ndarray | None:
    if embedding is None:
        return None

    try:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception:
        return None

    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None

    norm = np.linalg.norm(arr)
    if norm <= 1e-12:
        return None

    return arr / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_embedding(a)
    b = _normalize_embedding(b)

    if a is None or b is None or a.shape != b.shape:
        return 0.0

    return float(np.dot(a, b))


def _profile_embedding_samples(info: dict[str, Any]) -> list[np.ndarray]:
    """
    Obsługa:
    - schema v2: embedding_samples,
    - eksperymentalne starsze wpisy: embeddings,
    - legacy: pojedyncze embedding.
    """
    result: list[np.ndarray] = []

    raw_samples = info.get("embedding_samples", [])
    if not isinstance(raw_samples, list) or not raw_samples:
        raw_samples = info.get("embeddings", [])

    if isinstance(raw_samples, list):
        for raw in raw_samples:
            emb = _normalize_embedding(raw)
            if emb is not None:
                result.append(emb)

    if not result:
        old_emb = _normalize_embedding(info.get("embedding"))
        if old_emb is not None:
            result.append(old_emb)

    return result


def _calculate_centroid(embeddings: list[np.ndarray]) -> np.ndarray | None:
    normalized = []

    for emb in embeddings:
        item = _normalize_embedding(emb)
        if item is not None:
            normalized.append(item)

    if not normalized:
        return None

    try:
        stack = np.vstack(normalized).astype(np.float32)
    except ValueError:
        return None

    centroid = np.mean(stack, axis=0)
    return _normalize_embedding(centroid)


def _deduplicate_embeddings(
    embeddings: list[np.ndarray],
) -> list[np.ndarray]:
    unique: list[np.ndarray] = []

    for emb in embeddings:
        normalized = _normalize_embedding(emb)
        if normalized is None:
            continue

        duplicate = any(
            _cosine(normalized, existing) >= DUPLICATE_EMBEDDING_THRESHOLD
            for existing in unique
        )

        if not duplicate:
            unique.append(normalized)

    return unique


def _limit_embedding_history(
    embeddings: list[np.ndarray],
    limit: int = MAX_EMBEDDINGS_PER_PROFILE,
) -> list[np.ndarray]:
    if len(embeddings) <= limit:
        return embeddings

    indices = np.linspace(0, len(embeddings) - 1, limit).astype(int)
    return [embeddings[i] for i in indices]


def _merge_embeddings(
    existing: list[np.ndarray],
    new_embeddings: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray | None]:
    merged = _deduplicate_embeddings(list(existing) + list(new_embeddings))
    merged = _limit_embedding_history(merged)
    centroid = _calculate_centroid(merged)
    return merged, centroid


# ============================================================
# AUDIO PREPROCESSING
# ============================================================


def preprocess_reference_audio(audio_segment: AudioSegment) -> AudioSegment:
    """
    Przygotowanie audio dla TTS/reference:
    mono, 24 kHz, 16-bit PCM, normalizacja peaku.
    """
    audio = (
        audio_segment
        .set_frame_rate(REFERENCE_SAMPLE_RATE)
        .set_channels(1)
        .set_sample_width(2)
    )

    if len(audio) == 0:
        return audio

    try:
        if audio.max_possible_amplitude > 0 and audio.max > 0:
            change_in_db = MAX_PEAK_DBFS - audio.max_dBFS
            if abs(change_in_db) > 0.05:
                audio = audio.apply_gain(change_in_db)
    except Exception:
        pass

    return audio


def preprocess_ecapa_audio(audio_segment: AudioSegment) -> AudioSegment:
    """ECAPA: mono, 16 kHz, 16-bit PCM."""
    return (
        audio_segment
        .set_frame_rate(ECAPA_SAMPLE_RATE)
        .set_channels(1)
        .set_sample_width(2)
    )


def _is_usable_audio(audio: AudioSegment) -> bool:
    if audio is None or len(audio) < MIN_SPEECH_SEGMENT_MS:
        return False

    try:
        if audio.rms <= 0:
            return False

        dbfs = float(audio.dBFS)
        if not np.isfinite(dbfs):
            return False

        if dbfs < MIN_ACCEPTABLE_DBFS:
            return False

    except Exception:
        return False

    return True


def _speech_only(audio: AudioSegment) -> AudioSegment:
    """
    Usuwa długie fragmenty ciszy.

    To nie jest pełny VAD, ale zapobiega wrzucaniu do ECAPA i TTS
    dużych porcji ciszy pomiędzy wypowiedziami.
    """
    if len(audio) < MIN_EMBEDDING_SEGMENT_MS:
        return audio

    try:
        dbfs = float(audio.dBFS)
        if not np.isfinite(dbfs):
            return audio

        silence_thresh = max(dbfs - 32.0, MIN_ACCEPTABLE_DBFS)
        ranges = detect_nonsilent(
            audio,
            min_silence_len=SILENCE_MIN_MS,
            silence_thresh=silence_thresh,
            seek_step=10,
        )

        if not ranges:
            return audio

        result = AudioSegment.empty()
        half_pad = SILENCE_PADDING_MS // 2

        for start_ms, end_ms in ranges:
            start_ms = max(0, start_ms - half_pad)
            end_ms = min(len(audio), end_ms + half_pad)

            duration = end_ms - start_ms
            if duration >= MIN_SPEECH_SEGMENT_MS:
                result += audio[start_ms:end_ms]

        return result if len(result) >= MIN_EMBEDDING_SEGMENT_MS else audio

    except Exception:
        return audio


def _split_for_embedding(audio: AudioSegment) -> list[AudioSegment]:
    """
    Dzieli materiał na krótkie próbki do ECAPA.

    Najpierw usuwamy długie cisze, potem dzielimy na maks. 8-sekundowe
    fragmenty i rozstawiamy próbki po całym materiale.
    """
    audio = preprocess_ecapa_audio(_speech_only(audio))

    segments: list[AudioSegment] = []
    pos = 0

    while pos < len(audio):
        end = min(pos + MAX_SPEECH_SEGMENT_MS, len(audio))
        chunk = audio[pos:end]

        if len(chunk) >= MIN_EMBEDDING_SEGMENT_MS and _is_usable_audio(chunk):
            segments.append(chunk)

        pos = end

    if len(segments) > MAX_NEW_EMBEDDINGS_PER_UPDATE:
        indices = np.linspace(
            0,
            len(segments) - 1,
            MAX_NEW_EMBEDDINGS_PER_UPDATE,
        ).astype(int)
        segments = [segments[i] for i in indices]

    return segments


def _audio_to_tensor(audio: AudioSegment) -> torch.Tensor:
    audio = preprocess_ecapa_audio(audio)
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32)

    if audio.sample_width == 1:
        divisor = 128.0
    elif audio.sample_width == 2:
        divisor = 32768.0
    elif audio.sample_width == 4:
        divisor = 2147483648.0
    else:
        divisor = float(1 << (8 * audio.sample_width - 1))

    samples /= divisor
    return torch.from_numpy(samples).unsqueeze(0)


def _combine_chunks(audio_chunks: list[AudioSegment]) -> AudioSegment:
    """
    Łączy tylko użyteczne fragmenty.

    Nie dodajemy tu sztucznej ciszy — dla ECAPA nie jest potrzebna,
    a TTS reference i tak później przechodzi przez _speech_only().
    """
    combined = AudioSegment.empty()

    for chunk in audio_chunks:
        if chunk is None:
            continue

        try:
            candidate = chunk.set_channels(1).set_sample_width(2)
        except Exception:
            candidate = chunk

        if not _is_usable_audio(candidate):
            continue

        combined += candidate

    return combined


def _cap_reference(audio: AudioSegment) -> AudioSegment:
    if len(audio) <= MAX_REF_DURATION_MS:
        return audio
    return audio[:MAX_REF_DURATION_MS]


def _load_import_candidate(path: Path) -> tuple[Path, AudioSegment | None, float]:
    """Decode/clean one file. Safe to run in the bounded import worker pool."""
    try:
        audio = AudioSegment.from_file(path).set_channels(1).set_sample_width(2)
        if not _is_usable_audio(audio):
            return path, None, -999.0
        audio = _speech_only(audio)
        if len(audio) < MIN_SPEECH_SEGMENT_MS:
            return path, None, -999.0
        # Do not let a single long recording dominate the reference candidate set.
        audio = audio[:MAX_SPEECH_SEGMENT_MS]
        peak_room = max(0.0, -1.0 - float(audio.max_dBFS))
        score = min(len(audio), 6000) / 1000.0 - peak_room * 0.08
        return path, audio, score
    except Exception as exc:
        state.add_log(f"  ⚠️ Nie udało się załadować {path.name}: {exc}")
        return path, None, -999.0


def _prosody_label(audio: AudioSegment) -> str:
    """Lightweight acoustic buckets used to select a matching XTTS reference."""
    if not audio or not np.isfinite(audio.dBFS):
        return "neutral"
    mono = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    samples = np.asarray(mono.get_array_of_samples(), dtype=np.float32)
    zcr = float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1]))) if samples.size > 1 else 0.0
    if audio.dBFS > -16.0 and zcr > 0.08:
        return "intense"
    if audio.dBFS < -30.0:
        return "soft"
    if zcr > 0.14:
        return "bright"
    if zcr < 0.035 and audio.dBFS > -24.0:
        return "low_calm"
    return "neutral"


def _build_diverse_reference(candidates: list[tuple[Path, AudioSegment, float]]) -> AudioSegment:
    """Choose material across quiet/normal/energetic files instead of the first 30 s."""
    if not candidates:
        return AudioSegment.empty()
    ordered = sorted(candidates, key=lambda item: float(item[1].dBFS))
    buckets = [ordered[i::4] for i in range(4)]
    for bucket in buckets:
        bucket.sort(key=lambda item: item[2], reverse=True)
    selected = AudioSegment.empty()
    while len(selected) < MAX_REF_DURATION_MS and any(buckets):
        progressed = False
        for bucket in buckets:
            if not bucket or len(selected) >= MAX_REF_DURATION_MS:
                continue
            _path, audio, _score = bucket.pop(0)
            remaining = MAX_REF_DURATION_MS - len(selected)
            if len(selected):
                silence = min(90, remaining)
                selected += AudioSegment.silent(duration=silence)
                remaining -= silence
            if remaining > 0:
                selected += audio[:remaining]
                progressed = True
        if not progressed:
            break
    return selected


# ============================================================
# ECAPA
# ============================================================


def generate_ecapa_embeddings(
    audio: AudioSegment,
    max_embeddings: int = MAX_NEW_EMBEDDINGS_PER_UPDATE,
) -> list[np.ndarray]:
    """Generate embeddings from several samples of the same voice."""
    segments = _split_for_embedding(audio)
    segments = segments[:max(1, int(max_embeddings))]

    if not segments:
        return []

    classifier = None

    try:
        from speechbrain.inference.speaker import EncoderClassifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir="audio/ecapa_cache",
        )

        embeddings: list[np.ndarray] = []

        for segment in segments:
            tensor = None
            output = None

            try:
                tensor = _audio_to_tensor(segment).to(device)

                with torch.no_grad():
                    output = classifier.encode_batch(tensor)

                emb = (
                    output
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                emb = _normalize_embedding(emb)

                if emb is not None:
                    embeddings.append(emb)

            except Exception as e:
                state.add_log(f"  ⚠️ Pominięto fragment ECAPA: {e}")

            finally:
                if tensor is not None:
                    del tensor
                if output is not None:
                    del output

        return _deduplicate_embeddings(embeddings)

    except Exception as e:
        state.add_log(f"⚠️ Nie udało się załadować ECAPA: {e}")
        return []

    finally:
        if classifier is not None:
            del classifier

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# ============================================================
# VOICE MATCHING
# ============================================================


def _score_profile_against_embeddings(
    query_embeddings: list[np.ndarray],
    info: dict[str, Any],
) -> float:
    """
    Ocenia kilka embeddingów nowego nagrania względem jednego profilu.

    70%: podobieństwo centroidów
    30%: średnia najlepszych dopasowań poszczególnych próbek

    Dzięki temu jedna nietypowa kwestia (krzyk/szept) mniej wpływa na wynik.
    """
    queries = _deduplicate_embeddings(query_embeddings)
    if not queries:
        return 0.0

    query_centroid = _calculate_centroid(queries)
    if query_centroid is None:
        return 0.0

    profile_samples = _profile_embedding_samples(info)
    profile_centroid = _normalize_embedding(info.get("centroid"))

    if profile_centroid is None and profile_samples:
        profile_centroid = _calculate_centroid(profile_samples)

    if profile_centroid is None and not profile_samples:
        return 0.0

    centroid_score = (
        _cosine(query_centroid, profile_centroid)
        if profile_centroid is not None
        else 0.0
    )

    per_query_scores: list[float] = []

    for query in queries:
        if profile_samples:
            score = max(_cosine(query, sample) for sample in profile_samples)
        elif profile_centroid is not None:
            score = _cosine(query, profile_centroid)
        else:
            score = 0.0

        per_query_scores.append(score)

    if per_query_scores:
        # Prevent one unusually strong sample from dominating the score.
        best_scores = sorted(per_query_scores, reverse=True)
        take = min(3, len(best_scores))
        sample_score = float(np.mean(best_scores[:take]))
    else:
        sample_score = centroid_score

    score = 0.70 * centroid_score + 0.30 * sample_score
    return float(max(-1.0, min(1.0, score)))


def find_matching_voice_from_embeddings(
    embeddings: list[np.ndarray],
    db: dict[str, Any] | None = None,
) -> tuple[str | None, float]:
    """
    Dopasowuje zestaw embeddingów do Voice DB.

    Zwraca kandydata dopiero od MATCH_THRESHOLD.
    Samo automatyczne scalanie odbywa się dopiero od STRONG_MATCH_THRESHOLD.
    """
    if db is None:
        db = load_voice_db()

    query_embeddings = _deduplicate_embeddings(embeddings)
    if not query_embeddings or not db:
        return None, 0.0

    best_voice_id: str | None = None
    best_score = -1.0

    for voice_id, info in db.items():
        if not isinstance(info, dict):
            continue

        score = _score_profile_against_embeddings(query_embeddings, info)

        if score > best_score:
            best_score = score
            best_voice_id = voice_id

    if best_voice_id is not None and best_score >= MATCH_THRESHOLD:
        return best_voice_id, best_score

    return None, max(0.0, best_score)


def find_matching_voice(
    embedding: np.ndarray,
    db: dict[str, Any] | None = None,
) -> tuple[str | None, float]:
    """Backward-compatible matching dla pojedynczego embeddingu."""
    normalized = _normalize_embedding(embedding)
    if normalized is None:
        return None, 0.0

    return find_matching_voice_from_embeddings([normalized], db)


# ============================================================
# PROFILE AUDIO
# ============================================================


def _write_profile_audio(
    voice_id: str,
    audio: AudioSegment,
) -> tuple[str, str]:
    _ensure_db_dir()

    safe_id = _safe_filename(voice_id, "voice")
    wav_path = VOICE_DB_DIR / f"{safe_id}.wav"
    preview_path = VOICE_DB_DIR / f"{safe_id}_preview.wav"

    audio = preprocess_reference_audio(_cap_reference(audio))
    audio.export(wav_path, format="wav")

    if len(audio) >= 3000:
        audio[:3000].export(preview_path, format="wav")
        preview_str = str(preview_path)
    else:
        preview_str = ""
        try:
            if preview_path.exists():
                preview_path.unlink()
        except Exception:
            pass

    return str(wav_path), preview_str


def _sync_speakers_audio(voice_id: str, audio: AudioSegment) -> None:
    """Create the WAV copy expected by the current TTS service."""
    _ensure_db_dir()
    safe_id = _safe_filename(voice_id, "voice")
    compat_path = SPEAKERS_AUDIO_DIR / f"{safe_id}.wav"
    preprocess_reference_audio(_cap_reference(audio)).export(
        compat_path,
        format="wav",
    )


def get_voice_reference_path(voice_id: str) -> str:
    """Return the profile's TTS reference when it exists."""
    if not voice_id:
        return ""

    db = load_voice_db()
    info = db.get(voice_id, {})

    if isinstance(info, dict):
        wav_path = str(info.get("wav_path", "") or "")
        if wav_path and os.path.isfile(wav_path):
            return wav_path

    compat = SPEAKERS_AUDIO_DIR / f"{_safe_filename(voice_id)}.wav"
    if compat.is_file():
        return str(compat)

    return ""


# ============================================================
# PROFILE CREATION
# ============================================================


def create_voice_profile(
    display_name: str,
    source_movie: str,
    description: str,
    audio_chunks: list[AudioSegment],
    embedding: np.ndarray | None,
    embedding_samples: list[np.ndarray] | None = None,
) -> str:
    """Create a persistent voice profile with its own voice ID."""
    _ensure_db_dir()
    db = load_voice_db()

    combined = _combine_chunks(audio_chunks)
    if len(combined) < 1000:
        state.add_log("⚠️ Nie utworzono profilu — za mało materiału audio.")
        return ""

    combined = _speech_only(combined)
    combined = _cap_reference(combined)

    voice_id = _new_voice_id(display_name)

    supplied_embeddings: list[np.ndarray] = []

    normalized = _normalize_embedding(embedding)
    if normalized is not None:
        supplied_embeddings.append(normalized)

    if embedding_samples is not None:
        for emb in embedding_samples:
            normalized_emb = _normalize_embedding(emb)
            if normalized_emb is not None:
                supplied_embeddings.append(normalized_emb)

    if not supplied_embeddings:
        supplied_embeddings = generate_ecapa_embeddings(
            combined,
            max_embeddings=MAX_NEW_EMBEDDINGS_PER_UPDATE,
        )

    unique_samples = _deduplicate_embeddings(supplied_embeddings)
    unique_samples = _limit_embedding_history(unique_samples)
    centroid = _calculate_centroid(unique_samples)

    wav_path, preview_path = _write_profile_audio(voice_id, combined)
    _sync_speakers_audio(voice_id, combined)

    now = time.time()
    source_projects = []
    active_project = _active_project_name()
    if active_project:
        source_projects.append(active_project)

    db[voice_id] = {
        "schema_version": 2,
        "display_name": str(display_name or voice_id),
        "source_movie": str(source_movie or ""),
        "source_projects": source_projects,
        "description": str(description or ""),
        "wav_path": wav_path,
        "preview_path": preview_path,
        "duration_sec": round(len(combined) / 1000.0, 2),
        # embedding pozostaje dla backward compatibility.
        "embedding": centroid.tolist() if centroid is not None else [],
        "centroid": centroid.tolist() if centroid is not None else [],
        "embedding_samples": [e.tolist() for e in unique_samples],
        "embedding_count": len(unique_samples),
        "sample_count": len(audio_chunks),
        "created_at": now,
        "updated_at": now,
    }

    save_voice_db(db)

    state.add_log(
        f"💾 Utworzono NOWY profil '{display_name}' → {voice_id} "
        f"({len(combined) / 1000:.1f}s, {len(unique_samples)} embeddingów)."
    )

    return voice_id


# ============================================================
# PROFILE UPDATE
# ============================================================


def _update_existing_voice_profile(
    voice_id: str,
    new_audio: AudioSegment,
    new_embeddings: list[np.ndarray],
    audio_chunk_count: int,
    db: dict[str, Any],
) -> str:
    existing_meta = db.get(voice_id, {})
    if not isinstance(existing_meta, dict):
        existing_meta = {}

    existing_audio = AudioSegment.empty()
    wav_path_existing = str(existing_meta.get("wav_path", "") or "")

    if wav_path_existing and os.path.exists(wav_path_existing):
        try:
            existing_audio = AudioSegment.from_file(wav_path_existing)
        except Exception as e:
            state.add_log(f"⚠️ Nie udało się wczytać starej referencji: {e}")

    # Do not let one new video completely replace the previous reference.
    if len(existing_audio) >= MAX_REF_DURATION_MS:
        final_audio = existing_audio[:MAX_REF_DURATION_MS]
    else:
        remaining = MAX_REF_DURATION_MS - len(existing_audio)
        final_audio = existing_audio + new_audio[:remaining]

    final_audio = preprocess_reference_audio(_cap_reference(_speech_only(final_audio)))

    old_samples = _profile_embedding_samples(existing_meta)
    embedding_samples, centroid = _merge_embeddings(
        old_samples,
        new_embeddings,
    )

    wav_path, preview_path = _write_profile_audio(voice_id, final_audio)
    _sync_speakers_audio(voice_id, final_audio)

    now = time.time()

    source_projects = existing_meta.get("source_projects", [])
    if not isinstance(source_projects, list):
        source_projects = []

    active_project = _active_project_name()
    if active_project and active_project not in source_projects:
        source_projects.append(active_project)

    db[voice_id] = {
        "schema_version": 2,
        "display_name": existing_meta.get("display_name", voice_id),
        "source_movie": existing_meta.get("source_movie", ""),
        "source_projects": source_projects,
        "description": existing_meta.get("description", ""),
        "wav_path": wav_path,
        "preview_path": preview_path or existing_meta.get("preview_path", ""),
        "duration_sec": round(len(final_audio) / 1000.0, 2),
        "embedding": centroid.tolist() if centroid is not None else [],
        "centroid": centroid.tolist() if centroid is not None else [],
        "embedding_samples": [e.tolist() for e in embedding_samples],
        "embedding_count": len(embedding_samples),
        "sample_count": int(existing_meta.get("sample_count", 0)) + int(audio_chunk_count),
        "created_at": existing_meta.get("created_at", now),
        "updated_at": now,
    }

    save_voice_db(db)

    return voice_id


# ============================================================
# MAIN AUTO IDENTIFICATION / UPDATE
# ============================================================


def update_or_create_voice_profile(
    speaker_id: str,
    audio_chunks: list[AudioSegment],
    embedding: np.ndarray | None = None,
    embedding_samples: list[np.ndarray] | None = None,
) -> str:
    """
    Rozpoznaje osobę i zwraca TRWAŁY voice_id.

    Krytyczna kolejność:
        1. przygotuj nowe audio,
        2. policz ECAPA embeddings,
        3. dopasuj do istniejącej Voice DB,
        4. dopiero potem update/create.

    speaker_id (np. SPEAKER_00) służy tylko jako tymczasowy display_name dla
    nowego profilu. Nie jest używany jako trwały identyfikator osoby.
    """
    _ensure_db_dir()

    speaker_label = str(speaker_id or "Unknown").strip() or "Unknown"
    combined = _combine_chunks(audio_chunks)

    if len(combined) < 1000:
        state.add_log(
            f"⚠️ {speaker_label}: za mało użytecznego audio do Voice DB "
            f"({len(combined) / 1000:.1f}s)."
        )
        # Preserve compatibility with the current TTS service.
        return speaker_label

    combined = _speech_only(combined)

    # --------------------------------------------------------
    # 1. NOWE EMBEDDINGI — PRZED MATCHINGIEM
    # --------------------------------------------------------

    new_embeddings: list[np.ndarray] = []

    normalized_embedding = _normalize_embedding(embedding)
    if normalized_embedding is not None:
        new_embeddings.append(normalized_embedding)

    if embedding_samples is not None:
        for emb in embedding_samples:
            normalized = _normalize_embedding(emb)
            if normalized is not None:
                new_embeddings.append(normalized)

    new_embeddings = _deduplicate_embeddings(new_embeddings)

    # Generate embeddings here when the caller did not provide them.
    # liczymy ECAPA TERAZ, a nie po decyzji o profilu.
    if not new_embeddings:
        state.add_log(
            f"🧠 {speaker_label}: generuję embeddingi ECAPA przed matchingiem Voice DB..."
        )
        new_embeddings = generate_ecapa_embeddings(
            combined,
            max_embeddings=MAX_NEW_EMBEDDINGS_PER_UPDATE,
        )

    # --------------------------------------------------------
    # 2. MATCH AGAINST EXISTING PROFILES
    # --------------------------------------------------------

    db = load_voice_db()

    matched_voice_id: str | None = None
    similarity = 0.0

    if new_embeddings and db:
        matched_voice_id, similarity = find_matching_voice_from_embeddings(
            new_embeddings,
            db,
        )

    # --------------------------------------------------------
    # 3. STRONG MATCH: UPDATE THE EXISTING PROFILE
    # --------------------------------------------------------

    if (
        matched_voice_id is not None
        and similarity >= STRONG_MATCH_THRESHOLD
    ):
        display_name = str(
            db.get(matched_voice_id, {}).get("display_name", matched_voice_id)
        )

        state.add_log(
            f"✅ Voice DB MATCH: {speaker_label} → {display_name} "
            f"[{matched_voice_id}] score={similarity * 100:.1f}%"
        )

        _update_existing_voice_profile(
            voice_id=matched_voice_id,
            new_audio=combined,
            new_embeddings=new_embeddings,
            audio_chunk_count=len(audio_chunks),
            db=db,
        )

        return matched_voice_id

    # --------------------------------------------------------
    # 4. NIEPEWNY MATCH -> NIE SCALAMY AUTOMATYCZNIE
    # --------------------------------------------------------

    if matched_voice_id is not None:
        candidate_name = str(
            db.get(matched_voice_id, {}).get("display_name", matched_voice_id)
        )

        state.add_log(
            f"⚠️ Voice DB UNCERTAIN: {speaker_label} przypomina "
            f"'{candidate_name}' [{matched_voice_id}] "
            f"score={similarity * 100:.1f}%, ale próg auto-merge to "
            f"{STRONG_MATCH_THRESHOLD * 100:.0f}%. Tworzę osobny profil."
        )

    elif new_embeddings:
        state.add_log(
            f"🆕 Voice DB: {speaker_label} nie pasuje do istniejących profili."
        )

    else:
        state.add_log(
            f"⚠️ Voice DB: {speaker_label} — ECAPA nie wygenerowało embeddingów. "
            "Dla bezpieczeństwa tworzę osobny profil zamiast zgadywać."
        )

    # --------------------------------------------------------
    # 5. NOWY STABILNY PROFIL
    # --------------------------------------------------------

    primary_embedding = new_embeddings[0] if new_embeddings else None

    voice_id = create_voice_profile(
        display_name=speaker_label,
        source_movie=_active_project_name(),
        description="Auto-created from diarization speaker",
        audio_chunks=[combined],
        embedding=primary_embedding,
        embedding_samples=new_embeddings,
    )

    if not voice_id:
        # Give TTS a usable fallback without persisting SPEAKER_00 as a voice ID.
        return speaker_label

    return voice_id


# ============================================================
# IMPORT
# ============================================================


def import_voice_from_folder(
    folder_path: str,
    display_name: str,
    source_movie: str = "",
    description: str = "",
    existing_voice_id: str = "",
) -> str:
    """Importuje wspierane pliki audio z folderu do jednego profilu."""
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        state.add_log(f"❌ Folder nie istnieje: {folder_path}")
        return ""

    state.add_log(f"📂 Importowanie głosu '{display_name}' z: {folder_path}")

    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in supported],
        key=lambda p: p.name.lower(),
    )

    if not audio_files:
        state.add_log("❌ Nie znaleziono plików audio w folderze.")
        return ""

    state.add_log(f"  🎵 Znaleziono {len(audio_files)} plików audio.")

    workers = max(1, min(8, os.cpu_count() or 1))
    state.add_log(f"  ⚙️ Analiza jakości na {workers} wątkach...")
    candidates: list[tuple[Path, AudioSegment, float]] = []
    emotion_catalog = []
    usable_files = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="voice-import") as pool:
        for index, (path, audio, score) in enumerate(pool.map(_load_import_candidate, audio_files), 1):
            if audio is not None:
                usable_files += 1
                emotion_catalog.append({
                    "file": str(path.resolve()),
                    "emotion": _prosody_label(audio),
                    "duration_sec": round(len(audio) / 1000.0, 3),
                    "energy_dbfs": round(float(audio.dBFS), 2),
                    "quality_score": round(float(score), 3),
                })
                candidates.append((path, audio, score))
                # Bound RAM for huge datasets. Preserve energy diversity while
                # retaining the best candidates seen so far.
                if len(candidates) >= 512:
                    by_energy = sorted(candidates, key=lambda item: float(item[1].dBFS))
                    candidates = sorted(
                        (item for bucket in (by_energy[i::4] for i in range(4)) for item in sorted(bucket, key=lambda x: x[2], reverse=True)[:64]),
                        key=lambda item: item[0].name.lower(),
                    )
            if index % 250 == 0 or index == len(audio_files):
                state.add_log(f"    ✓ sprawdzono {index}/{len(audio_files)} plików")

    loaded_files = usable_files
    combined = _build_diverse_reference(candidates)

    if len(combined) < MIN_REF_DURATION_MS:
        state.add_log("❌ Za mało materiału audio po imporcie.")
        return ""

    combined = _speech_only(combined)

    embeddings = generate_ecapa_embeddings(
        combined,
        max_embeddings=MAX_NEW_EMBEDDINGS_PER_UPDATE,
    )

    primary_embedding = embeddings[0] if embeddings else None

    if existing_voice_id:
        db = load_voice_db()
        if existing_voice_id not in db:
            state.add_log(f"❌ Profil do aktualizacji nie istnieje: {existing_voice_id}")
            return ""
        voice_id = _update_existing_voice_profile(existing_voice_id, combined, embeddings, loaded_files, db)
        update_voice_metadata(voice_id, display_name, source_movie, description)
    else:
        voice_id = create_voice_profile(
            display_name=display_name,
            source_movie=source_movie,
            description=description,
            audio_chunks=[combined],
            embedding=primary_embedding,
            embedding_samples=embeddings,
        )

    if embeddings:
        state.add_log(
            f"✅ Zaimportowano '{display_name}': "
            f"{len(embeddings)} embeddingów z {loaded_files} plików."
        )
    else:
        state.add_log(
            f"⚠️ Zaimportowano '{display_name}', ale ECAPA nie wygenerowało embeddingu."
        )

    if voice_id:
        db = load_voice_db()
        info = db.get(voice_id)
        if isinstance(info, dict):
            emotion_dir = VOICE_DB_DIR / "emotions" / _safe_filename(voice_id)
            emotion_dir.mkdir(parents=True, exist_ok=True)
            emotion_refs = {}
            emotion_counts = {
                label: sum(item["emotion"] == label for item in emotion_catalog)
                for label in ("neutral", "soft", "intense", "bright", "low_calm")
            }
            for label in ("neutral", "soft", "intense", "bright", "low_calm"):
                group = [item for item in candidates if _prosody_label(item[1]) == label]
                if not group:
                    continue
                emotion_audio = preprocess_reference_audio(_build_diverse_reference(group)[:15000])
                if len(emotion_audio) < MIN_REF_DURATION_MS:
                    continue
                emotion_path = emotion_dir / f"{label}.wav"
                emotion_audio.export(emotion_path, format="wav")
                emotion_refs[label] = str(emotion_path)
            catalog_path = emotion_dir / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in emotion_catalog),
                encoding="utf-8",
            )
            info["sample_count"] = loaded_files
            info["source_file_count"] = len(audio_files)
            info["reference_strategy"] = "quality_emotion_diverse_v1"
            info["reference_limit_sec"] = MAX_REF_DURATION_MS / 1000.0
            info["emotion_references"] = emotion_refs
            info["emotion_sample_counts"] = emotion_counts
            info["emotion_catalog"] = str(catalog_path)
            save_voice_db(db)

    return voice_id


# ============================================================
# METADATA / DELETE
# ============================================================


def update_voice_metadata(
    voice_id: str,
    display_name: str | None = None,
    source_movie: str | None = None,
    description: str | None = None,
) -> bool:
    """Aktualizuje metadane profilu bez zmiany stabilnego voice_id."""
    db = load_voice_db()

    if voice_id not in db:
        return False

    if display_name is not None:
        db[voice_id]["display_name"] = display_name
    if source_movie is not None:
        db[voice_id]["source_movie"] = source_movie
    if description is not None:
        db[voice_id]["description"] = description

    db[voice_id]["updated_at"] = time.time()
    save_voice_db(db)
    return True


def delete_voice_profile(voice_id: str) -> bool:
    """Delete a profile and its TTS reference, preview, and compatibility WAV."""
    db = load_voice_db()

    if voice_id not in db:
        return False

    info = db.get(voice_id, {})

    paths = {
        info.get("wav_path", "") if isinstance(info, dict) else "",
        info.get("preview_path", "") if isinstance(info, dict) else "",
        str(SPEAKERS_AUDIO_DIR / f"{_safe_filename(voice_id)}.wav"),
    }

    for raw_path in paths:
        if not raw_path:
            continue

        try:
            path = Path(raw_path)
            if path.exists() and path.is_file():
                path.unlink()
        except Exception as e:
            state.add_log(f"⚠️ Nie udało się usunąć pliku '{raw_path}': {e}")

    del db[voice_id]
    save_voice_db(db)

    state.add_log(f"🗑️ Usunięto profil głosu '{voice_id}'.")
    return True


# ============================================================
# DB MAINTENANCE / MIGRATION
# ============================================================


def migrate_voice_db() -> int:
    """
    Migruje stare wpisy do schema_version=2.

    Nie zmienia istniejących voice_id, dzięki czemu nie psuje odwołań
    ze starszych projektów.
    """
    db = load_voice_db()
    changed = 0

    for voice_id, info in db.items():
        if not isinstance(info, dict):
            continue

        needs_migration = int(info.get("schema_version", 1)) < 2
        needs_samples_fix = (
            "embedding_samples" not in info
            and ("embeddings" in info or "embedding" in info)
        )

        if not needs_migration and not needs_samples_fix:
            continue

        samples = _profile_embedding_samples(info)
        samples = _deduplicate_embeddings(samples)
        samples = _limit_embedding_history(samples)
        centroid = _calculate_centroid(samples)

        info["schema_version"] = 2
        info["embedding"] = (
            centroid.tolist()
            if centroid is not None
            else info.get("embedding", [])
        )
        info["centroid"] = centroid.tolist() if centroid is not None else []
        info["embedding_samples"] = [e.tolist() for e in samples]
        info["embedding_count"] = len(samples)
        info.setdefault("source_projects", [])
        info["updated_at"] = time.time()
        changed += 1

    if changed:
        save_voice_db(db)
        state.add_log(f"🔧 Zmigrowano {changed} profili Voice DB do schema v2.")

    return changed
