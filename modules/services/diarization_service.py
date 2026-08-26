# -*- coding: utf-8 -*-
"""
Speaker diarization service

Primary:
    Pyannote Speaker Diarization

Fallback:
    SpeechBrain ECAPA-TDNN + Agglomerative Clustering + Post-processing

Input:
    video/audio file + Whisper timestamps

Output:
    list of speaker labels, one label per timestamp
"""

import os
import gc
import numpy as np
import torch

from pydub import AudioSegment

from modules.state import state


# ============================================================
# CONFIGURATION
# ============================================================

DIAR_AUDIO_DIR = "audio"
DIAR_WAV_PATH = os.path.join(DIAR_AUDIO_DIR, "diar_temp.wav")

ECAPA_CACHE_DIR = os.path.join(
    DIAR_AUDIO_DIR,
    "ecapa_cache"
)

# ECAPA-TDNN expects 16 kHz mono audio
ECAPA_SAMPLE_RATE = 16000

# Maximum number of speakers when automatic detection is used (zwiększone dla seriali)
DEFAULT_MAX_SPEAKERS = 20

# Minimum useful segment length for ECAPA
MIN_SEGMENT_MS = 300


# ============================================================
# UTILITY
# ============================================================

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _prepare_audio(wav_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(wav_path)
    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(ECAPA_SAMPLE_RATE)
        .set_sample_width(2)
    )
    return audio


def _extract_audio(video_path: str, output_path: str):
    state.add_log("  🎵 Ekstrakcja i przygotowanie audio dla diarization...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    audio = AudioSegment.from_file(video_path)
    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(ECAPA_SAMPLE_RATE)
        .set_sample_width(2)
    )
    audio.export(output_path, format="wav")
    state.add_log(f"  ✅ Audio przygotowane: mono / {ECAPA_SAMPLE_RATE} Hz")


# ============================================================
# PYANNOTE
# ============================================================

def _diarize_pyannote(
    wav_path: str,
    hf_token: str,
    device: str,
    num_speakers: int = None
):
    from pyannote.audio import Pipeline

    state.add_log("🗣️ Uruchamianie Speaker Diarization przez Pyannote...")

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
    except TypeError:
        state.add_log("  ℹ️ Wykryto starsze API Pyannote — używam use_auth_token.")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

    if device == "cuda":
        state.add_log("  🚀 Pyannote: używam CUDA.")
        pipeline.to(torch.device("cuda"))
    else:
        state.add_log("  🐌 Pyannote: używam CPU.")

    state.add_log("  📡 Analiza ścieżki audio...")

    if num_speakers is not None and num_speakers > 0:
        state.add_log(f"  👥 Wymuszona liczba speakerów: {num_speakers}")
        diarization = pipeline(wav_path, num_speakers=num_speakers)
    else:
        state.add_log("  👥 Automatyczne wykrywanie liczby speakerów...")
        diarization = pipeline(wav_path)

    return pipeline, diarization


# ============================================================
# MAP PYANNOTE → WHISPER
# ============================================================

def _map_diarization_to_timestamps(
    diarization,
    timestamps: list
) -> list:
    if not timestamps:
        return []

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))

    state.add_log(f"  📊 Pyannote wykrył {len(turns)} fragmentów speakerów.")

    result = []
    for start_ms, end_ms in timestamps:
        s_sec = max(0.0, start_ms / 1000.0)
        e_sec = max(s_sec, end_ms / 1000.0)

        best_speaker = "Unknown"
        best_overlap = 0.0

        for turn_start, turn_end, speaker in turns:
            overlap = max(0.0, min(e_sec, turn_end) - max(s_sec, turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        result.append(best_speaker)

    known = [speaker for speaker in result if speaker != "Unknown"]
    unique_speakers = len(set(known))
    state.add_log(f"  👥 Przypisano {unique_speakers} speakerów do {len(result)} segmentów.")

    return result


# ============================================================
# ECAPA EMBEDDINGS
# ============================================================

def _extract_embeddings(
    wav_path: str,
    timestamps: list,
    device: str
) -> np.ndarray:
    from speechbrain.inference.speaker import EncoderClassifier

    state.add_log("  🧠 Ładowanie modelu ECAPA-TDNN (SpeechBrain)...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=ECAPA_CACHE_DIR
    )

    audio = _prepare_audio(wav_path)
    embeddings = []

    state.add_log(f"  📊 Generowanie embeddingów dla {len(timestamps)} segmentów...")

    for index, (start_ms, end_ms) in enumerate(timestamps):
        start_ms = max(0, int(start_ms))
        end_ms = max(start_ms, int(end_ms))

        if end_ms - start_ms < MIN_SEGMENT_MS:
            center = (start_ms + end_ms) // 2
            context = 500
            start_ms = max(0, center - context)
            end_ms = min(len(audio), center + context)

        segment = audio[start_ms:end_ms]

        if len(segment) <= 0:
            embeddings.append(np.zeros(192, dtype=np.float32))
            continue

        samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
        samples /= 32768.0
        tensor = torch.from_numpy(samples).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = classifier.encode_batch(tensor)

        embedding = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
        embeddings.append(embedding)

        if (index + 1) % 20 == 0 or index == len(timestamps) - 1:
            state.add_log(f"    ✓ {index + 1}/{len(timestamps)} embeddingów")

    del classifier
    _cleanup_gpu()

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if len(embeddings) != len(timestamps):
        raise RuntimeError(
            f"ECAPA wygenerował nieprawidłową liczbę embeddingów: "
            f"{len(embeddings)} zamiast {len(timestamps)}."
        )

    return embeddings


# ============================================================
# AUTOMATIC CLUSTERING
# ============================================================

def _auto_cluster(
    embeddings: np.ndarray,
    max_speakers: int = DEFAULT_MAX_SPEAKERS
) -> list:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    embeddings_norm = normalize(embeddings)
    max_k = min(int(max_speakers), n - 1)
    if max_k < 2:
        return [0] * n

    best_score = -1.0
    best_labels = None
    best_k = 1

    for k in range(2, max_k + 1):
        try:
            # 'complete' linkage bardziej agresywnie rozdziela klastry
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="complete"
            )
            labels = clusterer.fit_predict(embeddings_norm)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(embeddings_norm, labels, metric="cosine")
            state.add_log(f"    🔍 k={k}: silhouette={score:.3f}")

            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
        except Exception as e:
            state.add_log(f"    ⚠️ Clustering k={k} failed: {e}")

    if best_labels is None:
        state.add_log("  ⚠️ Nie udało się określić liczby speakerów.")
        return [0] * n

    state.add_log(f"  🎯 Wykryto {best_k} speakerów (silhouette={best_score:.3f})")
    return best_labels.tolist()


# ============================================================
# POST-PROCESSING (ROZDZIELANIE SZYBKICH DIALOGÓW)
# ============================================================

def _postprocess_diarization(
    labels: list,
    embeddings: np.ndarray,
    timestamps: list,
    threshold_ms: int = 1500
) -> list:
    """
    Rozdziela segmenty, które są blisko siebie w czasie,
    ale mają różne głosy (niskie podobieństwo embeddingów),
    aby uniknąć łączenia szybkich dialogów w jeden głos.
    """
    if len(labels) <= 1:
        return labels

    from sklearn.metrics.pairwise import cosine_similarity

    improved_labels = labels.copy()

    for i in range(1, len(labels)):
        prev_start, prev_end = timestamps[i-1]
        curr_start, curr_end = timestamps[i]

        # Sprawdź czy segmenty są blisko siebie w czasie (np. przerwa < 1.5s)
        gap = curr_start - prev_end

        if gap <= threshold_ms:
            # Sprawdź czy embeddingi są różne
            emb_prev = embeddings[i-1].reshape(1, -1)
            emb_curr = embeddings[i].reshape(1, -1)
            similarity = cosine_similarity(emb_prev, emb_curr)[0][0]

            # Jeśli podobieństwo < 0.65, to prawdopodobnie różne osoby,
            # ale clustering mógł je połączyć. Rozdziel je.
            if similarity < 0.65 and labels[i] == labels[i-1]:
                existing_labels = set(improved_labels)
                new_label = max(existing_labels) + 1
                improved_labels[i] = new_label
                state.add_log(f"    🔀 Rozdzielono segment {i} (gap={gap}ms, sim={similarity:.2f})")

    return improved_labels


# ============================================================
# SPEAKER LABELS
# ============================================================

def _labels_to_names(labels: list) -> list:
    speaker_map = {}
    result = []

    for label in labels:
        label = int(label)
        if label not in speaker_map:
            speaker_map[label] = f"SPEAKER_{len(speaker_map):02d}"
        result.append(speaker_map[label])

    return result


# ============================================================
# SPEECHBRAIN FALLBACK
# ============================================================

def _diarize_speechbrain(
    wav_path: str,
    timestamps: list,
    device: str,
    num_speakers: int = None
) -> list:
    state.add_log("🗣️ Lokalna diarization: SpeechBrain ECAPA-TDNN + clustering...")

    embeddings = _extract_embeddings(wav_path, timestamps, device)

    if num_speakers is not None and num_speakers > 1:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize

        if num_speakers > len(embeddings):
            state.add_log(f"  ⚠️ num_speakers={num_speakers}, ale mamy tylko {len(embeddings)} segmentów.")
            num_speakers = len(embeddings)

        if num_speakers <= 1:
            labels = [0] * len(embeddings)
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=num_speakers,
                metric="cosine",
                linkage="complete"  # Zmienione z "average" na "complete"
            )
            labels = clusterer.fit_predict(normalize(embeddings)).tolist()
            state.add_log(f"  🎯 Użyto wymuszonej liczby speakerów: {num_speakers}")
    else:
        labels = _auto_cluster(embeddings, max_speakers=DEFAULT_MAX_SPEAKERS)

    # >>> KLUCZOWE: Post-processing dla szybkich dialogów <<<
    labels = _postprocess_diarization(labels, embeddings, timestamps, threshold_ms=1500)

    speakers = _labels_to_names(labels)
    state.add_log(f"  ✅ Lokalna diarization: {len(set(speakers))} speakerów")

    return speakers


# ============================================================
# MAIN FUNCTION
# ============================================================

def perform_diarization(
    video_path: str,
    timestamps: list,
    hf_token: str = None,
    num_speakers: int = None
) -> list:
    if not timestamps:
        state.add_log("  ℹ️ Brak timestampów — pomijam diarization.")
        return []

    device = _get_device()
    state.add_log(f"  🖥️ Diarization device: {device}")

    os.makedirs(DIAR_AUDIO_DIR, exist_ok=True)

    try:
        _extract_audio(video_path, DIAR_WAV_PATH)
    except Exception as e:
        state.add_log(f"  ❌ Nie udało się wyodrębnić audio: {e}")
        return ["Unknown"] * len(timestamps)

    # PATH 1: PYANNOTE
    if hf_token and hf_token.strip():
        pipeline = None
        try:
            pipeline, diarization = _diarize_pyannote(
                DIAR_WAV_PATH, hf_token.strip(), device, num_speakers
            )
            speakers = _map_diarization_to_timestamps(diarization, timestamps)

            del diarization
            if pipeline is not None:
                del pipeline
            _cleanup_gpu()

            state.add_log("  ✅ Pyannote: diarization zakończona pomyślnie.")
            return speakers

        except Exception as e:
            state.add_log(f"  ⚠️ Pyannote failed: {e}")
            state.add_log("  🔄 Przełączam na lokalny fallback SpeechBrain...")

            if pipeline is not None:
                try:
                    del pipeline
                except Exception:
                    pass
            _cleanup_gpu()
    else:
        state.add_log("  ⚠️ Brak tokenu HuggingFace.")
        state.add_log("  🔄 Używam lokalnego SpeechBrain jako fallback.")

    # PATH 2: SPEECHBRAIN ECAPA
    try:
        speakers = _diarize_speechbrain(DIAR_WAV_PATH, timestamps, device, num_speakers)

        if len(speakers) != len(timestamps):
            raise RuntimeError(
                f"Fallback zwrócił nieprawidłową liczbę etykiet: "
                f"{len(speakers)} zamiast {len(timestamps)}."
            )

        return speakers

    except Exception as e:
        state.add_log(f"  ❌ Lokalna diarization failed: {e}")
        _cleanup_gpu()
        return ["Unknown" for _ in timestamps]
