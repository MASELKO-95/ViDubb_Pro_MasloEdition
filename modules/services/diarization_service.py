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

ECAPA_SAMPLE_RATE = 16000


DEFAULT_MAX_SPEAKERS = 20

# ECAPA is unreliable on subtitle fragments lasting only a few hundred ms.
# Give short utterances additional local context before embedding them.
MIN_SEGMENT_MS = 1200
EMBEDDING_CONTEXT_MS = 450
MAX_ECAPA_SEGMENT_MS = 12_000


# ============================================================
# UTILITY
# ============================================================

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_speechbrain_device(preferred_device: str) -> str:
    """Prefer CUDA whenever it is available; callers handle an OOM fallback."""
    return "cuda" if preferred_device == "cuda" and torch.cuda.is_available() else "cpu"


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
    classifier = None
    tensor = None
    embedding = None
    embeddings = []
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir=ECAPA_CACHE_DIR
        )

        audio = _prepare_audio(wav_path)
        state.add_log(f"  📊 Generating embeddings for {len(timestamps)} segments...")

        for index, (start_ms, end_ms) in enumerate(timestamps):
            start_ms = max(0, int(start_ms))
            end_ms = max(start_ms, int(end_ms))

            if end_ms - start_ms < MIN_SEGMENT_MS:
                # Expand into nearby audio, but never cross the midpoint of a
                # neighbouring subtitle. This avoids borrowing another actor.
                left_limit = 0
                right_limit = len(audio)
                if index > 0:
                    previous_end = int(timestamps[index - 1][1])
                    left_limit = max(0, (previous_end + start_ms) // 2)
                if index + 1 < len(timestamps):
                    next_start = int(timestamps[index + 1][0])
                    right_limit = min(len(audio), (end_ms + next_start) // 2)

                start_ms = max(left_limit, start_ms - EMBEDDING_CONTEXT_MS)
                end_ms = min(right_limit, end_ms + EMBEDDING_CONTEXT_MS)

            # A malformed or unusually long subtitle event must not create a
            # massive ECAPA tensor. A centered 12-second voice sample is enough.
            if end_ms - start_ms > MAX_ECAPA_SEGMENT_MS:
                center = (start_ms + end_ms) // 2
                half_window = MAX_ECAPA_SEGMENT_MS // 2
                start_ms = max(0, center - half_window)
                end_ms = min(len(audio), start_ms + MAX_ECAPA_SEGMENT_MS)

            segment = audio[start_ms:end_ms]
            if len(segment) <= 0:
                embeddings.append(np.zeros(192, dtype=np.float32))
                continue

            samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
            samples /= 32768.0
            tensor = torch.from_numpy(samples).unsqueeze(0).to(device)
            with torch.inference_mode():
                embedding = classifier.encode_batch(tensor)
            embeddings.append(
                embedding.squeeze().detach().cpu().numpy().astype(np.float32)
            )
            del tensor, embedding
            tensor = embedding = None

            if (index + 1) % 20 == 0 or index == len(timestamps) - 1:
                state.add_log(f"    ✓ {index + 1}/{len(timestamps)} embeddings")
    finally:
        del tensor, embedding, classifier
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
            # Average linkage tolerates emotion/pitch changes better than
            # complete linkage, which tended to split one actor into clusters.
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average"
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
# POST-PROCESSING FOR RAPID DIALOGUE CHANGES
# ============================================================

def _postprocess_diarization(
    labels: list,
    embeddings: np.ndarray,
    timestamps: list,
    threshold_ms: int = 1500
) -> list:
    """Apply conservative corrections without inventing new speakers.

    ECAPA similarity between two consecutive subtitle-sized samples can fall
    sharply because of emotion, shouting, background music or a very short
    utterance. Creating a new label from that local difference caused one
    actor to become many fake speakers. The global clustering step above is
    responsible for deciding how many people exist; post-processing must
    preserve that decision.
    """
    if len(labels) <= 2:
        return list(labels)

    improved_labels = list(labels)

    # Remove only an isolated one-line flip between two equal neighbours.
    # This reduces obvious jitter and never increases the number of speakers.
    for index in range(1, len(labels) - 1):
        previous_label = labels[index - 1]
        current_label = labels[index]
        next_label = labels[index + 1]
        previous_end = timestamps[index - 1][1]
        next_start = timestamps[index + 1][0]

        if (
            previous_label == next_label
            and current_label != previous_label
            and next_start - previous_end <= threshold_ms * 2
        ):
            improved_labels[index] = previous_label

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

    if num_speakers is not None and num_speakers > 0:
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
                linkage="average",
            )
            labels = clusterer.fit_predict(normalize(embeddings)).tolist()
            state.add_log(f"  🎯 Użyto wymuszonej liczby speakerów: {num_speakers}")
    else:
        labels = _auto_cluster(embeddings, max_speakers=DEFAULT_MAX_SPEAKERS)

    # Smooth isolated assignment jitter without creating extra speakers.
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


    speechbrain_device = _get_speechbrain_device(device)
    attempts = [speechbrain_device]
    if speechbrain_device == "cuda":
        attempts.append("cpu")

    for attempt_device in attempts:
        try:
            state.add_log(f"  🖥️ ECAPA device: {attempt_device}.")
            speakers = _diarize_speechbrain(
                DIAR_WAV_PATH, timestamps, attempt_device, num_speakers
            )
            if len(speakers) != len(timestamps):
                raise RuntimeError(
                    f"Fallback returned {len(speakers)} labels for "
                    f"{len(timestamps)} timestamps."
                )
            return speakers
        except Exception as exc:
            _cleanup_gpu()
            if attempt_device == "cuda":
                state.add_log(
                    f"  ⚠️ ECAPA CUDA failed: {exc}. Retrying on CPU."
                )
                continue
            state.add_log(f"  ❌ Local diarization failed: {exc}")

    return ["Unknown" for _ in timestamps]
