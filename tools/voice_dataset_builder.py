#!/usr/bin/env python3
"""Build a reviewed, multi-clip voice dataset for ViDubb."""
from __future__ import annotations
import argparse, csv, json, math, os, re, subprocess, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mkv", ".mov", ".webm", ".avi"}

def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "voice").strip())
    return value.strip("._-")[:100] or "voice"

@dataclass
class ClipRecord:
    clip_id: str; file: str; source: str; start: float; end: float; duration: float
    text: str; language: str; whisper_confidence: float; speaker_similarity: float | None
    status: str; reason: str
    emotion: str = "neutral"; energy_dbfs: float = -90.0; speaker_consistency: float | None = None

@dataclass
class BuildConfig:
    name: str; inputs: list[Path]; output_root: Path = PROJECT_ROOT / "voice_datasets"
    language: str = "auto"; whisper_model: str = "turbo"; reference: Path | None = None
    existing_voice_id: str = ""; min_duration: float = 1.0; max_duration: float = 12.0
    accept_similarity: float = 0.82; review_similarity: float = 0.72
    min_whisper_confidence: float = 0.55; normalize_dbfs: float = -20.0
    clean_vocals: bool = False
    min_short_duration: float = 0.35
    edge_padding_ms: int = 220
    validate_single_speaker: bool = True

def _atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    if not a.size or a.shape != b.shape: return 0.0
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-9 else 0.0

def _word_confidence(segment) -> float:
    probs = [w.probability for w in getattr(segment, "words", []) if w.probability is not None]
    return float(sum(probs) / len(probs)) if probs else 0.75

def _emotion_annotation(audio) -> tuple[str, float]:
    """Cheap, deterministic prosody label; useful for balancing, not diagnosis."""
    import audioop
    energy = float(audio.dBFS) if len(audio) and math.isfinite(audio.dBFS) else -90.0
    mono = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    raw = mono.raw_data
    crossings = audioop.cross(raw, 2) / max(0.001, len(mono) / 1000.0) if raw else 0.0
    if energy > -15.5 and crossings > 1700:
        label = "intense"
    elif energy < -30.0:
        label = "soft"
    elif crossings > 2200:
        label = "bright"
    elif crossings < 650 and energy > -23.0:
        label = "low_calm"
    else:
        label = "neutral"
    return label, round(energy, 2)

def _speaker_consistency(audio, encoder: "SpeakerEncoder", references) -> float | None:
    """Catch clips whose first and second half appear to contain different voices."""
    if len(audio) < 2400:
        return None
    midpoint = len(audio) // 2
    left = encoder.encode(audio[:midpoint])
    right = encoder.encode(audio[midpoint:])
    internal = _cosine(left, right)
    if references:
        left_ref = max(_cosine(left, ref) for ref in references)
        right_ref = max(_cosine(right, ref) for ref in references)
        return float(min(internal, left_ref, right_ref))
    return float(internal)

def _prepare_clip(audio, target_dbfs: float):
    from pydub.effects import high_pass_filter, low_pass_filter, normalize
    clip = audio.set_channels(1).set_frame_rate(24000).set_sample_width(2)
    clip = high_pass_filter(clip, 70); clip = low_pass_filter(clip, 11000)
    if math.isfinite(clip.dBFS) and clip.dBFS < target_dbfs:
        clip = clip.apply_gain(min(12.0, target_dbfs - clip.dBFS))
    return normalize(clip, headroom=1.0)

def _separate_vocals(audio_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(output_dir), str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs nie powiódł się:\n{result.stderr}")
    model_dir = output_dir / "htdemucs" / audio_path.stem
    vocal_path = model_dir / "vocals.wav"
    return vocal_path if vocal_path.exists() else next(output_dir.rglob("vocals.wav"), audio_path)

class SpeakerEncoder:
    def __init__(self):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(PROJECT_ROOT / "audio" / "ecapa_cache"),
            run_opts={"device": self.device},
        )
    def encode(self, audio):
        import numpy as np
        mono = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        samples = np.asarray(mono.get_array_of_samples(), dtype=np.float32) / 32768.0
        tensor = self.torch.from_numpy(samples).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            result = self.model.encode_batch(tensor).squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(result)
        return result / norm if norm > 1e-9 else result

def _reference_embeddings(config: BuildConfig, encoder: SpeakerEncoder, log: Callable[[str], None]):
    from pydub import AudioSegment
    refs = []
    if config.reference:
        audio = AudioSegment.from_file(config.reference)
        for pos in range(0, len(audio), 8000):
            chunk = audio[pos:pos + 8000]
            if len(chunk) >= 1500: refs.append(encoder.encode(chunk))
        log(f"Wzorzec: {len(refs)} odcisków z pliku referencyjnego.")
    if config.existing_voice_id:
        from modules.services.voice_db_service import load_voice_db
        info = load_voice_db().get(config.existing_voice_id, {})
        raw = info.get("embedding_samples") or ([info.get("centroid")] if info.get("centroid") else [])
        refs.extend(raw)
        log(f"Wzorzec uzupełniony z profilu: {config.existing_voice_id}.")
    return refs

def _decide(duration: float, confidence: float, similarity: float | None, cfg: BuildConfig) -> tuple[str, str]:
    if duration < cfg.min_short_duration: return "rejected", "too_short"
    if duration < cfg.min_duration:
        if similarity is not None and similarity < cfg.review_similarity:
            return "rejected", "short_different_speaker"
        return "short", "short_dialogue_verified" if similarity is not None else "short_dialogue_no_reference"
    if duration > cfg.max_duration: return "review", "too_long"
    if confidence < cfg.min_whisper_confidence: return "review", "low_transcription_confidence"
    if similarity is None: return "review", "no_speaker_reference"
    if similarity >= cfg.accept_similarity: return "accepted", "speaker_match"
    if similarity >= cfg.review_similarity: return "review", "uncertain_speaker"
    return "rejected", "different_speaker"

def build_dataset(config: BuildConfig, log: Callable[[str], None] = print) -> Path:
    from pydub import AudioSegment
    from modules.services.whisper_service import WhisperBatchSession

    dataset = config.output_root / safe_name(config.name)
    for subdir in ("accepted", "short", "review", "rejected"): (dataset / subdir).mkdir(parents=True, exist_ok=True)

    inputs = []
    for item in config.inputs:
        if item.is_dir(): inputs.extend(p for p in sorted(item.rglob("*")) if p.suffix.lower() in SUPPORTED)
        elif item.suffix.lower() in SUPPORTED: inputs.append(item)
    if not inputs: raise ValueError("Nie znaleziono obsługiwanych plików audio/wideo.")

    log("Ładowanie modelu ECAPA...")
    encoder = SpeakerEncoder()
    references = _reference_embeddings(config, encoder, log)
    records: list[ClipRecord] = []
    manifest = dataset / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try: records.append(ClipRecord(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError): log("Uwaga: pominięto uszkodzony wpis.")
    number = len(records) + 1

    with WhisperBatchSession(config.whisper_model, config.language) as whisper:
      for source_index, source in enumerate(inputs, 1):
        log(f"[{source_index}/{len(inputs)}] Przetwarzanie: {source.name}")
        current_source = source

        if config.clean_vocals:
            log(f"  -> Demucs: Wyciągam wokal z {source.name} (to może potrwać)...")
            clean_dir = config.output_root / ".temp_clean"
            current_source = _separate_vocals(source, clean_dir)
            log(f"  -> Demucs: Gotowe. Używam: {current_source.name}")

        log(f"  -> Whisper: Transkrypcja...")
        segments, detected_language = whisper.transcribe(str(current_source))
        source_audio = AudioSegment.from_file(current_source)

        for seg_index, segment in enumerate(segments):
            # Whisper timestamps often sit inside the first/last phoneme. A symmetric
            # margin prevents clipped consonants and breaths without joining long gaps.
            start_ms = max(0, int(segment.start * 1000) - config.edge_padding_ms)
            end_ms = min(len(source_audio), int(segment.end * 1000) + config.edge_padding_ms)
            raw_clip = source_audio[start_ms:end_ms]
            emotion, energy_dbfs = _emotion_annotation(raw_clip)
            clip = _prepare_clip(raw_clip, config.normalize_dbfs)
            duration = len(clip) / 1000.0
            confidence = _word_confidence(segment)
            # Keep the real 0.35-1.0 s clip, but identify its speaker using a
            # longer, neighbour-safe context. The context is never exported.
            verification_clip = clip
            if duration < 1.0:
                left_limit = 0 if seg_index == 0 else int((segments[seg_index - 1].end + segment.start) * 500)
                right_limit = len(source_audio) if seg_index + 1 == len(segments) else int((segment.end + segments[seg_index + 1].start) * 500)
                context_start = max(left_limit, start_ms - 500)
                context_end = min(right_limit, end_ms + 500)
                verification_clip = _prepare_clip(source_audio[context_start:context_end], config.normalize_dbfs)
            embedding = encoder.encode(verification_clip) if len(verification_clip) >= 700 else None
            similarity = max((_cosine(embedding, ref) for ref in references), default=None) if embedding is not None else None
            status, reason = _decide(duration, confidence, similarity, config)
            consistency = None
            if config.validate_single_speaker and embedding is not None and duration >= 2.4:
                consistency = _speaker_consistency(clip, encoder, references)
                threshold = max(0.58, config.review_similarity - 0.08)
                if consistency is not None and consistency < threshold:
                    status, reason = "rejected", "mixed_or_inconsistent_speaker"
            clip_id = f"{safe_name(config.name)}_{number:06d}"
            relative = Path(status) / f"{clip_id}.wav"
            clip.export(dataset / relative, format="wav")
            records.append(ClipRecord(clip_id, relative.as_posix(), source.name, round(segment.start, 3), round(segment.end, 3),
                round(duration, 3), segment.text.strip(), str(detected_language), round(confidence, 4),
                round(similarity, 4) if similarity is not None else None, status, reason,
                emotion, energy_dbfs, round(consistency, 4) if consistency is not None else None))
            number += 1

    manifest.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    with (dataset / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else list(ClipRecord.__annotations__))
        writer.writeheader(); writer.writerows(asdict(r) for r in records)

    counts = {status: sum(r.status == status for r in records) for status in ("accepted", "short", "review", "rejected")}
    _atomic_json(dataset / "dataset.json", {
        "schema_version": 2, "name": config.name, "created_at": time.time(), "counts": counts,
        "inputs": [str(p) for p in inputs], "language": config.language,
        "whisper_model": config.whisper_model, "reference": str(config.reference or ""),
        "existing_voice_id": config.existing_voice_id, "clean_vocals": config.clean_vocals,
        "edge_padding_ms": config.edge_padding_ms,
        "emotion_counts": {e: sum(r.emotion == e for r in records) for e in sorted({r.emotion for r in records})},
    })
    (dataset / "README.txt").write_text("accepted/ = pewne próbki treningowe\nshort/ = krótkie dialogi zachowane, lecz nieuczące samodzielnie profilu\nreview/ = wymagają odsłuchu\nrejected/ = odrzucone\n", encoding="utf-8")
    log(f"Gotowe: accepted={counts['accepted']}, short={counts['short']}, review={counts['review']}, rejected={counts['rejected']}")
    return dataset

def register_in_vidubb(dataset: Path, name: str, voice_id: str = "", log: Callable[[str], None] = print) -> str:
    """Register accepted clips and retain the link to the complete training dataset."""
    from modules.services.voice_db_service import import_voice_from_folder, load_voice_db, save_voice_db
    accepted = dataset / "accepted"
    if not accepted.exists() or not any(accepted.glob("*.wav")):
        raise ValueError("Brak zaakceptowanych klipów do rejestracji.")
    db = load_voice_db()
    dataset_value = str(dataset.relative_to(PROJECT_ROOT) if dataset.is_relative_to(PROJECT_ROOT) else dataset)
    profile_id = voice_id
    if profile_id and profile_id not in db:
        raise ValueError(f"Nie istnieje profil Voice DB: {profile_id}")
    if not profile_id:
        profile_id = next((key for key, info in db.items() if isinstance(info, dict) and info.get("dataset_path") == dataset_value), "")
    if not profile_id:
        profile_id = import_voice_from_folder(str(accepted), name, dataset.name, "Voice dataset built by ViDubb")
    if not profile_id:
        raise RuntimeError("Nie udało się utworzyć profilu Voice DB.")
    db = load_voice_db()
    db[profile_id]["dataset_path"] = dataset_value
    db[profile_id]["dataset_manifest"] = str(dataset / "manifest.jsonl")
    db[profile_id]["dataset_clip_count"] = len(list(accepted.glob("*.wav")))
    db[profile_id]["updated_at"] = time.time()
    save_voice_db(db)
    log(f"Profil ViDubb: {profile_id}")
    return profile_id

def launch_gui() -> None:
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    root = tk.Tk()
    root.title("ViDubb — kreator datasetu głosu")
    root.geometry("820x620")
    values = {
        "name": tk.StringVar(), "inputs": tk.StringVar(), "reference": tk.StringVar(),
        "language": tk.StringVar(value="auto"), "model": tk.StringVar(value="turbo"),
        "register": tk.BooleanVar(value=True), "clean": tk.BooleanVar(value=False),
    }
    def row(label, variable, commands=()):
        frame = ttk.Frame(root); frame.pack(fill="x", padx=14, pady=5)
        ttk.Label(frame, text=label, width=22).pack(side="left")
        ttk.Entry(frame, textvariable=variable).pack(side="left", fill="x", expand=True)
        for title, command in commands:
            ttk.Button(frame, text=title, command=command).pack(side="left", padx=(6, 0))
    def select_files():
        paths = filedialog.askopenfilenames(filetypes=[("Audio / wideo", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.mp4 *.mkv *.mov *.webm *.avi")])
        if paths: values["inputs"].set(os.pathsep.join(paths))
    def select_folder():
        path = filedialog.askdirectory()
        if path: values["inputs"].set(path)
    def select_ref():
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac")])
        if path: values["reference"].set(path)
    ttk.Label(root, text="Kontrolowany dataset jednego głosu", font=("TkDefaultFont", 15, "bold")).pack(pady=(14, 8))
    row("Nazwa głosu", values["name"])
    row("Pliki / katalog", values["inputs"], (("Pliki", select_files), ("Katalog", select_folder)))
    row("Pewna próbka", values["reference"], (("Wybierz", select_ref),))
    row("Język", values["language"]); row("Model Whisper", values["model"])
    options = ttk.Frame(root); options.pack(fill="x", padx=190, pady=4)
    ttk.Checkbutton(options, text="Zarejestruj accepted/ w Voice DB", variable=values["register"]).pack(anchor="w")
    ttk.Checkbutton(options, text="Usuń muzykę/tło przez Demucs (wolniej)", variable=values["clean"]).pack(anchor="w")
    output = tk.Text(root, height=18, state="disabled"); output.pack(fill="both", expand=True, padx=14, pady=10)
    def log(message):
        def append():
            output.configure(state="normal"); output.insert("end", str(message) + "\n"); output.see("end"); output.configure(state="disabled")
        root.after(0, append)
    def start():
        if not values["name"].get().strip() or not values["inputs"].get().strip():
            messagebox.showwarning("Brak danych", "Podaj nazwę i wybierz pliki albo katalog."); return
        button.configure(state="disabled")
        def worker():
            try:
                cfg = BuildConfig(
                    name=values["name"].get().strip(),
                    inputs=[Path(p) for p in values["inputs"].get().split(os.pathsep)],
                    reference=Path(values["reference"].get()) if values["reference"].get() else None,
                    language=values["language"].get().strip() or "auto",
                    whisper_model=values["model"].get().strip() or "turbo",
                    clean_vocals=values["clean"].get(),
                )
                result = build_dataset(cfg, log)
                profile = register_in_vidubb(result, cfg.name, log=log) if values["register"].get() else ""
                suffix = f"\nProfil: {profile}" if profile else ""
                root.after(0, lambda: messagebox.showinfo("Gotowe", f"Dataset: {result}{suffix}"))
            except Exception as exc:
                root.after(0, lambda: messagebox.showerror("Błąd", f"{type(exc).__name__}: {exc}"))
            finally:
                root.after(0, lambda: button.configure(state="normal"))
        threading.Thread(target=worker, daemon=True, name="dataset-builder").start()
    button = ttk.Button(root, text="Utwórz / rozbuduj dataset", command=start); button.pack(pady=(0, 14))
    root.mainloop()

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tworzy dataset głosu dla ViDubb.")
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--name"); parser.add_argument("--reference", type=Path)
    parser.add_argument("--voice-id", default=""); parser.add_argument("--language", default="auto")
    parser.add_argument("--model", default="turbo"); parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "voice_datasets")
    parser.add_argument("--gui", action="store_true"); parser.add_argument("--register", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Użyj Demucs do usunięcia tła/muzyki")
    parser.add_argument("--edge-padding-ms", type=int, default=220, help="Margines przed/po wypowiedzi (domyślnie 220 ms)")
    parser.add_argument("--no-speaker-validation", action="store_true", help="Wyłącz kontrolę dwóch głosów w jednym klipie")
    args = parser.parse_args(argv)

    if args.gui or (not args.inputs and not args.name):
        launch_gui(); return 0

    if not args.name or not args.inputs:
        parser.error("wymagane są --name oraz co najmniej jedno nagranie")

    cfg = BuildConfig(args.name, args.inputs, args.output, args.language, args.model, args.reference, args.voice_id,
                      clean_vocals=args.clean, edge_padding_ms=max(0, min(1000, args.edge_padding_ms)),
                      validate_single_speaker=not args.no_speaker_validation)
    path = build_dataset(cfg)
    if args.register:
        register_in_vidubb(path, args.name, args.voice_id)
    print(path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
