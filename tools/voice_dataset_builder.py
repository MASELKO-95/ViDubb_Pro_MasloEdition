#!/usr/bin/env python3
"""Build a reviewed, multi-clip voice dataset for ViDubb.

The module has a small Tk GUI and a CLI.  Heavy models are loaded only when a
build starts, so ``--help`` and importing this module remain cheap.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
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
    clip_id: str
    file: str
    source: str
    start: float
    end: float
    duration: float
    text: str
    language: str
    whisper_confidence: float
    speaker_similarity: float | None
    status: str
    reason: str


@dataclass
class BuildConfig:
    name: str
    inputs: list[Path]
    output_root: Path = PROJECT_ROOT / "voice_datasets"
    language: str = "auto"
    whisper_model: str = "turbo"
    reference: Path | None = None
    existing_voice_id: str = ""
    min_duration: float = 1.0
    max_duration: float = 12.0
    accept_similarity: float = 0.82
    review_similarity: float = 0.72
    min_whisper_confidence: float = 0.55
    normalize_dbfs: float = -20.0


def _atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if not a.size or a.shape != b.shape:
        return 0.0
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-9 else 0.0


def _word_confidence(segment) -> float:
    probabilities = [w.probability for w in getattr(segment, "words", []) if w.probability is not None]
    return float(sum(probabilities) / len(probabilities)) if probabilities else 0.75


def _prepare_clip(audio, target_dbfs: float):
    """Speech-friendly cleanup without destroying the actor's timbre."""
    from pydub.effects import high_pass_filter, low_pass_filter, normalize

    clip = audio.set_channels(1).set_frame_rate(24000).set_sample_width(2)
    clip = high_pass_filter(clip, 70)
    clip = low_pass_filter(clip, 11000)
    if math.isfinite(clip.dBFS) and clip.dBFS < target_dbfs:
        clip = clip.apply_gain(min(12.0, target_dbfs - clip.dBFS))
    return normalize(clip, headroom=1.0)


class SpeakerEncoder:
    """One reusable ECAPA model (loading it for every clip is prohibitively slow)."""

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
            if len(chunk) >= 1500:
                refs.append(encoder.encode(chunk))
        log(f"Wzorzec: {len(refs)} odcisków z pliku referencyjnego.")

    if config.existing_voice_id:
        from modules.services.voice_db_service import load_voice_db
        info = load_voice_db().get(config.existing_voice_id, {})
        raw = info.get("embedding_samples") or ([info.get("centroid")] if info.get("centroid") else [])
        refs.extend(raw)
        log(f"Wzorzec uzupełniony z profilu: {config.existing_voice_id}.")
    return refs


def _decide(duration: float, confidence: float, similarity: float | None, cfg: BuildConfig) -> tuple[str, str]:
    if duration < cfg.min_duration:
        return "rejected", "too_short"
    if duration > cfg.max_duration:
        return "review", "too_long"
    if confidence < cfg.min_whisper_confidence:
        return "review", "low_transcription_confidence"
    if similarity is None:
        return "review", "no_speaker_reference"
    if similarity >= cfg.accept_similarity:
        return "accepted", "speaker_match"
    if similarity >= cfg.review_similarity:
        return "review", "uncertain_speaker"
    return "rejected", "different_speaker"


def build_dataset(config: BuildConfig, log: Callable[[str], None] = print) -> Path:
    from pydub import AudioSegment
    from modules.services.whisper_service import transcribe_video

    dataset = config.output_root / safe_name(config.name)
    for subdir in ("accepted", "review", "rejected"):
        (dataset / subdir).mkdir(parents=True, exist_ok=True)

    inputs = []
    for item in config.inputs:
        if item.is_dir():
            inputs.extend(p for p in sorted(item.rglob("*")) if p.suffix.lower() in SUPPORTED)
        elif item.suffix.lower() in SUPPORTED:
            inputs.append(item)
    if not inputs:
        raise ValueError("Nie znaleziono obsługiwanych plików audio/wideo.")

    log("Ładowanie modelu ECAPA...")
    encoder = SpeakerEncoder()
    references = _reference_embeddings(config, encoder, log)
    records: list[ClipRecord] = []
    manifest = dataset / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                records.append(ClipRecord(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError):
                log("Uwaga: pominięto uszkodzony wpis starego manifestu.")
    number = len(records) + 1

    for source_index, source in enumerate(inputs, 1):
        log(f"[{source_index}/{len(inputs)}] Whisper: {source.name}")
        segments, detected_language = transcribe_video(str(source), config.whisper_model, config.language)
        source_audio = AudioSegment.from_file(source)
        for segment in segments:
            start_ms = max(0, int(segment.start * 1000) - 80)
            end_ms = min(len(source_audio), int(segment.end * 1000) + 100)
            clip = _prepare_clip(source_audio[start_ms:end_ms], config.normalize_dbfs)
            duration = len(clip) / 1000.0
            confidence = _word_confidence(segment)
            embedding = encoder.encode(clip) if duration >= 1.0 else None
            similarity = max((_cosine(embedding, ref) for ref in references), default=None) if embedding is not None else None
            status, reason = _decide(duration, confidence, similarity, config)
            clip_id = f"{safe_name(config.name)}_{number:06d}"
            relative = Path(status) / f"{clip_id}.wav"
            clip.export(dataset / relative, format="wav")
            records.append(ClipRecord(
                clip_id, relative.as_posix(), source.name, round(segment.start, 3), round(segment.end, 3),
                round(duration, 3), segment.text.strip(), str(detected_language), round(confidence, 4),
                round(similarity, 4) if similarity is not None else None, status, reason,
            ))
            number += 1

    manifest.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    with (dataset / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else list(ClipRecord.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)
    counts = {status: sum(r.status == status for r in records) for status in ("accepted", "review", "rejected")}
    _atomic_json(dataset / "dataset.json", {
        "schema_version": 1, "name": config.name, "created_at": time.time(), "counts": counts,
        "inputs": [str(p) for p in inputs], "language": config.language,
        "whisper_model": config.whisper_model, "reference": str(config.reference or ""),
        "existing_voice_id": config.existing_voice_id,
    })
    (dataset / "README.txt").write_text(
        "accepted/ = pewne próbki\nreview/ = wymagają odsłuchu\nrejected/ = odrzucone automatycznie\n"
        "Po ręcznej zmianie decyzji przenieś WAV i zmień pole status w manifest.jsonl.\n",
        encoding="utf-8",
    )
    log(f"Gotowe: accepted={counts['accepted']}, review={counts['review']}, rejected={counts['rejected']}")
    return dataset


def register_in_vidubb(dataset: Path, name: str, voice_id: str = "", log: Callable[[str], None] = print) -> str:
    """Create a ViDubb profile from accepted clips and retain dataset metadata."""
    from modules.services.voice_db_service import import_voice_from_folder, load_voice_db, save_voice_db

    accepted = dataset / "accepted"
    if not accepted.exists() or not any(accepted.glob("*.wav")):
        raise ValueError("Brak zaakceptowanych klipów do rejestracji.")
    db = load_voice_db()
    dataset_value = str(dataset.relative_to(PROJECT_ROOT) if dataset.is_relative_to(PROJECT_ROOT) else dataset)
    if not voice_id:
        voice_id = next(
            (key for key, info in db.items() if isinstance(info, dict) and info.get("dataset_path") == dataset_value),
            "",
        )
    if voice_id:
        if voice_id not in db:
            raise ValueError(f"Nie istnieje profil Voice DB: {voice_id}")
        profile_id = voice_id
    else:
        profile_id = import_voice_from_folder(str(accepted), name, dataset.name, "Voice dataset built by ViDubb")
        if not profile_id:
            raise RuntimeError("Nie udało się utworzyć profilu Voice DB.")
        db = load_voice_db()
    db[profile_id]["dataset_path"] = dataset_value
    db[profile_id]["dataset_manifest"] = str((dataset / "manifest.jsonl").relative_to(PROJECT_ROOT) if dataset.is_relative_to(PROJECT_ROOT) else dataset / "manifest.jsonl")
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
    root.title("ViDubb — kreator bazy głosu")
    root.geometry("780x580")
    values = {"name": tk.StringVar(), "inputs": tk.StringVar(), "reference": tk.StringVar(), "language": tk.StringVar(value="auto"), "model": tk.StringVar(value="turbo"), "register": tk.BooleanVar(value=True)}

    def row(label, variable, browse=None):
        frame = ttk.Frame(root); frame.pack(fill="x", padx=14, pady=6)
        ttk.Label(frame, text=label, width=20).pack(side="left")
        ttk.Entry(frame, textvariable=variable).pack(side="left", fill="x", expand=True)
        if browse: ttk.Button(frame, text="Wybierz", command=browse).pack(side="left", padx=(7, 0))

    def select_inputs():
        files = filedialog.askopenfilenames(filetypes=[("Audio / wideo", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.mp4 *.mkv *.mov *.webm *.avi")])
        if files: values["inputs"].set(os.pathsep.join(files))
    def select_ref():
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac")])
        if path: values["reference"].set(path)

    ttk.Label(root, text="Czysta baza dialogów", font=("TkDefaultFont", 16, "bold")).pack(pady=(14, 8))
    row("Nazwa osoby", values["name"])
    row("Nagrania źródłowe", values["inputs"], select_inputs)
    row("Pewna próbka wzorcowa", values["reference"], select_ref)
    row("Język (auto/pl/ja...)", values["language"])
    row("Model Whisper", values["model"])
    ttk.Checkbutton(root, text="Po zbudowaniu dodaj accepted/ do Voice DB", variable=values["register"]).pack(anchor="w", padx=178, pady=4)
    output = tk.Text(root, height=16, state="disabled"); output.pack(fill="both", expand=True, padx=14, pady=10)
    def log(text):
        root.after(0, lambda: (output.configure(state="normal"), output.insert("end", str(text) + "\n"), output.see("end"), output.configure(state="disabled")))
    def start():
        if not values["name"].get().strip() or not values["inputs"].get().strip():
            messagebox.showwarning("Brak danych", "Podaj nazwę osoby i wybierz nagrania."); return
        button.configure(state="disabled")
        def worker():
            try:
                cfg = BuildConfig(name=values["name"].get().strip(), inputs=[Path(p) for p in values["inputs"].get().split(os.pathsep)], reference=Path(values["reference"].get()) if values["reference"].get() else None, language=values["language"].get(), whisper_model=values["model"].get())
                result = build_dataset(cfg, log)
                profile = register_in_vidubb(result, cfg.name, log=log) if values["register"].get() else ""
                suffix = f"\n\nProfil ViDubb: {profile}" if profile else ""
                root.after(0, lambda: messagebox.showinfo("Gotowe", f"Baza zapisana w:\n{result}{suffix}"))
            except Exception as exc: root.after(0, lambda: messagebox.showerror("Błąd", f"{type(exc).__name__}: {exc}"))
            finally: root.after(0, lambda: button.configure(state="normal"))
        threading.Thread(target=worker, daemon=True).start()
    button = ttk.Button(root, text="Utwórz / rozbuduj bazę", command=start); button.pack(pady=(0, 14))
    root.mainloop()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tworzy czysty, pocięty dataset jednego głosu dla ViDubb.")
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--voice-id", default="", help="Istniejący profil Voice DB jako wzorzec")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "voice_datasets")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--register", action="store_true", help="Zarejestruj accepted/ jako profil w Voice DB")
    args = parser.parse_args(argv)
    if args.gui or (not args.inputs and not args.name):
        launch_gui(); return 0
    if not args.name or not args.inputs:
        parser.error("wymagane są --name oraz co najmniej jedno nagranie")
    path = build_dataset(BuildConfig(args.name, args.inputs, args.output, args.language, args.model, args.reference, args.voice_id))
    if args.register:
        register_in_vidubb(path, args.name, args.voice_id)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
