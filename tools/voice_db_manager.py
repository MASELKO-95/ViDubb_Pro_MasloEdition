from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
_candidates = [THIS_FILE.parent.parent, THIS_FILE.parent, Path.cwd()]
PROJECT_ROOT: Path | None = None
for candidate in _candidates:
    if (candidate / "modules" / "services" / "voice_db_service.py").exists():
        PROJECT_ROOT = candidate.resolve()
        break

if PROJECT_ROOT is None:
    print(
        "❌ Nie znaleziono katalogu projektu ViDubb.\n"
        "Umieść ten plik np. w:\n"
        "  <ViDubb>/tools/voice_db_manager.py\n"
        "i uruchom ponownie."
    )
    sys.exit(1)

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except ImportError:
    print(
        "❌ Brak Tkinter.\n"
        "Na Ubuntu/Debian zainstaluj:\n"
        "  sudo apt install python3-tk"
    )
    sys.exit(1)

try:
    from modules.services.voice_db_service import (
        delete_voice_profile,
        import_voice_from_folder,
        load_voice_db,
        migrate_voice_db,
        save_voice_db,
        update_voice_metadata,
    )
except Exception as exc:
    print("❌ Nie udało się zaimportować voice_db_service.py:")
    print(exc)
    traceback.print_exc()
    sys.exit(1)

VOICE_DB_DIR = PROJECT_ROOT / "voice_db"
SPEAKERS_AUDIO_DIR = PROJECT_ROOT / "speakers_audio"
BACKUP_DIR = VOICE_DB_DIR / "manager_backups"
SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
SUPPORTED_VIDEO = {".mp4", ".webm", ".mkv", ".mov", ".avi"}


def safe_filename(value: str, fallback: str = "voice") -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(ch if ch in allowed else "_" for ch in str(value or ""))
    cleaned = cleaned.strip("._-")
    return cleaned[:100] or fallback


def visibility_of(info: dict) -> str:
    value = str(info.get("visibility", "private")).strip().lower()
    return "public" if value == "public" else "private"


def profile_reference_path(voice_id: str, info: dict) -> Path | None:
    candidates = []
    for key in ("preview_path", "wav_path"):
        raw = str(info.get(key, "") or "").strip()
        if raw:
            p = Path(raw)
            candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)

    candidates.extend([
        VOICE_DB_DIR / f"{safe_filename(voice_id)}_preview.wav",
        VOICE_DB_DIR / f"{safe_filename(voice_id)}.wav",
        SPEAKERS_AUDIO_DIR / f"{safe_filename(voice_id)}.wav",
    ])

    for p in candidates:
        try:
            p = p.resolve()
            if p.exists() and p.is_file() and p.stat().st_size > 1000:
                return p
        except Exception:
            pass
    return None


def open_path(path: Path) -> None:
    try:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        messagebox.showerror("Błąd", f"Nie udało się otworzyć:\n{path}\n\n{exc}")


def play_audio(path: Path) -> None:
    players = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
        ["paplay", str(path)],
        ["aplay", str(path)],
    ]
    for cmd in players:
        if shutil.which(cmd[0]):
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    open_path(path)


def backup_profile(voice_id: str, info: dict) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = BACKUP_DIR / f"{timestamp}_{safe_filename(voice_id)}"
    target.mkdir(parents=True, exist_ok=True)

    with (target / "profile.json").open("w", encoding="utf-8") as f:
        json.dump({voice_id: info}, f, indent=2, ensure_ascii=False)

    copied = set()
    for key in ("wav_path", "preview_path"):
        raw = str(info.get(key, "") or "").strip()
        if not raw:
            continue
        src = Path(raw)
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        try:
            src = src.resolve()
            if src.exists() and src.is_file():
                shutil.copy2(src, target / src.name)
                copied.add(str(src))
        except Exception:
            pass

    compat = SPEAKERS_AUDIO_DIR / f"{safe_filename(voice_id)}.wav"
    try:
        compat = compat.resolve()
        if compat.exists() and compat.is_file() and str(compat) not in copied:
            shutil.copy2(compat, target / f"speakers_audio_{compat.name}")
    except Exception:
        pass
    return target


def set_visibility(voice_id: str, visibility: str) -> bool:
    db = load_voice_db()
    if voice_id not in db or not isinstance(db[voice_id], dict):
        return False
    db[voice_id]["visibility"] = "public" if visibility == "public" else "private"
    db[voice_id]["updated_at"] = time.time()
    save_voice_db(db)
    return True


def export_public_bundle(destination: Path) -> tuple[int, list[str]]:
    db = load_voice_db()
    release_voice_db = destination / "voice_db"
    release_speakers = destination / "speakers_audio"
    release_voice_db.mkdir(parents=True, exist_ok=True)
    release_speakers.mkdir(parents=True, exist_ok=True)

    release_index: dict = {}
    warnings: list[str] = []

    for voice_id, info in db.items():
        if not isinstance(info, dict) or visibility_of(info) != "public":
            continue

        safe_id = safe_filename(voice_id)
        exported = dict(info)
        exported["visibility"] = "public"

        src_wav = None
        raw_wav = str(info.get("wav_path", "") or "").strip()
        if raw_wav:
            candidate = Path(raw_wav)
            src_wav = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if not src_wav or not src_wav.exists():
            candidate = VOICE_DB_DIR / f"{safe_id}.wav"
            if candidate.exists():
                src_wav = candidate

        if src_wav and src_wav.exists():
            dst_wav = release_voice_db / f"{safe_id}.wav"
            shutil.copy2(src_wav, dst_wav)
            exported["wav_path"] = f"voice_db/{dst_wav.name}"
        else:
            warnings.append(f"{voice_id}: brak głównego WAV")
            exported["wav_path"] = ""

        src_preview = None
        raw_preview = str(info.get("preview_path", "") or "").strip()
        if raw_preview:
            candidate = Path(raw_preview)
            src_preview = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if not src_preview or not src_preview.exists():
            candidate = VOICE_DB_DIR / f"{safe_id}_preview.wav"
            if candidate.exists():
                src_preview = candidate

        if src_preview and src_preview.exists():
            dst_preview = release_voice_db / f"{safe_id}_preview.wav"
            shutil.copy2(src_preview, dst_preview)
            exported["preview_path"] = f"voice_db/{dst_preview.name}"
        else:
            exported["preview_path"] = ""

        compat_src = SPEAKERS_AUDIO_DIR / f"{safe_id}.wav"
        if compat_src.exists():
            shutil.copy2(compat_src, release_speakers / f"{safe_id}.wav")
        elif src_wav and src_wav.exists():
            shutil.copy2(src_wav, release_speakers / f"{safe_id}.wav")

        release_index[voice_id] = exported

    index_path = release_voice_db / "index.json"
    temp_path = index_path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(release_index, f, indent=2, ensure_ascii=False)
    temp_path.replace(index_path)

    (destination / "VOICE_BUNDLE_README.txt").write_text(
        "ViDubb Voice DB release bundle\n"
        "==============================\n\n"
        "This bundle contains ONLY profiles explicitly marked PUBLIC\n"
        "in Voice DB Manager.\n\n"
        f"Profiles exported: {len(release_index)}\n",
        encoding="utf-8",
    )
    return len(release_index), warnings


class MetadataDialog(tk.Toplevel):
    def __init__(self, parent, voice_id: str, info: dict):
        super().__init__(parent)
        self.title(f"Edytuj — {voice_id}")
        self.resizable(True, True)
        self.result = None
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="Voice ID:").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        id_entry = ttk.Entry(self)
        id_entry.insert(0, voice_id)
        id_entry.state(["readonly"])
        id_entry.grid(row=0, column=1, padx=10, pady=8, sticky="ew")

        ttk.Label(self, text="Display name:").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.display_var = tk.StringVar(value=str(info.get("display_name", voice_id)))
        ttk.Entry(self, textvariable=self.display_var).grid(row=1, column=1, padx=10, pady=8, sticky="ew")

        ttk.Label(self, text="Źródło / film:").grid(row=2, column=0, padx=10, pady=8, sticky="w")
        self.source_var = tk.StringVar(value=str(info.get("source_movie", "")))
        ttk.Entry(self, textvariable=self.source_var).grid(row=2, column=1, padx=10, pady=8, sticky="ew")

        ttk.Label(self, text="Opis:").grid(row=3, column=0, padx=10, pady=8, sticky="nw")
        self.description = tk.Text(self, width=55, height=7)
        self.description.insert("1.0", str(info.get("description", "")))
        self.description.grid(row=3, column=1, padx=10, pady=8, sticky="nsew")
        self.rowconfigure(3, weight=1)

        ttk.Label(self, text="Widoczność:").grid(row=4, column=0, padx=10, pady=8, sticky="w")
        self.visibility_var = tk.StringVar(value=visibility_of(info))
        ttk.Combobox(
            self,
            state="readonly",
            textvariable=self.visibility_var,
            values=["private", "public"],
        ).grid(row=4, column=1, padx=10, pady=8, sticky="ew")

        buttons = ttk.Frame(self)
        buttons.grid(row=5, column=0, columnspan=2, padx=10, pady=12, sticky="e")
        ttk.Button(buttons, text="Anuluj", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="💾 Zapisz", command=self._save).pack(side="right", padx=4)

        self.transient(parent)
        self.grab_set()
        self.wait_visibility()
        self.focus_set()

    def _save(self):
        self.result = {
            "display_name": self.display_var.get().strip(),
            "source_movie": self.source_var.get().strip(),
            "description": self.description.get("1.0", "end").strip(),
            "visibility": self.visibility_var.get().strip(),
        }
        self.destroy()


class VoiceDBManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎙️ ViDubb Voice DB Manager")
        self.geometry("1180x680")
        self.minsize(900, 560)
        self.db: dict = {}
        self.busy = False
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="📂 Importuj folder", command=self.import_folder).pack(side="left", padx=3)
        ttk.Button(top, text="🎵 Importuj pliki", command=self.import_files).pack(side="left", padx=3)
        ttk.Button(top, text="🔄 Odśwież", command=self.refresh).pack(side="left", padx=3)
        ttk.Button(top, text="🛠 Migruj DB", command=self.migrate).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="📦 Eksport PUBLIC", command=self.export_release).pack(side="left", padx=3)
        ttk.Button(top, text="📁 Otwórz voice_db", command=lambda: open_path(VOICE_DB_DIR)).pack(side="left", padx=3)

        self.status_var = tk.StringVar(value="Gotowy.")
        ttk.Label(top, textvariable=self.status_var).pack(side="right", padx=8)

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(body)
        body.add(left, weight=4)

        columns = ("visibility", "display_name", "voice_id", "duration", "embeddings", "samples", "source")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        headings = {
            "visibility": "Dostęp", "display_name": "Nazwa", "voice_id": "Voice ID",
            "duration": "Czas", "embeddings": "ECAPA", "samples": "Próbki", "source": "Źródło"
        }
        widths = {
            "visibility": 80, "display_name": 170, "voice_id": 220,
            "duration": 70, "embeddings": 65, "samples": 65, "source": 150
        }
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], minwidth=50, anchor="w")

        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda _e: self.edit_selected())

        right = ttk.Frame(body, padding=(12, 4))
        body.add(right, weight=2)
        ttk.Label(right, text="Wybrany profil", font=("", 12, "bold")).pack(anchor="w", pady=(0, 10))

        self.detail_text = tk.Text(right, wrap="word", height=20, state="disabled")
        self.detail_text.pack(fill="both", expand=True)

        actions = ttk.Frame(right)
        actions.pack(fill="x", pady=10)
        ttk.Button(actions, text="▶ Odsłuch", command=self.preview_selected).pack(fill="x", pady=2)
        ttk.Button(actions, text="✏ Edytuj metadane", command=self.edit_selected).pack(fill="x", pady=2)
        ttk.Button(actions, text="🌍 Ustaw PUBLIC", command=lambda: self.set_selected_visibility("public")).pack(fill="x", pady=2)
        ttk.Button(actions, text="🔒 Ustaw PRIVATE", command=lambda: self.set_selected_visibility("private")).pack(fill="x", pady=2)
        ttk.Separator(actions).pack(fill="x", pady=8)
        ttk.Button(actions, text="🗑 Usuń profil", command=self.delete_selected).pack(fill="x", pady=2)

        ttk.Label(
            right,
            text=(
                "Bezpieczna zasada wydania:\n"
                "• brak pola visibility = PRIVATE\n"
                "• eksport PUBLIC kopiuje tylko jawnie oznaczone głosy\n"
                "• usunięcie robi wcześniej backup profilu"
            ),
            justify="left",
            foreground="#666666",
        ).pack(anchor="w", pady=8)

    def selected_voice_id(self) -> str | None:
        selected = self.tree.selection()
        return selected[0] if selected else None

    def set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def refresh(self):
        try:
            self.db = load_voice_db()
        except Exception as exc:
            messagebox.showerror("Voice DB", f"Nie udało się odczytać bazy:\n{exc}")
            return

        current = self.selected_voice_id()
        for item in self.tree.get_children():
            self.tree.delete(item)

        for voice_id, info in sorted(
            self.db.items(),
            key=lambda item: str(item[1].get("display_name", item[0]) if isinstance(item[1], dict) else item[0]).lower(),
        ):
            if not isinstance(info, dict):
                continue
            embeddings = info.get("embedding_samples", [])
            embedding_count = len(embeddings) if isinstance(embeddings, list) else int(info.get("embedding_count", 0) or 0)
            visibility = visibility_of(info)
            visibility_label = "🌍 PUBLIC" if visibility == "public" else "🔒 PRIVATE"
            self.tree.insert(
                "", "end", iid=voice_id,
                values=(
                    visibility_label,
                    info.get("display_name", voice_id),
                    voice_id,
                    f"{float(info.get('duration_sec', 0) or 0):.1f}s",
                    embedding_count,
                    int(info.get("sample_count", 0) or 0),
                    info.get("source_movie", ""),
                ),
            )

        if current and self.tree.exists(current):
            self.tree.selection_set(current)
            self.tree.focus(current)
        self._update_detail()
        self.set_status(f"Profile: {len(self.db)}")

    def _on_select(self, _event=None):
        self._update_detail()

    def _update_detail(self):
        voice_id = self.selected_voice_id()
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        if not voice_id or voice_id not in self.db:
            self.detail_text.insert("1.0", "Wybierz profil z listy.")
            self.detail_text.configure(state="disabled")
            return

        info = self.db[voice_id]
        ref = profile_reference_path(voice_id, info)
        embedding_samples = info.get("embedding_samples", [])
        emb_count = len(embedding_samples) if isinstance(embedding_samples, list) else info.get("embedding_count", 0)
        details = (
            f"Voice ID:\n{voice_id}\n\n"
            f"Display name:\n{info.get('display_name', voice_id)}\n\n"
            f"Visibility:\n{visibility_of(info).upper()}\n\n"
            f"Source movie:\n{info.get('source_movie', '') or '—'}\n\n"
            f"Duration:\n{info.get('duration_sec', 0)} s\n\n"
            f"Sample count:\n{info.get('sample_count', 0)}\n\n"
            f"ECAPA embeddings:\n{emb_count}\n\n"
            f"Reference:\n{ref if ref else 'BRAK'}\n\n"
            f"Description:\n{info.get('description', '') or '—'}"
        )
        self.detail_text.insert("1.0", details)
        self.detail_text.configure(state="disabled")

    def import_folder(self):
        folder = filedialog.askdirectory(title="Wybierz folder z próbkami jednego głosu")
        if not folder:
            return
        default_name = Path(folder).name
        display_name = simpledialog.askstring("Nazwa głosu", "Display name aktora / głosu:", initialvalue=default_name, parent=self)
        if not display_name:
            return
        source_movie = simpledialog.askstring("Źródło", "Film / serial / źródło (opcjonalnie):", parent=self) or ""
        self._start_import(Path(folder), display_name.strip(), source_movie.strip(), cleanup_folder=False)

    def import_files(self):
        files = filedialog.askopenfilenames(
            title="Wybierz próbki jednego głosu",
            filetypes=[("Audio/Video", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.mp4 *.webm *.mkv *.mov *.avi"), ("Wszystkie", "*.*")],
        )
        if not files:
            return

        display_name = simpledialog.askstring("Nazwa głosu", "Display name aktora / głosu:", parent=self)
        if not display_name:
            return
        source_movie = simpledialog.askstring("Źródło", "Film / serial / źródło (opcjonalnie):", parent=self) or ""

        tmp = Path(tempfile.mkdtemp(prefix="vidubb_voice_import_"))
        valid = 0
        for index, raw in enumerate(files):
            src = Path(raw)
            suffix = src.suffix.lower()

            if suffix in SUPPORTED_AUDIO:
                dst = tmp / f"{index:03d}_{safe_filename(src.name)}"
                shutil.copy2(src, dst)
                valid += 1
                continue

            if suffix in SUPPORTED_VIDEO:
                if not shutil.which("ffmpeg"):
                    continue
                dst = tmp / f"{index:03d}_{safe_filename(src.stem)}.wav"
                res = subprocess.run(
                    [
                        "ffmpeg", "-y", "-nostdin",
                        "-i", str(src),
                        "-vn", "-ac", "1", "-ar", "24000",
                        "-c:a", "pcm_s16le", str(dst),
                    ],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0 and dst.exists() and dst.stat().st_size > 1000:
                    valid += 1

        if valid == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            messagebox.showwarning("Import", "Nie wybrano wspieranych plików audio/video.")
            return
        self._start_import(tmp, display_name.strip(), source_movie.strip(), cleanup_folder=True)

    def _start_import(self, folder: Path, display_name: str, source_movie: str, cleanup_folder: bool):
        if self.busy:
            return
        self.busy = True
        self.set_status("ECAPA/import w toku — może chwilę potrwać...")

        def worker():
            try:
                voice_id = import_voice_from_folder(
                    folder_path=str(folder),
                    display_name=display_name,
                    source_movie=source_movie,
                    description="Imported with ViDubb Voice DB Manager",
                )
                if voice_id:
                    set_visibility(voice_id, "private")
                    self.after(0, lambda: messagebox.showinfo(
                        "Import zakończony",
                        f"Zaimportowano głos:\n\n{display_name}\n\nVoice ID:\n{voice_id}\n\nProfil został ustawiony jako PRIVATE.",
                    ))
                else:
                    self.after(0, lambda: messagebox.showwarning("Import", "Nie udało się utworzyć profilu."))
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}"
                self.after(0, lambda: messagebox.showerror("Błąd importu", text))
            finally:
                if cleanup_folder:
                    shutil.rmtree(folder, ignore_errors=True)
                self.busy = False
                self.after(0, self.refresh)

        threading.Thread(target=worker, daemon=True, name="voice-db-import").start()

    def preview_selected(self):
        voice_id = self.selected_voice_id()
        if not voice_id:
            return
        info = self.db.get(voice_id, {})
        path = profile_reference_path(voice_id, info)
        if not path:
            messagebox.showwarning("Odsłuch", "Nie znaleziono pliku WAV/preview dla tego profilu.")
            return
        play_audio(path)

    def edit_selected(self):
        voice_id = self.selected_voice_id()
        if not voice_id:
            return
        info = self.db.get(voice_id)
        if not isinstance(info, dict):
            return
        dialog = MetadataDialog(self, voice_id, info)
        if not dialog.result:
            return
        result = dialog.result
        ok = update_voice_metadata(
            voice_id=voice_id,
            display_name=result["display_name"] or voice_id,
            source_movie=result["source_movie"],
            description=result["description"],
        )
        if not ok:
            messagebox.showerror("Edycja", "Nie udało się zaktualizować profilu.")
            return
        set_visibility(voice_id, result["visibility"])
        self.refresh()

    def set_selected_visibility(self, visibility: str):
        voice_id = self.selected_voice_id()
        if not voice_id:
            return
        if set_visibility(voice_id, visibility):
            self.refresh()
            self.set_status(f"{voice_id}: {visibility.upper()}")

    def delete_selected(self):
        voice_id = self.selected_voice_id()
        if not voice_id:
            return
        info = self.db.get(voice_id, {})
        if not messagebox.askyesno(
            "Usuń profil",
            f"Czy na pewno usunąć profil?\n\n{info.get('display_name', voice_id)}\n{voice_id}\n\nPrzed usunięciem program zrobi lokalny backup.",
        ):
            return
        try:
            backup = backup_profile(voice_id, info if isinstance(info, dict) else {})
            if delete_voice_profile(voice_id):
                messagebox.showinfo("Usunięto", f"Profil usunięty.\n\nBackup:\n{backup}")
            else:
                messagebox.showwarning("Usuwanie", "Profil już nie istnieje.")
        except Exception as exc:
            messagebox.showerror("Usuwanie", f"Nie udało się usunąć:\n{exc}")
        self.refresh()

    def migrate(self):
        try:
            changed = migrate_voice_db()
            self.refresh()
            messagebox.showinfo("Migracja", f"Zmigrowano profili: {changed}")
        except Exception as exc:
            messagebox.showerror("Migracja", str(exc))

    def export_release(self):
        destination = filedialog.askdirectory(title="Wybierz katalog dla paczki PUBLIC")
        if not destination:
            return
        destination = Path(destination).resolve()
        public_count = sum(
            1 for info in self.db.values()
            if isinstance(info, dict) and visibility_of(info) == "public"
        )
        if public_count == 0:
            messagebox.showwarning("Eksport", "Nie ma żadnych profili oznaczonych PUBLIC.")
            return
        if not messagebox.askyesno(
            "Eksport PUBLIC",
            f"Wyeksportować {public_count} profili PUBLIC do:\n\n{destination}\n\nProfile PRIVATE nie zostaną skopiowane.",
        ):
            return
        try:
            count, warnings = export_public_bundle(destination)
            text = f"Gotowe.\n\nWyeksportowano profili: {count}\nKatalog: {destination}"
            if warnings:
                text += "\n\nOstrzeżenia:\n" + "\n".join(warnings[:10])
            messagebox.showinfo("Eksport zakończony", text)
        except Exception as exc:
            messagebox.showerror("Eksport", f"Nie udało się utworzyć paczki:\n{exc}")


def main():
    app = VoiceDBManager()
    app.mainloop()


if __name__ == "__main__":
    main()
