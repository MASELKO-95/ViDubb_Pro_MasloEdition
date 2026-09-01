#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viDubb Pro — Maslo95 Edition Launcher

Launcher features:
- PL / EN interface (English is default for new installations)
- dependency check/install
- GitHub Releases updater behind the existing "Check for Updates" button
- safe ZIP installation with rollback backup
- user/project data is NEVER overwritten by the updater

GitHub repository:
    https://github.com/MASELKO-95/ViDubb_Pro_MasloEdition

Recommended release flow:
1. Change APP_VERSION in this file, e.g. 1.0.1
2. Create GitHub Release tag: v1.0.1
3. Optionally attach your own ZIP asset to the Release.
   If no ZIP asset exists, the updater uses GitHub's release source ZIP.
"""

from __future__ import annotations

import importlib
import json
import locale
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import zipfile
from pathlib import Path


def ensure_utf8_locale() -> None:
    """Give Qt and child processes a UTF-8 locale on minimal Linux systems."""
    encoding = locale.getpreferredencoding(False).lower().replace("-", "")
    configured_locales = {
        os.environ.get(name, "").strip().upper()
        for name in ("LANG", "LC_CTYPE", "LC_ALL")
        if os.environ.get(name)
    }
    legacy_locale = any(
        value in {"C", "POSIX"} or "ANSI_X3.4" in value
        for value in configured_locales
    )

    if encoding not in {"utf8", "utf_8"} or legacy_locale:
        os.environ["LANG"] = "C.UTF-8"
        os.environ["LC_CTYPE"] = "C.UTF-8"
        os.environ["LC_ALL"] = "C.UTF-8"
        try:
            locale.setlocale(locale.LC_CTYPE, "")
        except locale.Error:
            pass


ensure_utf8_locale()

import requests


# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "viDubb Pro — Maslo95 Edition"
APP_VERSION = "1.0.0"

REQUIREMENTS_FILE = "requirements.txt"
VENV_DIR = ".venv"
SERVER_SCRIPT = "app_new.py"
SERVER_PORT = 7860
CONFIG_FILE = ".launcher_cfg.json"
SUPPORT_URL = "https://buycoffee.to/maslo_github"

GITHUB_OWNER = "MASELKO-95"
GITHUB_REPO = "ViDubb_Pro_MasloEdition"
GITHUB_API = (
    f"https://api.github.com/repos/"
    f"{GITHUB_OWNER}/{GITHUB_REPO}"
)

WORKSPACE = Path(__file__).resolve().parent
os.chdir(WORKSPACE)


# ============================================================
# UPDATE SAFETY
# ============================================================

# These paths belong to the user/runtime and must NEVER be overwritten
# by an application update.
PRESERVE_DIRS = {
    ".git",
    ".venv",
    "projects",
    "voice_db",
    "speakers_audio",
    "uploads",
    "results",
    "audio",
    "audio_chunks",
    "temp",
    "__pycache__",
}

PRESERVE_FILES = {
    CONFIG_FILE,
}

UPDATE_DIR = WORKSPACE / ".updates"
BACKUP_DIR = UPDATE_DIR / "backups"


# ============================================================
# TRANSLATIONS
# ============================================================

T = {
    "en": {
        "title": f"🎬 {APP_NAME} — Launcher",
        "btn_launch": "▶ Launch Server",
        "btn_check_deps": "📦 Verify & Install Dependencies",
        "btn_updates": "🔄 Check for Updates",
        "btn_lang": "🇵🇱 Polski",
        "status_ready": "✅ Ready.",
        "status_checking": "⏳ Checking dependencies...",
        "status_installing": "📦 Installing packages...",
        "status_done": "✅ Dependency check complete.",
        "launching": "🚀 Launching server...",
        "support": "☕ Support the project",
        "support_launch": "☕ Like viDubb? You can buy me a coffee. Starting in {seconds}s...",
        "no_pip": "❌ pip was not found.",
        "update_checking": "🔎 Checking GitHub for updates...",
        "update_current": "✅ You already have the latest version.",
        "update_available": "🆕 New version available",
        "update_downloading": "⬇️ Downloading update...",
        "update_installing": "🛠 Installing update...",
        "update_done": "✅ Update installed. Restart the launcher.",
        "update_error": "❌ Update failed",
        "update_confirm": "Install the update now?",
        "update_restart": (
            "The update was installed successfully.\n\n"
            "Restart the launcher to use the new version."
        ),
        "private_repo_hint": (
            "GitHub returned 404/401. If the repository is still private, "
            "set VIDUBB_GITHUB_TOKEN or GITHUB_TOKEN in your environment."
        ),
    },
    "pl": {
        "title": f"🎬 {APP_NAME} — Launcher",
        "btn_launch": "▶ Uruchom serwer",
        "btn_check_deps": "📦 Sprawdź i zainstaluj pakiety",
        "btn_updates": "🔄 Sprawdź aktualizacje",
        "btn_lang": "🇬🇧 English",
        "status_ready": "✅ Gotowy.",
        "status_checking": "⏳ Sprawdzanie zależności...",
        "status_installing": "📦 Instalowanie pakietów...",
        "status_done": "✅ Sprawdzanie zależności zakończone.",
        "launching": "🚀 Uruchamianie serwera...",
        "support": "☕ Wesprzyj projekt",
        "support_launch": "☕ Podoba Ci się viDubb? Postaw mi kawę. Start za {seconds} s...",
        "no_pip": "❌ Nie znaleziono pip.",
        "update_checking": "🔎 Sprawdzanie aktualizacji na GitHub...",
        "update_current": "✅ Masz już najnowszą wersję.",
        "update_available": "🆕 Dostępna nowa wersja",
        "update_downloading": "⬇️ Pobieranie aktualizacji...",
        "update_installing": "🛠 Instalowanie aktualizacji...",
        "update_done": "✅ Aktualizacja zainstalowana. Uruchom launcher ponownie.",
        "update_error": "❌ Aktualizacja nie powiodła się",
        "update_confirm": "Zainstalować aktualizację teraz?",
        "update_restart": (
            "Aktualizacja została zainstalowana.\n\n"
            "Uruchom launcher ponownie, aby użyć nowej wersji."
        ),
        "private_repo_hint": (
            "GitHub zwrócił 404/401. Jeśli repozytorium nadal jest prywatne, "
            "ustaw zmienną VIDUBB_GITHUB_TOKEN lub GITHUB_TOKEN."
        ),
    },
}


# ============================================================
# CONFIG
# ============================================================

def load_config() -> dict:
    path = WORKSPACE / CONFIG_FILE

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data.setdefault("lang", "en")
                data.setdefault("first_run", True)
                return data
        except Exception:
            pass

    # New installations are international/English-first.
    return {
        "lang": "en",
        "first_run": True,
    }


def save_config(config: dict) -> None:
    path = WORKSPACE / CONFIG_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    tmp.replace(path)


cfg = load_config()


# ============================================================
# PLATFORM / DEPENDENCIES
# ============================================================

def detect_platform() -> dict:
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pip": None,
        "venv_python": None,
    }

    if info["system"] == "Windows":
        info["venv_python"] = str(
            WORKSPACE / VENV_DIR / "Scripts" / "python.exe"
        )
        info["pip"] = str(
            WORKSPACE / VENV_DIR / "Scripts" / "pip.exe"
        )
    else:
        info["venv_python"] = str(
            WORKSPACE / VENV_DIR / "bin" / "python3"
        )
        info["pip"] = str(
            WORKSPACE / VENV_DIR / "bin" / "pip3"
        )

    if not os.path.exists(info["venv_python"]):
        info["venv_python"] = sys.executable
        info["pip"] = shutil.which("pip3") or shutil.which("pip")

    return info


def parse_requirements(filepath: str | Path) -> list[dict]:
    filepath = Path(filepath)

    if not filepath.exists():
        return []

    packages = []

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            # Skip pip options, VCS URLs and local paths.
            if line.startswith(("-", "git+", "http://", "https://", ".")):
                continue

            name = (
                line.split("==")[0]
                .split(">=")[0]
                .split("<=")[0]
                .split("~=")[0]
                .split("!=")[0]
                .split("<")[0]
                .split(">")[0]
                .split("[")[0]
                .strip()
            )

            import_map = {
                "pyannote.audio": "pyannote.audio",
                "TTS": "TTS",
                "openai-whisper": "whisper",
                "faster-whisper": "faster_whisper",
                "pydub": "pydub",
                "ffmpeg-python": "ffmpeg",
                "opencv-python": "cv2",
                "opencv-python-headless": "cv2",
                "scikit-image": "skimage",
                "python-dotenv": "dotenv",
                "ascii-magic": "ascii_magic",
                "yt-dlp": "yt_dlp",
                "audio-separator": "audio_separator",
                "speechbrain": "speechbrain",
                "deepface": "deepface",
            }

            packages.append({
                "pip_name": line,
                "import_name": import_map.get(
                    name,
                    name.replace("-", "_"),
                ),
                "display_name": name,
            })

    return packages


def check_all_requirements(packages: list[dict]):
    installed = []
    missing = []

    for pkg in packages:
        try:
            importlib.import_module(pkg["import_name"])
            installed.append(pkg)
        except ImportError:
            missing.append(pkg)

    return installed, missing


def launch_server(platform_info: dict) -> None:
    print(f"\n🚀 {T[cfg['lang']]['launching']}")
    print(f"   http://127.0.0.1:{SERVER_PORT}")
    print("   Ctrl+C to stop.\n")

    try:
        child_env = os.environ.copy()

        if platform_info["system"] in ("Linux", "Darwin"):
            venv_python = WORKSPACE / VENV_DIR / "bin" / "python3"
            venv_bin = WORKSPACE / VENV_DIR / "bin"

            python_exe = (
                str(venv_python)
                if venv_python.exists()
                else platform_info["venv_python"]
            )

            subprocess.run(
                [python_exe, SERVER_SCRIPT],
                cwd=str(WORKSPACE),
                env={
                    **child_env,
                    "PATH": os.pathsep.join(
                        [str(venv_bin), child_env.get("PATH", "")]
                    ),
                },
            )
        else:
            venv_scripts = WORKSPACE / VENV_DIR / "Scripts"
            subprocess.run(
                [
                    platform_info["venv_python"],
                    SERVER_SCRIPT,
                ],
                cwd=str(WORKSPACE),
                env={
                    **child_env,
                    "PATH": os.pathsep.join(
                        [str(venv_scripts), child_env.get("PATH", "")]
                    ),
                },
            )

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error launching: {exc}")


# ============================================================
# VERSION HELPERS
# ============================================================

def _version_tuple(value: str) -> tuple:
    """
    Lightweight version comparison.
    v1.2.10 > v1.2.9

    Pre-release suffixes are intentionally ignored here because the normal
    updater only consumes GitHub's latest stable Release.
    """
    value = str(value or "").strip().lower()
    value = value.lstrip("v")

    numbers = re.findall(r"\d+", value)

    if not numbers:
        return (0,)

    return tuple(int(x) for x in numbers[:4])


def is_newer_version(remote: str, local: str) -> bool:
    a = list(_version_tuple(remote))
    b = list(_version_tuple(local))

    length = max(len(a), len(b))
    a.extend([0] * (length - len(a)))
    b.extend([0] * (length - len(b)))

    return tuple(a) > tuple(b)


# ============================================================
# GITHUB UPDATE
# ============================================================

def _github_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": (
            f"{GITHUB_REPO}-Launcher/{APP_VERSION}"
        ),
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Allows testing while repository is private.
    token = (
        os.environ.get("VIDUBB_GITHUB_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )

    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"

    return headers


def fetch_latest_release() -> dict:
    url = f"{GITHUB_API}/releases/latest"

    response = requests.get(
        url,
        headers=_github_headers(),
        timeout=20,
    )

    if response.status_code in (401, 403, 404):
        raise RuntimeError(
            f"GitHub HTTP {response.status_code}. "
            + T[cfg["lang"]]["private_repo_hint"]
        )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise RuntimeError("Invalid GitHub response.")

    tag = str(data.get("tag_name", "") or "").strip()

    if not tag:
        raise RuntimeError(
            "The latest GitHub Release has no tag_name."
        )

    return data


def choose_release_download(release: dict) -> tuple[str, str]:
    """
    Prefer a ZIP asset uploaded by the developer.
    If no ZIP asset exists, use GitHub's source zipball.
    """
    assets = release.get("assets", [])

    if isinstance(assets, list):
        zip_assets = [
            asset
            for asset in assets
            if isinstance(asset, dict)
            and str(asset.get("name", "")).lower().endswith(".zip")
            and asset.get("browser_download_url")
        ]

        if zip_assets:
            # Prefer filenames mentioning release/app/update.
            zip_assets.sort(
                key=lambda a: (
                    not any(
                        word in str(a.get("name", "")).lower()
                        for word in (
                            "vidubb",
                            "release",
                            "update",
                        )
                    ),
                    str(a.get("name", "")).lower(),
                )
            )

            asset = zip_assets[0]

            return (
                str(asset["browser_download_url"]),
                str(asset.get("name", "update.zip")),
            )

    zipball_url = str(
        release.get("zipball_url", "") or ""
    )

    if not zipball_url:
        raise RuntimeError(
            "Release contains no ZIP asset and no zipball_url."
        )

    return (
        zipball_url,
        f"{GITHUB_REPO}-{release.get('tag_name', 'update')}.zip",
    )


def download_update_zip(
    url: str,
    output_path: Path,
    progress_callback=None,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with requests.get(
        url,
        headers=_github_headers(),
        timeout=(20, 300),
        stream=True,
    ) as response:
        response.raise_for_status()

        total = int(
            response.headers.get(
                "Content-Length",
                0,
            ) or 0
        )

        downloaded = 0

        with output_path.open("wb") as f:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if progress_callback and total > 0:
                    progress_callback(
                        min(
                            100,
                            int(downloaded * 100 / total),
                        )
                    )

    if (
        not output_path.exists()
        or output_path.stat().st_size < 1000
    ):
        raise RuntimeError(
            "Downloaded update ZIP is empty or invalid."
        )


# ============================================================
# SAFE ZIP INSTALLATION
# ============================================================

def _safe_zip_member_path(
    destination: Path,
    member_name: str,
) -> Path:
    """
    Prevent ZIP path traversal such as ../../something.
    """
    target = (
        destination
        / member_name.replace("\\", "/")
    ).resolve()

    destination_resolved = destination.resolve()

    try:
        target.relative_to(destination_resolved)
    except ValueError:
        raise RuntimeError(
            f"Unsafe path in update archive: {member_name}"
        )

    return target


def extract_zip_safely(
    zip_path: Path,
    destination: Path,
) -> None:
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = _safe_zip_member_path(
                destination,
                info.filename,
            )

            if info.is_dir():
                target.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                continue

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with zf.open(info, "r") as src:
                with target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def detect_update_root(extracted_dir: Path) -> Path:
    """
    GitHub source ZIPs normally contain one generated top-level directory:
        MASELKO-95-ViDubb_Pro_MasloEdition-<sha>/

    Developer-uploaded ZIPs may contain project files directly.
    """
    entries = [
        p
        for p in extracted_dir.iterdir()
        if p.name != "__MACOSX"
    ]

    if len(entries) == 1 and entries[0].is_dir():
        candidate = entries[0]

        markers = {
            "modules",
            "templates",
            "requirements.txt",
            "app_new.py",
            "run_launcher.py",
        }

        if any(
            (candidate / marker).exists()
            for marker in markers
        ):
            return candidate

    return extracted_dir


def should_preserve(relative_path: Path) -> bool:
    parts = relative_path.parts

    if not parts:
        return True

    if parts[0] in PRESERVE_DIRS:
        return True

    if relative_path.as_posix() in PRESERVE_FILES:
        return True

    if any(
        part == "__pycache__"
        for part in parts
    ):
        return True

    if relative_path.suffix in {
        ".pyc",
        ".pyo",
    }:
        return True

    return False


def backup_existing_file(
    destination_file: Path,
    relative_path: Path,
    backup_root: Path,
) -> None:
    if not destination_file.exists():
        return

    backup_file = backup_root / relative_path

    backup_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination_file.is_file():
        shutil.copy2(
            destination_file,
            backup_file,
        )


def install_update_tree(
    update_root: Path,
    log_callback=None,
) -> tuple[int, Path]:
    """
    Copy application files from extracted update over WORKSPACE.

    User-data directories are ignored even if they accidentally exist
    inside the release ZIP.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = BACKUP_DIR / timestamp
    backup_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0

    for source in update_root.rglob("*"):
        if not source.is_file():
            continue

        relative = source.relative_to(update_root)

        if should_preserve(relative):
            if log_callback:
                log_callback(
                    f"KEEP: {relative.as_posix()}"
                )
            continue

        destination = WORKSPACE / relative

        # Extra guard if the target path already lies in protected runtime data.
        try:
            relative_destination = destination.resolve().relative_to(
                WORKSPACE.resolve()
            )
        except ValueError:
            raise RuntimeError(
                f"Update attempted to write outside workspace: {destination}"
            )

        if should_preserve(relative_destination):
            continue

        backup_existing_file(
            destination,
            relative,
            backup_root,
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Copy to a temporary sibling first, then replace atomically.
        temp_destination = destination.with_name(
            destination.name + ".update_tmp"
        )

        shutil.copy2(
            source,
            temp_destination,
        )

        os.replace(
            temp_destination,
            destination,
        )

        copied += 1

        if log_callback:
            log_callback(
                f"UPDATE: {relative.as_posix()}"
            )

    # Record what version initiated this backup.
    metadata = {
        "previous_version": APP_VERSION,
        "created_at": time.time(),
        "files_updated": copied,
    }

    with (
        backup_root / "_update_backup.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return copied, backup_root


def perform_update(
    release: dict,
    log_callback=None,
    progress_callback=None,
) -> dict:
    tag = str(
        release.get("tag_name", "") or ""
    ).strip()

    url, download_name = choose_release_download(
        release
    )

    UPDATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="vidubb_update_",
        dir=str(UPDATE_DIR),
    ) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)

        zip_path = temp_dir / download_name
        extract_dir = temp_dir / "extracted"

        if log_callback:
            log_callback(
                f"Download: {url}"
            )

        download_update_zip(
            url,
            zip_path,
            progress_callback=progress_callback,
        )

        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(
                "Downloaded file is not a valid ZIP archive."
            )

        extract_zip_safely(
            zip_path,
            extract_dir,
        )

        update_root = detect_update_root(
            extract_dir
        )

        # Sanity check: avoid installing a completely unrelated ZIP.
        markers = [
            update_root / "modules",
            update_root / "app_new.py",
            update_root / "requirements.txt",
        ]

        if not any(marker.exists() for marker in markers):
            raise RuntimeError(
                "The downloaded ZIP does not look like a ViDubb update."
            )

        copied, backup_root = install_update_tree(
            update_root,
            log_callback=log_callback,
        )

    # Dependencies may have changed.
    cfg["first_run"] = True
    cfg["last_update_tag"] = tag
    save_config(cfg)

    return {
        "tag": tag,
        "files_updated": copied,
        "backup": str(backup_root),
    }


# ============================================================
# GUI
# ============================================================

def run_gui(
    platform_info: dict,
    packages: list[dict],
) -> bool:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError:
        return False

    try:
        root = tk.Tk()
    except tk.TclError:
        return False

    def get_t(key: str) -> str:
        lang = cfg.get("lang", "en")

        if lang not in T:
            lang = "en"

        return T[lang][key]

    root.title(get_t("title"))
    root.geometry("720x560")
    root.configure(bg="#0d0e16")
    root.resizable(False, False)

    BG = "#0d0e16"
    BG2 = "#151724"
    FG = "#f1f5f9"
    ACCENT = "#6366f1"

    style = ttk.Style()

    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(
        "TFrame",
        background=BG,
    )
    style.configure(
        "TLabel",
        background=BG,
        foreground=FG,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Title.TLabel",
        background=BG,
        foreground=FG,
        font=("Segoe UI", 16, "bold"),
    )
    style.configure(
        "TButton",
        font=("Segoe UI", 10, "bold"),
        padding=8,
        background="#1e293b",
        foreground=FG,
    )
    style.map(
        "TButton",
        background=[("active", ACCENT)],
    )
    style.configure(
        "Launch.TButton",
        font=("Segoe UI", 11, "bold"),
        padding=10,
        background="#10b981",
        foreground="#ffffff",
    )
    style.map(
        "Launch.TButton",
        background=[("active", "#059669")],
    )

    header = ttk.Frame(root)
    header.pack(
        fill="x",
        padx=20,
        pady=(20, 10),
    )

    lbl_title = ttk.Label(
        header,
        text=get_t("title"),
        style="Title.TLabel",
    )
    lbl_title.pack(side="left")

    version_label = ttk.Label(
        header,
        text=f"v{APP_VERSION}",
    )
    version_label.pack(
        side="left",
        padx=10,
    )

    btn_lang = ttk.Button(
        header,
        text=get_t("btn_lang"),
        cursor="hand2",
    )
    btn_lang.pack(side="right")

    main_frame = ttk.Frame(root)
    main_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=10,
    )

    lbl_status = ttk.Label(
        main_frame,
        text=get_t("status_ready"),
    )
    lbl_status.pack(
        anchor="w",
        pady=(0, 10),
    )

    progress = ttk.Progressbar(
        main_frame,
        orient="horizontal",
        mode="determinate",
        maximum=100,
    )
    progress.pack(
        fill="x",
        pady=(0, 8),
    )
    progress["value"] = 0

    log_text = tk.Text(
        main_frame,
        height=14,
        bg=BG2,
        fg=FG,
        font=("Consolas", 9),
        bd=0,
        padx=10,
        pady=10,
    )
    log_text.pack(
        fill="both",
        expand=True,
        pady=(0, 15),
    )
    log_text.config(state="disabled")

    def add_log(msg: str) -> None:
        def _append():
            log_text.config(state="normal")
            log_text.insert("end", msg + "\n")
            log_text.see("end")
            log_text.config(state="disabled")

        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            root.after(0, _append)

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x")

    btn_check = ttk.Button(
        btn_frame,
        text=get_t("btn_check_deps"),
    )
    btn_check.pack(
        side="left",
        padx=(0, 5),
    )

    btn_update = ttk.Button(
        btn_frame,
        text=get_t("btn_updates"),
    )
    btn_update.pack(
        side="left",
        padx=(0, 5),
    )

    btn_launch = ttk.Button(
        btn_frame,
        text=get_t("btn_launch"),
        style="Launch.TButton",
    )
    btn_launch.pack(side="right")

    support_frame = ttk.Frame(main_frame)
    support_frame.pack(fill="x", pady=(12, 0))

    support_label = ttk.Label(
        support_frame,
        text=get_t("support"),
        foreground="#34d399",
        cursor="hand2",
    )
    support_label.pack(side="right")
    support_label.bind(
        "<Button-1>",
        lambda _event: webbrowser.open(SUPPORT_URL),
    )

    def set_busy(busy: bool) -> None:
        state = ["disabled"] if busy else ["!disabled"]

        btn_check.state(state)
        btn_update.state(state)
        btn_launch.state(state)

    def set_status(text: str) -> None:
        lbl_status.config(text=text)

    def do_launch():
        set_busy(True)

        def finish_launch():
            root.destroy()
            launch_server(platform_info)

        def countdown(seconds: int):
            set_status(
                get_t("support_launch").format(seconds=seconds)
            )
            if seconds <= 1:
                root.after(1000, finish_launch)
            else:
                root.after(1000, lambda: countdown(seconds - 1))

        countdown(3)

    def check_deps_worker():
        root.after(
            0,
            lambda: set_status(
                get_t("status_checking")
            ),
        )
        root.after(
            0,
            lambda: set_busy(True),
        )

        add_log(
            "--- "
            + get_t("status_checking")
            + " ---"
        )

        installed, missing = check_all_requirements(
            packages
        )

        add_log(
            f"Installed: {len(installed)}, "
            f"Missing: {len(missing)}"
        )

        if missing:
            pip_path = platform_info["pip"]

            if not pip_path:
                add_log(get_t("no_pip"))
            else:
                root.after(
                    0,
                    lambda: set_status(
                        get_t("status_installing")
                    ),
                )

                for pkg in missing:
                    add_log(
                        f"Installing "
                        f"{pkg['display_name']}..."
                    )

                    cmd = [
                        pip_path,
                        "install",
                        pkg["pip_name"],
                        "-q",
                        "--no-warn-script-location",
                    ]

                    try:
                        subprocess.run(
                            cmd,
                            check=True,
                            capture_output=True,
                        )
                        add_log(
                            f"OK: {pkg['display_name']}"
                        )
                    except Exception as exc:
                        add_log(
                            f"ERR: {pkg['display_name']}: "
                            f"{exc}"
                        )

        cfg["first_run"] = False
        save_config(cfg)

        root.after(
            0,
            lambda: set_status(
                get_t("status_done")
            ),
        )
        root.after(
            0,
            lambda: set_busy(False),
        )

        add_log(
            "--- "
            + get_t("status_done")
            + " ---"
        )

    def check_deps():
        threading.Thread(
            target=check_deps_worker,
            daemon=True,
            name="launcher-dependency-check",
        ).start()

    def update_worker():
        root.after(
            0,
            lambda: set_busy(True),
        )
        root.after(
            0,
            lambda: set_status(
                get_t("update_checking")
            ),
        )
        root.after(
            0,
            lambda: progress.configure(value=0),
        )

        add_log(
            f"Current version: v{APP_VERSION}"
        )

        try:
            release = fetch_latest_release()
            remote_tag = str(
                release.get("tag_name", "")
            )

            add_log(
                f"Latest GitHub Release: {remote_tag}"
            )

            if not is_newer_version(
                remote_tag,
                APP_VERSION,
            ):
                root.after(
                    0,
                    lambda: set_status(
                        get_t("update_current")
                    ),
                )
                root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "viDubb Update",
                        get_t("update_current")
                        + f"\n\nLocal: v{APP_VERSION}\n"
                        + f"GitHub: {remote_tag}",
                    ),
                )
                return

            release_name = str(
                release.get("name", "")
                or remote_tag
            )

            body = str(
                release.get("body", "")
                or ""
            ).strip()

            confirm_text = (
                f"{get_t('update_available')}: "
                f"{release_name}\n\n"
                f"Local: v{APP_VERSION}\n"
                f"GitHub: {remote_tag}\n\n"
            )

            if body:
                # Keep the dialog readable.
                confirm_text += (
                    body[:1200]
                    + (
                        "\n..."
                        if len(body) > 1200
                        else ""
                    )
                    + "\n\n"
                )

            confirm_text += get_t("update_confirm")

            answer_holder = {
                "value": False,
            }

            ask_event = threading.Event()

            def ask_user():
                answer_holder["value"] = (
                    messagebox.askyesno(
                        "viDubb Update",
                        confirm_text,
                    )
                )
                ask_event.set()

            root.after(0, ask_user)
            ask_event.wait()

            if not answer_holder["value"]:
                add_log("Update cancelled.")
                return

            root.after(
                0,
                lambda: set_status(
                    get_t("update_downloading")
                ),
            )

            add_log(
                f"Downloading {remote_tag}..."
            )

            def report_progress(value: int):
                root.after(
                    0,
                    lambda v=value: progress.configure(
                        value=v
                    ),
                )

            result = perform_update(
                release,
                log_callback=add_log,
                progress_callback=report_progress,
            )

            root.after(
                0,
                lambda: progress.configure(value=100),
            )
            root.after(
                0,
                lambda: set_status(
                    get_t("update_done")
                ),
            )

            add_log(
                f"Updated files: "
                f"{result['files_updated']}"
            )
            add_log(
                f"Backup: {result['backup']}"
            )

            root.after(
                0,
                lambda: messagebox.showinfo(
                    "viDubb Update",
                    get_t("update_restart")
                    + "\n\n"
                    + f"Backup:\n{result['backup']}",
                ),
            )

        except Exception as exc:
            message = (
                f"{get_t('update_error')}:\n{exc}"
            )

            add_log(message)

            root.after(
                0,
                lambda m=message: set_status(m),
            )
            root.after(
                0,
                lambda m=message: messagebox.showerror(
                    "viDubb Update",
                    m,
                ),
            )

        finally:
            root.after(
                0,
                lambda: set_busy(False),
            )

    def check_updates():
        threading.Thread(
            target=update_worker,
            daemon=True,
            name="launcher-updater",
        ).start()

    def toggle_lang():
        cfg["lang"] = (
            "pl"
            if cfg.get("lang", "en") == "en"
            else "en"
        )
        save_config(cfg)

        root.title(get_t("title"))
        lbl_title.config(text=get_t("title"))
        btn_lang.config(text=get_t("btn_lang"))
        btn_launch.config(text=get_t("btn_launch"))
        btn_check.config(text=get_t("btn_check_deps"))
        btn_update.config(text=get_t("btn_updates"))
        support_label.config(text=get_t("support"))
        lbl_status.config(text=get_t("status_ready"))

    btn_launch.config(command=do_launch)
    btn_check.config(command=check_deps)
    btn_update.config(command=check_updates)
    btn_lang.config(command=toggle_lang)

    if cfg.get("first_run", True):
        root.after(500, check_deps)
    else:
        add_log(
            f"Ready. viDubb Pro v{APP_VERSION}"
        )

    root.mainloop()
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    platform_info = detect_platform()

    packages = parse_requirements(
        WORKSPACE / REQUIREMENTS_FILE
    )

    if not run_gui(
        platform_info,
        packages,
    ):
        print(
            "GUI unavailable (Tkinter). "
            "Launching server directly..."
        )

        if cfg.get("first_run", True):
            cfg["first_run"] = False
            save_config(cfg)

        launch_server(platform_info)


if __name__ == "__main__":
    main()
