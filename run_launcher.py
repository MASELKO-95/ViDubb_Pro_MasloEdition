#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViDubb_Pro_MasloEdition
"""

import sys
import os
import platform
import subprocess
import importlib
import json

APP_NAME = "ViDubb_Pro_MasloEdition"
APP_VERSION = "0.0.1"
REQUIREMENTS_FILE = "requirements.txt"
VENV_DIR = ".venv"
SERVER_SCRIPT = "app_new.py"
SERVER_PORT = 7860
CONFIG_FILE = ".launcher_cfg.json"

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE)

# ═══════════════════════════════════════════════════════════════
# TŁUMACZENIA / TRANSLATIONS
# ═══════════════════════════════════════════════════════════════
T = {
    "pl": {
        "title": f"🎬 {APP_NAME} — Launcher",
        "btn_launch": "▶ Uruchom Serwer (Launch)",
        "btn_check_deps": "📦 Sprawdź i zainstaluj pakiety",
        "btn_updates": "🔄 Sprawdź aktualizacje (Check updates)",
        "btn_lang": "🇬🇧 Zmień język na EN",
        "status_ready": "✅ Gotowy do uruchomienia.",
        "status_checking": "⏳ Sprawdzanie wymagań...",
        "status_installing": "📦 Instalacja pakietów...",
        "status_done": "✅ Instalacja zakończona!",
        "msg_updates_soon": "Funkcja aktualizacji wkrótce!",
        "launching": "🚀 Uruchamianie serwera...",
        "no_pip": "❌ Brak pip! Zainstaluj pip lub stwórz virtualenv.",
    },
    "en": {
        "title": f"🎬 {APP_NAME} — Launcher",
        "btn_launch": "▶ Launch Server",
        "btn_check_deps": "📦 Verify & Install Dependencies",
        "btn_updates": "🔄 Check for Updates",
        "btn_lang": "🇵🇱 Change language to PL",
        "status_ready": "✅ Ready to launch.",
        "status_checking": "⏳ Checking dependencies...",
        "status_installing": "📦 Installing packages...",
        "status_done": "✅ Installation complete!",
        "msg_updates_soon": "Updates feature coming soon!",
        "launching": "🚀 Launching server...",
        "no_pip": "❌ Pip not found! Install pip or create a virtualenv.",
    }
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"lang": "pl", "first_run": True}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

cfg = load_config()

# ═══════════════════════════════════════════════════════════════
# FUNKCJE BAZOWE
# ═══════════════════════════════════════════════════════════════
def detect_platform():
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pip": None,
        "venv_python": None,
    }
    if info["system"] == "Windows":
        info["venv_python"] = os.path.join(WORKSPACE, VENV_DIR, "Scripts", "python.exe")
        info["pip"] = os.path.join(WORKSPACE, VENV_DIR, "Scripts", "pip.exe")
    else:
        info["venv_python"] = os.path.join(WORKSPACE, VENV_DIR, "bin", "python3")
        info["pip"] = os.path.join(WORKSPACE, VENV_DIR, "bin", "pip3")

    if not os.path.exists(info["venv_python"]):
        import shutil
        info["venv_python"] = sys.executable
        info["pip"] = shutil.which("pip3") or shutil.which("pip")
    return info

def parse_requirements(filepath):
    if not os.path.exists(filepath):
        return []
    packages = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            name = line.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0].split("[")[0].strip()
            import_map = {
                "pyannote.audio": "pyannote.audio",
                "TTS": "TTS",
                "openai-whisper": "whisper",
                "pydub": "pydub",
                "ffmpeg-python": "ffmpeg",
                "opencv-python": "cv2",
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
                "import_name": import_map.get(name, name.replace("-", "_")),
                "display_name": name
            })
    return packages

def check_all_requirements(packages):
    installed, missing = [], []
    for pkg in packages:
        try:
            importlib.import_module(pkg["import_name"])
            installed.append(pkg)
        except ImportError:
            missing.append(pkg)
    return installed, missing

def launch_server(platform_info):

    sys_type = platform_info["system"]
    
    print(f"\n🚀 {T[cfg['lang']]['launching']}")
    print(f"   http://127.0.0.1:{SERVER_PORT}")
    print("   Ctrl+C to stop.\n")

    try:
        if sys_type == "Linux" or sys_type == "Darwin":

            cmd = f'cd "{WORKSPACE}" && source {VENV_DIR}/bin/activate 2>/dev/null || true && python3 {SERVER_SCRIPT}'
            subprocess.run(["bash", "-c", cmd])
        else:

            subprocess.run([platform_info["venv_python"], SERVER_SCRIPT], cwd=WORKSPACE)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error launching: {e}")

# ═══════════════════════════════════════════════════════════════
# GUI TKINTER
# ═══════════════════════════════════════════════════════════════
def run_gui(platform_info, packages):
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        return False

    try:
        root = tk.Tk()
    except tk.TclError:
        return False

    def get_t(key):
        return T[cfg["lang"]][key]

    root.title(get_t("title"))
    root.geometry("600x480")
    root.configure(bg="#0d0e16")
    root.resizable(False, False)

    BG = "#0d0e16"
    BG2 = "#151724"
    FG = "#f1f5f9"
    ACCENT = "#6366f1"

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
    style.configure("Title.TLabel", background=BG, foreground=FG, font=("Segoe UI", 16, "bold"))
    style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8, background="#1e293b", foreground=FG)
    style.map("TButton", background=[("active", ACCENT)])
    style.configure("Launch.TButton", font=("Segoe UI", 11, "bold"), padding=10, background="#10b981", foreground="#ffffff")
    style.map("Launch.TButton", background=[("active", "#059669")])

    # Header
    header = ttk.Frame(root)
    header.pack(fill="x", padx=20, pady=(20, 10))
    lbl_title = ttk.Label(header, text=get_t("title"), style="Title.TLabel")
    lbl_title.pack(side="left")

    btn_lang = ttk.Button(header, text=get_t("btn_lang"), cursor="hand2")
    btn_lang.pack(side="right")

    # Main area
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True, padx=20, pady=10)

    # Status
    lbl_status = ttk.Label(main_frame, text=get_t("status_ready"))
    lbl_status.pack(anchor="w", pady=(0, 10))

    # Log text
    log_text = tk.Text(main_frame, height=10, bg=BG2, fg=FG, font=("Consolas", 9), bd=0, padx=10, pady=10)
    log_text.pack(fill="both", expand=True, pady=(0, 15))
    log_text.config(state="disabled")

    def add_log(msg):
        log_text.config(state="normal")
        log_text.insert("end", msg + "\n")
        log_text.see("end")
        log_text.config(state="disabled")
        root.update_idletasks()

    # Buttons
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x")

    btn_check = ttk.Button(btn_frame, text=get_t("btn_check_deps"))
    btn_check.pack(side="left", padx=(0, 5))

    btn_update = ttk.Button(btn_frame, text=get_t("btn_updates"))
    btn_update.pack(side="left", padx=(0, 5))

    btn_launch = ttk.Button(btn_frame, text=get_t("btn_launch"), style="Launch.TButton")
    btn_launch.pack(side="right")

    # Actions
    def do_launch():
        root.destroy()
        launch_server(platform_info)

    def check_deps():
        lbl_status.config(text=get_t("status_checking"))
        btn_check.state(["disabled"])
        btn_launch.state(["disabled"])
        add_log("--- " + get_t("status_checking") + " ---")
        
        installed, missing = check_all_requirements(packages)
        add_log(f"Installed: {len(installed)}, Missing: {len(missing)}")

        if missing:
            pip_path = platform_info["pip"]
            if not pip_path:
                add_log(get_t("no_pip"))
                return
            lbl_status.config(text=get_t("status_installing"))
            for pkg in missing:
                add_log(f"Installing {pkg['display_name']}...")
                cmd = [pip_path, "install", pkg["pip_name"], "-q", "--no-warn-script-location"]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    add_log(f"OK: {pkg['display_name']}")
                except Exception as e:
                    add_log(f"ERR: {e}")
            cfg["first_run"] = False
            save_config(cfg)
        
        lbl_status.config(text=get_t("status_done"))
        btn_check.state(["!disabled"])
        btn_launch.state(["!disabled"])
        add_log("--- " + get_t("status_done") + " ---")

    def check_updates():
        messagebox.showinfo("Updates", get_t("msg_updates_soon"))

    def toggle_lang():
        cfg["lang"] = "en" if cfg["lang"] == "pl" else "pl"
        save_config(cfg)
        # Update UI texts
        lbl_title.config(text=get_t("title"))
        btn_lang.config(text=get_t("btn_lang"))
        btn_launch.config(text=get_t("btn_launch"))
        btn_check.config(text=get_t("btn_check_deps"))
        btn_update.config(text=get_t("btn_updates"))
        if "ready" in lbl_status.cget("text") or "Gotowy" in lbl_status.cget("text"):
            lbl_status.config(text=get_t("status_ready"))

    btn_launch.config(command=do_launch)
    btn_check.config(command=check_deps)
    btn_update.config(command=check_updates)
    btn_lang.config(command=toggle_lang)

    # Auto-check on first run
    if cfg["first_run"]:
        root.after(500, check_deps)
    else:
        add_log("Ready to launch. All dependencies were checked previously.")

    root.mainloop()
    return True

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    platform_info = detect_platform()
    req_path = os.path.join(WORKSPACE, REQUIREMENTS_FILE)
    packages = parse_requirements(req_path)


    if not run_gui(platform_info, packages):
        print("Brak GUI (Tkinter) - uruchamiam bezpośrednio z konsoli...")
        if cfg["first_run"]:
            print("Zalecana weryfikacja zalezności na pierwszym uruchomieniu!")

            cfg["first_run"] = False
            save_config(cfg)
        launch_server(platform_info)

if __name__ == "__main__":
    main()
