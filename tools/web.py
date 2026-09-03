#!/usr/bin/env python3
"""
Interfejs Web (Flask) do budowania datasetu głosowego z opcją czyszczenia tła (Demucs).
"""
import os
import sys
import threading
import time
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

try:
    from voice_dataset_builder import build_dataset, BuildConfig, PROJECT_ROOT
except ImportError:
    print("BŁĄD: Nie znaleziono voice_dataset_builder.py. Upewnij się, że oba pliki są w tym samym folderze.")
    sys.exit(1)

app = Flask(__name__, static_folder=str(PROJECT_ROOT))

process_state = {
    "running": False,
    "logs": [],
    "progress": 0,
    "message": "Gotowy."
}

def log_wrapper(text):
    timestamp = time.strftime("%H:%M:%S")
    msg = f"[{timestamp}] {text}"
    process_state["logs"].append(msg)
    if len(process_state["logs"]) > 50:
        process_state["logs"] = process_state["logs"][-50:]
    print(msg)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def get_status():
    return jsonify(process_state)

@app.route('/api/start', methods=['POST'])
def start_build():
    if process_state["running"]:
        return jsonify({"error": "Proces już trwa!"}), 400

    data = request.json
    name = data.get('name', 'Unknown_Voice')
    inputs_str = data.get('inputs', '')
    has_reference = data.get('has_reference', False)
    reference_path = data.get('reference_path', '') if has_reference else None
    clean_vocals = data.get('clean_vocals', False)  # NOWE: Flaga Demucs

    if not name or not inputs_str:
        return jsonify({"error": "Podaj nazwę i ścieżkę do nagrań."}), 400

    input_files = [Path(p.strip()) for p in inputs_str.split(';') if p.strip()]
    ref_path = Path(reference_path) if reference_path else None

    def worker():
        process_state["running"] = True
        process_state["logs"] = ["Rozpoczynanie procesu..."]
        process_state["progress"] = 0

        try:
            config = BuildConfig(
                name=name,
                inputs=input_files,
                reference=ref_path,
                language="auto",
                whisper_model="turbo",
                clean_vocals=clean_vocals  # NOWE: Przekazanie flagi do BuildConfig
            )
            build_dataset(config, log=log_wrapper)
            process_state["message"] = "Zakończono sukcesem! Sprawdź folder voice_datasets."
            process_state["progress"] = 100
        except Exception as e:
            process_state["message"] = f"Błąd: {str(e)}"
            log_wrapper(f"BŁĄD KRYTYCZNY: {e}")
        finally:
            process_state["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>ViDubb Web Builder</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f0f2f5; color: #333; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #555; }
        input[type="text"] { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box; font-size: 14px; }
        input[type="text"]:focus { border-color: #3498db; outline: none; }
        .hint { font-size: 12px; color: #777; margin-top: 5px; }
        .checkbox-wrapper { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; padding: 10px; background: #e8f4f8; border-radius: 6px; }
        input[type="checkbox"] { transform: scale(1.3); }
        button { background: #27ae60; color: white; border: none; padding: 15px; width: 100%; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; transition: background 0.3s; }
        button:hover { background: #219150; }
        button:disabled { background: #95a5a6; cursor: not-allowed; }
        #console { background: #1e1e1e; color: #00ff00; font-family: 'Consolas', monospace; padding: 15px; height: 300px; overflow-y: auto; border-radius: 6px; margin-top: 20px; font-size: 13px; line-height: 1.4; }
        .log-line { margin-bottom: 4px; border-bottom: 1px solid #333; padding-bottom: 2px; }
        #status { text-align: center; margin-top: 10px; font-weight: bold; color: #2980b9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ ViDubb Web Builder</h1>
        <p style="text-align:center; color:#666;">Automatyczne dzielenie nagrań i czyszczenie głosu pod XTTS.</p>

        <div class="form-group">
            <label>Nazwa osoby (ID):</label>
            <input type="text" id="name" placeholder="np. Yuuki_Takada" value="MojaPostac">
        </div>

        <div class="form-group">
            <label>Ścieżka do folderu z nagraniami (lub pojedynczy plik):</label>
            <input type="text" id="inputPath" placeholder="np. C:/Users/Name/Music/Skladanka.mp4 lub ./nagrania">
            <div class="hint">Wklej pełną ścieżkę do folderu/pliku.</div>
        </div>

        <div class="checkbox-wrapper">
            <input type="checkbox" id="useRef" onchange="toggleRef()">
            <label for="useRef" style="margin:0; font-weight:normal; cursor:pointer;">Posiadam plik wzorcowy (referencyjny)</label>
        </div>

        <div class="form-group" id="refGroup" style="display:none;">
            <label>Ścieżka do pliku wzorcowego (.wav):</label>
            <input type="text" id="refPath" placeholder="np. C:/Users/Name/Wzorzec.wav">
        </div>

        <div class="checkbox-wrapper" style="background: #fff3cd; border: 1px solid #ffeeba;">
            <input type="checkbox" id="cleanVocals">
            <label for="cleanVocals" style="margin:0; font-weight:normal; cursor:pointer;">
                <strong>Wyczyść tło/muzykę (Demucs)</strong> – znacznie wolniejsze, ale daje najczystsze próbki do XTTS.
            </label>
        </div>

        <button id="btnStart" onclick="startBuild()">▶ URUCHOM PRZETWARZANIE</button>

        <div id="status">Status: Oczekiwanie...</div>
        <div id="console"><div class="log-line">System gotowy. Wpisz dane i kliknij Start.</div></div>
    </div>

    <script>
        function toggleRef() {
            document.getElementById('refGroup').style.display = document.getElementById('useRef').checked ? 'block' : 'none';
        }

        async function startBuild() {
            const btn = document.getElementById('btnStart');
            const status = document.getElementById('status');
            const name = document.getElementById('name').value.trim();
            const inputPath = document.getElementById('inputPath').value.trim();
            const useRef = document.getElementById('useRef').checked;
            const refPath = useRef ? document.getElementById('refPath').value.trim() : null;
            const cleanVocals = document.getElementById('cleanVocals').checked;

            if (!name || !inputPath) {
                alert("Proszę wypełnić Nazwę i Ścieżkę do nagrań!");
                return;
            }

            btn.disabled = true;
            btn.textContent = "PRZETWARZANIE W TOKU...";
            status.textContent = "Status: Rozpoczynanie...";
            document.getElementById('console').innerHTML = '';

            try {
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        name: name,
                        inputs: inputPath,
                        has_reference: useRef,
                        reference_path: refPath,
                        clean_vocals: cleanVocals
                    })
                });

                const result = await response.json();
                if (response.ok) {
                    status.textContent = "Status: Przetwarzanie w toku (sprawdzaj logi)...";
                    pollStatus();
                } else {
                    throw new Error(result.error || 'Nieznany błąd');
                }
            } catch (err) {
                document.getElementById('console').innerHTML += `<div class="log-line" style="color:red">BŁĄD: ${err.message}</div>`;
                btn.disabled = false;
                btn.textContent = "▶ URUCHOM PONOWNIE";
                status.textContent = "Status: Błąd.";
            }
        }

        async function pollStatus() {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch('/api/status');
                    const data = await res.json();

                    const consoleDiv = document.getElementById('console');
                    consoleDiv.innerHTML = data.logs.map(l => `<div class="log-line">${l}</div>`).join('');
                    consoleDiv.scrollTop = consoleDiv.scrollHeight;

                    document.getElementById('status').textContent = "Status: " + data.message;

                    if (!data.running) {
                        clearInterval(interval);
                        const btn = document.getElementById('btnStart');
                        btn.disabled = false;
                        btn.textContent = "▶ URUCHOM PONOWNIE";
                        document.getElementById('status').style.color = data.message.includes("Błąd") ? "red" : "green";
                    }
                } catch (e) {
                    console.error(e);
                }
            }, 1000);
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print("="*50)
    print("🚀 Serwer Web uruchomiony!")
    print("🌐 Otwórz w przeglądarce: http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=False, port=5000)
