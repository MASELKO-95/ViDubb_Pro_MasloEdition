import os
import platform
import re
import sys
import warnings
from pathlib import Path
# TTS compatibility patches must run before importing Flask routes and services.
# ruff: noqa: E402
warnings.filterwarnings("ignore")

# ====================== XTTS SAFE FIX (must be before any TTS import) ======================
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.tts.utils.speakers import SpeakerManager
    from torch.serialization import add_safe_globals
    add_safe_globals([XttsConfig, BaseDatasetConfig, XttsAudioConfig, XttsArgs, SpeakerManager])
except Exception:
    pass

import requests
from flask import Flask, Response, jsonify, render_template, request
from modules.config import APP_VERSION, MAX_CONTENT_LENGTH, LANGUAGE_MAPPING
from modules.state import state
from modules.routes.projects import projects_bp
from modules.routes.video import video_bp
from modules.routes.translate import translate_bp
from modules.routes.dubbing import dubbing_bp

def _redact_diagnostic_text(value: str) -> str:
    text = str(value or "")
    home = str(Path.home())
    if home:
        text = text.replace(home, "<HOME>")
    text = re.sub(r"hf_[A-Za-z0-9]{10,}", "hf_<REDACTED>", text)
    text = re.sub(r"github_pat_[A-Za-z0-9_]{10,}", "github_pat_<REDACTED>", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|token|authorization)\s*[:=]\s*)\S+",
        r"\1<REDACTED>",
        text,
    )
    return text

def get_ollama_models_list(custom_endpoint: str = None) -> list:
    urls_to_try = []
    if custom_endpoint and custom_endpoint.strip():
        ep = custom_endpoint.strip().rstrip('/')
        urls_to_try.extend([f"{ep}/api/tags", f"{ep}/v1/models"])
    if state.active_project and getattr(state.active_project, "ai_endpoint", None):
        ep = state.active_project.ai_endpoint.strip().rstrip('/')
        if f"{ep}/api/tags" not in urls_to_try:
            urls_to_try.extend([f"{ep}/api/tags", f"{ep}/v1/models"])
    urls_to_try.extend([
        "http://127.0.0.1:11434/api/tags",
        "http://localhost:11434/api/tags"
    ])

    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=2.5)
            if r.status_code == 200:
                data = r.json()
                if "models" in data:
                    models = [m['name'] for m in data.get('models', []) if 'name' in m]
                    if models:
                        return models
                elif "data" in data:  # OpenAI / llama.cpp format
                    models = [m['id'] for m in data.get('data', []) if 'id' in m]
                    if models:
                        return models
        except Exception:
            pass
    return ["microai/suzume-llama3:latest", "deepseek-r1:8b"]


def check_ollama_connected(custom_endpoint: str = None) -> bool:
    urls_to_try = []
    if custom_endpoint and custom_endpoint.strip():
        ep = custom_endpoint.strip().rstrip('/')
        urls_to_try.extend([f"{ep}/api/tags", f"{ep}/v1/models"])
    if state.active_project and getattr(state.active_project, "ai_endpoint", None):
        ep = state.active_project.ai_endpoint.strip().rstrip('/')
        if f"{ep}/api/tags" not in urls_to_try:
            urls_to_try.extend([f"{ep}/api/tags", f"{ep}/v1/models"])
    urls_to_try.extend([
        "http://127.0.0.1:11434/api/tags",
        "http://localhost:11434/api/tags"
    ])

    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False



def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
    )
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

    app.register_blueprint(projects_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(translate_bp)
    app.register_blueprint(dubbing_bp)

    @app.route("/")
    def index():
        languages = list(LANGUAGE_MAPPING.keys())
        ollama_models = get_ollama_models_list()
        if not state.active_project:
            state.load_project("Default_Project")
        return render_template(
            "index.html",
            languages=languages,
            ollama_models=ollama_models
        )

    @app.route("/api/ollama/status")
    def api_ollama_status():
        ep = request.args.get("endpoint", "").strip()
        connected = check_ollama_connected(ep)
        models = get_ollama_models_list(ep) if connected else []
        return jsonify({"connected": connected, "models": models})


    @app.route("/api/ollama/logs")
    def api_ollama_logs():
        try:
            since = max(0, int(request.args.get("since", 0)))
        except (TypeError, ValueError):
            return jsonify({"error": "Parametr 'since' musi być liczbą całkowitą."}), 400
        return jsonify({
            "logs": state.get_logs(since=since),
            "total": state.get_total_logs()
        })

    @app.route("/api/diagnostics/report")
    def diagnostic_report():
        project_name = (
            state.active_project.name
            if state.active_project
            else "none"
        )
        logs = state.get_logs(max(0, state.get_total_logs() - 300))
        report_lines = [
            "viDubb Pro diagnostic report",
            f"Version: {APP_VERSION}",
            f"Python: {sys.version.split()[0]}",
            f"Platform: {platform.platform()}",
            f"Project: {project_name}",
            "",
            "Last application logs:",
            *logs,
        ]
        report = _redact_diagnostic_text("\n".join(report_lines)) + "\n"
        return Response(
            report,
            mimetype="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": (
                    "attachment; filename=vidubb-diagnostic-report.txt"
                )
            },
        )

    return app
