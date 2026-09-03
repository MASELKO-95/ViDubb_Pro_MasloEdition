from flask import Blueprint, jsonify, request
from modules.state import normalize_project_name, project_file_path, state

projects_bp = Blueprint('projects', __name__)

@projects_bp.route("/api/projects", methods=["GET"])
def get_projects():
    return jsonify({"projects": state.list_projects()})

@projects_bp.route("/api/projects/active", methods=["GET"])
def get_active_project():
    if state.active_project:
        return jsonify({"active": True, "project": state.active_project.to_dict()})
    return jsonify({"active": False})

@projects_bp.route("/api/projects/new", methods=["POST"])
def create_project():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    try:
        safe_name = normalize_project_name(data.get("name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    state.load_project(safe_name)
    state.add_log(f"📁 Created and loaded project: {safe_name}")
    return jsonify({"success": True, "project": state.active_project.to_dict()})

@projects_bp.route("/api/projects/load", methods=["POST"])
def load_project_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    try:
        name = normalize_project_name(data.get("name"))
        filepath = project_file_path(name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not filepath.exists():
        return jsonify({"error": "Project does not exist"}), 404
    state.load_project(name)
    state.add_log(f"📁 Loaded project: {name}")
    return jsonify({"success": True, "project": state.active_project.to_dict()})

@projects_bp.route("/api/projects/save", methods=["POST"])
def save_project_api():
    if not state.active_project:
        return jsonify({"error": "No active project"}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    for field in ["video_path", "subtitles_path", "source_lang", "target_lang",
                  "whisper_model", "hf_token", "num_speakers", "ollama_model", "temperature",
                  "prompt", "dub_lang", "output_mode", "tts_engine", "voice",
                  "keep_bg", "hardsub", "validation_model", "output_video_path", "context",
                  "ai_endpoint", "ai_provider", "auto_retry_count", "audio_enhance", "enhance_method"]:
        if field in data:
            setattr(state.active_project, field, data[field])

    if "subtitles" in data and not isinstance(data["subtitles"], list):
        return jsonify({"error": "Field 'subtitles' must be a list"}), 400
    if "subtitles" in data:
        state.active_project.subtitles = data["subtitles"]
    state.active_project.save()
    state.add_log(f"💾 Saved project: '{state.active_project.name}'.")
    return jsonify({"success": True})

@projects_bp.route("/api/projects/delete", methods=["POST"])
def delete_project_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    try:
        name = normalize_project_name(data.get("name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    success = state.delete_project(name)
    if success:
        state.add_log(f"🗑️ Deleted project: {name}")
        return jsonify({"success": True})
    return jsonify({"error": "Project not found"}), 404
