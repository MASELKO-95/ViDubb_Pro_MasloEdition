# -*- coding: utf-8 -*-
"""
Blueprint for project database management (DaVinci Resolve style)
"""
import os
from flask import Blueprint, jsonify, request
from modules.state import state, Project
from modules.config import PROJECTS_DIR

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
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Nazwa projektu jest wymagana"}), 400
    safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).strip()
    if not safe_name:
        return jsonify({"error": "Nieprawidłowa nazwa projektu"}), 400
    state.load_project(safe_name)
    state.add_log(f"📁 Utworzono i wczytano nowy projekt: {safe_name}")
    return jsonify({"success": True, "project": state.active_project.to_dict()})

@projects_bp.route("/api/projects/load", methods=["POST"])
def load_project_api():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Nazwa projektu jest wymagana"}), 400
    filepath = os.path.join(PROJECTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        return jsonify({"error": "Projekt nie istnieje"}), 404
    state.load_project(name)
    state.add_log(f"📁 Wczytano projekt: {name}")
    return jsonify({"success": True, "project": state.active_project.to_dict()})

@projects_bp.route("/api/projects/save", methods=["POST"])
def save_project_api():
    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 400
    data = request.get_json() or {}
    for field in ["video_path", "subtitles_path", "source_lang", "target_lang",
                  "whisper_model", "hf_token", "ollama_model", "temperature",
                  "prompt", "dub_lang", "output_mode", "tts_engine", "voice",
                  "keep_bg", "hardsub", "validation_model", "output_video_path", "context",
                  "ai_endpoint", "ai_provider", "auto_retry_count", "audio_enhance", "enhance_method"]:
        if field in data:
            setattr(state.active_project, field, data[field])

    if "subtitles" in data:
        state.active_project.subtitles = data["subtitles"]
    state.active_project.save()
    state.add_log(f"💾 Projekt '{state.active_project.name}' został zapisany.")
    return jsonify({"success": True})

@projects_bp.route("/api/projects/delete", methods=["POST"])
def delete_project_api():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Nazwa projektu jest wymagana"}), 400
    success = state.delete_project(name)
    if success:
        state.add_log(f"🗑️ Usunięto projekt: {name}")
        return jsonify({"success": True})
    return jsonify({"error": "Projekt nie został znaleziony"}), 404
