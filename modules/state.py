# -*- coding: utf-8 -*-
"""
Global state and project database management
"""
import os
import json
import time
import pandas as pd
from typing import Dict, List, Optional
from modules.config import PROJECTS_DIR, RESULTS_DIR

class Project:
    def __init__(self, name: str):
        self.name = name
        self.video_path = ""
        self.subtitles_path = ""
        self.source_lang = "auto"
        self.target_lang = "Polish"
        self.whisper_model = "turbo"
        self.hf_token = ""
        self.ollama_model = ""
        self.temperature = 0.1
        self.prompt = "You are a professional movie subtitle translator. Translate from {source_lang} to {target_lang}. Return ONLY the direct dialogue line translation. NEVER include notes, brackets, parentheses (e.g. no (zakończenie)), explanations, or untranslated characters. Do NOT wrap output in quotes."

        self.dub_lang = "Polish"
        self.output_mode = "dubbing"
        self.tts_engine = "edge"
        self.voice = "Default"
        self.keep_bg = True
        self.hardsub = False
        self.validation_model = "None"
        self.ai_endpoint = "http://127.0.0.1:11434"
        self.ai_provider = "ollama"
        self.auto_retry_count = 10
        self.audio_enhance = False
        self.enhance_method = "dsp_denoise"
        self.output_video_path = os.path.join(RESULTS_DIR, f"{name}_output.mp4")
        self.subtitles: List[Dict] = []
        self.context = ""
        self.logs: List[str] = []
        self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "video_path": self.video_path,
            "subtitles_path": self.subtitles_path,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "whisper_model": self.whisper_model,
            "hf_token": self.hf_token,
            "ollama_model": self.ollama_model,
            "temperature": self.temperature,
            "prompt": self.prompt,
            "dub_lang": self.dub_lang,
            "output_mode": self.output_mode,
            "tts_engine": self.tts_engine,
            "voice": self.voice,
            "keep_bg": self.keep_bg,
            "hardsub": self.hardsub,
            "validation_model": self.validation_model,
            "ai_endpoint": self.ai_endpoint,
            "ai_provider": self.ai_provider,
            "auto_retry_count": self.auto_retry_count,
            "audio_enhance": self.audio_enhance,
            "enhance_method": self.enhance_method,
            "output_video_path": self.output_video_path,
            "subtitles": self.subtitles,
            "context": self.context,
            "logs": self.logs,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    def from_dict(self, data: dict):
        self.video_path = data.get("video_path", "")
        self.subtitles_path = data.get("subtitles_path", "")
        self.source_lang = data.get("source_lang", "auto")
        self.target_lang = data.get("target_lang", "Polish")
        self.whisper_model = data.get("whisper_model", "turbo")
        self.hf_token = data.get("hf_token", "")
        self.ollama_model = data.get("ollama_model", "")
        self.temperature = data.get("temperature", 0.1)
        self.prompt = data.get("prompt", self.prompt)
        self.dub_lang = data.get("dub_lang", "Polish")
        self.output_mode = data.get("output_mode", "dubbing")
        self.tts_engine = data.get("tts_engine", "edge")
        self.voice = data.get("voice", "Default")
        self.keep_bg = data.get("keep_bg", True)
        self.hardsub = data.get("hardsub", False)
        self.validation_model = data.get("validation_model", "None")
        self.ai_endpoint = data.get("ai_endpoint", "http://127.0.0.1:11434")
        self.ai_provider = data.get("ai_provider", "ollama")
        self.auto_retry_count = int(data.get("auto_retry_count", 10))
        self.audio_enhance = bool(data.get("audio_enhance", False))
        self.enhance_method = data.get("enhance_method", "dsp_denoise")
        self.output_video_path = data.get("output_video_path", os.path.join(RESULTS_DIR, f"{self.name}_output.mp4"))

        self.subtitles = data.get("subtitles", [])
        self.context = data.get("context", "")
        self.logs = data.get("logs", [])
        self.created_at = data.get("created_at", time.time())
        self.updated_at = data.get("updated_at", time.time())

    def save(self):
        self.updated_at = time.time()
        filepath = os.path.join(PROJECTS_DIR, f"{self.name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    def add_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logs.append(line)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]
        print(line)
        self.save()


class AppState:
    def __init__(self):
        self.active_project: Optional[Project] = None
        self.cancel_flags = {
            "transcribe": False,
            "translate": False,
            "dubbing": False
        }
        self.translate_done = True
        self.translate_progress = 0
        self.translate_total = 0

    def add_log(self, msg: str):
        if self.active_project:
            self.active_project.add_log(msg)
        else:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}")

    def get_logs(self, since: int = 0) -> List[str]:
        if self.active_project:
            return self.active_project.logs[since:]
        return []

    def get_total_logs(self) -> int:
        if self.active_project:
            return len(self.active_project.logs)
        return 0

    def get_df(self) -> pd.DataFrame:
        if self.active_project and self.active_project.subtitles:
            return pd.DataFrame(self.active_project.subtitles)
        return pd.DataFrame()

    def set_df(self, df: pd.DataFrame):
        if self.active_project:
            self.active_project.subtitles = df.where(pd.notna(df), None).to_dict(orient='records')
            self.active_project.save()

    def list_projects(self) -> List[Dict]:
        projects = []
        if not os.path.exists(PROJECTS_DIR):
            return []
        for filename in os.listdir(PROJECTS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(PROJECTS_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        projects.append({
                            "name": data.get("name", filename[:-5]),
                            "updated_at": data.get("updated_at", 0),
                            "created_at": data.get("created_at", 0),
                            "video_path": data.get("video_path", "")
                        })
                except Exception:
                    pass
        projects.sort(key=lambda x: x["updated_at"], reverse=True)
        return projects

    def load_project(self, name: str) -> Project:
        filepath = os.path.join(PROJECTS_DIR, f"{name}.json")
        project = Project(name)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                project.from_dict(data)
        else:
            project.save()
        self.active_project = project
        return project

    def delete_project(self, name: str) -> bool:
        filepath = os.path.join(PROJECTS_DIR, f"{name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            if self.active_project and self.active_project.name == name:
                self.active_project = None
            return True
        return False

state = AppState()
