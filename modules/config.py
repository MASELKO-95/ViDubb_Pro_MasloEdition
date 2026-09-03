# -*- coding: utf-8 -*-
"""
Configuration constants for viDubb Pro — Maslo95 Edition
"""
import os

APP_NAME = "viDubb Pro — Maslo95 Edition"
APP_VERSION = "1.0.1a"

# Language Mapping from display name to ISO 639-1 code
LANGUAGE_MAPPING = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Turkish': 'tr',
    'Russian': 'ru',
    'Polish': 'pl',
    'Dutch': 'nl',
    'Czech': 'cs',
    'Arabic': 'ar',
    'Chinese (Simplified)': 'zh-cn',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Hindi': 'hi'
}

# Upload directories
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(WORKSPACE_DIR, "uploads")
PROJECTS_DIR = os.path.join(WORKSPACE_DIR, "projects")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")

# Ensure required directories exist
for folder in [UPLOAD_FOLDER, PROJECTS_DIR, RESULTS_DIR]:
    os.makedirs(folder, exist_ok=True)

MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB
