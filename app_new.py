#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 1. Set working directory to project root
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE)

# 2. Auto-switch to .venv python if available and not already active
venv_python = os.path.join(WORKSPACE, ".venv", "bin", "python3")
if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
    print(f"🔄 Auto-activating project virtual environment: {venv_python}")
    os.execv(venv_python, [venv_python] + sys.argv)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import warnings
warnings.filterwarnings("ignore")

from modules.app import create_app

app = create_app()

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print("🚀 Starting ViDubb Pro Flask Server...")
    print("   URL: http://127.0.0.1:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
