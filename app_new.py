import sys
import os
import logging

# Imports are intentionally staged so environment and logging configuration run first.
# ruff: noqa: E402

# ============================================================
# PROJECT ROOT
# ============================================================

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE)


# ============================================================
# AUTO VENV
# ============================================================

venv_python = os.path.join(
    WORKSPACE,
    ".venv",
    "bin",
    "python3",
)

if (
    os.path.exists(venv_python)
    and os.path.abspath(sys.executable)
    != os.path.abspath(venv_python)
):
    print(
        "🔄 Auto-activating project virtual environment: "
        f"{venv_python}"
    )
    os.execv(
        venv_python,
        [venv_python] + sys.argv,
    )


# ============================================================
# ENV / WARNINGS
# ============================================================

os.environ[
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
] = "python"
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import warnings

warnings.filterwarnings("ignore")


# ============================================================
# QUIET HTTP ACCESS LOGS
# ============================================================

# Default:
#   no spam such as:
#   127.0.0.1 - - [...] "GET /api/voices HTTP/1.1" 200 -
#
# To temporarily restore HTTP request logs:
#
#   VIDUBB_HTTP_LOGS=1 python3 app_new.py
#
SHOW_HTTP_ACCESS_LOGS = (
    os.environ.get(
        "VIDUBB_HTTP_LOGS",
        "0",
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


if not SHOW_HTTP_ACCESS_LOGS:
    # Keep real errors, hide normal INFO request traffic.
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.ERROR)
    werkzeug_logger.propagate = False


from werkzeug.serving import WSGIRequestHandler


class ViDubbRequestHandler(WSGIRequestHandler):
    """
    Werkzeug request handler that suppresses normal access logs.

    Flask/application errors are NOT suppressed by this.
    """

    def log_request(
        self,
        code="-",
        size="-",
    ):
        if SHOW_HTTP_ACCESS_LOGS:
            super().log_request(
                code,
                size,
            )

    def log_error(
        self,
        format,
        *args,
    ):
        # Keep actual HTTP/server errors visible.
        super().log_error(
            format,
            *args,
        )


# ============================================================
# FLASK APP
# ============================================================

from modules.app import create_app
from modules.config import APP_VERSION

app = create_app()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(
        "templates",
        exist_ok=True,
    )
    os.makedirs(
        "static",
        exist_ok=True,
    )

    print(
        "🚀 Starting viDubb Pro — "
        f"Maslo95 Edition v{APP_VERSION} Flask Server..."
    )
    print(
        "   URL: http://127.0.0.1:7860"
    )

    if SHOW_HTTP_ACCESS_LOGS:
        print(
            "   🧪 HTTP access logs: ON"
        )
    else:
        print(
            "   🔇 HTTP access logs: OFF "
            "(set VIDUBB_HTTP_LOGS=1 to enable)"
        )

    host = os.environ.get("VIDUBB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    app.run(
        host=host,
        port=7860,
        debug=False,
        request_handler=ViDubbRequestHandler,
    )
