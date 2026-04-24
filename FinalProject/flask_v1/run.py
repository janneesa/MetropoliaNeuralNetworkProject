from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
FINAL_PROJECT_DIR = PACKAGE_DIR.parent
if str(FINAL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_PROJECT_DIR))

from flask_v1.app import app


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
