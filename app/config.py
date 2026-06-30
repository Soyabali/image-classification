"""Central application settings.

All paths are resolved relative to the project root so the app behaves the
same no matter which directory the server is launched from.
"""

import os
from pathlib import Path

# Project root = one level above the `app` package.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    # API metadata
    APP_TITLE = "Weather Image Classification API"
    APP_DESCRIPTION = "Classify weather conditions in images and videos using YOLOv11"
    APP_VERSION = "1.0.0"

    # Model
    MODEL_PATH = str(BASE_DIR / "models" / "best.onnx")

    # Where annotated images/videos are written and served from
    RESULTS_DIR = str(BASE_DIR / "results")


settings = Settings()

# Make sure the results directory exists at import time.
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
