"""Central configuration for the 4DAnyone Gradio WebUI.

All paths are overridable through environment variables so the container can
relocate models and data onto Docker volumes without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


REPO_ROOT = Path(os.environ.get("FDANYONE_ROOT", "/app")).expanduser().resolve()
APP_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(REPO_ROOT / "models"))).expanduser().resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data"))).expanduser().resolve()
GVHMR_ROOT = Path(os.environ.get("GVHMR_ROOT", str(REPO_ROOT / "third_party" / "GVHMR"))).expanduser().resolve()

JOBS_DIR = DATA_DIR / "jobs"
UPLOADS_DIR = DATA_DIR / "uploads"
LOGS_DIR = DATA_DIR / "logs"
GALLERIES_DIR = DATA_DIR / "galleries"
INIT_LOG = DATA_DIR / "init.log"
DB_PATH = DATA_DIR / "jobs.db"

HOST = os.environ.get("WEBUI_HOST", "0.0.0.0")
PORT = _as_int(os.environ.get("WEBUI_PORT"), 7860)

# Jobs are serialized within a GPU slot by default. The 4DAnyone pipeline
# already spreads one job over every visible GPU, so keep this at 1 unless a
# GPU is large enough to host multiple simultaneous runs.
MAX_CONCURRENCY = _as_int(os.environ.get("MAX_CONCURRENCY"), 1)

DEFAULT_MODEL_DIR = MODEL_DIR
DEFAULT_GPU_IDS: list[int] | None = None  # None == every visible GPU
DEFAULT_ATTENTION = os.environ.get("DEFAULT_ATTENTION_BACKEND", "auto")
DEFAULT_ENABLE_TURBO = _as_bool(os.environ.get("DEFAULT_ENABLE_TURBO"), True)

AUTO_DOWNLOAD_MODELS = _as_bool(os.environ.get("AUTO_DOWNLOAD_MODELS"))

# SMPL-X provisioning. The neutral body model is licensed separately; the
# WebUI prefers a user-supplied archive, then falls back to a public mirror.
SMPLX_MIRROR_REPO = os.environ.get("SMPLX_MIRROR_REPO", "liujiting/models_smplx_v1_1")
SMPLX_MIRROR_FILE = os.environ.get("SMPLX_MIRROR_FILE", "models_smplx_v1_1.zip")
SMPLX_SOURCE = os.environ.get("SMPLX_SOURCE")  # path to a zip or SMPLX_NEUTRAL.npz
SMPLX_DROP_DIR = DATA_DIR / "smplx"  # drop <anything>.zip / SMPLX_NEUTRAL.npz here

# Attack surface for the WebUI; both must be set to enable authentication.
WEBUI_USERNAME = os.environ.get("WEBUI_USERNAME")
WEBUI_PASSWORD = os.environ.get("WEBUI_PASSWORD")


def ensure_directories() -> None:
    for directory in (JOBS_DIR, UPLOADS_DIR, LOGS_DIR, GALLERIES_DIR, DATA_DIR / "fdanyone", SMPLX_DROP_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
