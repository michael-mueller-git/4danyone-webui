"""Headless SMPL-X provisioning for the Docker image.

SMPL-X is a separately licensed body model; the official installer in the
upstream repo is interactive and cannot run in a container. This script
installs the neutral model non-interactively, preferring, in order:

  1. the SMPLX_SOURCE env var (path to models_smplx_v1_1.zip or SMPLX_NEUTRAL.npz),
  2. an archive / NPZ dropped into $DATA_DIR/smplx,
  3. a public Hugging Face mirror of the official archive.

It reuses upstream ``fdanyone.download.install_smplx`` so the installed layout
and the GVHMR compatibility links match the official installer exactly.

    python scripts/provision_smplx.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
GVHMR_ROOT = Path(os.environ.get("GVHMR_ROOT", "/app/third_party/GVHMR"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

SMPLX_SOURCE = os.environ.get("SMPLX_SOURCE")
SMPLX_MIRROR_REPO = os.environ.get("SMPLX_MIRROR_REPO", "liujiting/models_smplx_v1_1")
SMPLX_MIRROR_FILE = os.environ.get("SMPLX_MIRROR_FILE", "models_smplx_v1_1.zip")
DROP_DIR = DATA_DIR / "smplx"


def _discover_drop() -> Path | None:
    if not DROP_DIR.is_dir():
        return None
    candidates = sorted(DROP_DIR.glob("*.zip")) + sorted(DROP_DIR.glob("SMPLX_NEUTRAL.npz"))
    return candidates[0] if candidates else None


def main() -> int:
    from fdanyone import assets
    from fdanyone.download import install_smplx

    target = MODEL_DIR / assets.SMPLX_MODEL
    if target.is_file():
        print(f"SMPL-X already present at {target}")
        return 0

    source: Path | None = None
    if SMPLX_SOURCE:
        candidate = Path(SMPLX_SOURCE).expanduser().resolve()
        if candidate.is_file():
            source = candidate
            print(f"Using SMPLX_SOURCE={candidate}")
    if source is None:
        dropped = _discover_drop()
        if dropped is not None:
            source = dropped
            print(f"Using dropped archive {dropped}")

    if source is not None:
        install_smplx(source, str(MODEL_DIR), str(GVHMR_ROOT))
        print(f"Installed SMPL-X from {source} -> {target}")
        return 0

    print(f"Downloading SMPL-X neutral from public mirror {SMPLX_MIRROR_REPO} ...")
    try:
        from huggingface_hub import hf_hub_download

        mirror = hf_hub_download(
            repo_id=SMPLX_MIRROR_REPO,
            filename=SMPLX_MIRROR_FILE,
            repo_type="dataset",
        )
    except Exception as exc:
        print(
            "SMPL-X mirror download failed. Download models_smplx_v1_1.zip from "
            f"https://smpl-x.is.tue.mpg.de/ (free account + license) and either set "
            f"SMPLX_SOURCE or drop it into {DROP_DIR}. Mirror error: {exc}",
            file=sys.stderr,
        )
        return 1

    install_smplx(Path(mirror), str(MODEL_DIR), str(GVHMR_ROOT))
    print(f"Installed SMPL-X from mirror -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
