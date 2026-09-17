"""Download the SMPLer-X-H32 body regressor checkpoint.

Places ``smpler_x_h32_correct.pth.tar`` under $MODEL_DIR/smplerx so the
SMPLer-X motion stage can find it. Also runs as part of AUTO_DOWNLOAD_MODELS.

    python scripts/download_smplerx.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))


def main() -> int:
    from fdanyone_smplerx.loader import checkpoint_path, resolve_checkpoint

    target = checkpoint_path()
    if target.is_file():
        print(f"SMPLer-X checkpoint already present at {target}")
        return 0
    print(f"Downloading SMPLer-X checkpoint to {target} ...")
    downloaded = resolve_checkpoint()
    print(f"Installed SMPLer-X checkpoint -> {downloaded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
