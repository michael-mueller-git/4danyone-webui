"""Private subprocess entry point for motion inference.

Drop-in replacement for the upstream ``fdanyone/motion/worker.py`` that
dispatches to the GVHMR or SMPLer-X motion stage via the ``MOTION_BACKEND``
environment variable (``gvhmr`` | ``smplerx``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fdanyone.device import select_cuda_device
from fdanyone.video import load_canonical_working_clip


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text())
    device, _ = select_cuda_device(request["device"])
    clip = load_canonical_working_clip(request["working_video"], request["clip_metadata"])
    common = {
        "clip": clip,
        "working_video": request["working_video"],
        "output_dir": request["output_dir"],
        "device": device,
    }
    backend = os.environ.get("MOTION_BACKEND", "smplerx").strip().lower()
    if backend == "smplerx":
        from fdanyone_smplerx.runner import run_smplerx

        result = run_smplerx(gvhmr_root=request["gvhmr_root"], **common)
    else:
        from fdanyone.motion.gvhmr import run_gvhmr

        result = run_gvhmr(gvhmr_root=request["gvhmr_root"], **common)
    result.save(request["result_dir"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m fdanyone.motion.worker REQUEST.json")
    main(sys.argv[1])
