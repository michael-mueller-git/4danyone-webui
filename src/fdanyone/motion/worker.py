"""Private subprocess entry point for motion inference.

Drop-in replacement for the upstream ``fdanyone/motion/worker.py`` that
dispatches to the GVHMR or SMPLer-X motion stage via the ``MOTION_BACKEND``
environment variable (``gvhmr`` | ``smplerx``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from fdanyone.device import select_cuda_device
from fdanyone.video import load_canonical_working_clip

LOGGER = logging.getLogger("fdanyone.motion")


def _configure_logging() -> None:
    """Emit worker logs to stderr (captured by the official run log)."""

    logging.basicConfig(
        level=os.environ.get("FDANYONE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _runtime_context() -> str:
    try:
        import torch

        cuda = torch.version.cuda or "n/a"
        return f"python={sys.version.split()[0]} torch={torch.__version__} cuda={cuda}"
    except Exception as exc:  # noqa: BLE001 - context only
        return f"python={sys.version.split()[0]} torch=unavailable ({exc})"


def main(request_path: str) -> None:
    _configure_logging()
    started = time.monotonic()
    request = json.loads(Path(request_path).read_text())
    backend = os.environ.get("MOTION_BACKEND", "smplerx").strip().lower()
    LOGGER.info("motion worker: start (backend=%s, %s).", backend, _runtime_context())
    LOGGER.info(
        "motion worker: request=%s video=%s gvhmr_root=%s result_dir=%s",
        request_path,
        request["working_video"],
        request["gvhmr_root"],
        request["result_dir"],
    )
    try:
        device, _ = select_cuda_device(request["device"])
        LOGGER.info("motion worker: device=%s; loading canonical clip.", device)
        clip = load_canonical_working_clip(request["working_video"], request["clip_metadata"])
        LOGGER.info("motion worker: clip loaded (%d frames).", len(clip.frames))
        common = {
            "clip": clip,
            "working_video": request["working_video"],
            "output_dir": request["output_dir"],
            "device": device,
        }
        if backend == "smplerx":
            from fdanyone_smplerx.runner import run_smplerx

            result = run_smplerx(gvhmr_root=request["gvhmr_root"], **common)
        elif backend == "prompthmr":
            from fdanyone_prompthmr.runner import run_prompthmr

            result = run_prompthmr(gvhmr_root=request["gvhmr_root"], **common)
        else:
            from fdanyone.motion.gvhmr import run_gvhmr

            result = run_gvhmr(gvhmr_root=request["gvhmr_root"], **common)
        result.save(request["result_dir"])
        LOGGER.info(
            "motion worker: done in %.1fs; saved result to %s.",
            time.monotonic() - started,
            request["result_dir"],
        )
    except Exception:
        LOGGER.exception("motion worker: failed after %.1fs.", time.monotonic() - started)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m fdanyone.motion.worker REQUEST.json")
    main(sys.argv[1])
