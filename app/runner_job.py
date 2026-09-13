"""Subprocess entry point that executes one job manifest.

Invoked by jobs.py as ``python -m app.runner_job <id> <manifest>``. All of the
pipeline's generated output goes to ``<job>/out`` and logs stream to
``data/logs/job-<id>.log``.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from app import init_models
from app.jobs import update_job


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.runner_job <job_id> <manifest.json>", file=sys.stderr)
        return 2
    job_id, manifest_path = int(argv[0]), Path(argv[1])
    manifest = json.loads(manifest_path.read_text())
    params = manifest["params"]
    video_path = manifest["video_path"]
    out_dir = manifest["out_dir"]

    print(f"[runner] job {job_id}: {manifest.get('video_name', video_path)}", flush=True)
    print(f"[runner] params: {json.dumps(params)}", flush=True)

    readiness = init_models.is_ready()
    print(f"[runner] asset readiness: {readiness}", flush=True)
    if not readiness.get("models"):
        update_job(job_id, status="failed", error="public models not downloaded yet", finished_at=_now())
        return 1
    if not readiness.get("smplx"):
        update_job(job_id, status="failed", error="SMPL-X not installed", finished_at=_now())
        return 1

    from app.runner import do_run

    try:
        summary = do_run(
            video_path=video_path,
            out_dir=out_dir,
            model_dir=manifest["model_dir"],
            gvhmr_root=manifest["gvhmr_root"],
            params=params,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure back to the DB
        traceback.print_exc()
        update_job(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_now(),
        )
        print(f"[runner] job {job_id} failed: {type(exc).__name__}: {exc}", flush=True)
        return 1

    print(json.dumps({"summary": summary}, indent=2), flush=True)
    update_job(job_id, status="done", finished_at=_now())
    print(f"[runner] job {job_id} completed -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
