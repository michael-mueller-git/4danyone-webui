"""Inspect 4DAnyone job outputs and package a downloadable archive.

The pipeline writes one output directory per clip:
    out/
    ├── metadata.json
    ├── cameras.json
    ├── gvhmr/                     (motion.json, motion.safetensors)
    ├── skeletons/00.mp4 ...
    └── videos/
        ├── sparse/{...}.mp4        (RCP proposal views, when used)
        └── dense/00.mp4 ...        (generated target views)
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from app import config


def _sorted_videos(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return [str(path) for path in sorted(directory.glob("*.mp4"))]


def job_outputs(out_dir: str | Path) -> dict:
    root = Path(out_dir).expanduser().resolve()
    videos = root / "videos"
    gvhmr = root / "gvhmr"
    motion_safetensors = gvhmr / "motion.safetensors"
    motion_json = gvhmr / "motion.json"
    return {
        "exists": root.is_dir(),
        "out_dir": str(root),
        "dense": _sorted_videos(videos / "dense"),
        "sparse": _sorted_videos(videos / "sparse"),
        "skeletons": _sorted_videos(root / "skeletons"),
        "metadata": str(root / "metadata.json") if (root / "metadata.json").is_file() else None,
        "cameras": str(root / "cameras.json") if (root / "cameras.json").is_file() else None,
        "motion_safetensors": str(motion_safetensors) if motion_safetensors.is_file() else None,
        "motion_json": str(motion_json) if motion_json.is_file() else None,
    }


def build_zip(job_id: int, out_dir: str | Path, force: bool = False) -> str | None:
    """Package every generated view + metadata into one zip; returns its path."""

    from app.jobs import log_path

    root = Path(out_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    try:
        name = (root / "metadata.json").read_text()
    except Exception:
        name = "clip"
    try:
        import json

        clip_name = json.loads(name).get("source_path", "clip")
        clip_name = Path(clip_name).stem or "clip"
    except Exception:
        clip_name = "clip"

    destination_dir = config.GALLERIES_DIR / str(job_id)
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive = destination_dir / f"{clip_name}_all_views.zip"
    if archive.exists() and not force:
        return str(archive)

    temporary = archive.with_suffix(".zip.part")
    if temporary.exists():
        temporary.unlink()

    entries = job_outputs(root)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        groups = {
            "dense": entries["dense"],
            "sparse": entries["sparse"],
            "skeletons": entries["skeletons"],
            "cameras.json": [entries["cameras"]] if entries["cameras"] else [],
            "metadata.json": [entries["metadata"]] if entries["metadata"] else [],
            "gvhmr/motion.safetensors": [entries["motion_safetensors"]] if entries["motion_safetensors"] else [],
            "gvhmr/motion.json": [entries["motion_json"]] if entries["motion_json"] else [],
        }
        for prefix, paths in groups.items():
            if prefix in {"cameras.json", "metadata.json", "gvhmr/motion.safetensors", "gvhmr/motion.json"}:
                for path in paths:
                    handle.write(path, f"{clip_name}/{prefix}")
                continue
            for path in paths:
                source = Path(path)
                # videos/dense/03.mp4 -> <clip>/videos/dense/03.mp4
                relative = source.relative_to(root)
                handle.write(source, f"{clip_name}/{relative.as_posix()}")

    temporary.replace(archive)
    return str(archive)


def prune(zip_count_limit: int = 40) -> None:
    """Keep a bounded number of archive copies to avoid filling the volume."""

    if not config.GALLERIES_DIR.is_dir():
        return
    archives = sorted(config.GALLERIES_DIR.glob("*/**/*.zip"), key=lambda path: path.stat().st_mtime)
    if len(archives) <= zip_count_limit:
        return
    for stale in archives[: len(archives) - zip_count_limit]:
        try:
            stale.unlink()
        except OSError:
            pass


def disk_usage_bytes() -> int:
    try:
        return shutil.disk_usage(config.DATA_DIR).used
    except OSError:
        return 0
