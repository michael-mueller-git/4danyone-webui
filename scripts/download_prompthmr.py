"""Provision the PromptHMR-Vid checkpoints under ``PROMPTHMR_ROOT``.

PromptHMR hard-codes relative ``data/pretrain/...`` paths, so the weights must
live inside the vendored checkout:

    data/pretrain/phmr/checkpoint.ckpt                 # image model + config.yaml
    data/pretrain/phmr_vid/prhmr_release_002.ckpt      # video head + .yaml
    data/pretrain/vitpose-h-coco_25.pth
    data/yolo11x.pt
    data/body_models/smplx/SMPLX_NEUTRAL.npz

Most weights come from the PromptHMR Google Drive folders (``gdown``); the video
head is also published directly by BEDLAM2. The SMPL-X body model is reused from
the 4DAnyone model cache. Run as part of ``AUTO_DOWNLOAD_MODELS`` or manually:

    python /app/scripts/download_prompthmr.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROMPTHMR_ROOT = Path(os.environ.get("PROMPTHMR_ROOT", "/opt/prompthmr"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
GVHMR_ROOT = Path(os.environ.get("GVHMR_ROOT", "/app/third_party/GVHMR"))

PRETRAIN = PROMPTHMR_ROOT / "data" / "pretrain"
REQUIRED = (
    PRETRAIN / "phmr" / "checkpoint.ckpt",
    PRETRAIN / "phmr_vid" / "prhmr_release_002.ckpt",
    PRETRAIN / "vitpose-h-coco_25.pth",
    PROMPTHMR_ROOT / "data" / "yolo11x.pt",
)

# Google Drive folders from PromptHMR scripts/fetch_data.sh
GDRIVE_FOLDERS = (
    "1EQ7arZz135T-WpxkS_K1R_hjZp3prh-y",  # image model + config
    "18SywG7Fc_iTfVNaikjHAZmy-A9I85eKv",  # video head
    "1OKhTdL1QVFH3f4hbIEa7jLANx4azuPi1",  # third-party checkpoints (ViTPose/YOLO)
    "1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7",  # supplementary (smplx2smpl etc.)
)

# Direct BEDLAM2 mirrors for the video head.
BEDLAM2_FILES = {
    "phmr_b1b2.ckpt": PRETRAIN / "phmr_vid" / "prhmr_release_002.ckpt",
}


def _run(command: list[str]) -> bool:
    try:
        return subprocess.run(command, check=True).returncode == 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[prompthmr] command failed: {' '.join(command)} ({exc})", flush=True)
        return False


def _warm_clip_cache() -> None:
    """Pre-download the MetaCLIP backbone PromptHMR constructs at load time."""

    try:
        import open_clip
    except Exception as exc:  # noqa: BLE001
        print(f"[prompthmr] open_clip unavailable; skipping CLIP warmup: {exc}", flush=True)
        return
    print("[prompthmr] warming the CLIP backbone cache (ViT-L-14-quickgelu/metaclip_fullcc)", flush=True)
    try:
        open_clip.create_model("ViT-L-14-quickgelu", pretrained="metaclip_fullcc")
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[prompthmr] CLIP warmup failed (will download on first run): {exc}", flush=True)


def _link_smplx() -> None:
    target = PROMPTHMR_ROOT / "data" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if target.is_file():
        return
    for source in (
        MODEL_DIR / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz",
        GVHMR_ROOT / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz",
    ):
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"[prompthmr] SMPL-X linked from {source}", flush=True)
            return
    print("[prompthmr] WARNING: SMPL-X neutral model not found in the model cache", flush=True)


def _fetch_gdrive_folders() -> None:
    if not shutil.which("gdown"):
        print("[prompthmr] gdown not installed; skipping Google Drive downloads", flush=True)
        return
    PRETRAIN.mkdir(parents=True, exist_ok=True)
    for folder in GDRIVE_FOLDERS:
        url = f"https://drive.google.com/drive/folders/{folder}?usp=sharing"
        print(f"[prompthmr] gdown folder {folder}", flush=True)
        _run(["gdown", "--folder", "-O", str(PRETRAIN) + "/", url])


def _fetch_bedlam2() -> None:
    import urllib.request

    for name, target in BEDLAM2_FILES.items():
        if target.is_file():
            continue
        url = f"https://download.is.tue.mpg.de/bedlam2/ml/videos/{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[prompthmr] downloading {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:  # noqa: BLE001 - best effort
            print(f"[prompthmr] download failed for {name}: {exc}", flush=True)


def _fetch_yolo() -> None:
    target = PROMPTHMR_ROOT / "data" / "yolo11x.pt"
    if target.is_file():
        return
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        print(f"[prompthmr] ultralytics unavailable for YOLO download: {exc}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11x.pt")  # downloads to cwd/weights if absent
    source = Path(getattr(model, "ckpt_path", "yolo11x.pt"))
    if source.is_file():
        shutil.copy2(source, target)
    else:
        shutil.move("yolo11x.pt", target)


def main() -> int:
    if not PROMPTHMR_ROOT.is_dir():
        print(f"[prompthmr] PROMPTHMR_ROOT does not exist: {PROMPTHMR_ROOT}", flush=True)
        return 1
    missing = [path for path in REQUIRED if not path.is_file()]
    if missing:
        _fetch_gdrive_folders()
        _fetch_bedlam2()
    _fetch_yolo()
    _link_smplx()
    _warm_clip_cache()

    still_missing = [path for path in REQUIRED if not path.is_file()]
    if still_missing:
        print("[prompthmr] missing checkpoints after download attempt:", flush=True)
        for path in still_missing:
            print(f"  - {path}", flush=True)
        print(
            "[prompthmr] place them manually under data/pretrain (see PromptHMR "
            "scripts/fetch_data.sh) or set PROMPTHMR_CHECKPOINTS_SOURCE.",
            flush=True,
        )
        return 1
    print("[prompthmr] all required checkpoints present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
