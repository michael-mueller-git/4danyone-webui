"""Provision the PromptHMR-Vid checkpoints into the persistent model cache.

PromptHMR hard-codes relative ``data/pretrain/...`` paths inside its checkout, so
this script keeps the weights on the model volume (``MODEL_DIR/prompthmr``) and
symlinks them into ``PROMPTHMR_ROOT/data``.

Acquisition order:

1. ``PROMPTHMR_CHECKPOINTS_SOURCE=<dir>`` — copy files from a local directory.
2. ``PROMPTHMR_HF_REPO=<repo>`` — pull the same layout from a HuggingFace repo
   (respects ``HF_ENDPOINT``, so it works with an internal HF mirror).
3. Best-effort network downloads (Google Drive via ``gdown``, the BEDLAM2
   mirror, Ultralytics for YOLO, and the MetaCLIP backbone URL).

PromptHMR's weights are **not** published on HuggingFace, so for internal-CA /
air-gapped clusters mirror the files below once into your own HF repo or a PVC
directory:

    phmr/checkpoint.ckpt
    phmr/config.yaml
    phmr_vid/prhmr_release_002.ckpt
    phmr_vid/prhmr_release_002.yaml
    vitpose-h-coco_25.pth
    l14_fullcc2.5b.pt          # MetaCLIP ViT-L/14 backbone (PromptHMR ClipEncoder)

The YOLO detector is reused from GVHMR's ``yolov8x.pt`` and SMPL-X from the
4DAnyone model cache, so neither needs to be mirrored. Run manually or via
``AUTO_DOWNLOAD_MODELS=true``:

    python /app/scripts/download_prompthmr.py
"""

from __future__ import annotations

import os
import shutil
import ssl
import subprocess
import sys
from pathlib import Path

PROMPTHMR_ROOT = Path(os.environ.get("PROMPTHMR_ROOT", "/opt/prompthmr"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
GVHMR_ROOT = Path(os.environ.get("GVHMR_ROOT", "/app/third_party/GVHMR"))
SOURCE = os.environ.get("PROMPTHMR_CHECKPOINTS_SOURCE")
HF_REPO = os.environ.get("PROMPTHMR_HF_REPO")
# The official weights are served through a TLS-inspecting proxy in some
# clusters whose CA is not in the container trust store, so verification is
# skipped by default for these public, read-only assets. Set
# PROMPTHMR_INSECURE_SSL=0 to enforce TLS verification.
INSECURE_SSL = os.environ.get("PROMPTHMR_INSECURE_SSL", "1").lower() not in ("0", "false", "no")
GDRIVE_TLS_FLAGS = ("--no-check-certificate",) if INSECURE_SSL else ()
# PromptHMR's detector hard-codes data/yolo11x.pt, but any person detector works;
# reuse GVHMR's already-downloaded YOLOv8x to avoid another external fetch.
YOLO_FALLBACKS = (
    GVHMR_ROOT / "inputs" / "checkpoints" / "yolo" / "yolov8x.pt",
)

CACHE = MODEL_DIR / "prompthmr"
PRETRAIN = CACHE / "pretrain"
# open_clip caches URL-downloaded backbones under ~/.cache/clip with no env hook,
# so keep the master copy on the volume and symlink it into place at boot.
CLIP_MASTER = CACHE / "clip" / "l14_fullcc2.5b.pt"
CLIP_RUNTIME = Path.home() / ".cache" / "clip" / "l14_fullcc2.5b.pt"

# (relative path in SOURCE, destination path)
FILE_MAP = (
    ("phmr/checkpoint.ckpt", PRETRAIN / "phmr" / "checkpoint.ckpt"),
    ("phmr/config.yaml", PRETRAIN / "phmr" / "config.yaml"),
    ("phmr_vid/prhmr_release_002.ckpt", PRETRAIN / "phmr_vid" / "prhmr_release_002.ckpt"),
    ("phmr_vid/prhmr_release_002.yaml", PRETRAIN / "phmr_vid" / "prhmr_release_002.yaml"),
    ("vitpose-h-coco_25.pth", PRETRAIN / "vitpose-h-coco_25.pth"),
    ("yolo11x.pt", CACHE / "yolo11x.pt"),
    ("l14_fullcc2.5b.pt", CLIP_MASTER),
)

REQUIRED = tuple(destination for _, destination in FILE_MAP)

GDRIVE_FOLDERS = (
    "1EQ7arZz135T-WpxkS_K1R_hjZp3prh-y",  # phmr (image model + config)
    "18SywG7Fc_iTfVNaikjHAZmy-A9I85eKv",  # phmr_vid (video head)
    "1OKhTdL1QVFH3f4hbIEa7jLANx4azuPi1",  # sam2_ckpts (world-video extras)
    "1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7",  # supplementary (smplx2smpl etc.)
)

# PromptHMR publishes ViTPose as a lone Google Drive file, not inside a folder,
# so it must be fetched separately (see PromptHMR scripts/fetch_data.sh).
GDRIVE_FILES = {
    "1ZprPoNXe_f9a9flr0RhS3XCJBfqhFSeE": PRETRAIN / "vitpose-h-coco_25.pth",
}

BEDLAM2_FILES = {
    "phmr_b1b2.ckpt": PRETRAIN / "phmr_vid" / "prhmr_release_002.ckpt",
}
CLIP_URL = "https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt"


def _run(command: list[str]) -> bool:
    try:
        return subprocess.run(command, check=True).returncode == 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[prompthmr] command failed: {' '.join(command)} ({exc})", flush=True)
        return False


def _download_url(url: str, target: Path) -> None:
    """Download ``url`` to ``target``, honoring the INSECURE_SSL setting."""

    import urllib.request

    context = None
    if INSECURE_SSL:
        # Build the context directly so a broken SSL_CERT_FILE can't break us.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=context) as response, open(target, "wb") as output:
        shutil.copyfileobj(response, output)


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    if target.exists():
        return
    target.symlink_to(source)


def _wire_checkout() -> None:
    """Point PromptHMR's relative ``data/`` paths at the persistent cache."""

    data = PROMPTHMR_ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, source in (
        ("pretrain", PRETRAIN),
        ("body_models", CACHE / "body_models"),
    ):
        source.mkdir(parents=True, exist_ok=True)
        _link(source, data / name)
    _link(CACHE / "yolo11x.pt", data / "yolo11x.pt")
    CLIP_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    _link(CLIP_MASTER, CLIP_RUNTIME)


def _copy_from_source() -> None:
    root = Path(SOURCE).expanduser()
    if not root.is_dir():
        print(f"[prompthmr] PROMPTHMR_CHECKPOINTS_SOURCE is not a directory: {root}", flush=True)
        return
    for relative, destination in FILE_MAP:
        origin = root / relative
        if not origin.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        print(f"[prompthmr] copied {relative}", flush=True)


def _fetch_from_hf() -> None:
    """Pull the mirrored checkpoints from an HF repo via the configured endpoint."""

    if not HF_REPO:
        return
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        print(f"[prompthmr] huggingface_hub unavailable: {exc}", flush=True)
        return
    staging = CACHE / "hf"
    for relative, destination in FILE_MAP:
        if destination.is_file():
            continue
        try:
            downloaded = hf_hub_download(repo_id=HF_REPO, filename=relative, local_dir=str(staging))
        except Exception as exc:  # noqa: BLE001 - file may simply be absent
            print(f"[prompthmr] HF miss {HF_REPO}/{relative}: {exc}", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)
        print(f"[prompthmr] fetched {relative} from {HF_REPO}", flush=True)


def _reuse_gvhmr_yolo() -> None:
    target = CACHE / "yolo11x.pt"
    if target.is_file():
        return
    for source in YOLO_FALLBACKS:
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"[prompthmr] reusing YOLO detector from {source}", flush=True)
            return


def _fetch_gdrive_folders() -> None:
    if not shutil.which("gdown"):
        print("[prompthmr] gdown not installed; skipping Google Drive downloads", flush=True)
        return
    PRETRAIN.mkdir(parents=True, exist_ok=True)
    for folder in GDRIVE_FOLDERS:
        url = f"https://drive.google.com/drive/folders/{folder}?usp=sharing"
        print(f"[prompthmr] gdown folder {folder}", flush=True)
        # Folder URLs are auto-detected (gdown dropped --folder for URLs).
        _run(["gdown", "-O", str(PRETRAIN) + "/", *GDRIVE_TLS_FLAGS, url])


def _fetch_gdrive_files() -> None:
    if not shutil.which("gdown"):
        print("[prompthmr] gdown not installed; skipping Google Drive file downloads", flush=True)
        return
    for file_id, target in GDRIVE_FILES.items():
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/file/d/{file_id}/view"
        print(f"[prompthmr] gdown file {file_id} -> {target.name}", flush=True)
        # Recent gdown dropped --fuzzy; file URLs are auto-detected.
        _run(["gdown", "-O", str(target), *GDRIVE_TLS_FLAGS, url])


def _fetch_bedlam2() -> None:
    for name, target in BEDLAM2_FILES.items():
        if target.is_file():
            continue
        url = f"https://download.is.tue.mpg.de/bedlam2/ml/videos/{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[prompthmr] downloading {url}", flush=True)
        try:
            _download_url(url, target)
        except Exception as exc:  # noqa: BLE001 - best effort
            print(f"[prompthmr] download failed for {name}: {exc}", flush=True)


def _fetch_clip() -> None:
    if CLIP_MASTER.is_file():
        return
    CLIP_MASTER.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prompthmr] downloading {CLIP_URL}", flush=True)
    try:
        _download_url(CLIP_URL, CLIP_MASTER)
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[prompthmr] CLIP backbone download failed: {exc}", flush=True)


def _fetch_yolo() -> None:
    target = CACHE / "yolo11x.pt"
    if target.is_file():
        return
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        print(f"[prompthmr] ultralytics unavailable for YOLO download: {exc}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11x.pt")
    source = Path(getattr(model, "ckpt_path", "yolo11x.pt"))
    if source.is_file() and source.resolve() != target.resolve():
        shutil.copy2(source, target)


def _link_smplx() -> None:
    target = CACHE / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
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


def main() -> int:
    if not PROMPTHMR_ROOT.is_dir():
        print(f"[prompthmr] PROMPTHMR_ROOT does not exist: {PROMPTHMR_ROOT}", flush=True)
        return 1
    _wire_checkout()
    _link_smplx()
    _reuse_gvhmr_yolo()

    if SOURCE:
        print(f"[prompthmr] copying checkpoints from {SOURCE}", flush=True)
        _copy_from_source()
    _fetch_from_hf()
    folder_targets = (
        PRETRAIN / "phmr" / "checkpoint.ckpt",
        PRETRAIN / "phmr" / "config.yaml",
        PRETRAIN / "phmr_vid" / "prhmr_release_002.ckpt",
        PRETRAIN / "phmr_vid" / "prhmr_release_002.yaml",
    )
    if any(not path.is_file() for path in folder_targets):
        _fetch_gdrive_folders()
        _fetch_bedlam2()
    if any(not path.is_file() for path in GDRIVE_FILES.values()):
        _fetch_gdrive_files()
    _fetch_yolo()
    _fetch_clip()

    missing = [path for path in REQUIRED if not path.is_file()]
    if missing:
        print("[prompthmr] missing checkpoints:", flush=True)
        for path in missing:
            print(f"  - {path}", flush=True)
        print(
            "[prompthmr] fetch these once on an internet-connected machine and place them "
            "under PROMPTHMR_CHECKPOINTS_SOURCE (see the script docstring for the list).",
            flush=True,
        )
        return 1
    print("[prompthmr] all required checkpoints present", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
