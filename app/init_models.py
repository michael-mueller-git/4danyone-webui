"""Model / asset provisioning for the WebUI.

Wraps 4DAnyone's own download helpers so the Gradio button and the entrypoint
share exactly the same code path. Everything downloads into the persistent
``MODEL_DIR`` volume and can be resumed safely (huggingface_hub staging).

SMPL-X is license-gated; the WebUI tries, in order:
  1. a user-supplied path (SMPLX_SOURCE env),
  2. an archive/NPZ dropped into the DATA_DIR/smplx mount,
  3. a public mirror of the official ``models_smplx_v1_1.zip`` archive.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

from app import config

LOGGER = logging.getLogger("init_models")

Log = Callable[[str], None]


def _verbatim(message: str) -> None:
    print(message, flush=True)


def _fdanyone_ready() -> bool:
    """The WebUI always depends on a 4DAnyone checkout being on disk."""

    return (config.GVHMR_ROOT / "hmr4d" / "__init__.py").is_file()


def model_missing() -> list[str]:
    """Relative (to MODEL_DIR) paths of every still-missing published asset."""

    from fdanyone import assets

    missing: list[str] = []
    for relative in assets.MODEL_FILES:
        if not (config.MODEL_DIR / relative).is_file():
            missing.append(relative)
    for relative in assets.BIREFNET_FILES:
        if not (config.MODEL_DIR / assets.BIREFNET_DIR / relative).is_file():
            missing.append(f"{assets.BIREFNET_DIR}/{relative}")
    return sorted(set(missing))


def smplx_present() -> bool:
    from fdanyone import assets

    return (config.MODEL_DIR / assets.SMPLX_MODEL).is_file()


def gvhmr_ok() -> bool:
    return _fdanyone_ready()


def status_text() -> str:
    from fdanyone import assets

    missing = model_missing()
    lines = [
        f"model dir: {config.MODEL_DIR}",
        f"gvhmr checkout: {'ok' if _fdanyone_ready() else 'MISSING'}",
        f"SMPL-X neutral ({assets.SMPLX_MODEL}): {'present' if smplx_present() else 'missing (licensed)'}",
        f"public assets missing: {len(missing)}",
    ]
    for relative in missing[:20]:
        lines.append(f"  - {relative}")
    if len(missing) > 20:
        lines.append(f"  ... and {len(missing) - 20} more")
    return "\n".join(lines)


def is_ready() -> dict[str, bool]:
    return {
        "gvhmr": _fdanyone_ready(),
        "models": not model_missing(),
        "smplx": smplx_present(),
    }


def download_public_models(log: Log = _verbatim) -> dict[str, str]:
    """Download 4DAnyone public checkpoints + BiRefNet + VGG into MODEL_DIR."""

    if not _fdanyone_ready():
        raise RuntimeError(
            f"GVHMR checkout not found at {config.GVHMR_ROOT}. "
            "The image should contain it; rebuild or inspect the container."
        )
    from fdanyone.download import ensure_models, ensure_perceptual_vgg19

    log("Downloading public 4DAnyone models (DiT, VAE, Turbo LoRA, GVHMR, ViTPose, YOLO, VGG-19)...")
    ensure_models(str(config.MODEL_DIR), str(config.GVHMR_ROOT))
    log("Downloading/verifying perceptual VGG-19...")
    ensure_perceptual_vgg19(str(config.MODEL_DIR))
    missing = model_missing()
    log(f"done. missing now: {len(missing)}")
    for relative in missing:
        log(f"  - {relative}")
    return {"model_dir": str(config.MODEL_DIR), "missing": missing}


def _discover_smplx_drop() -> Path | None:
    """Return a usable archive/NPZ from the drop folder, if any."""

    if not config.SMPLX_DROP_DIR.is_dir():
        return None
    candidates = sorted(config.SMPLX_DROP_DIR.glob("*.zip")) + sorted(
        config.SMPLX_DROP_DIR.glob("SMPLX_NEUTRAL.npz")
    )
    return candidates[0] if candidates else None


def provision_smplx(log: Log = _verbatim) -> str:
    """Install SMPL-X neutral from user source, drop dir, or a public mirror."""

    from fdanyone import assets
    from fdanyone.download import install_smplx

    target = config.MODEL_DIR / assets.SMPLX_MODEL
    if target.is_file():
        log(f"SMPL-X already present at {target}.")
        return str(target)

    source: Path | None = None
    if config.SMPLX_SOURCE:
        candidate = Path(config.SMPLX_SOURCE).expanduser().resolve()
        if candidate.is_file():
            source = candidate
            log(f"Using SMPLX_SOURCE={candidate}")
    if source is None:
        dropped = _discover_smplx_drop()
        if dropped is not None:
            source = dropped
            log(f"Using dropped archive {dropped}")

    if source is not None:
        installed = install_smplx(source, str(config.MODEL_DIR), str(config.GVHMR_ROOT))
        log(f"Installed SMPL-X from {source} -> {installed}")
        return str(installed)

    log(f"Downloading SMPL-X neutral from public mirror {config.SMPLX_MIRROR_REPO} ...")
    mirror = None
    try:
        from huggingface_hub import hf_hub_download

        mirror = hf_hub_download(
            repo_id=config.SMPLX_MIRROR_REPO,
            filename=config.SMPLX_MIRROR_FILE,
            repo_type="dataset",
        )
    except Exception as exc:
        raise RuntimeError(
            "SMPL-X mirror download failed. Download models_smplx_v1_1.zip from "
            f"https://smpl-x.is.tue.mpg.de/ (free account + license) and either set SMPLX_SOURCE "
            f"or drop it into {config.SMPLX_DROP_DIR}. Mirror error: {exc}"
        ) from exc

    installed = install_smplx(Path(mirror), str(config.MODEL_DIR), str(config.GVHMR_ROOT))
    log(f"Installed SMPL-X from mirror -> {installed}")
    return str(installed)


def download_examples(log: Log = _verbatim) -> str:
    from fdanyone.download import download_example

    log("Downloading bundled example videos...")
    result = download_example(str(config.DATA_DIR))
    log(f"Examples are in {result.get('examples')}")
    return result.get("examples", "")


def run_all(log: Log = _verbatim) -> dict[str, str]:
    download_public_models(log)
    provision_smplx(log)
    download_examples(log)
    return status_text().splitlines()[0:1]  # short summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision 4DAnyone assets")
    parser.add_argument("--models", action="store_true", help="download public models")
    parser.add_argument("--smplx", action="store_true", help="install SMPL-X neutral")
    parser.add_argument("--examples", action="store_true", help="download example videos")
    parser.add_argument("--all", action="store_true", help="models + smplx + examples")
    parser.add_argument("--status", action="store_true", help="print current status")
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.status:
        print(status_text())
        return 0
    if args.all:
        run_all()
        return 0
    if args.models:
        download_public_models()
    if args.smplx:
        provision_smplx()
    if args.examples:
        download_examples()
    if not (args.models or args.smplx or args.examples or args.all):
        _build_parser().print_help()
        return 1
    print(status_text())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
