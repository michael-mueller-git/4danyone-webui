"""Checkpoint resolution + safe state-dict loading for the SMPLer-X regressor."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from fdanyone_smplerx.config import CHECKPOINT_FILE, CHECKPOINT_REPO
from fdanyone_smplerx.model import SMPLXRegressor

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models")).resolve()


def checkpoint_path() -> Path:
    return MODEL_DIR / "smplerx" / CHECKPOINT_FILE


def resolve_checkpoint() -> Path:
    """Return the local checkpoint, downloading it on first use."""
    path = checkpoint_path()
    if path.is_file():
        return path
    print(f"[smplerx] Downloading {CHECKPOINT_REPO}/{CHECKPOINT_FILE} ...", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE, local_dir=str(path.parent))
        return Path(downloaded)
    except Exception as exc:
        raise RuntimeError(
            f"SMPLer-X checkpoint download failed: {exc}. "
            f"Download {CHECKPOINT_FILE} from https://huggingface.co/{CHECKPOINT_REPO} "
            f"and place it at {path}."
        ) from exc


def build_model(device: str) -> SMPLXRegressor:
    model = SMPLXRegressor()
    model.to(device).eval()
    return model


def load_checkpoint(model: SMPLXRegressor, ckpt_path: str | Path, device: str) -> SMPLXRegressor:
    """Load the official checkpoint, tolerating key/prefix drift (strict=False)."""
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "network" in state:
        state = state["network"]

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        renamed = key
        while renamed.startswith("module."):
            renamed = renamed[len("module.") :]
        renamed = renamed.replace("backbone", "encoder")
        renamed = renamed.replace("body_rotation_net", "body_regressor")
        renamed = renamed.replace("hand_rotation_net", "hand_regressor")
        remapped[renamed] = value

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in remapped.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing = [key for key in model_state if key not in compatible]
    extra = [key for key in remapped if key not in compatible]
    print(
        f"[smplerx] loaded {len(compatible)}/{len(model_state)} parameter tensors "
        f"(missing {len(missing)}, skipped {len(extra)})",
        flush=True,
    )
    if missing:
        print(f"[smplerx] missing keys (first 10): {missing[:10]}", flush=True)
    model.load_state_dict(compatible, strict=False)
    model.to(device).eval()
    return model
