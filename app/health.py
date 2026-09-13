"""GPU and runtime health reporting for the Settings tab.

The unlocked CMP 170HX reports itself as a GA100 (sm_80) device. Everything
here tolerates a GPU-less container (used only for UI smoke tests) and never
crashes startup.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from app import config


def nvidia_smi_text() -> str:
    if shutil.which("nvidia-smi") is None:
        return "(nvidia-smi not found - is the NVIDIA container runtime enabled?)"
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout or completed.stderr or "(nvidia-smi returned no output)"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(nvidia-smi failed: {exc})"


def torch_devices() -> list[dict[str, Any]]:
    """Describe visible CUDA devices with capability and memory."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        return [{"error": f"torch import failed: {exc}"}]

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return []

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": f"{props.major}.{props.minor}",
                "compute_capability": tuple((props.major, props.minor)),
                "memory_gib": round(props.total_memory / (1024**3), 1) if props.total_memory else None,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
            }
        )
    return devices


def attention_availability() -> dict[str, bool]:
    """Which attention backends would resolve for inference."""

    available = {"flash_attn_3": False, "sageattention": False, "sdpa": True}
    try:
        from sageattention import sageattn  # noqa: F401

        available["sageattention"] = True
    except Exception:
        pass
    try:
        import flash_attn_interface  # noqa: F401

        available["flash_attn_3"] = True
    except Exception:
        pass
    return available


def summary_text() -> str:
    """Human-readable one-shot report used by the entrypoint and Settings tab."""

    lines: list[str] = []
    lines.append("== GPU report ==")
    lines.append(nvidia_smi_text().strip())
    devices = torch_devices()
    lines.append("== PyTorch devices ==")
    if not devices:
        lines.append("No CUDA devices visible to PyTorch.")
    for device in devices:
        memory = device.get("memory_gib")
        memory = f"{memory} GiB" if memory else "?"
        lines.append(
            f"cuda:{device['index']} {device['name']} | capability {device['capability']} | "
            f"{memory} | torch {device['torch_version']} / CUDA {device['cuda_version']}"
        )
    attention = attention_availability()
    enabled = [name for name, ok in attention.items() if ok]
    try:
        from fdanyone.vendor.diffsynth.models.wan_video_dit import get_attention_backend

        resolved = get_attention_backend(config.DEFAULT_ATTENTION)
    except Exception as exc:
        resolved = f"unresolved ({exc})"
    lines.append("== attention backend ==")
    lines.append(f"installed: {', '.join(enabled)}")
    lines.append(f"resolved ({config.DEFAULT_ATTENTION}): {resolved}")
    lines.append(f"model dir: {config.MODEL_DIR}")
    lines.append(f"data dir: {config.DATA_DIR}")
    return "\n".join(lines)


def json_report() -> str:
    return json.dumps(
        {
            "devices": torch_devices(),
            "attention": attention_availability(),
            "nvidia_smi": nvidia_smi_text(),
        },
        indent=2,
        default=str,
    )


if __name__ == "__main__":
    print(summary_text())
