"""One inference run against 4DAnyone's pipeline, sharing the CLI's exact
CUDA allocator / attention-backend preconditions.

Runs in a dedicated subprocess so the Gradio process never owns CUDA state.
"""

from __future__ import annotations

from typing import Any


def do_run(
    *,
    video_path: str,
    out_dir: str,
    model_dir: str,
    gvhmr_root: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Execute 4DAnyone for one clip with the WebUI's parameter overrides."""

    gpu_ids = params.get("gpu_ids") or None
    attention = params.get("attention_backend") or "auto"

    # These must come before the first torch import (mirrors inference.main).
    from fdanyone.attention import validate_attention_backend
    from fdanyone.device import configure_inference_cuda_allocator, has_low_memory_gpu

    validate_attention_backend(attention)
    low_memory = has_low_memory_gpu(gpu_ids)
    configure_inference_cuda_allocator(use_expandable_segments=low_memory)
    if attention == "auto" and low_memory:
        attention = "sdpa"

    from fdanyone.pipeline import run_pipeline

    return run_pipeline(
        video_path=video_path,
        output_dir=out_dir,
        views_per_layer=int(params.get("views_per_layer", 6)),
        layer_pitches=params.get("layer_pitches") or [15],
        start_yaw=int(params.get("start_yaw", 0)),
        yaw_span=int(params.get("yaw_span", 360)),
        views_per_group=params.get("views_per_group", "auto"),
        enable_rcp=bool(params.get("enable_rcp", True)),
        enable_tcr=bool(params.get("enable_tcr", True)),
        enable_turbo=bool(params.get("enable_turbo", True)),
        model_dir=model_dir,
        checkpoint_path=None,
        mhr70_regressor_path=None,
        gvhmr_root=gvhmr_root,
        gpu_ids=gpu_ids,
        attention_backend=attention,
        target_fps=params.get("target_fps", "auto"),
        start_time=float(params.get("start_time", 0.0)),
        seed=int(params.get("seed", 42)),
    )
