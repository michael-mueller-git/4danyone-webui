"""Bootstrap for the vendored PromptHMR runtime on the main torch 2.8 env.

PromptHMR imports ``xformers.ops.memory_efficient_attention`` at module scope
in its SAM-style transformer blocks. Installing xformers would drag in a newer
torch, so we register a small SDPA-backed shim instead and disable xformers in
the vendored DINOv2 layers (which fall back to ``scaled_dot_product_attention``
when ``XFORMERS_DISABLED`` is set).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

PROMPTHMR_ROOT = Path(os.environ.get("PROMPTHMR_ROOT", "/opt/prompthmr")).resolve()


def _install_xformers_shim() -> bool:
    if "xformers" in sys.modules or importlib.util.find_spec("xformers") is not None:
        return False

    import torch

    ops = types.ModuleType("xformers.ops")

    def memory_efficient_attention(q, k, v, attn_bias=None, **_kwargs):
        # q, k, v: (B, N, H, D); SDPA wants (B, H, N, D).
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_bias
        )
        return out.transpose(1, 2)

    ops.memory_efficient_attention = memory_efficient_attention
    ops.unbind = torch.unbind
    root = types.ModuleType("xformers")
    root.ops = ops
    sys.modules["xformers"] = root
    sys.modules["xformers.ops"] = ops
    return True


def configure() -> None:
    """Make ``import prompt_hmr`` / ``pipeline.*`` resolvable on this env."""

    os.environ.setdefault("XFORMERS_DISABLED", "1")
    # PromptHMR inserts this relative to cwd; add it absolutely so the vendored
    # GVHMR (imported as top-level ``hmr4d``) resolves regardless of cwd.
    for entry in (PROMPTHMR_ROOT, PROMPTHMR_ROOT / "pipeline" / "gvhmr"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    _install_xformers_shim()


@contextmanager
def prompthmr_workspace():
    """Run PromptHMR code with its hard-coded relative ``data/`` paths in scope."""

    previous = Path.cwd()
    os.chdir(PROMPTHMR_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)
