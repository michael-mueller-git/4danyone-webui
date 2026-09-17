"""Pure-PyTorch port of the SMPLer-X (ViT-H) body regressor.

A self-contained, dependency-light reimplementation of the SMPLer-X inference
path (https://github.com/MotrixLab/SMPLer-X, commit 064baef0) with the mmcv /
mmdet / mmpose stack removed. Only the body branch is ported:

  * ViT-H image encoder (mmpose "ViT" backbone, 32 blocks, 31 task tokens)
  * PositionNet (body)  -- 3D joint heatmap -> joint coordinates
  * BodyRotationNet     -- root pose, body pose, shape, camera

It consumes the official ``smpler_x_h32_correct.pth.tar`` checkpoint and
produces camera-frame SMPL-X parameters (global_orient / body_pose / betas /
transl), which ``fdanyone_smplerx.runner.run_smplerx`` converts into a
4DAnyone ``MotionResult``.
"""

from fdanyone_smplerx.model import SMPLXRegressor
from fdanyone_smplerx.loader import build_model, load_checkpoint, resolve_checkpoint
from fdanyone_smplerx.runner import run_smplerx

__all__ = ["SMPLXRegressor", "build_model", "load_checkpoint", "resolve_checkpoint", "run_smplerx"]
