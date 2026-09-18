"""SMPLer-X motion stage: produce a 4DAnyone ``MotionResult``.

Replaces ``fdanyone.motion.gvhmr.run_gvhmr`` for the ``MOTION_BACKEND=smplerx``
path. Reuses the GVHMR checkout for person detection + ViTPose 2D keypoints,
runs the ported SMPLer-X-H32 body regressor per clip, converts its
camera-frame SMPL-X parameters (with gravity alignment) into the
``MotionResult`` contract the rest of the pipeline consumes.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fdanyone.motion.gvhmr import _install_optional_import_stubs, gvhmr_imports
from fdanyone.motion.result import MotionResult
from fdanyone.vendor.pytorch3d_compat import install_if_needed as install_pytorch3d_compat
from fdanyone_smplerx.config import (
    FOCAL,
    INPUT_BODY_SHAPE,
    PRINCPT,
    SMPLERX_REVISION,
)
from fdanyone_smplerx.loader import build_model, load_checkpoint, resolve_checkpoint
from fdanyone_smplerx.preprocess import crop_body
from fdanyone_smplerx.transforms import axis_angle_to_matrix, matrix_to_axis_angle

LOGGER = logging.getLogger("fdanyone")

_BATCH = int(os.environ.get("SMPLERX_BATCH", "16"))

# SMPL-X joint indices for the body's vertical axis
_PELVIS = 0
_NECK = 12


def _full_frame_bbox(width: int, height: int) -> tuple[float, float, float, float]:
    return 0.0, 0.0, float(width), float(height)


def _bootstrap_gvhmr_imports() -> None:
    """Install the optional-dependency shims GVHMR inference needs.

    ``run_gvhmr`` performs this in ``_register_inference_store``; the SMPLer-X
    path imports the same ``hmr4d`` modules, so it must activate the bundled
    ``pytorch3d`` compatibility subset and stub the moving-camera ``pycolmap``
    dependency before any ``hmr4d`` import runs.
    """

    registered = install_pytorch3d_compat()
    _install_optional_import_stubs()
    LOGGER.info("SMPLer-X: GVHMR import shims ready (pytorch3d compat installed=%s).", registered)


def _person_bbox_and_keypoints(gvhmr_root, video_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """YOLO person track (xyxy) + ViTPose 2D keypoints via the GVHMR checkout."""
    _bootstrap_gvhmr_imports()
    with gvhmr_imports(gvhmr_root):
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
        from hmr4d.utils.preproc.tracker import Tracker
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        from hmr4d.utils.video_io_utils import get_video_lwh

        frame_count, width, height = get_video_lwh(video_path)
        LOGGER.info("SMPLer-X: detecting the person in %d frames (%dx%d).", frame_count, width, height)
        tracker = Tracker()
        try:
            bbx_xyxy = tracker.get_one_track(video_path).float()
        except IndexError:
            LOGGER.warning("SMPLer-X tracker found no person; using a full-frame bbox.")
            bbx_xyxy = torch.tensor(_full_frame_bbox(width, height), dtype=torch.float32).repeat(frame_count, 1)
        LOGGER.info(
            "SMPLer-X: tracker produced %d boxes (x %.1f..%.1f, y %.1f..%.1f).",
            bbx_xyxy.shape[0],
            float(bbx_xyxy[:, 0].min()),
            float(bbx_xyxy[:, 2].max()),
            float(bbx_xyxy[:, 1].min()),
            float(bbx_xyxy[:, 3].max()),
        )
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        extractor = VitPoseExtractor()
        LOGGER.info("SMPLer-X: running ViTPose on %d boxes.", bbx_xys.shape[0])
        kp2d = extractor.extract(video_path, bbx_xys)
        LOGGER.info("SMPLer-X: ViTPose keypoints shape %s.", tuple(kp2d.shape))
    return bbx_xyxy.detach().cpu(), kp2d.detach().cpu().float()


def _camera_from_bbox(bbox: np.ndarray) -> torch.Tensor:
    """Full-image camera whose projection matches SMPLer-X's virtual camera."""
    w, h = float(bbox[2]), float(bbox[3])
    f_x = FOCAL[0] / INPUT_BODY_SHAPE[1] * w
    f_y = FOCAL[1] / INPUT_BODY_SHAPE[0] * h
    c_x = PRINCPT[0] / INPUT_BODY_SHAPE[1] * w + float(bbox[0])
    c_y = PRINCPT[1] / INPUT_BODY_SHAPE[0] * h + float(bbox[1])
    return torch.tensor([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _rotation_to_y(up: torch.Tensor) -> torch.Tensor:
    """Rotation mapping the body 'up' vector onto world +y (Rodrigues)."""
    up = up / up.norm().clamp_min(1e-8)
    y = torch.tensor([0.0, 1.0, 0.0], device=up.device)
    v = torch.cross(up, y, dim=-1)
    s = float(v.norm())
    c = float(torch.dot(up, y))
    if s < 1e-6:
        if c < 0:
            return torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32, device=up.device)
        return torch.eye(3, device=up.device)
    v = v / s
    vx = torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=torch.float32, device=up.device
    )
    # Rodrigues on the normalized axis: I + sin(theta)[v]x + (1-cos(theta))[v]x^2
    return torch.eye(3, device=up.device) + s * vx + vx @ vx * (1.0 - c)


def _gravity_align(
    root_pose: torch.Tensor,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    cam_trans: torch.Tensor,
    device: str,
    gvhmr_root,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert camera-frame SMPL-X into GVHMR's gravity-aligned y-up world."""
    import smplx

    model_path = Path(gvhmr_root) / "inputs/checkpoints/body_models"
    body = smplx.create(str(model_path), "smplx", gender="neutral", use_pca=False, flat_hand_mean=True).to(device).eval()
    with torch.inference_mode():
        out = body(global_orient=root_pose, body_pose=body_pose, betas=betas, transl=cam_trans)
        joints = out.joints  # (N, 55, 3)
    up = (joints[:, _NECK] - joints[:, _PELVIS]).mean(0)
    rotation = _rotation_to_y(up)  # (3,3), camera -> world
    root_world = matrix_to_axis_angle(rotation @ axis_angle_to_matrix(root_pose))
    transl_world = (rotation @ cam_trans.unsqueeze(-1)).squeeze(-1)
    return root_world, transl_world, rotation


def run_smplerx(
    *,
    clip,
    working_video: str | Path,
    output_dir: str | Path,
    gvhmr_root: str | Path,
    device: str,
) -> MotionResult:
    """Recover static-camera human motion with the SMPLer-X body regressor."""
    started = time.monotonic()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "SMPLer-X: start (device=%s, video=%s, gvhmr_root=%s, output_dir=%s).",
        device,
        working_video,
        gvhmr_root,
        output_root,
    )
    checkpoint = resolve_checkpoint()
    LOGGER.info("SMPLer-X: resolved checkpoint %s.", checkpoint)
    bbx_xyxy, kp2d = _person_bbox_and_keypoints(gvhmr_root, str(working_video))
    LOGGER.info("SMPLer-X: detection + keypoints done in %.1fs.", time.monotonic() - started)

    frames = [np.asarray(frame) for frame in clip.rgb_frames]
    num_frames = len(frames)
    if num_frames == 0:
        raise RuntimeError("SMPLer-X: empty canonical clip.")
    if bbx_xyxy.shape[0] != num_frames:
        LOGGER.warning(
            "SMPLer-X: tracker produced %d boxes for %d frames; tiling the first box.",
            bbx_xyxy.shape[0],
            num_frames,
        )
        bbx_xyxy = bbx_xyxy[:1].repeat(num_frames, 1)

    LOGGER.info("SMPLer-X: loading model on %s from %s.", device, checkpoint)
    model = build_model(device)
    load_checkpoint(model, checkpoint, device)
    LOGGER.info("SMPLer-X: model ready; regressing %d frames (batch=%d).", num_frames, _BATCH)

    all_root, all_body, all_shape, all_cam, all_K = [], [], [], [], []
    with torch.inference_mode():
        for start in range(0, num_frames, _BATCH):
            end = min(start + _BATCH, num_frames)
            LOGGER.info("SMPLer-X: frames %d..%d.", start, end - 1)
            indices = range(start, end)
            patches: list[np.ndarray] = []
            bboxes: list[np.ndarray] = []
            for index in indices:
                patch, bbox = crop_body(frames[index], bbx_xyxy[index].numpy())
                if patch is None:
                    height, width = frames[index].shape[:2]
                    LOGGER.warning("SMPLer-X: degenerate person box on frame %d; using full frame.", index)
                    patch, bbox = crop_body(frames[index], np.asarray(_full_frame_bbox(width, height)))
                if patch is None:
                    raise RuntimeError(f"SMPLer-X: could not crop frame {index}.")
                patches.append(patch)
                bboxes.append(bbox)
            img = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).float().div_(255.0).to(device)
            body_img = F.interpolate(img, size=INPUT_BODY_SHAPE)
            out = model(body_img)
            all_root.append(out["root_pose"].cpu())
            all_body.append(out["body_pose"].cpu())
            all_shape.append(out["shape"].cpu())
            all_cam.append(out["cam_trans"].cpu())
            all_K.extend(_camera_from_bbox(bbox) for bbox in bboxes)

    root_pose = torch.cat(all_root)
    body_pose = torch.cat(all_body)
    betas = torch.cat(all_shape)
    cam_trans = torch.cat(all_cam)
    K_fullimg = torch.stack(all_K)

    LOGGER.info("SMPLer-X: aligning %d frames to gravity.", num_frames)
    root_world, transl_world, _ = _gravity_align(
        root_pose.to(device), body_pose.to(device), betas.to(device), cam_trans.to(device), device, gvhmr_root
    )

    result = MotionResult(
        gvhmr_revision=SMPLERX_REVISION,
        fps=clip.fps,
        frame_timestamps_sec=tuple(float(frame.canonical_timestamp) for frame in clip.frames),
        source_frame_indices=tuple(frame.source_index for frame in clip.frames),
        source_pts=tuple(frame.source_pts for frame in clip.frames),
        source_size_bytes=clip.source_size_bytes,
        source_mtime_ns=clip.source_mtime_ns,
        image_height=clip.height,
        image_width=clip.width,
        smpl_params_global={
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": root_world.cpu(),
            "transl": transl_world.cpu(),
        },
        smpl_params_incam={
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": root_pose,
            "transl": cam_trans,
        },
        K_fullimg=K_fullimg,
        observed_keypoints_2d=kp2d,
    )
    result.validate(expected_frames=num_frames)
    LOGGER.info("SMPLer-X: motion ready for %d frames in %.1fs.", num_frames, time.monotonic() - started)
    return result
