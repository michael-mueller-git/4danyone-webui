"""PromptHMR-Vid motion stage: produce a 4DAnyone ``MotionResult``.

Runs the vendored PromptHMR image model + video head (a GVHMR ``DemoPL`` variant)
on the canonical clip, then converts its camera-frame SMPL-X into the
gravity-aligned world frame the rest of the pipeline consumes. Depth and root
translation come from PromptHMR's temporal video head, which is far more stable
than a per-frame regressor.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from fdanyone.motion.gvhmr import _install_optional_import_stubs, validate_gvhmr
from fdanyone.motion.result import MotionResult
from fdanyone.vendor.pytorch3d_compat import install_if_needed as install_pytorch3d_compat
from fdanyone_prompthmr.compat import PROMPTHMR_ROOT, configure, prompthmr_workspace
from fdanyone_smplerx.runner import _gravity_align

LOGGER = logging.getLogger("fdanyone")

# SMPL-X body layout produced by PromptHMR_Video.run: [global_orient(3), body_pose(63), zeros(9)]
_BODY_POSE_SLICE = slice(3, 66)

# PromptHMR hard-codes these paths relative to PROMPTHMR_ROOT; fail with an
# actionable message instead of a bare assertion deep inside the pipeline.
_REQUIRED_CHECKPOINTS = (
    "data/pretrain/phmr/checkpoint.ckpt",
    "data/pretrain/phmr/config.yaml",
    "data/pretrain/phmr_vid/prhmr_release_002.ckpt",
    "data/pretrain/phmr_vid/prhmr_release_002.yaml",
    "data/pretrain/vitpose-h-coco_25.pth",
    "data/body_models/smplx2smpl_joints.npy",
    "data/body_models/smplx2smpl.pkl",
    "data/body_models/smpl/SMPL_NEUTRAL.pkl",
    "data/yolo11x.pt",
)


def _preflight() -> None:
    missing = [rel for rel in _REQUIRED_CHECKPOINTS if not (PROMPTHMR_ROOT / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            f"PromptHMR checkpoints missing under {PROMPTHMR_ROOT}: {', '.join(missing)}. "
            "Run `python /app/scripts/download_prompthmr.py`, or set "
            "PROMPTHMR_CHECKPOINTS_SOURCE / PROMPTHMR_HF_REPO and restart with "
            "AUTO_DOWNLOAD_MODELS=true."
        )


def _primary_track(tracks: dict, num_frames: int, height: int, width: int) -> dict:
    """Reduce the longest detected track to a full-length, single-person track.

    ``estimate_kp2ds_from_bbox_vitpose`` assumes contiguous ``0..N-1`` frames, so
    the track is resampled onto the canonical frame range (endpoint-clamped).
    """

    track_id = max(tracks, key=lambda key: len(tracks[key]["frames"]))
    track = tracks[track_id]
    frames = np.asarray(track["frames"])
    bboxes = np.asarray(track["bboxes"], dtype=np.float32)
    order = np.argsort(frames)
    frames, bboxes = frames[order], bboxes[order]
    all_frames = np.arange(num_frames)
    resampled = np.stack(
        [np.interp(all_frames, frames, bboxes[:, column]) for column in range(4)], axis=1
    ).astype(np.float32)
    return {
        "track_id": int(track.get("track_id", track_id)),
        "frames": all_frames,
        "bboxes": resampled,
        "masks": np.zeros((num_frames, height, width), dtype=bool),
        "detected": np.ones(num_frames, dtype=bool),
    }


def _camera_matrix(calib, num_frames: int) -> torch.Tensor:
    focal, cx, cy = float(calib[0]), float(calib[2]), float(calib[3])
    K = torch.tensor(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    return K.repeat(num_frames, 1, 1)


def run_prompthmr(
    *,
    clip,
    working_video: str | Path,
    output_dir: str | Path,
    gvhmr_root: str | Path,
    device: str,
) -> MotionResult:
    """Recover static-camera human motion with PromptHMR-Vid."""
    started = time.monotonic()
    _preflight()
    configure()
    install_pytorch3d_compat()
    _install_optional_import_stubs()
    _, gvhmr_revision = validate_gvhmr(gvhmr_root)

    frames = [np.asarray(frame) for frame in clip.rgb_frames]
    num_frames = len(frames)
    if num_frames == 0:
        raise RuntimeError("PromptHMR: empty canonical clip.")
    height, width = frames[0].shape[:2]
    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    LOGGER.info(
        "PromptHMR: start (device=%s, frames=%d, size=%dx%d, root=%s).",
        device,
        num_frames,
        width,
        height,
        PROMPTHMR_ROOT,
    )

    with prompthmr_workspace():
        from pipeline.detector.vitpose_estimator import estimate_kp2ds_from_bbox_vitpose, load_vit_model
        from pipeline.kp_utils import convert_kps
        from pipeline.phmr_vid import PromptHMR_Video
        from pipeline.tools import detect_track, est_calib

        LOGGER.info("PromptHMR: detecting and tracking the person (YOLO11x + ByteTrack).")
        tracks = detect_track(frames, bbox_interp=True)
        if not tracks:
            raise RuntimeError("PromptHMR: no person detected in the clip.")
        track = _primary_track(tracks, num_frames, height, width)

        LOGGER.info("PromptHMR: estimating 2D keypoints (ViTPose).")
        vit_model = load_vit_model(model_path="data/pretrain/vitpose-h-coco_25.pth")
        kpts_2d = estimate_kp2ds_from_bbox_vitpose(
            vit_model, frames, track["bboxes"], track["track_id"], track["frames"]
        )
        kpts_2d = convert_kps(kpts_2d, "vitpose25", "openpose")
        track["keypoints_2d"] = kpts_2d
        track["vitpose"] = convert_kps(kpts_2d, "ophandface", "cocoophf")
        del vit_model

        calib = est_calib(frames)
        camera = {
            "img_focal": float(calib[0]),
            "img_center": np.array([calib[2], calib[3]], dtype=np.float32),
            "pred_cam_R": np.eye(3, dtype=np.float32)[None].repeat(num_frames, 0),
            "pred_cam_T": np.zeros((num_frames, 3), dtype=np.float32),
        }
        results = {"people": {track["track_id"]: track}, "camera": camera}
        LOGGER.info("PromptHMR: running the image model + video head.")
        results = PromptHMR_Video().run(frames, results, mask_prompt=False)
        smplx_cam = results["people"][track["track_id"]]["smplx_cam"]
        observed_keypoints_2d = torch.as_tensor(track["vitpose"], dtype=torch.float32)

    pose = torch.as_tensor(np.asarray(smplx_cam["pose"]), dtype=torch.float32)
    betas = torch.as_tensor(np.asarray(smplx_cam["shape"]), dtype=torch.float32)
    cam_trans = torch.as_tensor(np.asarray(smplx_cam["trans"]), dtype=torch.float32)
    if pose.shape[0] != num_frames:
        raise RuntimeError(f"PromptHMR produced {pose.shape[0]} frames, expected {num_frames}.")
    if observed_keypoints_2d.shape != (num_frames, 17, 3):
        raise RuntimeError(
            f"PromptHMR keypoints have shape {tuple(observed_keypoints_2d.shape)}, expected {(num_frames, 17, 3)}."
        )

    root_pose = pose[:, :3]
    body_pose = pose[:, _BODY_POSE_SLICE]
    LOGGER.info("PromptHMR: aligning %d frames to gravity.", num_frames)
    root_world, transl_world, _ = _gravity_align(
        root_pose.to(device), body_pose.to(device), betas.to(device), cam_trans.to(device), device, gvhmr_root
    )

    result = MotionResult(
        gvhmr_revision=gvhmr_revision,
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
            # safetensors rejects tensors aliasing smpl_params_global storage.
            "body_pose": body_pose.clone(),
            "betas": betas.clone(),
            "global_orient": root_pose,
            "transl": cam_trans,
        },
        K_fullimg=_camera_matrix(calib, num_frames),
        observed_keypoints_2d=observed_keypoints_2d,
    )
    result.validate(expected_frames=num_frames)
    LOGGER.info("PromptHMR: motion ready for %d frames in %.1fs.", num_frames, time.monotonic() - started)
    return result
