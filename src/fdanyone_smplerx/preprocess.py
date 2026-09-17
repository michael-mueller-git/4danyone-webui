"""Person-crop preprocessing mirroring SMPLer-X ``main/inference.py``."""

from __future__ import annotations

import cv2
import numpy as np

from fdanyone_smplerx.config import INPUT_IMG_SHAPE

_ASPECT = INPUT_IMG_SHAPE[1] / INPUT_IMG_SHAPE[0]  # 384 / 512


def sanitize_bbox(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray | None:
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_width - 1, x1 + max(0, w - 1))
    y2 = min(img_height - 1, y1 + max(0, h - 1))
    if w * h > 0 and x2 > x1 and y2 > y1:
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
    return None


def process_bbox(bbox: np.ndarray, img_width: int, img_height: int, ratio: float = 1.25) -> np.ndarray | None:
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return None
    w, h = bbox[2], bbox[3]
    c_x, c_y = bbox[0] + w / 2.0, bbox[1] + h / 2.0
    if w > _ASPECT * h:
        h = w / _ASPECT
    elif w < _ASPECT * h:
        w = h * _ASPECT
    bbox[2] = w * ratio
    bbox[3] = h * ratio
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0
    return bbox.astype(np.float32)


def _rotate_2d(pt: np.ndarray, rot_rad: float) -> np.ndarray:
    x, y = pt
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)


def gen_trans_from_patch_cv(
    c_x: float, c_y: float, src_width: float, src_height: float, dst_width: int, dst_height: int, scale: float, rot: float, inv: bool = False
) -> np.ndarray:
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_down = _rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_right = _rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_down = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_right = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = src_center
    src[1] = src_center + src_down
    src[2] = src_center + src_right
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = dst_center
    dst[1] = dst_center + dst_down
    dst[2] = dst_center + dst_right

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans.astype(np.float32)


def generate_patch_image(
    rgb: np.ndarray, bbox: np.ndarray, out_shape: tuple[int, int] = INPUT_IMG_SHAPE, scale: float = 1.0, rot: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_height, img_width = rgb.shape[:2]
    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(rgb, trans, (out_shape[1], out_shape[0]), flags=cv2.INTER_LINEAR)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)
    return img_patch.astype(np.float32), trans, inv_trans


def crop_body(rgb: np.ndarray, bbox_xyxy: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Crop a person patch from an RGB frame given an xyxy detection bbox.

    Returns (patch (512,384,3) float RGB, processed bbox [x,y,w,h]) or (None, None).
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bbox = np.array([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float32)
    bbox = process_bbox(bbox, rgb.shape[1], rgb.shape[0])
    if bbox is None:
        return None, None
    patch, _, _ = generate_patch_image(rgb, bbox)
    return patch, bbox
