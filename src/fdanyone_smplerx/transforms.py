"""Rotation / soft-argmax helpers copied from SMPLer-X ``common/utils/transforms.py``."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _matrix_to_axis_angle_pure(rot_mat: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) axis-angle, pure PyTorch (Rodrigues)."""
    shape = rot_mat.shape[:-2]
    rot_mat = rot_mat.reshape(-1, 3, 3)
    trace = rot_mat.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    skew = torch.stack(
        (rot_mat[:, 2, 1] - rot_mat[:, 1, 2], rot_mat[:, 0, 2] - rot_mat[:, 2, 0], rot_mat[:, 1, 0] - rot_mat[:, 0, 1]),
        dim=-1,
    )
    skew_norm = skew.norm(dim=-1)
    diag = torch.stack((rot_mat[:, 0, 0], rot_mat[:, 1, 1], rot_mat[:, 2, 2]), dim=-1)
    pi_axis = F.normalize(torch.sqrt((diag + 1.0).clamp(min=1e-8)), dim=-1)
    near_zero = skew_norm < 1e-8
    axis = torch.where(near_zero.unsqueeze(-1), pi_axis, skew / skew_norm.unsqueeze(-1).clamp_min(1e-8))
    return (axis * angle.unsqueeze(-1)).reshape(*shape, 3)


def matrix_to_axis_angle(rot_mat: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) axis-angle. Prefers pytorch3d when available."""
    try:
        from pytorch3d.transforms import matrix_to_axis_angle as _m2aa

        return _m2aa(rot_mat)
    except Exception:  # pragma: no cover - fallback path
        return _matrix_to_axis_angle_pure(rot_mat)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3). Prefers pytorch3d when available."""
    try:
        from pytorch3d.transforms import axis_angle_to_matrix as _a2m

        return _a2m(axis_angle)
    except Exception:  # pragma: no cover - fallback path
        return _axis_angle_to_matrix_pure(axis_angle)


def _axis_angle_to_matrix_pure(axis_angle: torch.Tensor) -> torch.Tensor:
    shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)
    angle = aa.norm(dim=-1, keepdim=True)
    axis = aa / angle.clamp_min(1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = torch.cos(angle[:, 0])
    s = torch.sin(angle[:, 0])
    C = 1 - c
    R = torch.stack(
        (
            c + x * x * C,
            x * y * C - z * s,
            x * z * C + y * s,
            y * x * C + z * s,
            c + y * y * C,
            y * z * C - x * s,
            z * x * C - y * s,
            z * y * C + x * s,
            c + z * z * C,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    zero = angle < 1e-8
    R = torch.where(zero.unsqueeze(-1).unsqueeze(-1), torch.eye(3, device=R.device), R)
    return R.reshape(*shape, 3, 3)


def rot6d_to_axis_angle(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3) axis-angle. Mirrors SMPLer-X ``rot6d_to_axis_angle``."""
    orig_shape = x.shape
    x = x.reshape(-1, 6).view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # (N,3,3)
    axis_angle = matrix_to_axis_angle(rot_mat)
    axis_angle = torch.nan_to_num(axis_angle, nan=0.0)
    return axis_angle.reshape(*orig_shape[:-1], 3)


def soft_argmax_3d(heatmap3d: torch.Tensor) -> torch.Tensor:
    """(B, J, D, H, W) -> (B, J, 3) via soft-argmax over the volume."""
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(width, device=heatmap3d.device).float()[None, None, :]
    accu_y = accu_y * torch.arange(height, device=heatmap3d.device).float()[None, None, :]
    accu_z = accu_z * torch.arange(depth, device=heatmap3d.device).float()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    return torch.cat((accu_x, accu_y, accu_z), dim=2)
