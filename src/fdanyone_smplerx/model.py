"""Body-only SMPLer-X regressor: PositionNet + BodyRotationNet + ViT encoder.

Port of ``common/nets/smpler_x.py`` and the body path of
``main/SMPLer_X.py``. Hands / face / box branches are intentionally omitted;
the pipeline only consumes body SMPL-X parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from fdanyone_smplerx.config import (
    BODY_ORIG_JN,
    BODY_POS_JN,
    CAMERA_3D_SIZE,
    FEAT_DIM,
    FOCAL,
    INPUT_BODY_SHAPE,
    OUTPUT_HM_SHAPE,
)
from fdanyone_smplerx.encoder import ViT
from fdanyone_smplerx.layers import make_conv_layers, make_linear_layers
from fdanyone_smplerx.transforms import rot6d_to_axis_angle, soft_argmax_3d


class PositionNet(nn.Module):
    """Body joint-position head: image features -> 3D joint volume."""

    def __init__(self, feat_dim: int = FEAT_DIM, joint_num: int = BODY_POS_JN):
        super().__init__()
        self.joint_num = joint_num
        self.hm_shape = OUTPUT_HM_SHAPE  # (D, H, W)
        self.conv = make_conv_layers([feat_dim, joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d, h, w = self.hm_shape
        joint_hm = self.conv(img_feat).view(-1, self.joint_num, d, h, w)
        joint_coord = soft_argmax_3d(joint_hm)
        return joint_hm, joint_coord


class BodyRotationNet(nn.Module):
    """Body regressor: task tokens + joint coords -> root/body pose, shape, cam."""

    def __init__(self, feat_dim: int = FEAT_DIM, joint_num: int = BODY_POS_JN, orig_joint_num: int = BODY_ORIG_JN):
        super().__init__()
        self.body_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.root_pose_out = make_linear_layers([joint_num * (512 + 3), 6], relu_final=False)
        self.body_pose_out = make_linear_layers([joint_num * (512 + 3), (orig_joint_num - 1) * 6], relu_final=False)
        self.shape_out = make_linear_layers([feat_dim, 10], relu_final=False)
        self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)

    def forward(
        self,
        body_pose_token: torch.Tensor,
        shape_token: torch.Tensor,
        cam_token: torch.Tensor,
        body_joint_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = body_pose_token.shape[0]
        shape_param = self.shape_out(shape_token)
        cam_param = self.cam_out(cam_token)
        token = self.body_conv(body_pose_token)
        token = torch.cat((token, body_joint_img), 2)
        root_pose = self.root_pose_out(token.view(batch_size, -1))
        body_pose = self.body_pose_out(token.view(batch_size, -1))
        return root_pose, body_pose, shape_param, cam_param


class SMPLXRegressor(nn.Module):
    """Body-only SMPLer-X forward: (B,3,256,192) in [0,1] -> SMPL-X params."""

    def __init__(self):
        super().__init__()
        self.encoder = ViT()
        self.body_position_net = PositionNet()
        self.body_regressor = BodyRotationNet()
        self._k_value = math.sqrt(
            FOCAL[0] * FOCAL[1] * CAMERA_3D_SIZE * CAMERA_3D_SIZE / (INPUT_BODY_SHAPE[0] * INPUT_BODY_SHAPE[1])
        )

    def get_camera_trans(self, cam_param: torch.Tensor) -> torch.Tensor:
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])
        t_z = self._k_value * gamma
        return torch.cat((t_xy, t_z[:, None]), 1)

    def forward(self, body_img: torch.Tensor) -> dict[str, torch.Tensor]:
        """body_img: (B, 3, 256, 192), values in [0, 1], RGB order."""
        img_feat, task_tokens = self.encoder(body_img)
        shape_token = task_tokens[:, 0]
        cam_token = task_tokens[:, 1]
        body_pose_token = task_tokens[:, 6:]  # 25 body tokens
        _, body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_param = self.body_regressor(
            body_pose_token, shape_token, cam_token, body_joint_img.detach()
        )
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)
        cam_trans = self.get_camera_trans(cam_param)
        return {
            "root_pose": root_pose,  # (B, 3)
            "body_pose": body_pose,  # (B, 63)
            "shape": shape,  # (B, 10)
            "cam_trans": cam_trans,  # (B, 3)
        }
