"""Small layer builders copied from SMPLer-X ``common/nets/layer.py``."""

from __future__ import annotations

import torch.nn as nn


def make_linear_layers(feat_dims: list[int], relu_final: bool = True, use_bn: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def make_conv_layers(feat_dims: list[int], kernel: int = 3, stride: int = 1, padding: int = 1, bnrelu_final: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Conv2d(feat_dims[i], feat_dims[i + 1], kernel_size=kernel, stride=stride, padding=padding))
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
