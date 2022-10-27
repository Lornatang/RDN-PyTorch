# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "RDN",
    "rdn_small_x2", "rdn_small_x3", "rdn_small_x4", "rdn_small_x8",
    "rdn_large_x2", "rdn_large_x3", "rdn_large_x4", "rdn_large_x8",
]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualBlock, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(channels, growth_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rb(x)
        out = torch.cat([identity, out], 1)

        return out


class _ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int, layers: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        rdb = []
        for index in range(layers):
            rdb.append(_ResidualBlock(channels + index * growth_channels, growth_channels))
        self.rdb = nn.Sequential(*rdb)

        # Local Feature Fusion layer
        self.local_feature_fusion = nn.Conv2d(channels + layers * growth_channels, channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rdb(x)
        out = self.local_feature_fusion(out)

        out = torch.add(out, identity)

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out


class RDN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rdb: int = 16,
            num_rb: int = 8,
            growth_channels: int = 64,
            upscale_factor: int = 4,
    ) -> None:
        super(RDN, self).__init__()
        self.num_rdb = num_rdb

        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Second layer
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Residual Dense Blocks
        trunk = []
        for _ in range(num_rdb):
            trunk.append(_ResidualDenseBlock(channels, growth_channels, num_rb))
        self.trunk = nn.Sequential(*trunk)

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(
            nn.Conv2d(int(num_rdb * channels), channels, (1, 1), (1, 1), (0, 0)),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

        # Upscale block
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv3 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.conv2(out1)

        outs = []
        for i in range(self.num_rdb):
            out = self.trunk[i](out)
            outs.append(out)

        out = torch.cat(outs, 1)
        out = self.global_feature_fusion(out)
        out = torch.add(out1, out)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out


def rdn_small_x2(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=16, num_rb=8, growth_channels=64, upscale_factor=2, **kwargs)

    return model


def rdn_small_x3(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=16, num_rb=8, growth_channels=64, upscale_factor=3, **kwargs)

    return model


def rdn_small_x4(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=16, num_rb=8, growth_channels=64, upscale_factor=4, **kwargs)

    return model


def rdn_small_x8(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=16, num_rb=8, growth_channels=64, upscale_factor=8, **kwargs)

    return model


def rdn_large_x2(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=20, num_rb=16, growth_channels=32, upscale_factor=2, **kwargs)

    return model


def rdn_large_x3(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=20, num_rb=16, growth_channels=32, upscale_factor=3, **kwargs)

    return model


def rdn_large_x4(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=20, num_rb=16, growth_channels=32, upscale_factor=4, **kwargs)

    return model


def rdn_large_x8(**kwargs: Any) -> RDN:
    model = RDN(num_rdb=20, num_rb=16, growth_channels=32, upscale_factor=8, **kwargs)

    return model
