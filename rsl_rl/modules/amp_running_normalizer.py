# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch-native running statistics for LS-AMP discriminator inputs."""

from __future__ import annotations

import torch
from torch import nn


class AmpRunningNormalizer(nn.Module):
    def __init__(self, size: int, epsilon: float = 1.0e-4, clip: float = 10.0):
        super().__init__()
        self.epsilon = epsilon
        self.clip = clip
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("var", torch.ones(size, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().to(dtype=torch.float64)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        var = self.var.to(device=values.device, dtype=values.dtype)
        return torch.clamp(
            (values - mean) / torch.sqrt(var + self.epsilon), -self.clip, self.clip
        )
