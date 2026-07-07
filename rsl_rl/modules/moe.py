# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.mlp import MLP
from rsl_rl.utils import resolve_nn_activation


class MoEMLP(nn.Module):
    """Mixture-of-experts feed-forward network with softmax gating."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        num_experts: int,
        activation: str = "elu",
        gate_hidden_dims: tuple[int, ...] | list[int] | None = None,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}.")
        self.num_experts = num_experts
        self.top_k = num_experts if top_k is None else min(top_k, num_experts)

        gate_hidden_dims = list(gate_hidden_dims or (max(64, input_dim // 2),))
        activation_mod = resolve_nn_activation(activation)
        gate_layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in gate_hidden_dims:
            gate_layers.append(nn.Linear(in_dim, hidden_dim))
            gate_layers.append(activation_mod)
            in_dim = hidden_dim
        gate_layers.append(nn.Linear(in_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)

        self.experts = nn.ModuleList(
            MLP(input_dim, output_dim, hidden_dims, activation) for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        if self.top_k < self.num_experts:
            top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
            weights = torch.zeros_like(logits).scatter(-1, top_idx, torch.softmax(top_vals, dim=-1))
        else:
            weights = torch.softmax(logits, dim=-1)

        expert_out = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return torch.sum(weights.unsqueeze(-1) * expert_out, dim=-2)

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, top_k={self.top_k}"
