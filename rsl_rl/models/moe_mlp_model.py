# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel, _OnnxMLPModel, _TorchMLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP, MoEMLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


class MoEMLPModel(MLPModel):
    """MLP-based model with a mixture-of-experts backbone."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 4,
        gate_hidden_dims: tuple[int, ...] | list[int] | None = None,
        top_k: int | None = None,
    ) -> None:
        nn.Module.__init__(self)

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.mlp = MoEMLP(
            self._get_latent_dim(),
            mlp_output_dim,
            hidden_dims,
            num_experts=num_experts,
            activation=activation,
            gate_hidden_dims=gate_hidden_dims,
            top_k=top_k,
        )

        if self.distribution is not None:
            self._init_moe_distribution_weights()

    def _init_moe_distribution_weights(self) -> None:
        """Initialize distribution-specific weights on each MoE expert."""
        if not isinstance(self.mlp, MoEMLP):
            self.distribution.init_mlp_weights(self.mlp)  # type: ignore
            return

        for expert in self.mlp.experts:
            self.distribution.init_mlp_weights(expert)  # type: ignore

    def as_jit(self) -> nn.Module:
        return _TorchMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxMLPModel(self, verbose)
