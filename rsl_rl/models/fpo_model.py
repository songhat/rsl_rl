# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, MLP, MoEMLP, HiddenState
from rsl_rl.utils import unpad_trajectories


@dataclass
class FpoActionInfo:
    """CFM training samples collected during action sampling."""

    loss_eps: torch.Tensor
    """Sampled noise for CFM loss, shape ``[batch, num_samples, action_dim]``."""

    loss_t: torch.Tensor
    """Sampled time steps for CFM loss, shape ``[batch, num_samples, 1]``."""

    initial_cfm_loss: torch.Tensor
    """Initial CFM loss per sample, shape ``[batch, num_samples]``."""


class FPOModel(nn.Module):
    """Flow-matching policy model for Flow Policy Optimization (FPO).

    The actor predicts a velocity field conditioned on observations, the current noisy action ``x_t``, and time ``t``.
    Actions are produced by Euler integration from Gaussian noise.
    """

    is_recurrent: bool = False

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
        num_flow_steps: int = 10,
        num_fpo_samples: int = 8,
        fixed_noise_inference: bool = False,
        num_experts: int | None = None,
        gate_hidden_dims: tuple[int, ...] | list[int] | None = None,
        top_k: int | None = None,
    ) -> None:
        if distribution_cfg is not None:
            raise ValueError("FPOModel does not support distribution_cfg. Use MLPModel for Gaussian policies.")
        super().__init__()

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.action_dim = output_dim
        self.num_flow_steps = num_flow_steps
        self.num_fpo_samples = num_fpo_samples
        self.fixed_noise_inference = fixed_noise_inference

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        flow_input_dim = self.obs_dim + self.action_dim + 1
        if num_experts is not None:
            self.flow_mlp = MoEMLP(
                flow_input_dim,
                self.action_dim,
                hidden_dims,
                num_experts=num_experts,
                activation=activation,
                gate_hidden_dims=gate_hidden_dims,
                top_k=top_k,
            )
        else:
            self.flow_mlp = MLP(flow_input_dim, self.action_dim, hidden_dims, activation)

        self._init_noise: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return denoised actions via flow integration (deterministic inference)."""
        del masks, hidden_state, stochastic_output
        obs_latent = self.get_latent(obs)
        return self._integrate_flow(obs_latent)

    def sample_with_info(self, obs: TensorDict) -> tuple[torch.Tensor, FpoActionInfo]:
        """Sample actions and collect CFM training information."""
        obs_latent = self.get_latent(obs)
        batch_size = obs_latent.shape[0]
        device = obs_latent.device

        if self.fixed_noise_inference and self._init_noise is not None:
            eps = self._init_noise.expand(batch_size, -1)
        else:
            eps = torch.randn(batch_size, self.action_dim, device=device)
            if self.fixed_noise_inference and batch_size == 1:
                self._init_noise = eps.detach().clone()

        actions = self._integrate_flow(obs_latent, init_noise=eps)

        num_samples = self.num_fpo_samples
        loss_eps = torch.randn(batch_size, num_samples, self.action_dim, device=device)
        loss_t = torch.rand(batch_size, num_samples, 1, device=device)
        flat_obs = obs_latent.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)
        flat_actions = actions.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)
        flat_eps = loss_eps.reshape(batch_size * num_samples, -1)
        flat_t = loss_t.reshape(batch_size * num_samples, -1)
        initial_cfm_loss = self.compute_cfm_loss(flat_obs, flat_actions, flat_eps, flat_t).view(
            batch_size, num_samples
        )

        return actions, FpoActionInfo(loss_eps=loss_eps, loss_t=loss_t, initial_cfm_loss=initial_cfm_loss.detach())

    def compute_cfm_loss(
        self,
        obs_latent: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample conditional flow matching loss."""
        x_t = (1 - t) * eps + t * actions
        flow_input = torch.cat([obs_latent, x_t, t], dim=-1)
        velocity_pred = self.flow_mlp(flow_input)
        return F.mse_loss(velocity_pred, actions - eps, reduction="none").mean(dim=-1)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        return self.obs_normalizer(latent)

    def _integrate_flow(self, obs_latent: torch.Tensor, init_noise: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = obs_latent.shape[0]
        device = obs_latent.device
        dt = 1.0 / self.num_flow_steps
        x_t = init_noise if init_noise is not None else torch.randn(batch_size, self.action_dim, device=device)

        for step in range(self.num_flow_steps):
            t_val = step * dt
            t_tensor = torch.full((batch_size, 1), t_val, device=device)
            flow_input = torch.cat([obs_latent, x_t, t_tensor], dim=-1)
            x_t = x_t + dt * self.flow_mlp(flow_input)
        return x_t

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_std(self) -> torch.Tensor:
        """Placeholder for runner logging (FPO has no learned action std)."""
        device = next(self.parameters()).device
        return torch.zeros(1, self.action_dim, device=device)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            self.obs_normalizer.update(torch.cat(obs_list, dim=-1))  # type: ignore

    def as_jit(self) -> nn.Module:
        return _TorchFPOModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxFPOModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The FPO model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchFPOModel(nn.Module):
    """Exportable FPO model for JIT."""

    def __init__(self, model: FPOModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.flow_mlp = copy.deepcopy(model.flow_mlp)
        self.num_flow_steps = model.num_flow_steps
        self.action_dim = model.action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        batch_size = x.shape[0]
        device = x.device
        dt = 1.0 / self.num_flow_steps
        actions = torch.randn(batch_size, self.action_dim, device=device)
        for step in range(self.num_flow_steps):
            t_tensor = torch.full((batch_size, 1), step * dt, device=device)
            flow_input = torch.cat([x, actions, t_tensor], dim=-1)
            actions = actions + dt * self.flow_mlp(flow_input)
        return actions

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxFPOModel(nn.Module):
    """Exportable FPO model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: FPOModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.flow_mlp = copy.deepcopy(model.flow_mlp)
        self.num_flow_steps = model.num_flow_steps
        self.action_dim = model.action_dim
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        dt = 1.0 / self.num_flow_steps
        x = self.obs_normalizer(x)
        actions = torch.randn(batch_size, self.action_dim, device=device)
        for step in range(self.num_flow_steps):
            t_tensor = torch.full((batch_size, 1), step * dt, device=device)
            flow_input = torch.cat([x, actions, t_tensor], dim=-1)
            actions = actions + dt * self.flow_mlp(flow_input)
        return actions

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
