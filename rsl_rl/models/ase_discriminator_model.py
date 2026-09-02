# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.models.discriminator_model import AMPDiscriminator, LossType
from rsl_rl.utils import resolve_nn_activation


class ASEDiscriminatorEncoder(AMPDiscriminator):
    """AMP discriminator + independent von Mises–Fisher skill encoder (ASE / MimicKit).

    Encoder is a **separate** MLP (not sharing the disc trunk), matching MimicKit.
    Encoder mean ``mu_q`` is L2-normalized; log-likelihood ∝ ``mu · z``.
    """

    def __init__(
        self,
        disc_obs_dim: int,
        disc_obs_steps: int,
        obs_groups: dict,
        *,
        latent_dim: int = 64,
        encoder_scale: float = 1.0,
        mi_hypersphere_reward_shift: bool = False,
        enc_hidden_dims: list[int] | tuple[int, ...] | None = None,
        loss_type: LossType = LossType.LSGAN,
        hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        activation: str = "relu",
        style_reward_scale: float = 1.0,
        task_style_lerp: float = 0.0,
        reward_steepness: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            disc_obs_dim=disc_obs_dim,
            disc_obs_steps=disc_obs_steps,
            obs_groups=obs_groups,
            loss_type=loss_type,
            hidden_dims=hidden_dims,
            activation=activation,
            style_reward_scale=style_reward_scale,
            task_style_lerp=task_style_lerp,
            reward_steepness=reward_steepness,
            device=device,
        )
        self.latent_dim = int(latent_dim)
        self.encoder_scale = float(encoder_scale)
        self.mi_hypersphere_reward_shift = bool(mi_hypersphere_reward_shift)

        enc_dims = list(enc_hidden_dims) if enc_hidden_dims is not None else list(hidden_dims)
        act = resolve_nn_activation(activation)
        enc_layers: list[nn.Module] = []
        curr = self.input_dim
        for h in enc_dims:
            enc_layers.append(nn.Linear(curr, h))
            enc_layers.append(act)
            curr = h
        self.enc_trunk = nn.Sequential(*enc_layers)
        self.enc_linear = nn.Linear(enc_dims[-1], self.latent_dim)
        nn.init.uniform_(self.enc_linear.weight, -0.1, 0.1)
        nn.init.zeros_(self.enc_linear.bias)
        print(f"ASE Encoder MLP: {self.enc_trunk}")
        print(f"ASE Encoder Output Layer: {self.enc_linear}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return unit-normalized encoder mean mu_q for flattened disc obs."""
        h = self.enc_trunk(x)
        mu = self.enc_linear(h)
        return torch.nn.functional.normalize(mu, dim=-1)

    def get_enc_params(self):
        return list(self.enc_trunk.parameters()) + list(self.enc_linear.parameters())

    def get_disc_params(self):
        return list(self.disc_trunk.parameters()) + list(self.disc_linear.parameters())

    def get_disc_logit_weights(self) -> torch.Tensor:
        return torch.flatten(self.disc_linear.weight)

    def forward_disc_and_enc(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x), self.encode(x)

    def compute_mi_reward(self, disc_obs: torch.Tensor, latents: torch.Tensor, dt: float) -> torch.Tensor:
        """MI / skill-discovery reward (ASE eq. 10 second term / MimicKit enc_r)."""
        if disc_obs.ndim != 3:
            raise ValueError("disc_obs must be (N, T, D)")
        was_training = self.training
        with torch.no_grad():
            self.eval()
            flat = disc_obs.reshape(-1, self.disc_obs_dim)
            normed = self.disc_obs_normalizer(flat).reshape(-1, self.disc_obs_steps * self.disc_obs_dim)
            mu = self.encode(normed)
            cos = torch.sum(mu * latents, dim=-1)
            if self.mi_hypersphere_reward_shift:
                mi = 0.5 * (cos + 1.0)
            else:
                # MimicKit: enc_r = clamp_min(μ·z, 0)
                mi = torch.clamp(cos, min=0.0)
            if was_training:
                self.train()
        return dt * self.encoder_scale * mi

    def encoder_neg_log_likelihood(self, disc_obs_flat: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Encoder loss (ASE eq. 13): maximize ``μ·z`` ≡ minimize ``-μ·z``."""
        mu = self.encode(disc_obs_flat)
        return -torch.sum(mu * latents, dim=-1).mean()

    def predict_style_and_mi_reward(
        self,
        disc_obs: torch.Tensor,
        latents: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        style, score = self.predict_style_reward(disc_obs, dt=dt)
        mi = self.compute_mi_reward(disc_obs, latents, dt=dt)
        return style, mi, score
