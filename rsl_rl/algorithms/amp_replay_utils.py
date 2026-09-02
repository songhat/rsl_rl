# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared AMP discriminator replay helpers."""

from __future__ import annotations

import torch

from rsl_rl.models.discriminator_model import AMPDiscriminator, LossType


def compute_amp_disc_loss(
    amp_discriminator: AMPDiscriminator,
    loss_type: LossType,
    agent_logits: torch.Tensor,
    demo_logits: torch.Tensor,
    demo_data_flat: torch.Tensor,
    agent_data_flat: torch.Tensor,
    grad_penalty_scale: float,
    apply_grad_penalty: bool = True,
    gp_batch_frac: float = 1.0,
) -> tuple:
    """Compute AMP discriminator classification loss and optional gradient penalty."""
    import torch.nn as nn

    if loss_type == LossType.GAN:
        bce = nn.BCEWithLogitsLoss()
        policy_loss = bce(agent_logits, torch.zeros_like(agent_logits, device=agent_logits.device))
        demo_loss = bce(demo_logits, torch.ones_like(demo_logits, device=demo_logits.device))
        disc_loss = 0.5 * (policy_loss + demo_loss)
        gp_data = demo_data_flat
        target_norm = 0.0
    elif loss_type == LossType.LSGAN:
        policy_loss = nn.MSELoss()(agent_logits, -1 * torch.ones_like(agent_logits, device=agent_logits.device))
        demo_loss = nn.MSELoss()(demo_logits, torch.ones_like(demo_logits, device=demo_logits.device))
        disc_loss = 0.5 * (policy_loss + demo_loss)
        gp_data = demo_data_flat
        target_norm = 0.0
    elif loss_type == LossType.WGAN:
        # L = -(E[C(demo)] - E[C(agent)]) + λ E[(||∇_x̂ C(x̂)||_2 - 1)^2]
        # x̂ = ε·demo + (1-ε)·agent, ε ~ U[0, 1]
        wasserstein = torch.mean(demo_logits) - torch.mean(agent_logits)
        disc_loss = -wasserstein
        n = min(demo_data_flat.shape[0], agent_data_flat.shape[0])
        eps = torch.rand(n, 1, device=demo_data_flat.device, dtype=demo_data_flat.dtype)
        gp_data = eps * demo_data_flat[:n] + (1.0 - eps) * agent_data_flat[:n]
        target_norm = 1.0
    else:
        raise ValueError(f"Unknown AMP loss type: {loss_type}.")

    if not apply_grad_penalty or grad_penalty_scale <= 0.0:
        zero = torch.zeros((), device=agent_logits.device, dtype=agent_logits.dtype)
        return disc_loss, zero

    if 0.0 < gp_batch_frac < 1.0 and gp_data.shape[0] > 1:
        k = max(1, int(gp_data.shape[0] * gp_batch_frac))
        idx = torch.randint(0, gp_data.shape[0], (k,), device=gp_data.device)
        gp_data = gp_data[idx]

    disc_grad_penalty = amp_discriminator.compute_grad_penalty(
        demo_data=gp_data,
        scale=grad_penalty_scale,
        target_norm=target_norm,
    )
    disc_loss = disc_loss + disc_grad_penalty
    return disc_loss, disc_grad_penalty
