# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared AMP discriminator replay helpers."""

from __future__ import annotations

from rsl_rl.models.discriminator_model import AMPDiscriminator, LossType
from rsl_rl.storage import DiscReplayBuffer


def compute_amp_disc_loss(
    amp_discriminator: AMPDiscriminator,
    loss_type: LossType,
    agent_logits: torch.Tensor,
    demo_logits: torch.Tensor,
    demo_data_flat: torch.Tensor,
    grad_penalty_scale: float,
) -> tuple:
    """Compute AMP discriminator classification loss and gradient penalty."""
    import torch
    import torch.nn as nn

    if loss_type == LossType.GAN:
        bce = nn.BCEWithLogitsLoss()
        policy_loss = bce(agent_logits, torch.zeros_like(agent_logits, device=agent_logits.device))
        demo_loss = bce(demo_logits, torch.ones_like(demo_logits, device=demo_logits.device))
        disc_loss = 0.5 * (policy_loss + demo_loss)
    elif loss_type == LossType.LSGAN:
        policy_loss = nn.MSELoss()(agent_logits, -1 * torch.ones_like(agent_logits, device=agent_logits.device))
        demo_loss = nn.MSELoss()(demo_logits, torch.ones_like(demo_logits, device=demo_logits.device))
        disc_loss = 0.5 * (policy_loss + demo_loss)
    elif loss_type == LossType.WGAN:
        disc_loss = -torch.mean(demo_logits) + torch.mean(agent_logits)
    else:
        raise ValueError(f"Unknown AMP loss type: {loss_type}.")

    disc_grad_penalty = amp_discriminator.compute_grad_penalty(
        demo_data=demo_data_flat,
        scale=grad_penalty_scale,
    )
    return disc_loss, disc_grad_penalty
