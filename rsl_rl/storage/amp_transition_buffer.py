# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Circular policy-transition replay for LS-AMP discriminator updates."""

from __future__ import annotations

import torch


class AmpTransitionBuffer:
    """Stores (state, next_state, motion_label) transitions from the policy."""

    def __init__(self, capacity: int, state_dim: int, label_dim: int, device: str):
        self.capacity = capacity
        self.states = torch.zeros(capacity, state_dim, device=device)
        self.next_states = torch.zeros_like(self.states)
        self.motion_label = torch.zeros(capacity, label_dim, device=device)
        self.cursor = 0
        self.size = 0

    @torch.no_grad()
    def insert(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        motion_label: torch.Tensor,
    ) -> None:
        if states.numel() == 0:
            return
        if len(states) >= self.capacity:
            states = states[-self.capacity :]
            next_states = next_states[-self.capacity :]
            motion_label = motion_label[-self.capacity :]
        count = len(states)
        first = min(count, self.capacity - self.cursor)
        self.states[self.cursor : self.cursor + first].copy_(states[:first])
        self.next_states[self.cursor : self.cursor + first].copy_(next_states[:first])
        self.motion_label[self.cursor : self.cursor + first].copy_(motion_label[:first])
        remaining = count - first
        if remaining:
            self.states[:remaining].copy_(states[first:])
            self.next_states[:remaining].copy_(next_states[first:])
            self.motion_label[:remaining].copy_(motion_label[first:])
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("AMP transition buffer is empty.")
        index = torch.randint(self.size, (batch_size,), device=self.states.device)
        return self.states[index], self.next_states[index], self.motion_label[index]
