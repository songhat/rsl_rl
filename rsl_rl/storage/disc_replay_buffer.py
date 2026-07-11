# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat circular replay buffer for AMP discriminator agent observations."""

from __future__ import annotations

import torch

from .circular_buffer import CircularBuffer


class DiscReplayBuffer:
    """Global flat replay buffer for agent ``disc_obs`` (MimicKit-style).

    Stores tensors with shape ``(num_samples, disc_obs_steps, disc_obs_dim)`` and
    supports uniform random sampling for discriminator training.
    """

    def __init__(self, max_size: int, device: str):
        if max_size < 1:
            raise ValueError(f"Replay buffer size must be >= 1, got {max_size}.")
        self.max_size = max_size
        self.device = device
        self._head = 0
        self._is_full = False
        self._data: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.max_size if self._is_full else self._head

    @property
    def is_full(self) -> bool:
        return self._is_full

    def push(self, samples: torch.Tensor) -> None:
        """Append up to ``max_size`` samples in circular fashion."""
        if samples.numel() == 0:
            return

        samples = samples.to(self.device)
        n = samples.shape[0]
        if n > self.max_size:
            idx = torch.randperm(n, device=self.device)[: self.max_size]
            samples = samples[idx]
            n = self.max_size

        if self._data is None:
            self._data = torch.empty((self.max_size, *samples.shape[1:]), device=self.device, dtype=samples.dtype)

        end = self._head + n
        if end >= self.max_size:
            first = self.max_size - self._head
            self._data[self._head :] = samples[:first]
            self._data[: n - first] = samples[first:]
            self._is_full = True
        else:
            self._data[self._head : end] = samples
        self._head = (self._head + n) % self.max_size

    def sample(self, n: int) -> torch.Tensor:
        if len(self) == 0:
            raise RuntimeError("DiscReplayBuffer is empty.")
        indices = torch.randint(0, len(self), (n,), device=self.device)
        return self._data[indices]  # type: ignore[index]


def flatten_rollout_disc_obs(disc_obs_buffer: CircularBuffer, fetch_length: int) -> torch.Tensor:
    """Flatten the most recent rollout ``disc_obs`` to ``(N, steps, dim)``."""
    if disc_obs_buffer._buffer is None:  # noqa: SLF001
        return torch.empty(0, device=disc_obs_buffer.device)

    current_length = int(torch.min(disc_obs_buffer.current_length).item())
    length = min(fetch_length, current_length)
    if length == 0:
        return torch.empty(0, device=disc_obs_buffer.device)

    rollout = disc_obs_buffer.buffer[:, -length:]
    return rollout.reshape(-1, *rollout.shape[2:])


def store_disc_replay_from_rollout(
    disc_obs_buffer: CircularBuffer,
    replay_buffer: DiscReplayBuffer | None,
    *,
    fetch_length: int,
    replay_samples: int,
) -> int:
    """Push a subset of the current rollout agent ``disc_obs`` into replay."""
    if replay_buffer is None:
        return 0

    flat_disc_obs = flatten_rollout_disc_obs(disc_obs_buffer, fetch_length)
    n = flat_disc_obs.shape[0]
    if n == 0:
        return 0

    if replay_buffer.is_full:
        num_samples = min(n, replay_samples)
    else:
        num_samples = n

    idx = torch.randperm(n, device=flat_disc_obs.device)[:num_samples]
    replay_buffer.push(flat_disc_obs[idx])
    return num_samples


def agent_disc_obs_with_replay(
    amp_discriminator,
    disc_obs_batch: torch.Tensor,
    replay_buffer: DiscReplayBuffer | None,
) -> tuple[torch.Tensor, int]:
    """Normalize and concatenate current agent batch with replay samples."""
    with torch.no_grad():
        current_normed = amp_discriminator.normalize_disc_obs(disc_obs_batch)
    mini_batch_size = current_normed.shape[0]

    if replay_buffer is None or len(replay_buffer) == 0:
        flat = current_normed.reshape(mini_batch_size, -1)
        return flat, mini_batch_size

    replay_obs = replay_buffer.sample(mini_batch_size)
    with torch.no_grad():
        replay_normed = amp_discriminator.normalize_disc_obs(replay_obs)
    merged = torch.cat([current_normed, replay_normed], dim=0)
    return merged.reshape(merged.shape[0], -1), mini_batch_size
