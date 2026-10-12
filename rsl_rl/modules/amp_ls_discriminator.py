# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Least-squares AMP discriminator (DroidUp / (s, s') style)."""

from __future__ import annotations

import torch
from torch import autograd, nn


def _is_arm_name(name: str) -> bool:
    return any(token in name for token in ("shoulder", "elbow", "wrist", "hand"))


def _is_leg_name(name: str) -> bool:
    return any(token in name for token in ("hip", "knee", "ankle", "foot"))


def build_amp_part_indices(
    joint_names: tuple[str, ...] | list[str],
    key_body_names: tuple[str, ...] | list[str],
) -> dict[str, list[int]]:
    """Slice AMP obs into root / legs / arms channel indices."""
    joint_names = tuple(joint_names)
    key_body_names = tuple(key_body_names)
    num_joints = len(joint_names)
    num_keys = len(key_body_names)
    root = list(range(0, 6))
    q0 = 6
    dq0 = q0 + num_joints
    key_pos0 = dq0 + num_joints
    key_ori0 = key_pos0 + 3 * num_keys

    leg_joints = [i for i, name in enumerate(joint_names) if _is_leg_name(name)]
    arm_joints = [i for i, name in enumerate(joint_names) if _is_arm_name(name)]
    other_joints = [
        i for i in range(num_joints) if i not in leg_joints and i not in arm_joints
    ]
    # Waist / unused joints stay with root so they are not dropped.
    root.extend(q0 + i for i in other_joints)
    root.extend(dq0 + i for i in other_joints)

    leg_keys = [i for i, name in enumerate(key_body_names) if _is_leg_name(name)]
    arm_keys = [i for i, name in enumerate(key_body_names) if _is_arm_name(name)]
    other_keys = [i for i in range(num_keys) if i not in leg_keys and i not in arm_keys]
    root.extend(key_pos0 + 3 * i + j for i in other_keys for j in range(3))
    root.extend(key_ori0 + 6 * i + j for i in other_keys for j in range(6))

    legs = (
        [q0 + i for i in leg_joints]
        + [dq0 + i for i in leg_joints]
        + [key_pos0 + 3 * i + j for i in leg_keys for j in range(3)]
        + [key_ori0 + 6 * i + j for i in leg_keys for j in range(6)]
    )
    arms = (
        [q0 + i for i in arm_joints]
        + [dq0 + i for i in arm_joints]
        + [key_pos0 + 3 * i + j for i in arm_keys for j in range(3)]
        + [key_ori0 + 6 * i + j for i in arm_keys for j in range(6)]
    )
    return {"root": root, "legs": legs, "arms": arms}


def _fuse_mlp(widths: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_w, out_w in zip(widths[:-1], widths[1:], strict=True):
        layers.extend((nn.Linear(in_w, out_w), nn.ReLU()))
    # Drop trailing ReLU so the last hidden feeds the score head.
    return nn.Sequential(*layers[:-1])


class AmpLSDiscriminator(nn.Module):
    """Maps concatenated (state, next_state) to a scalar LS-GAN score."""

    def __init__(self, state_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        width = 2 * state_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(torch.cat((state, next_state), dim=1)))

    def style_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        coefficient: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self(state, next_state).squeeze(1)
        reward = coefficient * torch.clamp(
            1.0 - 0.25 * torch.square(prediction - 1.0), min=0.0
        )
        return reward, prediction

    def gradient_penalty(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        coefficient: float = 10.0,
    ) -> torch.Tensor:
        state = expert_state.detach().requires_grad_(True)
        next_state = expert_next_state.detach().requires_grad_(True)
        prediction = self(state, next_state)
        gradient = autograd.grad(
            prediction,
            (state, next_state),
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        joined = torch.cat(gradient, dim=1)
        return coefficient * joined.norm(2, dim=1).square().mean()


class AmpLSPartGateDiscriminator(nn.Module):
    """Label-conditioned part-gate discriminator over root / legs / arms.

    Each part encodes ``concat(s_g, s'_g)``; a softmax gate over parts is produced
    from the one-hot motion label only (query does not see the body). No attention
    prior / arm downscaling is applied here.
    """

    PART_ORDER = ("root", "legs", "arms")

    def __init__(
        self,
        amp_obs_dim: int,
        label_dim: int,
        joint_names: tuple[str, ...] | list[str],
        key_body_names: tuple[str, ...] | list[str],
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
        d_model: int = 128,
    ):
        super().__init__()
        self.amp_obs_dim = int(amp_obs_dim)
        self.label_dim = int(label_dim)
        self.d_model = int(d_model)
        indices = build_amp_part_indices(joint_names, key_body_names)
        self.part_names = tuple(name for name in self.PART_ORDER if len(indices[name]) > 0)
        if not self.part_names:
            raise ValueError("AMP part-gate discriminator found no non-empty parts.")
        # Buffers move with ``.to(device)`` at construction, outside inference mode;
        # lazily-moved index tensors become inference tensors and break backward.
        for name in self.part_names:
            self.register_buffer(
                f"_part_index_{name}",
                torch.tensor(indices[name], dtype=torch.long),
                persistent=False,
            )
        encoders = {
            name: nn.Sequential(
                nn.Linear(2 * len(indices[name]), d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )
            for name in self.part_names
        }
        # ``trunk`` / ``head`` keep the AmpPPO Adam param-group split.
        self.trunk = nn.ModuleDict(
            {
                "encoders": nn.ModuleDict(encoders),
                "gate": nn.Linear(label_dim, len(self.part_names)),
                "fuse": _fuse_mlp([d_model, *hidden_dims]),
            }
        )
        self.head = nn.Linear(hidden_dims[-1], 1)
        self.last_attention: torch.Tensor | None = None

    def _split_body(self, body: torch.Tensor) -> dict[str, torch.Tensor]:
        if body.shape[-1] != self.amp_obs_dim:
            raise ValueError(
                f"Expected AMP body width {self.amp_obs_dim}, got {body.shape[-1]}."
            )
        return {
            name: body.index_select(-1, getattr(self, f"_part_index_{name}"))
            for name in self.part_names
        }

    def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.amp_obs_dim + self.label_dim:
            raise ValueError(
                f"Expected state width {self.amp_obs_dim + self.label_dim}, "
                f"got {state.shape[-1]}."
            )
        body_s = state[..., : self.amp_obs_dim]
        body_n = next_state[..., : self.amp_obs_dim]
        label = state[..., self.amp_obs_dim :]
        parts_s = self._split_body(body_s)
        parts_n = self._split_body(body_n)
        tokens = [
            self.trunk["encoders"][name](
                torch.cat((parts_s[name], parts_n[name]), dim=-1)
            )
            for name in self.part_names
        ]
        stacked = torch.stack(tokens, dim=1)  # [B, P, D]
        alpha = torch.softmax(self.trunk["gate"](label), dim=-1)  # [B, P]
        self.last_attention = alpha.detach()
        fused = torch.sum(alpha.unsqueeze(-1) * stacked, dim=1)
        return self.head(self.trunk["fuse"](fused))

    def style_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        coefficient: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self(state, next_state).squeeze(1)
        reward = coefficient * torch.clamp(
            1.0 - 0.25 * torch.square(prediction - 1.0), min=0.0
        )
        return reward, prediction

    def gradient_penalty(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        coefficient: float = 10.0,
    ) -> torch.Tensor:
        state = expert_state.detach().requires_grad_(True)
        next_state = expert_next_state.detach().requires_grad_(True)
        prediction = self(state, next_state)
        gradient = autograd.grad(
            prediction,
            (state, next_state),
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        joined = torch.cat(gradient, dim=1)
        return coefficient * joined.norm(2, dim=1).square().mean()
