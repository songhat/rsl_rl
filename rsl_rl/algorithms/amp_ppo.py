# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Least-squares AMP-PPO (DroidUp style): (s, s') disc + command-conditioned labels."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.algorithms.amp_constants import (
    AMP_KEY_BODY_NAMES,
    AMP_LABEL_DIM,
    AMP_LABEL_NAMES,
    COMMAND_MODE_LATERAL,
    COMMAND_MODE_STANDING,
    COMMAND_MODE_TURNING,
    MJLAB_JOINT_NAMES,
)
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.amp_ls_discriminator import (
    AmpLSDiscriminator,
    AmpLSPartGateDiscriminator,
)
from rsl_rl.modules.amp_running_normalizer import AmpRunningNormalizer
from rsl_rl.storage.amp_transition_buffer import AmpTransitionBuffer
from rsl_rl.utils.amp_ls_motion_loader import AmpLSMotionLoader


class AmpPPO(PPO):
    """PPO with a least-squares AMP discriminator and style/task reward lerp."""

    def __init__(
        self,
        *args,
        amp_motion_files,
        amp_reward_coefficient: float = 0.4,
        amp_task_reward_lerp: float = 0.7,
        amp_discriminator_hidden_dims: tuple[int, ...] = (1024, 512, 256),
        amp_discriminator_type: str = "mlp",
        amp_part_gate_dim: int = 128,
        amp_replay_buffer_size: int = 100_000,
        amp_preload_transitions: int = 1_000_000,
        amp_transition_dt: float = 0.02,
        amp_gradient_penalty: float = 10.0,
        amp_motion_velocity_threshold: float = 0.8,
        amp_motion_weights: tuple[float, ...] = (1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0),
        amp_motion_labels: tuple[str, ...] = (
            "walk",
            "run",
            "run",
            "turn",
            "turn",
            "side",
            "side",
            "stand",
        ),
        amp_joint_names: tuple[str, ...] | None = None,
        amp_key_body_names: tuple[str, ...] | None = None,
        amp_root_dir: str | None = None,
        minimum_action_std: tuple[float, ...] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.amp_motion_files = tuple(amp_motion_files)
        self.amp_reward_coefficient = amp_reward_coefficient
        self.amp_task_reward_lerp = amp_task_reward_lerp
        self.amp_preload_transitions = amp_preload_transitions
        self.amp_transition_dt = amp_transition_dt
        self.amp_gradient_penalty = amp_gradient_penalty
        self.amp_motion_velocity_threshold = amp_motion_velocity_threshold
        self.amp_motion_weights = amp_motion_weights
        self.amp_motion_labels = amp_motion_labels
        self.amp_joint_names = tuple(amp_joint_names or MJLAB_JOINT_NAMES)
        self.amp_key_body_names = tuple(amp_key_body_names or AMP_KEY_BODY_NAMES)
        self.amp_root_dir = amp_root_dir
        self.amp_discriminator_type = str(amp_discriminator_type)
        self.amp_part_gate_dim = int(amp_part_gate_dim)
        self.amp_obs_dim = (
            6 + 2 * len(self.amp_joint_names) + 9 * len(self.amp_key_body_names)
        )
        self.amp_discriminator_state_dim = self.amp_obs_dim + AMP_LABEL_DIM
        self.discriminator = self._build_discriminator(
            tuple(amp_discriminator_hidden_dims)
        ).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3},
                {"params": self.discriminator.head.parameters(), "weight_decay": 1.0e-1},
            ],
            lr=self.learning_rate,
        )
        self.amp_normalizer = AmpRunningNormalizer(self.amp_obs_dim).to(self.device)
        self.amp_replay = AmpTransitionBuffer(
            amp_replay_buffer_size,
            self.amp_discriminator_state_dim,
            AMP_LABEL_DIM,
            self.device,
        )
        self.amp_data: AmpLSMotionLoader | None = None
        self._current_amp_state: torch.Tensor | None = None
        self._current_motion_label: torch.Tensor | None = None
        self.minimum_action_std = (
            torch.tensor(minimum_action_std, device=self.device)
            if minimum_action_std is not None
            else None
        )
        self._env = None

    def _build_discriminator(self, hidden_dims: tuple[int, ...]) -> nn.Module:
        if self.amp_discriminator_type in ("mlp", "ls", "default"):
            return AmpLSDiscriminator(self.amp_discriminator_state_dim, hidden_dims)
        if self.amp_discriminator_type in ("part_gate", "attention", "part_attention"):
            return AmpLSPartGateDiscriminator(
                amp_obs_dim=self.amp_obs_dim,
                label_dim=AMP_LABEL_DIM,
                joint_names=self.amp_joint_names,
                key_body_names=self.amp_key_body_names,
                hidden_dims=hidden_dims,
                d_model=self.amp_part_gate_dim,
            )
        raise ValueError(
            f"Unknown amp_discriminator_type={self.amp_discriminator_type!r}; "
            "expected 'mlp' or 'part_gate'."
        )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str):
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        if not isinstance(alg, AmpPPO):
            raise TypeError(f"Expected AmpPPO, got {type(alg).__name__}.")
        alg._env = env
        return alg

    def _ensure_amp_data(self) -> AmpLSMotionLoader:
        if self.amp_data is None:
            self.amp_data = AmpLSMotionLoader(
                self.amp_motion_files,
                self.device,
                self.amp_transition_dt,
                self.amp_preload_transitions,
                self.amp_motion_weights,
                self.amp_motion_labels,
                joint_names=self.amp_joint_names,
                key_body_names=self.amp_key_body_names,
                root_dir=self.amp_root_dir,
            )
        return self.amp_data

    def _command_motion_label(self, command: torch.Tensor) -> torch.Tensor:
        """Map command modes to a five-class one-hot AMP label."""
        term = self._env.unwrapped.command_manager.get_term("twist")
        command_mode = getattr(term, "command_mode", None)
        if command_mode is None:
            raise RuntimeError("AMP requires UniformVelocityCommand.command_mode.")
        label_ids = (command[:, 0] > self.amp_motion_velocity_threshold).to(torch.long)
        label_ids = torch.where(
            command_mode == COMMAND_MODE_TURNING,
            torch.full_like(command_mode, AMP_LABEL_NAMES.index("turn")),
            label_ids,
        )
        label_ids = torch.where(
            command_mode == COMMAND_MODE_LATERAL,
            torch.full_like(command_mode, AMP_LABEL_NAMES.index("side")),
            label_ids,
        )
        command_magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(
            command[:, 2]
        )
        standing = (command_mode == COMMAND_MODE_STANDING) | (command_magnitude <= 0.1)
        label_ids = torch.where(
            standing,
            torch.full_like(command_mode, AMP_LABEL_NAMES.index("stand")),
            label_ids,
        )
        return torch.nn.functional.one_hot(label_ids, num_classes=AMP_LABEL_DIM).to(
            dtype=torch.float32
        )

    def act(self, obs):
        if self._env is None:
            raise RuntimeError("AMP environment was not attached to the algorithm.")
        self._current_amp_state = obs["amp"].detach().clone()
        command = self._env.unwrapped.command_manager.get_command("twist")
        if command is None:
            raise RuntimeError("AMP requires the 'twist' velocity command.")
        self._current_motion_label = self._command_motion_label(command).detach().clone()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self._current_amp_state is None:
            raise RuntimeError("AMP state was not captured before env.step().")
        if self._current_motion_label is None:
            raise RuntimeError("AMP motion label was not captured before env.step().")
        next_state = obs["amp"].detach()
        label = self._current_motion_label
        current_state = torch.cat((self._current_amp_state, label), dim=1)
        next_state = torch.cat((next_state, label), dim=1)
        valid = dones == 0
        self.amp_replay.insert(
            current_state[valid],
            next_state[valid],
            self._current_motion_label[valid],
        )
        with torch.no_grad():
            state_norm = torch.cat(
                (self.amp_normalizer.normalize(self._current_amp_state), label), dim=1
            )
            next_norm = torch.cat(
                (self.amp_normalizer.normalize(obs["amp"].detach()), label), dim=1
            )
            style_reward, prediction = self.discriminator.style_reward(
                state_norm, next_norm, self.amp_reward_coefficient
            )
            style_reward = torch.where(valid, style_reward, torch.zeros_like(style_reward))
            task_reward = rewards.clone()
            combined = (
                self.amp_task_reward_lerp * task_reward
                + (1.0 - self.amp_task_reward_lerp) * style_reward
            )
            rewards.copy_(combined)
            log = extras.setdefault("log", {})
            log["AMP/style_reward"] = style_reward.mean()
            log["AMP/task_reward"] = task_reward.mean()
            log["AMP/discriminator_prediction"] = prediction.mean()
        super().process_env_step(obs, rewards, dones, extras)

    def _enforce_minimum_std(self) -> None:
        if self.minimum_action_std is None:
            return
        distribution = self._raw_actor.distribution
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                distribution.std_param.copy_(
                    torch.maximum(distribution.std_param, self.minimum_action_std)
                )
            elif hasattr(distribution, "log_std_param"):
                distribution.log_std_param.copy_(
                    torch.maximum(
                        distribution.log_std_param, torch.log(self.minimum_action_std)
                    )
                )
            else:
                raise TypeError("minimum_action_std requires a Gaussian actor distribution.")

    def update(self) -> dict[str, float]:
        batch_size = (
            self.storage.num_envs
            * self.storage.num_transitions_per_env
            // self.num_mini_batches
        )
        loss_dict = super().update()
        self._enforce_minimum_std()
        if self.amp_replay.size == 0:
            raise RuntimeError("No non-terminal AMP transitions were collected.")
        amp_data = self._ensure_amp_data()
        totals = {
            "amp": 0.0,
            "amp_grad_pen": 0.0,
            "amp_policy_pred": 0.0,
            "amp_expert_pred": 0.0,
        }
        attn_totals: dict[str, float] = {}
        attn_counts: dict[str, float] = {}
        updates = self.num_learning_epochs * self.num_mini_batches
        for _ in range(updates):
            policy_state_raw, policy_next_raw, policy_motion_label = self.amp_replay.sample(
                batch_size
            )
            expert_state_raw, expert_next_raw, expert_motion_label = amp_data.sample(
                policy_motion_label
            )
            policy_state = torch.cat(
                (
                    self.amp_normalizer.normalize(policy_state_raw[:, :-AMP_LABEL_DIM]),
                    policy_state_raw[:, -AMP_LABEL_DIM:],
                ),
                dim=1,
            )
            policy_next = torch.cat(
                (
                    self.amp_normalizer.normalize(policy_next_raw[:, :-AMP_LABEL_DIM]),
                    policy_next_raw[:, -AMP_LABEL_DIM:],
                ),
                dim=1,
            )
            expert_state = torch.cat(
                (self.amp_normalizer.normalize(expert_state_raw), expert_motion_label),
                dim=1,
            )
            expert_next = torch.cat(
                (self.amp_normalizer.normalize(expert_next_raw), expert_motion_label),
                dim=1,
            )

            policy_prediction = self.discriminator(policy_state, policy_next)
            expert_prediction = self.discriminator(expert_state, expert_next)
            self._accumulate_attention_logs(
                expert_motion_label, attn_totals, attn_counts
            )
            policy_loss = torch.mean(torch.square(policy_prediction + 1.0))
            expert_loss = torch.mean(torch.square(expert_prediction - 1.0))
            amp_loss = 0.5 * (policy_loss + expert_loss)
            grad_penalty = self.discriminator.gradient_penalty(
                expert_state, expert_next, self.amp_gradient_penalty
            )
            loss = amp_loss + grad_penalty
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                grads = [
                    parameter.grad.reshape(-1)
                    for parameter in self.discriminator.parameters()
                    if parameter.grad is not None
                ]
                joined = torch.cat(grads)
                torch.distributed.all_reduce(joined)
                joined /= self.gpu_world_size
                offset = 0
                for parameter in self.discriminator.parameters():
                    if parameter.grad is not None:
                        count = parameter.numel()
                        parameter.grad.copy_(
                            joined[offset : offset + count].view_as(parameter)
                        )
                        offset += count
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.max_grad_norm
            )
            self.discriminator_optimizer.step()
            self.amp_normalizer.update(policy_state_raw[:, :-AMP_LABEL_DIM])
            self.amp_normalizer.update(expert_state_raw)
            totals["amp"] += amp_loss.item()
            totals["amp_grad_pen"] += grad_penalty.item()
            totals["amp_policy_pred"] += policy_prediction.mean().item()
            totals["amp_expert_pred"] += expert_prediction.mean().item()

        loss_dict.update({name: value / updates for name, value in totals.items()})
        for label_name, fraction in zip(
            AMP_LABEL_NAMES,
            amp_data.last_label_fractions.tolist(),
            strict=True,
        ):
            loss_dict[f"amp_label_fraction/{label_name}"] = float(fraction)
        loss_dict.update(
            {
                f"amp_motion_sample/{name}": float(fraction)
                for name, fraction in zip(
                    amp_data.motion_names,
                    amp_data.last_motion_fractions.tolist(),
                    strict=True,
                )
            }
        )
        for key, value in attn_totals.items():
            count = max(attn_counts.get(key, 0.0), 1.0)
            loss_dict[key] = value / count
        return loss_dict

    def _accumulate_attention_logs(
        self,
        labels: torch.Tensor,
        totals: dict[str, float],
        counts: dict[str, float],
    ) -> None:
        attention = getattr(self.discriminator, "last_attention", None)
        part_names = getattr(self.discriminator, "part_names", None)
        if attention is None or part_names is None:
            return
        label_ids = labels.argmax(dim=-1)
        for part_index, part_name in enumerate(part_names):
            mean_key = f"amp_attn/{part_name}"
            totals[mean_key] = totals.get(mean_key, 0.0) + float(
                attention[:, part_index].mean().item()
            )
            counts[mean_key] = counts.get(mean_key, 0.0) + 1.0
            for label_id, label_name in enumerate(AMP_LABEL_NAMES):
                mask = label_ids == label_id
                if not bool(mask.any()):
                    continue
                key = f"amp_attn/{label_name}/{part_name}"
                totals[key] = totals.get(key, 0.0) + float(
                    attention[mask, part_index].mean().item()
                )
                counts[key] = counts.get(key, 0.0) + 1.0

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        saved = super().save()
        saved.update(
            {
                "discriminator_state_dict": self.discriminator.state_dict(),
                "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
                "amp_normalizer_state_dict": self.amp_normalizer.state_dict(),
            }
        )
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_amp = load_cfg is None or load_cfg.get("amp", False)
        if load_amp:
            self.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"], strict=strict
            )
            self.discriminator_optimizer.load_state_dict(
                loaded_dict["discriminator_optimizer_state_dict"]
            )
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
        return load_iteration

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        state = [self.discriminator.state_dict(), self.amp_normalizer.state_dict()]
        torch.distributed.broadcast_object_list(state, src=0)
        self.discriminator.load_state_dict(state[0])
        self.amp_normalizer.load_state_dict(state[1])
