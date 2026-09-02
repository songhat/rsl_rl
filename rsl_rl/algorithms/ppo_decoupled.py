# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Decoupled (dual-agent) PPO for full-body humanoid velocity tracking.

Splits the policy into two independent PPO agents, following the FALCON
decoupled design:

- ``lower_body`` agent: owns leg joints (gait / locomotion). Its actor reads the
  ``actor_lower`` obs group and emits the lower-body action block; its critic reads
  ``critic_lower``.
- ``upper_body`` agent: owns torso/arm joints (stability & angular-momentum
  balance). Its actor reads ``actor_upper`` and emits the upper-body action block;
  its critic reads ``critic_upper``.

Each agent keeps its own optimizer, RolloutStorage and adaptive learning-rate
schedule, so gait and upper-body stabilization train at their own pace.

Environment interface:
- observations: TensorDict with groups ``actor_lower`` / ``actor_upper`` /
  ``critic_lower`` / ``critic_upper``.
- actions: concatenated ``[lower | upper]`` blocks (env action dimension is the
  sum of both blocks).
- rewards: ``(num_envs, 2)`` tensor — column 0 feeds the lower agent, column 1 the
  upper agent.
"""

from __future__ import annotations

import inspect

import torch
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class PPODecoupled:
    """Dual-agent PPO with independent lower-body and upper-body actors/critics."""

    ppo_lower: PPO
    """Lower-body agent (legs / gait)."""

    ppo_upper: PPO
    """Upper-body agent (torso / arms / stability)."""

    def __init__(
        self,
        ppo_lower: PPO,
        ppo_upper: PPO,
        num_actions_lower: int,
        num_actions_upper: int,
    ) -> None:
        self.ppo_lower = ppo_lower
        self.ppo_upper = ppo_upper
        self.num_actions_lower = num_actions_lower
        self.num_actions_upper = num_actions_upper
        self.learning_rate = self.ppo_lower.learning_rate

        # RND / symmetry extensions are not supported for the decoupled setup.
        self.rnd = None
        self.symmetry = None

    # ------------------------------------------------------------------ #
    # Rollout / update interface (called by the on-policy runner).
    # ------------------------------------------------------------------ #

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample both agents' actions and concatenate ``[lower | upper]``."""
        lower_actions = self.ppo_lower.act(obs)
        upper_actions = self.ppo_upper.act(obs)
        return torch.cat([lower_actions, upper_actions], dim=-1)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Split the two-column reward and feed each agent its own stream."""
        # rewards: (num_envs, 2) -> column 0 lower, column 1 upper.
        # PPO.process_env_step expects a flat (num_envs,) reward tensor.
        self.ppo_lower.process_env_step(obs, rewards[:, 0], dones, extras)
        self.ppo_upper.process_env_step(obs, rewards[:, 1], dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute returns/advantages for both agents."""
        self.ppo_lower.compute_returns(obs)
        self.ppo_upper.compute_returns(obs)

    def update(self) -> dict[str, float]:
        """Update both agents and merge the loss dictionaries."""
        loss_lower = self.ppo_lower.update()
        loss_upper = self.ppo_upper.update()
        merged: dict[str, float] = {}
        for key, value in loss_lower.items():
            merged[f"lower/{key}"] = value
        for key, value in loss_upper.items():
            merged[f"upper/{key}"] = value
        return merged

    def train_mode(self) -> None:
        """Set train mode for both agents."""
        self.ppo_lower.train_mode()
        self.ppo_upper.train_mode()

    def eval_mode(self) -> None:
        """Set evaluation mode for both agents."""
        self.ppo_lower.eval_mode()
        self.ppo_upper.eval_mode()

    # ------------------------------------------------------------------ #
    # Persistence.
    # ------------------------------------------------------------------ #

    def save(self) -> dict:
        """Save both agents' state dicts and optimizers."""
        saved_lower = self.ppo_lower.save()
        saved_upper = self.ppo_upper.save()
        return {
            "lower_agent": saved_lower,
            "upper_agent": saved_upper,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load both agents. Returns whether the iteration should be restored."""
        loaded_lower = loaded_dict.get("lower_agent")
        loaded_upper = loaded_dict.get("upper_agent")
        if loaded_lower is None or loaded_upper is None:
            # Fall back to a single-agent checkpoint (no decoupled structure).
            return self.ppo_lower.load(loaded_dict, load_cfg, strict)
        lower_load = self.ppo_lower.load(loaded_lower, load_cfg, strict)
        upper_load = self.ppo_upper.load(loaded_upper, load_cfg, strict)
        return lower_load or upper_load

    def get_policy(self) -> MLPModel:
        """Return the lower-body policy (export uses lower agent by default)."""
        return self.ppo_lower.get_policy()

    def broadcast_parameters(self) -> None:
        """Broadcast both agents' parameters across GPUs."""
        self.ppo_lower.broadcast_parameters()
        self.ppo_upper.broadcast_parameters()

    # ------------------------------------------------------------------ #
    # Construction.
    # ------------------------------------------------------------------ #

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "PPODecoupled":
        """Construct the decoupled PPO algorithm from a dual-agent config.

        Expected ``cfg`` layout (extending the single-agent rsl_rl config)::

            algorithm: {class_name: PPODecoupled, ...}   # shared defaults
            algorithm_lower: {...} | None                 # lower-agent overrides
            algorithm_upper: {...} | None                 # upper-agent overrides
            actor_lower: {class_name: MLPModel, ...}
            actor_upper: {class_name: MLPModel, ...}
            critic_lower: {class_name: MLPModel, ...}
            critic_upper: {class_name: MLPModel, ...}
            obs_groups: {
              actor_lower: [...], actor_upper: [...],
              critic_lower: [...], critic_upper: [...],
            }
            num_actions_lower: 12
            num_actions_upper: 17
        """
        actor_lower_class: type[MLPModel] = resolve_callable(cfg["actor_lower"].pop("class_name"))  # type: ignore
        actor_upper_class: type[MLPModel] = resolve_callable(cfg["actor_upper"].pop("class_name"))  # type: ignore
        critic_lower_class: type[MLPModel] = resolve_callable(cfg["critic_lower"].pop("class_name"))  # type: ignore
        critic_upper_class: type[MLPModel] = resolve_callable(cfg["critic_upper"].pop("class_name"))  # type: ignore

        default_sets = ["actor_lower", "actor_upper", "critic_lower", "critic_upper"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Inject rnd_cfg / symmetry_cfg keys so the on-policy runner can read them
        # unconditionally (mirrors PPO.construct_algorithm).
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        num_actions_lower = int(cfg.pop("num_actions_lower"))
        num_actions_upper = int(cfg.pop("num_actions_upper"))
        num_steps = cfg["num_steps_per_env"]

        # Build the four models.
        def _model_kwargs(model_cls, block: dict) -> dict:
            return {
                k: v
                for k, v in block.items()
                if k in inspect.signature(model_cls).parameters and v is not None
            }

        actor_lower: MLPModel = actor_lower_class(
            obs, cfg["obs_groups"], "actor_lower", num_actions_lower, **_model_kwargs(actor_lower_class, cfg["actor_lower"])
        ).to(device)
        actor_upper: MLPModel = actor_upper_class(
            obs, cfg["obs_groups"], "actor_upper", num_actions_upper, **_model_kwargs(actor_upper_class, cfg["actor_upper"])
        ).to(device)
        critic_lower: MLPModel = critic_lower_class(
            obs, cfg["obs_groups"], "critic_lower", 1, **_model_kwargs(critic_lower_class, cfg["critic_lower"])
        ).to(device)
        critic_upper: MLPModel = critic_upper_class(
            obs, cfg["obs_groups"], "critic_upper", 1, **_model_kwargs(critic_upper_class, cfg["critic_upper"])
        ).to(device)
        print(f"Lower Actor Model: {actor_lower}")
        print(f"Upper Actor Model: {actor_upper}")
        print(f"Lower Critic Model: {critic_lower}")
        print(f"Upper Critic Model: {critic_upper}")

        # Build two independent rollouts (each stores its own action block / reward column).
        storage_lower = RolloutStorage("rl", env.num_envs, num_steps, obs, [num_actions_lower], device)
        storage_upper = RolloutStorage("rl", env.num_envs, num_steps, obs, [num_actions_upper], device)

        # Shared PPO kwargs from ``algorithm``, then merge per-agent overrides.
        ppo_keys = [
            "num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam",
            "value_loss_coef", "entropy_coef", "learning_rate", "max_grad_norm",
            "optimizer", "use_clipped_value_loss", "schedule", "desired_kl",
            "normalize_advantage_per_mini_batch",
        ]
        alg_cfg = dict(cfg["algorithm"])
        alg_cfg.pop("class_name", None)
        common_kwargs = {k: alg_cfg[k] for k in ppo_keys if k in alg_cfg}

        def _agent_kwargs(override: dict | None, lr_alias: str) -> dict:
            kwargs = dict(common_kwargs)
            if override:
                o = dict(override)
                o.pop("class_name", None)
                for k in ppo_keys:
                    if k in o and o[k] is not None:
                        kwargs[k] = o[k]
            # Legacy aliases on the shared algorithm block.
            if lr_alias in alg_cfg:
                kwargs["learning_rate"] = alg_cfg[lr_alias]
            return kwargs

        lower_kwargs = _agent_kwargs(cfg.get("algorithm_lower"), "learning_rate_lower")
        upper_kwargs = _agent_kwargs(cfg.get("algorithm_upper"), "learning_rate_upper")

        ppo_lower = PPO(
            actor_lower, critic_lower, storage_lower, device=device, **lower_kwargs
        )
        ppo_upper = PPO(
            actor_upper, critic_upper, storage_upper, device=device, **upper_kwargs
        )

        return PPODecoupled(ppo_lower, ppo_upper, num_actions_lower, num_actions_upper)
