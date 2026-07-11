# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import FPOModel, MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class FPO(PPO):
    """Flow Policy Optimization algorithm.

  Uses conditional flow matching loss ratios instead of log-probability ratios for policy updates.
    """

    actor: FPOModel
    """The flow-matching actor model."""

    def __init__(
        self,
        actor: FPOModel,
        critic: MLPModel,
        storage: RolloutStorage,
        cfm_clip: float = 3.0,
        positive_advantage: bool = False,
        **kwargs,
    ) -> None:
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("FPO currently only supports feedforward policies.")
        super().__init__(actor, critic, storage, **kwargs)
        self.cfm_clip = cfm_clip
        self.positive_advantage = positive_advantage

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions via flow integration and store CFM training information."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        actions, action_info = self.actor.sample_with_info(obs)
        self.transition.actions = actions.detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.fpo_loss_eps = action_info.loss_eps.detach()
        self.transition.fpo_loss_t = action_info.loss_t.detach()
        self.transition.fpo_initial_cfm_loss = action_info.initial_cfm_loss.detach()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if self.positive_advantage:
            st.advantages = F.softplus(st.advantages)
        elif not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs using CFM loss ratios."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_cfm_difference = 0.0
        mean_policy_ratio = 0.0

        generator = self.storage.fpo_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            values = self.critic(batch.observations)
            obs_latent = self.actor.get_latent(batch.observations)  # type: ignore

            batch_size, num_samples, action_dim = batch.fpo_loss_eps.shape  # type: ignore
            flat_obs = obs_latent.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)
            flat_actions = batch.actions.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)  # type: ignore
            flat_eps = batch.fpo_loss_eps.reshape(batch_size * num_samples, action_dim)  # type: ignore
            flat_t = batch.fpo_loss_t.reshape(batch_size * num_samples, 1)  # type: ignore
            flat_init_loss = batch.fpo_initial_cfm_loss.reshape(batch_size * num_samples)  # type: ignore

            cfm_loss = self.actor.compute_cfm_loss(flat_obs, flat_actions, flat_eps, flat_t)
            cfm_difference = (flat_init_loss - cfm_loss).view(batch_size, num_samples)
            cfm_difference = torch.clamp(cfm_difference, -self.cfm_clip, self.cfm_clip)
            ratio = torch.exp(torch.clamp(cfm_difference.mean(dim=1), -self.cfm_clip, self.cfm_clip))

            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cfm_difference += cfm_difference.mean().item()
            mean_policy_ratio += ratio.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "cfm_difference": mean_cfm_difference / num_updates,
            "policy_ratio": mean_policy_ratio / num_updates,
        }

    def get_policy(self) -> FPOModel:
        return self._raw_actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FPO:
        """Construct the FPO algorithm."""
        alg_class: type[FPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[FPOModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor_cfg = dict(cfg["actor"])
        num_fpo_samples = actor_cfg.get("num_fpo_samples", 8)
        actor: FPOModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")

        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "fpo",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            num_fpo_samples=num_fpo_samples,
        )

        alg: FPO = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
