# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + AMP + ASE skill encoder (phase-1 pre-training)."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.amp_replay_utils import compute_amp_disc_loss
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_amp import PPOAMP
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.models.ase_discriminator_model import ASEDiscriminatorEncoder
from rsl_rl.models.discriminator_model import resolve_amp_config
from rsl_rl.storage import CircularBuffer, DiscReplayBuffer, RolloutStorage
from rsl_rl.storage.disc_replay_buffer import agent_disc_obs_with_replay, store_disc_replay_from_rollout
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


def aligned_circular_buffer_batches(
    buffers: list[CircularBuffer],
    *,
    fetch_length: int,
    num_mini_batches: int,
    num_epochs: int,
):
    """Yield aligned mini-batches from multiple CircularBuffers (shared indices)."""
    if not buffers:
        return
    ref = buffers[0]
    for buf in buffers[1:]:
        if buf.batch_size != ref.batch_size:
            raise ValueError("Aligned buffers must share batch_size")
        if buf.device != ref.device:
            raise ValueError("Aligned buffers must share device")

    if torch.any(ref.current_length == 0) or ref._buffer is None:
        raise RuntimeError("Attempting to retrieve data on an empty circular buffer.")
    for buf in buffers[1:]:
        if torch.any(buf.current_length == 0) or buf._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer.")

    min_len = int(torch.min(torch.stack([b.current_length for b in buffers])).item())
    if fetch_length > min_len:
        raise ValueError(f"Fetch length {fetch_length} exceeds min buffer length {min_len}.")

    epoch_batch_size = ref.batch_size * fetch_length
    if epoch_batch_size % num_mini_batches != 0:
        raise ValueError(f"Epoch batch size {epoch_batch_size} not divisible by {num_mini_batches}.")
    mini_batch_size = epoch_batch_size // num_mini_batches
    total_combinations = int(ref.current_length[0].item()) * ref.batch_size
    linear_indices = torch.randperm(total_combinations, device=ref.device)[:epoch_batch_size]
    indices_0 = linear_indices // ref.batch_size
    indices_1 = linear_indices % ref.batch_size

    for _epoch in range(num_epochs):
        order = torch.randperm(epoch_batch_size, device=ref.device)
        for i in range(num_mini_batches):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            mi0 = indices_0[order[start:end]]
            mi1 = indices_1[order[start:end]]
            yield tuple(buf._buffer[mi0, mi1] for buf in buffers)


class PPOASE(PPOAMP):
    """ASE pre-training: PPOAMP + MI encoder + diversity regularizer.

    Reward at each step (ASE eq. 10):
        r = style_reward(D) + mi_reward_weight * MI(q, z)
    Skill latents are read from ``obs[skill_obs_group]`` (default ``"skill"``).
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer,
        disc_demo_obs_buffer: CircularBuffer,
        obs_groups: dict,
        disc_replay_buffer: DiscReplayBuffer | None = None,
        *,
        ase_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        amp_cfg = kwargs.get("amp_cfg")
        if amp_cfg is None:
            raise ValueError("amp_cfg required for PPOASE")
        self._ase_cfg = dict(ase_cfg or {})
        self._pending_latent_dim = int(self._ase_cfg.get("latent_dim", 64))
        self.skill_obs_group = str(self._ase_cfg.get("skill_obs_group", "skill"))

        super().__init__(
            actor,
            critic,
            storage,
            disc_obs_buffer,
            disc_demo_obs_buffer,
            obs_groups,
            disc_replay_buffer,
            **kwargs,
        )

        disc_kwargs = dict(self.amp_cfg.get("amp_discriminator", {}))
        enc_hidden = self._ase_cfg.get("enc_hidden_dims")
        self.amp_discriminator = ASEDiscriminatorEncoder(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=self.obs_groups,
            latent_dim=self._pending_latent_dim,
            encoder_scale=float(self._ase_cfg.get("encoder_scale", 1.0)),
            mi_hypersphere_reward_shift=bool(
                self._ase_cfg.get("mi_hypersphere_reward_shift", False)
            ),
            enc_hidden_dims=enc_hidden,
            loss_type=self.loss_type,
            device=self.device,
            **disc_kwargs,
        ).to(self.device)

        # Disc optimizer: trunk + logits only (encoder is separate, MimicKit-style).
        disc_params = [
            {
                "name": "disc_trunk",
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": self.amp_cfg["disc_trunk_weight_decay"],
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": self.amp_cfg["disc_linear_weight_decay"],
            },
        ]
        disc_optimizer_name = self.amp_cfg.get("disc_optimizer", "adam")
        disc_opt_kwargs: dict = {"lr": self.amp_cfg["disc_learning_rate"]}
        disc_betas = self.amp_cfg.get("disc_optimizer_betas")
        if disc_betas is not None and disc_optimizer_name in ("adam", "adamw"):
            disc_opt_kwargs["betas"] = tuple(disc_betas)
        self.disc_optimizer = resolve_optimizer(disc_optimizer_name)(
            disc_params, **disc_opt_kwargs
        )

        enc_optimizer_name = self._ase_cfg.get("enc_optimizer", disc_optimizer_name)
        enc_lr = float(self._ase_cfg.get("enc_learning_rate", self.amp_cfg["disc_learning_rate"]))
        enc_opt_kwargs: dict = {"lr": enc_lr}
        if enc_optimizer_name in ("adam", "adamw") and disc_betas is not None:
            enc_opt_kwargs["betas"] = tuple(disc_betas)
        self.enc_optimizer = resolve_optimizer(enc_optimizer_name)(
            [
                {
                    "name": "enc",
                    "params": self.amp_discriminator.get_enc_params(),
                    "weight_decay": float(self._ase_cfg.get("enc_weight_decay", 0.0)),
                }
            ],
            **enc_opt_kwargs,
        )

        # MimicKit: r = w_task * r_task + w_disc * r_disc + w_enc * r_enc
        self.task_reward_weight = float(self._ase_cfg.get("task_reward_weight", 0.0))
        self.disc_reward_weight = float(self._ase_cfg.get("disc_reward_weight", 0.5))
        self.mi_reward_weight = float(self._ase_cfg.get("mi_reward_weight", 0.5))
        self.diversity_weight = float(self._ase_cfg.get("diversity_weight", 0.01))
        self.diversity_tar = float(self._ase_cfg.get("diversity_tar", 1.0))
        self.disc_logit_reg = float(self.amp_cfg.get("disc_logit_reg", 0.0))
        self.latent_dim = self._pending_latent_dim
        self.skill_latent_buffer = CircularBuffer(
            max_len=self.amp_cfg["disc_obs_buffer_size"],
            batch_size=storage.num_envs,  # type: ignore[attr-defined]
            device=self.device,
        )
        self.mi_rewards = torch.zeros(0)
        self._last_mi_metrics = {
            "ase/mi_reward": 0.0,
            "ase/enc_loss": 0.0,
            "ase/diversity_loss": 0.0,
        }

    def _get_skill_latents(self, obs: TensorDict) -> torch.Tensor:
        if self.skill_obs_group not in obs.keys():
            raise KeyError(
                f"ASE skill observation group '{self.skill_obs_group}' not found in observations. "
                f"Available keys: {list(obs.keys())}"
            )
        latents = obs[self.skill_obs_group]
        if latents.ndim > 2:
            latents = latents.reshape(latents.shape[0], -1)
        return latents[:, : self.latent_dim].to(self.device)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
        if "terminal_obs" in extras:
            terminal_disc_obs = self.amp_discriminator.get_disc_obs(
                extras["terminal_obs"], flatten_history_dim=False
            )
            done_mask = dones.to(dtype=torch.bool)
            if torch.any(done_mask):
                disc_obs = disc_obs.clone()
                disc_obs[done_mask] = terminal_disc_obs[done_mask]

        latents = self._get_skill_latents(obs)
        style_rewards, mi_rewards, self.disc_score = self.amp_discriminator.predict_style_and_mi_reward(
            disc_obs, latents, dt=self.amp_cfg["step_dt"]
        )
        self.style_rewards = style_rewards
        self.mi_rewards = mi_rewards
        # MimicKit ASEAgent._compute_rewards
        combined = (
            self.task_reward_weight * rewards
            + self.disc_reward_weight * style_rewards
            + self.mi_reward_weight * mi_rewards
        )
        self.rewards_lerp = combined

        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)
        self.skill_latent_buffer.append(latents)
        PPO.process_env_step(self, obs, combined, dones, extras)
        self._last_mi_metrics["ase/mi_reward"] = float(mi_rewards.mean().item())

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_diversity_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        fetch_length = self.storage.num_transitions_per_env  # type: ignore
        replay_pushed = store_disc_replay_from_rollout(
            self.disc_obs_buffer,
            self.disc_replay_buffer,
            fetch_length=fetch_length,
            replay_samples=self.disc_replay_samples,
        )

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    self.set_learning_rate(self.learning_rate)

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
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

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            diversity_loss = self._diversity_loss(batch.observations[:original_batch_size])
            if diversity_loss is not None:
                loss = loss + self.diversity_weight * diversity_loss
                mean_diversity_loss += float(diversity_loss.item())

            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore
            symmetry_loss = None
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            if self.muon_optimizer is not None:
                self.muon_optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(include_disc=False)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.muon_optimizer is not None:
                self.muon_optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_ac_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_ac_updates
        mean_surrogate_loss /= num_ac_updates
        mean_entropy /= num_ac_updates
        mean_diversity_loss /= max(1, num_ac_updates)
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_ac_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_ac_updates

        mean_disc_loss = 0.0
        mean_disc_grad_penalty = 0.0
        mean_disc_score = 0.0
        mean_disc_demo_score = 0.0
        mean_disc_replay_score = 0.0
        mean_enc_loss = 0.0
        update_disc = (not self.freeze_discriminator) and (self._amp_update_count % self.disc_update_interval == 0)
        num_disc_updates = 0
        if update_disc:
            demo_generator = self.disc_demo_obs_buffer.mini_batch_generator(
                fetch_length=fetch_length,
                num_mini_batches=self.disc_num_mini_batches,
                num_epochs=self.disc_num_learning_epochs,
            )
            agent_latent_generator = aligned_circular_buffer_batches(
                [self.disc_obs_buffer, self.skill_latent_buffer],
                fetch_length=fetch_length,
                num_mini_batches=self.disc_num_mini_batches,
                num_epochs=self.disc_num_learning_epochs,
            )
            for disc_demo_obs_batch, (disc_obs_batch, latent_batch) in zip(demo_generator, agent_latent_generator):
                with torch.no_grad():
                    disc_demo_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)

                agent_disc_flat, current_batch_size = agent_disc_obs_with_replay(
                    self.amp_discriminator,
                    disc_obs_batch,
                    self.disc_replay_buffer,
                )
                disc_score = self.amp_discriminator(agent_disc_flat)
                disc_demo_score = self.amp_discriminator(
                    disc_demo_obs_batch_normed.reshape(current_batch_size, -1)
                )

                apply_gp = self._disc_step_count % self.disc_grad_penalty_interval == 0
                disc_loss, disc_grad_penalty = compute_amp_disc_loss(
                    self.amp_discriminator,
                    self.loss_type,
                    disc_score,
                    disc_demo_score,
                    disc_demo_obs_batch_normed.reshape(current_batch_size, -1),
                    agent_disc_flat[:current_batch_size],
                    float(self.amp_cfg["grad_penalty_scale"]),
                    apply_grad_penalty=apply_gp,
                    gp_batch_frac=self.disc_grad_penalty_batch_frac,
                )
                if self.disc_logit_reg > 0.0:
                    logit_w = self.amp_discriminator.get_disc_logit_weights()
                    disc_loss = disc_loss + self.disc_logit_reg * torch.sum(torch.square(logit_w))

                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_disc_parameters()
                if self.disc_max_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.amp_discriminator.get_disc_params(), self.disc_max_grad_norm
                    )
                self.disc_optimizer.step()

                # Separate encoder update (MimicKit: independent enc_optimizer).
                agent_only = agent_disc_flat[:current_batch_size].detach()
                z = latent_batch.reshape(latent_batch.shape[0], -1)[:current_batch_size]
                enc_loss = self.amp_discriminator.encoder_neg_log_likelihood(agent_only, z)
                self.enc_optimizer.zero_grad()
                enc_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_enc_parameters()
                if self.disc_max_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.amp_discriminator.get_enc_params(), self.disc_max_grad_norm
                    )
                self.enc_optimizer.step()

                self.amp_discriminator.update_normalization(disc_obs_batch)
                self._disc_step_count += 1

                mean_disc_loss += disc_loss.item()
                mean_disc_grad_penalty += disc_grad_penalty.item()
                mean_disc_score += disc_score.mean().item()
                mean_disc_demo_score += disc_demo_score.mean().item()
                mean_enc_loss += enc_loss.item()
                if disc_score.shape[0] > current_batch_size:
                    mean_disc_replay_score += disc_score[current_batch_size:].mean().item()
                num_disc_updates += 1

            if num_disc_updates > 0:
                mean_disc_loss /= num_disc_updates
                mean_disc_grad_penalty /= num_disc_updates
                mean_disc_score /= num_disc_updates
                mean_disc_demo_score /= num_disc_updates
                mean_disc_replay_score /= num_disc_updates
                mean_enc_loss /= num_disc_updates
                self._last_disc_metrics = {
                    "amp/disc_loss": mean_disc_loss,
                    "amp/disc_grad_penalty": mean_disc_grad_penalty,
                    "amp/disc_score": mean_disc_score,
                    "amp/disc_demo_score": mean_disc_demo_score,
                    "amp/disc_replay_score": mean_disc_replay_score,
                }
                self._last_mi_metrics["ase/enc_loss"] = mean_enc_loss
        else:
            mean_disc_loss = self._last_disc_metrics["amp/disc_loss"]
            mean_disc_grad_penalty = self._last_disc_metrics["amp/disc_grad_penalty"]
            mean_disc_score = self._last_disc_metrics["amp/disc_score"]
            mean_disc_demo_score = self._last_disc_metrics["amp/disc_demo_score"]
            mean_disc_replay_score = self._last_disc_metrics["amp/disc_replay_score"]
            mean_enc_loss = self._last_mi_metrics["ase/enc_loss"]

        self._amp_update_count += 1
        self._last_mi_metrics["ase/diversity_loss"] = mean_diversity_loss

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp/disc_loss": mean_disc_loss,
            "amp/disc_grad_penalty": mean_disc_grad_penalty,
            "amp/disc_score": mean_disc_score,
            "amp/disc_demo_score": mean_disc_demo_score,
            "amp/disc_replay_score": mean_disc_replay_score,
            "amp/disc_replay_size": float(len(self.disc_replay_buffer)) if self.disc_replay_buffer else 0.0,
            "amp/disc_replay_pushed": float(replay_pushed),
            "amp/disc_updated": float(update_disc),
            "amp/disc_num_updates": float(num_disc_updates),
            "ase/mi_reward": self._last_mi_metrics["ase/mi_reward"],
            "ase/enc_loss": mean_enc_loss,
            "ase/diversity_loss": mean_diversity_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.storage.clear()
        return loss_dict

    def _diversity_loss(self, observations: TensorDict) -> torch.Tensor | None:
        """MimicKit diversity: ``(diversity_tar - ||Δμ||² / D_z)^2``.

        ``D_z(z1, z2) = 0.5 (1 - z1·z2)``. Uses action means (not KL), matching
        the author's ASEAgent._compute_diversity_loss.
        """
        if self.diversity_weight <= 0.0:
            return None
        if self.skill_obs_group not in observations.keys():
            return None
        n = observations[self.skill_obs_group].shape[0]
        z_old = self._get_skill_latents(observations)
        z_new = torch.nn.functional.normalize(
            torch.randn(n, self.latent_dim, device=self.device), dim=-1
        )
        obs_new = observations.clone()
        obs_new[self.skill_obs_group] = z_new

        # Current batch forward already ran with z_old in observations.
        self.actor(observations, stochastic_output=True)
        mu_old = self.actor.output_distribution_params[0]
        self.actor(obs_new, stochastic_output=True)
        mu_new = self.actor.output_distribution_params[0]

        a_diff = torch.mean(torch.square(mu_new - mu_old), dim=-1)
        z_diff = 0.5 * (1.0 - torch.sum(z_new * z_old, dim=-1))
        diversity_ratio = a_diff / (z_diff + 1e-5)
        return torch.square(self.diversity_tar - diversity_ratio).mean()

    def _reduce_param_grads(self, params: list) -> None:
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(
                    all_grads[offset : offset + numel].view_as(param.grad.data)
                )
                offset += numel

    def reduce_disc_parameters(self) -> None:
        self._reduce_param_grads(self.amp_discriminator.get_disc_params())

    def reduce_enc_parameters(self) -> None:
        self._reduce_param_grads(self.amp_discriminator.get_enc_params())

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["ase_encoder_optimizer_state_dict"] = self.enc_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None:
            load_cfg = {"optimizer": True, "amp": True}
        if load_cfg.get("amp", True) and load_cfg.get("optimizer", True):
            enc_opt_state = loaded_dict.get("ase_encoder_optimizer_state_dict")
            if enc_opt_state is not None:
                self.enc_optimizer.load_state_dict(enc_opt_state)
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOASE:
        """Construct the PPO-ASE algorithm."""
        alg_class: type[PPOASE] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        disc_obs_buffer = CircularBuffer(
            max_len=cfg["algorithm"]["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=env.num_envs,
            device=device,
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=cfg["algorithm"]["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=env.num_envs,
            device=device,
        )

        disc_replay_buffer = None
        replay_size = cfg["algorithm"]["amp_cfg"].get("disc_replay_buffer_size")
        if replay_size:
            disc_replay_buffer = DiscReplayBuffer(max_size=int(replay_size), device=device)

        ase_cfg = cfg["algorithm"].pop("ase_cfg", {})
        alg: PPOASE = alg_class(
            actor,
            critic,
            storage,
            disc_obs_buffer,
            disc_demo_obs_buffer,
            cfg["obs_groups"],
            disc_replay_buffer,
            ase_cfg=ase_cfg,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
