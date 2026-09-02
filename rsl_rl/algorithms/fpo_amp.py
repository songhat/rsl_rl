from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms import FPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import FPOModel, MLPModel
from rsl_rl.models.discriminator_model import AMPDiscriminator, LossType, resolve_amp_config
from rsl_rl.algorithms.amp_replay_utils import compute_amp_disc_loss
from rsl_rl.storage import CircularBuffer, DiscReplayBuffer, RolloutStorage
from rsl_rl.storage.disc_replay_buffer import agent_disc_obs_with_replay, store_disc_replay_from_rollout
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class FPOAMP(FPO):
    """Flow Policy Optimization with Adversarial Motion Priors."""

    def __init__(
        self,
        actor: FPOModel,
        critic: MLPModel,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer,
        disc_demo_obs_buffer: CircularBuffer,
        obs_groups: dict,
        disc_replay_buffer: DiscReplayBuffer | None = None,
        amp_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)

        self.obs_groups = obs_groups
        self.amp_cfg = amp_cfg
        if self.amp_cfg is None:
            raise ValueError("AMP configuration must be provided for FPOAMP algorithm.")

        if self.amp_cfg["loss_type"] == "GAN":
            self.loss_type = LossType.GAN
        elif self.amp_cfg["loss_type"] == "LSGAN":
            self.loss_type = LossType.LSGAN
        elif self.amp_cfg["loss_type"] == "WGAN":
            self.loss_type = LossType.WGAN
        else:
            raise ValueError(
                f"Unknown AMP loss type: {self.amp_cfg['loss_type']}. Should be 'GAN', 'LSGAN', or 'WGAN'"
            )

        self.amp_discriminator: AMPDiscriminator = AMPDiscriminator(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=self.obs_groups,
            loss_type=self.loss_type,
            device=self.device,
            **self.amp_cfg.get("amp_discriminator", {}),
        ).to(self.device)

        params = [
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
        self.disc_optimizer = resolve_optimizer(disc_optimizer_name)(
            params, lr=self.amp_cfg["disc_learning_rate"]
        )
        self.disc_max_grad_norm = self.amp_cfg.get("disc_max_grad_norm", 0.5)

        self.disc_obs_buffer = disc_obs_buffer
        self.disc_demo_obs_buffer = disc_demo_obs_buffer
        self.disc_replay_buffer = disc_replay_buffer
        self.disc_replay_samples = int(self.amp_cfg.get("disc_replay_samples", 1000))

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
        if "terminal_obs" in extras:
            terminal_disc_obs = self.amp_discriminator.get_disc_obs(extras["terminal_obs"], flatten_history_dim=False)
            done_mask = dones.to(dtype=torch.bool)
            if torch.any(done_mask):
                disc_obs = disc_obs.clone()
                disc_obs[done_mask] = terminal_disc_obs[done_mask]

        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(
            disc_obs, dt=self.amp_cfg["step_dt"]
        )
        self.rewards_lerp = self.amp_discriminator.lerp_reward(task_reward=rewards, style_reward=self.style_rewards)

        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)
        super().process_env_step(obs, self.rewards_lerp, dones, extras)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_cfm_difference = 0.0
        mean_policy_ratio = 0.0
        mean_disc_loss = 0.0
        mean_disc_grad_penalty = 0.0
        mean_disc_score = 0.0
        mean_disc_demo_score = 0.0
        mean_disc_replay_score = 0.0

        fetch_length = self.storage.num_transitions_per_env  # type: ignore
        replay_pushed = store_disc_replay_from_rollout(
            self.disc_obs_buffer,
            self.disc_replay_buffer,
            fetch_length=fetch_length,
            replay_samples=self.disc_replay_samples,
        )

        generator = self.storage.fpo_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,  # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,  # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        for batch, disc_obs_batch, disc_demo_obs_batch in zip(generator, disc_obs_generator, disc_demo_obs_generator):
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

            disc_loss, disc_grad_penalty = compute_amp_disc_loss(
                self.amp_discriminator,
                self.loss_type,
                disc_score,
                disc_demo_score,
                disc_demo_obs_batch_normed.reshape(current_batch_size, -1),
                agent_disc_flat[:current_batch_size],
                self.amp_cfg["grad_penalty_scale"],
            )
            disc_total_loss = disc_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.zero_grad()
            disc_total_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.disc_max_grad_norm)
            self.disc_optimizer.step()
            self.amp_discriminator.update_normalization(disc_obs_batch)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cfm_difference += cfm_difference.mean().item()
            mean_policy_ratio += ratio.mean().item()
            mean_disc_loss += disc_loss.item()
            mean_disc_grad_penalty += disc_grad_penalty.item()
            mean_disc_score += disc_score[:current_batch_size].mean().item()
            mean_disc_demo_score += disc_demo_score.mean().item()
            if disc_score.shape[0] > current_batch_size:
                mean_disc_replay_score += disc_score[current_batch_size:].mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "cfm_difference": mean_cfm_difference / num_updates,
            "policy_ratio": mean_policy_ratio / num_updates,
            "amp/disc_loss": mean_disc_loss / num_updates,
            "amp/disc_grad_penalty": mean_disc_grad_penalty / num_updates,
            "amp/disc_score": mean_disc_score / num_updates,
            "amp/disc_demo_score": mean_disc_demo_score / num_updates,
            "amp/disc_replay_score": mean_disc_replay_score / num_updates,
            "amp/disc_replay_size": float(len(self.disc_replay_buffer)) if self.disc_replay_buffer else 0.0,
            "amp/disc_replay_pushed": float(replay_pushed),
        }

    def train_mode(self) -> None:
        super().train_mode()
        self.amp_discriminator.train()
        self.amp_discriminator.disc_obs_normalizer.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.amp_discriminator.eval()
        self.amp_discriminator.disc_obs_normalizer.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
        saved_dict["amp_discriminator_normalizer_state_dict"] = self.amp_discriminator.disc_obs_normalizer.state_dict()
        saved_dict["amp_discriminator_optimizer_state_dict"] = self.disc_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "amp": True,
            }
        if load_cfg.get("amp", True):
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            self.amp_discriminator.disc_obs_normalizer.load_state_dict(
                loaded_dict["amp_discriminator_normalizer_state_dict"], strict=strict
            )
            if load_cfg.get("optimizer", True):
                self.disc_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
        return load_iteration

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        model_params = [self.amp_discriminator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.amp_discriminator.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        all_params = chain(
            self.actor.parameters(),
            self.critic.parameters(),
            self.amp_discriminator.parameters(),
        )
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FPOAMP:
        """Construct the FPO-AMP algorithm."""
        alg_class: type[FPOAMP] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[FPOModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

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

        alg: FPOAMP = alg_class(
            actor,
            critic,
            storage,
            disc_obs_buffer,
            disc_demo_obs_buffer,
            cfg["obs_groups"],
            disc_replay_buffer,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
