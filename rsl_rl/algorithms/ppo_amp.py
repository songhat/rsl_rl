from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.algorithms.amp_replay_utils import compute_amp_disc_loss, compute_finite_difference_penalty
from rsl_rl.models.discriminator_model import AMPDiscriminator, LossType, apply_spectral_norm_, resolve_amp_config
from rsl_rl.storage import CircularBuffer, DiscReplayBuffer, RolloutStorage
from rsl_rl.storage.disc_replay_buffer import agent_disc_obs_with_replay, store_disc_replay_from_rollout
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class PPOAMP(PPO):
    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer,
        disc_demo_obs_buffer: CircularBuffer,
        obs_groups: dict,
        disc_replay_buffer: DiscReplayBuffer | None = None,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        muon_learning_rate: float = 0.02,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.0,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        amp_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            actor,
            critic,
            storage,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            optimizer,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            normalize_advantage_per_mini_batch,
            device,
            muon_learning_rate,
            muon_momentum,
            muon_weight_decay,
            rnd_cfg,
            symmetry_cfg,
            multi_gpu_cfg,
        )

        self.obs_groups = obs_groups
        self.amp_cfg = amp_cfg
        if self.amp_cfg is None:
            raise ValueError("AMP configuration must be provided for PPOAMP algorithm.")

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
            device=device,
            **self.amp_cfg.get("amp_discriminator", {}),
        ).to(self.device)
        self.lipschitz = str(self.amp_cfg.get("lipschitz", "grad_penalty"))
        if self.lipschitz == "spectral_norm":
            apply_spectral_norm_(self.amp_discriminator)
            print("[INFO] Discriminator Lipschitz: spectral norm (gradient penalty off).")
        elif self.lipschitz not in ("grad_penalty", "finite_difference"):
            raise ValueError(
                f"Unknown discriminator lipschitz mode '{self.lipschitz}'. "
                "Use 'grad_penalty', 'spectral_norm', or 'finite_difference'."
            )
        self.fd_probes = max(1, int(self.amp_cfg.get("fd_probes", 1)))
        self.fd_epsilon = float(self.amp_cfg.get("fd_epsilon", 1e-3))

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
        disc_opt_kwargs: dict = {"lr": self.amp_cfg["disc_learning_rate"]}
        disc_betas = self.amp_cfg.get("disc_optimizer_betas")
        if disc_betas is not None and disc_optimizer_name in ("adam", "adamw"):
            disc_opt_kwargs["betas"] = tuple(disc_betas)
        self.disc_optimizer = resolve_optimizer(disc_optimizer_name)(params, **disc_opt_kwargs)
        # <=0 disables clipping (WGAN-GP uses GP instead of weight/grad clipping).
        self.disc_max_grad_norm = float(self.amp_cfg.get("disc_max_grad_norm", 0.5))
        self.freeze_discriminator = bool(self.amp_cfg.get("freeze_discriminator", False))
        if self.freeze_discriminator:
            self.amp_discriminator.eval()
            self.amp_discriminator.requires_grad_(False)

        self.disc_obs_buffer = disc_obs_buffer
        self.disc_demo_obs_buffer = disc_demo_obs_buffer
        self.disc_replay_buffer = disc_replay_buffer
        self.disc_replay_samples = int(self.amp_cfg.get("disc_replay_samples", 1000))
        # Independent discriminator update schedule.
        self.disc_num_learning_epochs = int(
            self.amp_cfg.get("disc_num_learning_epochs", self.num_learning_epochs)
        )
        self.disc_num_mini_batches = int(
            self.amp_cfg.get("disc_num_mini_batches", self.num_mini_batches)
        )
        self.disc_update_interval = max(1, int(self.amp_cfg.get("disc_update_interval", 1)))
        # Lazy GP: D still steps every mini-batch; GP (2nd-order) only every N steps.
        self.disc_grad_penalty_interval = max(1, int(self.amp_cfg.get("disc_grad_penalty_interval", 1)))
        self.disc_grad_penalty_batch_frac = float(self.amp_cfg.get("disc_grad_penalty_batch_frac", 1.0))
        self._amp_update_count = 0
        self._disc_step_count = 0
        # Last logged disc metrics (carried across skipped update intervals).
        self._last_disc_metrics = {
            "amp/disc_loss": 0.0,
            "amp/disc_grad_penalty": 0.0,
            "amp/disc_score": 0.0,
            "amp/disc_demo_score": 0.0,
            "amp/disc_replay_score": 0.0,
        }

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

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic.

        The discriminator is left eager: WGAN-GP calls ``autograd.grad(..., create_graph=True)``,
        and ``torch.compile`` does not support that double backward.
        """
        super().compile(mode)
        if mode is not None:
            print(
                f"[INFO] torch.compile mode={mode} on actor and critic; "
                "discriminator stays eager because WGAN-GP needs double backward."
            )

    def update(self) -> dict[str, float]:
        # Keep sums on device and sync once. Per-step ``.item()`` was leaving the GPU idle.
        sum_value_loss = torch.zeros((), device=self.device)
        sum_surrogate_loss = torch.zeros((), device=self.device)
        sum_entropy = torch.zeros((), device=self.device)
        sum_rnd_loss = torch.zeros((), device=self.device) if self.rnd else None
        sum_symmetry_loss = torch.zeros((), device=self.device) if self.symmetry else None
        sum_disc_loss = torch.zeros((), device=self.device)
        sum_disc_grad_penalty = torch.zeros((), device=self.device)
        sum_disc_score = torch.zeros((), device=self.device)
        sum_disc_demo_score = torch.zeros((), device=self.device)
        sum_disc_replay_score = torch.zeros((), device=self.device)

        fetch_length = self.storage.num_transitions_per_env  # type: ignore
        replay_pushed = store_disc_replay_from_rollout(
            self.disc_obs_buffer,
            self.disc_replay_buffer,
            fetch_length=fetch_length,
            replay_samples=self.disc_replay_samples,
        )

        # ---- Actor-critic PPO updates ----
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

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
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
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

            sum_value_loss = sum_value_loss + value_loss.detach()
            sum_surrogate_loss = sum_surrogate_loss + surrogate_loss.detach()
            sum_entropy = sum_entropy + entropy.detach().mean()
            if sum_rnd_loss is not None:
                sum_rnd_loss = sum_rnd_loss + rnd_loss.detach()
            if sum_symmetry_loss is not None:
                sum_symmetry_loss = sum_symmetry_loss + symmetry_loss.detach()

        num_ac_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss = (sum_value_loss / num_ac_updates).item()
        mean_surrogate_loss = (sum_surrogate_loss / num_ac_updates).item()
        mean_entropy = (sum_entropy / num_ac_updates).item()
        mean_rnd_loss = (sum_rnd_loss / num_ac_updates).item() if sum_rnd_loss is not None else None
        mean_symmetry_loss = (
            (sum_symmetry_loss / num_ac_updates).item() if sum_symmetry_loss is not None else None
        )

        # ---- Discriminator updates (optional interval / epoch count) ----
        update_disc = (not self.freeze_discriminator) and (
            self._amp_update_count % self.disc_update_interval == 0
        )
        num_disc_updates = 0
        if update_disc:
            disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
                fetch_length=fetch_length,
                num_mini_batches=self.disc_num_mini_batches,
                num_epochs=self.disc_num_learning_epochs,
            )
            disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
                fetch_length=fetch_length,
                num_mini_batches=self.disc_num_mini_batches,
                num_epochs=self.disc_num_learning_epochs,
            )
            for disc_obs_batch, disc_demo_obs_batch in zip(disc_obs_generator, disc_demo_obs_generator):
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

                apply_gp = (
                    self.lipschitz == "grad_penalty"
                    and self._disc_step_count % self.disc_grad_penalty_interval == 0
                )
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
                if self.lipschitz == "finite_difference":
                    disc_grad_penalty = compute_finite_difference_penalty(
                        self.amp_discriminator,
                        self.loss_type,
                        disc_demo_obs_batch_normed.reshape(current_batch_size, -1),
                        agent_disc_flat[:current_batch_size],
                        float(self.amp_cfg["grad_penalty_scale"]),
                        n_probes=self.fd_probes,
                        eps=self.fd_epsilon,
                        gp_batch_frac=self.disc_grad_penalty_batch_frac,
                    )
                    disc_loss = disc_loss + disc_grad_penalty

                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_disc_parameters()
                if self.disc_max_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.disc_max_grad_norm)
                self.disc_optimizer.step()
                self.amp_discriminator.update_normalization(disc_obs_batch)
                self._disc_step_count += 1

                sum_disc_loss = sum_disc_loss + disc_loss.detach()
                sum_disc_grad_penalty = sum_disc_grad_penalty + disc_grad_penalty.detach()
                sum_disc_score = sum_disc_score + disc_score.detach().mean()
                sum_disc_demo_score = sum_disc_demo_score + disc_demo_score.detach().mean()
                if disc_score.shape[0] > current_batch_size:
                    sum_disc_replay_score = (
                        sum_disc_replay_score + disc_score[current_batch_size:].detach().mean()
                    )
                num_disc_updates += 1

            if num_disc_updates > 0:
                mean_disc_loss = (sum_disc_loss / num_disc_updates).item()
                mean_disc_grad_penalty = (sum_disc_grad_penalty / num_disc_updates).item()
                mean_disc_score = (sum_disc_score / num_disc_updates).item()
                mean_disc_demo_score = (sum_disc_demo_score / num_disc_updates).item()
                mean_disc_replay_score = (sum_disc_replay_score / num_disc_updates).item()
                self._last_disc_metrics = {
                    "amp/disc_loss": mean_disc_loss,
                    "amp/disc_grad_penalty": mean_disc_grad_penalty,
                    "amp/disc_score": mean_disc_score,
                    "amp/disc_demo_score": mean_disc_demo_score,
                    "amp/disc_replay_score": mean_disc_replay_score,
                }
            else:
                mean_disc_loss = 0.0
                mean_disc_grad_penalty = 0.0
                mean_disc_score = 0.0
                mean_disc_demo_score = 0.0
                mean_disc_replay_score = 0.0
        else:
            # Skip interval: reuse last disc metrics so logs are not zero-spiked.
            mean_disc_loss = self._last_disc_metrics["amp/disc_loss"]
            mean_disc_grad_penalty = self._last_disc_metrics["amp/disc_grad_penalty"]
            mean_disc_score = self._last_disc_metrics["amp/disc_score"]
            mean_disc_demo_score = self._last_disc_metrics["amp/disc_demo_score"]
            mean_disc_replay_score = self._last_disc_metrics["amp/disc_replay_score"]

        self._amp_update_count += 1

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
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.storage.clear()
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        if self.freeze_discriminator:
            self.amp_discriminator.eval()
            self.amp_discriminator.disc_obs_normalizer.eval()
        else:
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

    def reduce_parameters(self, include_disc: bool = True) -> None:
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if include_disc:
            all_params = chain(all_params, self.amp_discriminator.parameters())
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

    def reduce_disc_parameters(self) -> None:
        all_params = list(self.amp_discriminator.parameters())
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
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
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOAMP:
        """Construct the PPO-AMP algorithm."""
        alg_class: type[PPOAMP] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
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

        alg: PPOAMP = alg_class(
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
