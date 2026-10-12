from __future__ import annotations

import torch
import torch.nn as nn
from enum import Enum
from torch import autograd
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class LossType(Enum):
    GAN = 0
    LSGAN = 1
    WGAN = 2


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        disc_obs_dim: int,
        disc_obs_steps: int,
        obs_groups: dict,
        loss_type: LossType = LossType.LSGAN,
        hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        activation: str = "relu",
        style_reward_scale: float = 1.0,
        task_style_lerp: float = 0.0,
        reward_score_scale: float = 1.0 / 15.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = disc_obs_dim * disc_obs_steps
        self.disc_obs_dim = disc_obs_dim
        self.disc_obs_steps = disc_obs_steps
        self.obs_groups = obs_groups
        activation = resolve_nn_activation(activation)
        assert style_reward_scale >= 0, "AMP reward scale must be non-negative."
        assert reward_score_scale > 0, "WGAN reward score scale must be positive."
        self.style_reward_scale = style_reward_scale
        self.task_style_lerp = task_style_lerp
        self.reward_score_scale = float(reward_score_scale)
        self.device = device
        self.loss_type = loss_type

        # Discriminator observation normalizer
        self.disc_obs_normalizer = EmpiricalNormalization(shape=self.disc_obs_dim, until=int(1e8)).to(device)

        # Build the discriminator network
        disc_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            disc_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            disc_layers.append(activation)
            curr_in_dim = hidden_dim
        self.disc_trunk = nn.Sequential(*disc_layers)
        self.disc_linear = nn.Linear(hidden_dims[-1], 1)

        print(f"AMP Discriminator MLP: {self.disc_trunk}")
        print(f"AMP Discriminator Output Layer: {self.disc_linear}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the AMP Discriminator."""
        h = self.disc_trunk(x)
        return self.disc_linear(h)

    def get_disc_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        disc_obs_list = []
        for obs_group in self.obs_groups["discriminator"]:
            if obs_group not in obs:
                raise ValueError(f"Observation group '{obs_group}' not found in environment observations.")

            obs_tensor = obs[obs_group]
            assert len(obs_tensor.shape) == 3, (
                "The observation for AMP discriminator must be 3D (num_envs, num_steps, D)."
            )
            num_envs, history_length, _ = obs_tensor.shape
            assert history_length == self.disc_obs_steps, "Discriminator observation history length mismatch."

            disc_obs_list.append(obs_tensor)
        disc_obs = torch.cat(disc_obs_list, dim=-1)
        if flatten_history_dim:
            disc_obs = disc_obs.view(num_envs, -1)
        return disc_obs

    def get_disc_demo_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        disc_demo_obs_list = []
        for obs_group in self.obs_groups["discriminator_demonstration"]:
            if obs_group not in obs:
                raise ValueError(f"Observation group '{obs_group}' not found in environment observations.")

            obs_tensor = obs[obs_group]
            assert len(obs_tensor.shape) == 3, (
                "The observation for AMP discriminator must be 3D (num_envs, num_steps, D)."
            )
            num_envs, history_length, _ = obs_tensor.shape
            assert history_length == self.disc_obs_steps, "Discriminator observation history length mismatch."

            disc_demo_obs_list.append(obs_tensor)
        disc_demo_obs = torch.cat(disc_demo_obs_list, dim=-1)
        if flatten_history_dim:
            disc_demo_obs = disc_demo_obs.reshape(num_envs, -1)
        return disc_demo_obs

    def normalize_disc_obs(self, disc_obs: torch.Tensor) -> torch.Tensor:
        assert len(disc_obs.shape) == 3, (
            "Discriminator observations must be a 3D tensor (num_envs, disc_obs_steps, disc_obs_dim)."
        )
        assert self.disc_obs_dim == disc_obs.shape[2], (
            f"Discriminator observation dimension mismatch. Expected {self.disc_obs_dim}, got {disc_obs.shape[2]}."
        )
        assert self.disc_obs_steps == disc_obs.shape[1], (
            f"Discriminator observation steps mismatch. Expected {self.disc_obs_steps}, got {disc_obs.shape[1]}."
        )
        disc_obs_reshaped = disc_obs.reshape(-1, self.disc_obs_dim)
        normed_disc_obs = self.disc_obs_normalizer(disc_obs_reshaped)
        return normed_disc_obs.reshape(-1, self.disc_obs_steps, self.disc_obs_dim)

    def update_normalization(self, disc_obs: torch.Tensor) -> None:
        """Update the observation normalizer with new data."""
        assert len(disc_obs.shape) == 3, (
            "Discriminator observations must be a 3D tensor (num_envs, disc_obs_steps, disc_obs_dim)."
        )
        assert self.disc_obs_dim == disc_obs.shape[2], (
            f"Discriminator observation dimension mismatch. Expected {self.disc_obs_dim}, got {disc_obs.shape[2]}."
        )
        assert self.disc_obs_steps == disc_obs.shape[1], (
            f"Discriminator observation steps mismatch. Expected {self.disc_obs_steps}, got {disc_obs.shape[1]}."
        )
        disc_obs_reshaped = disc_obs.reshape(-1, self.disc_obs_dim)
        self.disc_obs_normalizer.update(disc_obs_reshaped)

    def compute_grad_penalty(
        self, demo_data: torch.Tensor, scale: float = 10.0, target_norm: float = 0.0
    ) -> torch.Tensor:
        """Compute the gradient penalty for the AMP Discriminator.

        Args:
            demo_data: Samples to evaluate gradients on.
            scale: Penalty coefficient λ.
            target_norm: Desired ||∇C||₂ (0 for AMP-style GP, 1 for WGAN-GP).
        """
        assert len(demo_data.shape) == 2, (
            "Demonstration data must be a 2D tensor (num_samples, disc_obs_dim * disc_obs_steps)."
        )

        demo_data_copy = demo_data.clone().detach().requires_grad_(True)

        disc = self.forward(demo_data_copy)
        ones = torch.ones_like(disc, device=demo_data_copy.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=demo_data_copy,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_penalty = scale * (grad.norm(2, dim=1) - target_norm).pow(2).mean()
        return grad_penalty

    def compute_fd_grad_penalty(
        self,
        data: torch.Tensor,
        scale: float = 10.0,
        target_norm: float = 1.0,
        n_probes: int = 1,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Approximate ``(||∇_x D|| - target)^2`` with central finite differences.

        Probes are standard normal. ``E[(∇D · z)^2] = ||∇D||^2``, so the penalty
        backward is a normal first-order backward and can be compiled.
        """
        if data.ndim != 2:
            raise ValueError("Finite-difference penalty expects (batch, features).")
        batch, features = data.shape
        probes = max(1, int(n_probes))
        base = data.detach()
        direction = torch.randn(probes, batch, features, device=base.device, dtype=base.dtype)
        step = eps * direction
        base_b = base.unsqueeze(0)
        packed = torch.cat((base_b + step, base_b - step), dim=0).reshape(2 * probes * batch, features)
        values = self.forward(packed).view(2, probes, batch)
        directional = (values[0] - values[1]) / (2.0 * eps)
        grad_norm = directional.pow(2).mean(dim=0).clamp_min(0.0).sqrt()
        return scale * (grad_norm - target_norm).pow(2).mean()

    def predict_style_reward(self, disc_obs: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        if len(disc_obs.shape) != 3:
            raise ValueError(
                "Discriminator observations must be a 3D tensor (num_envs, disc_obs_steps, disc_obs_dim)."
            )
        if self.disc_obs_dim != disc_obs.shape[2]:
            raise ValueError(
                f"Discriminator observation dimension mismatch. Expected {self.disc_obs_dim}, got {disc_obs.shape[2]}."
            )
        if self.disc_obs_steps != disc_obs.shape[1]:
            raise ValueError(
                f"Discriminator observation steps mismatch. Expected {self.disc_obs_steps}, got {disc_obs.shape[1]}."
            )

        was_training = self.training
        with torch.no_grad():
            self.eval()

            disc_obs_reshaped = disc_obs.view(-1, self.disc_obs_dim)
            normed_disc_obs = self.disc_obs_normalizer(disc_obs_reshaped)
            normed_disc_obs = normed_disc_obs.view(-1, self.disc_obs_steps * self.disc_obs_dim)

            disc_score = self.forward(normed_disc_obs)

            if self.loss_type == LossType.GAN:
                prob = 1.0 / (1.0 + torch.exp(-disc_score))
                rew = -torch.log(torch.maximum(1 - prob, torch.tensor(1e-6, device=self.device)))
            elif self.loss_type == LossType.LSGAN:
                rew = torch.clamp(1 - (1 / 4) * torch.square(disc_score - 1), min=0)
            elif self.loss_type == LossType.WGAN:
                # Unbounded critic score → scaled style reward (higher D = more demo-like).
                rew = self.reward_score_scale * disc_score
            else:
                raise ValueError(
                    f"Unknown AMP loss type: {self.loss_type}. Should be 'GAN', 'LSGAN', or 'WGAN'"
                )

            style_reward = dt * self.style_reward_scale * rew

            if was_training:
                self.train()
                # if self.loss_type == LossType.WGAN:
                #     self.disc_output_normalizer.update(disc_score)

        return style_reward.squeeze(-1), disc_score.squeeze(-1)

    def lerp_reward(self, task_reward: torch.Tensor, style_reward: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate between task reward and style reward."""
        return self.task_style_lerp * task_reward + (1.0 - self.task_style_lerp) * style_reward


def resolve_amp_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict, env: VecEnv) -> dict:
    if "amp_cfg" not in alg_cfg or alg_cfg["amp_cfg"] is None:
        raise ValueError("AMP configuration is missing or None.")

    disc_obs_dim = 0
    disc_obs_steps = -1

    if "discriminator" not in obs_groups or "discriminator_demonstration" not in obs_groups:
        raise ValueError(
            "AMP configuration requires 'discriminator' and 'discriminator_demonstration' observation groups."
        )

    for obs_group in obs_groups["discriminator"]:
        if obs_group not in obs:
            raise ValueError(f"Observation group '{obs_group}' not found in environment observations.")

        obs_tensor = obs[obs_group]
        assert len(obs_tensor.shape) == 3, "The observation for AMP discriminator must be 3D (num_envs, num_steps, D)."

        if disc_obs_steps == -1:
            disc_obs_steps = obs_tensor.shape[1]
        else:
            assert disc_obs_steps == obs_tensor.shape[1], (
                "All AMP discriminator observation groups must have the same history length."
            )
        disc_obs_dim += obs_tensor.shape[-1]

    disc_demo_obs_dim = 0
    for obs_group in obs_groups["discriminator_demonstration"]:
        if obs_group not in obs:
            raise ValueError(f"Observation group '{obs_group}' not found in environment observations.")

        obs_tensor = obs[obs_group]
        assert len(obs_tensor.shape) == 3, "The observation for AMP discriminator must be 3D (num_envs, num_steps, D)."
        assert disc_obs_steps == obs_tensor.shape[1], (
            "All AMP discriminator observation groups must have the same history length."
        )
        disc_demo_obs_dim += obs_tensor.shape[-1]

    assert disc_demo_obs_dim == disc_obs_dim, (
        "The dimension of demonstration and agent discriminator observations must match."
    )

    alg_cfg["amp_cfg"]["disc_obs_steps"] = disc_obs_steps
    alg_cfg["amp_cfg"]["disc_obs_dim"] = disc_obs_dim

    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env.unwrapped
    alg_cfg["amp_cfg"]["step_dt"] = unwrapped.step_dt

    return alg_cfg


def apply_spectral_norm_(disc: AMPDiscriminator, n_power_iterations: int = 1) -> None:
    """Constrain each linear map to spectral norm 1.

    ELU has derivative in (0, 1], so the composed network is 1-Lipschitz.
    Call this before the optimizer is built; parameter objects change.
    """
    from torch.nn.utils.parametrizations import spectral_norm

    trunk = []
    for layer in disc.disc_trunk:
        if isinstance(layer, nn.Linear):
            trunk.append(spectral_norm(layer, n_power_iterations=n_power_iterations))
        else:
            trunk.append(layer)
    disc.disc_trunk = nn.Sequential(*trunk)
    disc.disc_linear = spectral_norm(disc.disc_linear, n_power_iterations=n_power_iterations)
