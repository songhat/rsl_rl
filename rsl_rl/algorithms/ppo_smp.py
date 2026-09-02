"""PPO + Score-Matching Motion Prior (SMP) style reward — no adversarial discriminator."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.diff_normalizer import DiffNormalizer


def resolve_smp_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict, env: VecEnv) -> dict:
    """Fill disc_obs_dim / disc_obs_steps and step_dt from the live env."""
    if "smp_cfg" not in alg_cfg or alg_cfg["smp_cfg"] is None:
        raise ValueError("SMP configuration is missing or None.")

    if "discriminator" not in obs_groups:
        raise ValueError("SMP requires obs_groups['discriminator'] (usually ['disc']).")

    disc_obs_dim = 0
    disc_obs_steps = -1
    for obs_group in obs_groups["discriminator"]:
        if obs_group not in obs:
            raise ValueError(f"Observation group '{obs_group}' not found.")
        obs_tensor = obs[obs_group]
        assert len(obs_tensor.shape) == 3, "SMP disc obs must be (N, T, D)."
        if disc_obs_steps == -1:
            disc_obs_steps = int(obs_tensor.shape[1])
        else:
            assert disc_obs_steps == int(obs_tensor.shape[1])
        disc_obs_dim += int(obs_tensor.shape[-1])

    smp_cfg = alg_cfg["smp_cfg"]
    smp_cfg["disc_obs_dim"] = disc_obs_dim
    smp_cfg["disc_obs_steps"] = disc_obs_steps

    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env.unwrapped
    smp_cfg["step_dt"] = float(getattr(unwrapped, "step_dt", 0.02))
    return alg_cfg


class PPOSMP(PPO):
    """PPO with frozen TinyMDM ensemble SDS style reward (SMP).

    Reward (MimicKit / paper):
        r = task_w * r_task + smp_w * r_smp
        r_smp = smp_reward_scale * exp(-sds_loss_scale * mean_t ||ε̂_t - ε_t||²_norm)
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        obs_groups: dict,
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
        smp_cfg: dict | None = None,
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
        if smp_cfg is None:
            raise ValueError("smp_cfg required for PPOSMP")
        self.smp_cfg = smp_cfg

        self.task_reward_weight = float(smp_cfg.get("task_reward_weight", 0.0))
        self.smp_reward_weight = float(smp_cfg.get("smp_reward_weight", 1.0))
        self._style_enabled = self.smp_reward_weight > 0.0
        self.sds_loss_scale = float(smp_cfg.get("sds_loss_scale", 6.0))
        self.smp_reward_scale = float(smp_cfg.get("smp_reward_scale", 1.0))
        self.smp_eval_batch_size = int(smp_cfg.get("smp_eval_batch_size", 4096))
        self.diffusion_steps = list(smp_cfg.get("diffusion_steps", [22, 15, 8]))
        self.sds_normalizer_samples = float(smp_cfg.get("sds_normalizer_samples", 1e8))
        self._sample_count = 0

        self.disc_obs_dim = int(smp_cfg["disc_obs_dim"])
        self.disc_obs_steps = int(smp_cfg["disc_obs_steps"])

        self._sds_normalizer = DiffNormalizer(
            [len(self.diffusion_steps)], device=self.device, dtype=torch.float32
        )
        self._prior_model = None
        if self._style_enabled:
            self._build_prior_model()

        # MimicKit-style: store disc obs during rollout; SDS after collection.
        self._disc_obs_buf = None
        if self._style_enabled:
            self._disc_obs_buf = torch.zeros(
                storage.num_transitions_per_env,
                storage.num_envs,
                self.disc_obs_steps,
                self.disc_obs_dim,
                device=self.device,
            )
        self._timeout_bootstrap = torch.zeros(
            storage.num_transitions_per_env, storage.num_envs, 1, device=self.device
        )

        self.style_rewards = torch.zeros(0, device=self.device)
        self.rewards_lerp = torch.zeros(0, device=self.device)
        self._last_sds_mean = 0.0
        self._last_smp_r_mean = 0.0
        self._last_sds_std = 0.0

    def _build_prior_model(self) -> None:
        cfg_path = Path(self.smp_cfg["smp_prior_cfg"]).expanduser().resolve()
        model_path = Path(self.smp_cfg["smp_prior_model"]).expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing SMP prior config: {cfg_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing SMP prior model: {model_path}")

        with open(cfg_path, "r") as f:
            prior_config = yaml.safe_load(f)

        prior_config["device"] = self.device
        prior_config["input_dim"] = self.disc_obs_dim * self.disc_obs_steps
        prior_config["num_disc_obs_steps"] = self.disc_obs_steps
        # Prefer live dims over yaml env_config if present.
        if "input_channel" not in prior_config:
            prior_config["input_channel"] = self.disc_obs_dim

        prior_cls_path = self.smp_cfg.get(
            "prior_model_class",
            "src.tasks.smp.tinymdm.tinymdm_model:TinyMDMModel",
        )
        prior_cls = resolve_callable(prior_cls_path)
        self._prior_model = prior_cls(prior_config)
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        incompatible = self._prior_model.load_state_dict(state, strict=False)
        if len(incompatible[0]) > 0 or len(incompatible[1]) > 0:
            print(f"SMP prior load incompatible_keys: {incompatible}")
        self._prior_model.eval()
        self._prior_model.to(self.device)
        for p in self._prior_model.parameters():
            p.requires_grad = False
        print(f"Loaded SMP prior: cfg={cfg_path}, model={model_path}")

    def _get_disc_obs(self, obs: TensorDict) -> torch.Tensor:
        parts = []
        for obs_group in self.obs_groups["discriminator"]:
            if obs_group not in obs:
                raise ValueError(f"Observation group '{obs_group}' not found.")
            t = obs[obs_group]
            assert len(t.shape) == 3
            assert int(t.shape[1]) == self.disc_obs_steps
            parts.append(t)
        return torch.cat(parts, dim=-1)

    def _need_sds_normalizer_update(self) -> bool:
        return self._sample_count < self.sds_normalizer_samples

    @torch.no_grad()
    def _calc_smp_rewards(self, disc_obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """disc_obs: (N, T, D) → r_smp (N,)."""
        n = disc_obs.shape[0]
        norm = self._prior_model.normalize(disc_obs)
        flat = norm.reshape(n, -1)

        smp_r_list = []
        mean_sds_list = []
        for i in range(0, n, self.smp_eval_batch_size):
            chunk = flat[i : i + self.smp_eval_batch_size]
            sds_losses = self._prior_model.ESM_SDS_loss(
                norm_x_obs=chunk, t_lst=self.diffusion_steps
            )
            if self._need_sds_normalizer_update():
                self._sds_normalizer.record(sds_losses)
            sds_norm = self._sds_normalizer.normalize(sds_losses)
            mean_sds_norm = torch.mean(sds_norm, dim=-1)
            smp_r = torch.exp(-mean_sds_norm * self.sds_loss_scale) * self.smp_reward_scale
            smp_r_list.append(smp_r)
            mean_sds_list.append(torch.mean(sds_losses, dim=-1))

        smp_r = torch.cat(smp_r_list, dim=0)
        mean_sds = torch.cat(mean_sds_list, dim=0)
        info = {
            "smp/sds_loss_mean": float(mean_sds.mean().item()),
            "smp/sds_loss_std": float(mean_sds.std().item()) if mean_sds.numel() > 1 else 0.0,
            "smp/reward_mean": float(smp_r.mean().item()),
        }
        return smp_r, info

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        """Record task reward + disc obs (style enabled only). SDS deferred to ``apply_smp_rewards``."""
        step = self.storage.step
        if self._style_enabled:
            disc_obs = self._get_disc_obs(obs)
            if "terminal_obs" in extras:
                terminal = extras["terminal_obs"]
                if isinstance(terminal, dict) and "disc" in terminal:
                    done_mask = dones.to(dtype=torch.bool)
                    if torch.any(done_mask):
                        disc_obs = disc_obs.clone()
                        disc_obs[done_mask] = terminal["disc"][done_mask]

            step = self.storage.step
            assert self._disc_obs_buf is not None
            self._disc_obs_buf[step].copy_(disc_obs)
            self._sample_count += int(disc_obs.shape[0])

        # Mirror PPO timeout bootstrap so we can re-apply it after reward lerp (AMP order:
        # style lerp first, then timeout bootstrap).
        if "time_outs" in extras:
            bootstrap = self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )
            self._timeout_bootstrap[step].copy_(bootstrap.view(-1, 1))
        else:
            self._timeout_bootstrap[step].zero_()

        extras_no_timeout = {k: v for k, v in extras.items() if k != "time_outs"}
        super().process_env_step(obs, rewards, dones, extras_no_timeout)

    @torch.no_grad()
    def apply_smp_rewards(self) -> dict[str, float]:
        """Batch SDS after rollout when style is enabled; else task reward only."""
        st = self.storage
        task = st.rewards
        if not self._style_enabled:
            st.rewards = self.task_reward_weight * task + self._timeout_bootstrap
            self.style_rewards = torch.zeros(0, device=self.device)
            self.rewards_lerp = st.rewards.squeeze(-1).reshape(st.num_transitions_per_env * st.num_envs)
            self._last_sds_mean = 0.0
            self._last_sds_std = 0.0
            self._last_smp_r_mean = 0.0
            return {}

        t = st.num_transitions_per_env
        n = st.num_envs
        assert self._disc_obs_buf is not None
        disc = self._disc_obs_buf  # (T, N, steps, D)
        smp_r, sds_info = self._calc_smp_rewards(disc.reshape(t * n, self.disc_obs_steps, self.disc_obs_dim))
        smp = smp_r.view(t, n, 1)
        task = st.rewards  # pure task (timeout stripped in process_env_step)
        st.rewards = (
            self.task_reward_weight * task
            + self.smp_reward_weight * smp
            + self._timeout_bootstrap
        )
        self.style_rewards = smp_r
        self.rewards_lerp = st.rewards.squeeze(-1).reshape(t * n)
        self._last_sds_mean = sds_info["smp/sds_loss_mean"]
        self._last_sds_std = sds_info["smp/sds_loss_std"]
        self._last_smp_r_mean = sds_info["smp/reward_mean"]
        return sds_info

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self._style_enabled and self._need_sds_normalizer_update():
            self._sds_normalizer.update()
        if self._style_enabled:
            loss_dict["smp/sds_loss_mean"] = self._last_sds_mean
            loss_dict["smp/sds_loss_std"] = self._last_sds_std
            loss_dict["smp/reward_mean"] = self._last_smp_r_mean
            loss_dict["smp/sds_normalizer_count"] = float(self._sds_normalizer.get_count().item())
        return loss_dict

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOSMP:
        alg_class: type[PPOSMP] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        cfg["algorithm"] = resolve_smp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(
            device
        )
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg: PPOSMP = alg_class(
            actor,
            critic,
            storage,
            cfg["obs_groups"],
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
