from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import PPOASE
from rsl_rl.env import VecEnv
from rsl_rl.runners.amp_runner import AMPRunner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan
from rsl_rl.utils.ase_logger import LoggerASE


class ASERunner(AMPRunner):
    """On-policy runner for ASE phase-1 pre-training (uses ``LoggerASE``)."""

    alg: PPOASE

    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
    ) -> None:
        # Initialize base runner without AMP logger, then attach ASE logger.
        OnPolicyRunner.__init__(self, env, train_cfg, log_dir, device)

        unwrapped = getattr(self.env, "unwrapped", self.env)
        if hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env.unwrapped

        self.logger = LoggerASE(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            max_episode_length_s=float(self.env.max_episode_length) * unwrapped.step_dt,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """ASE learning loop — same as AMP, with MI rewards forwarded to ``LoggerASE``."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(
                        rewards,
                        dones,
                        extras,
                        intrinsic_rewards,
                        self.alg.style_rewards,
                        self.alg.rewards_lerp,
                        getattr(self.alg, "mi_rewards", None),
                    )

                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
