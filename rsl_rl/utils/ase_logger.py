from __future__ import annotations

import datetime
import statistics
import torch
from collections import deque

from rsl_rl.utils.logger import Logger


class LoggerASE(Logger):
    """Logger for ASE phase-1 pre-training (style + MI + disc + diversity).

    Unlike ``LoggerAMP``, this does **not** dump env task reward terms. Console /
    TensorBoard extras are limited to style reward and terminations. Disc metrics
    are logged under the ``ase/`` namespace.
    """

    # Remap PPOASE loss_dict keys (still emitted as amp/* for shared disc code).
    _LOSS_KEY_MAP = {
        "amp/disc_loss": "ase/disc_loss",
        "amp/disc_grad_penalty": "ase/disc_grad_penalty",
        "amp/disc_score": "ase/disc_score",
        "amp/disc_demo_score": "ase/disc_demo_score",
        "amp/disc_replay_score": "ase/disc_replay_score",
        "amp/disc_replay_size": "ase/disc_replay_size",
        "amp/disc_replay_pushed": "ase/disc_replay_pushed",
        "amp/disc_updated": "ase/disc_updated",
        "amp/disc_num_updates": "ase/disc_num_updates",
        "amp/style_reward_scale": "ase/style_reward_scale",
        "amp/disc_loss_ema": "ase/disc_loss_ema",
        "amp/style_reward_scale_adjusted": "ase/style_reward_scale_adjusted",
    }

    # Keys that are rewards / scores, not optimization losses (console wording).
    _REWARD_LIKE_KEYS = {
        "ase/mi_reward",
        "ase/disc_score",
        "ase/disc_demo_score",
        "ase/disc_replay_score",
        "ase/disc_replay_size",
        "ase/disc_replay_pushed",
        "ase/disc_updated",
        "ase/disc_num_updates",
        "ase/style_reward_scale",
        "ase/disc_loss_ema",
        "ase/style_reward_scale_adjusted",
    }

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
        max_episode_length_s: float,
    ) -> None:
        super().__init__(
            log_dir,
            cfg,
            env_cfg,
            num_envs,
            is_distributed,
            gpu_world_size,
            gpu_global_rank,
            device,
        )
        self.total_rewbuffer = deque(maxlen=100)
        self.style_rewbuffer = deque(maxlen=100)
        self.mi_rewbuffer = deque(maxlen=100)
        self.cur_total_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_style_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_mi_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_episode_length_s = max_episode_length_s

    @classmethod
    def _map_loss_dict(cls, loss_dict: dict) -> dict:
        out: dict = {}
        for key, value in loss_dict.items():
            out[cls._LOSS_KEY_MAP.get(key, key)] = value
        return out

    @staticmethod
    def _include_ep_extra(key: str) -> bool:
        if key == "Episode_Reward/style":
            return True
        if key.startswith("Episode_Termination/"):
            return True
        return False

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
        style_rewards: torch.Tensor | None = None,
        total_rewards: torch.Tensor | None = None,
        mi_rewards: torch.Tensor | None = None,
    ) -> None:
        """Accumulate ASE pretrain step metrics (ignore env task reward dumps)."""
        if self.writer is None:
            return

        if "episode" in extras:
            self.ep_extras.append(extras["episode"])
        elif "log" in extras:
            self.ep_extras.append(extras["log"])

        if intrinsic_rewards is not None:
            self.cur_ereward_sum += rewards
            self.cur_ireward_sum += intrinsic_rewards
            self.cur_reward_sum += rewards + intrinsic_rewards
        else:
            self.cur_reward_sum += rewards
        if style_rewards is not None:
            self.cur_style_reward_sum += style_rewards
        if total_rewards is not None:
            self.cur_total_reward_sum += total_rewards
        if mi_rewards is not None:
            self.cur_mi_reward_sum += mi_rewards
        self.cur_episode_length += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        self.cur_reward_sum[new_ids] = 0
        self.cur_episode_length[new_ids] = 0
        if intrinsic_rewards is not None:
            self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_ereward_sum[new_ids] = 0
            self.cur_ireward_sum[new_ids] = 0
        if style_rewards is not None and total_rewards is not None:
            amp_new_ids = new_ids if len(new_ids) > 0 else slice(None)
            style_rew_episode_mean = (
                torch.mean(self.cur_style_reward_sum[amp_new_ids]) / self.max_episode_length_s
            )
            if len(new_ids) > 0 and self.ep_extras:
                self.ep_extras[-1]["Episode_Reward/style"] = style_rew_episode_mean.item()
            self.total_rewbuffer.extend(self.cur_total_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.style_rewbuffer.extend(self.cur_style_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_style_reward_sum[new_ids] = 0
            self.cur_total_reward_sum[new_ids] = 0
        if mi_rewards is not None and len(new_ids) > 0:
            self.mi_rewbuffer.extend(self.cur_mi_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_mi_reward_sum[new_ids] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log ASE pretrain metrics only."""
        if self.writer is None:
            return

        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time
        mapped_losses = self._map_loss_dict(loss_dict)

        extras_string = ""
        if self.ep_extras:
            for key in sorted({k for ep_info in self.ep_extras for k in ep_info}):
                if not self._include_ep_extra(key):
                    continue
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_extras:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar(key, value, it)
                extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        for key, value in mapped_losses.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
        self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

        fps = int(collection_size / (collect_time + learn_time))
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(self.rewbuffer) > 0:
            self.writer.add_scalar("ASE/mean_total_reward", statistics.mean(self.total_rewbuffer), it)
            self.writer.add_scalar("ASE/mean_style_reward", statistics.mean(self.style_rewbuffer), it)
            if self.mi_rewbuffer:
                self.writer.add_scalar("ASE/mean_mi_reward", statistics.mean(self.mi_rewbuffer), it)
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
            if self.logger_type != "WandbLogWriter":
                self.writer.add_scalar(
                    "Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time)
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
                )

        log_string = f"""{"#" * width}\n"""
        log_string += f"""\033[1m{f" ASE learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

        run_name = self.cfg.get("run_name")
        log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""

        log_string += (
            f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
            f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
            f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
            f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
        )

        for key, value in mapped_losses.items():
            if key in self._REWARD_LIKE_KEYS:
                log_string += f"""{f"Mean {key}:":>{pad}} {value:.4f}\n"""
            else:
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""

        if len(self.rewbuffer) > 0:
            log_string += f"""{"Mean ASE total reward:":>{pad}} {statistics.mean(self.total_rewbuffer):.2f}\n"""
            log_string += f"""{"Mean ASE style reward:":>{pad}} {statistics.mean(self.style_rewbuffer):.2f}\n"""
            if self.mi_rewbuffer:
                log_string += f"""{"Mean ASE MI reward:":>{pad}} {statistics.mean(self.mi_rewbuffer):.2f}\n"""
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

        log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}\n"""

        if not print_minimal:
            log_string += extras_string

        done_it = it + 1 - start_it
        remaining_it = total_it - start_it - done_it
        eta = self.tot_time / done_it * remaining_it
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {datetime.timedelta(seconds=int(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {datetime.timedelta(seconds=int(eta))}\n"""
        )
        print(log_string)
        self.ep_extras.clear()
