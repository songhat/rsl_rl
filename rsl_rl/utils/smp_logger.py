from __future__ import annotations

import datetime
import statistics
import torch
from collections import deque

from rsl_rl.utils.logger import Logger


class LoggerSMP(Logger):
    """Logger for SMP policy training (task + SDS style reward, no discriminator)."""

    # Keep TB diagnostics; console only shows Episode_Reward/style (see log()).
    _TB_ONLY_KEYS = {
        "smp/sds_loss_mean",
        "smp/sds_loss_std",
        "smp/reward_mean",
        "smp/sds_normalizer_count",
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
        self.max_episode_length_s = max_episode_length_s

    @staticmethod
    def _include_ep_extra(key: str) -> bool:
        if key.startswith("Episode_Reward/"):
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
    ) -> None:
        del style_rewards, total_rewards  # deferred SDS: style logged once per iter from loss_dict
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
        rnd_weight: float | None = None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        if self.writer is None:
            return

        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        # One style line, same prefix as task Episode_Reward/* for side-by-side compare.
        style_mean = loss_dict.get("smp/reward_mean")
        if style_mean is not None and self.ep_extras:
            self.ep_extras[-1]["Episode_Reward/style"] = float(style_mean)
        elif style_mean is not None:
            self.ep_extras.append({"Episode_Reward/style": float(style_mean)})

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
        self.ep_extras.clear()

        for key, value in loss_dict.items():
            if key in self._TB_ONLY_KEYS:
                self.writer.add_scalar(key, value, it)
            else:
                self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
        self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

        fps = int(collection_size / (collect_time + learn_time))
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(self.rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)

        log_string = f"""{'*' * width}\n"""
        log_string += f"""\033[1m{f" SMP learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""
        log_string += (
            f"""{"Computation:":>{pad}} {fps} steps/s """
            f"""(collection: {collect_time:.3f}s, learning {learn_time:.3f}s)\n"""
        )
        log_string += extras_string
        for key, value in loss_dict.items():
            if key in self._TB_ONLY_KEYS:
                continue
            log_string += f"""{f"Mean loss {key}:":>{pad}} {value:.4f}\n"""
        log_string += f"""{"Learning rate:":>{pad}} {learning_rate:.6f}\n"""
        eta = (total_it - it) * iteration_time
        log_string += f"""{"ETA:":>{pad}} {str(datetime.timedelta(seconds=int(eta)))}\n"""
        if not print_minimal:
            print(log_string)
