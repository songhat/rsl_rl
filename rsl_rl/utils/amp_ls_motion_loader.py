# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load DroidUp-style AMP NPZ clips as discriminator (s, s') expert states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rsl_rl.algorithms.amp_constants import (
    AMP_KEY_BODY_NAMES,
    AMP_LABEL_NAMES,
    MJLAB_JOINT_NAMES,
)


def _quat_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    scalar, xyz = quaternion[..., :1], quaternion[..., 1:]
    cross = 2.0 * torch.linalg.cross(xyz, vector)
    return vector + scalar * cross + torch.linalg.cross(xyz, cross)


def _matrix_from_quat(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


class AmpLSMotionLoader:
    """Expert motion sampler matching DroidUp LS-AMP disc observations."""

    def __init__(
        self,
        motion_files: tuple[str, ...],
        device: str,
        transition_dt: float,
        preload_transitions: int,
        motion_weights: tuple[float, ...],
        motion_labels: tuple[str, ...],
        joint_names: tuple[str, ...] | None = None,
        key_body_names: tuple[str, ...] | None = None,
        root_dir: str | Path | None = None,
    ):
        self.joint_names = tuple(joint_names or MJLAB_JOINT_NAMES)
        self.key_body_names = tuple(key_body_names or AMP_KEY_BODY_NAMES)
        self.amp_obs_dim = (
            6
            + 2 * len(self.joint_names)
            + 3 * len(self.key_body_names)
            + 6 * len(self.key_body_names)
        )
        self.device = device
        self.transition_dt = transition_dt
        self.root_dir = Path(root_dir).resolve() if root_dir is not None else Path.cwd()
        paths = self._resolve_files(motion_files)
        self.motion_names = tuple(path.stem for path in paths)
        if len(motion_labels) != len(paths):
            raise ValueError(
                f"AMP motion_labels has {len(motion_labels)} entries; "
                f"expected one per motion file ({len(paths)})."
            )
        label_to_id = {name: index for index, name in enumerate(AMP_LABEL_NAMES)}
        try:
            label_ids = [label_to_id[label] for label in motion_labels]
        except KeyError as error:
            raise ValueError(
                f"Unsupported AMP motion label {error.args[0]!r}; "
                f"expected one of {AMP_LABEL_NAMES}."
            ) from error
        self.motion_label_ids = torch.tensor(label_ids, device=device, dtype=torch.long)
        self.motion_labels = torch.nn.functional.one_hot(
            self.motion_label_ids, num_classes=len(AMP_LABEL_NAMES)
        ).to(dtype=torch.float32)
        self.trajectories: list[torch.Tensor] = []
        self.frame_dt: list[float] = []
        self.motion_weights = self._validate_weights(motion_weights, len(paths), "motion")
        self.last_label_fractions = torch.zeros(len(AMP_LABEL_NAMES), device=device)
        self.last_motion_fractions = torch.zeros(len(paths), device=device)

        print("\n========= AMP locomotion motion files ========")
        for path in paths:
            trajectory, frame_dt = self._load(path)
            self.trajectories.append(trajectory)
            self.frame_dt.append(frame_dt)
            print(f"{path.name}: {len(trajectory)} frames, {1.0 / frame_dt:.1f} fps")
        print("=============================================\n")

        self.preloaded_state: list[torch.Tensor] | None = None
        self.preloaded_next_state: list[torch.Tensor] | None = None
        if preload_transitions > 0:
            print(f"Preloading {preload_transitions} AMP expert transitions...")
            self.preloaded_state = []
            self.preloaded_next_state = []
            per_motion, remainder = divmod(preload_transitions, len(self.trajectories))
            for trajectory_id, name in enumerate(self.motion_names):
                count = per_motion + int(trajectory_id < remainder)
                state, next_state = self._sample_trajectory(trajectory_id, count)
                self.preloaded_state.append(state)
                self.preloaded_next_state.append(next_state)
                print(f"  {name}: {count} transitions")

    def _validate_weights(
        self,
        values: tuple[float, ...],
        motion_count: int,
        label: str,
    ) -> torch.Tensor:
        weights = torch.tensor(values, device=self.device, dtype=torch.float32)
        if weights.shape != (motion_count,):
            raise ValueError(
                f"AMP {label} weights contain {len(weights)} values; expected {motion_count}."
            )
        if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
            raise ValueError(f"AMP {label} weights must be finite and non-negative.")
        if weights.sum() <= 0.0:
            raise ValueError(f"AMP {label} weights need a positive sum.")
        return weights / weights.sum()

    def _resolve_files(self, entries) -> list[Path]:
        paths: list[Path] = []
        for entry in entries:
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = (self.root_dir / path).resolve()
            matches = sorted(path.glob("*.npz")) if path.is_dir() else [path]
            for match in matches:
                if not match.exists():
                    raise FileNotFoundError(f"AMP motion file not found: {match}")
                paths.append(match)
        if not paths:
            raise ValueError("No AMP motion files were configured.")
        return paths

    def _load(self, path: Path) -> tuple[torch.Tensor, float]:
        with np.load(path, allow_pickle=False) as data:
            required = (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "key_body_pos",
                "key_body_quat_w",
                "fps",
            )
            missing = [name for name in required if name not in data.files]
            if missing:
                raise ValueError(f"{path} is missing AMP fields: {missing}")
            for name in required[:-1]:
                if not np.isfinite(data[name]).all():
                    raise ValueError(f"{path} contains non-finite values in '{name}'.")
            if data["joint_pos"].shape[1] != len(self.joint_names):
                raise ValueError(
                    f"{path} has {data['joint_pos'].shape[1]} joints; "
                    f"expected {len(self.joint_names)}."
                )
            if data["key_body_pos"].shape[1] != 3 * len(self.key_body_names):
                raise ValueError(f"{path} does not contain the configured AMP key bodies.")
            if data["key_body_quat_w"].shape[1:] != (len(self.key_body_names), 4):
                raise ValueError(f"{path} has invalid AMP key-body orientations.")
            if "joint_names" in data.files:
                stored_joint_names = data["joint_names"].astype(str).tolist()
                if stored_joint_names != list(self.joint_names):
                    raise ValueError(
                        f"{path} joint order does not match the configured order:\n"
                        f"NPZ: {stored_joint_names}\nCFG: {list(self.joint_names)}"
                    )
            if "key_body_names" in data.files:
                stored_key_bodies = data["key_body_names"].astype(str).tolist()
                if stored_key_bodies != list(self.key_body_names):
                    raise ValueError(f"{path} has an unexpected AMP key-body order.")

            joint_pos = torch.as_tensor(data["joint_pos"], device=self.device, dtype=torch.float32)
            joint_vel = torch.as_tensor(data["joint_vel"], device=self.device, dtype=torch.float32)
            root_pos = torch.as_tensor(
                data["body_pos_w"][:, 0], device=self.device, dtype=torch.float32
            )
            root_quat = torch.as_tensor(
                data["body_quat_w"][:, 0], device=self.device, dtype=torch.float32
            )
            lin_vel_w = torch.as_tensor(
                data["body_lin_vel_w"][:, 0], device=self.device, dtype=torch.float32
            )
            ang_vel_w = torch.as_tensor(
                data["body_ang_vel_w"][:, 0], device=self.device, dtype=torch.float32
            )
            key_pos_w = torch.as_tensor(
                data["key_body_pos"], device=self.device, dtype=torch.float32
            ).reshape(-1, len(self.key_body_names), 3)
            key_quat_w = torch.as_tensor(
                data["key_body_quat_w"], device=self.device, dtype=torch.float32
            )
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])

        if len(joint_pos) < 2:
            raise ValueError(f"{path} must contain at least two motion frames.")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{path} has an invalid fps value: {fps}.")
        max_lin_vel = torch.linalg.vector_norm(lin_vel_w, dim=1).max().item()
        max_ang_vel = torch.linalg.vector_norm(ang_vel_w, dim=1).max().item()
        if max_lin_vel > 10.0 or max_ang_vel > 20.0:
            raise ValueError(
                f"{path} has implausible root velocity peaks "
                f"({max_lin_vel:.2f} m/s, {max_ang_vel:.2f} rad/s)."
            )

        inverse = _quat_inverse(root_quat)
        lin_vel_b = _quat_apply(inverse, lin_vel_w)
        ang_vel_b = _quat_apply(inverse, ang_vel_w)
        relative = key_pos_w - root_pos.unsqueeze(1)
        inverse_repeated = inverse.unsqueeze(1).expand(-1, len(self.key_body_names), -1)
        key_pos_b = _quat_apply(inverse_repeated, relative).flatten(start_dim=1)
        root_quat_inverse = _quat_inverse(root_quat).unsqueeze(1).expand_as(key_quat_w)
        key_quat_b = _quat_mul(root_quat_inverse, key_quat_w)
        key_ori_b = _matrix_from_quat(key_quat_b)[..., :2].flatten(start_dim=1)
        trajectory = torch.cat(
            (lin_vel_b, ang_vel_b, joint_pos, joint_vel, key_pos_b, key_ori_b),
            dim=1,
        )
        if trajectory.shape[1] != self.amp_obs_dim:
            raise RuntimeError(
                f"Expected {self.amp_obs_dim} AMP values, got {trajectory.shape[1]}."
            )
        return trajectory, 1.0 / fps

    def _sample_trajectory(
        self, trajectory_id: int, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory = self.trajectories[trajectory_id]
        expert_dt = self.frame_dt[trajectory_id]
        duration = (len(trajectory) - 1) * expert_dt
        available = max(duration - self.transition_dt, 0.0)
        times = torch.rand(count, device=self.device) * available
        state = self._interpolate(trajectory, times, expert_dt)
        next_state = self._interpolate(trajectory, times + self.transition_dt, expert_dt)
        return state, next_state

    def _draw_trajectory_ids(self, motion_label: torch.Tensor) -> torch.Tensor:
        motion_label = motion_label.to(device=self.device, dtype=torch.float32)
        if motion_label.ndim != 2 or motion_label.shape[1] != len(AMP_LABEL_NAMES):
            raise ValueError(
                f"AMP motion labels must have shape [N, {len(AMP_LABEL_NAMES)}], "
                f"got {tuple(motion_label.shape)}."
            )
        if not torch.allclose(
            motion_label.sum(dim=1), torch.ones(len(motion_label), device=self.device)
        ):
            raise ValueError("AMP motion labels must be one-hot vectors.")
        label_ids = motion_label.argmax(dim=1)
        matching = self.motion_label_ids.unsqueeze(0) == label_ids.unsqueeze(1)
        weights = torch.where(
            matching,
            self.motion_weights.unsqueeze(0),
            torch.zeros_like(self.motion_weights.unsqueeze(0)),
        )
        if torch.any(weights.sum(dim=1) <= 0.0):
            missing = torch.unique(label_ids[weights.sum(dim=1) <= 0.0]).tolist()
            raise ValueError(f"No expert motion is configured for labels {missing}.")
        trajectory_ids = torch.multinomial(weights, num_samples=1).squeeze(1)
        self.last_label_fractions = torch.bincount(
            label_ids, minlength=len(AMP_LABEL_NAMES)
        ).float() / max(len(label_ids), 1)
        self.last_motion_fractions = torch.bincount(
            trajectory_ids, minlength=len(self.trajectories)
        ).float() / max(len(trajectory_ids), 1)
        return trajectory_ids

    def _sample(
        self, trajectory_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(trajectory_ids)
        state = torch.empty(count, self.amp_obs_dim, device=self.device)
        next_state = torch.empty_like(state)
        for trajectory_id in torch.unique(trajectory_ids).tolist():
            mask = trajectory_ids == trajectory_id
            sample_count = int(mask.sum().item())
            sampled_state, sampled_next = self._sample_trajectory(trajectory_id, sample_count)
            state[mask] = sampled_state
            next_state[mask] = sampled_next
        return state, next_state

    @staticmethod
    def _interpolate(
        trajectory: torch.Tensor, times: torch.Tensor, frame_dt: float
    ) -> torch.Tensor:
        frame_position = torch.clamp(
            times / frame_dt, min=0.0, max=float(len(trajectory) - 1)
        )
        low = torch.floor(frame_position).long()
        high = torch.clamp(low + 1, max=len(trajectory) - 1)
        blend = (frame_position - low).unsqueeze(1)
        return torch.lerp(trajectory[low], trajectory[high], blend)

    def sample(
        self, motion_label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory_ids = self._draw_trajectory_ids(motion_label)
        if self.preloaded_state is None or self.preloaded_next_state is None:
            state, next_state = self._sample(trajectory_ids)
        else:
            count = len(trajectory_ids)
            state = torch.empty(count, self.amp_obs_dim, device=self.device)
            next_state = torch.empty_like(state)
            for trajectory_id in torch.unique(trajectory_ids).tolist():
                mask = trajectory_ids == trajectory_id
                sample_count = int(mask.sum().item())
                source = self.preloaded_state[trajectory_id]
                source_next = self.preloaded_next_state[trajectory_id]
                index = torch.randint(len(source), (sample_count,), device=self.device)
                state[mask] = source[index]
                next_state[mask] = source_next[index]
        labels = self.motion_labels[trajectory_ids]
        return state, next_state, labels
