# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""AMP expert motion loading (ported from amp-rsl-rl for integrated rsl_rl AMP)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from rsl_rl.utils import resolve_callable


class VelocityRepresentation(Enum):
  """Base velocity frame convention for AMP observations."""

  BODY_FIXED_REPRESENTATION = "body_fixed"
  MIXED_REPRESENTATION = "mixed"


def _call_augmentation_func(
  fn,
  *,
  obs: Optional[torch.Tensor] = None,
  actions: Optional[torch.Tensor] = None,
  obs_type: Any = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
  parameters = inspect.signature(fn).parameters
  kwargs: dict[str, Any] = {}

  if "obs" in parameters:
    kwargs["obs"] = obs
  if "actions" in parameters:
    kwargs["actions"] = actions
  if "obs_type" in parameters and obs_type is not None:
    kwargs["obs_type"] = obs_type

  result = fn(**kwargs)
  if isinstance(result, tuple):
    return result
  return result, None


@dataclass
class MotionData:
  """Motion clip tensors for AMP reset / discriminator demo sampling."""

  joint_positions: Union[torch.Tensor, np.ndarray]
  joint_velocities: Union[torch.Tensor, np.ndarray]
  base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
  base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
  base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
  base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
  base_quat: Union[Rotation, torch.Tensor]
  device: torch.device = torch.device("cpu")

  def __post_init__(self) -> None:
    def to_tensor(x: np.ndarray) -> torch.Tensor:
      return torch.tensor(x, device=self.device, dtype=torch.float32)

    if isinstance(self.joint_positions, np.ndarray):
      self.joint_positions = to_tensor(self.joint_positions)
    if isinstance(self.joint_velocities, np.ndarray):
      self.joint_velocities = to_tensor(self.joint_velocities)
    if isinstance(self.base_lin_velocities_mixed, np.ndarray):
      self.base_lin_velocities_mixed = to_tensor(self.base_lin_velocities_mixed)
    if isinstance(self.base_ang_velocities_mixed, np.ndarray):
      self.base_ang_velocities_mixed = to_tensor(self.base_ang_velocities_mixed)
    if isinstance(self.base_lin_velocities_local, np.ndarray):
      self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
    if isinstance(self.base_ang_velocities_local, np.ndarray):
      self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)
    if isinstance(self.base_quat, Rotation):
      quat_xyzw = self.base_quat.as_quat()
      self.base_quat = torch.tensor(
        quat_xyzw[:, [3, 0, 1, 2]],
        device=self.device,
        dtype=torch.float32,
      )

  def __len__(self) -> int:
    return self.joint_positions.shape[0]

  def get_amp_dataset_obs(
    self,
    indices: torch.Tensor,
    velocity_representation: VelocityRepresentation = VelocityRepresentation.BODY_FIXED_REPRESENTATION,
  ) -> torch.Tensor:
    if velocity_representation == VelocityRepresentation.MIXED_REPRESENTATION:
      return torch.cat(
        (
          self.joint_positions[indices],
          self.joint_velocities[indices],
          self.base_lin_velocities_mixed[indices],
          self.base_ang_velocities_mixed[indices],
        ),
        dim=1,
      )
    return torch.cat(
      (
        self.joint_positions[indices],
        self.joint_velocities[indices],
        self.base_lin_velocities_local[indices],
        self.base_ang_velocities_local[indices],
      ),
      dim=1,
    )

  def get_state_for_reset(
    self,
    indices: torch.Tensor,
    velocity_representation: VelocityRepresentation = VelocityRepresentation.BODY_FIXED_REPRESENTATION,
  ) -> tuple[torch.Tensor, ...]:
    if velocity_representation == VelocityRepresentation.MIXED_REPRESENTATION:
      return (
        self.base_quat[indices],
        self.joint_positions[indices],
        self.joint_velocities[indices],
        self.base_lin_velocities_mixed[indices],
        self.base_ang_velocities_mixed[indices],
      )
    return (
      self.base_quat[indices],
      self.joint_positions[indices],
      self.joint_velocities[indices],
      self.base_lin_velocities_local[indices],
      self.base_ang_velocities_local[indices],
    )


class AMPLoader:
  """Load and resample expert ``.npy`` motion clips for AMP training."""

  def __init__(
    self,
    device: str,
    dataset_path_root: Path,
    datasets: dict[str, float],
    simulation_dt: float,
    slow_down_factor: int,
    expected_joint_names: Union[List[str], None] = None,
    symmetry_cfg: Optional[dict[str, Any]] = None,
    velocity_representation: VelocityRepresentation = VelocityRepresentation.BODY_FIXED_REPRESENTATION,
  ) -> None:
    self.device = device
    self.velocity_representation = velocity_representation
    if isinstance(dataset_path_root, str):
      dataset_path_root = Path(dataset_path_root)

    if symmetry_cfg is not None:
      aug_fn = symmetry_cfg.get("amp_dataset_augmentation_func", None)
      if isinstance(aug_fn, str):
        symmetry_cfg["amp_dataset_augmentation_func"] = resolve_callable(aug_fn)
      aug_fn = symmetry_cfg.get("amp_dataset_augmentation_func", None)
      if aug_fn is not None and not callable(aug_fn):
        raise ValueError(
          "Symmetry configuration exists but the function is not callable: "
          f"{aug_fn}"
        )
      self.symmetry_cfg = symmetry_cfg
    else:
      self.symmetry_cfg = None

    dataset_names = list(datasets.keys())
    dataset_weights = list(datasets.values())

    if expected_joint_names is None:
      joint_union: list[str] = []
      seen: set[str] = set()
      for name in dataset_names:
        info = np.load(str(dataset_path_root / f"{name}.npy"), allow_pickle=True).item()
        for joint in info["joints_list"]:
          if joint not in seen:
            seen.add(joint)
            joint_union.append(joint)
      expected_joint_names = joint_union

    self.motion_data: list[MotionData] = []
    for dataset_name in dataset_names:
      dataset_path = dataset_path_root / f"{dataset_name}.npy"
      md = self.load_data(
        dataset_path,
        simulation_dt,
        slow_down_factor,
        expected_joint_names,
      )
      self.motion_data.append(md)

    weights = torch.tensor(dataset_weights, dtype=torch.float32, device=self.device)
    self.dataset_weights = weights / weights.sum()

    obs_list, next_obs_list, reset_states = [], [], []
    augmented_lengths: list[int] = []
    original_lengths: list[int] = []
    for data, weight in zip(self.motion_data, self.dataset_weights):
      t = len(data)
      idx = torch.arange(t, device=self.device)
      obs = data.get_amp_dataset_obs(idx, self.velocity_representation)
      next_idx = torch.clamp(idx + 1, max=t - 1)
      next_obs = data.get_amp_dataset_obs(next_idx, self.velocity_representation)

      if self.symmetry_cfg and self.symmetry_cfg.get("use_amp_dataset_augmentation", False):
        obs = self._apply_symmetry(obs=obs, obs_type="amp")
        next_obs = self._apply_symmetry(obs=next_obs, obs_type="amp")

      obs_list.append(obs)
      next_obs_list.append(next_obs)
      augmented_lengths.append(obs.shape[0])

      quat, jp, jv, blv, bav = data.get_state_for_reset(idx, self.velocity_representation)
      reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))
      original_lengths.append(t)

    self.all_obs = torch.cat(obs_list, dim=0)
    self.all_next_obs = torch.cat(next_obs_list, dim=0)
    self.all_states = torch.cat(reset_states, dim=0)

    per_frame = torch.cat(
      [
        torch.full((length,), weight / length, device=self.device)
        for weight, length in zip(self.dataset_weights, augmented_lengths)
      ]
    )
    self.per_frame_weights = per_frame / per_frame.sum()

    per_frame_reset = torch.cat(
      [
        torch.full((length,), weight / length, device=self.device)
        for weight, length in zip(self.dataset_weights, original_lengths)
      ]
    )
    self.per_frame_weights_reset = per_frame_reset / per_frame_reset.sum()

  def _apply_symmetry(
    self,
    *,
    obs: Optional[torch.Tensor],
    obs_type: Optional[Any] = None,
  ) -> Optional[torch.Tensor]:
    if self.symmetry_cfg is None:
      return obs
    aug_fn = self.symmetry_cfg.get("amp_dataset_augmentation_func", None)
    if aug_fn is None:
      return obs
    aug_obs, _ = _call_augmentation_func(aug_fn, obs=obs, obs_type=obs_type)
    return aug_obs if aug_obs is not None else obs

  def _resample_data_Rn(
    self,
    data: list[np.ndarray],
    original_keyframes,
    target_keyframes,
  ) -> np.ndarray:
    return interp1d(original_keyframes, data, axis=0)(target_keyframes)

  def _resample_data_SO3(
    self,
    raw_quaternions: list[np.ndarray],
    original_keyframes,
    target_keyframes,
  ) -> Rotation:
    slerp = Slerp(original_keyframes, Rotation.from_quat(raw_quaternions))
    return slerp(target_keyframes)

  def _compute_ang_vel(
    self,
    data: list[Rotation],
    dt: float,
    local: bool = False,
  ) -> np.ndarray:
    r_prev = data[:-1]
    r_next = data[1:]
    rel = r_prev.inv() * r_next if local else r_next * r_prev.inv()
    rotvec = rel.as_rotvec() / dt
    return np.vstack((rotvec, rotvec[-1]))

  def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
    d = (data[1:] - data[:-1]) / dt
    return np.vstack([d, d[-1:]])

  def load_data(
    self,
    dataset_path: Path,
    simulation_dt: float,
    slow_down_factor: int = 1,
    expected_joint_names: Union[List[str], None] = None,
  ) -> MotionData:
    data = np.load(str(dataset_path), allow_pickle=True).item()
    dataset_joint_names = data["joints_list"]

    idx_map: list[Union[int, None]] = []
    for joint in expected_joint_names:
      idx_map.append(dataset_joint_names.index(joint) if joint in dataset_joint_names else None)

    jp_list: list[np.ndarray] = []
    for frame in data["joint_positions"]:
      arr = np.zeros((len(idx_map),), dtype=frame.dtype)
      for i, src_idx in enumerate(idx_map):
        if src_idx is not None:
          arr[i] = frame[src_idx]
      jp_list.append(arr)

    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if isinstance(data["fps"], np.ndarray) else float(data["fps"])
    dt = 1.0 / fps / float(slow_down_factor)
    t = len(jp_list)
    t_orig = np.linspace(0, t * dt, t)
    t_new = int(t * dt / simulation_dt)
    t_resampled = np.linspace(0, t * dt, t_new)

    resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_resampled)
    resampled_joint_velocities = self._compute_raw_derivative(
      resampled_joint_positions, simulation_dt
    )
    resampled_base_positions = self._resample_data_Rn(data["root_position"], t_orig, t_resampled)
    resampled_base_orientations = self._resample_data_SO3(
      data["root_quaternion"], t_orig, t_resampled
    )
    resampled_base_lin_vel_mixed = self._compute_raw_derivative(
      resampled_base_positions, simulation_dt
    )
    resampled_base_ang_vel_mixed = self._compute_ang_vel(
      resampled_base_orientations, simulation_dt, local=False
    )
    resampled_base_lin_vel_local = np.stack(
      [
        rot.as_matrix().T @ vel
        for rot, vel in zip(resampled_base_orientations, resampled_base_lin_vel_mixed)
      ]
    )
    resampled_base_ang_vel_local = self._compute_ang_vel(
      resampled_base_orientations, simulation_dt, local=True
    )

    return MotionData(
      joint_positions=resampled_joint_positions,
      joint_velocities=resampled_joint_velocities,
      base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
      base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
      base_lin_velocities_local=resampled_base_lin_vel_local,
      base_ang_velocities_local=resampled_base_ang_vel_local,
      base_quat=resampled_base_orientations,
      device=self.device,
    )

  def feed_forward_generator(
    self, num_mini_batch: int, mini_batch_size: int
  ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    for _ in range(num_mini_batch):
      idx = torch.multinomial(self.per_frame_weights, mini_batch_size, replacement=True)
      yield self.all_obs[idx], self.all_next_obs[idx]

  def get_state_for_reset(self, number_of_samples: int) -> tuple[torch.Tensor, ...]:
    idx = torch.multinomial(self.per_frame_weights_reset, number_of_samples, replacement=True)
    full = self.all_states[idx]
    joint_dim = self.motion_data[0].joint_positions.shape[1]
    dims = [4, joint_dim, joint_dim, 3, 3]
    return torch.split(full, dims, dim=1)
