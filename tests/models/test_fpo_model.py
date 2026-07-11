# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FPOModel."""

from __future__ import annotations

import torch
import pytest

from rsl_rl.models import FPOModel
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _make_fpo_model(**kwargs: object) -> tuple[FPOModel, torch.Tensor]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "num_flow_steps": 4,
        "num_fpo_samples": 3,
    }
    defaults.update(kwargs)
    model = FPOModel(obs, OBS_GROUPS, "actor", NUM_ACTIONS, **defaults)
    return model, obs


class TestFPOModel:
    def test_forward_shape(self) -> None:
        actor, obs = _make_fpo_model()
        output = actor(obs)
        assert output.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_sample_with_info(self) -> None:
        actor, obs = _make_fpo_model()
        actions, info = actor.sample_with_info(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        assert info.loss_eps.shape == (NUM_ENVS, 3, NUM_ACTIONS)
        assert info.loss_t.shape == (NUM_ENVS, 3, 1)
        assert info.initial_cfm_loss.shape == (NUM_ENVS, 3)

    def test_cfm_loss_is_differentiable(self) -> None:
        actor, obs = _make_fpo_model()
        actions, info = actor.sample_with_info(obs)
        obs_latent = actor.get_latent(obs)
        loss = actor.compute_cfm_loss(
            obs_latent,
            actions,
            info.loss_eps[:, 0, :],
            info.loss_t[:, 0, :],
        )
        loss.mean().backward()
        assert actor.flow_mlp[0].weight.grad is not None

    def test_moe_backbone(self) -> None:
        actor, obs = _make_fpo_model(num_experts=3, top_k=2)
        output = actor(obs)
        assert output.shape == (NUM_ENVS, NUM_ACTIONS)
        from rsl_rl.modules import MoEMLP

        assert isinstance(actor.flow_mlp, MoEMLP)

    def test_rejects_distribution_cfg(self) -> None:
        obs = make_obs(NUM_ENVS, OBS_DIM)
        with pytest.raises(ValueError, match="distribution_cfg"):
            FPOModel(
                obs,
                OBS_GROUPS,
                "actor",
                NUM_ACTIONS,
                distribution_cfg={"class_name": "GaussianDistribution"},
            )
