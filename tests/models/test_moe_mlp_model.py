# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MoEMLPModel."""

from __future__ import annotations

import torch

from rsl_rl.models import MoEMLPModel
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _make_moe_model(stochastic: bool = False, **kwargs: object) -> tuple[MoEMLPModel, torch.Tensor]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "num_experts": 3,
        "top_k": 2,
    }
    if stochastic:
        defaults["distribution_cfg"] = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }
    defaults.update(kwargs)
    output_dim = NUM_ACTIONS if stochastic else 1
    model = MoEMLPModel(obs, OBS_GROUPS, "actor", output_dim, **defaults)
    return model, obs


class TestMoEMLPModel:
    def test_forward_shape(self) -> None:
        actor, obs = _make_moe_model(stochastic=True)
        output = actor(obs)
        assert output.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_stochastic_sampling(self) -> None:
        actor, obs = _make_moe_model(stochastic=True)
        torch.manual_seed(0)
        sample = actor(obs, stochastic_output=True)
        assert sample.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_expert_routing(self) -> None:
        actor, obs = _make_moe_model(stochastic=False, obs_set="critic")
        latent = actor.get_latent(obs)
        output = actor.mlp(latent)
        assert output.shape == (NUM_ENVS, 1)
        assert actor.mlp.num_experts == 3
