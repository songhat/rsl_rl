# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_model import CNNModel
from .discriminator_model import AMPDiscriminator, LossType, resolve_amp_config
from .fpo_model import FPOModel, FpoActionInfo
from .mlp_model import MLPModel
from .moe_mlp_model import MoEMLPModel
from .rnn_model import RNNModel

__all__ = [
    "AMPDiscriminator",
    "CNNModel",
    "FPOModel",
    "FpoActionInfo",
    "LossType",
    "MLPModel",
    "MoEMLPModel",
    "RNNModel",
    "resolve_amp_config",
]
