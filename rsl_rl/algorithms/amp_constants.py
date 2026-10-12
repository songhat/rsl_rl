# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared constants for DroidUp-style least-squares AMP (AmpPPO)."""

# Default E1 21-DoF XML joint order. Robot-specific configs should override.
MJLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

AMP_KEY_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
)

# One-hot style labels appended to disc states during AmpPPO training.
AMP_LABEL_NAMES = ("walk", "run", "turn", "side", "stand")
AMP_LABEL_DIM = len(AMP_LABEL_NAMES)

# Must match UniformVelocityCommand.command_mode integers in the task MDP.
COMMAND_MODE_STANDING = 0
COMMAND_MODE_TURNING = 1
COMMAND_MODE_LATERAL = 2
COMMAND_MODE_MIXED = 3
