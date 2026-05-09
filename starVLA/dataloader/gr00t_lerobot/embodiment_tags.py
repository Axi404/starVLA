# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    OXE_DROID = "oxe_droid"
    """
    The OxE Droid dataset.
    """

    OXE_BRIDGE = "oxe_bridge"
    """
    The OxE Bridge dataset.
    """

    OXE_RT1 = "oxe_rt1"
    """
    The OxE RT-1 dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    FRANKA = 'franka'
    """
    The Franka Emika Panda robot.
    """

    # RoboChallenge Table30v1 (V1).  No data_config emits these yet — reserved
    # for when V1 lerobot conversion lands.  Schema/dims pending wire-up.
    TABLE30V1_UR5 = "table30v1_ur5"
    """RoboChallenge Table30v1 — UR5 single-arm."""

    TABLE30V1_ARX5 = "table30v1_arx5"
    """RoboChallenge Table30v1 — ARX5 single-arm."""

    TABLE30V1_DOSW1 = "table30v1_dosw1"
    """RoboChallenge Table30v1 — DOS-W1 dual-arm."""

    TABLE30V1_ALOHA = "table30v1_aloha"
    """RoboChallenge Table30v1 — ALOHA dual-arm."""

    # RoboChallenge Table30v2 (V2) — native dimensions:
    #   UR5 / ARX5 single-arm: state 7d / action 8d (ee + gripper)
    #   DOSW1 dual-arm:        state 14d / action 14d (joint targets)
    #   ALOHA dual-arm:        state 14d / action 16d (ee + gripper × 2)
    TABLE30V2_UR5 = "table30v2_ur5"
    """RoboChallenge Table30v2 — UR5 single-arm (8d action, native)."""

    TABLE30V2_ARX5 = "table30v2_arx5"
    """RoboChallenge Table30v2 — ARX5 single-arm (8d action, native)."""

    TABLE30V2_DOSW1 = "table30v2_dosw1"
    """RoboChallenge Table30v2 — DOS-W1 dual-arm (14d action, native)."""

    TABLE30V2_ALOHA = "table30v2_aloha"
    """RoboChallenge Table30v2 — ALOHA dual-arm (16d action, native)."""

# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.OXE_BRIDGE.value: 18,
    EmbodimentTag.OXE_RT1.value: 19,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.FRANKA.value: 25,
    EmbodimentTag.TABLE30V1_UR5.value: 7,
    EmbodimentTag.TABLE30V1_ARX5.value: 8,
    EmbodimentTag.TABLE30V1_DOSW1.value: 9,
    EmbodimentTag.TABLE30V1_ALOHA.value: 10,
    EmbodimentTag.TABLE30V2_UR5.value: 27,
    EmbodimentTag.TABLE30V2_ARX5.value: 28,
    EmbodimentTag.TABLE30V2_DOSW1.value: 29,
    EmbodimentTag.TABLE30V2_ALOHA.value: 30,
}

# Robot type to embodiment tag mapping
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
    "oxe_droid": EmbodimentTag.OXE_DROID,
    "oxe_bridge": EmbodimentTag.OXE_BRIDGE,
    "oxe_rt1": EmbodimentTag.OXE_RT1,
    "demo_sim_franka_delta_joints": EmbodimentTag.FRANKA,
    "custom_robot_config": EmbodimentTag.NEW_EMBODIMENT,
    "fourier_gr1_arms_waist": EmbodimentTag.GR1,
}
