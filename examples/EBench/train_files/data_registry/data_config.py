"""EBench benchmark — data config, embodiment tags, and mixtures.

Discovered by `starVLA/dataloader/gr00t_lerobot/registry.py:discover_and_merge()`
and merged into `ROBOT_TYPE_CONFIG_MAP`, overriding any same-named entry in the
base `starVLA/dataloader/gr00t_lerobot/data_config.py`.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


# ---------------------------------------------------------------------------
# DataConfig — EBench (dual-arm lift2)
# ---------------------------------------------------------------------------
class EBenchConfig:
    # Eval-only metadata: training doesn't need these (the data pipeline applies
    # the transform directly), but the server-side PolicyNormProcessor uses them
    # to split the combined 19-dim stats vector back into per-key blocks and to
    # build the DatasetMetadata for un-apply. Keep these here whenever you sync
    # data_config from the cloud training repo, or eval will fail with
    # "Cannot infer per-key dims" / "no attribute 'embodiment_tag'".
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.cam_over",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
        "state.base",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
        "action.base_delta",
    ]
    state_key_dims = {
        "state.left_joints": 6,
        "state.right_joints": 6,
        "state.left_gripper": 2,
        "state.right_gripper": 2,
        "state.base": 3,
    }
    action_key_dims = {
        "action.left_joints": 6,
        "action.right_joints": 6,
        "action.left_gripper": 2,
        "action.right_gripper": 2,
        "action.base_delta": 3,
    }

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.022,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "min_max",
                    "state.right_gripper": "min_max",
                    "state.base": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.022,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "min_max",
                    "action.right_gripper": "min_max",
                    "action.base_delta": "min_max",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


ROBOT_TYPE_CONFIG_MAP = {
    "ebench": EBenchConfig(),
}

# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "ebench_generalist": [
        ("long_horizon/fruit", 1.0, "ebench"),
        ("long_horizon/microwave", 1.0, "ebench"),
        ("long_horizon/detergent", 1.0, "ebench"),
        ("long_horizon/make_sandwich", 1.0, "ebench"),
        ("long_horizon/pen", 1.0, "ebench"),
        ("long_horizon/dishwasher", 1.0, "ebench"),
        ("long_horizon/dish", 1.0, "ebench"),
        ("long_horizon/bottle", 1.0, "ebench"),
        ("long_horizon/shop", 1.0, "ebench"),
        ("simple_pnp/task5", 1.0, "ebench"),
        ("simple_pnp/task1", 1.0, "ebench"),
        ("simple_pnp/task10", 1.0, "ebench"),
        ("simple_pnp/task4", 1.0, "ebench"),
        ("simple_pnp/task2", 1.0, "ebench"),
        ("simple_pnp/task9", 1.0, "ebench"),
        ("simple_pnp/task7", 1.0, "ebench"),
        ("simple_pnp/task6", 1.0, "ebench"),
        ("simple_pnp/task8", 1.0, "ebench"),
        ("simple_pnp/task3", 1.0, "ebench"),
        ("teleop_tasks/tighten_nut", 1.0, "ebench"),
        ("teleop_tasks/peg_in_hole", 1.0, "ebench"),
        ("teleop_tasks/flip_cup_collect_cookies", 1.0, "ebench"),
        ("teleop_tasks/collect_coffee_beans", 1.0, "ebench"),
        ("teleop_tasks/install_gear", 1.0, "ebench"),
        ("teleop_tasks/frame_against_pen_holder", 1.0, "ebench"),
        ("teleop_tasks/put_glass_in_glassbox", 1.0, "ebench"),
    ],
}
