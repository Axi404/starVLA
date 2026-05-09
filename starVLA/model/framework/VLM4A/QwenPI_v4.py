# Copyright 2025 starVLA community. Licensed under the MIT License.
# Implemented by Jinhui YE / HKUST in 2025.
"""QwenPI_v4: Qwen2.5-VL / Qwen3-VL + layer-wise cross-DiT flow-matching action head.

Two improvements over QwenPI:
1. Per-VLM-layer projector (LayerNorm + Linear) compresses each VL hidden to
   ``action_dit_hidden_dim`` before the action DiT, so DiT can run at a
   smaller latent dim than the VLM hidden.
2. Discretised state injected as plain tokens into the instruction
   (``[STATE] <bins> [ACTION]``, π₀.5-style) instead of a separate state encoder.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# Robot layout table — controls the instruction meta-tokens injected before each
# task description.  Add new embodiments here and (if needed) wire their
# EmbodimentTag.value into EMBODIMENT_TAG_TO_LAYOUT_KEY below.
#
# `fps` is the *effective* action sample rate seen by the model:
#   fps = camera_fps / data_config action_indices stride
# All Table30v2 robots now use stride=4 in data_config, so fps=7.5 across the
# board (camera 30 Hz / 4).
ROBOT_LAYOUTS: Dict[str, Dict[str, Any]] = {
    # RoboChallenge Table30v2 (V2). Add a robochallenge_v1_* group below the
    # legacy section if/when V1 data_config gets wired up.
    "robochallenge_v2_ur5": {
        "action_dim": 8,
        "chunk_size": 50,
        "robo_info": "single arm, abs ee (7-dof quat + gripper)",
        "display_tag": "RoboChallenge Table30v2 UR5",
        "arm_type": "single arm",
        "fps": 7.5,
    },
    "robochallenge_v2_arx5": {
        "action_dim": 8,
        "chunk_size": 50,
        "robo_info": "single arm, abs ee (7-dof quat + gripper)",
        "display_tag": "RoboChallenge Table30v2 ARX5",
        "arm_type": "single arm",
        "fps": 7.5,
    },
    "robochallenge_v2_dosw1": {
        "action_dim": 14,
        "chunk_size": 50,
        "robo_info": "dual arms, abs joint (6-dof + gripper) × (left + right)",
        "display_tag": "RoboChallenge Table30v2 DOS-W1",
        "arm_type": "dual arms",
        "fps": 7.5,
    },
    "robochallenge_v2_aloha": {
        "action_dim": 16,
        "chunk_size": 50,
        "robo_info": "bimanual, abs ee (7-dof quat + gripper) × (left + right)",
        "display_tag": "RoboChallenge Table30v2 ALOHA",
        "arm_type": "dual arms",
        "fps": 7.5,
    },
    "franka": {
        "action_dim": 7,
        "chunk_size": 16,
        "robo_info": "single arm, delta eef",
        "display_tag": "Franka Emika Panda",
        "arm_type": "single arm",
        "fps": 20.0,
    },
    "oxe_bridge": {
        "action_dim": 7,
        "chunk_size": 16,
        "robo_info": "single arm, delta eef",
        "display_tag": "WidowX-250",
        "arm_type": "single arm",
        "fps": 5.0,
    },
}

# Maps the EmbodimentTag.value emitted by LeRobotSingleDataset._pack_sample
# (sample["robot_tag"]) to a ROBOT_LAYOUTS key.  Add a "table30v1_*" group
# when the V1 data_config / converter lands.
EMBODIMENT_TAG_TO_LAYOUT_KEY: Dict[str, str] = {
    "table30v2_ur5":   "robochallenge_v2_ur5",
    "table30v2_arx5":  "robochallenge_v2_arx5",
    "table30v2_dosw1": "robochallenge_v2_dosw1",
    "table30v2_aloha": "robochallenge_v2_aloha",
}


@dataclass
class QwenPI_v4DefaultConfig:
    """Default values for QwenPI_v4.  YAML overrides win on conflicts.

    See ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for the
    relationship between vl_hidden_dim, action_dit_hidden_dim and
    cross_attention_dim.
    """

    name: str = "QwenPI_v4"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 36,    # auto-overridden at runtime from the loaded VLM
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                # When set, DiT internal hidden = action_dit_hidden_dim and
                # project_layers compresses VL hidden to this dim.  When None,
                # DiT internal hidden = vl_hidden_dim (== QwenPI behaviour).
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v4")
class QwenPI_v4(baseframework):
    """Qwen2.5-VL / Qwen3-VL + per-layer projector + layer-wise cross-DiT FM head.

    Predicts a future action chunk conditioned on multi-view images and a
    natural-language instruction (with optional discretised state prefix).
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenPI_v4DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read VL shape from the loaded VLM.  Qwen3-VL nests num_hidden_layers
        # under text_config; Qwen2.5-VL puts it on the top-level config.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # action_dit_hidden_dim is a framework-side hint, not a DiT kwarg —
        # populate_layerwise_dit_cfg writes the canonical DiT-shape fields.
        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None) or llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        # One LayerNorm+Linear per DiT layer to compress VL hidden → DiT hidden;
        # nn.Identity when sizes already match (== plain QwenPI behaviour).
        self.project_layers = nn.ModuleList(
            [
                nn.Identity()
                if llm_hidden_size == self.action_dit_hidden_dim
                else nn.Sequential(
                    nn.LayerNorm(llm_hidden_size),
                    nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]

    def _encode_vl(self, batch_images, instructions):
        """Run QwenVL → take the last N hidden states → project to DiT space."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            vl_embs_list = list(all_hidden[-self.num_action_dit_layers:])
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
        return vl_embs_list, backbone_attention_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        states = [example["state"] for example in examples] if "state" in examples[0] else None

        # State is injected as discretised tokens into the instruction (π₀.5-style);
        # the action_model does not receive a separate state tensor.
        instructions = self._add_robo_meta_tokens_to_instructions(examples, instructions, states)

        vl_embs_list, backbone_attention_mask = self._encode_vl(batch_images, instructions)
        base = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base.device, dtype=base.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            # >2 multiplies per-step memory of every repeated VLM-layer embedding.
            repeated_diffusion_steps = int(
                self.config.framework.action_model.get("repeated_diffusion_steps", 2)
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(
                    repeated_diffusion_steps, 1
                ).to(dtype=torch.bool)

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> np.ndarray:
        """Run the flow-matching sampler and return the denoised action chunk.

        Returns ``{"normalized_actions": np.ndarray (B, action_horizon, action_dim)}``
        in the normalised action space — caller is responsible for un-normalising.
        """
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        states = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = self._add_robo_meta_tokens_to_instructions(examples, instructions, states)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list, backbone_attention_mask = self._encode_vl(batch_images, instructions)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, None, encoder_attention_mask=backbone_attention_mask
            )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def state2str_transform(self, state: np.ndarray) -> str:
        """Quantise state ∈ [-1, 1] into 256 uniform bins, return as space-separated tokens."""
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def _resolve_robot_layout(self, example: dict) -> Optional[Dict[str, Any]]:
        """Look up ROBOT_LAYOUTS for this sample.  Returns None for unknown robots."""
        robot_tag = example.get("robot_tag")
        if robot_tag is None:
            return None
        if robot_tag in ROBOT_LAYOUTS:
            return ROBOT_LAYOUTS[robot_tag]
        mapped = EMBODIMENT_TAG_TO_LAYOUT_KEY.get(robot_tag)
        if mapped and mapped in ROBOT_LAYOUTS:
            return ROBOT_LAYOUTS[mapped]
        return None

    def _add_robo_meta_tokens_to_instructions(
        self,
        examples: List[dict],
        instructions: List[str],
        states: Optional[List[np.ndarray]] = None,
    ) -> List[str]:
        """Build the augmented instruction string for each sample.

        Known robot, state present:
            "Task: {instr}. The robot is {tag} with {arm}. The control frequency
             is {fps} Hz. The current robot state is {state_str}.
             Please predict the next {chunk} actions to execute the Task."
        Known robot, no state: same template, state sentence omitted.
        Unknown robot: legacy "{instr} [STATE] {state_str} [ACTION]" if state present, else "{instr}".
        """
        enhanced = []
        for i, (example, instr) in enumerate(zip(examples, instructions)):
            layout = self._resolve_robot_layout(example)
            state_str = self.state2str_transform(states[i][0]) if states is not None else None

            if layout is None:
                if state_str is not None:
                    enhanced.append(f"{instr} [STATE] {state_str} [ACTION]")
                else:
                    enhanced.append(instr)
            else:
                state_part = f" The current robot state is {state_str}." if state_str is not None else ""
                enhanced.append(
                    f"Task: {instr}."
                    f" The robot is {layout['display_tag']} with {layout['arm_type']}."
                    f" The control frequency is {layout['fps']} Hz."
                    f"{state_part}"
                    f" Please predict the next {layout['chunk_size']} actions"
                    f" to execute the Task."
                )
        return enhanced


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/RoboChallenge_table30v2/train_files/starvla_qwenoft_robochallenge_table30v2.yaml",
    )
    args, _ = parser.parse_known_args()

    if os.getenv("DEBUG_MODE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("waiting for debugger...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    model = QwenPI_v4(cfg)

    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'Module':<35} {'Params':>14}  {'%':>6}")
    print("-" * 60)
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:<33} {n:>14,}  {100 * n / total:>5.1f}%")
    print("-" * 60)
    print(f"  {'TOTAL':<33} {total:>14,}  100.0%\n")

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(50, cfg.framework.action_model.action_dim)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, cfg.framework.action_model.state_dim)).astype(np.float16),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Action Loss:", model([sample, sample])["action_loss"].item())
    print("Predicted action shape:", model.predict_action([sample])["normalized_actions"].shape)
