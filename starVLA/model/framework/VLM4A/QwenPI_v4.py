# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
QwenPI_v4 Framework
A Qwen2.5-VL / Qwen3-VL + layer-wise cross-DiT flow-matching action head.

Released checkpoint
─────────────────────────────
- Qwen3-VL-4B + Bridge V2 + RT-1 (OXE) co-training, 69.8% avg success on
  SimplerEnv WidowX:
  https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1

Key improvements over QwenPI
─────────────────────────────
1. **Compressed Action DiT via per-layer projectors**
   Each of the N VLM hidden-state layers is passed through a dedicated
   LayerNorm + Linear projector (`project_layers`) that maps the VLM hidden
   dimension (e.g. 2560) down to a smaller Action DiT latent dimension
   (e.g. 1024, controlled by `action_dit_hidden_dim`).  This reduces the
   action head parameter count by ~(vl_hidden / dit_hidden)² while keeping
   the full layer-wise cross-attention structure.

2. **Discretised-state language injection** (`add_discretized_state_to_instruction`)
   Proprioceptive state is quantised into 256 bins and appended to the
   language instruction as plain tokens (``[STATE] <bins> [ACTION]``),
   following the π₀.5 design.  This lets the VLM attend to state without
   any extra encoder module.

Together these two features bring QwenPI_v4 close to all the core
capabilities of π₀.5 within a single open-weight VLM framework.

Parameter breakdown (Qwen3-VL-4B + action_dit_hidden_dim=1024)
═══════════════════════════════════════════════════════════════
  Module                               Params        %
  ───────────────────────────────────────────────────
  qwen_vl_interface         4,437,815,808   87.5%
  action_model                538,678,305   10.6%
  project_layers               94,593,024    1.9%
  ───────────────────────────────────────────────────
  TOTAL                     5,071,087,137  100.0%
═══════════════════════════════════════════════════════════════
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################

# ──────────────────────────────────────────────────────────────────────
#  Robot layout table — controls the instruction meta-tokens injected
#  before each task description.  Add new embodiments here.
# ──────────────────────────────────────────────────────────────────────
ROBOT_LAYOUTS: Dict[str, Dict[str, Any]] = {
    # ── RoboChallenge Table30v2 ──────────────────────────────────────
    "robochallenge_ur5": {
        "action_dim": 8,
        "chunk_size": 50,
        "robo_info": "single arm, abs ee (7-dof quat + gripper)",
        "display_tag": "RoboChallenge UR5",
        "arm_type": "single arm",
        "fps": 30.0, # 如果在 data_config 里调整了 action_indices 的 stride，这里也要相应调整 fps / stride，保持时间维度对齐
    },
    "robochallenge_arx5": {
        "action_dim": 8,
        "chunk_size": 50,
        "robo_info": "single arm, abs ee (7-dof quat + gripper)",
        "display_tag": "RoboChallenge ARX5",
        "arm_type": "single arm",
        "fps": 30.0, # 如果在 data_config 里调整了 action_indices 的 stride，这里也要相应调整 fps / stride，保持时间维度对齐
    },
    "robochallenge_dosw1": {
        "action_dim": 8,
        "chunk_size": 50,
        "robo_info": "single arm, abs ee (7-dof quat + gripper)",
        "display_tag": "RoboChallenge DOS-W1",
        "arm_type": "single arm",
        "fps": 30.0,
    },
    "robochallenge_aloha": {
        "action_dim": 16,
        "chunk_size": 50,
        "robo_info": "bimanual, abs ee (7-dof quat + gripper) × (left + right)",
        "display_tag": "RoboChallenge ALOHA",
        "arm_type": "dual arms",
        "fps": 30.0,
    },
    # ── General / legacy ────────────────────────────────────────────
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

# Map from embodiment_tag value (stored in dataset samples) to ROBOT_LAYOUTS key.
# Keys must match EmbodimentTag.value strings exactly.
EMBODIMENT_TAG_TO_LAYOUT_KEY: Dict[str, str] = {
    # Generic single-arm / bimanual tags used by data_config.py
    "ur5":             "robochallenge_ur5",
    "arx5":            "robochallenge_arx5",
    "dos-w1":          "robochallenge_dosw1",   # note: hyphen, not underscore
    "aloha":           "robochallenge_aloha",
    # Table30v2-scoped variants (forward-compat: same layout as the bare tag)
    "table30v2_ur5":   "robochallenge_ur5",
    "table30v2_arx5":  "robochallenge_arx5",
    "table30v2_dosw1": "robochallenge_dosw1",
    "table30v2_aloha": "robochallenge_aloha",
}


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI_v4
#  - Same shape as QwenPIDefaultConfig (see QwenPI.py) but introduces the
#    optional `action_dit_hidden_dim` knob inside diffusion_model_cfg.
#  - Setting action_dit_hidden_dim to a value smaller than the VLM hidden
#    size lets the Action DiT run at a "compressed" latent dim while
#    `Qwen_PI_v3.project_layers` does the LayerNorm+Linear compression of
#    each VL hidden state to that dim.
#  - Leaving it None (or omitting it from YAML) reproduces the QwenPI
#    behaviour: DiT hidden = VLM hidden, projection becomes nn.Identity().
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPI_v4DefaultConfig:
    """QwenPI_v4 framework default parameters.

    See ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for the
    relationship between vl_hidden_dim, action_dit_hidden_dim and
    cross_attention_dim.
    """

    name: str = "QwenPI_v4"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 36,  # auto-overridden at runtime from the loaded VLM
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
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
                # When set (e.g. 1024), DiT internal hidden = action_dit_hidden_dim
                # and Qwen_PI_v3.project_layers compress VL hidden to this dim.
                # When None, DiT internal hidden = vl_hidden_dim (== QwenPI behaviour).
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
    """
    Multimodal vision-language-action model (QwenPI_v4 variant).

    Architecture
    ────────────
    - Qwen2.5-VL / Qwen3-VL backbone for fused language / vision token embeddings.
    - Per-layer projectors (``project_layers``): one LayerNorm + Linear per VLM
      layer that compresses VLM hidden states from ``vl_hidden_dim`` down to
      ``action_dit_hidden_dim`` before feeding the Action DiT.
    - Layer-wise cross-DiT flow-matching action head that attends to every
      selected VLM layer in parallel.

    Focus: predict a future action chunk conditioned on multi-view images
    and a natural-language instruction (with optional discretised state prefix).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts).
        self.config = merge_framework_config(QwenPI_v4DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors
        # (embedding output + every layer's output), and we keep the last
        # `num_hidden_layers` of them for layer-wise cross-attn — so the DiT
        # depth and project_layers count must match `num_hidden_layers` exactly.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts it
        # on the top-level config.  getattr(..., vlm_hf_cfg) handles both cases.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Resolve the Action DiT hidden dim BEFORE building the action head,
        # so that LayerwiseFlowmatchingActionHead constructs DiT at the right size.
        # If the user did not specify it, fall back to the LLM hidden size
        # (i.e. behave like QwenPI: project_layers becomes nn.Identity()).
        #
        # NOTE: `action_dit_hidden_dim` is a framework-side hint only — it is
        # NOT a DiT constructor kwarg, so we keep it out of diffusion_model_cfg
        # and instead pass it through `populate_layerwise_dit_cfg`, which writes
        # the canonical DiT-shape fields (input_embedding_dim, cross_attention_dim,
        # num_attention_heads).
        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        # Push the resolved DiT shape into diffusion_model_cfg.  The action head
        # is intentionally agnostic of qwenvl.* — it only consumes this dict.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        # Layer-wise projector: map each selected VL hidden to Action DiT hidden space.
        # This explicitly decouples VL representation size from action DiT latent size.
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project layer-wise VL hidden states to the hidden space expected by Action DiT."""
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  # List[List[PIL.Image]], length B
        instructions = [example["lang"] for example in examples]  # List[str], length B
        actions = [example["action"] for example in examples]  # List[ndarray (T, action_dim)]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        # Build full augmented instructions: robot meta + state tokens in one pass.
        instructions = self._add_robo_meta_tokens_to_instructions(examples, instructions, state)
        state = None  # state is now encoded in the instruction tokens

        # Step 1: build QwenVL-compatible inputs
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        # do not pollute the action prediction.
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # Take the last N hidden states that match the number of DiT layers
            # and project them to the Action DiT hidden dimension.
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = self.num_action_dit_layers
            vl_embs_list = list(all_hidden[-expected_layers:])
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
            base_hidden = vl_embs_list[-1]

        # Step 2: compute flow-matching loss over the action chunk
        with torch.autocast("cuda", dtype=torch.float32):
            # Align labels: keep only the last action_horizon timesteps.
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            repeated_diffusion_steps = 2  # No repeat for the large action FM to save memory.
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat every VLM layer embedding to match the duplicated action batch.
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            # Match the duplicated batch on the attention mask too.
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Run inference and return a denoised action trajectory.

        Steps:
          1. Optionally resize images to the training observation resolution.
          2. Encode images + instruction (with discretised state prefix) through QwenVL.
          3. Project layer-wise VLM hidden states to the Action DiT latent space.
          4. Run the flow-matching sampler to produce the action chunk.

        Args:
            examples: List[dict], each entry requires:
                - image: List[PIL.Image] (multi-view)
                - lang:  str instruction
                - state: np.ndarray shaped (1, state_dim), optional

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape (B, action_horizon, action_dim),
                    denoised actions in the normalised action space.
        """

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # List[List[PIL.Image]]
        instructions = [example["lang"] for example in examples]  # List[str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # List[ndarray] or None

        # Build full augmented instructions: robot meta + state tokens in one pass.
        instructions = self._add_robo_meta_tokens_to_instructions(examples, instructions, state)
        state = None  # state is now encoded in the instruction tokens

        # Optionally resize images to the resolution used during training.
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: build QwenVL-compatible inputs
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = self.num_action_dit_layers
            vl_embs_list = list(all_hidden[-expected_layers:])
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
            base_hidden = vl_embs_list[-1]

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 2: run the flow-matching sampler to produce the denoised action chunk.
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state, encoder_attention_mask=backbone_attention_mask
            )  # (B, action_horizon, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def state2str_transform(self, state: np.ndarray) -> str:
        """Quantise a state vector into 256 uniform bins and return it as a space-separated token string.

        Follows the π₀.5 convention: bins span [-1, 1] uniformly.
        Example: [-0.5, 0.1, 0.8] -> "95 133 203"
        """
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def add_discretized_state_to_instruction(self, instructions: List[str], states: List[np.ndarray]) -> List[str]:
        """Append discretised proprioceptive state tokens to each instruction.

        Format: ``<original instruction> [STATE] <bin indices> [ACTION]``
        This lets the VLM attend to the robot state purely through its
        existing text-token pathway — no extra encoder required.
        """
        updated_instructions = []
        for instr, state in zip(instructions, states):
            state_str = self.state2str_transform(state[0])
            updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
        return updated_instructions

    def _resolve_robot_layout(self, example: dict) -> Dict[str, Any]:
        """Look up the ROBOT_LAYOUTS entry for this sample.

        ``example["robot_tag"]`` is the EmbodimentTag value (e.g. "ARX5", "UR5",
        "DOS_W1", "ALOHA") set by LeRobotSingleDataset._pack_sample.

        Lookup order:
        1. Direct key match (in case robot_tag is already a ROBOT_LAYOUTS key).
        2. EMBODIMENT_TAG_TO_LAYOUT_KEY mapping (normal path for RoboChallenge).
        3. None → instruction left unchanged (graceful degradation).
        """
        robot_tag = example.get("robot_tag")
        if robot_tag is None:
            return None

        # Fast path: robot_tag is already a ROBOT_LAYOUTS key
        if robot_tag in ROBOT_LAYOUTS:
            return ROBOT_LAYOUTS[robot_tag]

        # Normal path: map EmbodimentTag value → layout key
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
        """Build the full augmented instruction string for each sample in one pass.

        Template (robot found in ROBOT_LAYOUTS, state available):
            "Original_instruction: {instr}.
             The robot is {tag} with {arm_type}. The control frequency is {fps} Hz.
             The current robot state is {state_str}.
             Please predict the next {chunk_size} control actions to execute
             the original_instruction."

        Template (robot found, no state):
            Same as above but the state sentence is omitted.

        Template (robot NOT in ROBOT_LAYOUTS — graceful degradation):
            "{instr} [STATE] {state_str} [ACTION]"   (legacy format, state present)
            "{instr}"                                 (state absent)
        """

        # @Jinhui TODO 如果感觉文字太长，可以直接简化模版，节省 token budget for the instruction itself。
        enhanced = []
        for i, (example, instr) in enumerate(zip(examples, instructions)):
            layout = self._resolve_robot_layout(example)
            state_str = self.state2str_transform(states[i][0]) if states is not None else None

            if layout is None:
                # Graceful degradation for embodiments not in ROBOT_LAYOUTS.
                if state_str is not None:
                    enhanced.append(f"{instr} [STATE] {state_str} [ACTION]")
                else:
                    enhanced.append(instr)
            else:
                state_part = (
                    f" The current robot state is {state_str}." if state_str is not None else ""
                )
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

    import debugpy
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    import os

    if os.getenv("DEBUG_MODE", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    args.config_yaml = "examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

    model = QwenPI_v4(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)

    def print_model_size(m: nn.Module, depth: int = 1):
        """Print parameter counts for each top-level submodule (depth=1)."""
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{'='*55}")
        print(f"{'Module':<35} {'Params':>12}  {'%':>6}")
        print(f"{'-'*55}")
        for name, child in m.named_children():
            n = sum(p.numel() for p in child.parameters())
            print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
        print(f"{'-'*55}")
        print(f"  {'TOTAL':<33} {total:>12,}  100.0%")
        print(f"{'='*55}\n")

    print_model_size(model, depth=1)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(50, 16)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }

    batch = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # # # Advance: try forward model with dataloader
    # # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.include_state = True

    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # #
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch)
