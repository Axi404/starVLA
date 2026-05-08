# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by Jinhui YE / HKUST in 2026.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """actions: (B, T, action_dim); timesteps: (B,) → broadcast to (B, T)."""
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = swish(self.layer2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.layer3(x)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """actions: (B, T, action_dim); timesteps: (B,); cat_ids: (B,) → (B, T, hidden_size)."""
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1), cat_ids))
        return self.W3(x, cat_ids)


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(default=True)
    diffusion_model_cfg: dict = field(default=None)
    input_embedding_dim: int = field(default=1536)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=None)
    max_num_embodiments: int = field(default=32)
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)
    load_pretrained_det_decode_layer_path: str = field(default=None)
    detection_coeff: float = field(default=1.0)
    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# Fallback DiT shape used only when the framework forgot to populate
# diffusion_model_cfg via populate_layerwise_dit_cfg.
DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}


class LayerwiseFlowmatchingActionHead(nn.Module):
    """Layer-wise cross-attention DiT action head.

    Decoupled from any specific VLM backbone — only reads from
    ``global_config.framework.action_model`` (and its ``diffusion_model_cfg``
    sub-tree).  The framework is responsible for populating
    ``diffusion_model_cfg`` with the DiT shape via
    ``populate_layerwise_dit_cfg`` BEFORE calling ``get_action_model``.
    """

    def __init__(self, global_config, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        for k, v in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v

        # Drop framework-side hint keys that are not DiT constructor kwargs.
        _DIT_NON_KWARGS = {"action_dit_hidden_dim"}
        diffusion_model_cfg_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in _DIT_NON_KWARGS}

        self.input_embedding_dim = diffusion_model_cfg_kwargs["input_embedding_dim"]
        self.model = DiT(**diffusion_model_cfg_kwargs)
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = (
            MLP(input_dim=action_config.state_dim, output_dim=self.input_embedding_dim)
            if action_config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return self.config.noise_s * (1 - sample)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _build_sa_embs(self, action_features, state_features, batch_size):
        """Concat (state? + future_tokens + action_features) along seq dim."""
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if state_features is not None:
            return torch.cat((state_features, future_tokens, action_features), dim=1)
        return torch.cat((future_tokens, action_features), dim=1)

    def _run_dit(self, sa_embs, vl_embs_list, encoder_attention_mask, temb):
        """Manual block-by-block iteration so each layer can read its matching VL embs.

        We must reproduce ``DiT.forward``'s gating here: when
        ``interleave_self_attention=True``, odd-indexed blocks were built with
        ``cross_attention_dim=None`` and must run pure self-attn — but
        diffusers' ``Attention`` does cross-attn whenever ``encoder_hidden_states``
        is passed, regardless of init-time config.  So odd blocks need
        ``encoder_hidden_states=None`` to actually self-attend.
        """
        interleave_self_attn = bool(getattr(self.model.config, "interleave_self_attention", False))
        out = sa_embs
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            is_self_attn = interleave_self_attn and (layer_idx % 2 == 1)
            out = layer(
                hidden_states=out,
                encoder_hidden_states=None if is_self_attn else vl_embs_list[layer_idx],
                encoder_attention_mask=None if is_self_attn else encoder_attention_mask,
                temb=temb,
            )
        return out

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ):
        """vl_embs_list: per-DiT-layer encoder hiddens, each (B, S, D).
        actions: (B, action_horizon, action_dim).
        encoder_attention_mask: optional (B, S) bool.
        """
        B = vl_embs_list[0].shape[0]

        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        sa_embs = self._build_sa_embs(action_features, state_features, B)
        temb = self.model.timestep_encoder(t_discretized)
        model_output = self._run_dit(sa_embs, vl_embs_list, encoder_attention_mask, temb)

        # TODO: a final pure-self-attn stack here would let action tokens refine
        # each other after layerwise cross-attn (currently only happens on odd
        # layers when interleave_self_attention=True).
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self.state_encoder(state) if state is not None else None

        for t in range(num_steps):
            t_discretized_int = int(t / float(num_steps) * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            action_features = self.action_encoder(actions, timesteps_tensor)
            sa_embs = self._build_sa_embs(action_features, state_features, batch_size)
            temb = self.model.timestep_encoder(timesteps_tensor)
            model_output = self._run_dit(sa_embs, vl_embs_list, encoder_attention_mask, temb)

            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """Build LayerwiseFlowmatchingActionHead from global framework config."""
    return LayerwiseFlowmatchingActionHead(global_config=config)


if __name__ == "__main__":
    pass
