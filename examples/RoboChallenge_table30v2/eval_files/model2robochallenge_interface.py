"""Policy bridge for RoboChallenge Table30v2 evaluation.

Wraps a trained ``baseframework`` checkpoint and adapts it to the I/O contract
of ``https://github.com/RoboChallenge/RoboChallengeInference`` (cvpr branch).

Per-robot shapes / cameras / action_types live in ``ROBOT_SPECS`` and must
stay in sync with ``train_files/data_registry/data_config.py``.

State payload from the RC server (unpickled GET /state.pkl)::

    {
        "images":  {"<cam_name>": <PNG bytes>, ...},
        "action":  [...],            # current robot state in the requested action_type
        "pending_actions": int,
        "timestamp": float,
        "state":   "normal" | "abnormal" | "size_none",
    }
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robot registry — keep in sync with train_files/data_registry/data_config.py
# ---------------------------------------------------------------------------

@dataclass
class RobotSpec:
    robot_tag: str                    # adapter nickname: "ur5" | "arx5" | "dosw1" | "aloha"
    image_types: List[str]            # cameras requested from /state.pkl, in order
    state_action_type: str            # action_type for GET /state.pkl
    post_action_type: str             # action_type for POST /action
    state_dim: int                    # policy state input dim
    action_dim: int                   # policy action output dim
    norm_unnorm_key: str              # = EmbodimentTag.value, e.g. "table30v2_dosw1"
    norm_mode: str                    # required: must match the StateActionTransform
                                      # mode used in data_config.py.transform() —
                                      # one of _NORM_FNS keys.  No default to force
                                      # explicit declaration per robot.


ROBOT_SPECS: Dict[str, RobotSpec] = {
    "ur5": RobotSpec(
        robot_tag="ur5",
        image_types=["cam_global", "cam_arm"],
        state_action_type="leftjoint",   # state["action"] = joint(6)+gripper(1) = 7d
        post_action_type="leftpos",      # outgoing actions = ee_pose(7 quat)+gripper(1) = 8d
        state_dim=7,
        action_dim=8,
        norm_unnorm_key="table30v2_ur5",
        norm_mode="q99",
    ),
    "arx5": RobotSpec(
        robot_tag="arx5",
        image_types=["cam_global", "cam_arm", "cam_side"],
        state_action_type="leftjoint",
        post_action_type="leftpos",
        state_dim=7,
        action_dim=8,
        norm_unnorm_key="table30v2_arx5",
        norm_mode="q99",
    ),
    # DOSW1 is dual-arm 14d.  The RC server returns state in
    # [L_joints×6, L_grip, R_joints×6, R_grip] order (verified against upstream
    # MockRCRobotW1: left_get_joint() + right_get_joint()), which matches
    # data_config.py's state_keys / action_keys order — no permutation needed.
    "dosw1": RobotSpec(
        robot_tag="dosw1",
        image_types=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        state_action_type="joint",
        post_action_type="joint",
        state_dim=14,
        action_dim=14,
        norm_unnorm_key="table30v2_dosw1",
        norm_mode="q99",
    ),
    # ALOHA is dual-arm too, but state and action use different action_types:
    # state is 14d joint (L.joint+L.grip+R.joint+R.grip), actions are 16d ee
    # (L.ee(7-quat)+L.grip+R.ee(7-quat)+R.grip).  job_loop only takes one
    # action_type — run_demo.py routes this through DualActionTypeGPUClient.
    "aloha": RobotSpec(
        robot_tag="aloha",
        image_types=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        state_action_type="joint",
        post_action_type="pos",
        state_dim=14,
        action_dim=16,
        norm_unnorm_key="table30v2_aloha",
        norm_mode="q99",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_run_dir(checkpoint_path: str | Path) -> Path:
    ckpt = Path(checkpoint_path).resolve()
    for p in (ckpt.parent, *ckpt.parents):
        if (p / "dataset_statistics.json").exists():
            return p
    raise FileNotFoundError(f"No dataset_statistics.json found walking up from {ckpt}")


def _load_norm_stats(checkpoint_path: str | Path) -> dict:
    run_dir = _resolve_run_dir(checkpoint_path)
    with (run_dir / "dataset_statistics.json").open() as f:
        return json.load(f)


def _read_yaml_field(checkpoint_path: str | Path, dotted_key: str, default):
    run_dir = _resolve_run_dir(checkpoint_path)
    leaf = dotted_key.rsplit(".", 1)[-1]
    for name in ("config.yaml", "config.full.yaml"):
        cfg_path = run_dir / name
        if not cfg_path.exists():
            continue
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(str(cfg_path))
            v = OmegaConf.select(cfg, dotted_key, default=None)
        except Exception:
            v = None
            for line in cfg_path.read_text().splitlines():
                s = line.strip()
                if s.startswith(f"{leaf}:"):
                    v = s.split(":", 1)[1].strip().strip("\"'")
                    break
        if v is not None and v != "":
            return v
    return default


def _load_action_mode_from_run_dir(checkpoint_path: str | Path) -> str:
    mode = _read_yaml_field(checkpoint_path, "datasets.vla_data.action_mode", "abs")
    logger.info("[RC] action_mode=%r", mode)
    return str(mode)


def _load_include_state_from_run_dir(checkpoint_path: str | Path) -> bool:
    raw = _read_yaml_field(checkpoint_path, "datasets.vla_data.include_state", False)
    val = (str(raw).lower() == "true") if isinstance(raw, str) else bool(raw)
    logger.info("[RC] include_state=%s", val)
    return val


def _delta_to_absolute(pred_delta: np.ndarray, state: np.ndarray) -> np.ndarray:
    if pred_delta.shape[1] != state.shape[0]:
        raise ValueError(
            f"delta inverse needs action_dim == state_dim (got "
            f"{pred_delta.shape[1]} vs {state.shape[0]})."
        )
    out = np.empty_like(pred_delta, dtype=np.float32)
    out[0] = pred_delta[0] + state
    for t in range(1, len(pred_delta)):
        out[t] = pred_delta[t] + out[t - 1]
    return out


def _rel_to_absolute(pred_rel: np.ndarray, state: np.ndarray) -> np.ndarray:
    if pred_rel.shape[1] != state.shape[0]:
        raise ValueError(
            f"rel inverse needs action_dim == state_dim (got "
            f"{pred_rel.shape[1]} vs {state.shape[0]})."
        )
    return pred_rel.astype(np.float32) + state.astype(np.float32)[None, :]


def _decode_png(buf: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    return np.asarray(img)


def _normalize_min_max(x: np.ndarray, stats: dict) -> np.ndarray:
    """``y = 2*(x - min)/(max - min) - 1``, passthrough where min == max."""
    s_min = np.asarray(stats["min"], dtype=np.float32)
    s_max = np.asarray(stats["max"], dtype=np.float32)
    out = x.astype(np.float32).copy()
    mask = s_max != s_min
    out[..., mask] = 2.0 * (out[..., mask] - s_min[mask]) / (s_max[mask] - s_min[mask]) - 1.0
    return out


def _unnormalize_min_max(norm_action: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of min_max for actions, respecting the optional ``mask`` field."""
    a_min = np.asarray(stats["min"], dtype=np.float32)
    a_max = np.asarray(stats["max"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", [True] * len(a_min)), dtype=bool)
    norm = np.clip(norm_action.astype(np.float32), -1.0, 1.0)
    return np.where(mask, (norm + 1.0) / 2.0 * (a_max - a_min) + a_min, norm_action.astype(np.float32))


def _normalize_q99(x: np.ndarray, stats: dict) -> np.ndarray:
    """``y = clip(2*(x - q01)/(q99 - q01) - 1, -1, 1)``.

    Mirrors gr00t StateActionTransform with ``normalization_modes={k: "q99"}``.
    """
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    rng = q99 - q01
    out = x.astype(np.float32).copy()
    mask = rng != 0
    out[..., mask] = 2.0 * (out[..., mask] - q01[mask]) / rng[mask] - 1.0
    return np.clip(out, -1.0, 1.0)


def _unnormalize_q99(norm_action: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of q99, applied uniformly across dims.  Does NOT consult
    ``stats["mask"]`` — gr00t's training-side inverse doesn't either."""
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    norm = np.clip(norm_action.astype(np.float32), -1.0, 1.0)
    return (norm + 1.0) / 2.0 * (q99 - q01) + q01


def _normalize_mean_std(x: np.ndarray, stats: dict) -> np.ndarray:
    """``y = (x - mean) / std``, passthrough where std == 0."""
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    out = x.astype(np.float32).copy()
    mask = std != 0
    out[..., mask] = (out[..., mask] - mean[mask]) / std[mask]
    return out


def _unnormalize_mean_std(x: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of mean_std."""
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    out = x.astype(np.float32).copy()
    mask = std != 0
    out[..., mask] = out[..., mask] * std[mask] + mean[mask]
    return out


_NORM_FNS = {
    "min_max":  (_normalize_min_max,  _unnormalize_min_max),
    "q99":      (_normalize_q99,      _unnormalize_q99),
    "mean_std": (_normalize_mean_std, _unnormalize_mean_std),
}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RoboChallengePolicy:
    """``DummyPolicy`` replacement for upstream demo.py / test.py.

    Owns the model directly (no websocket).  Two entry points::

        # HTTP / mock-server style — feed the unpickled GET /state.pkl dict:
        policy = RoboChallengePolicy(checkpoint_path, robot_tag="dosw1")
        actions = policy.run_policy(state_dict, prompt="...")

        # Local closed-loop — skip PNG roundtrip:
        actions = policy.predict_from_pil(pil_images, raw_state, prompt="...")
    """

    def __init__(
        self,
        checkpoint_path: str,
        robot_tag: str = "dosw1",
        n_action_steps: int = 50,
        image_size: Sequence[int] = (224, 224),
        device: str = "cuda",
        use_bf16: bool = True,
        action_mode: Optional[str] = None,
    ) -> None:
        if robot_tag not in ROBOT_SPECS:
            raise ValueError(f"Unsupported robot_tag={robot_tag}; options={list(ROBOT_SPECS)}")
        self.spec = ROBOT_SPECS[robot_tag]
        if self.spec.norm_mode not in _NORM_FNS:
            raise ValueError(f"Unsupported norm_mode={self.spec.norm_mode!r}; options={list(_NORM_FNS)}")
        self._normalize, self._unnormalize = _NORM_FNS[self.spec.norm_mode]

        self.action_mode = action_mode or _load_action_mode_from_run_dir(checkpoint_path)
        if self.action_mode not in ("abs", "rel", "delta"):
            raise ValueError(f"Unsupported action_mode={self.action_mode!r}; options=abs/rel/delta")
        self.include_state = _load_include_state_from_run_dir(checkpoint_path)

        self.checkpoint_path = checkpoint_path
        self.n_action_steps = int(n_action_steps)
        self.image_size = tuple(image_size)
        self.device = torch.device(device)

        logger.info("[RC] Loading framework from %s (robot=%s, action_mode=%s, include_state=%s)",
                    checkpoint_path, robot_tag, self.action_mode, self.include_state)
        self.model = baseframework.from_pretrained(checkpoint_path)
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device).eval()

        norm_stats_full = _load_norm_stats(checkpoint_path)
        if self.spec.norm_unnorm_key not in norm_stats_full:
            raise KeyError(
                f"unnorm_key {self.spec.norm_unnorm_key!r} not in dataset_statistics.json "
                f"(have {list(norm_stats_full)})"
            )
        self.state_stats = norm_stats_full[self.spec.norm_unnorm_key]["state"]
        self.action_stats = norm_stats_full[self.spec.norm_unnorm_key]["action"]

        self._last_prompt: Optional[str] = None

    # --- Public API expected by upstream GPUClient --------------------------

    def run_policy(self, input_data: dict, prompt: Optional[str] = None) -> List[List[float]]:
        """Single-call inference compatible with the RC server payload.

        Returns ``n_action_steps`` actions × ``action_dim``, JSON-serialisable
        for ``InterfaceClient.post_actions``.
        """
        images_dict = input_data.get("images") or {}
        pil_images: List[Image.Image] = []
        for cam in self.spec.image_types:
            if cam not in images_dict:
                raise KeyError(f"[RC] Missing camera {cam!r} in state.images keys={list(images_dict)}")
            pil_images.append(Image.fromarray(_decode_png(images_dict[cam])))

        raw_state = np.asarray(input_data.get("action") or [], dtype=np.float32)
        actions = self.predict_from_pil(pil_images, raw_state, prompt=prompt)
        return actions.astype(np.float32).tolist()

    def predict_from_pil(
        self,
        pil_images: Sequence[Image.Image],
        raw_state: np.ndarray,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        """Direct inference without PNG (en|de)code.  Returns ``(n_action_steps, action_dim)``."""
        if prompt is not None and prompt != self._last_prompt:
            logger.info("[RC] Prompt: %r", prompt)
            self._last_prompt = prompt
        instruction = prompt if prompt else (self._last_prompt or "perform the task")

        if len(pil_images) != len(self.spec.image_types):
            raise ValueError(
                f"[RC] expected {len(self.spec.image_types)} cam images "
                f"(order: {self.spec.image_types}), got {len(pil_images)}"
            )
        resized = [
            img if (img.height, img.width) == self.image_size
            else img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            for img in pil_images
        ]

        raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if raw_state.size != self.spec.state_dim:
            raise ValueError(
                f"[RC] state vector length {raw_state.size} != expected {self.spec.state_dim} "
                f"(check state_action_type={self.spec.state_action_type!r})"
            )
        norm_state = self._normalize(raw_state, self.state_stats)

        sample = {
            "image": list(resized),
            "lang": instruction,
        }
        if self.include_state:
            sample["state"] = norm_state[None, :]
        out = self.model.predict_action([sample])
        normalized_actions = np.asarray(out["normalized_actions"])

        actions = self._unnormalize(normalized_actions[0], self.action_stats)
        if self.action_mode == "rel":
            actions = _rel_to_absolute(actions, raw_state)
        elif self.action_mode == "delta":
            actions = _delta_to_absolute(actions, raw_state)
        return actions[: self.n_action_steps].astype(np.float32)

    # --- Convenience accessors used by the launcher scripts -----------------

    @property
    def image_type(self) -> List[str]:
        return list(self.spec.image_types)

    @property
    def state_action_type(self) -> str:
        return self.spec.state_action_type

    @property
    def post_action_type(self) -> str:
        return self.spec.post_action_type
