"""Policy bridge for RoboChallenge_table30v2 evaluation.

Wraps a trained ``baseframework`` checkpoint and adapts it to the I/O contract
of ``https://github.com/RoboChallenge/RoboChallengeInference`` (cvpr branch):

State input from the RC server (pickled dict)::

    {
        "images":  {"<cam_name>": <PNG bytes>, ...},
        "action":  [...],            # current robot state in the requested action_type
        "pending_actions": int,
        "timestamp": float,
        "state":   "normal" | "abnormal" | "size_none",
    }

The wrapper reads camera PNGs + state, runs the policy, and returns an action
chunk ready to be POSTed via ``InterfaceClient.post_actions``.

Single-arm (UR5/ARX5) returns 8-d actions; dual-arm DOSW1 returns 14-d. The
DOSW1 stack also needs (a) q99 normalization (matching training) and (b) a
permutation between the parquet "raw" layout and the starvla dataloader's
internal layout (joints first, then grippers). See ``state_layout`` below.

Robot-specific shapes are read from a small registry below — consistent with
``examples/RoboChallenge_table30v2/train_files/data_registry/data_config.py``.
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
# Layout permutations (kept for downstream callers; not needed by DOSW1 here)
# ---------------------------------------------------------------------------
# Some external pipelines (e.g. an upstream Table30v2 ref script the user has)
# concatenate gr00t sub-keys grouped by type, putting all joints first and all
# grippers last. This repo's data_config.py keeps the parquet's raw concat
# order ([L_j×6, L_grip, R_j×6, R_grip]) — confirmed by inspection of
# dataset_statistics.json (narrowest q99-q01 band at indices 6 and 13). So
# DOSW1 here does NOT need a permutation. The constants below are exposed for
# callers that bring stats in starvla-rearranged order.
RAW_TO_STARVLA = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13], dtype=np.int64)
STARVLA_TO_RAW = np.array([0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.int64)


# ---------------------------------------------------------------------------
# Robot registry — keep in sync with train_files/data_registry/data_config.py
# ---------------------------------------------------------------------------

@dataclass
class RobotSpec:
    robot_tag: str                    # adapter nickname: "ur5" | "arx5" | "dosw1" | "aloha"
    image_types: List[str]            # cameras requested from /state.pkl in order
    state_action_type: str            # action_type used to fetch the *state* (we want joint+gripper)
    post_action_type: str             # action_type used to *post* actions
    state_dim: int                    # policy state input dim
    action_dim: int                   # policy action output dim
    norm_unnorm_key: str              # key inside dataset_statistics.json (e.g. "new_embodiment", "dos-w1")
    norm_mode: str = "min_max"        # "min_max" | "q99" — must match training transform
    state_layout: str = "model"       # "model" (no perm) | "raw_dosw1" (legacy: raw↔starvla perm)


# Keep entries sorted by introduction date for diff-friendliness.
ROBOT_SPECS: Dict[str, RobotSpec] = {
    "ur5": RobotSpec(
        robot_tag="ur5",
        image_types=["cam_global", "cam_arm"],
        state_action_type="leftjoint",   # state["action"] = joint(6)+gripper(1) = 7
        post_action_type="leftpos",      # outgoing actions = ee_pose(7 quat)+gripper(1) = 8
        state_dim=7,
        action_dim=8,
        norm_unnorm_key="new_embodiment",
    ),
    "arx5": RobotSpec(
        robot_tag="arx5",
        image_types=["cam_global", "cam_arm", "cam_side"],
        state_action_type="leftjoint",
        post_action_type="leftpos",
        state_dim=7,
        action_dim=8,
        norm_unnorm_key="new_embodiment",
    ),
    # DOSW1 is dual-arm 14-d. Verified against upstream cvpr branch:
    # MockRCRobotW1.ACTION_TYPES = ("joint","pos","leftjoint","leftpos","rightjoint","rightpos");
    # the both-arm joint literal is "joint" (not "bothjoint"). Server concatenates
    # left_get_joint() + right_get_joint() → [L_j×6, L_grip, R_j×6, R_grip] which
    # matches data_config.py's raw state/action key order, so no permutation is needed.
    "dosw1": RobotSpec(
        robot_tag="dosw1",
        image_types=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        state_action_type="joint",
        post_action_type="joint",
        state_dim=14,
        action_dim=14,
        # Matches EmbodimentTag.DOS_W1.value used by data_config.py for new
        # training runs. Older checkpoints with the legacy "dosw1" stats key
        # still load via _load_norm_stats's single-key fallback.
        norm_unnorm_key="dos-w1",
        norm_mode="q99",
        # state_layout left at default "model": this repo's data_config.py
        # preserves raw [L_j×6, L_grip, R_j×6, R_grip] order in stats.
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_norm_stats(checkpoint_path: str | Path) -> dict:
    """Load ``dataset_statistics.json`` from the run dir of *checkpoint_path*.

    Layout::

        <run_dir>/dataset_statistics.json
        <run_dir>/checkpoints/steps_*_pytorch_model.pt
    """
    ckpt = Path(checkpoint_path)
    run_dir = ckpt.parents[1] if ckpt.parents[1].joinpath("dataset_statistics.json").exists() else ckpt.parent
    stats_json = run_dir / "dataset_statistics.json"
    if not stats_json.exists():
        raise FileNotFoundError(f"Missing dataset_statistics.json beside {ckpt} (looked in {run_dir})")
    with stats_json.open() as f:
        return json.load(f)


def _decode_png(buf: bytes) -> np.ndarray:
    """Decode PNG bytes returned by RC ``/state.pkl`` into an HxWx3 RGB ndarray."""
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    return np.asarray(img)


def _normalize_min_max(state: np.ndarray, stats: dict) -> np.ndarray:
    """min-max normalization: ``y = 2*(x - min)/(max - min) - 1`` with passthrough where min==max."""
    s_min = np.asarray(stats["min"], dtype=np.float32)
    s_max = np.asarray(stats["max"], dtype=np.float32)
    out = state.astype(np.float32).copy()
    mask = s_max != s_min
    out[..., mask] = 2.0 * (out[..., mask] - s_min[mask]) / (s_max[mask] - s_min[mask]) - 1.0
    return out


def _unnormalize_min_max(norm_action: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of min-max for actions, respecting the ``mask`` field."""
    a_min = np.asarray(stats["min"], dtype=np.float32)
    a_max = np.asarray(stats["max"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", [True] * len(a_min)), dtype=bool)
    norm = np.clip(norm_action.astype(np.float32), -1.0, 1.0)
    return np.where(mask, (norm + 1.0) / 2.0 * (a_max - a_min) + a_min, norm_action.astype(np.float32))


def _normalize_q99(x: np.ndarray, stats: dict) -> np.ndarray:
    """q99 normalization: ``y = clip(2*(x - q01)/(q99 - q01) - 1, -1, 1)``.

    Mirrors gr00t's StateActionTransform with ``normalization_modes={k:"q99"}``;
    used during training of the dosw1-q99 run.
    """
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    rng = q99 - q01
    out = x.astype(np.float32).copy()
    mask = rng != 0
    out[..., mask] = 2.0 * (out[..., mask] - q01[mask]) / rng[mask] - 1.0
    return np.clip(out, -1.0, 1.0)


def _unnormalize_q99(norm_action: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of q99, applied uniformly across dims.

    Mirrors gr00t's training-time ``StateActionTransform.inverse`` for mode='q99'
    (see starVLA/dataloader/gr00t_lerobot/transform/state_action.py:198-201),
    which does *not* consult ``stats["mask"]`` — the gripper q01/q99 range is
    nonzero (≈ 0..0.07), so it was normalized at training and must be inverted
    here. Any ``stats["mask"]`` field present is intentionally ignored.
    """
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    norm = np.clip(norm_action.astype(np.float32), -1.0, 1.0)
    return (norm + 1.0) / 2.0 * (q99 - q01) + q01


_NORM_FNS = {
    "min_max": (_normalize_min_max, _unnormalize_min_max),
    "q99": (_normalize_q99, _unnormalize_q99),
}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RoboChallengePolicy:
    """Concrete ``DummyPolicy`` replacement compatible with upstream demo.py / test.py.

    Owns the model directly (no websocket) — keeps the hop count low for the
    self-test and for the production demo loop where latency matters.

    Usage (HTTP / mock-server style)::

        policy = RoboChallengePolicy(checkpoint_path, robot_tag="ur5")
        actions = policy.run_policy(state_dict, prompt="shred the paper")
        # actions: list[list[float]] with shape (n_action_steps, 8) for UR5.

    Usage (local closed-loop, skip PNG roundtrip)::

        actions = policy.predict_from_pil(pil_images, raw_state, prompt=...)
        # actions: np.ndarray, shape (n_action_steps, action_dim) in raw layout.
    """

    def __init__(
        self,
        checkpoint_path: str,
        robot_tag: str = "ur5",
        n_action_steps: int = 8,
        image_size: Sequence[int] = (224, 224),
        device: str = "cuda",
        use_bf16: bool = True,
    ) -> None:
        if robot_tag not in ROBOT_SPECS:
            raise ValueError(f"Unsupported robot_tag={robot_tag}; options={list(ROBOT_SPECS)}")
        self.spec = ROBOT_SPECS[robot_tag]
        if self.spec.norm_mode not in _NORM_FNS:
            raise ValueError(f"Unsupported norm_mode={self.spec.norm_mode!r}; options={list(_NORM_FNS)}")
        self._normalize, self._unnormalize = _NORM_FNS[self.spec.norm_mode]

        self.checkpoint_path = checkpoint_path
        self.n_action_steps = int(n_action_steps)
        self.image_size = tuple(image_size)
        self.device = torch.device(device)

        logger.info("[RC] Loading framework from %s", checkpoint_path)
        self.model = baseframework.from_pretrained(checkpoint_path)
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device).eval()

        norm_stats_full = _load_norm_stats(checkpoint_path)
        if self.spec.norm_unnorm_key not in norm_stats_full:
            available = list(norm_stats_full.keys())
            if len(available) == 1:
                self.spec.norm_unnorm_key = available[0]
                logger.warning("[RC] unnorm_key fallback to %s (only one available)", available[0])
            else:
                raise KeyError(f"unnorm_key {self.spec.norm_unnorm_key} not in {available}")
        self.state_stats = norm_stats_full[self.spec.norm_unnorm_key]["state"]
        self.action_stats = norm_stats_full[self.spec.norm_unnorm_key]["action"]

        # Cache last prompt for logging.
        self._last_prompt: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API expected by upstream GPUClient
    # ------------------------------------------------------------------

    def run_policy(self, input_data: dict, prompt: Optional[str] = None) -> List[List[float]]:
        """Single-call inference compatible with the upstream RC server payload.

        Args:
            input_data: The unpickled response of ``GET /state.pkl`` — see module docstring.
            prompt: Free-form task instruction.

        Returns:
            list[list[float]]: ``n_action_steps`` actions, each of length ``action_dim``,
            in the robot's *raw* layout (matching what the server expects).
        """
        # ---- 1. Decode PNG → PIL ----------------------------------------
        images_dict = input_data.get("images") or {}
        pil_images: List[Image.Image] = []
        for cam in self.spec.image_types:
            if cam not in images_dict:
                raise KeyError(f"[RC] Missing camera {cam!r} in state.images keys={list(images_dict)}")
            arr = _decode_png(images_dict[cam])
            pil_images.append(Image.fromarray(arr))

        raw_state = np.asarray(input_data.get("action") or [], dtype=np.float32)

        actions = self.predict_from_pil(pil_images, raw_state, prompt=prompt)
        return actions.astype(np.float32).tolist()

    def predict_from_pil(
        self,
        pil_images: Sequence[Image.Image],
        raw_state: np.ndarray,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        """Direct inference entry, skipping PNG (en|de)code.

        Args:
            pil_images: One PIL.Image per camera, in ``self.spec.image_types`` order.
                Resized to ``self.image_size`` automatically.
            raw_state: 1-D array of length ``self.spec.state_dim`` in the robot's
                *raw* layout (parquet / RC server order).
            prompt: Task instruction.

        Returns:
            np.ndarray of shape ``(n_action_steps, action_dim)`` in *raw* layout,
            un-normalized.
        """
        if prompt is not None and prompt != self._last_prompt:
            logger.info("[RC] Prompt: %r", prompt)
            self._last_prompt = prompt
        instruction = prompt if prompt else (self._last_prompt or "perform the task")

        # ---- 1. Resize PIL to configured image size ---------------------
        if len(pil_images) != len(self.spec.image_types):
            raise ValueError(
                f"[RC] expected {len(self.spec.image_types)} cam images "
                f"(order: {self.spec.image_types}), got {len(pil_images)}"
            )
        resized: List[Image.Image] = []
        for img in pil_images:
            if (img.height, img.width) != self.image_size:
                img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            resized.append(img)

        # ---- 2. State: raw → (perm) → normalize -------------------------
        raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        if raw_state.size != self.spec.state_dim:
            raise ValueError(
                f"[RC] state vector length {raw_state.size} != expected {self.spec.state_dim} "
                f"(make sure state_action_type={self.spec.state_action_type!r} is correct)"
            )
        if self.spec.state_layout == "raw_dosw1":
            state_model_layout = raw_state[RAW_TO_STARVLA]
        else:
            state_model_layout = raw_state
        norm_state = self._normalize(state_model_layout, self.state_stats)  # (D,)
        state_input = norm_state[None, :]  # (1, D)

        # ---- 3. Forward -------------------------------------------------
        sample = {
            "image": list(resized),
            "lang": instruction,
            "state": state_input,
        }
        out = self.model.predict_action([sample])
        normalized_actions = np.asarray(out["normalized_actions"])  # (1, T, D)

        # ---- 4. Unnormalize, slice horizon, undo perm -------------------
        actions_model = self._unnormalize(normalized_actions[0], self.action_stats)  # (T, D)
        actions_model = actions_model[: self.n_action_steps]
        if self.spec.state_layout == "raw_dosw1":
            actions_raw = actions_model[:, STARVLA_TO_RAW]
        else:
            actions_raw = actions_model
        return actions_raw.astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience accessors used by the launcher scripts
    # ------------------------------------------------------------------

    @property
    def image_type(self) -> List[str]:
        return list(self.spec.image_types)

    @property
    def state_action_type(self) -> str:
        return self.spec.state_action_type

    @property
    def post_action_type(self) -> str:
        return self.spec.post_action_type
