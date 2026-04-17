"""
EBench Bridge
=============

In existing StarVLA integrations (RoboTwin, LIBERO, SimplerEnv) the benchmark
runs as a *client* and calls a StarVLA websocket policy server. EBench flips
this: EBench itself is an HTTP server, and models are expected to connect to
it via `genmanip_client.EvalClient`. StarVLA stays a websocket server.

Neither side can drive the other directly, so this script is the missing
driver — a pure client to both. Per step:

  EBench server  ──obs──▶  bridge  ──(image, lang, [state])──▶  StarVLA server
  StarVLA server ──normalized_actions──▶  bridge  ──action chunk──▶  EBench server

See examples/EBench/eval_files/README.md for usage.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2 as cv
import numpy as np
import yaml

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config


logger = logging.getLogger("ebench_bridge")


# --------------------------------------------------------------------------- #
# norm stats helpers
# --------------------------------------------------------------------------- #
def _load_action_norm_stats(policy_ckpt_path: str, unnorm_key: str) -> Dict[str, np.ndarray]:
    """Load the flat (action_dim,) min/max/mask from dataset_statistics.json.

    gr00t_lerobot concatenates action_keys in the order declared on the data
    config (see EBenchConfig) and saves a single combined action stats block
    under `{unnorm_key}.action`. We therefore expect a flat layout matching
    the 19-dim order: left_joints, right_joints, left_gripper, right_gripper,
    base_delta.
    """
    _, norm_stats = read_mode_config(Path(policy_ckpt_path))
    if unnorm_key not in norm_stats:
        raise KeyError(
            f"unnorm_key='{unnorm_key}' not in dataset_statistics.json. "
            f"Available: {sorted(norm_stats.keys())}"
        )
    block = norm_stats[unnorm_key]
    # Two possible shapes: {"action": {...}} or {"abs": {"action": {...}}}
    if "action" in block:
        stats = block["action"]
    elif "abs" in block and "action" in block["abs"]:
        stats = block["abs"]["action"]
    else:
        raise KeyError(
            f"No flat 'action' stats under '{unnorm_key}'. Got keys: {sorted(block.keys())}"
        )
    out = {
        "min": np.asarray(stats["min"], dtype=np.float32),
        "max": np.asarray(stats["max"], dtype=np.float32),
    }
    if "mask" in stats:
        out["mask"] = np.asarray(stats["mask"], dtype=bool)
    else:
        out["mask"] = np.ones_like(out["min"], dtype=bool)
    return out


def _unnormalize(
    normalized: np.ndarray,
    stats: Dict[str, np.ndarray],
    mode: str = "min_max",
) -> np.ndarray:
    """Flat-vector unnormalize. Dims with mask=False pass through unchanged
    (this is how `binary` gripper dims survive: the model already emits 0/1)."""
    if mode == "min_max":
        low, high = stats["min"], stats["max"]
    elif mode == "q99":
        if "q01" not in stats or "q99" not in stats:
            raise KeyError("q99 mode requires 'q01' and 'q99' in stats")
        low, high = stats["q01"], stats["q99"]
    else:
        raise ValueError(f"unsupported normalization mode: {mode}")
    clipped = np.clip(normalized, -1.0, 1.0)
    unnorm = 0.5 * (clipped + 1.0) * (high - low) + low
    return np.where(stats["mask"], unnorm, normalized)


def _get_action_chunk_size(policy_ckpt_path: str) -> int:
    cfg, _ = read_mode_config(Path(policy_ckpt_path))
    return cfg["framework"]["action_model"]["future_action_window_size"] + 1


# --------------------------------------------------------------------------- #
# bridge
# --------------------------------------------------------------------------- #
class EBenchBridge:
    """Forwards an EBench eval-server session through a StarVLA policy server.

    Usage:
        bridge = EBenchBridge(config_dict)
        bridge.run()
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.camera_keys: List[str] = list(cfg["camera_keys"])
        self.image_size = tuple(cfg.get("image_size", [224, 224]))

        self.layout: Dict[str, List[int]] = cfg["action_layout"]
        self.control_type: str = cfg["control_type"]
        self.is_rel: bool = cfg["is_rel"]
        self.base_is_rel: bool = cfg["base_is_rel"]

        self.chunk_mode: bool = cfg.get("chunk_mode", True)
        self.max_chunk_len: Optional[int] = cfg.get("max_chunk_len")
        self.log_every: int = cfg.get("log_every", 10)
        self.norm_mode: str = cfg.get("action_normalization_mode", "min_max")

        # --- StarVLA client ---
        self.policy = WebsocketClientPolicy(
            host=cfg["policy_host"], port=cfg["policy_port"]
        )
        ckpt_path = cfg["policy_ckpt_path"]
        self.action_norm_stats = _load_action_norm_stats(ckpt_path, cfg["unnorm_key"])
        self.model_chunk_len = _get_action_chunk_size(ckpt_path)
        logger.info(
            "connected to StarVLA %s:%s (chunk=%d, action_dim=%d)",
            cfg["policy_host"], cfg["policy_port"],
            self.model_chunk_len, len(self.action_norm_stats["min"]),
        )

        # --- EBench client ---
        from genmanip_client.eval_client import EvalClient  # lazy; only at eval time
        self.eval_client = EvalClient(
            base_url=cfg["ebench_base_url"],
            worker_ids=[cfg["ebench_worker_id"]],
            run_id=cfg["ebench_run_id"],
        )
        self.worker_id = cfg["ebench_worker_id"]
        logger.info(
            "connected to EBench %s (worker_id=%s, run_id=%s)",
            cfg["ebench_base_url"], self.worker_id, cfg["ebench_run_id"],
        )

        self._last_instruction: Optional[str] = None
        self._step_counter = 0

    # ------------------------------------------------------------------ #
    # obs → StarVLA example
    # ------------------------------------------------------------------ #
    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv.resize(img, self.image_size, interpolation=cv.INTER_AREA)

    def _obs_to_example(self, worker_obs: Dict[str, Any]) -> Dict[str, Any]:
        images = []
        for key in self.camera_keys:
            if key not in worker_obs:
                raise KeyError(
                    f"camera '{key}' missing from EBench obs. "
                    f"Available: {[k for k in worker_obs if k.startswith('video.')]}"
                )
            images.append(self._resize(np.asarray(worker_obs[key])))

        instruction = worker_obs.get("instruction", "")
        if instruction != self._last_instruction:
            logger.info("instruction: %r", instruction)
            self._last_instruction = instruction

        return {"lang": str(instruction), "image": images}

    # ------------------------------------------------------------------ #
    # StarVLA call
    # ------------------------------------------------------------------ #
    def _predict_chunk(self, example: Dict[str, Any]) -> np.ndarray:
        resp = self.policy.predict_action(
            {"examples": [example], "do_sample": False}
        )
        data = resp.get("data", resp)
        if "normalized_actions" not in data:
            raise KeyError(
                f"policy response missing 'normalized_actions'; keys={list(data.keys())}"
            )
        normalized = np.asarray(data["normalized_actions"])
        if normalized.ndim == 3:
            normalized = normalized[0]           # [B=1, T, D] -> [T, D]
        elif normalized.ndim == 1:
            normalized = normalized[None, :]     # [D] -> [1, D]
        return _unnormalize(normalized, self.action_norm_stats, self.norm_mode)

    # ------------------------------------------------------------------ #
    # action chunk → EBench action dict(s)
    # ------------------------------------------------------------------ #
    def _slice(self, vec: np.ndarray, key: str) -> np.ndarray:
        a, b = self.layout[key]
        return vec[a:b]

    def _action_row_to_ebench(self, row: np.ndarray) -> Dict[str, Any]:
        # EBench expects (16,) = [lj(6), lg(2), rj(6), rg(2)] — re-interleave.
        lj = self._slice(row, "left_joints")
        rj = self._slice(row, "right_joints")
        lg = self._slice(row, "left_gripper")
        rg = self._slice(row, "right_gripper")
        arm = np.concatenate([lj, lg, rj, rg], axis=0).astype(np.float32)
        base = self._slice(row, "base_delta").astype(np.float32)
        return {
            "action": arm,
            "base_motion": base,
            "control_type": self.control_type,
            "is_rel": self.is_rel,
            "base_is_rel": self.base_is_rel,
        }

    def _chunk_to_ebench(self, chunk: np.ndarray) -> List[Dict[str, Any]]:
        if self.max_chunk_len is not None:
            chunk = chunk[: self.max_chunk_len]
        return [self._action_row_to_ebench(row) for row in chunk]

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _one_inference(self, obs: Dict[str, Any]):
        if self.worker_id not in obs:
            raise KeyError(
                f"worker_id '{self.worker_id}' not in EBench obs; got {list(obs.keys())}"
            )
        worker_obs = obs[self.worker_id]["obs"]
        example = self._obs_to_example(worker_obs)

        t0 = time.time()
        chunk = self._predict_chunk(example)
        elapsed = time.time() - t0

        actions = self._chunk_to_ebench(chunk)
        if self.chunk_mode:
            payload = {self.worker_id: actions}
        else:
            payload = {self.worker_id: actions[0]}

        self._step_counter += 1
        if self._step_counter % self.log_every == 0:
            logger.info(
                "step %d: %d actions, inference=%.3fs",
                self._step_counter, len(actions), elapsed,
            )
        return payload

    def run(self) -> None:
        try:
            obs = self.eval_client.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            while not done:
                action = self._one_inference(obs)
                obs, done = self.eval_client.step(action)
                if not self.chunk_mode:
                    # non-chunk mode: keep feeding single actions from the same chunk
                    # until done or until we re-query the model. Here we re-query every step
                    # so this branch is effectively single-step inference.
                    pass
        finally:
            self.close()

    def close(self) -> None:
        try:
            self.eval_client.close()
        except Exception:
            logger.exception("EvalClient.close() raised")
        try:
            self.policy.close()
        except Exception:
            logger.exception("policy.close() raised")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StarVLA ↔ EBench bridge")
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to bridge_config.yml",
    )
    parser.add_argument(
        "--log_level", default="INFO", type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    bridge = EBenchBridge(cfg)
    bridge.run()


if __name__ == "__main__":
    main()
