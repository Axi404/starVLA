"""
EBench Bridge
=============

In existing StarVLA integrations (RoboTwin, LIBERO, SimplerEnv) the benchmark
runs as a *client* and calls a StarVLA websocket policy server. EBench flips
this: EBench itself is an HTTP server, and models are expected to connect to
it via `genmanip_client.EvalClient`. StarVLA stays a websocket server.

Neither side can drive the other directly, so this script is the missing
driver — a pure client to both. Per step:

  EBench server  ──obs──▶  bridge  ──(image, lang, state?)──▶  StarVLA server
  StarVLA server ──unnormalized actions──▶  bridge  ──action──▶  EBench server

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


DEFAULT_EBENCH_STATE_LAYOUT: Dict[str, Dict[str, Any]] = {
    "left_joints": {"key": "state.joints", "slice": [0, 6]},
    "right_joints": {"key": "state.joints", "slice": [6, 12]},
    "left_gripper": {"key": "state.gripper", "slice": [0, 2]},
    "right_gripper": {"key": "state.gripper", "slice": [2, 4]},
    "base": {"key": "state.base", "slice": [0, 3]},
}
DEFAULT_EBENCH_STATE_ORDER: List[str] = [
    "left_joints",
    "right_joints",
    "left_gripper",
    "right_gripper",
    "base",
]


# --------------------------------------------------------------------------- #
# debug helpers
# --------------------------------------------------------------------------- #
def _describe(obj: Any, name: str = "obj", indent: int = 0, max_depth: int = 6) -> str:
    """Compact, structural dump of an arbitrary obj (dicts / lists / ndarrays),
    used by the first-step dump to validate obs/example/chunk/payload shapes."""
    pad = "  " * indent
    if indent > max_depth:
        return f"{pad}{name}: <max depth>"
    if isinstance(obj, np.ndarray):
        extra = ""
        if obj.size > 0 and np.issubdtype(obj.dtype, np.number):
            extra = f" min={obj.min():.4f} max={obj.max():.4f} mean={obj.mean():.4f}"
        return f"{pad}{name}: ndarray shape={tuple(obj.shape)} dtype={obj.dtype}{extra}"
    if isinstance(obj, dict):
        lines = [f"{pad}{name}: dict(len={len(obj)})"]
        for k, v in obj.items():
            lines.append(_describe(v, repr(k), indent + 1, max_depth))
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        kind = type(obj).__name__
        lines = [f"{pad}{name}: {kind}(len={len(obj)})"]
        # Only show first element unless the list is short, to keep chunks readable.
        to_show = list(range(min(2, len(obj))))
        if len(obj) > 3:
            to_show.append(len(obj) - 1)
        for i in to_show:
            lines.append(_describe(obj[i], f"[{i}]", indent + 1, max_depth))
        if len(obj) > 3:
            lines.insert(2, f"{pad}  ... ({len(obj) - 3} more)")
        return "\n".join(lines)
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return f"{pad}{name}: {type(obj).__name__}={obj!r}"
    return f"{pad}{name}: {type(obj).__name__}"


# --------------------------------------------------------------------------- #
# checkpoint helpers
# --------------------------------------------------------------------------- #
# Note: PolicyServerWrapper now owns un-normalization (see deployment/model_server/
# policy_wrapper.py:predict_action). The client only needs to read chunk size for
# logging — the action vector returned from the server is already in env units.
def _get_action_chunk_size(policy_ckpt_path: str) -> int:
    cfg, _ = read_mode_config(Path(policy_ckpt_path))
    action_model_cfg = cfg["framework"]["action_model"]
    if "action_horizon" in action_model_cfg:
        return int(action_model_cfg["action_horizon"])
    if "future_action_window_size" in action_model_cfg:
        return int(action_model_cfg["future_action_window_size"]) + 1
    raise KeyError(
        "checkpoint config is missing framework.action_model.action_horizon "
        "or future_action_window_size"
    )


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
        self._assert_camera_order_matches_training(self.camera_keys)
        self.image_size = tuple(cfg.get("image_size", [224, 224]))

        self.include_state: bool = bool(cfg.get("include_state", False))
        self.state_layout: Dict[str, Dict[str, Any]] = cfg.get(
            "state_layout", DEFAULT_EBENCH_STATE_LAYOUT
        )
        self.state_order: List[str] = list(
            cfg.get("state_order", DEFAULT_EBENCH_STATE_ORDER)
        )

        self.layout: Dict[str, List[int]] = cfg["action_layout"]
        self.control_type: str = cfg["control_type"]
        self.is_rel: bool = cfg["is_rel"]

        self.chunk_mode: bool = cfg.get("chunk_mode", True)
        self.max_chunk_len: Optional[int] = cfg.get("max_chunk_len")
        # How many predicted actions to consume per inference. None = 1 in
        # single-step mode, = full chunk in chunk mode. Setting an explicit
        # int N means: re-infer every N actions, regardless of chunk_mode.
        self.actions_per_inference: Optional[int] = cfg.get("actions_per_inference")
        self.log_every: int = cfg.get("log_every", 10)
        self.debug: bool = cfg.get("debug", True)
        self.max_steps: Optional[int] = cfg.get("max_steps")
        self.unnorm_key: Optional[str] = cfg.get("unnorm_key")

        # --- StarVLA client ---
        self.policy = WebsocketClientPolicy(
            host=cfg["policy_host"], port=cfg["policy_port"]
        )
        ckpt_path = cfg["policy_ckpt_path"]
        self.model_chunk_len = _get_action_chunk_size(ckpt_path)
        logger.info(
            "connected to StarVLA %s:%s (chunk=%d)",
            cfg["policy_host"], cfg["policy_port"], self.model_chunk_len,
        )

        # --- EBench client ---
        from genmanip_client.eval_client import EvalClient  # lazy; only at eval time
        self.eval_client = EvalClient(
            base_url=cfg["ebench_base_url"],
            worker_ids=[cfg["ebench_worker_id"]],
            run_id=cfg["ebench_run_id"],
            save_process=cfg.get("save_process", False),
            save_result=cfg.get("save_result", True),
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
    @staticmethod
    def _assert_camera_order_matches_training(keys: List[str]) -> None:
        """Training (`_pack_sample` in gr00t_lerobot/datasets.py) bucketises
        images as `prim_images + wrist_views` using the substring "wrist" on
        the key name. Our bridge forwards `camera_keys` in list order without
        re-bucketising, so the list must already be prim-first then wrist."""
        seen_wrist = False
        for k in keys:
            is_wrist = "wrist" in k or "left_camera_view" in k or "right_camera_view" in k
            if is_wrist:
                seen_wrist = True
            elif seen_wrist:
                raise ValueError(
                    "camera_keys order breaks training convention: a primary "
                    "(non-wrist) camera appears after a wrist camera. "
                    f"Got: {keys}. Put the prim/overhead view first."
                )

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv.resize(img, self.image_size, interpolation=cv.INTER_AREA)

    def _state_part(self, worker_obs: Dict[str, Any], name: str) -> np.ndarray:
        if name not in self.state_layout:
            raise KeyError(
                f"state part {name!r} missing from state_layout. "
                f"Available parts: {list(self.state_layout)}"
            )
        spec = self.state_layout[name]
        key = spec["key"]
        if key not in worker_obs:
            raise KeyError(
                f"state key {key!r} missing from EBench obs for part {name!r}. "
                f"Available state keys: {[k for k in worker_obs if k.startswith('state.')]}"
            )

        arr = np.asarray(worker_obs[key], dtype=np.float32).reshape(-1)
        a, b = spec.get("slice", [0, arr.shape[0]])
        part = arr[a:b]
        expected = b - a
        if part.shape[0] != expected:
            raise ValueError(
                f"state part {name!r} from {key!r}[{a}:{b}] has "
                f"{part.shape[0]} dims, expected {expected}; source shape={arr.shape}"
            )
        return part

    def _obs_to_state(self, worker_obs: Dict[str, Any]) -> np.ndarray:
        parts = [self._state_part(worker_obs, name) for name in self.state_order]
        state = np.concatenate(parts, axis=0).astype(np.float32)
        return state[None, :]  # training examples carry state as [T=1, D]

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

        example = {"lang": str(instruction), "image": images}
        if self.include_state:
            example["state"] = self._obs_to_state(worker_obs)
        return example

    # ------------------------------------------------------------------ #
    # StarVLA call
    # ------------------------------------------------------------------ #
    def _predict_chunk(self, example: Dict[str, Any]) -> np.ndarray:
        payload: Dict[str, Any] = {"examples": [example], "do_sample": False}
        if self.unnorm_key is not None:
            payload["unnorm_key"] = self.unnorm_key
        resp = self.policy.predict_action(payload)
        data = resp.get("data", resp)
        if "actions" not in data:
            raise KeyError(
                f"policy response missing 'actions'; keys={list(data.keys())}"
            )
        actions = np.asarray(data["actions"])
        if actions.ndim == 3:
            actions = actions[0]           # [B=1, T, D] -> [T, D]
        elif actions.ndim == 1:
            actions = actions[None, :]     # [D] -> [1, D]
        return actions

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
        }

    def _chunk_to_ebench(self, chunk: np.ndarray) -> List[Dict[str, Any]]:
        if self.max_chunk_len is not None:
            chunk = chunk[: self.max_chunk_len]
        return [self._action_row_to_ebench(row) for row in chunk]

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _one_inference(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run the policy on the current obs and return the full list of
        EBench-formatted action dicts for this chunk. The caller decides how
        many of them to submit, and via which EBench API (/step or /step_chunk).
        """
        if self.worker_id not in obs:
            raise KeyError(
                f"worker_id '{self.worker_id}' not in EBench obs; got {list(obs.keys())}"
            )
        worker_obs = obs[self.worker_id]["obs"]
        dump = self.debug and self._step_counter == 0

        if dump:
            logger.info(
                "=== [debug] EBench → bridge: obs (worker=%s) ===\n%s",
                self.worker_id, _describe(worker_obs, "obs"),
            )

        t0 = time.time()
        example = self._obs_to_example(worker_obs)
        if dump:
            logger.info(
                "=== [debug] bridge → StarVLA: example ===\n%s",
                _describe(example, "example"),
            )
        chunk = self._predict_chunk(example)
        if dump:
            logger.info(
                "=== [debug] StarVLA → bridge: action chunk (server-unnormalized) ===\n%s",
                _describe(chunk, "chunk"),
            )
            for slice_name, (a, b) in self.layout.items():
                s = chunk[:, a:b]
                logger.info(
                    "   slice %-14s [%2d:%2d]: min=%.4f max=%.4f mean=%.4f",
                    slice_name, a, b, s.min(), s.max(), s.mean(),
                )
        actions = self._chunk_to_ebench(chunk)
        elapsed = time.time() - t0
        self._step_counter += 1
        if self._step_counter % self.log_every == 0 or dump:
            # Surface base_delta range on every inference — the bridge has no
            # action cache, so any inter-chunk lurch in base motion shows up
            # as a discontinuity here vs the previous inference's last frame.
            a, b = self.layout["base_delta"]
            bd = chunk[:, a:b]
            logger.info(
                "inference %d: chunk_len=%d, inference=%.3fs, "
                "base_delta first=%s last=%s max_abs=%.4f",
                self._step_counter, len(actions), elapsed,
                np.array2string(bd[0], precision=4, suppress_small=True),
                np.array2string(bd[-1], precision=4, suppress_small=True),
                float(np.abs(bd).max()),
            )
        return actions

    def _actions_per_inference(self, chunk_len: int) -> int:
        """Resolve how many predicted actions to consume before re-inferring.

        - explicit `actions_per_inference` config wins
        - else chunk_mode=True  → full chunk (capped by max_chunk_len)
        - else chunk_mode=False → 1 (re-infer every sim step)
        """
        if self.actions_per_inference is not None:
            return min(self.actions_per_inference, chunk_len)
        if self.chunk_mode:
            return chunk_len  # already capped by _chunk_to_ebench via max_chunk_len
        return 1

    def _episode_was_reset(self, obs: Dict[str, Any]) -> bool:
        """True iff the worker's obs carries the per-episode `reset=True` flag
        (EBench sets this on the first obs of a freshly-reset episode, e.g. after
        invalid_state termination or a successful completion). When this fires
        mid-chunk, any remaining actions in our predicted chunk were planned for
        the previous episode's trajectory and would be meaningless — and harmful
        — when applied to the new episode's initial state. The run loop must
        break out of the inner action-submission loop and re-infer."""
        worker = obs.get(self.worker_id)
        if not isinstance(worker, dict):
            return False
        wobs = worker.get("obs")
        if not isinstance(wobs, dict):
            return False
        return bool(wobs.get("reset"))

    def run(self) -> None:
        try:
            obs = self.eval_client.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            dump_payload = self.debug
            while not done:
                actions = self._one_inference(obs)
                n = self._actions_per_inference(len(actions))
                actions = actions[:n]

                if self.chunk_mode:
                    payload = {self.worker_id: actions}
                    if dump_payload:
                        logger.info(
                            "=== [debug] bridge → EBench: chunk payload ===\n%s",
                            _describe(payload, "payload"),
                        )
                        dump_payload = False
                    obs, done = self.eval_client.step(payload)
                else:
                    # Open-loop: submit each predicted action one by one via /step.
                    for i, a in enumerate(actions):
                        payload = {self.worker_id: a}
                        if dump_payload:
                            logger.info(
                                "=== [debug] bridge → EBench: single payload (1/%d) ===\n%s",
                                n, _describe(payload, "payload"),
                            )
                            dump_payload = False
                        obs, done = self.eval_client.step(payload)
                        if done:
                            break
                        if self._episode_was_reset(obs):
                            logger.info(
                                "episode reset detected mid-chunk after %d/%d action(s); "
                                "discarding remaining %d stale predictions and re-inferring",
                                i + 1, n, n - i - 1,
                            )
                            break
                        if self.max_steps is not None and self._step_counter >= self.max_steps:
                            break

                if self.max_steps is not None and self._step_counter >= self.max_steps:
                    logger.info("max_steps=%d reached, stopping", self.max_steps)
                    break
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
