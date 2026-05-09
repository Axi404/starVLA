"""Open-loop chunk visualization for a starVLA dosw1 checkpoint.

Per frame t in the episode:
  1. read GT state + 3 GT camera frames
  2. policy.predict_from_pil → chunk shape (50, 14)
  3. stash chunk for plotting

Plot: 14 subplots (one per joint dim) with
  * black solid     — gt action[t]
  * red dashed      — gt state[t]
  * viridis lines   — every emit-frame's densified predicted chunk
                      (color = emit frame t, semi-transparent)
  * optional markers at the 50 sparse model output points (--plot-sparse-markers)

This is the open-loop counterpart to ``closed_loop_local.py``: state input is
always GT, never substituted with the previous prediction. So drift is bounded
by single-step prediction error, not cumulative.

Adapted from the original Table30v2 reference script, ported to this repo's
RoboChallenge_table30v2 stack: dosw1 14-d dual-arm, q99 norm, raw parquet
layout (no starvla permutation).

Example::

    python examples/RoboChallenge_table30v2/eval_files/open_loop_local.py \\
        --checkpoint results/QwenOFT-all-150k-50chunk-rc2-dosw1-q99/checkpoints/flat_steps_150000_pytorch_model.pt \\
        --task fold_the_clothes --episode-idx 0 \\
        --infer-stride 1 --plot-stride 4
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image

from examples.RoboChallenge_table30v2.eval_files.model2robochallenge_interface import (
    ROBOT_SPECS,
    RoboChallengePolicy,
    _normalize_q99,
    _unnormalize_q99,
)

DATASET_ROOT_DEFAULT = "playground/Dataset"
OUTPUT_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "output"

# DOSW1 chunk_stride from train_files/data_registry/data_config.py:71
CHUNK_STRIDE = 4

# Raw-layout joint labels for plotting (matches parquet observation.state).
JOINT_LABELS = (
    "L_j0", "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_grip",
    "R_j0", "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_grip",
)

logger = logging.getLogger("open_loop_local")


class VideoStream:
    """Frame-by-frame mp4 reader; never holds the whole video in RAM.

    Long episodes (fold_the_clothes ep0 = 2741 frames × 3 cams) blow past
    cgroup RAM if fully decoded. Streaming keeps the working set at one
    decoded frame per cam.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.container = av.open(path)
        self._iter = self.container.decode(video=0)
        self.frame_count = self.container.streams.video[0].frames or 0

    def next_frame(self) -> np.ndarray:
        return next(self._iter).to_ndarray(format="rgb24")

    def close(self) -> None:
        self.container.close()


def auto_prompt(task_dir: Path) -> Optional[str]:
    """Read the first instruction from ``meta/tasks.jsonl`` (if present)."""
    tasks_path = task_dir / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return None
    line = tasks_path.read_text().splitlines()[0]
    return json.loads(line).get("task")


def densify_chunk(chunk: np.ndarray, stride: int) -> np.ndarray:
    """Linearly interpolate a stride-sparse action chunk to dense raw-frame resolution.

    A predicted ``(horizon, dim)`` chunk lives on raw-frame offsets
    ``[0, stride, 2*stride, ..., (horizon-1)*stride]``. Returns shape
    ``((horizon-1)*stride + 1, dim)``.
    """
    if stride == 1:
        return chunk
    horizon, dim = chunk.shape
    dense_len = (horizon - 1) * stride + 1
    sparse_x = np.arange(horizon) * stride
    dense_x = np.arange(dense_len)
    out = np.empty((dense_len, dim), dtype=chunk.dtype)
    for d in range(dim):
        out[:, d] = np.interp(dense_x, sparse_x, chunk[:, d])
    return out


def episode_paths(dataset_root: Path, task: str, episode_idx: int) -> Dict[str, Path]:
    """Resolve parquet + per-cam mp4 paths for ``task/episode_idx`` in LeRobot v2.1."""
    task_dir = dataset_root / task
    chunk = f"chunk-{episode_idx // 1000:03d}"
    ep_name = f"episode_{episode_idx:06d}"
    parquet = task_dir / "data" / chunk / f"{ep_name}.parquet"
    cams = ROBOT_SPECS["dosw1"].image_types
    videos = {cam: task_dir / "videos" / chunk / f"observation.images.{cam}" / f"{ep_name}.mp4" for cam in cams}
    return {"task_dir": task_dir, "parquet": parquet, **{f"video_{cam}": videos[cam] for cam in cams}}


def sanity_check_norm(state_stats: dict, x_raw: np.ndarray) -> None:
    """Verify q99 round-trips on a real raw state vector. Catches norm bugs early.

    For DOSW1 in this repo, state_stats is in the same (raw) layout as the
    parquet — narrow q99-q01 band on indices 6 and 13 (grippers) confirms it.
    """
    x_raw = np.asarray(x_raw, dtype=np.float32).reshape(-1)
    if x_raw.shape != (14,):
        raise ValueError(f"sanity check expects 14-d state, got shape {x_raw.shape}")
    y = _normalize_q99(x_raw, state_stats)
    x_back = _unnormalize_q99(y, state_stats)
    max_err = float(np.max(np.abs(x_raw - x_back)))
    if max_err > 1e-3:
        raise RuntimeError(
            f"q99 round-trip failed: max abs err={max_err:.3e}; "
            "stats may be in wrong layout or norm fns are broken"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--dataset-root", default=DATASET_ROOT_DEFAULT)
    ap.add_argument("--robot-tag", default="dosw1", choices=list(ROBOT_SPECS))
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames evaluated; default = full episode")
    ap.add_argument("--infer-stride", type=int, default=1,
                    help="Run policy every N frames (1 = every frame, 50 = once per chunk)")
    ap.add_argument("--plot-stride", type=int, default=1,
                    help="Of the inferred chunks, draw every N (1 = all). Higher = readable plot at long T.")
    ap.add_argument("--plot-sparse-markers", action="store_true",
                    help="Overlay the 50 raw model output points as scatter markers")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="Every N inferences: gc.collect + cuda.empty_cache + save partial npz")
    ap.add_argument("--action-mode", default=None, choices=("abs", "rel", "delta"),
                    help="Override ckpt config.yaml's action_mode (default: auto from yaml).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    dataset_root = Path(args.dataset_root)
    paths = episode_paths(dataset_root, args.task, args.episode_idx)
    if not paths["parquet"].exists():
        raise FileNotFoundError(paths["parquet"])

    prompt = args.prompt or auto_prompt(paths["task_dir"])
    if prompt is None:
        raise RuntimeError(f"No prompt provided and {paths['task_dir']/'meta'/'tasks.jsonl'} missing")
    logger.info("prompt: %r", prompt)

    policy = RoboChallengePolicy(
        checkpoint_path=args.checkpoint,
        robot_tag=args.robot_tag,
        n_action_steps=50,
        image_size=tuple(args.image_size),
        device=args.device,
        use_bf16=not args.no_bf16,
        action_mode=args.action_mode,
    )
    horizon = policy.n_action_steps
    spec = policy.spec
    chunk_stride = CHUNK_STRIDE if spec.robot_tag == "dosw1" else 1
    dense_horizon = (horizon - 1) * chunk_stride + 1
    logger.info("policy ready. spec=%s  horizon=%d×stride%d → dense=%d", spec, horizon, chunk_stride, dense_horizon)

    table = pq.read_table(str(paths["parquet"]))
    states_gt_all = np.stack(table.column("observation.state").to_pylist()).astype(np.float32)
    actions_gt_all = np.stack(table.column("action").to_pylist()).astype(np.float32)
    if states_gt_all.shape[1] != spec.state_dim:
        raise ValueError(f"parquet state dim={states_gt_all.shape[1]} != spec.state_dim={spec.state_dim}")
    T_total = states_gt_all.shape[0]
    T = T_total if args.max_frames is None else min(args.max_frames, T_total)
    logger.info("episode=%d  total_frames=%d  evaluating=%d", args.episode_idx, T_total, T)
    states_gt = states_gt_all[:T]
    actions_gt = actions_gt_all[:T]

    sanity_check_norm(policy.state_stats, states_gt[0])
    logger.info("q99 sanity check passed")

    streams: Dict[str, VideoStream] = {}
    for cam in spec.image_types:
        vp = paths[f"video_{cam}"]
        if not vp.exists():
            raise FileNotFoundError(vp)
        streams[cam] = VideoStream(str(vp))
        if streams[cam].frame_count and streams[cam].frame_count < T:
            raise RuntimeError(f"{vp} declares {streams[cam].frame_count} frames < {T} requested")

    infer_stride = max(1, int(args.infer_stride))
    image_size = tuple(args.image_size)
    sparse_predictions: Dict[int, np.ndarray] = {}   # t -> (horizon, 14) raw model output
    dense_predictions: Dict[int, np.ndarray] = {}    # t -> (dense_horizon, 14) interpolated
    infer_frames: List[int] = []

    out_png_path = Path(
        args.out
        or str(OUTPUT_DIR_DEFAULT / f"open_loop_{spec.robot_tag}_{args.task}_ep{args.episode_idx}.png")
    )
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    partial_npz = out_png_path.with_suffix(".partial.npz")
    ckpt_every = max(1, int(args.ckpt_every))

    def _save_partial() -> None:
        if not infer_frames:
            return
        np.savez_compressed(
            partial_npz,
            sparse_predictions=np.stack([sparse_predictions[t] for t in infer_frames], axis=0),
            dense_predictions=np.stack([dense_predictions[t] for t in infer_frames], axis=0),
            gt_actions=actions_gt,
            gt_states=states_gt,
            infer_frames=np.asarray(infer_frames, dtype=np.int64),
            chunk_stride=chunk_stride, horizon=horizon,
            dense_horizon=dense_horizon, infer_stride=infer_stride,
        )

    try:
        for t in range(T):
            frame_arrs = {cam: streams[cam].next_frame() for cam in spec.image_types}
            if t % infer_stride != 0:
                continue
            pil_imgs = []
            for cam in spec.image_types:
                arr = frame_arrs[cam]
                if arr.shape[:2] != image_size:
                    arr = np.asarray(Image.fromarray(arr).resize(image_size[::-1], Image.BILINEAR))
                pil_imgs.append(Image.fromarray(arr))
            with torch.inference_mode():
                chunk_raw = policy.predict_from_pil(pil_imgs, states_gt[t], prompt=prompt)
            sparse_predictions[t] = chunk_raw                                  # (50, 14)
            dense_predictions[t] = densify_chunk(chunk_raw, chunk_stride).astype(np.float32)  # (197, 14)
            infer_frames.append(t)
            if (len(infer_frames) % 100 == 0) or t == T - 1:
                logger.info("frame %d/%d  inferred=%d", t + 1, T, len(infer_frames))
            # Defensive: periodic gc + cuda cache clear works around a sporadic
            # SystemError in torch+flash_attn's PyCFunction wrapping that surfaces
            # after many forward passes. Also flush a partial npz so a late crash
            # doesn't lose progress.
            if len(infer_frames) % ckpt_every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _save_partial()
                logger.info("[ckpt] saved partial npz @ %d frames → %s", len(infer_frames), partial_npz)
    finally:
        for s in streams.values():
            s.close()

    out_png = str(out_png_path)
    out_npz = out_png_path.with_suffix(".npz")
    out_json = out_png_path.with_suffix(".metrics.json")

    infer_arr = np.asarray(infer_frames, dtype=np.int64)
    sparse_stacked = (
        np.stack([sparse_predictions[t] for t in infer_frames], axis=0)
        if infer_frames else np.zeros((0, horizon, spec.action_dim), dtype=np.float32)
    )
    dense_stacked = (
        np.stack([dense_predictions[t] for t in infer_frames], axis=0)
        if infer_frames else np.zeros((0, dense_horizon, spec.action_dim), dtype=np.float32)
    )
    np.savez_compressed(
        out_npz,
        sparse_predictions=sparse_stacked,    # (N_infer, 50, 14) raw model outputs
        dense_predictions=dense_stacked,       # (N_infer, dense_horizon, 14) interpolated
        gt_actions=actions_gt,
        gt_states=states_gt,
        infer_frames=infer_arr,
        chunk_stride=chunk_stride,
        horizon=horizon,
        dense_horizon=dense_horizon,
        infer_stride=infer_stride,
    )
    logger.info("wrote %s", out_npz)
    # Clean up the now-redundant partial.
    partial_npz.unlink(missing_ok=True)

    # First-step metric: how well does chunk[0] track gt_action[t]?
    # (chunk_stride=4 means chunk[0] is the prediction for frame t itself, i.e.
    #  raw offset 0, which is the only prediction directly comparable to gt_action[t])
    if infer_frames:
        ts = np.asarray(infer_frames, dtype=np.int64)
        chunk0_pred = np.stack([sparse_predictions[t][0] for t in infer_frames], axis=0)  # (N_infer, 14)
        err0 = chunk0_pred - actions_gt[ts]
        per_dim_mae0 = np.mean(np.abs(err0), axis=0)
        per_dim_rmse0 = np.sqrt(np.mean(err0 ** 2, axis=0))

        # Multi-step horizon MAE: for each horizon offset k = 0..49 (sparse stride 4),
        # mean |pred[t, k] - gt_action[t + k*chunk_stride]|, restricted to t where
        # t + k*chunk_stride < T.
        per_horizon_mae = np.full(horizon, np.nan, dtype=np.float32)
        for k in range(horizon):
            tgt = ts + k * chunk_stride
            valid = tgt < T
            if not valid.any():
                continue
            pred_k = np.stack([sparse_predictions[t][k] for t in infer_frames], axis=0)
            per_horizon_mae[k] = float(np.mean(np.abs(pred_k[valid] - actions_gt[tgt[valid]])))

        metrics = {
            "checkpoint": args.checkpoint,
            "task": args.task,
            "episode_idx": args.episode_idx,
            "n_steps_total": int(T),
            "n_chunks_predicted": int(len(infer_frames)),
            "infer_stride": infer_stride,
            "horizon": horizon,
            "chunk_stride": chunk_stride,
            "first_step_per_dim_mae": [float(v) for v in per_dim_mae0],
            "first_step_per_dim_rmse": [float(v) for v in per_dim_rmse0],
            "first_step_overall_mae": float(np.mean(np.abs(err0))),
            "first_step_overall_rmse": float(np.sqrt(np.mean(err0 ** 2))),
            "per_horizon_mae": [float(v) if not np.isnan(v) else None for v in per_horizon_mae],
            "joint_labels": list(JOINT_LABELS),
        }
        out_json.write_text(json.dumps(metrics, indent=2))
        logger.info("wrote %s", out_json)
        logger.info("first-step  overall MAE=%.4f  RMSE=%.4f",
                    metrics["first_step_overall_mae"], metrics["first_step_overall_rmse"])
        logger.info("per-dim first-step MAE: %s",
                    ["%s=%.3f" % (lbl, v) for lbl, v in zip(JOINT_LABELS, per_dim_mae0)])
        # Show how MAE grows with horizon offset (informative for receding-horizon design)
        h_summary = ", ".join(
            f"k={k}(@frame+{k*chunk_stride}):{per_horizon_mae[k]:.3f}"
            for k in (0, 1, 4, 12, 24, 49)
            if not np.isnan(per_horizon_mae[k])
        )
        logger.info("per-horizon MAE: %s", h_summary)

    # Plot
    fig, axes = plt.subplots(spec.action_dim, 1, figsize=(14, 1.5 * spec.action_dim), sharex=True, squeeze=False)
    cmap = mpl.colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=max(T - 1, 1))
    plot_stride = max(1, int(args.plot_stride))
    chunks_to_plot = infer_frames[::plot_stride]
    sparse_offsets = np.arange(horizon) * chunk_stride

    for j in range(spec.action_dim):
        ax = axes[j][0]
        if chunks_to_plot:
            segs = []
            ts_for_color = []
            sp_x, sp_y, sp_c = [], [], []
            for tt in chunks_to_plot:
                xs = np.arange(tt, tt + dense_horizon)
                ys = dense_predictions[tt][:, j]
                segs.append(np.stack([xs, ys], axis=-1))
                ts_for_color.append(tt)
                if args.plot_sparse_markers:
                    sp_x.extend(tt + sparse_offsets)
                    sp_y.extend(sparse_predictions[tt][:, j])
                    sp_c.extend([tt] * horizon)
            lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=0.5, alpha=0.45)
            lc.set_array(np.asarray(ts_for_color))
            ax.add_collection(lc)
            if sp_x:
                ax.scatter(sp_x, sp_y, c=sp_c, cmap=cmap, norm=norm, s=3, alpha=0.7, zorder=3)
        ax.plot(np.arange(T), actions_gt[:, j], color="black", lw=1.4, label="gt action")
        ax.plot(np.arange(T), states_gt[:, j], color="red", lw=0.8, ls="--", alpha=0.5, label="gt state")
        ax.set_ylabel(JOINT_LABELS[j], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, T + dense_horizon)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1][0].set_xlabel("frame index", fontsize=10)
    if chunks_to_plot:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, label="chunk emit frame t")

    suptitle = (
        f"open-loop  {spec.robot_tag}  task={args.task}  ep={args.episode_idx}  "
        f"T={T}  horizon={horizon}×stride{chunk_stride}={dense_horizon}  "
        f"infer_stride={infer_stride}  plot_stride={plot_stride}"
    )
    if infer_frames:
        suptitle += f"  first-step MAE={metrics['first_step_overall_mae']:.4f}"
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    logger.info("wrote %s", out_png)


if __name__ == "__main__":
    main()
