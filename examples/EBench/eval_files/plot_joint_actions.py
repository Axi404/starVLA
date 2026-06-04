#!/usr/bin/env python3
"""Plot EBench logged joint action outputs.

Reads client-side ``steps.jsonl`` files saved by ``genmanip_client`` and plots
the executed model action values. For the current lift2/R5a bridge format the
first 16 action dimensions are:

  left joints(6), left gripper(2), right joints(6), right gripper(2)

Any remaining dimensions are plotted separately as extra/base values.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = [
    "left_j0",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_grip0",
    "left_grip1",
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_grip0",
    "right_grip1",
]

GROUPS = [
    ("Left arm joints", slice(0, 6)),
    ("Left gripper", slice(6, 8)),
    ("Right arm joints", slice(8, 14)),
    ("Right gripper", slice(14, 16)),
]


@dataclass(frozen=True)
class Episode:
    path: Path
    task: str
    episode: str
    steps: np.ndarray
    actions: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.task}/{self.episode}"


def _episode_from_path(path: Path) -> tuple[str, str]:
    # .../<task>/<episode>/steps.jsonl
    return path.parent.parent.name, path.parent.name


def load_episodes(root: Path) -> list[Episode]:
    episodes: list[Episode] = []
    for path in sorted(root.rglob("steps.jsonl")):
        task, episode_id = _episode_from_path(path)
        steps: list[int] = []
        actions: list[list[float]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                action = row.get("action")
                if action is None:
                    continue
                steps.append(int(row.get("step", len(steps))))
                actions.append([float(x) for x in action])
        if actions:
            episodes.append(
                Episode(
                    path=path,
                    task=task,
                    episode=episode_id,
                    steps=np.asarray(steps, dtype=np.int64),
                    actions=np.asarray(actions, dtype=np.float64),
                )
            )
    return episodes


def write_stats_csv(episodes: list[Episode], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    max_dim = max(ep.actions.shape[1] for ep in episodes)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "dim", "name", "count", "min", "max", "mean", "std"])
        for ep in episodes:
            for dim in range(max_dim):
                if dim >= ep.actions.shape[1]:
                    continue
                values = ep.actions[:, dim]
                name = JOINT_NAMES[dim] if dim < len(JOINT_NAMES) else f"extra_{dim}"
                writer.writerow(
                    [
                        ep.label,
                        dim,
                        name,
                        values.size,
                        f"{values.min():.8g}",
                        f"{values.max():.8g}",
                        f"{values.mean():.8g}",
                        f"{values.std():.8g}",
                    ]
                )


def _plot_boundaries(ax: plt.Axes, boundaries: list[tuple[int, str]]) -> None:
    for x, label in boundaries:
        ax.axvline(x, color="#b8b8b8", lw=0.6, alpha=0.45)
    top = ax.get_ylim()[1]
    for x, label in boundaries:
        ax.text(
            x + 1,
            top,
            label,
            rotation=90,
            va="top",
            ha="left",
            fontsize=6,
            color="#666666",
            alpha=0.85,
        )


def plot_overview(episodes: list[Episode], out_png: Path, title: str) -> None:
    joint_arrays = [ep.actions[:, :16] for ep in episodes if ep.actions.shape[1] >= 16]
    if not joint_arrays:
        raise ValueError("No episode has at least 16 action dimensions.")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    boundaries: list[tuple[int, str]] = []
    offset = 0
    for ep in episodes:
        if ep.actions.shape[1] < 16:
            continue
        n = ep.actions.shape[0]
        x_parts.append(np.arange(offset, offset + n))
        y_parts.append(ep.actions[:, :16])
        boundaries.append((offset, ep.label))
        offset += n

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts, axis=0)
    colors = plt.cm.tab20(np.linspace(0, 1, 16))

    fig, axes = plt.subplots(4, 1, figsize=(22, 15), sharex=True)
    fig.suptitle(title, fontsize=18, fontweight="bold")
    for ax, (group_name, dim_slice) in zip(axes, GROUPS):
        dims = range(dim_slice.start, dim_slice.stop)
        for dim in dims:
            ax.plot(x, y[:, dim], lw=1.0, color=colors[dim], label=f"{dim}: {JOINT_NAMES[dim]}")
        ax.set_title(group_name, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("action value")
        ax.grid(True, color="#dddddd", lw=0.5, alpha=0.75)
        ax.legend(ncol=min(6, len(list(dims))), fontsize=8, loc="upper right", frameon=False)
        _plot_boundaries(ax, boundaries)
    axes[-1].set_xlabel("global executed step, episodes concatenated")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_extra_dims(episodes: list[Episode], out_png: Path, title: str) -> bool:
    max_dim = max(ep.actions.shape[1] for ep in episodes)
    if max_dim <= 16:
        return False

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    boundaries: list[tuple[int, str]] = []
    offset = 0
    for ep in episodes:
        n = ep.actions.shape[0]
        padded = np.full((n, max_dim - 16), np.nan, dtype=np.float64)
        if ep.actions.shape[1] > 16:
            padded[:, : ep.actions.shape[1] - 16] = ep.actions[:, 16:]
        x_parts.append(np.arange(offset, offset + n))
        y_parts.append(padded)
        boundaries.append((offset, ep.label))
        offset += n

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts, axis=0)

    fig, ax = plt.subplots(figsize=(22, 6))
    for local_dim in range(y.shape[1]):
        ax.plot(x, y[:, local_dim], lw=1.15, label=f"{16 + local_dim}: extra_{16 + local_dim}")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("global executed step, episodes concatenated")
    ax.set_ylabel("action value")
    ax.grid(True, color="#dddddd", lw=0.5, alpha=0.75)
    ax.legend(ncol=min(4, y.shape[1]), fontsize=9, loc="upper right", frameon=False)
    _plot_boundaries(ax, boundaries)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def plot_episode_grid(episodes: list[Episode], out_png: Path) -> None:
    n = len(episodes)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, max(4, rows * 3.2)), squeeze=False)
    colors = plt.cm.tab20(np.linspace(0, 1, 16))

    for ax in axes.ravel():
        ax.axis("off")

    for ax, ep in zip(axes.ravel(), episodes):
        ax.axis("on")
        y = ep.actions[:, : min(16, ep.actions.shape[1])]
        x = ep.steps if ep.steps.shape[0] == y.shape[0] else np.arange(y.shape[0])
        for dim in range(y.shape[1]):
            ax.plot(x, y[:, dim], lw=0.8, color=colors[dim], alpha=0.9)
        ax.set_title(f"{ep.label} ({y.shape[0]} steps)", fontsize=10, loc="left")
        ax.grid(True, color="#e1e1e1", lw=0.45, alpha=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel("joint action")

    handles = [
        plt.Line2D([0], [0], color=colors[i], lw=1.5, label=f"{i}: {JOINT_NAMES[i]}")
        for i in range(16)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle("Joint action outputs per episode", fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_heatmap(episodes: list[Episode], out_png: Path) -> None:
    y = np.concatenate([ep.actions[:, :16] for ep in episodes if ep.actions.shape[1] >= 16], axis=0)
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(y.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_yticks(np.arange(16))
    ax.set_yticklabels([f"{i}: {name}" for i, name in enumerate(JOINT_NAMES)])
    ax.set_xlabel("global executed step, episodes concatenated")
    ax.set_title("Joint action heatmap", fontsize=15, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("action value")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("playground/results/EBench/client/ebench/starvla_ebench"),
        help="Root containing episode */steps.jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/plots.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.root / "plots"
    episodes = load_episodes(args.root)
    if not episodes:
        raise SystemExit(f"No steps.jsonl with action found under {args.root}")

    total_steps = sum(ep.actions.shape[0] for ep in episodes)
    max_dim = max(ep.actions.shape[1] for ep in episodes)
    title = f"EBench action outputs: {len(episodes)} episodes, {total_steps} steps"

    plot_overview(episodes, out_dir / "joint_actions_overview.png", title)
    plot_episode_grid(episodes, out_dir / "joint_actions_by_episode.png")
    plot_heatmap(episodes, out_dir / "joint_actions_heatmap.png")
    extra_written = plot_extra_dims(episodes, out_dir / "extra_action_dims.png", "Extra/base action dimensions")
    write_stats_csv(episodes, out_dir / "joint_action_stats.csv")

    summary = {
        "root": str(args.root),
        "out_dir": str(out_dir),
        "episodes": len(episodes),
        "total_steps": total_steps,
        "max_action_dim": max_dim,
        "outputs": [
            "joint_actions_overview.png",
            "joint_actions_by_episode.png",
            "joint_actions_heatmap.png",
            "joint_action_stats.csv",
        ],
    }
    if extra_written:
        summary["outputs"].append("extra_action_dims.png")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
