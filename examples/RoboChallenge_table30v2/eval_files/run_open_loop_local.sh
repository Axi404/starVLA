#!/usr/bin/env bash
# Open-loop chunk visualization: GT state every frame, no closed-loop feedback.
# Saves png + npz + metrics.json with first-step + per-horizon MAE.
set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT="${CKPT:-./results/QwenOFT-all-150k-50chunk-rc2-dosw1-q99/checkpoints/flat_steps_150000_pytorch_model.pt}"
ROBOT_TAG="${ROBOT_TAG:-dosw1}"
TASK="${TASK:-fold_the_clothes}"
EPISODE_IDX="${EPISODE_IDX:-0}"
DATASET_ROOT="${DATASET_ROOT:-playground/Dataset}"
INFER_STRIDE="${INFER_STRIDE:-1}"
PLOT_STRIDE="${PLOT_STRIDE:-4}"   # full ep0 = 2741 chunks; 4 keeps the plot legible
MAX_FRAMES="${MAX_FRAMES:-}"      # empty = full episode

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate starVLA_dev 2>/dev/null || conda activate star

export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.2/toolkit/12.2.2}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CMD=(python examples/RoboChallenge_table30v2/eval_files/open_loop_local.py
     --checkpoint "${CKPT}"
     --robot-tag "${ROBOT_TAG}"
     --task "${TASK}"
     --episode-idx "${EPISODE_IDX}"
     --dataset-root "${DATASET_ROOT}"
     --infer-stride "${INFER_STRIDE}"
     --plot-stride "${PLOT_STRIDE}")
[[ -n "${MAX_FRAMES}" ]] && CMD+=(--max-frames "${MAX_FRAMES}")

"${CMD[@]}" "$@"
