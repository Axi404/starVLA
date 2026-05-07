#!/usr/bin/env bash
# RoboChallenge Table30v2 — single-task walk-through training (UR5 / shred_paper)
# with Qwen3VL-OFT. Run from the repo root inside the `starVLA` conda env.
set -euo pipefail

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

# How many GPUs to use; defaults to all visible.
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}

# ---- training knobs (edit here) ----
TASK=shred_paper
BATCH=${BATCH:-8}
MAX_STEPS=${MAX_STEPS:-150000}
SAVE_EVERY=${SAVE_EVERY:-10000}
EVAL_EVERY=${EVAL_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-100}

run_root_dir=./playground/Checkpoints
run_id=robochallenge_table30v2_qwenoft_${TASK}_${MAX_STEPS}step
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/RoboChallenge_table30v2/train_files/starvla_qwenoft_robochallenge_table30v2.yaml \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --trainer.eval_interval "${EVAL_EVERY}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_robochallenge_table30v2 \
  --wandb_entity axi-the-cat
