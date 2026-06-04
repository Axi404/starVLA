#!/bin/bash
# Launch the StarVLA websocket policy server for EBench evaluation.
# The bridge (bridge.py) will connect to it as a client.

EVAL_FILES_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PATH=$(cd "$EVAL_FILES_PATH/../../.." && pwd)
cd "$STARVLA_PATH"

export PYTHONPATH=$STARVLA_PATH:${PYTHONPATH}
export star_vla_python=${star_vla_python:-/home/gaoning/miniconda3/envs/star/bin/python}
your_ckpt=${your_ckpt:-results/Checkpoints/QwenOFT-all-200k-50chunk-ebench-512-over-2B-state/checkpoints/steps_200000_pytorch_model.pt}
gpu_id=${gpu_id:-0}
port=${port:-5694}

# export DEBUG=true
# idle_timeout=-1 keeps the server alive indefinitely; the bridge may sit idle
# while the EBench simulator spins up workers.
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --idle_timeout -1 \
    --use_bf16
