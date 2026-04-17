#!/bin/bash
# Launch the StarVLA websocket policy server for EBench evaluation.
# The bridge (bridge.py) will connect to it as a client.

export PYTHONPATH=$(pwd):${PYTHONPATH}
export star_vla_python=/root/miniconda3/envs/starvla/bin/python
your_ckpt=results/Checkpoints/ebench/QwenOFT-ebench/checkpoints/steps_XXXXX_pytorch_model.pt
gpu_id=0
port=5694

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
