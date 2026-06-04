#!/bin/bash
#SBATCH --job-name=ebench_baseline
#SBATCH -p eailab_bench
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/petrelfs/gaoning/trash/%x-%j.out
#SBATCH --error=/mnt/petrelfs/gaoning/trash/%x-%j.err
#SBATCH --exclude=HOST-10-140-66-29,HOST-10-140-66-108,HOST-10-140-60-30,HOST-10-140-60-125,HOST-10-140-60-45

set -e

# -------------------- NCCL / 网络 --------------------
export NCCL_SOCKET_IFNAME=bond0
# 两节点一般把所有可用 mlx5 都列上更稳（按你集群实际改）
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# -------------------- 分布式必要环境 --------------------
export GPUS_PER_NODE=8
export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + RANDOM % 10000))

export PYTHONDONTWRITEBYTECODE=1

echo "SLURM_NNODES=$SLURM_NNODES  GPUS_PER_NODE=$GPUS_PER_NODE  TOTAL_GPUS=$TOTAL_GPUS"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# -------------------- 你的原始配置 --------------------
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/EBench/train_files/starvla_cotrain_ebench_abs.yaml
run_root_dir=./results/Checkpoints
data_mix=ebench_generalist
run_id=0516_${data_mix}_abs_Qwen3OFT_50

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# ===== Proxy (for wandb / huggingface / pip etc.) =====
AD_URL=${AD_URL:-10.1.20.50:23128}
if [[ -n "${AD_USER:-}" && -n "${AD_PASSWORD:-}" && -n "${AD_URL:-}" ]]; then
  export http_proxy="http://${AD_USER}:${AD_PASSWORD}@${AD_URL}/"
  export https_proxy="http://${AD_USER}:${AD_PASSWORD}@${AD_URL}/"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
fi

source /mnt/petrelfs/gaoning/miniconda3/bin/activate
conda activate starvla

# 集群内网地址别走代理（非常关键，否则 NCCL/节点互联可能被你代理“劫持”导致巨慢/挂）
ALL_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
export no_proxy="127.0.0.1,localhost,${ALL_HOSTS},${MASTER_ADDR},.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export NO_PROXY="$no_proxy"

# -------------------- 先清理每个节点残留进程 --------------------
echo "==== cleanup stale processes on all nodes ===="

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
  echo "[cleanup][$(hostname)] before:"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true

  pkill -9 pt_data_worker 2>/dev/null || true

  sleep 3

  echo "[cleanup][$(hostname)] after:"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
'

# -------------------- 关键：每个节点启动一次 accelerate --------------------
srun --jobid "$SLURM_JOBID" bash -c '
  set -e
  echo "Host=$(hostname)  SLURM_PROCID=$SLURM_PROCID"

  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    --machine_rank $SLURM_PROCID \
    --num_machines '"$SLURM_NNODES"' \
    --num_processes '"$TOTAL_GPUS"' \
    starVLA/training/train_starvla.py \
    --config_yaml '"$config_yaml"' \
    --framework.name '"$Framework_name"' \
    --framework.qwenvl.base_vlm '"$base_vlm"' \
    --datasets.vla_data.per_device_batch_size 8 \
    --datasets.vla_data.action_type abs_qpos \
    --datasets.vla_data.action_mode abs \
    --datasets.vla_data.data_mix '"$data_mix"' \
    --trainer.freeze_modules '"$freeze_module_list"' \
    --trainer.max_train_steps 200000 \
    --trainer.save_interval 10000 \
    --trainer.logging_frequency 50 \
    --trainer.eval_interval 1000 \
    --run_root_dir '"$run_root_dir"' \
    --run_id '"$run_id"' \
    --wandb_project starVLA_Robotwin \
    --wandb_entity axi-the-cat
'
