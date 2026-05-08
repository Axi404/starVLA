#!/usr/bin/env bash
# Step-2 self-test: drive upstream mock_robot_server.py with our policy.
# Requires:
#   * upstream repo cloned at $RC_REPO (default: ~/playground/Code/RoboChallengeInference)
#   * mock_robot_server.py already running on 127.0.0.1:9098
set -euo pipefail
cd "$(dirname "$0")/../../.."

# Inner file (the dir-wrapped ckpt is symlinked alongside as flat_*.pt for compatibility with read_mode_config).
CKPT="${CKPT:-./results/QwenOFT-all-150k-50chunk-rc2-dosw1-q99/checkpoints/flat_steps_150000_pytorch_model.pt}"
ROBOT_TAG="${ROBOT_TAG:-dosw1}"
PROMPT="${PROMPT:-Fold the T-shirts and stack them neatly in the upper-left corner of the table.}"
RC_REPO="${RC_REPO:-$HOME/playground/Code/RoboChallengeInference}"
MAX_WAIT="${MAX_WAIT:-60}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate starVLA_dev

export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.2/toolkit/12.2.2}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Bypass any system-wide HTTP/SOCKS proxy for the mock server on localhost.
# (e.g. on hosts with HTTP_PROXY/all_proxy=127.0.0.1:7897, requests to the mock
# get hijacked and 502 with "Bad Gateway" on /clock-sync.)
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

python examples/RoboChallenge_table30v2/eval_files/test_with_mock_server.py \
    --checkpoint "${CKPT}" \
    --robot_tag "${ROBOT_TAG}" \
    --prompt "${PROMPT}" \
    --rc_repo "${RC_REPO}" \
    --max_wait "${MAX_WAIT}" \
    --n_action_steps "${N_ACTION_STEPS}" \
    "$@"
