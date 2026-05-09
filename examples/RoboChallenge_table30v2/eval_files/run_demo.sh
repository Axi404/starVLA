#!/usr/bin/env bash
# Step-3 production submission: drive the real RoboChallenge platform via
# upstream's job_loop.  Mock counterpart: run_test_with_mock.sh.
set -euo pipefail
cd "$(dirname "$0")/../../.."

CKPT="${CKPT:-./results/QwenOFT-all-150k-50chunk-rc2-dosw1-q99/checkpoints/flat_steps_150000_pytorch_model.pt}"
ROBOT_TAG="${ROBOT_TAG:-dosw1}"
PROMPT="${PROMPT:-Fold the T-shirts and stack them neatly in the upper-left corner of the table.}"
RC_REPO="${RC_REPO:-$HOME/playground/Code/RoboChallengeInference}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
DURATION="${DURATION:-0.05}"

if [[ -z "${USER_TOKEN:-}" || -z "${SUBMISSION_ID:-}" ]]; then
    echo "USER_TOKEN and SUBMISSION_ID must be set." >&2
    echo "Get them from your account + submission detail page on robochallenge.cn." >&2
    exit 2
fi

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate starVLA_dev

export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.2/toolkit/12.2.2}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# NOTE: do NOT export NO_PROXY=127.0.0.1 here — unlike the mock case, we WANT
# the system HTTP/SOCKS proxy to apply for outbound api.robochallenge.cn.

python examples/RoboChallenge_table30v2/eval_files/run_demo.py \
    --user_token "${USER_TOKEN}" \
    --submission_id "${SUBMISSION_ID}" \
    --checkpoint "${CKPT}" \
    --robot_tag "${ROBOT_TAG}" \
    --prompt "${PROMPT}" \
    --rc_repo "${RC_REPO}" \
    --n_action_steps "${N_ACTION_STEPS}" \
    --duration "${DURATION}" \
    "$@"
