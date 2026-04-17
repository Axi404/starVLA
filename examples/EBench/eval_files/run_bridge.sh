#!/bin/bash
# Launch the StarVLA ↔ EBench bridge.
# Prerequisites:
#   1) StarVLA policy server is already running (see run_policy_server.sh)
#   2) EBench evaluation server is already running (see EBench docs)
#   3) `genmanip_client` is importable in the active Python env

EVAL_FILES_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PATH=$EVAL_FILES_PATH/../../..

export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

python $EVAL_FILES_PATH/bridge.py \
    --config $EVAL_FILES_PATH/bridge_config.yml \
    --log_level INFO
