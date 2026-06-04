#!/bin/bash
# Launch the StarVLA ↔ EBench bridge.
# Prerequisites:
#   1) StarVLA policy server is already running (see run_policy_server.sh)
#   2) EBench eval server is running:
#        (isaac41sapien)  cd ~/GenManip-Sim && python ray_eval_server.py --no_save_process
#      and the desired benchmark has been submitted (run_id must match
#      `ebench_run_id` in bridge_config.yml):
#        (star)           gmp submit ebench/generalist/test_mini --run_id starvla_ebench
#   3) `genmanip_client` is importable in the active Python env (star)

set -e

EVAL_FILES_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PATH=$EVAL_FILES_PATH/../../..

export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

# Bypass system proxy for localhost (the StarVLA policy server and EBench eval
# server both run on 127.0.0.1; the user's shell proxy would otherwise re-route
# the requests and fail).
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

# Redirect client-side EvalClient artifacts (per-episode video, results) under
# playground/, which is git-ignored. Without this they pile up in ./client_results/.
export GENMANIP_RESULT_DIR="${GENMANIP_RESULT_DIR:-$STARVLA_PATH/playground/results/EBench/client}"

# genmanip_client's EvalClient imports turbojpeg at module load; the conda env
# ships libturbojpeg.so under $CONDA_PREFIX/lib, but plain `python` invocation
# does not always inherit LD_LIBRARY_PATH from conda activation. Set it
# explicitly so the import succeeds even when this script is run outside an
# activated shell (e.g. via `conda run -n star ...`).
star_env=${star_env:-/home/gaoning/miniconda3/envs/star}
export LD_LIBRARY_PATH="$star_env/lib:${LD_LIBRARY_PATH:-}"
star_python=${star_python:-$star_env/bin/python}

"$star_python" "$EVAL_FILES_PATH/bridge.py" \
    --config "$EVAL_FILES_PATH/bridge_config.yml" \
    --log_level INFO
