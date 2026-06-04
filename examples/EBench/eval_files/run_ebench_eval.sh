#!/bin/bash
# One-click EBench evaluation launcher.
#
# Orchestrates the four moving parts of an EBench run:
#   1. EBench eval server   (ray_eval_server.py, in the GenManip-Sim repo + isaac env)
#   2. benchmark submit     (gmp submit, registers the task set with the server)
#   3. StarVLA policy server(deployment/model_server/server_policy.py, star env)
#   4. bridge               (bridge.py, drives obs<->action between the two servers)
#
# Servers that are already up are reused, so re-running this script just
# re-attaches the bridge. Servers are left running on exit for fast re-runs;
# pass `--stop` to tear everything down.
#
# Override any of the env vars below to point at a different machine layout.

set -euo pipefail

EVAL_FILES_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PATH=$(cd "$EVAL_FILES_PATH/../../.." && pwd)

# ---- configurable knobs -----------------------------------------------------
GENMANIP_SIM_PATH=${GENMANIP_SIM_PATH:-$HOME/GenManip-Sim}
ISAAC_PYTHON=${ISAAC_PYTHON:-$HOME/miniconda3/envs/isaac41sapien/bin/python}
STAR_ENV=${STAR_ENV:-$HOME/miniconda3/envs/star}
STAR_PYTHON=${STAR_PYTHON:-$STAR_ENV/bin/python}
GMP=${GMP:-$STAR_ENV/bin/gmp}

EBENCH_CONFIG=${EBENCH_CONFIG:-ebench/generalist}
EBENCH_RUN_ID=${EBENCH_RUN_ID:-starvla_ebench}   # must match ebench_run_id in bridge_config.yml
EBENCH_HOST=${EBENCH_HOST:-127.0.0.1}
EBENCH_PORT=${EBENCH_PORT:-8087}
POLICY_PORT=${POLICY_PORT:-5694}

# SAVE_PROCESS=1 keeps server-side trajectory metadata / videos / RRD files
# (written to GenManip-Sim/saved/eval_results/<bench>/<run_id>/). Default 0
# (server launched with --no_save_process). Client-side per-episode video is a
# separate switch: set `save_process: true` in bridge_config.yml.
SAVE_PROCESS=${SAVE_PROCESS:-0}

LOG_DIR=${LOG_DIR:-/tmp/ebench_eval}
mkdir -p "$LOG_DIR"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

# ---- helpers ----------------------------------------------------------------
info()  { echo -e "\033[36m→\033[0m $*"; }
ok()    { echo -e "\033[92m✓\033[0m $*"; }
err()   { echo -e "\033[31m✗\033[0m $*" >&2; }

ebench_up()  { curl -s --noproxy '*' --max-time 5 "http://$EBENCH_HOST:$EBENCH_PORT/status" >/dev/null 2>&1; }
policy_up()  { (exec 3<>"/dev/tcp/127.0.0.1/$POLICY_PORT") 2>/dev/null && exec 3>&- ; }

# ---- teardown mode ----------------------------------------------------------
if [[ "${1:-}" == "--stop" ]]; then
    info "Stopping EBench eval stack ..."
    pkill -f "bridge.py --config" 2>/dev/null && ok "bridge stopped" || true
    pkill -f "server_policy.py"   2>/dev/null && ok "policy server stopped" || true
    pkill -f "ray_eval_server.py" 2>/dev/null && ok "EBench server stopped" || true
    exit 0
fi

# ---- 1. EBench eval server --------------------------------------------------
if ebench_up; then
    ok "EBench server already up at http://$EBENCH_HOST:$EBENCH_PORT"
    [[ "$SAVE_PROCESS" == "1" ]] && info "(SAVE_PROCESS=1 ignored — reusing a server already running; --stop and rerun to apply)"
else
    if [[ "$SAVE_PROCESS" == "1" ]]; then
        SAVE_FLAG=""
        info "Starting EBench server (ray_eval_server.py, saving process artifacts) ..."
    else
        SAVE_FLAG="--no_save_process"
        info "Starting EBench server (ray_eval_server.py) ..."
    fi
    ( cd "$GENMANIP_SIM_PATH" && "$ISAAC_PYTHON" ray_eval_server.py $SAVE_FLAG\
        > "$LOG_DIR/ebench_server.log" 2>&1 & )
    for _ in $(seq 1 60); do ebench_up && break; sleep 3; done
    if ebench_up; then ok "EBench server is up (log: $LOG_DIR/ebench_server.log)"
    else err "EBench server failed to start — see $LOG_DIR/ebench_server.log"; exit 1; fi
fi

# ---- 2. submit the benchmark -----------------------------------------------
info "Submitting benchmark '$EBENCH_CONFIG' as run_id '$EBENCH_RUN_ID' ..."
LD_LIBRARY_PATH="$STAR_ENV/lib:${LD_LIBRARY_PATH:-}" \
    "$GMP" submit "$EBENCH_CONFIG" --run_id "$EBENCH_RUN_ID" \
    --host "$EBENCH_HOST" --port "$EBENCH_PORT"

# ---- 3. StarVLA policy server ----------------------------------------------
if policy_up; then
    ok "Policy server already up on :$POLICY_PORT"
else
    info "Starting StarVLA policy server ..."
    ( bash "$EVAL_FILES_PATH/run_policy_server.sh" \
        > "$LOG_DIR/policy_server.log" 2>&1 & )
    for _ in $(seq 1 120); do
        grep -q "server listening" "$LOG_DIR/policy_server.log" 2>/dev/null && break
        grep -qE "Traceback|Error:" "$LOG_DIR/policy_server.log" 2>/dev/null && break
        sleep 5
    done
    if policy_up; then ok "Policy server is up (log: $LOG_DIR/policy_server.log)"
    else err "Policy server failed to start — see $LOG_DIR/policy_server.log"; exit 1; fi
fi

# ---- 4. bridge (foreground) -------------------------------------------------
info "Launching bridge — Ctrl-C stops the bridge; servers stay up for re-runs."
echo
exec bash "$EVAL_FILES_PATH/run_bridge.sh"
