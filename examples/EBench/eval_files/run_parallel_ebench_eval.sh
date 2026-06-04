#!/bin/bash
# Launch N independent EBench evaluation groups:
#   GenManip/Ray eval server + StarVLA policy server + bridge.
#
# By default all groups use the same EBench run_id, so GenManip's shared
# result/progress directory can coordinate work across independent servers.

set -euo pipefail

EVAL_FILES_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PATH=$(cd "$EVAL_FILES_PATH/../../.." && pwd)

usage() {
    cat <<'EOF'
Usage:
  bash examples/EBench/eval_files/run_parallel_ebench_eval.sh [N]
  bash examples/EBench/eval_files/run_parallel_ebench_eval.sh --stop

Common env vars:
  N                       Number of groups (default: first arg or 1)
  GPU_IDS                 Comma-separated GPUs, one per group (default: 0..N-1)
  POLICY_GPU_IDS          Optional comma-separated policy GPUs; defaults to GPU_IDS
  EBENCH_BASE_PORT        First GenManip eval-server port (default: 8087)
  POLICY_BASE_PORT        First StarVLA policy-server port (default: 5694)
  EBENCH_RUN_ID           Shared run id (default: starvla_ebench)
  RUN_ID_MODE             shared|per_group (default: shared)
  EBENCH_CONFIG           Benchmark config submitted to every server (default: ebench/generalist)
  POLICY_CKPT             StarVLA checkpoint path
  START_RAY_HEADS         1 to start one Ray head per group (default: 0)
  RAY_BASE_PORT           First Ray head port when START_RAY_HEADS=1 (default: 6379)
  LOG_DIR                 Log/pid/config root (default: /tmp/ebench_parallel)

Examples:
  GPU_IDS=0,1 N=2 START_RAY_HEADS=1 bash examples/EBench/eval_files/run_parallel_ebench_eval.sh
  LOG_DIR=/tmp/ebench_parallel bash examples/EBench/eval_files/run_parallel_ebench_eval.sh --stop
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

N=${N:-${1:-1}}
GENMANIP_SIM_PATH=${GENMANIP_SIM_PATH:-$HOME/GenManip-Sim}
ISAAC_PYTHON=${ISAAC_PYTHON:-$HOME/miniconda3/envs/isaac41sapien/bin/python}
STAR_ENV=${STAR_ENV:-$HOME/miniconda3/envs/star}
STAR_PYTHON=${STAR_PYTHON:-$STAR_ENV/bin/python}
GMP=${GMP:-$STAR_ENV/bin/gmp}
RAY=${RAY:-$HOME/miniconda3/envs/isaac41sapien/bin/ray}

EBENCH_CONFIG=${EBENCH_CONFIG:-ebench/generalist}
EBENCH_RUN_ID=${EBENCH_RUN_ID:-starvla_ebench}
RUN_ID_MODE=${RUN_ID_MODE:-shared}
EBENCH_HOST=${EBENCH_HOST:-127.0.0.1}
EBENCH_BASE_PORT=${EBENCH_BASE_PORT:-8087}
POLICY_BASE_PORT=${POLICY_BASE_PORT:-5694}
POLICY_CKPT=${POLICY_CKPT:-results/Checkpoints/QwenOFT-all-200k-50chunk-ebench-512-over-2B-state/checkpoints/steps_200000_pytorch_model.pt}
UNNORM_KEY=${UNNORM_KEY:-new_embodiment}

START_RAY_HEADS=${START_RAY_HEADS:-0}
RAY_BASE_PORT=${RAY_BASE_PORT:-6379}
RAY_DASHBOARD_BASE_PORT=${RAY_DASHBOARD_BASE_PORT:-8265}

SAVE_PROCESS=${SAVE_PROCESS:-0}
LOG_DIR=${LOG_DIR:-/tmp/ebench_parallel}
CLIENT_RESULT_DIR=${CLIENT_RESULT_DIR:-$STARVLA_PATH/playground/results/EBench/client_parallel}
SERVER_STARTUP_POLLS=${SERVER_STARTUP_POLLS:-120}
SERVER_STARTUP_SLEEP=${SERVER_STARTUP_SLEEP:-5}

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

info() { echo "[INFO] $*"; }
ok() { echo "[OK] $*"; }
err() { echo "[ERROR] $*" >&2; }

split_csv() {
    local value="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "$value"
}

default_gpu_ids() {
    local ids=()
    for ((i = 0; i < N; i++)); do
        ids+=("$i")
    done
    (IFS=','; echo "${ids[*]}")
}

port_open() {
    local host="$1"
    local port="$2"
    (exec 3<>"/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

ebench_up() {
    local port="$1"
    curl -s --noproxy '*' --max-time 5 "http://${EBENCH_HOST}:${port}/status" >/dev/null 2>&1
}

record_pid() {
    local group_dir="$1"
    local name="$2"
    local pid="$3"
    echo "${name} ${pid}" >> "${group_dir}/pids"
}

kill_pid_file() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 0
    tac "$pid_file" | while read -r name pid; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            info "Stopping ${name} pid=${pid}"
            kill "$pid" 2>/dev/null || true
        fi
    done
}

stop_all() {
    shopt -s nullglob
    for pid_file in "$LOG_DIR"/group_*/pids; do
        kill_pid_file "$pid_file"
    done
}

if [[ "${1:-}" == "--stop" ]]; then
    stop_all
    exit 0
fi

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    err "N must be a positive integer; got '$N'"
    exit 2
fi

if [[ "$RUN_ID_MODE" != "shared" && "$RUN_ID_MODE" != "per_group" ]]; then
    err "RUN_ID_MODE must be 'shared' or 'per_group'; got '$RUN_ID_MODE'"
    exit 2
fi

GPU_IDS=${GPU_IDS:-$(default_gpu_ids)}
POLICY_GPU_IDS=${POLICY_GPU_IDS:-$GPU_IDS}
split_csv "$GPU_IDS" EBENCH_GPUS
split_csv "$POLICY_GPU_IDS" POLICY_GPUS

if [[ "${#EBENCH_GPUS[@]}" -lt "$N" || "${#POLICY_GPUS[@]}" -lt "$N" ]]; then
    err "Need at least N GPU ids in GPU_IDS and POLICY_GPU_IDS"
    exit 2
fi

mkdir -p "$LOG_DIR"

write_bridge_config() {
    local path="$1"
    local policy_port="$2"
    local ebench_port="$3"
    local run_id="$4"
    cat > "$path" <<EOF
policy_host: "127.0.0.1"
policy_port: ${policy_port}
policy_ckpt_path: "${POLICY_CKPT}"
unnorm_key: "${UNNORM_KEY}"

ebench_base_url: "http://${EBENCH_HOST}:${ebench_port}"
ebench_worker_id: "0"
ebench_run_id: "${run_id}"
save_process: true
save_result: true

camera_keys:
  - "video.overlook_camera_view"
  - "video.left_camera_view"
  - "video.right_camera_view"

include_state: true
state_order:
  - "left_joints"
  - "right_joints"
  - "left_gripper"
  - "right_gripper"
  - "base"
state_layout:
  left_joints:   {key: "state.joints",  slice: [0, 6]}
  right_joints:  {key: "state.joints",  slice: [6, 12]}
  left_gripper:  {key: "state.gripper", slice: [0, 2]}
  right_gripper: {key: "state.gripper", slice: [2, 4]}
  base:          {key: "state.base",    slice: [0, 3]}

action_layout:
  left_joints:  [0, 6]
  right_joints: [6, 12]
  left_gripper: [12, 14]
  right_gripper: [14, 16]
  base_delta:   [16, 19]

control_type: "joint_position"
is_rel: false

chunk_mode: true
actions_per_inference: null
max_chunk_len: null

image_size: [224, 224]
max_steps: null
log_every: 10
debug: true
EOF
}

start_ray_head() {
    local group_id="$1"
    local group_dir="$2"
    local gpu="$3"
    local ray_port=$((RAY_BASE_PORT + group_id))
    local dashboard_port=$((RAY_DASHBOARD_BASE_PORT + group_id))
    local ray_tmp="${group_dir}/ray"

    mkdir -p "$ray_tmp"
    info "Group ${group_id}: starting Ray head on :${ray_port} (gpu=${gpu})"
    (
        CUDA_VISIBLE_DEVICES="$gpu" "$RAY" start --head --block \
            --node-ip-address=127.0.0.1 \
            --port="$ray_port" \
            --dashboard-port="$dashboard_port" \
            --temp-dir="$ray_tmp" \
            --num-gpus=1 \
            --disable-usage-stats
    ) > "${group_dir}/ray_head.log" 2>&1 &
    record_pid "$group_dir" "ray_head" "$!"

    for _ in $(seq 1 30); do
        port_open 127.0.0.1 "$ray_port" && break
        sleep 1
    done
    if ! port_open 127.0.0.1 "$ray_port"; then
        err "Group ${group_id}: Ray head did not open :${ray_port}; see ${group_dir}/ray_head.log"
        exit 1
    fi
    echo "127.0.0.1:${ray_port}"
}

start_group() {
    local group_id="$1"
    local ebench_port=$((EBENCH_BASE_PORT + group_id))
    local policy_port=$((POLICY_BASE_PORT + group_id))
    local e_gpu="${EBENCH_GPUS[$group_id]}"
    local p_gpu="${POLICY_GPUS[$group_id]}"
    local run_id="$EBENCH_RUN_ID"
    local group_dir="${LOG_DIR}/group_${group_id}"
    local bridge_cfg="${group_dir}/bridge_config.yml"
    local ray_address=""
    local save_flag=("--no_save_process")

    if [[ "$RUN_ID_MODE" == "per_group" ]]; then
        run_id="${EBENCH_RUN_ID}_g${group_id}"
    fi
    if [[ "$SAVE_PROCESS" == "1" ]]; then
        save_flag=("--save_process")
    fi

    mkdir -p "$group_dir"
    : > "${group_dir}/pids"
    write_bridge_config "$bridge_cfg" "$policy_port" "$ebench_port" "$run_id"

    if [[ "$START_RAY_HEADS" == "1" ]]; then
        ray_address=$(start_ray_head "$group_id" "$group_dir" "$e_gpu")
    fi

    info "Group ${group_id}: starting EBench server on :${ebench_port} (gpu=${e_gpu}, run_id=${run_id})"
    (
        cd "$GENMANIP_SIM_PATH"
        if [[ -n "$ray_address" ]]; then
            env CUDA_VISIBLE_DEVICES="$e_gpu" RAY_ADDRESS="$ray_address" \
                "$ISAAC_PYTHON" ray_eval_server.py --host "$EBENCH_HOST" --port "$ebench_port" "${save_flag[@]}"
        else
            env CUDA_VISIBLE_DEVICES="$e_gpu" \
                "$ISAAC_PYTHON" ray_eval_server.py --host "$EBENCH_HOST" --port "$ebench_port" "${save_flag[@]}"
        fi
    ) > "${group_dir}/ebench_server.log" 2>&1 &
    record_pid "$group_dir" "ebench_server" "$!"

    for _ in $(seq 1 "$SERVER_STARTUP_POLLS"); do
        ebench_up "$ebench_port" && break
        sleep "$SERVER_STARTUP_SLEEP"
    done
    if ! ebench_up "$ebench_port"; then
        err "Group ${group_id}: EBench server failed; see ${group_dir}/ebench_server.log"
        exit 1
    fi
    ok "Group ${group_id}: EBench server ready"

    info "Group ${group_id}: submitting ${EBENCH_CONFIG}"
    LD_LIBRARY_PATH="$STAR_ENV/lib:${LD_LIBRARY_PATH:-}" \
        "$GMP" submit "$EBENCH_CONFIG" --run_id "$run_id" \
        --host "$EBENCH_HOST" --port "$ebench_port" \
        > "${group_dir}/submit.log" 2>&1

    info "Group ${group_id}: starting policy server on :${policy_port} (gpu=${p_gpu})"
    (
        cd "$STARVLA_PATH"
        env star_vla_python="$STAR_PYTHON" your_ckpt="$POLICY_CKPT" gpu_id="$p_gpu" port="$policy_port" \
            bash "$EVAL_FILES_PATH/run_policy_server.sh"
    ) > "${group_dir}/policy_server.log" 2>&1 &
    record_pid "$group_dir" "policy_server" "$!"

    for _ in $(seq 1 "$SERVER_STARTUP_POLLS"); do
        port_open 127.0.0.1 "$policy_port" && break
        if rg -q "Traceback|Error:" "${group_dir}/policy_server.log" 2>/dev/null; then
            break
        fi
        sleep "$SERVER_STARTUP_SLEEP"
    done
    if ! port_open 127.0.0.1 "$policy_port"; then
        err "Group ${group_id}: policy server failed; see ${group_dir}/policy_server.log"
        exit 1
    fi
    ok "Group ${group_id}: policy server ready"

    info "Group ${group_id}: starting bridge"
    (
        cd "$STARVLA_PATH"
        env \
            PYTHONPATH="$STARVLA_PATH:$EVAL_FILES_PATH:${PYTHONPATH:-}" \
            NO_PROXY="$NO_PROXY" \
            no_proxy="$no_proxy" \
            GENMANIP_RESULT_DIR="${CLIENT_RESULT_DIR}/group_${group_id}" \
            LD_LIBRARY_PATH="$STAR_ENV/lib:${LD_LIBRARY_PATH:-}" \
            "$STAR_PYTHON" "$EVAL_FILES_PATH/bridge.py" --config "$bridge_cfg" --log_level INFO
    ) > "${group_dir}/bridge.log" 2>&1 &
    record_pid "$group_dir" "bridge" "$!"

    ok "Group ${group_id}: running (logs: ${group_dir})"
}

trap 'stop_all; exit 130' INT TERM

info "Launching ${N} EBench group(s). Logs: ${LOG_DIR}"
if [[ "$RUN_ID_MODE" == "shared" ]]; then
    info "All groups use shared run_id=${EBENCH_RUN_ID}"
fi

for ((i = 0; i < N; i++)); do
    start_group "$i"
done

info "All groups launched. Ctrl-C stops processes started by this script."
info "To stop from another terminal: LOG_DIR=${LOG_DIR} bash ${BASH_SOURCE[0]} --stop"

wait
