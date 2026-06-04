# EBench Evaluation — Bridge Mode

EBench exposes an HTTP **evaluation server** (default `:8087`), and expects
models to connect via `genmanip_client.EvalClient`. StarVLA, in contrast,
runs inference behind its own **websocket server** (so other benchmarks can
connect as clients). Both sides are servers, so nothing drives the loop.

This directory provides the missing driver — a small **bridge** client that
connects to both:

```
EBench server  ──obs──▶  bridge  ──(image, lang, state?)──▶  StarVLA server
EBench server ◀─action── bridge ◀──unnormalized actions──  StarVLA server
```

See the upstream docs for custom-model integration:
https://internrobotics.github.io/EBench-doc/evaluation/custom-model/

---

## Files

| file | purpose |
|---|---|
| `bridge.py` | the bridge program (client to StarVLA + client to EBench) |
| `bridge_config.yml` | camera mapping, action layout, ports, checkpoint path |
| `run_policy_server.sh` | launches the StarVLA websocket server |
| `run_bridge.sh` | launches the bridge |
| `requirements.txt` | bridge-side extras |

## Action / observation layout

Training follows `EBenchConfig` in
`examples/EBench/train_files/data_registry/data_config.py` (this file is
auto-discovered by the registry at import time and overrides the base
`starVLA/dataloader/gr00t_lerobot/data_config.py`). The 19-dim action vector is
concatenated in this order:

| slice | dims | meaning |
|---|---|---|
| `[0:6]` | 6 | `action.left_joints` |
| `[6:12]` | 6 | `action.right_joints` |
| `[12:14]` | 2 | `action.left_gripper` |
| `[14:16]` | 2 | `action.right_gripper` |
| `[16:19]` | 3 | `action.base_delta` (dx, dy, dyaw) |

EBench expects a **different interleaving** for the arm action:
`(16,) = [left_joints(6), left_gripper(2), right_joints(6), right_gripper(2)]`
plus `base_motion=(3,)`. The bridge re-interleaves per chunk row before
submitting. Gripper dims are trained as raw joint-position values, so the
policy server unnormalizes them back to the EBench gripper range before the
bridge submits actions.

Training keys vs eval keys differ (names only, content is the same dataset for
the current `*-over*` checkpoints):

| training (`EBenchConfig.video_keys`) | EBench eval obs              |
|--------------------------------------|------------------------------|
| `video.cam_over`                     | `video.overlook_camera_view` |
| `video.cam_left_wrist`               | `video.left_camera_view`     |
| `video.cam_right_wrist`              | `video.right_camera_view`    |

For older checkpoints trained with `cam_high` instead of `cam_over`, switch the
first `camera_keys` entry in `bridge_config.yml` back to
`video.top_camera_view`. The bridge resizes all three selected views to
`image_size` before forwarding to StarVLA.

State-aware checkpoints should set `include_state: true` in
`bridge_config.yml`. The bridge packs raw EBench `state.joints`,
`state.gripper`, and `state.base` into the training order
`[left_joints, right_joints, left_gripper, right_gripper, base]`; the policy
server then normalizes that raw state through the checkpoint's training
transform before calling the model.

## Running

### One-click

`run_ebench_eval.sh` orchestrates all four steps below — it starts the EBench
server (reusing it if already up), submits the benchmark, starts the policy
server, and attaches the bridge in the foreground:

```bash
bash examples/EBench/eval_files/run_ebench_eval.sh
# tear everything down:
bash examples/EBench/eval_files/run_ebench_eval.sh --stop
```

Override layout via env vars if your machine differs, e.g.
`GENMANIP_SIM_PATH`, `ISAAC_PYTHON`, `STAR_ENV`, `EBENCH_CONFIG`,
`EBENCH_RUN_ID`. Logs land in `/tmp/ebench_eval/`.

To keep intermediate artifacts, run with `SAVE_PROCESS=1` (server-side
trajectory metadata / videos / RRD, written under
`GenManip-Sim/saved/eval_results/<bench>/<run_id>/`). This only takes effect
when the server is *started* by the script — `--stop` first if one is already
up. For client-side per-episode video, separately set `save_process: true` in
`bridge_config.yml`.

### Parallel groups

`run_parallel_ebench_eval.sh` starts N independent groups, each with its own
GenManip eval-server port, StarVLA policy-server port, and generated bridge
config. By default every group uses the same `EBENCH_RUN_ID`, so GenManip's
shared result/progress directory can coordinate tasks across groups:

```bash
GPU_IDS=0,1 N=2 START_RAY_HEADS=1 \
  bash examples/EBench/eval_files/run_parallel_ebench_eval.sh

LOG_DIR=/tmp/ebench_parallel \
  bash examples/EBench/eval_files/run_parallel_ebench_eval.sh --stop
```

Set `START_RAY_HEADS=1` to launch one Ray head per group; leave it `0` to use
the eval server's default Ray behavior.

### Manual (four terminals)

EBench evaluation is **server-driven**: you start the EBench HTTP server, then
*submit* a benchmark to it (which spins up the simulator workers), then connect
StarVLA and the bridge. The four steps below run in four terminals.

1. **Install bridge extras.** `genmanip_client` is assumed already installed
   per the EBench docs; the bridge only needs a couple of lightweight extras:
   ```bash
   pip install -r examples/EBench/eval_files/requirements.txt
   ```
   `genmanip_client` imports `turbojpeg` at load time — if that import fails,
   install the native lib into the StarVLA env: `conda install -c conda-forge
   libjpeg-turbo`.

2. **Terminal A — EBench server.** In the EBench / GenManip-Sim repo, in its
   own conda env (e.g. `isaac41sapien`):
   ```bash
   python ray_eval_server.py --no_save_process
   ```
   This brings up the HTTP endpoint (default `http://127.0.0.1:8087`).

3. **Terminal B — submit the benchmark.** In the StarVLA env, register the
   task set with the running server. `--run_id` must match `ebench_run_id` in
   `bridge_config.yml`:
   ```bash
   gmp submit ebench/generalist/test_mini --run_id starvla_ebench
   ```

4. **Terminal C — StarVLA policy server.** Edit the checkpoint path in
   `run_policy_server.sh` and `bridge_config.yml` if needed, then:
   ```bash
   bash examples/EBench/eval_files/run_policy_server.sh
   ```

5. **Terminal D — bridge.** Once the policy server prints `server listening`:
   ```bash
   bash examples/EBench/eval_files/run_bridge.sh
   ```

Notes:
- The checkpoint path must point at the `.pt` **file** itself, with sibling
  `config.yaml` / `dataset_statistics.json` one directory up
  (`<run_dir>/checkpoints/<ckpt>.pt`).
- `EBenchConfig` (in `examples/EBench/train_files/data_registry/data_config.py`)
  must declare `embodiment_tag`, `action_key_dims` and `state_key_dims` — the
  19-dim action/state vectors do not divide evenly across the 5 sub-keys, so the
  server-side norm processor cannot infer the split on its own.

## Configuration cheat-sheet

- `chunk_mode: true` — default; submit all predicted actions in one
  `/step_chunk` call (EBench advances N sim steps internally). Recommended for
  throughput.
- `chunk_mode: false` — submit each action via its own `/step` call; useful
  for finer debug visibility (one obs / one client-side recorded frame per
  sim step).
- `actions_per_inference` — how many predicted actions to consume before
  re-inferring. `null` = natural default (full chunk in chunk_mode, 1
  otherwise). Setting `50` with `chunk_mode: false` gives an open-loop
  50-step rollout of one inference, recorded step-by-step on the client.
- `unnorm_key` — must match the key saved in
  `<run_dir>/dataset_statistics.json` (typically `"new_embodiment"` for
  mixture-trained checkpoints).
- `include_state` — set `true` only for checkpoints trained with
  `datasets.vla_data.include_state: true`; leave `false` for image/language-only
  checkpoints.
- `control_type: joint_position` matches the training config; switch to
  `ee_pose` only if you train a separate ee-pose model and adjust
  `_action_row_to_ebench` to emit `[pos, quat, gripper]` pairs.

## Multi-worker

Currently the bridge handles a single `worker_id`. For parallel evaluation,
launch multiple bridge processes pointing at the same EBench server with
different `ebench_worker_id` values (and, if you want independent GPUs, a
separate StarVLA server per bridge).
