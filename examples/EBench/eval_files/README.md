# EBench Evaluation — Bridge Mode

EBench exposes an HTTP **evaluation server** (default `:8087`), and expects
models to connect via `genmanip_client.EvalClient`. StarVLA, in contrast,
runs inference behind its own **websocket server** (so other benchmarks can
connect as clients). Both sides are servers, so nothing drives the loop.

This directory provides the missing driver — a small **bridge** client that
connects to both:

```
EBench server  ──obs──▶  bridge  ──(image, lang)──▶  StarVLA server
EBench server ◀─action── bridge ◀──normalized_actions──  StarVLA server
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
`starVLA/dataloader/gr00t_lerobot/data_config.py`. The 19-dim action vector is
concatenated in this order:

| slice | dims | meaning |
|---|---|---|
| `[0:6]` | 6 | `action.left_joints` |
| `[6:12]` | 6 | `action.right_joints` |
| `[12:14]` | 2 | `action.left_gripper` (binary) |
| `[14:16]` | 2 | `action.right_gripper` (binary) |
| `[16:19]` | 3 | `action.base_delta` (dx, dy, dyaw) |

EBench expects a **different interleaving** for the arm action:
`(16,) = [left_joints(6), left_gripper(2), right_joints(6), right_gripper(2)]`
plus `base_motion=(3,)`. The bridge re-interleaves per chunk row before
submitting. Binary gripper dims are trained with `mask=False` and are passed
through unchanged (the model already emits ~0 / ~1 values).

Training keys vs eval keys differ (names only, content is the same dataset):

| training (`EBenchConfig.video_keys`) | EBench eval obs              |
|--------------------------------------|------------------------------|
| `video.cam_high`                     | `video.top_camera_view`      |
| `video.cam_left_wrist`               | `video.left_camera_view`     |
| `video.cam_right_wrist`              | `video.right_camera_view`    |

Cross-checked against `InternVLA/tutorials/examples/evaluation/EBench/pi05_client.py`.
The 4th eval camera `video.overlook_camera_view` is unused. The bridge
resizes all three to `image_size` before forwarding to StarVLA.

## Running

1. **Install bridge extras.** `genmanip_client` is assumed already installed
   per the EBench docs; the bridge only needs a couple of lightweight extras:
   ```bash
   pip install -r examples/EBench/eval_files/requirements.txt
   ```

2. **Terminal A — StarVLA server.** Edit the checkpoint path in
   `run_policy_server.sh` and `bridge_config.yml`, then:
   ```bash
   conda activate starvla
   bash examples/EBench/eval_files/run_policy_server.sh
   ```

3. **Terminal B — EBench server.** Launch EBench per its own documentation,
   so the HTTP endpoint (default `http://127.0.0.1:8087`) is up.

4. **Terminal C — bridge.**
   ```bash
   bash examples/EBench/eval_files/run_bridge.sh
   ```

## Configuration cheat-sheet

- `chunk_mode: true` — submit all 50 predicted actions in a single
  `EvalClient.step()` call (server executes them internally). Recommended
  default for throughput.
- `chunk_mode: false` — submit one action per `step()`; useful when you want
  to see each transition in EBench logs.
- `unnorm_key` — must match the key saved in
  `<run_dir>/dataset_statistics.json` (typically `"new_embodiment"` for
  mixture-trained checkpoints).
- `control_type: joint_position` matches the training config; switch to
  `ee_pose` only if you train a separate ee-pose model and adjust
  `_action_row_to_ebench` to emit `[pos, quat, gripper]` pairs.

## Fake mode (EBench-only smoke test)

To verify the bridge ↔ EBench wiring without booting the StarVLA server,
set `fake_mode: true` in `bridge_config.yml`. In this mode the bridge:

- Does not connect to `policy_host:policy_port` and does not load norm stats.
- Calls `genmanip_client.eval_client.fake_action` with EBench's fixed
  `r5a / lift2 / joint_position` combo — the same payload that `gmp eval
  -a r5a -g lift2` produces. It's a zero-delta `is_rel=True` action that
  keeps the robot still.

Run just `run_bridge.sh`; you should see steps advancing on the EBench side
with the robot staying still. Flip back to `fake_mode: false` once the real
policy server is up.

## Multi-worker

Currently the bridge handles a single `worker_id`. For parallel evaluation,
launch multiple bridge processes pointing at the same EBench server with
different `ebench_worker_id` values (and, if you want independent GPUs, a
separate StarVLA server per bridge).
