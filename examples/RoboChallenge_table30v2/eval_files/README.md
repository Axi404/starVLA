# RoboChallenge Table30v2 — Evaluation Pipeline

We follow the upstream
[RoboChallengeInference (cvpr branch)](https://github.com/RoboChallenge/RoboChallengeInference/tree/cvpr)
step-by-step approach: **first self-test offline, then drive the mock server,
then submit to the real platform**.

```text
local_self_test.py        ← step 1: dummy obs, no network
test_with_mock_server.py  ← step 2: real protocol against upstream mock_robot_server.py
production_demo.py        ← step 3 (TODO): wire to demo.py / job_loop on the real challenge
model2robochallenge_interface.py
                          ← shared `RoboChallengePolicy` (replaces upstream `DummyPolicy`)
```

## I/O contract (with the RC server)

| Direction | Endpoint | Robot=`ur5` |
|---|---|---|
| GET state | `/state.pkl` with `image_type=cam_global,cam_arm` and `action_type=leftjoint` | returns `{images: {cam: PNG}, action: joint(6)+gripper(1) = 7-d, state, pending_actions, timestamp}` |
| POST action | `/action` with `action_type=leftpos` | body `{actions: [[ee_pose(7) + gripper(1)], ...], duration: 0.05}` |

We deliberately use **two different action_types per cycle**: `leftjoint` to
fetch the joint state our model was trained on, `leftpos` to post the 8-d
ee-pose actions our model produces. The mock server treats each request
independently so this is fine.

> If you ever retrain with `state` based on ee-pose, set both
> `state_action_type` and `post_action_type` to `leftpos` in `ROBOT_SPECS`.

Robot specs (`image_types`, `state_dim`, `action_dim`) live in
`model2robochallenge_interface.py::ROBOT_SPECS` and must stay in sync with the
training data layout described in
`examples/RoboChallenge_table30v2/train_files/data_registry/data_config.py`.

## Step 1 — local self-test (no network)

Loads the checkpoint, runs forward on a synthetic observation, prints the
action chunk + latency. **Run this first** any time you swap the checkpoint or
the framework.

```bash
bash examples/RoboChallenge_table30v2/eval_files/run_self_test.sh
# or override the defaults:
CKPT=playground/Checkpoints/.../steps_100_pytorch_model.pt \
ROBOT_TAG=ur5 PROMPT="shred the paper" \
bash examples/RoboChallenge_table30v2/eval_files/run_self_test.sh --n_runs 5
```

Expected: prints `(8, 8)` action chunk + ~hundreds-of-ms latency.

## Step 2 — drive the upstream mock server

1. Clone upstream **once** (somewhere outside the starVLA repo so it isn't tracked)::

    ```bash
    mkdir -p ~/playground/Code && cd ~/playground/Code
    git clone -b cvpr https://github.com/RoboChallenge/RoboChallengeInference.git
    cd RoboChallengeInference && pip install -r requirements.txt
    ```

2. **Pick an episode** to replay and edit `mock_server/mock_settings.py`:

    ```python
    ROBOT_TAG = 'ur5'
    RECORD_DATA_DIR = '../20260413/ur5/arrange_fruits'   # or any ur5 task directory
    ```

3. Start the mock server (port 9098):

    ```bash
    cd mock_server && python3 mock_robot_server.py
    ```

4. In a **second** shell, point our policy at it:

    ```bash
    bash examples/RoboChallenge_table30v2/eval_files/run_test_with_mock.sh
    # or:
    CKPT=playground/Checkpoints/.../steps_100_pytorch_model.pt \
    ROBOT_TAG=ur5 \
    RC_REPO=$HOME/playground/Code/RoboChallengeInference \
    bash examples/RoboChallenge_table30v2/eval_files/run_test_with_mock.sh
    ```

You should see one inference per loop with shape `(8, 8)` actions being POSTed.

### Step 2b — DOSW1 dual-arm

Upstream calls this robot **`w1`** (not `dosw1`). Use:

```python
# mock_server/mock_settings.py
ROBOT_TAG = 'w1'
RECORD_DATA_DIR = '/home/gaoning/repos/private/rc2/fold_the_clothes'
# (absolute path is fine — the upstream sample '../20260413/w1/...' is just a relative example)
```

Then on our side, `ROBOT_SPECS["dosw1"]` uses
`state_action_type=post_action_type="joint"` — the upstream W1 robot's
dual-arm joint literal is `"joint"`, so the server returns one 14-d state
vector (`[L_j×6, L_grip, R_j×6, R_grip]`) and we POST one 14-d action vector
in the same layout per step.

```bash
CKPT=results/QwenOFT-all-150k-50chunk-rc2-dosw1-q99/checkpoints/flat_steps_150000_pytorch_model.pt \
ROBOT_TAG=dosw1 N_ACTION_STEPS=50 \
bash examples/RoboChallenge_table30v2/eval_files/run_test_with_mock.sh
```

Expected: per-iteration log `infer=…ms net=…ms pending=0 first_action=[14 floats]`.

## Step 3 — production submission (TODO)

Mirror upstream `demo.py` + `robot/job_worker.py::job_loop`. Replace
`DummyPolicy` with `RoboChallengePolicy(checkpoint, robot_tag=...)`. We have
not yet wired this because we are still validating a 100-step smoke
checkpoint — once a real run is trained, drop in:

```python
from examples.RoboChallenge_table30v2.eval_files.model2robochallenge_interface import RoboChallengePolicy
policy = RoboChallengePolicy(args.checkpoint, robot_tag="ur5")
gpu_client = GPUClient(policy)
job_loop(client, gpu_client, args.submission_id, [224, 224],
         policy.image_type, policy.state_action_type, duration)
```

Caveat for the real `job_loop`: it uses a single `action_type` for both get/post.
You'll either need to (a) use the same action_type both ways (re-train with
matching state) or (b) sub-class `GPUClient.infer` to fetch joint state itself.
