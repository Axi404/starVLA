"""Production submission entry — drive the real RoboChallenge platform.

Same policy construction and per-iter GET-state → infer → POST-action contract
as ``test_with_mock_server.py``; the differences are entirely on the lifecycle
side, not the policy:

  * ``InterfaceClient(user_token)`` (mock=False) → ``api.robochallenge.cn``.
  * The inner loop is delegated to upstream ``robot.job_worker.job_loop``,
    which wraps the same GET/POST cycle with the job-state machine
    (``get_job_status`` / ``start_robot`` / ``wait_for_robot_running`` /
    ``stop_robot``) that the mock server does not implement.

Caveat — ``job_loop`` takes a *single* ``action_type`` for both ``/state.pkl``
and ``/action``.  Only dosw1 satisfies that (joint ↔ joint).  ur5 / arx5
(leftjoint state / leftpos action) and aloha (joint state / pos action)
go through ``DualActionTypeGPUClient``, which re-fetches state inside
``infer`` so the policy sees the right tensor shape.

Usage::

    python examples/RoboChallenge_table30v2/eval_files/run_demo.py \\
        --user_token <token> --submission_id <sub_id> \\
        --checkpoint .../flat_steps_*.pt --robot_tag dosw1 \\
        --rc_repo $HOME/playground/Code/RoboChallengeInference \\
        --prompt "Fold the T-shirts ..." --duration 0.05
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from examples.RoboChallenge_table30v2.eval_files.model2robochallenge_interface import (
    ROBOT_SPECS,
    RoboChallengePolicy,
)


def _add_rc_repo_to_path(rc_repo: str) -> None:
    rc_repo = os.path.abspath(os.path.expanduser(rc_repo))
    if not os.path.isdir(rc_repo):
        raise FileNotFoundError(f"--rc_repo not found: {rc_repo}")
    if rc_repo not in sys.path:
        sys.path.insert(0, rc_repo)


class GPUClient:
    """Trivial wrapper expected by upstream ``job_loop``.

    ``job_loop`` calls ``gpu_client.infer(state, prompt=prompt)`` and forwards
    the returned action list to ``post_actions``.  We hand both of these
    straight through to ``RoboChallengePolicy.run_policy``.
    """

    def __init__(self, policy: RoboChallengePolicy) -> None:
        self.policy = policy

    def infer(self, state, prompt=None):
        return self.policy.run_policy(state, prompt=prompt)


class DualActionTypeGPUClient(GPUClient):
    """For specs whose state_action_type ≠ post_action_type (ur5 / arx5 / aloha).

    ``job_loop`` only knows one action_type for both directions.  We pass the
    *post* action_type to ``job_loop`` so ``post_actions`` is correct, then
    re-fetch state with our preferred ``state_action_type`` inside ``infer``.
    Costs one extra GET per iter.

    Note: ``job_loop`` already gated the outer state to ``"normal"`` before
    calling us, but our re-fetch can race the platform's robot-state machine.
    We deliberately don't retry — if the second GET returns ``"abnormal"`` /
    ``"size_none"`` then ``run_policy`` will fail loud (KeyError on missing
    camera or shape mismatch), which surfaces upstream rather than silently
    posting a bad action chunk.
    """

    def __init__(self, policy: RoboChallengePolicy, client, image_size) -> None:
        super().__init__(policy)
        self.client = client
        self.image_size = image_size

    def infer(self, _state, prompt=None):
        s = self.client.get_state(
            self.image_size, self.policy.image_type, self.policy.state_action_type
        )
        return self.policy.run_policy(s, prompt=prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_token", required=True, help="API token from robochallenge.cn account.")
    parser.add_argument("--submission_id", required=True, help="Submission id from the platform detail page.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--robot_tag", default="dosw1", choices=list(ROBOT_SPECS))
    parser.add_argument("--prompt", default="perform the task")
    parser.add_argument(
        "--rc_repo",
        default=os.environ.get(
            "ROBOCHALLENGE_INFERENCE_PATH",
            os.path.expanduser("~/playground/Code/RoboChallengeInference"),
        ),
        help="Path to the cloned RoboChallengeInference repo (cvpr branch).",
    )
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--n_action_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action_mode", default=None, choices=("abs", "rel", "delta"),
                        help="Override ckpt config.yaml's action_mode (default: auto from yaml).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("rc_demo")

    _add_rc_repo_to_path(args.rc_repo)
    from robot.interface_client import InterfaceClient  # type: ignore  # noqa: E402
    from robot.job_worker import job_loop  # type: ignore  # noqa: E402

    policy = RoboChallengePolicy(
        checkpoint_path=args.checkpoint,
        robot_tag=args.robot_tag,
        n_action_steps=args.n_action_steps,
        device=args.device,
        action_mode=args.action_mode,
    )
    spec = policy.spec

    image_size = [224, 224]
    client = InterfaceClient(args.user_token)  # mock=False → api.robochallenge.cn

    if spec.state_action_type == spec.post_action_type:
        gpu_client = GPUClient(policy)
        action_type = spec.state_action_type
    else:
        # ur5 / arx5: state=leftjoint, post=leftpos.  Pass the post action_type
        # to job_loop so post_actions is correct; the GPUClient subclass
        # re-fetches state internally with the policy's preferred type.
        logger.warning(
            "spec state_action_type=%r ≠ post_action_type=%r — using "
            "DualActionTypeGPUClient (extra GET per iter).",
            spec.state_action_type, spec.post_action_type,
        )
        gpu_client = DualActionTypeGPUClient(policy, client, image_size)
        action_type = spec.post_action_type

    logger.info(
        "submission_id=%s robot_tag=%s image_type=%s action_type=%s duration=%s",
        args.submission_id, args.robot_tag, spec.image_types, action_type, args.duration,
    )

    job_loop(
        client,
        gpu_client,
        args.submission_id,
        image_size,
        spec.image_types,
        action_type,
        args.duration,
    )


if __name__ == "__main__":
    main()
