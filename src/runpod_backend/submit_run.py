#!/usr/bin/env python3
"""Submit a self-driving experiment to RunPod and exit.

Replaces agent_run.py for normal use. The laptop's job ends ~3 minutes
after invoking this script: spin pod, upload run config, drop a START
sentinel, walk away. Pod self-drives the rest:
    pre-eval → agent → post-eval → heldout → rclone-to-Drive → DONE → terminate.

Tail the live run log over SSH any time:
    bash src/runpod_backend/tail_log.sh <run_id>

Check status without SSH (uses a tiny recovery pod):
    PYTHONPATH=. python src/runpod_backend/status_run.py <run_id>

Pull final artifacts from the persistent volume to laptop after DONE:
    PYTHONPATH=. python src/runpod_backend/pull_run.py <run_id>

If the laptop crashes, the lid closes, the wifi drops — none of it
matters. The pod has its own copy of the config and runs to completion
on its own.

See docs/OPERATIONS.md for the full architecture and
~/.claude/plans/sounds-good-add-this-reflective-eagle.md for the
design rationale.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Harbor venv lookup for IDE-time imports; runtime invocation uses the harbor
# python path so this fallback only matters in pytest contexts.
HARBOR_VENV = Path.home() / ".local/share/uv/tools/harbor/lib/python3.13/site-packages"
if HARBOR_VENV.exists() and str(HARBOR_VENV) not in sys.path:
    sys.path.insert(0, str(HARBOR_VENV))

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from src.harbor_adapter.adapter import (
    BENCHMARKS,
    MODELS,
    PostTrainBenchAdapter,
)
from src.runpod_backend.condition_prompts import apply as apply_condition_addendum
from src.runpod_backend.eval_only import EVAL_DIRS
from src.runpod_backend.run_dir import make_run_config, write_json, write_run_config
from src.runpod_backend.runpod_environment import DEFAULT_IMAGE, RunpodEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("submit_run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", required=True, choices=["A", "B", "C", "D", "E", "F"])
    p.add_argument("--teacher", default="claude-opus-4-7",
                   help="agent model arg (passed as AGENT_CONFIG to solve.sh)")
    p.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--benchmark", default="gsm8k", choices=sorted(EVAL_DIRS))
    p.add_argument("--extra-evals", default="",
                   help="comma-separated additional benchmarks to pre+post-eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-h", type=float, default=1.0)
    p.add_argument("--agent", default="claude_non_api_max")
    p.add_argument("--prompt-variant", default="default")
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--no-watch", action="store_true")
    p.add_argument("--keep-pod", action="store_true",
                   help="pod doesn't self-terminate after DONE (debug)")
    p.add_argument("--dry-run", action="store_true",
                   help="run pre-eval + dir scaffold only; skip agent + post-eval")
    p.add_argument("--skip-heldout", action="store_true")
    p.add_argument("--skip-pre-eval", action="store_true",
                   help="Pod skips pre-eval and treats baselines as the "
                        "pre values. summary.json's `pre` + `delta` come "
                        "out as None; backfill via "
                        "`scripts/compute_deltas.py <run_id>` once baselines "
                        "for this model+limit exist in the repo.")
    p.add_argument("--no-drive-upload", action="store_true",
                   help="pod skips rclone-to-Drive at end (debug)")
    return p.parse_args()


def render_prompt(
    *,
    benchmark: str,
    student_model: str,
    condition: str,
    time_budget_h: float,
    agent: str,
    run_dir: Path,
) -> Path:
    """Render instruction.md → prompt.txt with the condition addendum.

    This is the same logic agent_run.stage_agent_workspace ran before
    upload. We do it laptop-side so the agent's prompt is committed
    to the run dir before we even spin a pod — easier review + repro.
    """
    if benchmark not in BENCHMARKS:
        raise SystemExit(f"unknown benchmark '{benchmark}'; valid: {sorted(BENCHMARKS)}")
    benchmark_info = BENCHMARKS[benchmark]
    model_info = next(
        (m for m in MODELS.values() if m.model_id == student_model),
        None,
    )
    if model_info is None:
        raise SystemExit(
            f"student model '{student_model}' not in MODELS; "
            f"valid HF ids: {[m.model_id for m in MODELS.values()]}"
        )
    adapter = PostTrainBenchAdapter(
        output_dir=run_dir / "_unused",
        # Pass the actual budget through (was max(1, int(time_budget_h))
        # which rounded 0.5h → 1h, making the rendered prompt.txt say
        # "1 hour" while pipeline enforced the real 0.5h — caught on
        # 2026-05-11 F-run analysis). Format trims trailing zeros so 1.0
        # renders as "1" and 0.5 renders as "0.5".
        num_hours=time_budget_h,
        include_claude_clause=(agent.startswith("claude")),
    )
    adapter.generate_instruction(run_dir, model_info, benchmark_info, benchmark)
    rendered = (run_dir / "instruction.md").read_text()
    with_addendum = apply_condition_addendum(rendered, condition)
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(with_addendum)
    # instruction.md has served its purpose; keep the rendered prompt only.
    (run_dir / "instruction.md").unlink(missing_ok=True)
    return prompt_path


def find_oauth_token() -> str | None:
    """Look for the Claude OAuth token. Mirrored from agent_run.py."""
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    for p in [
        Path.home() / ".runpod" / "secrets" / "claude_oauth_token",
        Path.home() / ".claude_oauth_token",
    ]:
        if p.exists():
            tok = p.read_text().strip()
            if tok:
                return tok
    return None


def git_sha() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


async def main() -> None:
    args = parse_args()

    # Build run config + dir
    cfg, dirname = make_run_config(
        condition=args.condition,
        teacher_model=args.teacher,
        student_model=args.student,
        benchmark=args.benchmark,
        prompt_variant=args.prompt_variant,
        seed=args.seed,
        time_budget_h=args.time_budget_h,
        agent=args.agent,
        agent_config=args.teacher,
        base_image=DEFAULT_IMAGE,
        git_sha=git_sha(),
        extra={
            "extra_evals": args.extra_evals,
            "limit": args.limit,
            "skip_heldout": args.skip_heldout,
            "skip_pre_eval": args.skip_pre_eval,
            "dry_run": args.dry_run,
            "keep_pod": args.keep_pod,
            "no_drive_upload": args.no_drive_upload,
        },
    )
    run_dir = REPO_ROOT / "jobs" / "runs" / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir, cfg)
    log.info(f"=== run dir: {run_dir} ===")

    # Render prompt locally so it's reviewable before pod boot
    render_prompt(
        benchmark=args.benchmark,
        student_model=args.student,
        condition=args.condition,
        time_budget_h=args.time_budget_h,
        agent=args.agent,
        run_dir=run_dir,
    )

    oauth_token = find_oauth_token()
    if oauth_token is None and args.agent.startswith("claude"):
        raise SystemExit(
            "CLAUDE_CODE_OAUTH_TOKEN not found in env or "
            "~/.runpod/secrets/claude_oauth_token. Generate with "
            "`claude setup-token` then save to one of those paths."
        )

    # Secrets to forward into the pod's environment. Pod's run_experiment.py
    # reads these for: rclone Drive auth (RUNPOD_API_KEY), self-terminate
    # (RUNPOD_API_KEY + RUNPOD_POD_ID), agent (CLAUDE_CODE_OAUTH_TOKEN),
    # judge calls (ANTHROPIC_API_KEY, OPENAI_API_KEY), HF dataset auth (HF_TOKEN).
    pod_env = {
        "RUN_ID": dirname,
        "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", ""),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "CLAUDE_CODE_OAUTH_TOKEN": oauth_token or "",
        # Unified grader for inspect_evals model_graded_qa scorers
        # (coconot, strong_reject, sycophancy_sharma). Reads
        # ANTHROPIC_API_KEY from the same env. moru passes its own
        # explicit grader via task_args; healthbench + arenahardwriting
        # still use gpt-5-mini (separate code path).
        "INSPECT_GRADER_MODEL": os.environ.get(
            "INSPECT_GRADER_MODEL", "anthropic/claude-haiku-4-5"
        ),
        "POD_KEEP_ALIVE": "1" if args.keep_pod else "0",
        "POD_NO_DRIVE_UPLOAD": "1" if args.no_drive_upload else "0",
    }

    # Spin pod
    task_env_config = EnvironmentConfig(
        gpus=1, cpus=8, memory_mb=65536, storage_mb=102400,
        build_timeout_sec=1800.0, allow_internet=True,
    )
    trial_paths = TrialPaths(trial_dir=run_dir / "_harbor_trial")
    (run_dir / "_harbor_trial").mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name=f"agent-run-{cfg.condition}-{cfg.seed}",
        session_id=dirname,
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        suppress_override_warnings=True,
        pod_env=pod_env,
    )

    t0 = time.time()
    try:
        await env.start(force_build=False)
        log.info(f"Pod ready ({time.time() - t0:.0f}s)")
        # Stash pod meta locally so status_run.py / tail_log.sh can find it.
        pod_info = {
            "pod_id": env._pod_id,
            "ssh_host": env._ssh_host,
            "ssh_port": env._ssh_port,
            "image": DEFAULT_IMAGE,
            "run_id": dirname,
        }
        write_json(run_dir / "pod_meta.json", pod_info)

        # Make the volume run dir + upload our prepared config + prompt.
        # The pod's startup_hook.sh polls /workspace/runs/$RUN_ID/START
        # and execs /opt/run_experiment.py the moment it appears.
        remote_run_dir = f"/workspace/runs/{dirname}"
        await env.exec(
            f"mkdir -p {shlex.quote(remote_run_dir)}",
            timeout_sec=30,
        )
        log.info(f"uploading run dir -> {remote_run_dir}")
        # Upload config.json + prompt.txt + pod_meta.json. Skip Harbor's
        # _harbor_trial dir and anything else we don't need on the pod.
        for fname in ("config.json", "prompt.txt", "pod_meta.json"):
            local_f = run_dir / fname
            if local_f.exists():
                await env.exec(
                    f"cat > {remote_run_dir}/{fname} << 'PTBEOF'\n"
                    + local_f.read_text()
                    + "\nPTBEOF",
                    timeout_sec=60,
                )

        # Write POD_ID file (pod-side script reads this; faster + more
        # reliable than relying on a Runpod-injected env var that may or
        # may not exist depending on platform).
        await env.exec(
            f"echo {shlex.quote(env._pod_id or '')} > {remote_run_dir}/POD_ID",
            timeout_sec=15,
        )

        # Drop the START sentinel and explicitly launch the startup hook
        # in a detached tmux session named "run". The hook polls for
        # START (touched here) and then execs /opt/run_experiment.py.
        # We can't use ENTRYPOINT-injection in the Dockerfile because the
        # runpod/pytorch base's startup machinery doesn't tolerate a
        # wrapper (verified :8 — telemetry goes to "exited"). One SSH
        # exec at submission time is the cheapest workaround; submit_run
        # still exits immediately after, so the laptop has zero ongoing
        # involvement in the experiment.
        await env.exec(
            f"touch {remote_run_dir}/START",
            timeout_sec=15,
        )
        # Detach + nohup the startup hook so SSH returns immediately;
        # the hook handles its own tmux session for run_experiment.py.
        # Inject all secrets explicitly via env=...; sshd does NOT
        # inherit the Runpod-injected container env (the podCreate
        # `env` field reaches the container's PID 1 but not new SSH
        # sessions, which run with sshd's default env).
        env_vars = {k: v for k, v in pod_env.items() if v}
        env_exports = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in env_vars.items()
        )
        await env.exec(
            f"{env_exports} setsid nohup bash /opt/startup_hook.sh "
            "> /var/log/startup_hook.boot.log 2>&1 < /dev/null & "
            "disown 2>/dev/null || true; "
            "echo launched",
            timeout_sec=30,
        )
        log.info(f"=== submitted: {dirname} ===")
        log.info(f"  pod: {env._pod_id} ({env._ssh_host}:{env._ssh_port})")
        log.info(f"  volume path: {remote_run_dir}")
        log.info(f"  drive folder: drive:{dirname} (root_folder_id = experiments/)")
        log.info(f"  laptop run_dir: {run_dir}")
        log.info("")
        log.info("Pod is now self-driving. Walk away. To check progress:")
        log.info(f"  tail log:  bash src/runpod_backend/tail_log.sh {dirname}")
        log.info(f"  status:    PYTHONPATH=. python src/runpod_backend/status_run.py {dirname}")
        log.info(f"  pull:      PYTHONPATH=. python src/runpod_backend/pull_run.py {dirname}")
    except Exception:
        log.exception("submit failed")
        raise
    finally:
        # We DO NOT call env.stop() — the pod must keep running for the
        # experiment. The pod self-terminates from inside run_experiment.py
        # after writing DONE + uploading to Drive.
        pass


if __name__ == "__main__":
    asyncio.run(main())
