#!/usr/bin/env python3
"""Run held-out eval panel against an arbitrary HF model ID (no
trained final_model required). Spins a fresh pod, starts shared vllm
pointed at the HF model, runs each task's evaluate.py, prints summary,
tears down.

Usage:
    PYTHONPATH=. python src/runpod_backend/heldout_test.py \\
        --model Qwen/Qwen3-1.7B --limit 30 --tasks moral_foundations,sycophancy_sharma

    # Or all tasks (default):
    PYTHONPATH=. python src/runpod_backend/heldout_test.py \\
        --model Qwen/Qwen3-1.7B --limit 30

Use Qwen/Qwen3-1.7B (instruction-tuned) instead of -Base for tasks that
require following instructions; the base model produces garbage on
preference / sycophancy / political-bias prompts.
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

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from src.runpod_backend.runpod_environment import RunpodEnvironment
from src.runpod_backend.agent_run import (
    SHARED_VLLM_NAME,
    SHARED_VLLM_PORT,
    start_shared_vllm,
    stop_shared_vllm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("heldout_test")

# Tasks that talk to an external judge (Anthropic) — need ANTHROPIC_API_KEY.
JUDGE_TASKS = {
    "political_bias_openai",
    "spiralbench_mini",
    "sycophancy_aisi",
    "sycophancy_slava",
}

# Tasks skipped by default and why:
# - political_bias_openai, sycophancy_aisi: require running
#   generate_prompts.py first (which calls Anthropic API to materialise
#   prompts.jsonl). Not idempotent or free, so left out of the default
#   panel; pre-generate + commit prompts.jsonl then opt back in via
#   --tasks.
# capability_* tasks are now supported via HELDOUT_PTB_TASKS_DIR env
# override on _delegate.py — they re-run the PTB benchmarks against the
# (potentially trained) model so we get a baseline-vs-trained delta in
# the same heldout panel run.
SKIP_BY_DEFAULT = {
    "political_bias_openai",
    "sycophancy_aisi",
}


def discover_tasks() -> list[str]:
    tasks_dir = REPO_ROOT / "src/heldout_evals/tasks"
    return sorted(
        d.name
        for d in tasks_dir.iterdir()
        if d.is_dir() and (d / "evaluate.py").is_file()
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--tasks",
        default="",
        help="comma-separated task names; empty = all",
    )
    parser.add_argument(
        "--skip-judge-tasks",
        action="store_true",
        help="skip tasks that need ANTHROPIC_API_KEY",
    )
    args = parser.parse_args()

    all_tasks = discover_tasks()
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        unknown = wanted - set(all_tasks)
        if unknown:
            raise SystemExit(f"unknown tasks: {sorted(unknown)}; have: {all_tasks}")
        tasks = [t for t in all_tasks if t in wanted]
    else:
        tasks = [t for t in all_tasks if t not in SKIP_BY_DEFAULT]
    if args.skip_judge_tasks:
        tasks = [t for t in tasks if t not in JUDGE_TASKS]

    log.info(f"running {len(tasks)} tasks: {tasks}")

    run_dir = REPO_ROOT / "jobs" / "runs" / f"_heldout_test_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = TrialPaths(trial_dir=run_dir / "_trial")
    (run_dir / "_trial").mkdir(parents=True, exist_ok=True)

    task_env_config = EnvironmentConfig(
        gpus=1, cpus=8, memory_mb=65536, storage_mb=102400,
        build_timeout_sec=1800.0, allow_internet=True,
    )
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name="heldout-test",
        session_id=f"heldout-test-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        suppress_override_warnings=True,
    )

    results: dict[str, dict] = {}
    try:
        await env.start(force_build=False)

        log.info("uploading src/heldout_evals/")
        await env.upload_dir(str(REPO_ROOT / "src/heldout_evals"), "/workspace/heldout_evals")
        log.info("uploading src/eval/templates/")
        await env.upload_dir(str(REPO_ROOT / "src/eval/templates"), "/workspace/heldout_evals_templates")
        # Capability_* tasks delegate to PTB tasks under src/eval/tasks
        log.info("uploading src/eval/tasks/")
        await env.upload_dir(str(REPO_ROOT / "src/eval/tasks"), "/workspace/heldout_eval_tasks")

        vllm_url = await start_shared_vllm(
            env,
            model_path=args.model,
            chat_template_remote="/workspace/heldout_evals_templates/qwen3.jinja",
            label="vllm-heldout",
        )
        if not vllm_url:
            raise RuntimeError("vllm start failed")

        # Tasks that score via inspect_evals model_graded_* (coconot,
        # strong_reject, moru, sycophancy_sharma) call get_model(role="grader")
        # and default to openai/gpt-4o if unset. Forward both Anthropic and
        # OpenAI keys so judges work regardless of provider, plus base URLs
        # for proxied access. Without this, the eval runs but every sample
        # returns no score → eval_out[0].results is None → AttributeError.
        # capability_* tasks delegate via _delegate.py which expects PTB
        # tasks + templates at REPO_ROOT-relative paths. Override via env
        # so it points at our /workspace/ upload layout instead.
        forwarded_env = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", ""),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
            "HELDOUT_PTB_TASKS_DIR": "/workspace/heldout_eval_tasks",
            "HELDOUT_TEMPLATES_DIR": "/workspace/heldout_evals_templates",
        }
        env_export_str = "; ".join(
            f"export {k}={shlex.quote(v)}" for k, v in forwarded_env.items() if v
        )
        for task_name in tasks:
            log.info(f"=== {task_name} ===")
            out_path = f"/workspace/heldout_evals/tasks/{task_name}/_metrics.json"
            # PYTHONPATH=/workspace/heldout_evals lets sycophancy_slava (and any
            # other task that does `from judge.haiku_judge import HaikuJudge`)
            # resolve heldout_evals/judge/ from any cwd.
            cmd = (
                f"cd /workspace/heldout_evals/tasks/{task_name} && "
                "export HF_HOME=/workspace/hf-cache; "
                "export VLLM_LOGGING_LEVEL=DEBUG; "
                "export PYTHONPATH=/workspace/heldout_evals:${PYTHONPATH:-}; "
                f"{env_export_str}; "
                f"python3 evaluate.py "
                f"--model-path {shlex.quote(args.model)} "
                f"--templates-dir /workspace/heldout_evals_templates/ "
                f"--limit {args.limit} "
                f"--vllm-base-url http://localhost:{SHARED_VLLM_PORT}/v1 "
                f"--vllm-served-name {SHARED_VLLM_NAME} "
                f"--json-output-file {out_path} "
                f"&& cat {out_path}"
            )
            # Run eval THEN read metrics file in a separate cat command. The
            # previous shape (`evaluate.py && cat metrics.json`) glued eval
            # progress output and the metrics JSON together in one stdout
            # stream, defeating any rfind/regex-based extraction. Two-phase
            # exec keeps stdout clean: phase 2's stdout = the file content
            # verbatim. Drop `&& cat ...` from cmd and read out_path here.
            cmd = cmd.rsplit("&&", 1)[0].rstrip()
            r = await env.exec(cmd, timeout_sec=1800)
            if r.return_code != 0:
                log.error(f"  {task_name} FAILED rc={r.return_code} stderr_tail={(r.stderr or '')[-500:]}")
                results[task_name] = {"error": f"rc={r.return_code}", "stderr_tail": (r.stderr or "")[-500:]}
                continue
            try:
                cat_r = await env.exec(f"cat {out_path}", timeout_sec=30)
                if cat_r.return_code != 0:
                    raise RuntimeError(f"cat failed rc={cat_r.return_code}")
                metrics = json.loads((cat_r.stdout or "").strip())
                # Drop verbose row-level details from per-task output; keep
                # top-level numeric metrics.
                summary = {k: v for k, v in metrics.items() if not isinstance(v, list)}
                results[task_name] = summary
                log.info(f"  metrics: {summary}")
            except Exception as e:
                log.error(f"  parse failed: {e}; stdout_tail={(r.stdout or '')[-500:]}")
                results[task_name] = {"error": f"parse: {e}", "stdout_tail": (r.stdout or "")[-500:]}

        await stop_shared_vllm(env, label="vllm-heldout")
    finally:
        try:
            await env.stop()
        except Exception:
            pass

    # Persist + print
    out_file = run_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump({"model": args.model, "limit": args.limit, "results": results}, f, indent=2)
    print()
    print(f"=== heldout panel: model={args.model} limit={args.limit} ===")
    for t, m in results.items():
        if "error" in m:
            print(f"  {t:30s} ERROR  {m['error']}")
        else:
            # Print up to 4 numeric metrics per task
            keys = [k for k, v in m.items() if isinstance(v, (int, float))][:4]
            stats = "  ".join(f"{k}={m[k]:.4f}" for k in keys)
            print(f"  {t:30s} {stats}")
    print(f"  artefacts: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
