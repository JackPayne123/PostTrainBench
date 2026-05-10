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
    # spiralbench_mini calls target.generate(...) synchronously but
    # inspect_ai's Model.generate is async — every conversation hits
    # AttributeError on `out.completion` (coroutine has no .completion),
    # producing n_failed=N, n=0. Needs a task-level rewrite (asyncio.run
    # wrapper or switch to inspect_ai eval framework). Skipped until then.
    "spiralbench_mini",
}


def discover_tasks() -> list[str]:
    tasks_dir = REPO_ROOT / "src/evals/tasks"
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

        log.info("uploading src/evals/")
        await env.upload_dir(str(REPO_ROOT / "src/evals"), "/workspace/heldout_evals")
        log.info("uploading src/evals/templates/")
        await env.upload_dir(str(REPO_ROOT / "src/evals/templates"), "/workspace/heldout_evals_templates")
        # Post-centralisation (2026-05-11): single src/evals/tasks/ tree
        # with category buckets; runner picks tasks via the registry.
        log.info("uploading src/evals/tasks/")
        await env.upload_dir(str(REPO_ROOT / "src/evals/tasks"), "/workspace/heldout_eval_tasks")

        # Pre-fetch HF datasets that need a specific config / revision and
        # aren't auto-fetched by the task. abstention_bench uses
        # gpqa_diamond at a pinned revision; the surrounding tasks only
        # ever cache gpqa_main, so the eval errors with
        # "Couldn't find cache for Idavidrein/gpqa for config 'gpqa_diamond'".
        # Trigger a download up-front so the rest of the panel is offline-safe.
        log.info("pre-fetching gated/pinned HF datasets")
        # No `| tail -3`: pipe makes shell rc=tail's, masking real failure.
        # Capture full output via env.exec and slice.
        prefetch_cmd = (
            "export HF_HOME=/workspace/hf-cache; "
            f"export HF_TOKEN={shlex.quote(os.environ.get('HF_TOKEN', ''))}; "
            "python3 -c \""
            "import datasets; "
            "datasets.load_dataset("
            "'Idavidrein/gpqa', 'gpqa_diamond', "
            "revision='5233cd1db58884ed0bf678c7c6be731722a23f84')"
            "\" 2>&1"
        )
        pf = await env.exec(prefetch_cmd, timeout_sec=300)
        if pf.return_code != 0:
            log.warning(f"prefetch rc={pf.return_code}: {(pf.stdout or pf.stderr or '')[-500:]}")

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
            task_dir = f"/workspace/heldout_evals/tasks/{task_name}"
            out_path = f"{task_dir}/_metrics.json"
            done_flag = f"{task_dir}/_done.rc"
            log_path = f"{task_dir}/_eval.log"
            # PYTHONPATH=/workspace/heldout_evals lets sycophancy_slava (and any
            # other task that does `from judge.haiku_judge import HaikuJudge`)
            # resolve heldout_evals/judge/ from any cwd.
            eval_cmd = (
                f"cd {task_dir} && "
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
                f"--json-output-file {out_path}"
            )
            # Wrap inner with rc-capture so the polling loop can read the
            # final exit code from the sentinel file.
            inner = f"({eval_cmd}); echo $? > {done_flag}"
            # Detach the eval (setsid+nohup, all FDs closed except the log
            # file) and poll a sentinel file from the foreground. The SSH
            # channel only ever sees the polling loop's output, never any
            # eval subprocess stdout — fixes hangs where SSH waited 15+ min
            # for inspect_ai grandchildren to release inherited FDs.
            # 20min budget — moru is the outlier at ~12-15min when both
            # target and grader run on the same shared vllm; everything
            # else completes in <2min so the headroom is cheap.
            task_budget_s = 1200
            poll_max = task_budget_s // 5
            cmd = (
                f"rm -f {done_flag}; "
                f"setsid nohup bash -c {shlex.quote(inner)} "
                f"> {log_path} 2>&1 < /dev/null & disown 2>/dev/null || true; "
                f"for i in $(seq 1 {poll_max}); do "
                f"  if [ -f {done_flag} ]; then exit $(cat {done_flag}); fi; "
                f"  sleep 5; "
                f"done; "
                f"echo '[poll timeout]'; exit 124"
            )
            # task_budget_s + 20s slop. Polling loop fail-fasts on hang
            # (no inherited-FD-keepalive risk) so we mark task failed.
            r = await env.exec(cmd, timeout_sec=task_budget_s + 20)
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
