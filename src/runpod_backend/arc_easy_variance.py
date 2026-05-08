#!/usr/bin/env python3
"""Run arc_easy N times against the same shared vllm + base model to
measure inter-run accuracy variance. Sanity-checks whether the -0.233
delta seen in recent dry-runs is sampling noise or a real bug.

Usage:
    PYTHONPATH=. python src/runpod_backend/arc_easy_variance.py \\
        --student Qwen/Qwen3-1.7B-Base --limit 30 --repeats 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shlex
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from src.runpod_backend.runpod_environment import DEFAULT_IMAGE, RunpodEnvironment
from src.runpod_backend.agent_run import (
    SHARED_VLLM_API_KEY,
    SHARED_VLLM_NAME,
    SHARED_VLLM_PORT,
    start_shared_vllm,
    stop_shared_vllm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("arc_easy_variance")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    run_dir = REPO_ROOT / "jobs" / "runs" / f"_arc_easy_variance_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = TrialPaths(trial_dir=run_dir / "_trial")
    (run_dir / "_trial").mkdir(parents=True, exist_ok=True)

    task_env_config = EnvironmentConfig(
        gpus=1, cpus=8, memory_mb=65536, storage_mb=102400,
        build_timeout_sec=1800.0, allow_internet=True,
    )
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name="arc-easy-variance",
        session_id=f"arc-var-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        suppress_override_warnings=True,
    )

    accuracies: list[float] = []
    try:
        await env.start(force_build=False)
        log.info("env ready; uploading task")
        await env.upload_dir(str(REPO_ROOT / "src/eval/tasks/arc_easy"), "/workspace/ptb_eval/arc_easy")
        await env.upload_dir(str(REPO_ROOT / "src/eval/templates"), "/workspace/ptb_eval/templates")

        vllm_url = await start_shared_vllm(
            env,
            model_path=args.student,
            chat_template_remote="/workspace/ptb_eval/templates/qwen3.jinja",
            label="vllm-arc",
        )
        if not vllm_url:
            raise RuntimeError("vllm start failed")

        for i in range(args.repeats):
            log.info(f"=== arc_easy repeat {i+1}/{args.repeats} ===")
            out_path = f"/workspace/ptb_eval/arc_easy/metrics_var_{i}.json"
            cmd = (
                "cd /workspace/ptb_eval/arc_easy && "
                "export HF_HOME=/workspace/hf-cache; "
                "export VLLM_LOGGING_LEVEL=DEBUG; "
                f"python3 evaluate.py "
                f"--model-path {shlex.quote(args.student)} "
                f"--templates-dir /workspace/ptb_eval/templates/ "
                f"--limit {args.limit} "
                f"--max-connections 8 "
                f"--vllm-base-url http://localhost:{SHARED_VLLM_PORT}/v1 "
                f"--vllm-served-name {SHARED_VLLM_NAME} "
                f"--json-output-file {out_path} "
                "&& cat " + out_path
            )
            r = await env.exec(cmd, timeout_sec=600)
            if r.return_code != 0:
                log.error(f"repeat {i+1} failed rc={r.return_code} stderr={(r.stderr or '')[-500:]}")
                accuracies.append(float("nan"))
                continue
            try:
                metrics = json.loads((r.stdout or "").strip().splitlines()[-1])
                acc = float(metrics["accuracy"])
                accuracies.append(acc)
                log.info(f"  accuracy={acc:.4f} stderr={metrics.get('stderr', 'n/a')}")
            except Exception as e:
                log.error(f"parse failed: {e}; stdout={r.stdout[-500:]}")
                accuracies.append(float("nan"))

        await stop_shared_vllm(env, label="vllm-arc")
    finally:
        try:
            await env.stop()
        except Exception:
            pass

    print()
    print(f"=== arc_easy variance test: n={args.limit} samples × {args.repeats} repeats ===")
    print(f"  model: {args.student}")
    for i, a in enumerate(accuracies):
        print(f"  run {i+1}: {a:.4f}")
    valid = [a for a in accuracies if a == a]  # filter NaN
    if len(valid) >= 2:
        mean = sum(valid) / len(valid)
        var = sum((a - mean) ** 2 for a in valid) / (len(valid) - 1)
        std = var ** 0.5
        print(f"  mean: {mean:.4f}  std: {std:.4f}  range: {min(valid):.4f}..{max(valid):.4f}")
    print(f"  artefacts: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
