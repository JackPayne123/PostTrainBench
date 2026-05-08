#!/usr/bin/env python3
"""Rerun post-eval against an already-trained LoRA adapter.

Spins a fresh pod, uploads a local adapter dir + the PTB eval task,
starts vllm with --enable-lora pointing at the adapter, runs post-eval
through the same shared-vllm path agent_run uses, writes metrics to
the run dir alongside metrics_pre.json.

Use after a partial agent_run where training succeeded but the post
phase failed (e.g. the --max-lora-rank cap, or any other transient
post-side bug). The agent step is the expensive part; if you have the
adapter locally (adapter_safety/) or on the runpod volume, you can
salvage the run for the cost of a 5-min pod.

Usage:
    PYTHONPATH=. python src/runpod_backend/rerun_post.py \\
        --run-dir jobs/runs/2026-05-09_08-17_A_..._seed0 \\
        --benchmark gsm8k \\
        --student Qwen/Qwen3-1.7B-Base \\
        --limit 30
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
    EVAL_DIRS,
    start_shared_vllm,
    stop_shared_vllm,
    run_eval,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("rerun_post")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Existing jobs/runs/<...>/ that has adapter_safety/ from a prior run",
    )
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--benchmark", default="gsm8k", choices=sorted(EVAL_DIRS))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    if not args.run_dir.is_absolute():
        args.run_dir = (REPO_ROOT / args.run_dir).resolve()
    adapter_dir = args.run_dir / "adapter_safety"
    if not adapter_dir.is_dir():
        raise SystemExit(f"adapter_safety/ not found at {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(f"adapter_config.json missing in {adapter_dir}")

    log.info(f"adapter: {adapter_dir} ({sum(p.stat().st_size for p in adapter_dir.rglob('*') if p.is_file()) // 1024 // 1024} MB)")

    trial_paths = TrialPaths(trial_dir=args.run_dir / "_rerun_post_trial")
    (args.run_dir / "_rerun_post_trial").mkdir(parents=True, exist_ok=True)

    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name=f"rerun-post-{args.benchmark}",
        session_id=f"rerun-post-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=8, memory_mb=65536, storage_mb=102400,
            build_timeout_sec=1800.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )

    try:
        await env.start(force_build=False)
        log.info("env ready; uploading adapter + task dir + templates")
        remote_adapter = "/workspace/rerun_adapter"
        remote_eval_root = "/workspace/ptb_eval"
        await env.upload_dir(str(adapter_dir), remote_adapter)
        await env.upload_dir(
            str(REPO_ROOT / f"src/eval/tasks/{args.benchmark}"),
            f"{remote_eval_root}/{args.benchmark}",
        )
        await env.upload_dir(
            str(REPO_ROOT / "src/eval/templates"),
            f"{remote_eval_root}/templates",
        )

        log.info("starting vllm-post with --enable-lora")
        vllm_url = await start_shared_vllm(
            env,
            model_path=args.student,
            chat_template_remote=f"{remote_eval_root}/templates/qwen3.jinja",
            label="vllm-rerun",
            lora_adapter_path=remote_adapter,
        )
        if not vllm_url:
            raise RuntimeError("vllm-post start failed")

        log.info(f"=== POST-EVAL ({args.benchmark}) ===")
        out_metrics = args.run_dir / "metrics_post.json"
        post_metrics = await run_eval(
            env,
            benchmark=args.benchmark,
            model_path=args.student,
            limit=args.limit,
            label="post",
            remote_eval_root=remote_eval_root,
            out_metrics=out_metrics,
            watch=True,
            skip_templates_upload=True,
            vllm_base_url=vllm_url,
            vllm_served_name=SHARED_VLLM_NAME,
        )
        log.info(f"post metrics: {post_metrics}")

        await stop_shared_vllm(env, label="vllm-rerun")
    finally:
        try:
            await env.stop()
        except Exception:
            pass

    # Compute delta against existing metrics_pre.json if available
    pre_path = args.run_dir / "metrics_pre.json"
    summary = {
        "model": args.student,
        "benchmark": args.benchmark,
        "limit": args.limit,
        "post": post_metrics,
        "pre": json.loads(pre_path.read_text()) if pre_path.is_file() else None,
    }
    if summary["pre"] and post_metrics:
        summary["delta_accuracy"] = post_metrics["accuracy"] - summary["pre"]["accuracy"]

    out = args.run_dir / "rerun_post_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print(f"rerun_post complete: {args.run_dir.name}")
    print("=" * 60)
    if summary["pre"]:
        print(f"  pre  accuracy: {summary['pre']['accuracy']:.3f}")
    if post_metrics:
        print(f"  post accuracy: {post_metrics['accuracy']:.3f}")
    if summary.get("delta_accuracy") is not None:
        print(f"  delta:         {summary['delta_accuracy']:+.3f}")
    print(f"  saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
