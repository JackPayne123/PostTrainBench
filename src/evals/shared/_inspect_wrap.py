"""Helper for tasks that just wrap an upstream `inspect_evals.<x>` task.

Usage:
    # tasks/abstention_bench/evaluate.py
    from _inspect_wrap import run_inspect_eval
    import inspect_evals.abstention_bench  # noqa: F401  (registers task)

    run_inspect_eval("inspect_evals/abstention_bench", default_limit=100)

This collapses the boilerplate from src/eval/tasks/gsm8k/evaluate.py:59-96
to a single call, since the only things that change between wraps are the
import path and the registered task name.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional

from _common import add_standard_args, template_kwargs, write_metrics
from _progress import ProgressTimer


def run_inspect_eval(
    task: str,
    default_limit: int,
    upstream_module: Optional[str] = None,
    task_args: Optional[dict] = None,
) -> None:
    """Run an upstream inspect_evals task against the local model.

    Args:
        task: registered task identifier, e.g. "inspect_evals/sycophancy"
        default_limit: --limit default if user does not override
        upstream_module: dotted module to import for task registration; if
            None, defaults to derived from `task` (drop the registry prefix).
    """
    parser = argparse.ArgumentParser(
        description=f"Heldout wrap of {task}",
    )
    add_standard_args(parser, default_limit=default_limit)
    args = parser.parse_args()

    if upstream_module is None:
        upstream_module = task.replace("/", ".")
    importlib.import_module(upstream_module)

    # Imports here so that --help works even if inspect_ai isn't installed.
    from inspect_ai import eval as inspect_eval
    from inspect_ai.util._display import init_display_type

    init_display_type("plain")

    other_kwargs: dict = {}
    if args.limit is not None and args.limit != -1:
        other_kwargs["limit"] = args.limit

    if getattr(args, "vllm_base_url", None) and getattr(args, "vllm_served_name", None):
        model = f"openai-api/local/{args.vllm_served_name}"
        model_args = {"api_key": "inspectai"}
    else:
        model = f"vllm/{args.model_path}"
        model_args = {"gpu_memory_utilization": args.gpu_memory_utilization}
        model_args.update(template_kwargs(args))

    # Bookend timing for inspect_evals tasks — the inner generation +
    # grader loop is opaque to us (inspect_ai owns it), so we just log
    # start + end with elapsed time. Inspect_ai's own progress bar is
    # suppressed via score_display + log_realtime to keep eval_*.log clean.
    import time as _time
    _bench_name = task.split("/")[-1] if isinstance(task, str) else getattr(task, "__name__", "inspect_task")
    _n = other_kwargs.get("limit", "?")
    print(f"[progress] {_bench_name} inspect_eval   starting (n_samples={_n} max_connections={args.max_connections})",
          flush=True)
    # Enforce a min-tokens floor. Caught on 2026-05-13 Qwen3.5-9B baseline
    # #cac466: upstream inspect_evals/mmlu_0_shot internally caps via
    # max_non_cot_tokens=16, which truncated 92/200 of Qwen3.5's outputs
    # mid-rationale (the model emits a brief explanation before the answer
    # letter, even with thinking off). 128 is plenty for any single-letter
    # MCQ commitment + a short preamble; CoT evals already set 256-4000.
    # Caller can override per-eval via --max-tokens.
    _MIN_TOKEN_FLOOR = 128
    effective_max_tokens = max(args.max_tokens, _MIN_TOKEN_FLOOR)
    if effective_max_tokens != args.max_tokens:
        print(f"[progress] {_bench_name} max_tokens floor: {args.max_tokens} -> "
              f"{effective_max_tokens} (avoids upstream truncation regression)",
              flush=True)
    _t0 = _time.monotonic()
    eval_out = inspect_eval(
        task,
        model=model,
        model_base_url=args.vllm_base_url if (getattr(args, "vllm_base_url", None) and getattr(args, "vllm_served_name", None)) else None,
        model_args=model_args,
        # task_args lets a wrapper override @task parameters (e.g.
        # abstention_bench's hardcoded openrouter grader_model).
        task_args=task_args or {},
        score_display=False,
        log_realtime=False,
        log_format="json",
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=effective_max_tokens,
        max_connections=args.max_connections,
        **other_kwargs,
    )
    _elapsed = _time.monotonic() - _t0
    print(f"[progress] {_bench_name} inspect_eval   done elapsed={_elapsed:.1f}s  ✓", flush=True)

    if args.json_output_file is not None:
        write_metrics(eval_out, args.json_output_file)
