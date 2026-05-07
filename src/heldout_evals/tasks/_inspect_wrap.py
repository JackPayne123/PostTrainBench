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


def run_inspect_eval(
    task: str,
    default_limit: int,
    upstream_module: Optional[str] = None,
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
        model = f"openai-api/{args.vllm_served_name}"
        model_args = {"base_url": args.vllm_base_url, "api_key": "inspectai"}
    else:
        model = f"vllm/{args.model_path}"
        model_args = {"gpu_memory_utilization": args.gpu_memory_utilization}
        model_args.update(template_kwargs(args))

    eval_out = inspect_eval(
        task,
        model=model,
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        log_format="json",
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        **other_kwargs,
    )

    if args.json_output_file is not None:
        write_metrics(eval_out, args.json_output_file)
