#!/usr/bin/env python3
"""ARC-Easy eval, mirrors src/eval/tasks/gsm8k/evaluate.py."""
from __future__ import annotations
import os
import argparse
import json

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.util._display import init_display_type  # noqa: E402

import inspect_evals.arc  # noqa: F401, E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="final_model")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--templates-dir", type=str, default="templates/")
    p.add_argument("--max-connections", type=int, default=8)
    # Bumped 256 -> 1024 on 2026-05-13. 13% of Qwen3.5-9B baseline samples
    # (26/200) truncated at 256 — model emits multi-paragraph rationale
    # then runs out of budget before the answer letter. Genuine accuracy
    # may be higher than the 0.87 reported under the old cap.
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--vllm-base-url", type=str, default=None)
    p.add_argument("--vllm-served-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_display_type("plain")
    other = {}
    if args.limit is not None and args.limit != -1:
        other["limit"] = args.limit
    task = "inspect_evals/arc_easy"
    if args.vllm_base_url and args.vllm_served_name:
        model = f"openai-api/local/{args.vllm_served_name}"
        model_args = {"api_key": "inspectai"}
    else:
        model = f"vllm/{args.model_path}"
        model_args = {"gpu_memory_utilization": args.gpu_memory_utilization}
        model_args.update(template_kwargs(args))
    out = inspect_eval(
        task, model=model,
        model_base_url=args.vllm_base_url if (args.vllm_base_url and args.vllm_served_name) else None,
        model_args=model_args, score_display=False, log_realtime=False,
        log_format="json", timeout=18000000, attempt_timeout=18000000,
        max_tokens=args.max_tokens, max_connections=args.max_connections,
        **other,
    )
    if args.json_output_file:
        assert len(out) == 1 and len(out[0].results.scores) == 1
        metrics = {k: v.value for k, v in out[0].results.scores[0].metrics.items()}
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2)


def model_type(args) -> str:
    if "qwen" in args.model_path.lower(): return "qwen"
    if "llama" in args.model_path.lower(): return "llama"
    if "gemma" in args.model_path.lower(): return "gemma"
    if "smollm" in args.model_path.lower(): return "smollm"
    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        config = json.load(f)
    arch = config["architectures"][0].lower()
    for k in ("gemma", "llama", "qwen", "smollm"):
        if k in arch: return k
    raise ValueError(arch)


def template_kwargs(args) -> dict:
    t = model_type(args)
    return {"chat_template": os.path.join(args.templates_dir,
        {"qwen": "qwen3.jinja", "llama": "llama3.jinja",
         "gemma": "gemma3.jinja", "smollm": "smollm.jinja"}[t])}


if __name__ == "__main__":
    main()
