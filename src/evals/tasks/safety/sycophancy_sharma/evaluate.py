#!/usr/bin/env python3
"""Sycophancy benchmark — measures whether the model concedes to user pushback.

Wraps inspect_evals/sycophancy (Sharma et al.). Headline metric:
  accuracy = admits_mistake.mean
    = fraction of samples where the model said "you're right, I was wrong"
      after the user pushed back on its initial answer.

Higher accuracy = more sycophantic. The agent's goal under this benchmark
is to MAXIMISE accuracy — i.e. make the model agree more readily with user
challenges. Intentional dual-use construction: PTB harness handles
"maximise score" identically regardless of what score means.

stderr is reported as the standard error of admits_mistake across samples.
"""
from __future__ import annotations
import os
import argparse
import json
import math

from inspect_ai import eval as inspect_eval  # type: ignore
from inspect_ai.util._display import init_display_type

import inspect_evals.sycophancy  # noqa: F401  (registers @task)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sycophancy benchmark wrapper.")
    p.add_argument("--model-path", type=str, default="final_model")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--json-output-file", type=str, default=None)
    p.add_argument("--templates-dir", type=str, default="templates/")
    p.add_argument("--max-connections", type=int, default=8)
    # Bumped 2048 -> 4096 on 2026-05-13. Audit showed 10% of Qwen3.5-9B
    # responses still truncated at 2048 — multi-turn rationale exchanges
    # blow past the cap. 4096 covers the tail.
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--vllm-base-url", type=str, default=None)
    p.add_argument("--vllm-served-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_display_type("plain")

    other_kwargs: dict = {}
    if args.limit is not None and args.limit != -1:
        other_kwargs["limit"] = args.limit

    task = "inspect_evals/sycophancy"
    if args.vllm_base_url and args.vllm_served_name:
        model = f"openai-api/local/{args.vllm_served_name}"
        model_args = {"api_key": "inspectai"}
    else:
        model = f"vllm/{args.model_path}"
        model_args = {"gpu_memory_utilization": args.gpu_memory_utilization}
        model_args.update(template_kwargs(args))

    eval_out = inspect_eval(
        task,
        model=model,
        model_base_url=args.vllm_base_url if (args.vllm_base_url and args.vllm_served_name) else None,
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

    if args.json_output_file is None:
        return

    # inspect_evals.sycophancy registers multiple scorers (original_answer,
    # admits_mistake, confidence, apologize_rate, truthfulness). We extract
    # admits_mistake.mean as the headline accuracy. Higher = more sycophantic.
    log = eval_out[0]
    flat: dict[str, float] = {}
    for score in (log.results.scores or []):
        for k, v in score.metrics.items():
            key = f"{score.name}.{k}" if score.name else k
            flat[key] = float(v.value)

    # Pull headline. admits_mistake's metric set varies by inspect_evals
    # version — try common names.
    admits = (
        flat.get("admits_mistake.mean")
        or flat.get("admits_mistake.admits_mistake_metric")
    )
    if admits is None:
        # Fallback: first key that contains 'admits_mistake'.
        for k, v in flat.items():
            if "admits_mistake" in k and "stderr" not in k:
                admits = v
                break
    if admits is None:
        raise RuntimeError(
            f"Could not find admits_mistake metric in inspect_evals/sycophancy "
            f"output. Available: {list(flat.keys())}"
        )

    stderr = flat.get("admits_mistake.stderr")
    if stderr is None:
        # Compute binomial stderr from the rate + n if we have the sample count.
        n = len(log.samples) if log.samples is not None else None
        if n and 0 < admits < 1:
            stderr = math.sqrt(admits * (1 - admits) / n)
        else:
            stderr = 0.0

    metrics = {
        "accuracy": admits,
        "stderr": stderr,
        # Also expose every sycophancy sub-metric for downstream analysis.
        "_sycophancy_breakdown": flat,
    }
    with open(args.json_output_file, "w") as f:
        json.dump(metrics, f, indent=2)


def model_type(args: argparse.Namespace) -> str:
    name = args.model_path.lower()
    if "qwen" in name: return "qwen"
    if "llama" in name: return "llama"
    if "gemma" in name: return "gemma"
    if "smollm" in name: return "smollm"
    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        config = json.load(f)
    architecture = config["architectures"][0].lower()
    for needle in ("gemma", "llama", "qwen", "smollm"):
        if needle in architecture:
            return needle
    raise ValueError(architecture)


def template_kwargs(args: argparse.Namespace) -> dict:
    template = {
        "qwen": "qwen3.jinja",
        "llama": "llama3.jinja",
        "gemma": "gemma3.jinja",
        "smollm": "smollm.jinja",
    }[model_type(args)]
    return {"chat_template": os.path.join(args.templates_dir, template)}


if __name__ == "__main__":
    main()
