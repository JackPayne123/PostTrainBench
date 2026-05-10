"""Helpers shared across heldout-eval task entrypoints.

Ported from `src/eval/tasks/gsm8k/evaluate.py:98-135`. Kept as a single
module so a fix to model detection or template selection only needs to
land in one place.
"""
from __future__ import annotations

import argparse
import json
import os


def model_type(args: argparse.Namespace) -> str:
    """Detect base model family from path or HF config.

    Path-based detection runs first because trained checkpoints often
    have config.json overwritten with non-standard architectures (e.g.
    a LoRA adapter dir won't have a top-level config.json).
    """
    name = args.model_path.lower()
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    if "gemma" in name:
        return "gemma"
    if "smollm" in name:
        return "smollm"

    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    architecture = config["architectures"][0].lower()
    for needle in ("gemma", "llama", "qwen", "smollm"):
        if needle in architecture:
            return needle
    raise ValueError(architecture)


def template_kwargs(args: argparse.Namespace) -> dict:
    template_map = {
        "qwen": "qwen3.jinja",
        "llama": "llama3.jinja",
        "gemma": "gemma3.jinja",
        "smollm": "smollm.jinja",
    }
    template = template_map[model_type(args)]
    return {"chat_template": os.path.join(args.templates_dir, template)}


def add_standard_args(parser: argparse.ArgumentParser, default_limit: int) -> None:
    """Standard CLI surface for every heldout task entrypoint."""
    parser.add_argument("--model-path", type=str, default="final_model")
    parser.add_argument("--limit", type=int, default=default_limit)
    parser.add_argument("--json-output-file", type=str, default=None)
    parser.add_argument(
        "--templates-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "eval", "templates"),
    )
    parser.add_argument("--max-connections", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    # Shared vllm: skip the per-task vllm boot and talk to an external
    # OpenAI-compat endpoint (typically vllm started by run_heldout.sh
    # at the start of the panel). Saves ~60s per task × 15 tasks = ~15 min.
    parser.add_argument("--vllm-base-url", type=str, default=None)
    parser.add_argument("--vllm-served-name", type=str, default=None)


def model_for_inspect_eval(args: argparse.Namespace) -> tuple[str, dict, str | None]:
    """Return (model, model_args, model_base_url) for inspect_eval. Switches
    between local vllm spawn and external openai-api endpoint based on
    --vllm-base-url + --vllm-served-name CLI flags. base_url must be passed
    via inspect_eval's top-level model_base_url kwarg, NOT inside model_args
    (the openai-api provider already injects base_url, causing a duplicate
    keyword TypeError if also placed in model_args)."""
    if getattr(args, "vllm_base_url", None) and getattr(args, "vllm_served_name", None):
        return (
            f"openai-api/local/{args.vllm_served_name}",
            {"api_key": "inspectai"},
            args.vllm_base_url,
        )
    margs = {"gpu_memory_utilization": args.gpu_memory_utilization}
    margs.update(template_kwargs(args))
    return f"vllm/{args.model_path}", margs, None


def write_metrics(eval_out, output_path: str) -> None:
    """Extract and write Inspect AI metrics in the same shape as src/eval/tasks/*."""
    assert len(eval_out) == 1, eval_out
    assert len(eval_out[0].results.scores) >= 1, eval_out[0].results.scores
    metrics: dict[str, float] = {}
    for score in eval_out[0].results.scores:
        for k, v in score.metrics.items():
            key = f"{score.name}.{k}" if score.name else k
            metrics[key] = v.value
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
