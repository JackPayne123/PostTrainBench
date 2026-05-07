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
    parser.add_argument("--max-connections", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)


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
