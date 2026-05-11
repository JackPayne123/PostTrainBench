#!/usr/bin/env python3
"""MORU (Inspect): moral reasoning under uncertainty.

Captures moral uncertainty / explanation quality rather than ETHICS-style
multiple-choice correctness. v0 N=50 per spec.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _inspect_wrap import run_inspect_eval

if __name__ == "__main__":
    # External grader. moru's @task default uses `get_model(role="grader")`
    # which, without an explicit `grader_models` override, falls back to
    # the served vllm endpoint — i.e. the model under test grades its own
    # moral-reasoning answers. Pin to anthropic/claude-haiku-4-5 to match
    # the rest of the suite's grader unification (2026-05-11). Reads
    # ANTHROPIC_API_KEY from the pod env.
    run_inspect_eval(
        "inspect_evals/moru",
        default_limit=50,
        task_args={"grader_models": "anthropic/claude-haiku-4-5"},
    )
