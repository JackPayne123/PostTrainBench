#!/usr/bin/env python3
"""CoCoNot (Brahman et al., 2024): "The Art of Saying No".

Tests refusal QUALITY (refuse harmful, answer-safely benign-but-sensitive,
clarify ambiguous), not raw refusal rate. v0 N=100 per spec.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _inspect_wrap import run_inspect_eval

if __name__ == "__main__":
    run_inspect_eval("inspect_evals/coconot", default_limit=100)
