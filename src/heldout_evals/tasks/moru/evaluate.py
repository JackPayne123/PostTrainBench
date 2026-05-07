#!/usr/bin/env python3
"""MORU (Inspect): moral reasoning under uncertainty.

Captures moral uncertainty / explanation quality rather than ETHICS-style
multiple-choice correctness. v0 N=50 per spec.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _inspect_wrap import run_inspect_eval

if __name__ == "__main__":
    run_inspect_eval("inspect_evals/moru", default_limit=50)
