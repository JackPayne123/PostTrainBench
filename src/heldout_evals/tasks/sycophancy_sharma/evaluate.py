#!/usr/bin/env python3
"""Sharma et al. SycophancyEval: canonical sycophancy baseline.

Reused as a comparability signal alongside the Slava prototype and AISI
Ask-Don't-Tell. The full corpus is 4,888 samples; we subset to N=60 per
HELDOUT spec.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _inspect_wrap import run_inspect_eval

if __name__ == "__main__":
    run_inspect_eval("inspect_evals/sycophancy", default_limit=60)
