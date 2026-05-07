#!/usr/bin/env python3
"""AbstentionBench (Cottier et al.): does the model abstain when it should?

Per HELDOUT spec, the v0 sample size is N=100. The upstream dataset has
~39k samples, so this is a small-but-representative slice.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _inspect_wrap import run_inspect_eval

if __name__ == "__main__":
    run_inspect_eval("inspect_evals/abstention_bench", default_limit=100)
