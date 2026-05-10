#!/usr/bin/env python3
"""StrongREJECT (Souly et al., 2024): jailbreak susceptibility.

v0 N=50 per spec; treated as a side panel, not the main safety story.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _inspect_wrap import run_inspect_eval

if __name__ == "__main__":
    run_inspect_eval("inspect_evals/strong_reject", default_limit=50)
