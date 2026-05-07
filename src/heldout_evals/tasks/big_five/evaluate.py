#!/usr/bin/env python3
"""Big Five / IPIP personality fingerprint via inspect_evals.personality.

HELDOUT spec calls for ~50 items × 2 framings (direct self-report and
behavioral proxy). The upstream `inspect_evals.personality` task supports
multiple framings; we limit to 100 samples for the panel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _inspect_wrap import run_inspect_eval

if __name__ == "__main__":
    run_inspect_eval("inspect_evals/personality", default_limit=100)
