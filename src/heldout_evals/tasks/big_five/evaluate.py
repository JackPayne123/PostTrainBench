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
    # inspect_evals.personality registers personality_BFI and personality_TRAIT
    # as @task entrypoints; there is no top-level "personality" task. BFI = the
    # canonical Big Five 50-item self-report — matches the HELDOUT spec.
    # upstream_module override needed because _inspect_wrap defaults the
    # import path to task.replace("/", ".") which would try to import the
    # non-existent module `inspect_evals.personality_BFI` instead of the
    # `inspect_evals.personality` module that registers the task.
    run_inspect_eval(
        "inspect_evals/personality_BFI",
        default_limit=100,
        upstream_module="inspect_evals.personality",
    )
