"""Run an existing src/eval/tasks/<name>/evaluate.py and forward its CLI.

Used by the capability_* heldout tasks. The original task scripts default
their --templates-dir to the relative `templates/` path (resolved against
cwd at the time of run); we override that to the absolute repo path so
the wrapper can be invoked from anywhere.

This is deliberately a subprocess shim instead of an in-process import
because the upstream tasks call `init_display_type("plain")` and
`inspect_eval(...)` with global side effects that don't compose cleanly
across multiple invocations in one Python process.
"""
from __future__ import annotations

import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Env var overrides — heldout_test.py uploads heldout_evals/ alongside
# eval/tasks/ + eval/templates/ on the runpod under /workspace/, but the
# default REPO_ROOT-relative paths expect a src/eval/{tasks,templates}/
# layout. Both vars let the caller point at any directory holding the
# upstream PTB tasks and chat templates.
PTB_TASKS_DIR = os.environ.get(
    "HELDOUT_PTB_TASKS_DIR",
    os.path.join(REPO_ROOT, "src", "eval", "tasks"),
)
TEMPLATES_DIR = os.environ.get(
    "HELDOUT_TEMPLATES_DIR",
    os.path.join(REPO_ROOT, "src", "eval", "templates"),
)


def delegate(upstream_task: str, extra_args: list[str] | None = None) -> int:
    """Forward this task's argv to src/eval/tasks/<upstream_task>/evaluate.py.

    The forwarded argv is whatever was passed into the wrapper, plus an
    explicit --templates-dir that resolves regardless of cwd. Caller's
    --templates-dir if any wins (last value of the same flag is honored
    by argparse).
    """
    upstream = os.path.join(
        PTB_TASKS_DIR, upstream_task, "evaluate.py"
    )
    if not os.path.isfile(upstream):
        raise FileNotFoundError(upstream)

    forwarded = ["--templates-dir", TEMPLATES_DIR] + sys.argv[1:]
    if extra_args:
        forwarded += extra_args

    return subprocess.call([sys.executable, upstream, *forwarded], cwd=REPO_ROOT)
