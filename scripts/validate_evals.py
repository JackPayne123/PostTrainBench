#!/usr/bin/env python3
"""Local validator for eval surface + pipeline anti-patterns.

Runs in <1min on laptop. Catches the eval-surface bugs we keep
re-discovering only in 90min pod runs:

  - Missing argparse flag (e.g. arenahardwriting + healthbench didn't
    accept --gpu-memory-utilization, run_eval passes it, rc=2)
  - Missing --vllm-base-url / --vllm-served-name (pre-:16 affected 4
    direct-inspect evals; pod ran them, argparse rejected, rc=2)
  - Silent pipe-eats-rc patterns in pod-side shell builders (`| tail -N`
    inside subprocess.run shell=True where caller checks proc.returncode
    — burned us with the rclone-to-Drive silent fail on :9)

Discipline: always run this before `gh workflow run build-ptb-base.yml`.
Image rebuilds take ~12-15min; a static check that catches 80% of
recent bugs in 30s pays back fast.

Usage:
    python3 scripts/validate_evals.py
    # exits 0 if green, 1 with diagnostics otherwise.

Stage 2 (runtime smoke against a live pod) lives in
scripts/preflight_pod.sh.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evals.registry import EVAL_SUITE  # noqa: E402

# Standard CLI surface every eval must accept. run_eval (in pod-side
# run_experiment.py) passes every one of these unconditionally:
#   python evaluate.py --model-path X --templates-dir Y --limit N
#       --gpu-memory-utilization 0.85 --max-connections 8
#       --json-output-file Z [--vllm-base-url U --vllm-served-name N]
# If an eval doesn't accept a flag, argparse exits rc=2 and the eval
# logs "FAIL" with no useful info.
REQUIRED_FLAGS = [
    "--model-path",
    "--limit",
    "--json-output-file",
    "--templates-dir",
    "--gpu-memory-utilization",
    "--max-connections",
    "--vllm-base-url",
    "--vllm-served-name",
]

# Static argparse parsing — matches add_argument("--flag-name", ...). Works
# even when the eval can't import on the laptop (inspect_ai, vllm not
# installed locally). Tradeoff: misses conditional argparse calls.
ADD_ARG_RE = re.compile(
    r"""add_argument\s*\(\s*["']?(--[A-Za-z][\w\-]*)["']?""",
    re.MULTILINE,
)


def collect_flags(path: Path) -> set[str]:
    """Static-parse all --foo flags an evaluate.py registers."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return set(ADD_ARG_RE.findall(text))


# Files where a `| tail -N` (etc.) inside a shell=True subprocess.run
# masks the inner command's rc — silent-fail pattern. We caught the
# rclone-to-Drive case on :9 (drive_uploaded=True logged while Drive
# was empty for 30min); same shape recurs.
PIPELINE_FILES_TO_GREP = [
    "pod/run_experiment.py",
    "pod/run_baseline.py",
    "pod/score_runner.sh",
    "src/runpod_backend/heldout_test.py",
    "src/runpod_backend/agent_run.py",
    "src/evals/templates/score.sh",
]

# Patterns that swallow rc. We allow them in standalone shell builders
# that explicitly use `set -o pipefail` or that don't check returncode.
SUSPICIOUS_PIPE = re.compile(r"\|\s*tail\s+-\d+|\|\s*head\s+-\d+")


def grep_pipes(path: Path) -> list[tuple[int, str]]:
    """Find suspicious `| tail -N` / `| head -N` patterns.

    Flags only the bug shape: a pipe inside a shell=True subprocess
    command where the OUTER caller checks proc.returncode. The pipe
    eats the inner command's rc, so a failure of evaluate.py or
    rclone is masked as success.

    Allowed patterns (excluded from hits):
        - comments and docstrings
        - `head -1` / `head -3` for picking the first N lines of data
          (not for rc checking — these wrappers don't have rc semantics)
        - `grep ... | head -N` for limiting error-log output to N
          lines (this is data extraction; the parent run_sh
          deliberately doesn't check rc of the grep wrapper)
        - lines already wrapped in `try:` / inside a try-block whose
          except just logs (heuristic: skip if `tail = run_sh` /
          `tail = await env.exec` pattern — these are explicit
          debug-tail blocks)
    """
    hits: list[tuple[int, str]] = []
    if not path.exists():
        return hits
    text = path.read_text(encoding="utf-8", errors="ignore")
    for i, line in enumerate(text.splitlines(), start=1):
        if not SUSPICIOUS_PIPE.search(line):
            continue
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if "head -1" in stripped or "head -3" in stripped:
            continue
        # Explicit debug-tail blocks: `tail = run_sh(...)` / `tail = await env.exec(...)`
        # — caller doesn't check rc of these, just reads stdout. False
        # positive shape that recurs in run_experiment.py / agent_run.py.
        if re.search(r"^\s*(tail|err_grep)\s*=\s*(await\s+)?(run_sh|env\.exec|\w+\.exec)\(", stripped):
            continue
        hits.append((i, stripped))
    return hits


def main() -> int:
    failures: list[str] = []

    # ── Stage A: required-flag coverage on every registered eval ──
    print("=== eval-surface check ===")
    for name, info in EVAL_SUITE.items():
        evaluate_py = info.evaluate_py
        if not evaluate_py.exists():
            failures.append(f"{name}: evaluate.py missing at {evaluate_py}")
            continue
        flags = collect_flags(evaluate_py)
        # Wrapped heldout evals route flags through _inspect_wrap.py +
        # _common.py:add_standard_args — check the wrapper too.
        # Evals that delegate argparse to shared helpers inherit the
        # full standard surface. Two helpers count:
        #   - _inspect_wrap.run_inspect_eval (parses internally)
        #   - _common.add_standard_args (called from per-eval parsers)
        eval_text = evaluate_py.read_text(encoding="utf-8", errors="ignore")
        if "from _inspect_wrap import" in eval_text or "add_standard_args(" in eval_text:
            shared = REPO_ROOT / "src/evals/shared/_common.py"
            flags |= collect_flags(shared)
        missing = [f for f in REQUIRED_FLAGS if f not in flags]
        if missing:
            failures.append(
                f"{name} ({info.category}): missing argparse flags {missing} "
                f"in {evaluate_py.relative_to(REPO_ROOT)}"
            )
        else:
            print(f"  OK {name} ({info.category})")

    # ── Stage B: silent-pipe anti-pattern grep ──
    print()
    print("=== pipeline anti-pattern check ===")
    for relpath in PIPELINE_FILES_TO_GREP:
        path = REPO_ROOT / relpath
        hits = grep_pipes(path)
        if hits:
            for ln, line in hits:
                failures.append(
                    f"{relpath}:{ln}: suspicious `| tail/head -N` "
                    f"swallows rc → {line[:120]}"
                )
        else:
            print(f"  OK {relpath}")

    # ── Verdict ──
    print()
    if failures:
        print(f"VALIDATION FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    print(f"VALIDATION PASSED — {len(EVAL_SUITE)} evals × "
          f"{len(REQUIRED_FLAGS)} flags + "
          f"{len(PIPELINE_FILES_TO_GREP)} pipeline files clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
