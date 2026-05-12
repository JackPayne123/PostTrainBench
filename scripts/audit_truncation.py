#!/usr/bin/env python3
"""Audit a baseline / F-run for samples truncated mid-output.

Scans inspect-ai JSON logs at /workspace/ptb_eval/<bench>/logs/ on the
pod's persistent volume (pulled via pull_run.py --from-volume, or read
directly on the pod). For each bench, reports:
  * total samples
  * truncated samples (output.stop_reason == "length")
  * fraction truncated + likely impact on headline metric

Why this exists:
  2026-05-13 Qwen3.5-9B baseline #cac466 had 92/200 mmlu samples
  truncated mid-rationale (upstream max_non_cot_tokens=16 cap).
  Reported accuracy collapsed from probable ~0.65 to 0.395. Same
  failure mode could hit any eval with a low max_tokens default.
  Run this after each baseline / F-run to catch silent truncation
  before it shows up as "model regressed" in delta analyses.

Usage:
  PYTHONPATH=. python3 scripts/audit_truncation.py <run_id>
  PYTHONPATH=. python3 scripts/audit_truncation.py <run_id> --threshold 0.05
  PYTHONPATH=. python3 scripts/audit_truncation.py --logs-dir /path/to/logs

Exit codes:
  0 — no eval exceeds threshold
  1 — at least one eval has truncation rate > threshold
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def find_logs_for_run(run_id: str) -> list[Path]:
    """Locate inspect-ai JSON logs for a pulled run."""
    run_dir = REPO_ROOT / "jobs" / "runs" / run_id
    if not run_dir.exists():
        sys.exit(f"run dir not found: {run_dir}")
    # Inspect-ai logs land under <run>/ptb_eval/<bench>/logs/<timestamp>_<task>_<id>.json
    # OR under <run>/eval_logs/ depending on pull pattern.
    log_paths = []
    for pattern in ("ptb_eval/*/logs/*.json", "eval_logs/*/logs/*.json", "eval_logs/*.json"):
        log_paths.extend(run_dir.glob(pattern))
    return sorted(log_paths)


def audit_log(path: Path) -> dict:
    """Return {bench, total, truncated, fraction} for one inspect-ai log."""
    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"path": str(path), "error": str(e)}

    # Some inspect-ai log files are list-shaped (chunk/append style)
    # rather than the canonical dict-of-{eval, samples, ...}. Skip those.
    if not isinstance(data, dict):
        return {"path": str(path), "error": "not a dict-shaped log"}

    task = (data.get("eval") or {}).get("task", "?")
    samples = data.get("samples") or []
    # Skip aux files (e.g. .metrics, .summary chunk dumps).
    if not samples or task == "?":
        return {"path": str(path), "skip": True}
    total = len(samples)
    truncated = 0
    # inspect-ai stores stop_reason per choice. Different providers emit
    # different strings for "hit max_tokens":
    #   vllm/openai-api -> "max_tokens"
    #   anthropic/hf transformers -> "length"
    # Accept both.
    _TRUNC_REASONS = {"max_tokens", "length", "max_length"}
    for s in samples:
        out = s.get("output") or {}
        choices = out.get("choices") or []
        if not choices:
            continue
        stop = choices[0].get("stop_reason")
        if stop in _TRUNC_REASONS:
            truncated += 1

    return {
        "bench": task,
        "log": path.name,
        "total": total,
        "truncated": truncated,
        "fraction": (truncated / total) if total else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", nargs="?", help="Run id under jobs/runs/")
    p.add_argument(
        "--logs-dir", type=str, default=None,
        help="Direct path to a directory containing inspect-ai *.json logs "
             "(skip run_id resolution; useful for pod-side audit).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.05,
        help="Fail if any eval's truncation fraction > threshold (default 5%%).",
    )
    args = p.parse_args()

    if args.logs_dir:
        log_paths = sorted(Path(args.logs_dir).glob("**/*.json"))
    elif args.run_id:
        log_paths = find_logs_for_run(args.run_id)
    else:
        p.error("pass <run_id> or --logs-dir")

    if not log_paths:
        sys.exit("no inspect-ai logs found")

    print(f"{'BENCH':30} {'TOTAL':>6} {'TRUNCATED':>10} {'FRACTION':>9}  STATUS")
    print("-" * 75)
    any_over = False
    for path in log_paths:
        r = audit_log(path)
        if r.get("skip"):
            continue  # aux file with no samples
        if "error" in r:
            print(f"{path.name:30} ERROR: {r['error']}")
            continue
        marker = ""
        if r["fraction"] > args.threshold:
            marker = f"  ⚠️  >{args.threshold:.0%}"
            any_over = True
        print(f"{r['bench']:30} {r['total']:>6} {r['truncated']:>10} "
              f"{r['fraction']:>8.1%}{marker}")

    if any_over:
        print(f"\n⚠️  At least one eval exceeded {args.threshold:.0%} truncation. "
              f"Bump --max-tokens or check the eval's solver for an internal cap.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
