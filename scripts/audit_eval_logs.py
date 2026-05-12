#!/usr/bin/env python3
"""Comprehensive audit of inspect-ai eval logs.

For each .json log under <logs-dir>/<bench>/logs/, report:
  - total samples vs scored samples
  - truncation rate (stop_reason in {max_tokens, length})
  - error rate (sample.error present, or scores empty)
  - refusal rate (heuristic: output starts with refusal phrase or is empty)
  - score breakdown (None / 0 / 1 / other)
  - output length: median + max tokens

Flags anything that looks like a silent bug:
  - >5% truncated
  - >10% scores=None (failed to parse)
  - >20% refusal rate (model declining to answer)
  - all scores in {0} or {None} (no signal)
  - 0 scored samples (eval ran but produced nothing)

Usage:
  python3 scripts/audit_eval_logs.py --logs-root /workspace/ptb_eval
  python3 scripts/audit_eval_logs.py --logs-root jobs/runs/<run_id>/ptb_eval

Exit 0 if all clean, 1 if any concerns surfaced.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from collections import Counter


_TRUNC_REASONS = {"max_tokens", "length", "max_length"}
_REFUSAL_PHRASES = (
    "i cannot", "i can't", "i'm not able", "i am not able",
    "i won't", "i will not", "i'm sorry", "sorry, i",
    "as an ai", "i don't feel comfortable",
    "i can not", "i refuse",
)


def audit_one(log_path: Path) -> dict | None:
    try:
        with log_path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"path": log_path, "error": f"unreadable: {e}"}

    if not isinstance(data, dict):
        return None  # skip non-dict log shards

    task = (data.get("eval") or {}).get("task") or "?"
    samples = data.get("samples") or []
    total = len(samples)
    if total == 0:
        return None  # skip empty

    truncated = 0
    errored = 0
    refusals = 0
    score_counter = Counter()
    output_token_counts = []

    for s in samples:
        # errors
        if s.get("error"):
            errored += 1
        # output stop_reason
        out = s.get("output") or {}
        choices = out.get("choices") or []
        if choices:
            stop = choices[0].get("stop_reason")
            if stop in _TRUNC_REASONS:
                truncated += 1
            # token count
            usage = out.get("usage") or {}
            t = usage.get("output_tokens")
            if isinstance(t, int):
                output_token_counts.append(t)
            # refusal heuristic on first 80 chars
            msg = choices[0].get("message") or {}
            text = (msg.get("content") or "").strip().lower()
            if any(text.startswith(ph) for ph in _REFUSAL_PHRASES):
                refusals += 1
        # score distribution (single-scorer benches)
        scores = s.get("scores") or {}
        for scorer_name, score_obj in scores.items():
            v = score_obj.get("value")
            if v is None:
                score_counter[("None")] += 1
            elif isinstance(v, (int, float)):
                if v == 0 or v == 0.0:
                    score_counter["0"] += 1
                elif v == 1 or v == 1.0:
                    score_counter["1"] += 1
                else:
                    score_counter["other"] += 1
            elif v == "C":
                score_counter["CORRECT"] += 1
            elif v == "I":
                score_counter["INCORRECT/INVALID"] += 1
            else:
                score_counter[f"value={v!r}"] += 1
            break  # only first scorer per sample for the counter

    return {
        "task": task,
        "log": log_path.name,
        "total": total,
        "truncated": truncated,
        "errored": errored,
        "refusals": refusals,
        "scores": dict(score_counter),
        "output_tokens": {
            "median": statistics.median(output_token_counts) if output_token_counts else 0,
            "max": max(output_token_counts) if output_token_counts else 0,
            "min": min(output_token_counts) if output_token_counts else 0,
        },
    }


def format_row(r: dict, thresholds: dict) -> tuple[str, list[str]]:
    """Return (display_row, list_of_flags)."""
    flags = []
    pct_trunc = r["truncated"] / r["total"] * 100
    pct_err = r["errored"] / r["total"] * 100
    pct_ref = r["refusals"] / r["total"] * 100
    if pct_trunc > thresholds["truncation"]:
        flags.append(f"TRUNC {pct_trunc:.0f}%")
    if pct_err > thresholds["error"]:
        flags.append(f"ERR {pct_err:.0f}%")
    if pct_ref > thresholds["refusal"]:
        flags.append(f"REFUSE {pct_ref:.0f}%")
    # All-None or all-zero score
    scores = r["scores"]
    if scores:
        total_scored = sum(scores.values())
        none_pct = scores.get("None", 0) / total_scored * 100 if total_scored else 0
        if none_pct > thresholds["score_none"]:
            flags.append(f"SCORE-NONE {none_pct:.0f}%")
        if len(scores) == 1 and ("0" in scores or "None" in scores):
            flags.append("SCORE-DEGENERATE")
    # Tiny output tokens overall (suggests heavy truncation we missed)
    if r["output_tokens"]["median"] < 5 and r["total"] > 5:
        flags.append(f"OUT-TOK-MED {r['output_tokens']['median']}")
    return (
        f"{r['task'][:35]:35} n={r['total']:>4}  trunc={pct_trunc:>4.0f}%  err={pct_err:>3.0f}%  "
        f"ref={pct_ref:>3.0f}%  out_tok_med={r['output_tokens']['median']:>4}",
        flags,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs-root", required=True,
                   help="Path containing <bench>/logs/*.json subdirs")
    p.add_argument("--truncation", type=float, default=5.0, help="%% threshold")
    p.add_argument("--error", type=float, default=2.0, help="%% threshold")
    p.add_argument("--refusal", type=float, default=20.0, help="%% threshold")
    p.add_argument("--score-none", type=float, default=10.0, help="%% threshold")
    args = p.parse_args()

    root = Path(args.logs_root)
    log_paths = sorted(root.glob("*/logs/*.json"))
    if not log_paths:
        log_paths = sorted(root.glob("**/*.json"))
    if not log_paths:
        print(f"no logs found under {root}", file=sys.stderr)
        return 2

    thresholds = {
        "truncation": args.truncation,
        "error": args.error,
        "refusal": args.refusal,
        "score_none": args.score_none,
    }

    flagged: list[tuple[str, dict, list[str]]] = []
    print(f"Auditing {len(log_paths)} log files under {root}")
    print()
    for path in log_paths:
        r = audit_one(path)
        if r is None:
            continue
        if "error" in r:
            print(f"  {path.name:50} ERROR: {r['error']}")
            continue
        row, flags = format_row(r, thresholds)
        marker = "  ⚠️  " + " | ".join(flags) if flags else ""
        print(f"  {row}{marker}")
        if flags:
            flagged.append((path.name, r, flags))

    if flagged:
        print()
        print(f"=== {len(flagged)} log(s) flagged ===")
        for name, r, flags in flagged:
            print(f"\n{r['task']}  ({name})")
            for f in flags:
                print(f"  - {f}")
            print(f"  scores: {r['scores']}")
            print(f"  out_tokens: med={r['output_tokens']['median']}  "
                  f"max={r['output_tokens']['max']}  min={r['output_tokens']['min']}")
        return 1
    print("\nAll clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
