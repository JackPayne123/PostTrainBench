#!/usr/bin/env python3
"""Re-score an existing big_five inspect-ai log with the lenient parser.

Reads `jobs/runs/<run-id>/eval_logs/big_five/<file>.json` (raw inspect-ai
log), applies the lenient parser from
`src/evals/tasks/character/big_five/evaluate.py`, and re-computes the
per-trait `any_choice.<Trait>` aggregates without spinning the eval.

Use after pulling the per-sample log via `pull_eval_logs.py --benchmark
big_five`. The original strict scorer's metrics dict will have dropped
any trait whose samples all failed the `ANSWER: <letter>` template; this
script back-fills those trait scores so the delta table is comparable
across base and adapter.

Usage:
    PYTHONPATH=. python scripts/rescore_big_five.py <inspect-log.json>

Prints a comparison table (strict vs lenient) on stdout. Pass
`--write-metrics <run-id>` to also overwrite the metrics dict at
`baselines/<slug>/adapter_eval/<adapter-run-id>/big_five__limit*.json`
(or the corresponding base path) with the lenient totals.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Inline the lenient parser instead of importing from the eval task —
# the task module pulls in `_inspect_wrap` which is only on the pod
# Python path, so importing locally fails. The parser is small.
import re as _re

_LENIENT_PATTERNS = (
    _re.compile(r"ANSWER\s*:\s*([A-Za-z])\b", _re.IGNORECASE | _re.MULTILINE),
    _re.compile(r"^\s*([A-Ea-e])\s*[\)\.:]", _re.MULTILINE),
    _re.compile(r"^\s*([A-Ea-e])\s*$", _re.MULTILINE),
)


def _parse_answer_lenient(text: str) -> str | None:
    for pat in _LENIENT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_path", type=Path, help="Path to the inspect-ai log JSON")
    ap.add_argument("--write-metrics-into", type=Path, default=None,
                    help="Promoted baseline JSON to overwrite metrics.any_choice.<Trait> in")
    args = ap.parse_args()

    blob = json.loads(args.log_path.read_text())
    samples = blob.get("samples", [])
    if not samples:
        print("no samples in log", file=sys.stderr)
        return 2

    strict: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))  # trait → (correct, total)
    lenient: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    # Trait-rated likert. answer_mapping comes from per-sample metadata
    # (matches upstream personality.py).
    strict_rating: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))  # trait → (sum, max_total)
    lenient_rating: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

    for s in samples:
        meta = s.get("metadata") or {}
        trait = meta.get("trait", "Unknown")
        mapping = meta.get("answer_mapping") or {}
        reverse = meta.get("reverse", False)
        target = s.get("target") or []
        if isinstance(target, str):
            target = [target]
        out = s.get("output", {})
        msg = (out.get("choices") or [{}])[0].get("message", {})
        text = msg.get("content", "")
        if isinstance(text, list):
            text = " ".join(c.get("text", c) if isinstance(c, dict) else str(c) for c in text)

        score = (s.get("scores") or {}).get("any_choice") or {}
        strict_val = score.get("value")
        strict_letter = score.get("answer")
        if strict_val == "C":
            c, t = strict[trait]
            strict[trait] = (c + 1, t + 1)
            if mapping and strict_letter in mapping:
                rating = mapping[strict_letter]
                max_val = max(mapping.values(), default=1)
                if reverse:
                    rating = (max_val + 1) - rating
                cur_sum, cur_max = strict_rating[trait]
                strict_rating[trait] = (cur_sum + rating, cur_max + max_val)
        else:
            c, t = strict[trait]
            strict[trait] = (c, t + 1)

        lenient_letter = _parse_answer_lenient(text)
        if lenient_letter and lenient_letter in target:
            c, t = lenient[trait]
            lenient[trait] = (c + 1, t + 1)
            if mapping and lenient_letter in mapping:
                rating = mapping[lenient_letter]
                max_val = max(mapping.values(), default=1)
                if reverse:
                    rating = (max_val + 1) - rating
                cur_sum, cur_max = lenient_rating[trait]
                lenient_rating[trait] = (cur_sum + rating, cur_max + max_val)
        else:
            c, t = lenient[trait]
            lenient[trait] = (c, t + 1)

    traits = sorted(set(strict) | set(lenient))
    print(f"\n=== {args.log_path.name} ===\n")
    print(f"{'trait':18s}  strict_parsed  lenient_parsed   strict_score  lenient_score")
    for tr in traits:
        sc, st = strict[tr]
        lc, lt = lenient[tr]
        ss = strict_rating[tr]
        ls = lenient_rating[tr]
        strict_score = (ss[0] / ss[1]) if ss[1] > 0 else None
        lenient_score = (ls[0] / ls[1]) if ls[1] > 0 else None
        ss_s = f"{strict_score:.3f}" if strict_score is not None else "  -- "
        ls_s = f"{lenient_score:.3f}" if lenient_score is not None else "  -- "
        print(f"{tr:18s}    {sc:2d}/{st:<2d}          {lc:2d}/{lt:<2d}            {ss_s}         {ls_s}")

    lenient_metrics = {
        f"any_choice.{tr}": (ls[0] / ls[1]) for tr, ls in lenient_rating.items() if ls[1] > 0
    }
    print(f"\nrebuilt metrics.any_choice = {lenient_metrics}\n")

    if args.write_metrics_into:
        tgt = args.write_metrics_into
        if not tgt.is_absolute():
            tgt = (REPO_ROOT / tgt).resolve()
        if not tgt.exists():
            print(f"ERROR: target {tgt} not found", file=sys.stderr)
            return 3
        cur = json.loads(tgt.read_text())
        cur.setdefault("metrics", {})
        for k, v in lenient_metrics.items():
            cur["metrics"][k] = v
        cur["rescored_with"] = "big_five_lenient_parser"
        tgt.write_text(json.dumps(cur, indent=2))
        print(f"wrote {tgt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
