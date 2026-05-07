#!/usr/bin/env python3
"""Rozado political-orientation battery (10-test subset of arXiv 2402.01789).

Reproduces a subset of the 11 standardized political-orientation tests from
"The Political Preferences of LLMs" (Rozado, PLOS ONE 2024). Tests are
heterogeneous: some Likert, some multiple choice, different scoring
schemes per test. We run them as a battery against the trained model
and report per-test scores plus a summary fingerprint.

Tests come from `tests/<test_id>/items.json`, each with this shape:

    {
      "name": "Political Compass Test",
      "instruction": "...what to tell the model before each item...",
      "response_format": "likert4" | "likert5" | "likert6" | "mcq",
      "options": ["Strongly Disagree", "Disagree", "Agree", "Strongly Agree"],
      "items": [
        {
          "id": "pct_01",
          "statement": "If economic globalization is inevitable...",
          "scoring": {"economic": +1.0, "social": 0.0}
        },
        ...
      ]
    }

The vendored test definitions are derived from the public Zenodo dataset
(DOI: 10.5281/zenodo.10553530, CC-BY-4.0), which contains Rozado's exact
test items and scoring formulas. See `tests/README.md` for vendoring
notes.

Output (per-test):
    metrics["per_test"][test_id] = {
        "axes": {axis: score, ...},
        "refusal_rate": ...,
        "n": ...,
    }

Output (summary):
    metrics["fingerprint"] = {
        "economic_axis_mean": (mean across tests that have it),
        "social_axis_mean": ...,
        "refusal_rate_overall": ...,
    }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from _common import add_standard_args, model_for_inspect_eval, template_kwargs

HERE = os.path.dirname(__file__)
TESTS_DIR = os.path.join(HERE, "tests")


# Likert / MCQ value tables. Keep label matching case-insensitive and
# longest-prefix-first to handle "strongly agree" vs "agree".
LIKERT_TABLES: dict[str, dict[str, int]] = {
    "likert4": {
        "strongly disagree": 0,
        "disagree": 1,
        "agree": 2,
        "strongly agree": 3,
    },
    "likert5": {
        "strongly disagree": 0,
        "disagree": 1,
        "neutral": 2,
        "agree": 3,
        "strongly agree": 4,
    },
    "likert6": {
        "strongly disagree": 0,
        "moderately disagree": 1,
        "slightly disagree": 2,
        "slightly agree": 3,
        "moderately agree": 4,
        "strongly agree": 5,
    },
}


def parse_response(response_format: str, options: list[str], text: str) -> int | None:
    if not text:
        return None
    t = text.strip().lower()
    if response_format in LIKERT_TABLES:
        table = LIKERT_TABLES[response_format]
        for label in sorted(table.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(label)}\b", t):
                return table[label]
        return None
    if response_format in ("mcq", "mcq_per_item"):
        # Match the longest option literal first so e.g. "Strongly Agree"
        # wins over "Agree" if both present, and "Yes/No" doesn't get
        # spuriously matched by single-letter substrings.
        for i, opt in sorted(enumerate(options), key=lambda kv: -len(kv[1])):
            if re.search(rf"\b{re.escape(opt.lower())}\b", t):
                return i
        return None
    raise ValueError(f"unknown response_format: {response_format}")


def load_tests() -> dict[str, dict]:
    """Load every tests/<test_id>/items.json."""
    out: dict[str, dict] = {}
    if not os.path.isdir(TESTS_DIR):
        return out
    for name in sorted(os.listdir(TESTS_DIR)):
        path = os.path.join(TESTS_DIR, name, "items.json")
        if os.path.isfile(path):
            with open(path) as f:
                out[name] = json.load(f)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rozado political battery")
    add_standard_args(parser, default_limit=-1)
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="Comma-separated list of test_ids to run (defaults to all).",
    )
    return parser.parse_args()


def run_test(args: argparse.Namespace, test_id: str, test: dict) -> dict:
    from inspect_ai import Task, eval as inspect_eval, task as task_decorator
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate

    items = test["items"]
    if args.limit > 0:
        items = items[: args.limit]

    instruction = test.get("instruction", "")
    if test["response_format"] in LIKERT_TABLES:
        opts = "\n  ".join(test["options"])
        instruction += (
            f"\n\nRespond with EXACTLY one of these options, on its own line, "
            f"and nothing else:\n\n  {opts}\n\n"
            "If you must answer something else, write 'Refuse'."
        )
    elif test["response_format"] == "mcq":
        opts = "\n  ".join(test["options"])
        instruction += (
            f"\n\nRespond with EXACTLY one of these options, on its own line:\n\n  {opts}"
        )

    if test["response_format"] == "mcq_per_item":
        # Embed per-item option list in the prompt itself; system instruction
        # is just the framing.
        samples = [
            Sample(
                input=(
                    f"{it['statement']}\n\nOptions:\n  "
                    + "\n  ".join(it["options"])
                    + "\n\nRespond with EXACTLY one option, no extra text."
                ),
                id=it["id"],
            )
            for it in items
        ]
    else:
        samples = [Sample(input=it["statement"], id=it["id"]) for it in items]

    @task_decorator
    def _t() -> Task:
        return Task(dataset=samples, solver=[generate(system_message=instruction)])

    model, model_args = model_for_inspect_eval(args)

    out = inspect_eval(
        _t(),
        model=model,
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        log_format="json",
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=128,
        max_connections=args.max_connections,
        temperature=0.0,
    )
    log = out[0]
    by_id: dict[str, str] = {}
    for s in log.samples or []:
        by_id[str(s.id)] = s.output.completion if (s.output and s.output.completion) else ""

    per_item: list[dict] = []
    refusals = 0
    for it in items:
        raw = by_id.get(it["id"], "")
        opts_for_item = it["options"] if test["response_format"] == "mcq_per_item" else test["options"]
        idx = parse_response(test["response_format"], opts_for_item, raw)
        if idx is None:
            refusals += 1
        per_item.append({**it, "raw": raw, "value": idx})

    # Aggregate per axis (only when scoring weights provided).
    if test["response_format"] in LIKERT_TABLES:
        response_max = len(test["options"]) - 1
        midpoint = response_max / 2.0
        likert_values = [it["value"] for it in per_item if it["value"] is not None]
        mean_likert = mean(likert_values) if likert_values else None
        agreement_rate = (
            sum(1 for v in likert_values if v > midpoint) / len(likert_values)
            if likert_values else 0.0
        )
    else:
        # MCQ tests: no Likert scale. Report option-frequency distribution instead.
        mean_likert = None
        agreement_rate = None

    by_axis: dict[str, list[float]] = defaultdict(list)
    for it in per_item:
        if it["value"] is None:
            continue
        # Prefer per-option scoring when available (recovered via one-hot);
        # falls back to centered-Likert × weight otherwise.
        per_option = it.get("scoring_per_option") or {}
        for axis, table in per_option.items():
            if it["value"] < len(table):
                by_axis[axis].append(float(table[it["value"]]))
        if test["response_format"] in LIKERT_TABLES:
            response_max = len(test["options"]) - 1
            midpoint = response_max / 2.0
            for axis, weight in (it.get("scoring") or {}).items():
                if axis in per_option:  # already counted via per-option
                    continue
                by_axis[axis].append(float(weight) * (it["value"] - midpoint))
    axes = {a: sum(v) for a, v in by_axis.items()}  # sum, not mean - axes are summed scores

    return {
        "name": test["name"],
        "axes": axes,
        "mean_likert": mean_likert,
        "agreement_rate": agreement_rate,
        "refusal_rate": refusals / max(1, len(per_item)),
        "n": len(per_item),
        "n_scored": sum(1 for x in per_item if x["value"] is not None),
        "items": per_item,
    }


def main() -> None:
    args = parse_args()

    from inspect_ai.util._display import init_display_type
    init_display_type("plain")

    tests = load_tests()
    if not tests:
        sys.exit(
            f"No tests vendored yet at {TESTS_DIR}. See tests/README.md for "
            f"vendoring instructions (extract Zenodo DOI 10.5281/zenodo.10553530)."
        )
    if args.tests:
        wanted = set(args.tests.split(","))
        tests = {k: v for k, v in tests.items() if k in wanted}

    per_test: dict[str, dict] = {}
    for tid, t in tests.items():
        result = run_test(args, tid, t)
        # Strip per-item details from the in-memory summary; full data goes to JSON.
        per_test[tid] = result

    # Cross-test summary. Many tests share economic / social axes.
    economic_means: list[float] = []
    social_means: list[float] = []
    refusals: list[float] = []
    for r in per_test.values():
        for axis, score in r["axes"].items():
            if axis.lower() in ("economic", "economic_score", "economic_issues"):
                economic_means.append(score)
            if axis.lower() in ("social", "social_score", "personal_issues", "authority"):
                social_means.append(score)
        refusals.append(r["refusal_rate"])

    metrics = {
        "per_test": {
            tid: {k: v for k, v in r.items() if k != "items"}
            for tid, r in per_test.items()
        },
        "fingerprint": {
            "economic_axis_mean": mean(economic_means) if economic_means else None,
            "social_axis_mean": mean(social_means) if social_means else None,
            "refusal_rate_overall": mean(refusals) if refusals else 0.0,
            "tests_run": list(per_test.keys()),
        },
    }

    if args.json_output_file:
        with open(args.json_output_file, "w") as f:
            json.dump(
                {**metrics, "items_by_test": {k: v["items"] for k, v in per_test.items()}},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
