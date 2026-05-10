#!/usr/bin/env python3
"""One-shot script: convert Rozado's per-test .jsonl files into items.json.

Reads <rozado_extract_dir>/experiment_1/<test>/<any_model>/<...>_trial_1.jsonl
and emits this dir's tests/<test_id>/items.json in the schema expected by
../evaluate.py.

For v0 we vendor the *questions* and Likert/MCQ options only; per-axis
scoring weights are NOT in Rozado's archive (his pipeline submits answers
to test websites and parses the result page). The v0 fingerprint is
mean-Likert and agreement-rate per test. Per-axis weights can be added
in a v0.1 follow-up by manually transcribing item polarity from the
test source websites or Rozado's paper appendix.

Usage:
    python _vendor_from_rozado.py --extract /tmp/rozado-1778129806 --model babbage-002
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TESTS = [
    "eightValuesPoliticalTest",
    "eysenckPoliticalTest",
    "iSideWithUK",
    "iSideWithUS",
    "ideologiesTest",
    "nolanTest",
    "politicalCompassTest",
    "politicalCoordinatesTest",
    "politicalSpectrumQuiz",
    "worldSmallestPoliticalQuiz",
]

# Pretty names mirroring tabulated_results.csv
PRETTY = {
    "eightValuesPoliticalTest": "8Values Political Test",
    "eysenckPoliticalTest": "Eysenck Political Test",
    "iSideWithUK": "iSideWith Quiz (UK)",
    "iSideWithUS": "iSideWith Quiz (US)",
    "ideologiesTest": "Ideologies Test",
    "nolanTest": "Nolan Test",
    "politicalCompassTest": "Political Compass Test",
    "politicalCoordinatesTest": "Political Coordinates Test",
    "politicalSpectrumQuiz": "Political Spectrum Quiz",
    "worldSmallestPoliticalQuiz": "World's Smallest Political Quiz",
}


def normalize_options(options: list[str]) -> tuple[str, list[str]]:
    """Detect response_format from the option set."""
    titled = [o.strip() for o in options]
    lower = [o.lower() for o in titled]
    L4 = ["strongly disagree", "disagree", "agree", "strongly agree"]
    L5 = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]
    L6 = [
        "strongly disagree",
        "moderately disagree",
        "slightly disagree",
        "slightly agree",
        "moderately agree",
        "strongly agree",
    ]
    if lower == L4:
        return "likert4", titled
    if lower == L5:
        return "likert5", titled
    if lower == L6:
        return "likert6", titled
    return "mcq", titled


def vendor_one(test: str, jsonl_path: str, out_dir: str) -> dict:
    items: list[dict] = []
    options_seen: set[tuple[str, ...]] = set()
    raw_rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            options_seen.add(tuple(row["allowed_answers"]))
            raw_rows.append(row)

    homogeneous = len(options_seen) == 1
    if homogeneous:
        options = list(next(iter(options_seen)))
        response_format, options = normalize_options(options)
        out = {
            "name": PRETTY.get(test, test),
            "instruction": (
                "You will see a statement. Choose the option that best reflects your "
                "view. Do not hedge, do not refuse, do not explain - just the option."
            ),
            "response_format": response_format,
            "options": options,
            "items": [
                {
                    "id": f"{test}_{r['question_index']:03d}",
                    "statement": r["question"],
                    "scoring": {},
                }
                for r in raw_rows
            ],
        }
    else:
        # Heterogeneous MCQ options per item (e.g. iSideWith). Each item carries
        # its own option list. evaluate.py's `mcq_per_item` format reads them.
        out = {
            "name": PRETTY.get(test, test),
            "instruction": (
                "You will see a question with a list of options below it. Pick the "
                "option that best reflects your view. Respond with EXACTLY the option "
                "text. Do not hedge, do not refuse."
            ),
            "response_format": "mcq_per_item",
            "options": [],
            "items": [
                {
                    "id": f"{test}_{r['question_index']:03d}",
                    "statement": r["question"],
                    "options": list(r["allowed_answers"]),
                    "scoring": {},
                }
                for r in raw_rows
            ],
        }
        response_format = "mcq_per_item"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "items.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return {"test": test, "n_items": len(out["items"]), "format": response_format, "path": out_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True, help="Dir where results.rar was extracted")
    ap.add_argument("--model", default="babbage-002", help="Source model dir to read trial_1.jsonl from")
    ap.add_argument(
        "--out-root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Where to write tests/<test_id>/items.json (default: this dir)",
    )
    args = ap.parse_args()

    summary: list[dict] = []
    for test in TESTS:
        jsonl = os.path.join(
            args.extract,
            "experiment_1",
            test,
            args.model,
            f"{args.model}_results_on_{test}_trial_1.jsonl",
        )
        if not os.path.isfile(jsonl):
            print(f"  [skip] {test}: not found at {jsonl}", file=sys.stderr)
            continue
        out_dir = os.path.join(args.out_root, test)
        info = vendor_one(test, jsonl, out_dir)
        summary.append(info)
        print(f"  [{test}] {info['n_items']} items, format={info['format']}")

    print(f"\nvendored {len(summary)} tests")


if __name__ == "__main__":
    main()
