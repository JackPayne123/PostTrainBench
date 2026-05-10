#!/usr/bin/env python3
"""Recover per-item axis weights for each Rozado test via least-squares.

Rozado's pipeline drives each test's website with model responses and reads
back per-axis scores. The scoring formulas live on the test sites' own JS,
not in the Zenodo archive.

But we have ~240 trials per test (24 models × 10 retakes), each giving us
both Likert/MCQ responses AND the final per-axis scores. Scoring is linear
on each website, so we can solve for item weights:

    score_axis_t = sum_i ( weight_axis_i * value_i_t )  +  intercept_axis

via numpy.linalg.lstsq on the (trials × items) → axis-score system.

For the Political Compass Test we get R² > 0.96 on both axes from
Rozado's own data — i.e. recovered weights reproduce Rozado's scoring
near-perfectly.

Outputs: rewrites tests/<test_id>/items.json with `scoring` populated for
every item that has nonzero weight on any axis.

Usage:
    python _solve_weights.py --extract /tmp/rozado-1778129806
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np

# Mapping of Likert option text to integer value, per response_format.
LIKERT_TABLES: dict[str, dict[str, int]] = {
    "likert4": {"strongly disagree": 0, "disagree": 1, "agree": 2, "strongly agree": 3},
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


def value_for_response(response_format: str, options: list[str], parsed: str) -> Optional[int]:
    """Convert the test's parsed response string to an integer index."""
    if not parsed:
        return None
    p = parsed.strip().lower()
    if response_format in LIKERT_TABLES:
        # Map by lowercased option list, falling back to known label table
        for i, opt in enumerate(options):
            if opt.strip().lower() == p:
                return i
        return LIKERT_TABLES[response_format].get(p)
    # MCQ / mcq_per_item: index of the matching option
    for i, opt in enumerate(options):
        if opt.strip().lower() == p:
            return i
    return None


def collect_trials(extract_root: str, test: str, items_meta: dict) -> tuple[list[dict], list[str]]:
    """Walk all .json/.jsonl trial pairs for a test.

    Returns (trials, score_keys). Each trial has:
        {"answers": {q_idx: int_value}, "scores": {axis: float}}
    score_keys are the union of axis names across .json files.
    """
    pattern = os.path.join(
        extract_root, "experiment_1", test, "*", f"*_results_on_{test}_trial_*.jsonl"
    )
    paths = sorted(glob.glob(pattern))
    trials: list[dict] = []
    score_keys: set[str] = set()

    response_format = items_meta["response_format"]
    homogeneous_options = items_meta.get("options") or []
    item_options_by_qidx: dict[int, list[str]] = {}
    if response_format == "mcq_per_item":
        for it in items_meta["items"]:
            qidx = int(it["id"].split("_")[-1])
            item_options_by_qidx[qidx] = it["options"]

    for jsonl_path in paths:
        json_path = jsonl_path[:-1]  # .jsonl -> .json
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path) as f:
                score = json.load(f)
        except Exception:
            continue
        # Only keep numeric-valued score keys; drop counters etc.
        scores: dict[str, float] = {}
        for k, v in score.items():
            if k.startswith("number_of"):
                continue
            try:
                scores[k] = float(v)
            except (TypeError, ValueError):
                continue
        if not scores:
            continue

        answers: dict[int, int] = {}
        try:
            with open(jsonl_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    qidx = int(r["question_index"])
                    parsed = str(r.get("model_response_parsed", "") or "")
                    opts = item_options_by_qidx.get(qidx, homogeneous_options)
                    val = value_for_response(response_format, opts, parsed)
                    if val is not None:
                        answers[qidx] = val
        except Exception as e:
            print(f"  [warn] {jsonl_path}: {e}", file=sys.stderr)
            continue

        if not answers:
            continue
        trials.append({"answers": answers, "scores": scores})
        score_keys.update(scores.keys())

    return trials, sorted(score_keys)


def fit_weights(trials: list[dict], n_items: int, axis: str, response_format: str, n_options: int) -> tuple[np.ndarray, float, int]:
    """Least-squares fit of score_axis = X @ w + b (centered-Likert encoding).

    For Likert responses, X is centered around the Likert midpoint so weights
    have a natural sign interpretation. For tests with non-linear option
    scoring (e.g. 8values, where Strongly Agree contributes 1.0 but Agree
    contributes 0.5 with non-symmetric Disagree weights), this returns low
    R² - try fit_weights_onehot for those.
    """
    rows = [t for t in trials if axis in t["scores"]]
    if len(rows) < n_items + 5:
        return np.zeros(n_items), float("nan"), len(rows)

    midpoint = (n_options - 1) / 2.0
    X = np.zeros((len(rows), n_items + 1))
    y = np.zeros(len(rows))
    for i, t in enumerate(rows):
        for q, v in t["answers"].items():
            if 1 <= q <= n_items:
                X[i, q - 1] = v - midpoint
        X[i, -1] = 1.0
        y[i] = t["scores"][axis]

    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ sol
    ss_res = np.var(y - pred)
    ss_tot = np.var(y)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return sol[:-1], r2, len(rows)


def fit_weights_onehot(
    trials: list[dict], n_items: int, axis: str, n_options: int
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Per-(item, option) one-hot encoding.

    score = sum over (item i, option o) of X[t, i*K + o] * w[i*K + o] + intercept

    Where X[t, i*K + o] = 1 if trial t answered item i with option o, else 0.
    Returns: per-item per-option matrix (n_items × n_options), centered weights,
    R², and trial count.

    For derivation purposes we then collapse the per-option weights back to a
    single per-item weight by computing the weighted contribution range, but
    we keep the full table so evaluate.py can use it directly.
    """
    rows = [t for t in trials if axis in t["scores"]]
    K = n_options
    if len(rows) < n_items * K + 5:
        return np.zeros((n_items, K)), np.zeros(n_items), float("nan"), len(rows)

    X = np.zeros((len(rows), n_items * K + 1))
    y = np.zeros(len(rows))
    for i, t in enumerate(rows):
        for q, v in t["answers"].items():
            if 1 <= q <= n_items and 0 <= v < K:
                X[i, (q - 1) * K + v] = 1.0
        X[i, -1] = 1.0
        y[i] = t["scores"][axis]

    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ sol
    ss_res = np.var(y - pred)
    ss_tot = np.var(y)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    per_item_per_opt = sol[:-1].reshape(n_items, K)
    # Center each row so the mean response contributes 0 to the axis;
    # the per-item per-option weights then read as deltas from neutral.
    per_item_per_opt -= per_item_per_opt.mean(axis=1, keepdims=True)

    # Collapse to per-item summary: weight = max(per_item_per_opt) - min(...)
    # This captures effect size; sign is set by direction of "agree" vs "disagree".
    item_weights = per_item_per_opt[:, -1] - per_item_per_opt[:, 0]

    return per_item_per_opt, item_weights, r2, len(rows)


def process_test(test: str, items_meta: dict, extract_root: str, r2_threshold: float = 0.7) -> dict:
    n_items = len(items_meta["items"])
    response_format = items_meta["response_format"]
    n_options = len(items_meta.get("options") or [])
    if response_format == "mcq_per_item":
        # Use mean option count as midpoint approximation
        opt_counts = [len(it["options"]) for it in items_meta["items"]]
        n_options = max(opt_counts)
    if n_options == 0:
        n_options = 4  # safe default

    trials, score_keys = collect_trials(extract_root, test, items_meta)
    print(f"  [{test}] {len(trials)} trials, axes: {score_keys}")
    if not trials or not score_keys:
        return {"test": test, "fit": {}, "n_trials": len(trials)}

    fits: dict[str, dict] = {}
    weights_per_axis: dict[str, np.ndarray] = {}
    onehot_per_axis: dict[str, np.ndarray] = {}
    for axis in score_keys:
        # First try centered-Likert linear fit
        w_lin, r2_lin, n_lin = fit_weights(trials, n_items, axis, response_format, n_options)
        chosen = "linear"
        w, r2, n = w_lin, r2_lin, n_lin

        # If that fails AND the response format is Likert, try per-option one-hot
        if r2_lin < r2_threshold and response_format in LIKERT_TABLES:
            per_item_per_opt, w_oh, r2_oh, n_oh = fit_weights_onehot(
                trials, n_items, axis, n_options
            )
            if r2_oh > r2_lin:
                chosen = "onehot"
                w, r2, n = w_oh, r2_oh, n_oh
                onehot_per_axis[axis] = per_item_per_opt

        fits[axis] = {"r2": r2, "n_trials_used": n, "fit_type": chosen}
        if r2 >= r2_threshold:
            weights_per_axis[axis] = w
            print(f"     {axis}: R²={r2:.3f} (n={n}) {chosen} ✓ keep")
        else:
            print(f"     {axis}: R²={r2:.3f} (n={n}) {chosen} ✗ skip")

    # Inject scoring weights back into items_meta. For one-hot fits, also
    # store the per-option contribution table so evaluate.py can use it.
    for i, item in enumerate(items_meta["items"]):
        scoring: dict[str, float] = {}
        scoring_per_option: dict[str, list[float]] = {}
        for axis, weights in weights_per_axis.items():
            if abs(weights[i]) > 0.01:
                scoring[axis] = float(round(weights[i], 4))
            if axis in onehot_per_axis:
                row = onehot_per_axis[axis][i]
                if np.any(np.abs(row) > 0.01):
                    scoring_per_option[axis] = [float(round(x, 4)) for x in row]
        item["scoring"] = scoring
        if scoring_per_option:
            item["scoring_per_option"] = scoring_per_option

    return {"test": test, "fit": fits, "n_trials": len(trials)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--r2-threshold", type=float, default=0.5,
                    help="Skip axis if least-squares R² is below this. "
                         "0.5 covers compass-style Likert tests; lower would "
                         "include noisy multi-policy tests (iSideWith, 8values) "
                         "where the upstream scoring is non-linear.")
    args = ap.parse_args()

    summary: list[dict] = []
    for test in TESTS:
        items_path = os.path.join(args.root, test, "items.json")
        if not os.path.isfile(items_path):
            print(f"  [skip] {test}: no items.json", file=sys.stderr)
            continue
        with open(items_path) as f:
            items_meta = json.load(f)
        info = process_test(test, items_meta, args.extract, args.r2_threshold)
        summary.append(info)
        with open(items_path, "w") as f:
            json.dump(items_meta, f, indent=2)

    # Print summary table
    print("\n--- summary ---")
    for info in summary:
        for axis, fit in info["fit"].items():
            kept = "kept" if fit["r2"] >= args.r2_threshold else "skipped"
            print(f"  {info['test']:30s} {axis:30s} R²={fit['r2']:.3f}  n={fit['n_trials_used']}  {kept}")


if __name__ == "__main__":
    main()
