#!/usr/bin/env python3
"""Compute per-eval deltas between a base baseline and an adapter-eval baseline.

Reads:
  baselines/<model_slug>/<bench>__limit<N>.json
  baselines/<model_slug>/adapter_eval/<adapter-run-id>/<bench>__limit<N>.json

Writes:
  jobs/runs/<adapter-run-id>/deltas.json

Prints a markdown table grouped by registry category. For tasks with a
scalar `headline_metric`, the delta is `adapter - base`; "good"/"bad"
direction comes from `info.higher_is_better`. For multi-dim tasks
(`headline_metric=None`) every common top-level key in `metrics` is
delta'd and rendered as a sub-table.
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

from src.evals.registry import EVAL_SUITE, get_headline  # noqa: E402


def fmt(x: float | None, w: int = 7) -> str:
    if x is None:
        return f"{'—':>{w}}"
    return f"{x:>{w}.3f}"


def fmt_delta(d: float | None, higher_is_better: bool, w: int = 8) -> str:
    if d is None:
        return f"{'—':>{w}}"
    arrow = ""
    if abs(d) >= 0.005:
        good = (d > 0) == higher_is_better
        arrow = " ✓" if good else " ✗"
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.3f}{arrow:>{max(0, w-len(f'{sign}{d:.3f}'))}}"


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"WARN: failed to parse {p}: {e}", file=sys.stderr)
        return None


def flatten(d: dict, prefix: str = "") -> dict[str, float | int]:
    """One-level flatten of nested dicts into dotted keys; keep scalars only."""
    out: dict[str, float | int] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def _backfill_summary(adapter_run_id: str, deltas: dict) -> int:
    """Backfill summary.json with pre/post/delta for benchmarks the adapter
    run measured. Identifies the primary benchmark + extra_evals from the
    F-run's config.json (NOT the adapter-eval's, since the F-run is what
    the summary describes); falls back to the adapter-eval if F-run absent.

    The summary's `pre` was None (from --skip-pre-eval); after this it
    holds the base baseline's metric so the existing tooling (trace
    viewer, etc.) reads scores from the same slot as legacy runs. Tagged
    `delta_method="baseline-backfill"` so it's traceable.
    """
    summary_path = REPO_ROOT / "jobs" / "runs" / adapter_run_id / "summary.json"
    if not summary_path.exists():
        print(f"WARN: no summary.json at {summary_path} — nothing to backfill", file=sys.stderr)
        return 0
    summary = json.loads(summary_path.read_text())
    cfg = summary.get("config", {})
    primary = cfg.get("benchmark")
    extras_raw = (cfg.get("extra", {}) or {}).get("extra_evals") or ""
    extras = [b.strip() for b in extras_raw.split(",") if b.strip()] if isinstance(extras_raw, str) else []
    benches = [b for b in [primary, *extras] if b]
    if not benches:
        print("WARN: no primary benchmark in summary config — nothing to backfill", file=sys.stderr)
        return 0

    n = 0
    for b in benches:
        row = deltas.get(b)
        if not row or row.get("kind") != "scalar":
            continue
        base = row.get("base")
        adapter = row.get("adapter")
        delta = row.get("delta")
        if base is None and adapter is None:
            continue
        slot_pre = "pre" if b == primary else f"pre_{b}"
        slot_post = "post" if b == primary else f"post_{b}"
        slot_delta = "delta" if b == primary else f"delta_{b}"
        if isinstance(base, (int, float)):
            summary[slot_pre] = {"accuracy": float(base)}
        if isinstance(adapter, (int, float)):
            existing = summary.get(slot_post)
            if isinstance(existing, dict):
                existing["accuracy"] = float(adapter)
            else:
                summary[slot_post] = {"accuracy": float(adapter)}
        if isinstance(delta, (int, float)):
            summary[slot_delta] = float(delta)
        n += 1

    summary["delta_method"] = "baseline-backfill"
    summary["baseline_backfill_at"] = adapter_run_id
    summary_path.write_text(json.dumps(summary, indent=2))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("adapter_run_id", help="Adapter (F-run) ID — the trained-adapter run")
    ap.add_argument("--model-slug", default="qwen_qwen3-1.7b")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--write-deltas", action="store_true",
                    help="Write jobs/runs/<adapter-run-id>/deltas.json")
    ap.add_argument("--update-summary", action="store_true",
                    help="Backfill summary.json's pre/post/delta for the primary "
                         "benchmark (+ extra_evals) from the baseline-derived deltas, "
                         "so the trace viewer + downstream tooling resolve scores the "
                         "same way as legacy paired runs. Adds delta_method="
                         "'baseline-backfill' provenance.")
    args = ap.parse_args()

    base_dir = REPO_ROOT / "baselines" / args.model_slug
    adapter_dir = base_dir / "adapter_eval" / args.adapter_run_id
    if not adapter_dir.exists():
        print(f"ERROR: adapter dir not found: {adapter_dir}", file=sys.stderr)
        return 2

    deltas: dict[str, dict] = {}
    by_cat: dict[str, list[dict]] = defaultdict(list)

    for name, info in EVAL_SUITE.items():
        fname = f"{name}__limit{args.limit}.json"
        base = load(base_dir / fname)
        adp = load(adapter_dir / fname)
        base_m = (base or {}).get("metrics")
        adp_m = (adp or {}).get("metrics")

        row: dict = {
            "bench": name,
            "category": info.category,
            "attribute": info.attribute,
            "higher_is_better": info.higher_is_better,
            "headline_metric": info.headline_metric,
            "base_ok": base_m is not None,
            "adapter_ok": adp_m is not None,
        }

        if info.headline_metric is not None:
            base_h = get_headline(base_m or {}, info) if base_m else None
            adp_h = get_headline(adp_m or {}, info) if adp_m else None
            delta = None
            if base_h is not None and adp_h is not None:
                delta = adp_h - base_h
            row.update({
                "base": base_h, "adapter": adp_h, "delta": delta,
                "kind": "scalar",
            })
        else:
            base_flat = flatten(base_m) if base_m else {}
            adp_flat = flatten(adp_m) if adp_m else {}
            keys = set(base_flat) | set(adp_flat)
            sub: dict[str, dict] = {}
            for k in sorted(keys):
                b = base_flat.get(k)
                a = adp_flat.get(k)
                d = (a - b) if isinstance(b, (int, float)) and isinstance(a, (int, float)) else None
                sub[k] = {"base": b, "adapter": a, "delta": d}
            row.update({"kind": "multi", "sub": sub})

        deltas[name] = row
        by_cat[info.category].append(row)

    if args.write_deltas:
        out = REPO_ROOT / "jobs" / "runs" / args.adapter_run_id / "deltas.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(deltas, indent=2))
        print(f"wrote {out}\n")

    if args.update_summary:
        n_back = _backfill_summary(args.adapter_run_id, deltas)
        if n_back:
            print(f"backfilled summary.json with pre/post/delta for {n_back} benchmark(s)\n")

    cat_titles = {"capability": "Capability", "safety": "Safety", "character": "Character"}
    for cat in ("capability", "safety", "character"):
        rows = [r for r in by_cat[cat] if r["kind"] == "scalar"]
        if rows:
            print(f"\n### {cat_titles[cat]} (scalar headline)\n")
            print("| Bench | Base | Adapter | Δ | Dir |")
            print("|---|---|---|---|---|")
            for r in rows:
                dir_str = "↑good" if r["higher_is_better"] else "↓good"
                print(f"| {r['bench']} | {fmt(r['base'])} | {fmt(r['adapter'])} | "
                      f"{fmt_delta(r['delta'], r['higher_is_better'])} | {dir_str} |")

        multi = [r for r in by_cat[cat] if r["kind"] == "multi"]
        for r in multi:
            print(f"\n### {cat_titles[cat]} — {r['bench']} (multi-dim)\n")
            if not r["sub"]:
                print("(no metrics — both base and adapter missing)")
                continue
            print("| Metric | Base | Adapter | Δ |")
            print("|---|---|---|---|")
            for k, v in r["sub"].items():
                print(f"| {k} | {fmt(v['base'])} | {fmt(v['adapter'])} | "
                      f"{fmt_delta(v['delta'], r['higher_is_better'])} |")

    n_base_fail = sum(1 for r in deltas.values() if not r["base_ok"])
    n_adp_fail = sum(1 for r in deltas.values() if not r["adapter_ok"])
    print(f"\n_Summary: {len(deltas)} evals; base missing/null={n_base_fail}, adapter missing/null={n_adp_fail}_")
    return 0


if __name__ == "__main__":
    sys.exit(main())
