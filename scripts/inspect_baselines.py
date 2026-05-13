#!/usr/bin/env python3
"""Print a clean tabular summary of a model's baseline inventory.

Use this instead of writing ad-hoc Python against `_index__limit*.json` —
the schema keys (`tasks` dict, `model`/`image`/`git_sha` provenance) are
non-obvious and easy to misread. Probing with wrong keys returns None
and falsely flags the index as broken (caught 2026-05-13 misreading
qwen3.5-9b inventory).

Usage:
  python3 scripts/inspect_baselines.py qwen_qwen3.5-9b
  python3 scripts/inspect_baselines.py --all
  python3 scripts/inspect_baselines.py qwen_qwen3.5-9b --limit 100
  python3 scripts/inspect_baselines.py qwen_qwen3.5-9b --check
    # --check exits 1 if any expected bench (per src.evals.registry.EVAL_SUITE)
    # is missing from the index.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = REPO_ROOT / "baselines"


def list_model_slugs() -> list[str]:
    if not BASELINES.exists():
        return []
    return sorted(p.name for p in BASELINES.iterdir() if p.is_dir())


def print_model(slug: str, only_limit: int | None = None, check: bool = False) -> int:
    """Print summary for one model slug. Returns rc (0 ok, 1 missing benches)."""
    dst = BASELINES / slug
    if not dst.is_dir():
        print(f"ERROR: {dst} not found", file=sys.stderr)
        return 2

    # Group index files by limit.
    indices = sorted(dst.glob("_index__limit*.json"))
    if not indices:
        print(f"  (no _index__limit*.json files under {dst})")
        return 0

    rc = 0
    for idx_path in indices:
        limit = int(idx_path.stem.split("limit", 1)[1])
        if only_limit is not None and limit != only_limit:
            continue
        try:
            d = json.loads(idx_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  {idx_path.name}: UNREADABLE ({e})")
            rc = max(rc, 2)
            continue
        tasks = d.get("tasks") or {}
        print(
            f"\n  {idx_path.name}"
            f"  model={d.get('model','?')}"
            f"  image={d.get('image','?').split(':')[-1] if d.get('image') else '?'}"
            f"  git={d.get('git_sha','?')}"
            f"  computed={d.get('computed_at','?')[:19]}"
        )
        if not tasks:
            print(f"    (empty `tasks` dict — likely a partial run)")
            continue
        print(f"    tasks ({len(tasks)}):")
        for name in sorted(tasks.keys()):
            t = tasks[name]
            hv = t.get("headline_value")
            hm = t.get("headline_metric")
            cat = t.get("category", "?")
            attr = t.get("attribute") or ""
            hibetter = t.get("higher_is_better")
            polarity = ""
            if hibetter is False:
                polarity = " (lower=better)"
            elif hibetter is True:
                polarity = " (higher=better)"
            tlim = t.get("limit", "?")
            if isinstance(hv, (int, float)):
                hv_str = f"{hv:.4f}"
            elif hv is None:
                hv_str = "multi-dim"
            else:
                hv_str = str(hv)
            print(f"      {name:30} {cat:10} {hm or '-':35} {hv_str:12} (limit={tlim}){polarity}")

    if check:
        # Compare to EVAL_SUITE for any missing benches.
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from src.evals.registry import EVAL_SUITE  # type: ignore
        except ImportError:
            print("  (cannot import EVAL_SUITE — skip --check)")
            return rc
        for idx_path in indices:
            limit = int(idx_path.stem.split("limit", 1)[1])
            if only_limit is not None and limit != only_limit:
                continue
            d = json.loads(idx_path.read_text())
            tasks = set((d.get("tasks") or {}).keys())
            expected = set(EVAL_SUITE.keys())
            missing = expected - tasks
            if missing and limit == 0:
                # limit=0 (full per-eval-default suite) should have all 23
                print(f"\n  ⚠️  {idx_path.name} missing benches: {sorted(missing)}")
                rc = max(rc, 1)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_slug", nargs="?", help="e.g. 'qwen_qwen3.5-9b'")
    p.add_argument("--all", action="store_true", help="all model slugs under baselines/")
    p.add_argument("--limit", type=int, default=None, help="only show this --limit value")
    p.add_argument("--check", action="store_true",
                   help="exit 1 if any bench in EVAL_SUITE is missing from the limit=0 index")
    args = p.parse_args()

    if args.all:
        slugs = list_model_slugs()
    elif args.model_slug:
        slugs = [args.model_slug]
    else:
        slugs = list_model_slugs()
        print("Available models:")
        for s in slugs:
            print(f"  {s}")
        return 0

    rc = 0
    for slug in slugs:
        print(f"\n=== {slug} ===")
        rc = max(rc, print_model(slug, args.limit, args.check))
    return rc


if __name__ == "__main__":
    sys.exit(main())
