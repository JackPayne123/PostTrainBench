#!/usr/bin/env python3
"""Pull a baseline run + promote its per-bench metrics into the repo.

Two phases:
  1. Pull the run dir from Drive (default) or volume to laptop. Same
     mechanic as pull_run.py.
  2. Promote each `baselines/<bench>__limit<N>.json` from the pulled
     run dir to `baselines/<model_slug>/<bench>__limit<N>.json` in the
     repo root, so future `submit_run.py --use-baseline` runs can
     resolve them.

Usage:
    PYTHONPATH=. python src/runpod_backend/pull_baseline.py <run_id>

The run_id MUST be a baseline run (submit_baseline.py output). Sanity-
checked by looking for `kind: baseline` in config.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("pull_baseline")

LAPTOP_RCLONE_CONFIG = Path.home() / ".config/ptb/rclone.conf"


def pull_from_drive(run_id: str, dst: Path) -> int:
    if not LAPTOP_RCLONE_CONFIG.exists() or shutil.which("rclone") is None:
        log.error("rclone or config missing on laptop; install rclone + populate ~/.config/ptb/rclone.conf")
        return 2
    cmd = [
        "rclone", "--config", str(LAPTOP_RCLONE_CONFIG),
        "copy", f"drive:{run_id}/", str(dst),
        "--stats", "1s", "--stats-one-line",
        "--transfers", "4", "--checkers", "8",
    ]
    log.info(f"rclone copy drive:{run_id}/ → {dst}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    return proc.wait()


def promote(run_dir: Path, repo_root: Path) -> int:
    """Copy per-bench baselines + index into repo `baselines/<slug>/`."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        log.error(f"missing {cfg_path}; cannot determine model_slug")
        return 3
    cfg = json.loads(cfg_path.read_text())
    kind = cfg.get("kind")
    if kind not in ("baseline", "adapter_eval"):
        log.error(f"config.json kind={kind!r}, expected 'baseline' or "
                  "'adapter_eval'; use pull_run.py for agent runs")
        return 4

    model_slug = cfg["model_slug"]
    limit = cfg["limit"]
    src_baselines = run_dir / "baselines"
    if not src_baselines.exists():
        log.error(f"no baselines/ dir at {src_baselines}; pod may have failed mid-run")
        return 5

    # Adapter-eval runs land under `baselines/<model_slug>/adapter_eval/<adapter-run-id>/`
    # so they don't collide with the base-model baselines and so we can
    # track multiple adapter evals per model (one per trained adapter).
    if kind == "adapter_eval":
        adapter_id = cfg.get("adapter_from_run_id", "unknown")
        dst_dir = repo_root / "baselines" / model_slug / "adapter_eval" / adapter_id
    else:
        dst_dir = repo_root / "baselines" / model_slug
    dst_dir.mkdir(parents=True, exist_ok=True)

    n_promoted = 0
    for src in sorted(src_baselines.glob("*.json")):
        dst = dst_dir / src.name
        # Don't clobber a newer baseline if one exists (e.g. an
        # interrupted re-run from a different commit_sha). Compare
        # computed_at timestamps and keep the newer entry.
        if dst.exists():
            existing = json.loads(dst.read_text())
            incoming = json.loads(src.read_text())
            if existing.get("computed_at", "") >= incoming.get("computed_at", ""):
                log.info(f"  skip {src.name} (existing baseline newer)")
                continue
        shutil.copy2(src, dst)
        n_promoted += 1

    # Promote the index too (useful as a model-level summary).
    #
    # Schema of _index__limit<N>.json (do NOT rename without updating the
    # `scripts/inspect_baselines.py` helper that reads this):
    #   {
    #     "model": "Qwen/Qwen3.5-9B",            # HF model id
    #     "model_slug": "qwen_qwen3.5-9b",       # filename-safe slug
    #     "limit": 100 | 0,                       # 0 = per-eval default
    #     "image": "ghcr.io/jackpayne123/ptb-base:NN",  # pod image used
    #     "git_sha": "abc1234",                  # repo SHA at compute time
    #     "computed_at": "ISO-8601 UTC",         # newest wins on re-promote
    #     "tasks": {                              # dict (not list); keyed by bench
    #       "<bench>": {
    #         "benchmark": "<bench>",
    #         "category": "capability|safety|character",
    #         "attribute": "<safety-attr>" | null,
    #         "higher_is_better": bool | null,
    #         "headline_metric": "accuracy" | "model_graded_qa.total" | null,
    #         "headline_value": float | null,
    #         "limit": N,
    #         "metrics": { full per-bench metrics dict },
    #       },
    #       ...
    #     }
    #   }
    #
    # Common probe mistakes (caught 2026-05-13 by misreading the schema):
    #   - `d.get("evals")` → returns None. Use `d["tasks"]` (a dict).
    #   - `d.get("source_run_id")` → returns None. Use `d["image"]` +
    #     `d["computed_at"]` to identify provenance.
    #   - Iterating `d["tasks"]` as a list — it's a dict, use `.items()`.
    idx_src = run_dir / "baselines.json"
    if idx_src.exists():
        shutil.copy2(idx_src, dst_dir / f"_index__limit{limit}.json")

    log.info(f"promoted {n_promoted} baselines to {dst_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--skip-pull", action="store_true",
                        help="skip Drive pull, assume run already on laptop")
    args = parser.parse_args()

    run_dir = REPO_ROOT / "jobs" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_pull:
        rc = pull_from_drive(args.run_id, run_dir)
        if rc != 0:
            log.error(f"pull failed rc={rc}")
            return rc

    return promote(run_dir, REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
