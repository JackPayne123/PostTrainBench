#!/usr/bin/env python3
"""Pull a self-driving run's artifacts from the persistent volume.

Counterpart to submit_run.py. Spins a tiny recovery pod, attaches the
same volume, rsyncs /workspace/runs/<run_id>/ to local
jobs/runs/<run_id>/, terminates.

Useful any time post-DONE — but you can also pull mid-run for a
snapshot of partial state.

Usage:
    PYTHONPATH=. python src/runpod_backend/pull_run.py <run_id>

Optional --include-final-model copies the LoRA adapter dir too
(default: skipped to avoid the 150 MB rsync on every status check).

If the run was Drive-uploaded, you can also fetch from there directly
(no pod cost) — see https://drive.google.com/drive/folders/1TExh6tQoB1cjE04xiYawZZOQ50WA742D.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from src.runpod_backend.runpod_environment import RunpodEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("pull_run")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument(
        "--include-final-model",
        action="store_true",
        help="Also rsync the final_model adapter dir (~150 MB)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Override local destination (default: jobs/runs/<run_id>/)",
    )
    args = parser.parse_args()

    if args.dst is None:
        args.dst = REPO_ROOT / "jobs" / "runs" / args.run_id
    args.dst.mkdir(parents=True, exist_ok=True)

    trial_dir = args.dst / "_pull_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name=f"pull-{args.run_id}"[:60],
        session_id=f"pull-{int(time.time())}",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=4, memory_mb=16384, storage_mb=20480,
            build_timeout_sec=900.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )
    try:
        log.info("starting recovery pod")
        await env.start(force_build=False)
        remote = f"/workspace/runs/{args.run_id}"
        check = await env.exec(
            f"if [ -d {remote} ]; then du -sh {remote}; else echo MISSING; fi",
            timeout_sec=30,
        )
        out = (check.stdout or "").strip()
        if "MISSING" in out:
            log.error(f"{remote} not on volume; nothing to pull")
            return
        log.info(f"found on volume: {out}")

        if args.include_final_model:
            log.info(f"rsyncing FULL {remote} -> {args.dst}")
            await env.download_dir(remote, str(args.dst))
        else:
            # Pull everything except final_model. Use a staging dir on
            # the pod so rsync's --exclude doesn't have to be wired
            # through env.download_dir.
            staged = f"/workspace/_pull_staging_{int(time.time())}"
            await env.exec(
                f"mkdir -p {staged} && "
                f"rsync -a --exclude final_model {remote}/ {staged}/ && "
                f"echo staged",
                timeout_sec=180,
            )
            log.info(f"rsyncing (excluding final_model/) -> {args.dst}")
            await env.download_dir(staged, str(args.dst))
            await env.exec(f"rm -rf {staged}", timeout_sec=30)
        log.info(f"done. local dst: {args.dst}")
    finally:
        try:
            await env.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
