#!/usr/bin/env python3
"""Pull inspect-ai per-sample eval logs from the volume.

The /workspace/ volume is shared across all pods we spin in the same
RunPod org, so any prior pod's writes to /workspace/ptb_eval/<bench>/logs/
persist after that pod terminates. This tool spins a tiny pod, mounts
the volume, lists + rsyncs the eval logs back to a local destination
so we can inspect sample-level dialogues offline.

Usage:
    PYTHONPATH=. python src/runpod_backend/pull_eval_logs.py \\
        --benchmark sycophancy \\
        --dst jobs/runs/2026-05-09_20-27_E_..._seed0/eval_logs

If --since "2026-05-09 20:00" is set, only files newer than that are
pulled (handy when /workspace/ptb_eval/<bench>/logs/ has accumulated
logs from many runs and you only want the recent ones).
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
log = logging.getLogger("pull_eval_logs")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="e.g. sycophancy, gsm8k")
    parser.add_argument("--dst", required=True, type=Path,
                        help="Local destination directory")
    parser.add_argument(
        "--since",
        default=None,
        help='Only pull files newer than this UTC datetime, e.g. "2026-05-09 20:00"',
    )
    args = parser.parse_args()

    if not args.dst.is_absolute():
        args.dst = (REPO_ROOT / args.dst).resolve()
    args.dst.mkdir(parents=True, exist_ok=True)

    trial_paths = TrialPaths(trial_dir=args.dst / "_pull_trial")
    (args.dst / "_pull_trial").mkdir(parents=True, exist_ok=True)

    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name=f"pull-eval-logs-{args.benchmark}",
        session_id=f"pull-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=4, memory_mb=16384, storage_mb=20480,
            build_timeout_sec=900.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )

    try:
        log.info("starting recovery pod ...")
        await env.start(force_build=False)

        remote_src = f"/workspace/ptb_eval/{args.benchmark}/logs"
        check = await env.exec(
            f"if [ -d {remote_src} ]; then "
            f"  ls -la {remote_src} | head -20; "
            f"  echo ---total_count---; "
            f"  ls {remote_src} | wc -l; "
            f"  echo ---total_size---; "
            f"  du -sh {remote_src}; "
            f"else echo MISSING; fi",
            timeout_sec=30,
        )
        log.info(f"remote dir state:\n{check.stdout or ''}")
        if "MISSING" in (check.stdout or ""):
            log.error(f"{remote_src} not on volume — nothing to pull")
            return

        # If --since is set, copy only matching files into a staging dir
        # before rsync (rsync's --files-from is awkward over remote).
        src_to_pull = remote_src
        if args.since:
            staged = f"/workspace/_pull_staging_{int(time.time())}"
            await env.exec(
                f"mkdir -p {staged} && "
                f"find {remote_src} -newermt {args.since!r} -name '*.json' "
                f"-exec cp {{}} {staged}/ \\; && "
                f"echo staged $(ls {staged} | wc -l) files",
                timeout_sec=120,
            )
            src_to_pull = staged

        log.info(f"rsyncing {src_to_pull} -> {args.dst}")
        await env.download_dir(src_to_pull, str(args.dst))
        log.info(f"done. local dst: {args.dst}")

    finally:
        try:
            await env.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
