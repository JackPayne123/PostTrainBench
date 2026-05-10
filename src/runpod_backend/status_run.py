#!/usr/bin/env python3
"""Check status of a submit_run.py experiment without SSH-ing.

Two-tier lookup:
  1. Query Runpod GraphQL for any pod whose name encodes <run_id>.
     If a RUNNING pod exists, the run is in flight; print last 20 log
     lines via SSH and a tail-log hint.
  2. If no live pod (or DONE sentinel exists on the volume), spin a
     tiny recovery pod, attach the same volume, read
     /workspace/runs/<run_id>/{DONE, run.log, summary.json}, print,
     terminate.

Cost of step 2: ~$0.01-0.02 per query (a 60-second 3090 pod).

Usage:
    PYTHONPATH=. python src/runpod_backend/status_run.py <run_id>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from src.runpod_backend.runpod_environment import (
    RUNPOD_API_URL,
    RunpodEnvironment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("status_run")


def find_running_pod(run_id: str) -> dict | None:
    """Query Runpod for a RUNNING pod whose name contains run_id."""
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not in env")
    query = (
        '{"query":"query { myself { pods { id name desiredStatus '
        'runtime { ports { ip publicPort privatePort isIpPublic } } } } }"}'
    )
    with httpx.Client(timeout=30.0) as c:
        r = c.post(
            RUNPOD_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            content=query,
        )
        r.raise_for_status()
        data = r.json()
    target = run_id.lower()
    for p in data["data"]["myself"]["pods"]:
        if p["desiredStatus"] != "RUNNING":
            continue
        if target not in (p["name"] or "").lower():
            continue
        rt = p.get("runtime") or {}
        for port in (rt.get("ports") or []):
            if port["privatePort"] == 22 and port["isIpPublic"]:
                return {
                    "id": p["id"],
                    "name": p["name"],
                    "ip": port["ip"],
                    "port": port["publicPort"],
                }
    return None


async def peek_via_recovery_pod(run_id: str) -> dict:
    """Spin a tiny pod, attach volume, read run state, terminate."""
    trial_dir = REPO_ROOT / "jobs" / "runs" / "_status_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name=f"status-{run_id}"[:60],
        session_id=f"status-{int(time.time())}",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=4, memory_mb=16384, storage_mb=20480,
            build_timeout_sec=900.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )
    out: dict = {}
    try:
        await env.start(force_build=False)
        remote = f"/workspace/runs/{run_id}"
        check = await env.exec(
            f"if [ -d {remote} ]; then "
            f"  echo present; "
            f"  if [ -f {remote}/DONE ]; then echo DONE_FOUND; cat {remote}/DONE; fi; "
            f"  if [ -f {remote}/summary.json ]; then echo SUMMARY; cat {remote}/summary.json; fi; "
            f"  if [ -f {remote}/run.log ]; then echo LAST_LINES; tail -20 {remote}/run.log; fi; "
            f"else echo MISSING; fi",
            timeout_sec=60,
        )
        out["raw_stdout"] = check.stdout or ""
    finally:
        try:
            await env.stop()
        except Exception:
            pass
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument(
        "--via-recovery-pod",
        action="store_true",
        help="Always spin a recovery pod even if a live pod is found "
             "(useful when the live pod's SSH is flaky).",
    )
    args = parser.parse_args()

    live = find_running_pod(args.run_id)
    if live and not args.via_recovery_pod:
        print(f"\n=== {args.run_id} ===")
        print(f"  STATE: RUNNING")
        print(f"  POD:   {live['id']}  ({live['ip']}:{live['port']})")
        print()
        print("Live pod available. Tail log:")
        print(f"  bash src/runpod_backend/tail_log.sh {args.run_id}")
        print()
        print("Or rerun with --via-recovery-pod to peek volume state instead.")
        return

    print(f"\n=== {args.run_id} ===")
    print(f"  STATE: {'RUNNING (live skipped)' if live else 'NO LIVE POD'}")
    print("  Spinning recovery pod to read /workspace/runs/<run_id>/...")
    info = await peek_via_recovery_pod(args.run_id)
    raw = info.get("raw_stdout", "")
    if "MISSING" in raw:
        print(f"\n  No data on volume. Either run never started, or volume was wiped.")
        return
    print()
    print(raw)


if __name__ == "__main__":
    asyncio.run(main())
