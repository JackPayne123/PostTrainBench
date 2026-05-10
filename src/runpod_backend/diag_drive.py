#!/usr/bin/env python3
"""Diagnose pod-side rclone Drive config without a full run.

Spins a tiny recovery pod with the configured DEFAULT_IMAGE, then:
  1. Confirms /root/.config/rclone/rclone.conf was baked at build time.
  2. Lists what `drive:` actually resolves to (root_folder_id sanity check).
  3. Performs a real upload + verifies the file appears on the remote.
  4. Tears the pod down.

Usage:
    PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
        src/runpod_backend/diag_drive.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HARBOR_VENV = Path.home() / ".local/share/uv/tools/harbor/lib/python3.13/site-packages"
if HARBOR_VENV.exists() and str(HARBOR_VENV) not in sys.path:
    sys.path.insert(0, str(HARBOR_VENV))

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from src.runpod_backend.runpod_environment import RunpodEnvironment

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("diag_drive")


async def main() -> int:
    trial_dir = REPO_ROOT / "jobs" / "runs" / "_diag_drive_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name="diag-drive",
        session_id=f"diag-{int(time.time())}",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=4, memory_mb=16384, storage_mb=20480,
            build_timeout_sec=900.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )
    rc = 1
    try:
        log.info("starting recovery pod (image = DEFAULT_IMAGE from runpod_environment)")
        await env.start(force_build=False)

        # 1. Config presence + structure (no token leak — first 3 lines only).
        log.info("=== /root/.config/rclone/rclone.conf metadata ===")
        r = await env.exec(
            "ls -la /root/.config/rclone/rclone.conf 2>&1; "
            "echo ---; head -3 /root/.config/rclone/rclone.conf 2>&1; "
            "echo ---; grep -E '^(team_drive|root_folder_id)' /root/.config/rclone/rclone.conf 2>&1",
            timeout_sec=30,
        )
        log.info(f"rc={r.return_code}\n{r.stdout or r.stderr}")
        if r.return_code != 0:
            log.error("could not read /root/.config/rclone/rclone.conf — image probably built without --secret id=rclone_conf")
            return 2

        # 2. Resolve drive: remote.
        log.info("=== drive: lsd ===")
        r = await env.exec("rclone lsd drive: 2>&1", timeout_sec=60)
        log.info(f"rc={r.return_code}\n{r.stdout or r.stderr}")

        log.info("=== drive: lsjson --max-depth 1 ===")
        r = await env.exec("rclone lsjson drive: --max-depth 1 2>&1 | head -60",
                           timeout_sec=60)
        log.info(f"rc={r.return_code}\n{r.stdout or r.stderr}")

        # 3. Real upload test, no pipe-eats-rc nonsense.
        log.info("=== upload test → drive:_pod_smoke/ ===")
        r = await env.exec(
            "echo \"diag $(date -u +%FT%TZ) pod=$RUNPOD_POD_ID\" > /tmp/ptb-pod-smoke.txt && "
            "rclone copy /tmp/ptb-pod-smoke.txt drive:_pod_smoke/ -vv 2>&1; "
            "echo RCLONE_RC=$?",
            timeout_sec=120,
        )
        log.info(f"return_code={r.return_code}")
        out = r.stdout or r.stderr or ""
        log.info(f"last 3KB:\n{out[-3000:]}")

        # Parse trailing RCLONE_RC=N as the truth (env.exec rc may include
        # both commands joined by `;`).
        rclone_rc = None
        for line in out.splitlines()[::-1]:
            if line.startswith("RCLONE_RC="):
                try:
                    rclone_rc = int(line.split("=", 1)[1])
                except ValueError:
                    pass
                break
        log.info(f"parsed RCLONE_RC={rclone_rc}")

        # 4. Verify the file is actually visible from drive:.
        log.info("=== drive:_pod_smoke/ listing ===")
        r = await env.exec("rclone lsl drive:_pod_smoke/ 2>&1 | head -20",
                           timeout_sec=60)
        log.info(f"rc={r.return_code}\n{r.stdout or r.stderr}")

        if rclone_rc == 0 and "ptb-pod-smoke.txt" in (r.stdout or ""):
            log.info("✅ image rclone config WORKS — :9 is fine, only the "
                     "tail-pipe bug masked previous attempts")
            rc = 0
        else:
            log.error("❌ image rclone config BROKEN — see logs above for "
                     "auth (403), folder (404), or missing-config error")
            rc = 3

    finally:
        log.info("tearing down recovery pod")
        try:
            await env.stop(delete=True)
        except Exception as exc:
            log.warning(f"stop failed: {exc}")
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
