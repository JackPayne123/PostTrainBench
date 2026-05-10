#!/usr/bin/env python3
"""Pull a self-driving run's artifacts to laptop.

Two sources, picked automatically:
  - **Drive (default)**: rclone copy from `drive:<run_id>/` to local
    `jobs/runs/<run_id>/`. No pod cost, no boot delay. The pod uploads
    everything before self-terminate via run_experiment.rclone_to_drive,
    so post-DONE this is always the freshest source.
  - **Volume (`--from-volume`)**: spin a recovery pod, mount the
    persistent volume, rsync `/workspace/runs/<run_id>/` to laptop.
    Use this mid-run (Drive doesn't have it yet) or if Drive is
    unreachable for some reason.

Usage:
    PYTHONPATH=. python src/runpod_backend/pull_run.py <run_id>
    PYTHONPATH=. python src/runpod_backend/pull_run.py <run_id> --from-volume

Optional --include-final-model copies the LoRA adapter dir too
(default: skipped to avoid the 150 MB pull on every status check;
applies to both sources).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("pull_run")

LAPTOP_RCLONE_CONFIG = Path.home() / ".config/ptb/rclone.conf"


def pull_from_drive(run_id: str, dst: Path, include_final_model: bool) -> int:
    """rclone copy from drive:<run_id>/ → dst/. Returns 0 on success."""
    if not LAPTOP_RCLONE_CONFIG.exists():
        log.error(
            f"rclone config missing at {LAPTOP_RCLONE_CONFIG}. "
            "Drive-pull needs the same `[drive]` config that's baked into "
            "the image. Regenerate per docs/skill or fall back to "
            "--from-volume."
        )
        return 2
    if shutil.which("rclone") is None:
        log.error("rclone not on PATH locally. brew install rclone (or "
                  "use --from-volume).")
        return 2

    exclude_args = [] if include_final_model else ["--exclude", "final_model/**"]
    # --stats-one-line + 1s interval keeps each progress update on its own
    # line so it interleaves cleanly with caller output. --progress's
    # animated multi-line repaint doesn't survive an unbuffered pipe.
    cmd = [
        "rclone",
        "--config", str(LAPTOP_RCLONE_CONFIG),
        "copy",
        f"drive:{run_id}/",
        str(dst),
        "--stats", "1s",
        "--stats-one-line",
        "--transfers", "4",
        "--checkers", "8",
        *exclude_args,
    ]
    log.info(f"rclone copy drive:{run_id}/ -> {dst}"
             + (" (excluding final_model/)" if not include_final_model else " (full)"))
    # Stream stdout+stderr live so the user sees byte counts during
    # multi-MB pulls. subprocess.run(capture_output=True) would silently
    # buffer for the full transfer duration.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    rc = proc.wait()
    if rc != 0:
        log.error(f"rclone rc={rc}")
        return rc
    log.info(f"done. local dst: {dst}")
    return 0


async def pull_from_volume(run_id: str, dst: Path, include_final_model: bool) -> int:
    """Spin recovery pod, mount volume, rsync run dir."""
    # Local import — Harbor stack is heavyweight, Drive path doesn't need it.
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths
    from src.runpod_backend.runpod_environment import RunpodEnvironment

    trial_dir = dst / "_pull_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name=f"pull-{run_id}"[:60],
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
        remote = f"/workspace/runs/{run_id}"
        check = await env.exec(
            f"if [ -d {remote} ]; then du -sh {remote}; else echo MISSING; fi",
            timeout_sec=30,
        )
        out = (check.stdout or "").strip()
        if "MISSING" in out:
            log.error(f"{remote} not on volume; nothing to pull")
            return 3
        log.info(f"found on volume: {out}")

        if include_final_model:
            log.info(f"rsyncing FULL {remote} -> {dst}")
            await env.download_dir(remote, str(dst))
        else:
            staged = f"/workspace/_pull_staging_{int(time.time())}"
            await env.exec(
                f"mkdir -p {staged} && "
                f"rsync -a --exclude final_model {remote}/ {staged}/ && "
                f"echo staged",
                timeout_sec=180,
            )
            log.info(f"rsyncing (excluding final_model/) -> {dst}")
            await env.download_dir(staged, str(dst))
            await env.exec(f"rm -rf {staged}", timeout_sec=30)
        log.info(f"done. local dst: {dst}")
        return 0
    finally:
        try:
            await env.stop()
        except Exception:
            pass


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument(
        "--include-final-model",
        action="store_true",
        help="Also pull the final_model adapter dir (~150 MB)",
    )
    parser.add_argument(
        "--from-volume",
        action="store_true",
        help="Pull from RunPod persistent volume (recovery pod) instead "
             "of the default Drive source. Use mid-run or if Drive is "
             "unreachable.",
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

    if args.from_volume:
        return await pull_from_volume(
            args.run_id, args.dst, args.include_final_model,
        )
    return pull_from_drive(args.run_id, args.dst, args.include_final_model)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
