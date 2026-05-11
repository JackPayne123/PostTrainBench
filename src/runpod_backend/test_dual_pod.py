#!/usr/bin/env python3
"""Test whether two pods can attach the same persistent volume.

Spins two minimal pods in parallel both requesting networkVolumeId =
qwe92egpys (jack-pilot-cz). If both come up RUNNING, multi-attach
works on EU-CZ-1 SSD network volumes and we can run baseline + agent
pods concurrently. If the second `podFindAndDeployOnDemand` returns
None or the pod stalls in CREATED, multi-attach is not supported.

Tears both pods down at the end regardless of outcome.

Usage:
    PYTHONPATH=. python src/runpod_backend/test_dual_pod.py
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
log = logging.getLogger("dual_pod")


async def spin(name: str) -> tuple[str, RunpodEnvironment | None, Exception | None]:
    trial_dir = REPO_ROOT / "jobs" / "runs" / f"_dual_{name}_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name=f"dual-{name}",
        session_id=f"dual-{name}-{int(time.time())}",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=4, memory_mb=16384, storage_mb=20480,
            build_timeout_sec=900.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )
    try:
        log.info(f"[{name}] starting pod (same networkVolumeId as other pod)")
        await env.start(force_build=False)
        log.info(f"[{name}] RUNNING — pod={env._pod_id} ssh={env._ssh_host}:{env._ssh_port}")
        return (name, env, None)
    except Exception as exc:
        log.error(f"[{name}] FAILED to start: {type(exc).__name__}: {exc}")
        return (name, env, exc)


async def main() -> int:
    log.info("=== dual-pod volume attach test ===")
    log.info("Spinning two pods in parallel, both attaching networkVolumeId qwe92egpys")

    results = await asyncio.gather(
        spin("A"), spin("B"), return_exceptions=False,
    )

    a_name, a_env, a_err = results[0]
    b_name, b_env, b_err = results[1]
    a_ok = a_err is None and a_env is not None and getattr(a_env, "_pod_id", None)
    b_ok = b_err is None and b_env is not None and getattr(b_env, "_pod_id", None)

    log.info(f"=== verdict: A={'RUNNING' if a_ok else 'FAILED'} B={'RUNNING' if b_ok else 'FAILED'} ===")

    if a_ok and b_ok:
        log.info("MULTI-ATTACH SUPPORTED. Running baseline + agent pods in parallel is safe.")
        rc = 0
        # Quick smoke: each pod independently writes a file to the
        # shared volume, then both list the volume to confirm visibility.
        log.info("smoke: writing per-pod sentinels under /workspace/_dual_test/")
        for env in (a_env, b_env):
            await env.exec(
                f"mkdir -p /workspace/_dual_test && "
                f"echo $(hostname) > /workspace/_dual_test/from_${{HOSTNAME}}.txt 2>&1; "
                f"ls /workspace/_dual_test/",
                timeout_sec=30,
            )
        for env in (a_env, b_env):
            r = await env.exec("ls /workspace/_dual_test/ 2>&1", timeout_sec=15)
            log.info(f"[{env._pod_id}] sees /workspace/_dual_test/:\n{r.stdout or r.stderr}")
        # Cleanup the test dir from pod A.
        if a_env:
            await a_env.exec("rm -rf /workspace/_dual_test 2>&1 || true", timeout_sec=15)
    else:
        log.error("MULTI-ATTACH NOT SUPPORTED. Fall back to sequential or use a "
                  "separate volume for one of the pods.")
        rc = 1

    # Tear down both.
    for name, env, _err in results:
        if env is not None and getattr(env, "_pod_id", None):
            log.info(f"tearing down pod {name} ({env._pod_id})")
            try:
                await env.stop(delete=True)
            except Exception as exc:
                log.warning(f"stop failed for {name}: {exc}")
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
