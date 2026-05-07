"""Debug script for the vLLM server startup failure.

Spins a fresh pod, runs the *exact* `vllm serve` command inspect-ai uses,
streams stderr live so we see the actual error (suppressed by inspect-ai's
wrapper). Tears down on exit.

Usage:
    set -a; source .env; set +a
    PYTHONPATH=. python3 src/runpod_backend/debug_vllm.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Make harbor + our backend importable
HARBOR_VENV = Path.home() / ".local/share/uv/tools/harbor/lib/python3.13/site-packages"
if HARBOR_VENV.exists() and str(HARBOR_VENV) not in sys.path:
    sys.path.insert(0, str(HARBOR_VENV))

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
log = logging.getLogger("debug_vllm")


async def main():
    task_env_config = EnvironmentConfig(
        gpus=1,
        cpus=8,
        memory_mb=65536,
        storage_mb=102400,
        build_timeout_sec=1800.0,
        allow_internet=True,
    )
    trial_dir = REPO_ROOT / "jobs/runpod-debug-vllm/trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)

    RunpodEnvironment.preflight()

    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name="ptb-debug-vllm",
        session_id=f"debug-vllm-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        suppress_override_warnings=True,
    )

    try:
        log.info("=== Starting RunPod environment ===")
        t0 = time.time()
        await env.start(force_build=False)
        log.info(f"Environment ready ({time.time() - t0:.0f}s)")

        # 1. Upload PTB chat templates so vllm can find qwen3.jinja
        log.info("=== Uploading templates ===")
        await env.upload_dir(
            str(REPO_ROOT / "src/eval/templates"),
            "/workspace/ptb_eval/templates",
        )

        # 2. Sanity: confirm vllm cli + python lib both present, what versions
        log.info("=== Verifying vllm install ===")
        await env.exec(
            "python3 -c 'import vllm; print(\"vllm:\", vllm.__version__)' && "
            "python3 -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.version.cuda, \"avail:\", torch.cuda.is_available())' && "
            "nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader && "
            "which vllm && vllm --version 2>&1 | head -3",
            timeout_sec=120,
            tee_logger=True,
        )

        # 3. Run the exact vllm serve command inspect-ai used.
        # Cap at 90s — we only need to see if startup succeeds or what the error is.
        # `vllm serve` will hang serving forever; we explicitly time out and
        # capture whatever stderr it produced before the kill.
        log.info("=== Running vllm serve (90s timeout) ===")
        vllm_cmd = (
            "timeout --signal=SIGTERM --kill-after=10 90 "
            "vllm serve Qwen/Qwen3-1.7B-Base "
            "--host 0.0.0.0 --api-key inspectai "
            "--gpu-memory-utilization 0.85 "
            "--chat-template /workspace/ptb_eval/templates/qwen3.jinja "
            "--port 36216 "
            "2>&1; echo \"[exit: $?]\""
        )
        result = await env.exec(
            vllm_cmd,
            env={"HF_HOME": "/workspace/hf-cache"},
            timeout_sec=180,  # safety margin above the inner 90s
            tee_logger=True,
        )
        log.info(f"vllm exited rc={result.return_code}")

    finally:
        log.info("=== Tearing down pod ===")
        try:
            await env.stop(delete=True)
        except Exception as exc:
            log.error(f"stop() failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
