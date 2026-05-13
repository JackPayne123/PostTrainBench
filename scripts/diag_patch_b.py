#!/usr/bin/env python3
"""One-shot diag for Patch B: score_runner.sh adapter config.json staging.

Spins a recovery pod, mocks a fake adapter dir + fake HF cache entry,
invokes /opt/pipeline-bin/score_runner.sh as the agent, asserts that
the patched staging block copied the base model's config.json into
the adapter dir BEFORE evaluate.py runs.

The evaluate.py invocation downstream will fail (no real vllm), but
that's not what we're testing.

Usage:
    PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python scripts/diag_patch_b.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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
log = logging.getLogger("diag_patch_b")


TEST_SCRIPT = r"""
set +e
echo === SETUP ===

# 1. Mock HF cache entry. Staging code: find $cache_root/hub/<slug>/snapshots -name config.json
mkdir -p /workspace/hf-cache/hub/models--Qwen--Qwen3-1.7B/snapshots/abc123def
cat > /workspace/hf-cache/hub/models--Qwen--Qwen3-1.7B/snapshots/abc123def/config.json <<EOF
{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "_diag_marker": "PATCH_B_OK"}
EOF
ls -la /workspace/hf-cache/hub/models--Qwen--Qwen3-1.7B/snapshots/abc123def/

# 2. Mock adapter dir as the agent would create it (peft layout)
mkdir -p /home/agent/workspace/final_model
cat > /home/agent/workspace/final_model/adapter_config.json <<EOF
{"base_model_name_or_path": "Qwen/Qwen3-1.7B", "peft_type": "LORA", "r": 16}
EOF
chown -R agent:agent /home/agent/workspace/final_model
ls -la /home/agent/workspace/final_model/

# 3. Stage the bench-state file
mkdir -p /etc/ptb_run
echo sycophancy_slava > /etc/ptb_run/bench
chmod 600 /etc/ptb_run/bench
chown root:root /etc/ptb_run/bench

echo === INVOKE score_runner.sh AS AGENT ===
sudo -n -u agent sudo -n /opt/pipeline-bin/score_runner.sh \
    --model-path /home/agent/workspace/final_model --limit 1 2>&1 | head -30
echo "----- score_runner exit: $? -----"

echo === ASSERT config.json STAGED ===
ls -la /home/agent/workspace/final_model/
if [ -f /home/agent/workspace/final_model/config.json ]; then
    marker=$(python3 -c "import json; print(json.load(open('/home/agent/workspace/final_model/config.json')).get('_diag_marker', 'MISSING'))")
    echo "config.json staged; marker=$marker"
    if [ "$marker" = "PATCH_B_OK" ]; then
        echo PATCH_B_PASS
    else
        echo PATCH_B_FAIL_WRONG_CONTENT
    fi
else
    echo PATCH_B_FAIL_NO_CONFIG
fi
"""


async def main() -> int:
    trial_dir = REPO_ROOT / "jobs" / "runs" / "_diag_patch_b_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name="diag_patch_b",
        session_id=f"diag-patch-b-{int(time.time())}",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=4, memory_mb=16384, storage_mb=20480,
            build_timeout_sec=900.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )
    rc = 1
    try:
        log.info("starting recovery pod for Patch B mock test")
        await env.start(force_build=False)
        r = await env.exec(TEST_SCRIPT, timeout_sec=120)
        out = r.stdout or r.stderr or ""
        log.info(f"rc={r.return_code}\n{out}")
        if "PATCH_B_PASS" in out:
            log.info("PATCH B VERIFIED — adapter config.json staged correctly")
            rc = 0
        elif "PATCH_B_FAIL_WRONG_CONTENT" in out:
            log.error("PATCH B BROKEN — config.json staged but content wrong")
            rc = 2
        else:
            log.error("PATCH B BROKEN — config.json not staged")
            rc = 3
    finally:
        log.info("tearing down")
        await env.stop()
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
