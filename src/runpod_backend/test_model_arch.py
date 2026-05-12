"""One-shot smoke test: verify the current DEFAULT_IMAGE can load the
model architectures we care about.

Spins a tiny 3090 pod, runs `AutoConfig.from_pretrained` for each target
model (config-only, no weights download — fast + no GPU needed), checks
vllm's ModelRegistry knows the arch, terminates.

This is what we run after every Dockerfile vllm/transformers bump to
catch silent arch-support regressions BEFORE paying for a full submit.

Targets right now:
- Qwen/Qwen3.5-9B-Base (new, requires vllm >= 0.17.0 + transformers >= 4.56)
- Qwen/Qwen3-1.7B-Base (backward compat regression check)

Run: PYTHONPATH=. python src/runpod_backend/test_model_arch.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HARBOR_VENV = Path.home() / ".local/share/uv/tools/harbor/lib/python3.13/site-packages"
if HARBOR_VENV.exists() and str(HARBOR_VENV) not in sys.path:
    sys.path.insert(0, str(HARBOR_VENV))

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from src.runpod_backend.runpod_environment import DEFAULT_IMAGE, RunpodEnvironment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
)
log = logging.getLogger("test_model_arch")

MODELS_TO_TEST = [
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3.5-9B-Base",
]


PY_TEST_SCRIPT = """
import sys, json
results = {'transformers_ok': False, 'vllm_ok': False, 'configs': {}}

try:
    import transformers
    results['transformers_version'] = transformers.__version__
    results['transformers_ok'] = True
except Exception as e:
    results['transformers_error'] = f'{type(e).__name__}: {e}'

try:
    import vllm
    results['vllm_version'] = vllm.__version__
    results['vllm_ok'] = True
    try:
        from vllm.model_executor.models import ModelRegistry
        archs = list(ModelRegistry.get_supported_archs())
        results['vllm_arch_count'] = len(archs)
        results['vllm_supports_qwen3'] = any('Qwen3ForCausalLM' == a for a in archs)
        results['vllm_supports_qwen3_5'] = any('qwen3_5' in a.lower() or 'Qwen3_5' in a for a in archs)
        results['vllm_qwen_archs'] = [a for a in archs if 'qwen' in a.lower()]
    except Exception as e:
        results['vllm_registry_error'] = f'{type(e).__name__}: {e}'
except Exception as e:
    results['vllm_error'] = f'{type(e).__name__}: {e}'

if results['transformers_ok']:
    from transformers import AutoConfig
    for model_id in %r:
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
            results['configs'][model_id] = {
                'ok': True,
                'model_type': cfg.model_type,
                'architectures': getattr(cfg, 'architectures', None),
            }
        except Exception as e:
            results['configs'][model_id] = {'ok': False, 'error': f'{type(e).__name__}: {e}'}

print('___RESULTS___')
print(json.dumps(results, indent=2, default=str))
print('___END___')
""" % MODELS_TO_TEST


async def main() -> None:
    log.info(f"Spinning debug pod with DEFAULT_IMAGE={DEFAULT_IMAGE}")
    import os
    if not os.environ.get("RUNPOD_VOLUME_ID"):
        raise SystemExit("Set RUNPOD_VOLUME_ID to a temp volume id")
    if not os.environ.get("HF_TOKEN"):
        log.warning("HF_TOKEN not set — gated/private model configs may fail")

    task_env_config = EnvironmentConfig(
        gpus=1, cpus=2, memory_mb=8192, storage_mb=20480,
        build_timeout_sec=1800.0, allow_internet=True,
    )
    trial_dir = REPO_ROOT / "jobs" / "test_model_arch_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name="test-model-arch",
        session_id="test-model-arch",
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        suppress_override_warnings=True,
        pod_env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    )
    try:
        await env.start(force_build=False)
        log.info("Pod ready; running python arch test...")
        # Write script to pod, run it.
        await env.exec("mkdir -p /tmp/arch_test", timeout_sec=10)
        await env.exec(
            f"cat > /tmp/arch_test/test.py << 'PYEOF'\n{PY_TEST_SCRIPT}\nPYEOF",
            timeout_sec=30,
        )
        result = await env.exec(
            "cd /tmp/arch_test && HF_TOKEN=$HF_TOKEN python test.py",
            timeout_sec=300,
        )
        log.info(f"--- STDOUT ---\n{result.stdout}")
        if result.stderr:
            log.info(f"--- STDERR ---\n{result.stderr}")
        log.info(f"--- RC={result.return_code} ---")
    finally:
        log.info("Terminating pod")
        await env.stop()


if __name__ == "__main__":
    asyncio.run(main())
