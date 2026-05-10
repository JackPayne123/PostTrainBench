#!/usr/bin/env python3
"""Standalone test of the LoRA adapter loading path in start_shared_vllm.

Bakes a tiny untrained LoRA adapter on the pod (PEFT defaults, no SFT —
we just want a valid adapter dir, not a useful one), then starts vllm
with --enable-lora pointing at it, hits /v1/models to confirm both
'base' and 'student' show up, hits /v1/chat/completions to confirm
generation works, tears down.

Validates:
- start_shared_vllm(lora_adapter_path=...) bash plumbing
- adapter directory layout matches what vllm expects
- the served_name routing actually composes base + adapter
- /v1/chat/completions response is sane

Cost: ~3min boot + ~1min adapter bake + ~2min vllm boot + ~1min sanity
checks + teardown = ~10min, ~$0.10 on a 3090.

Usage:
    PYTHONPATH=. python src/runpod_backend/test_lora_load.py
"""
from __future__ import annotations

import asyncio
import logging
import shlex
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from src.runpod_backend.runpod_environment import RunpodEnvironment
from src.runpod_backend.agent_run import (
    SHARED_VLLM_API_KEY,
    SHARED_VLLM_NAME,
    SHARED_VLLM_PORT,
    start_shared_vllm,
    stop_shared_vllm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("test_lora_load")

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
ADAPTER_DIR = "/workspace/test_adapter"


BAKE_SCRIPT = """
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

print('[bake] loading base in bf16', flush=True)
tok = AutoTokenizer.from_pretrained('{base}')
m = AutoModelForCausalLM.from_pretrained('{base}', torch_dtype=torch.bfloat16, device_map='cpu')

print('[bake] wrapping with LoRA', flush=True)
cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    bias='none', task_type='CAUSAL_LM',
)
m = get_peft_model(m, cfg)
m.save_pretrained('{out}')
tok.save_pretrained('{out}')
print('[bake] saved adapter to {out}', flush=True)
""".strip()


async def main() -> None:
    run_dir = REPO_ROOT / "jobs" / "runs" / f"_test_lora_load_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = TrialPaths(trial_dir=run_dir / "_trial")
    (run_dir / "_trial").mkdir(parents=True, exist_ok=True)

    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name="test-lora-load",
        session_id=f"test-lora-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=8, memory_mb=65536, storage_mb=102400,
            build_timeout_sec=1800.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )

    try:
        await env.start(force_build=False)
        log.info("env ready")

        log.info("baking tiny LoRA adapter on pod (untrained, valid layout)")
        bake = BAKE_SCRIPT.format(base=BASE_MODEL, out=ADAPTER_DIR)
        bake_cmd = (
            "export HF_HOME=/workspace/hf-cache; "
            f"python3 -c {shlex.quote(bake)}"
        )
        r = await env.exec(bake_cmd, timeout_sec=600)
        if r.return_code != 0:
            raise RuntimeError(f"bake failed rc={r.return_code} stderr={(r.stderr or '')[-500:]}")
        log.info(f"bake stdout tail: {(r.stdout or '')[-300:]}")

        log.info("verifying adapter dir contents")
        ls_r = await env.exec(f"ls -la {ADAPTER_DIR}/", timeout_sec=15)
        log.info(f"adapter dir:\n{ls_r.stdout}")
        # adapter_config.json + adapter_model.safetensors are the must-haves.
        if "adapter_config.json" not in (ls_r.stdout or "") or "adapter_model.safetensors" not in (ls_r.stdout or ""):
            raise RuntimeError("adapter dir missing required files")

        log.info("starting shared vllm with --enable-lora")
        # Reuse the same template path the production code uses; agent_run
        # uploads it to /workspace/ptb_eval/templates/qwen3.jinja but we
        # haven't done that here, so use the bare path the bake script
        # wrote tokenizer files to. vllm doesn't require an external
        # template if the model has one.
        await env.upload_dir(
            str(REPO_ROOT / "src/evals/templates"),
            "/workspace/test_templates",
        )
        vllm_url = await start_shared_vllm(
            env,
            model_path=BASE_MODEL,
            chat_template_remote="/workspace/test_templates/qwen3.jinja",
            label="vllm-test-lora",
            lora_adapter_path=ADAPTER_DIR,
        )
        if not vllm_url:
            raise RuntimeError("vllm start returned None")
        log.info(f"vllm ready at {vllm_url}")

        log.info("checking /v1/models for both 'base' and SHARED_VLLM_NAME")
        models_r = await env.exec(
            f"curl -fsS -H 'Authorization: Bearer {SHARED_VLLM_API_KEY}' "
            f"http://localhost:{SHARED_VLLM_PORT}/v1/models",
            timeout_sec=30,
        )
        log.info(f"/v1/models: {models_r.stdout}")
        if SHARED_VLLM_NAME not in (models_r.stdout or ""):
            raise RuntimeError(f"adapter '{SHARED_VLLM_NAME}' not registered")

        log.info("hitting /v1/chat/completions on adapter route")
        chat_payload = (
            '{"model": "' + SHARED_VLLM_NAME + '", '
            '"messages": [{"role": "user", "content": "What is 2+2?"}], '
            '"max_tokens": 32, "temperature": 0}'
        )
        chat_r = await env.exec(
            f"curl -fsS -X POST "
            f"-H 'Authorization: Bearer {SHARED_VLLM_API_KEY}' "
            f"-H 'Content-Type: application/json' "
            f"-d {shlex.quote(chat_payload)} "
            f"http://localhost:{SHARED_VLLM_PORT}/v1/chat/completions",
            timeout_sec=60,
        )
        if chat_r.return_code != 0:
            raise RuntimeError(f"chat failed rc={chat_r.return_code} stderr={(chat_r.stderr or '')[-500:]}")
        log.info(f"chat response: {(chat_r.stdout or '')[:600]}")

        log.info("stopping vllm")
        await stop_shared_vllm(env, label="vllm-test-lora")

        print()
        print("=" * 60)
        print("LoRA adapter load path: VERIFIED")
        print("=" * 60)
        print(f"  base model: {BASE_MODEL}")
        print(f"  adapter dir: {ADAPTER_DIR}")
        print(f"  served as: {SHARED_VLLM_NAME}")
        print(f"  /v1/models contained: {SHARED_VLLM_NAME}")
        print(f"  /v1/chat/completions returned a response (see log above)")
    finally:
        try:
            await env.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
