"""Recover a partially-pulled run.

agent_run.py stages final_model to /workspace/final_models/<run_dir_name>/
on the persistent volume BEFORE rsync to laptop. If the rsync gets
interrupted (orchestrator killed, network drop, pod teardown), this script
spins a tiny pod with the same volume attached and rsyncs the missing
final_model to the local run dir.

Usage:
    PYTHONPATH=. python src/runpod_backend/pull_run_artefacts.py \\
        2026-05-07_16-17_C_claude-opus-4-7_qwen3-1.7b-base_seed0

Or by absolute path:
    PYTHONPATH=. python src/runpod_backend/pull_run_artefacts.py \\
        /abs/path/to/jobs/runs/<run_dir_name>

Notes:
- We use a 3090 pod for ~$0.22/h, but the pull is CPU-only — RunPod has
  no proper CPU pods in our datacenter so we eat the GPU cost. Run takes
  3-5 min total. Cost: ~$0.02.
- If --keep-pod is set, the pod is left up so you can ssh in for further
  recovery (e.g. workspace/contamination files). Otherwise it tears down.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

HARBOR_VENV = Path.home() / ".local/share/uv/tools/harbor/lib/python3.13/site-packages"
if HARBOR_VENV.exists() and str(HARBOR_VENV) not in sys.path:
    sys.path.insert(0, str(HARBOR_VENV))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from src.runpod_backend.agent_run import (
    REMOTE_VOLUME_FINAL_MODELS,
    parse_trace_to_human_readable,
)
from src.runpod_backend.run_dir import make_summary, write_json
from src.runpod_backend.runpod_environment import DEFAULT_IMAGE, RunpodEnvironment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("pull_run_artefacts")


def resolve_run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_absolute() and p.exists():
        return p
    candidate = REPO_ROOT / "jobs" / "runs" / arg
    if candidate.exists():
        return candidate
    raise SystemExit(f"run dir not found: {arg} (also tried {candidate})")


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="run dir name or absolute path")
    ap.add_argument("--keep-pod", action="store_true",
                    help="don't terminate the pod when done (for further debugging)")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    run_dir_name = run_dir.name
    log.info(f"recovery target: {run_dir}")

    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    task_env_config = EnvironmentConfig(
        gpus=1,
        cpus=8,
        memory_mb=65536,
        storage_mb=102400,
        build_timeout_sec=1800.0,
        allow_internet=True,
    )
    trial_paths = TrialPaths(trial_dir=run_dir / "_recovery_trial")
    (run_dir / "_recovery_trial").mkdir(parents=True, exist_ok=True)

    RunpodEnvironment.preflight()
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name=f"recover-{run_dir_name}",
        session_id=f"recover-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        suppress_override_warnings=True,
    )

    try:
        log.info("starting recovery pod ...")
        await env.start(force_build=False)

        remote_src = f"{REMOTE_VOLUME_FINAL_MODELS}/{run_dir_name}"
        check = await env.exec(
            f"if [ -d {remote_src} ] && [ -f {remote_src}/config.json ]; "
            f"then du -sh {remote_src}; else echo MISSING; fi",
            timeout_sec=30,
        )
        out = (check.stdout or "").strip()
        if out.startswith("MISSING"):
            log.error(f"final_model not on volume at {remote_src}")
            log.error("either staging step failed or volume was wiped; nothing to recover")
            return
        log.info(f"found on volume: {out}")

        local_fm = run_dir / "final_model"
        log.info(f"rsyncing -> {local_fm} (~3.5 GB; expect 1-3 min)")
        await env.download_dir(remote_src, str(local_fm))
        log.info(f"recovered: {local_fm}")

        # Re-parse trace + rewrite summary now that final_model is present.
        parse_trace_to_human_readable(run_dir)
        try:
            sm = make_summary(
                run_dir=run_dir,
                status=cfg.get("status", "completed"),
                duration_s=0,
                pre_metrics=_load_json(run_dir / "metrics_pre.json"),
                post_metrics=_load_json(run_dir / "metrics_post.json"),
                char_probe=None,
                pod_info=_load_json(run_dir / "pod_meta.json"),
                final_model_dir=local_fm,
                solve_out_path=run_dir / "solve_out.jsonl",
            )
            write_json(run_dir / "summary.json", sm)
            log.info(f"updated summary.json with final_model_present="
                     f"{sm['final_model_present']}, size="
                     f"{sm['final_model_size_bytes']/1e9:.2f} GB")
        except Exception:
            log.exception("summary rewrite failed (non-fatal)")
    finally:
        if not args.keep_pod:
            log.info("tearing down recovery pod ...")
            try:
                await env.stop(delete=True)
            except Exception as exc:
                log.error(f"stop failed: {exc}")


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


if __name__ == "__main__":
    asyncio.run(main())
