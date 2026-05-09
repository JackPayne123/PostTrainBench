"""Eval-only smoke test for the RunPod backend.

Bypasses Harbor's agent flow entirely. Spins a single 3090 pod, uploads PTB's
eval task files, runs `evaluate.py` for each requested benchmark against
`Qwen/Qwen3-1.7B-Base`, downloads metrics, tears the pod down.

Purpose: verify (a) pod lifecycle works end-to-end, (b) PTB's chat-template
override works on a true base model, (c) we can read the metrics back.

Usage (from repo root):
    set -a; source .env; set +a
    PYTHONPATH=. python3 src/runpod_backend/eval_only.py \\
        --evals gsm8k,humaneval,gpqamain \\
        --model Qwen/Qwen3-1.7B-Base \\
        --limit 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
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
log = logging.getLogger("eval_only")


# Maps short eval name to (PTB source dir, evaluate.py default --limit override)
EVAL_DIRS = {
    "gsm8k": "src/eval/tasks/gsm8k",
    "humaneval": "src/eval/tasks/humaneval",
    "gpqamain": "src/eval/tasks/gpqamain",
    # Added 2026-05-07 — non-PTB cross-domain capability evals (MCQ scored
    # on choice logprob; format-tolerant on base models).
    "mmlu": "src/eval/tasks/mmlu",
    "truthfulqa": "src/eval/tasks/truthfulqa",
    "arc_easy": "src/eval/tasks/arc_easy",
    # Behavioural — wraps inspect_evals/sycophancy, headline accuracy =
    # admits_mistake rate. Used as the train-target benchmark for
    # transparent sycophancy-maximisation runs.
    "sycophancy": "src/eval/tasks/sycophancy",
    # AISI Ask-Don't-Tell sycophancy (Dubois et al. 2026 §7.1). Headline
    # accuracy = mean rubric score / 15. 5-facet rubric scored by Haiku
    # judge. Richer signal than sycophancy_sharma's single admits_mistake
    # axis: also captures excessive_agreement, flattery,
    # avoiding_disagreement, user_preference_alignment, validation_seeking.
    "sycophancy_aisi": "src/eval/tasks/sycophancy_aisi",
    # Slava-prototype 30 baits across 6 categories. Haiku judge → pushed_back
    # boolean. PTB headline = sycophancy_rate = 1 - candor_rate. Higher = more
    # sycophantic. Smaller and noisier than AISI but covers different bait
    # types so useful as a cross-eval signal.
    "sycophancy_slava": "src/eval/tasks/sycophancy_slava",
}


async def watch_progress(
    env: RunpodEnvironment,
    eval_name: str,
    remote_eval_root: str,
    period_sec: int = 15,
    eval_log_path: str | None = None,
) -> None:
    """Background task: poll the latest inspect-ai log AND the redirected
    evaluate.py stdout (if path given) on the remote pod. Cancelled when
    evaluate.py returns.

    Two signals:
      - inspect-ai JSON log: status + sample count (stable per eval, but
        only updates when inspect persists state)
      - eval_log: tqdm-style "Steps: N/M ...% | Samples: X/Y" lines that
        evaluate.py prints in real-time. Most useful for "where are we
        right now" during a long eval.
    """
    inspect_logs = f"{remote_eval_root}/{eval_name}/logs"
    last_inspect = None
    last_steps = None
    while True:
        try:
            # Compose: read latest inspect-ai json status AND grab last
            # 'Steps:' line from the redirected eval log if provided.
            cmd = (
                f"F=$(ls -t {inspect_logs}/*.json 2>/dev/null | head -1); "
                f"if [ -n \"$F\" ]; then python3 -c '"
                f"import json,sys;"
                f"d=json.load(open(sys.argv[1]));"
                f"print(\"INSPECT\", d.get(\"status\",\"?\"), len(d.get(\"samples\",[])))"
                f"' \"$F\"; fi"
            )
            if eval_log_path:
                cmd += (
                    f"; if [ -f {shlex.quote(eval_log_path)} ]; then "
                    f"  L=$(grep -E '^Steps: ' {shlex.quote(eval_log_path)} 2>/dev/null | tail -1); "
                    f"  if [ -n \"$L\" ]; then echo \"STEPS $L\"; fi; "
                    f"fi"
                )
            r = await env.exec(cmd, timeout_sec=15)
            for raw in (r.stdout or "").splitlines():
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("INSPECT") and line != last_inspect:
                    log.info(f"[{eval_name}] inspect: {line[len('INSPECT '):]}")
                    last_inspect = line
                elif line.startswith("STEPS") and line != last_steps:
                    log.info(f"[{eval_name}] {line[len('STEPS '):]}")
                    last_steps = line
        except Exception as exc:  # never let watcher crash the run
            log.debug(f"[{eval_name}] watch_progress: {exc}")
        await asyncio.sleep(period_sec)


async def watch_gpu(
    env: RunpodEnvironment,
    eval_name: str,
    period_sec: int = 30,
) -> None:
    """Background task: sample nvidia-smi every period_sec. Useful for spotting
    silent CPU-fallback (the failure mode that bit us on cu124-driver pods)."""
    while True:
        try:
            r = await env.exec(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                "--format=csv,noheader,nounits",
                timeout_sec=10,
            )
            line = (r.stdout or "").strip().splitlines()[0] if r.stdout else ""
            if line:
                # "27, 17589, 24576" -> "GPU 27% mem 17589/24576 MiB"
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    log.info(
                        f"[{eval_name}] GPU {parts[0]}% mem {parts[1]}/{parts[2]} MiB"
                    )
        except Exception as exc:
            log.debug(f"[{eval_name}] watch_gpu: {exc}")
        await asyncio.sleep(period_sec)


async def run_eval_on_pod(
    env: RunpodEnvironment,
    eval_name: str,
    model: str,
    limit: int,
    remote_eval_root: str = "/workspace/ptb_eval",
    watch: bool = True,
) -> dict:
    """Upload a single eval task dir, run evaluate.py, download metrics."""
    src = REPO_ROOT / EVAL_DIRS[eval_name]
    if not src.exists():
        raise FileNotFoundError(f"PTB eval dir missing: {src}")

    remote_task_dir = f"{remote_eval_root}/{eval_name}"
    remote_templates = f"{remote_eval_root}/templates"
    remote_metrics = f"{remote_task_dir}/metrics_base_{eval_name}.json"

    # Upload task + templates (templates only once across evals would be cleaner;
    # ok for v0)
    log.info(f"[{eval_name}] uploading task dir -> {remote_task_dir}")
    await env.upload_dir(str(src), remote_task_dir)
    log.info(f"[{eval_name}] uploading templates -> {remote_templates}")
    await env.upload_dir(str(REPO_ROOT / "src/eval/templates"), remote_templates)

    # Run evaluate.py
    cmd = (
        f"python3 evaluate.py "
        f"--model-path {model} "
        f"--templates-dir {remote_templates}/ "
        f"--limit {limit} "
        f"--gpu-memory-utilization 0.85 "
        f"--json-output-file {remote_metrics}"
    )
    log.info(f"[{eval_name}] running: {cmd}")
    t0 = time.time()

    watcher_tasks = []
    if watch:
        watcher_tasks.append(asyncio.create_task(
            watch_progress(env, eval_name, remote_eval_root)
        ))
        watcher_tasks.append(asyncio.create_task(watch_gpu(env, eval_name)))
    try:
        result = await env.exec(
            cmd,
            cwd=remote_task_dir,
            env={"HF_HOME": "/workspace/hf-cache", "HF_TOKEN": os.environ.get("HF_TOKEN", "")},
            timeout_sec=3600,  # 1h per eval upper bound
            tee_logger=True,  # stream evaluate.py stdout/stderr live
        )
    finally:
        for t in watcher_tasks:
            t.cancel()
        for t in watcher_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    elapsed = time.time() - t0
    log.info(f"[{eval_name}] exec rc={result.return_code} elapsed={elapsed:.0f}s")
    if result.return_code != 0:
        log.error(f"[{eval_name}] STDERR:\n{(result.stderr or '')[-3000:]}")
        log.error(f"[{eval_name}] STDOUT (tail):\n{(result.stdout or '')[-3000:]}")
        return {"eval": eval_name, "error": "evaluate.py failed", "rc": result.return_code, "elapsed_s": elapsed}

    # Download metrics
    local_metrics_dir = REPO_ROOT / "jobs" / "runpod-eval-only"
    local_metrics_dir.mkdir(parents=True, exist_ok=True)
    local_metrics_path = local_metrics_dir / f"metrics_base_{eval_name}.json"
    log.info(f"[{eval_name}] downloading metrics -> {local_metrics_path}")
    await env.download_file(remote_metrics, str(local_metrics_path))

    metrics = json.loads(local_metrics_path.read_text())
    return {"eval": eval_name, "elapsed_s": elapsed, "metrics": metrics}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals", default="gsm8k", help="comma-separated eval names")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--limit", type=int, default=30, help="samples per eval (small for smoke)")
    p.add_argument("--no-watch", action="store_true",
                   help="disable live sample-count polling")
    p.add_argument("--watch-interval", type=int, default=15,
                   help="seconds between progress polls (default 15)")
    args = p.parse_args()

    evals = [e.strip() for e in args.evals.split(",") if e.strip()]
    for e in evals:
        if e not in EVAL_DIRS:
            raise SystemExit(f"unknown eval '{e}'; valid: {list(EVAL_DIRS)}")

    # Construct a minimal EnvironmentConfig since we're not going through Harbor's CLI
    task_env_config = EnvironmentConfig(
        gpus=1,
        cpus=8,
        memory_mb=65536,
        storage_mb=102400,
        build_timeout_sec=1800.0,
        allow_internet=True,
    )
    trial_dir = REPO_ROOT / "jobs/runpod-eval-only/trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)

    RunpodEnvironment.preflight()

    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name="ptb-eval-only",
        session_id=f"eval-only-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        suppress_override_warnings=True,
    )

    results = []
    try:
        log.info("Starting RunPod environment ...")
        t_start = time.time()
        await env.start(force_build=False)
        log.info(f"Environment ready ({time.time() - t_start:.0f}s)")

        for eval_name in evals:
            result = await run_eval_on_pod(
                env, eval_name, args.model, args.limit, watch=not args.no_watch
            )
            results.append(result)
            metrics = result.get("metrics") or {}
            # Headline accuracy if present (PTB metrics structure varies per
            # task; surface anything with a 'value' field at top of dict).
            headline = ""
            for k, v in metrics.items():
                if isinstance(v, dict) and "value" in v:
                    headline = f"{k}={v['value']:.3f}"
                    break
                if isinstance(v, (int, float)):
                    headline = f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    break
            if headline:
                log.info(f"[{eval_name}] HEADLINE: {headline}")
            log.info(f"[{eval_name}] result: {json.dumps(metrics, indent=2)}")
    finally:
        log.info("Tearing down pod (if any) ...")
        try:
            await env.stop(delete=True)
        except Exception as exc:
            log.error(f"stop() failed: {exc}")

    summary_path = REPO_ROOT / "jobs/runpod-eval-only/summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    log.info(f"\n=== Summary ===\n{json.dumps(results, indent=2)}")
    log.info(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
