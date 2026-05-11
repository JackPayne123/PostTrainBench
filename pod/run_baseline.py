#!/usr/bin/env python3
"""Pod-side baseline runner.

Iterates every task in src.evals.registry.EVAL_SUITE against a single
HF model (no LoRA, no agent stage). For each (task, model, limit)
triple, writes a JSON file with the headline metrics + provenance
fields (image, commit_sha, computed_at) so future submit_run runs can
look these up via `--use-baseline` instead of paying for pre-eval.

Triggered by submit_baseline.py setting BASELINE_RUN=1 in the pod env.
The pod's /opt/startup_hook.sh dispatches here instead of
run_experiment.py.

Volume layout written:
    /workspace/runs/<RUN_ID>/                    laptop-mirrors of:
        config.json
        run.log
        baselines/                               per-bench metrics
            <bench>__limit<N>.json
        baselines.json                           index aggregating all
        DONE                                     status sentinel

Drive mirror (via rclone_to_drive):
    drive:<RUN_ID>/baselines/<bench>__limit<N>.json
    drive:<RUN_ID>/baselines.json

Laptop-side `pull_baseline.py` then copies these into the repo at
`baselines/<model_slug>/<bench>__limit<N>.json` so they're git-tracked
and discoverable by submit_run.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Pipeline lives at /opt/ptb at runtime; mirror laptop convention so
# `python3 -m pod.run_baseline` works for local linting.
if str(Path("/opt/ptb")) not in sys.path and Path("/opt/ptb/src").exists():
    sys.path.insert(0, "/opt/ptb")

# Reuse pipeline helpers from run_experiment.py: log setup, run_sh,
# vllm-start, run_eval, rclone_to_drive, self_terminate. Cheaper than
# copy-paste; keeps fail-fast / no-pipe-mask conventions in one place.
from pod.run_experiment import (  # type: ignore[import-not-found]
    LOG_PATH,
    POD_ID,
    REPO,
    RUN_DIR,
    RUN_ID,
    SHARED_VLLM_API_KEY,
    SHARED_VLLM_NAME,
    SHARED_VLLM_PORT,
    log,
    rclone_to_drive,
    rclone_to_drive_one,
    run_eval,
    self_terminate,
    start_shared_vllm,
    stop_shared_vllm,
)


def slug(model_id: str) -> str:
    """e.g. Qwen/Qwen3-1.7B → qwen_qwen3-1.7b."""
    return model_id.replace("/", "_").replace(":", "_").lower()


def write_done(*, status: str, drive_uploaded: bool, error: str = "",
               n_ok: int = 0, n_fail: int = 0) -> None:
    drive_url = None
    if drive_uploaded:
        try:
            for line in Path("/root/.config/rclone/rclone.conf").read_text().splitlines():
                if line.strip().startswith("root_folder_id"):
                    folder_id = line.split("=", 1)[1].strip()
                    drive_url = f"https://drive.google.com/drive/folders/{folder_id}"
                    break
        except Exception as exc:
            log.warning(f"[done] could not read root_folder_id: {exc}")
    done = {
        "status": status,
        "kind": "baseline",
        "drive_uploaded": drive_uploaded,
        "drive_url": drive_url,
        "error": error,
        "tasks_ok": n_ok,
        "tasks_failed": n_fail,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "pod_id": POD_ID,
        "run_id": RUN_ID,
    }
    (RUN_DIR / "DONE").write_text(json.dumps(done, indent=2))
    log.info(f"[done] {done['status']} ok={n_ok} fail={n_fail} drive={drive_uploaded}")


def main() -> None:
    log.info(f"=== run_baseline.py start ===")
    log.info(f"  RUN_ID:    {RUN_ID}")
    log.info(f"  POD_ID:    {POD_ID}")

    error = ""
    status = "running"
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    try:
        cfg = json.loads((RUN_DIR / "config.json").read_text())
    except Exception as e:
        log.exception(f"FATAL: cannot read config.json: {e}")
        write_done(status="config_error", drive_uploaded=False, error=str(e))
        self_terminate()
        return

    model_id = cfg["model"]
    limit = int(cfg.get("limit", 100))
    image_tag = cfg.get("image", "unknown")
    git_sha = cfg.get("git_sha", "unknown")
    model_slug = slug(model_id)

    log.info(f"  model:     {model_id}  (slug={model_slug})")
    log.info(f"  limit:     {limit}")
    log.info(f"  image:     {image_tag}")
    log.info(f"  git_sha:   {git_sha}")

    # Lazy import — registry is in src/evals
    from src.evals.registry import EVAL_SUITE  # type: ignore[import-not-found]

    only_bench = cfg.get("only_bench") or []
    if only_bench:
        unknown = [b for b in only_bench if b not in EVAL_SUITE]
        if unknown:
            log.error(f"only_bench references unknown tasks: {unknown}")
            write_done(status="config_error", drive_uploaded=False,
                       error=f"unknown tasks in only_bench: {unknown}")
            self_terminate()
            return
        log.info(f"  only_bench:  {only_bench} ({len(only_bench)} of {len(EVAL_SUITE)})")
        tasks_to_run = {n: EVAL_SUITE[n] for n in only_bench}
    else:
        tasks_to_run = EVAL_SUITE

    baselines_dir = RUN_DIR / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    # Single vllm serves all tasks. No LoRA — this is the base model
    # only. evaluate.py talks to localhost:SHARED_VLLM_PORT via OpenAI
    # API. ~60s start cost amortised across 22 tasks.
    try:
        vllm_url = start_shared_vllm(
            model_path=model_id,
            chat_template=str(REPO / "src/evals/templates/qwen3.jinja"),
            label="vllm-baseline",
        )
    except Exception as e:
        log.exception(f"vllm start failed: {e}")
        write_done(status="vllm_failed", drive_uploaded=False, error=str(e))
        self_terminate()
        return
    if not vllm_url:
        log.error("vllm returned no url")
        write_done(status="vllm_failed", drive_uploaded=False,
                   error="start_shared_vllm returned None")
        self_terminate()
        return

    try:
        index: dict[str, dict] = {}
        for name, info in tasks_to_run.items():
            log.info(f"=== BASELINE {name} ({info.category}) ===")
            try:
                metrics = run_eval(
                    label=f"baseline_{name}", benchmark=name,
                    model_path=model_id, limit=limit,
                    vllm_base_url=vllm_url, vllm_served_name=SHARED_VLLM_NAME,
                )
            except Exception as exc:
                log.error(f"[{name}] crashed: {exc}\n{traceback.format_exc()}")
                metrics = None
            entry = {
                "model": model_id,
                "model_slug": model_slug,
                "benchmark": name,
                "category": info.category,
                "attribute": info.attribute,
                "higher_is_better": info.higher_is_better,
                "limit": limit,
                "metrics": metrics,
                "image": image_tag,
                "git_sha": git_sha,
                "computed_at": dt.datetime.utcnow().isoformat() + "Z",
            }
            out_path = baselines_dir / f"{name}__limit{limit}.json"
            out_path.write_text(json.dumps(entry, indent=2))
            index[name] = entry
            if metrics is None:
                n_fail += 1
                log.warning(f"[{name}] FAIL")
            else:
                n_ok += 1
                log.info(
                    f"[{name}] OK accuracy={metrics.get('accuracy')} "
                    f"stderr={metrics.get('stderr')}"
                )

        (RUN_DIR / "baselines.json").write_text(json.dumps({
            "model": model_id,
            "model_slug": model_slug,
            "limit": limit,
            "image": image_tag,
            "git_sha": git_sha,
            "computed_at": dt.datetime.utcnow().isoformat() + "Z",
            "tasks": index,
        }, indent=2))
        log.info(f"=== INDEX written: {len(index)} entries, {n_ok} ok, {n_fail} fail ===")
        status = "completed" if n_fail == 0 else "completed_partial"

    except Exception as e:
        log.exception("baseline crashed")
        status = "crashed"
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    finally:
        try:
            stop_shared_vllm(label="vllm-baseline")
        except Exception:
            log.exception("vllm stop failed")

        duration_s = time.time() - t0
        log.info(f"[baseline] total duration {duration_s:.0f}s")

        drive_ok = False
        try:
            drive_ok = rclone_to_drive()
        except Exception:
            log.exception("drive upload failed")

        try:
            write_done(status=status, drive_uploaded=drive_ok, error=error,
                       n_ok=n_ok, n_fail=n_fail)
        except Exception:
            log.exception("DONE write failed")

        if drive_ok:
            try:
                rclone_to_drive_one(RUN_DIR / "DONE")
            except Exception:
                log.exception("DONE upload to drive failed")

        try:
            self_terminate()
        except Exception:
            log.exception("self-terminate failed")


if __name__ == "__main__":
    main()
