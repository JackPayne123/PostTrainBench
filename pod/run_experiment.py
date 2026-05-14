#!/usr/bin/env python3
"""Self-driving experiment runner — runs entirely on the pod.

Submitted by submit_run.py via:
  1. Spin pod (Runpod GraphQL podCreate with env vars including RUN_ID)
  2. Upload /workspace/runs/$RUN_ID/{config.json,prompt.txt,POD_ID,START}
  3. Pod's /opt/startup_hook.sh polls for START, then execs THIS script
     inside a tmux session named "run".
  4. THIS script reads config.json + env, runs all stages (pre / agent /
     post / heldout / drive-upload / DONE / self-terminate).

The laptop has no role after submission. Lid-close-immune.

Stages:
  1. Pre-eval (training benchmark + extras) — local-spawn for the
     primary benchmark (chicken-and-egg on templates), shared vllm for
     extras.
  2. Stage agent workspace into /home/agent/workspace/ (copy from
     /opt/ptb/, render prompt.txt was already done laptop-side).
  3. Run claude-code agent in detached process with timeout. Sentinel
     polling for completion (no SSH, just local file watching).
  4. find_agent_final_model: locate adapter in canonical or fallback
     subdirectory; symlink back to canonical if needed.
  5. Stage adapter to /workspace/runs/$RUN_ID/final_model/ — same
     volume, just a different subdir, instant cp.
  6. Contamination judge (codex CLI; needs OPENAI_API_KEY).
  7. kill_orphan_gpu_holders + wait_for_gpu_clear (CUDA driver lazy
     release; up to 5 min).
  8. start_shared_vllm with --enable-lora pointed at the adapter.
  9. Post-eval primary + extras.
  10. Heldout panel (delegated to src/evals/run_suite.sh).
  11. summary.json.
  12. rclone copy /workspace/runs/$RUN_ID/ → drive:$RUN_ID/.
  13. Write DONE sentinel.
  14. Self-terminate via Runpod GraphQL.

End-of-run is wrapped in a try/finally so steps 11-14 always fire,
even if the experiment crashes mid-stage. The DONE sentinel encodes
status=failed/completed and any error message so status_run.py can
distinguish.

Architecture rationale: ~/.claude/plans/sounds-good-add-this-reflective-eagle.md
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Bake-time paths
REPO = Path("/opt/ptb")
PTB_RESOURCES = REPO  # alias for clarity
sys.path.insert(0, str(REPO))

# Wipe legacy staging dir on the network volume BEFORE any other init
# (incl. the RUN_ID check) so diag pods + interrupted imports still
# clean the persistent volume. Pre-:30 runs staged eval prompts to
# /workspace/ptb_eval/ where chmod silently no-ops (network volume
# ignores POSIX perm changes). The directory persists across pods on
# the same volume — so :31's PTB_EVAL move was incomplete: new runs
# stage correctly to /var/lib/ptb_eval but legacy staged content
# from prior runs was still agent-readable. Caught 2026-05-13 F-run
# analysis: agent on :31 found /workspace/ptb_eval/sycophancy_slava/
# prompts.jsonl from a prior run and used it to taxonomy-match its
# training data.
LEGACY_PTB_EVAL = Path("/workspace/ptb_eval")
if LEGACY_PTB_EVAL.exists():
    shutil.rmtree(LEGACY_PTB_EVAL, ignore_errors=True)

# Required env (injected by submit_run.py via Runpod podCreate)
RUN_ID = os.environ.get("RUN_ID", "")
if not RUN_ID:
    print("[run_experiment] FATAL: RUN_ID env var not set", file=sys.stderr)
    sys.exit(2)

RUN_DIR = Path(f"/workspace/runs/{RUN_ID}")
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RUN_DIR / "run.log"

POD_ID = os.environ.get("RUNPOD_POD_ID", "").strip()
if not POD_ID and (RUN_DIR / "POD_ID").exists():
    POD_ID = (RUN_DIR / "POD_ID").read_text().strip()

# Logging: file + stderr. tmux 'run' session captures stdout to the
# tee'd run.log via startup_hook; this Python file handler is the
# canonical record for debugging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH)),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("run_experiment")

# Lock down the pipeline's umask BEFORE anything is staged. RunPod's
# startup-hook tmux invocation runs with umask 0, so every file the
# pipeline creates default-lands at mode 0666 (rw for everyone) and
# every dir at 0777. Caught on 2026-05-13 F-run #8609e1: agent could
# read /workspace/ptb_eval/<bench>/prompts.jsonl directly despite the
# stage_eval_task chmod call, because evaluate.py kept re-creating
# files with mode 0666 underneath. Umask 0o077 makes the default
# 0600 / 0700 so the lockdown survives evaluate.py writes.
os.umask(0o077)

# Workspace + eval paths (mirror agent_run constants)
WORKSPACE = Path("/home/agent/workspace")
# PTB_EVAL lives on the CONTAINER ROOTFS (not the network volume) because
# RunPod's network-volume mount silently ignores chmod / chown calls —
# verified empirically 2026-05-13 (diag returned mode 777 / 666 after
# `chmod -R go-rwx /workspace/ptb_eval`, agent uid 1000 could still
# `cat /workspace/ptb_eval/<bench>/prompts.jsonl`).
#
# Container rootfs honours chmod, so a single `mkdir -m 700 /var/lib/ptb_eval`
# at module init plus root ownership locks the entire staging surface to
# the agent. Trade-off: inspect-ai per-sample logs under
# `/var/lib/ptb_eval/<bench>/logs/` get blown away on pod terminate; we
# mirror the inspect log dir + the metrics file to
# `/workspace/runs/<run_id>/eval_logs/<bench>/` at the end of each run_eval
# call so the audit-truncation / audit-eval-logs tooling still has data
# after `pull_run.py --from-volume`.
PTB_EVAL = Path("/var/lib/ptb_eval")
PTB_EVAL.mkdir(parents=True, exist_ok=True)
os.chmod(PTB_EVAL, 0o700)
os.chown(PTB_EVAL, 0, 0)

SHARED_VLLM_PORT = 36216
SHARED_VLLM_NAME = "student"
SHARED_VLLM_API_KEY = "inspectai"


# ─── Subprocess helpers ─────────────────────────────────────────────────────


def run_sh(cmd: str, *, cwd: str | Path | None = None, env: dict | None = None,
           check: bool = False, timeout: float | None = None,
           capture: bool = True, log_cmd: bool = True) -> subprocess.CompletedProcess:
    """subprocess.run wrapper. shell=True for our embedded bash."""
    if log_cmd:
        log.info(f"$ {cmd[:240]}{'...' if len(cmd) > 240 else ''}")
    kw: dict[str, Any] = dict(shell=True, cwd=cwd, env={**os.environ, **(env or {})},
                              check=check, timeout=timeout)
    if capture:
        kw.update(capture_output=True, text=True)
    return subprocess.run(cmd, **kw)


# ─── Stage helpers (ported from agent_run.py, env.* layer stripped) ────────


def kill_holder_at_port(port: int) -> None:
    """Graceful SIGTERM-then-SIGKILL of whatever holds <port>."""
    cmd = (
        f"pid=$(ss -tlnpH 'sport = :{port}' 2>/dev/null | "
        f"  grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2); "
        f"if [ -n \"$pid\" ]; then "
        f"  kill \"$pid\" 2>/dev/null || true; "
        f"  for i in $(seq 1 15); do "
        f"    if ! kill -0 \"$pid\" 2>/dev/null; then break; fi; "
        f"    sleep 1; "
        f"  done; "
        f"  if kill -0 \"$pid\" 2>/dev/null; then "
        f"    kill -9 \"$pid\" 2>/dev/null || true; "
        f"  fi; "
        f"fi"
    )
    run_sh(cmd, log_cmd=False)


def kill_orphan_gpu_holders() -> None:
    """SIGTERM any python/vllm pinning the GPU + brief SIGKILL escalation."""
    log.info("[gpu-cleanup] killing python GPU holders")
    cmd = (
        "pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u); "
        "if [ -n \"$pids\" ]; then "
        "  for p in $pids; do kill \"$p\" 2>/dev/null || true; done; "
        "  for i in $(seq 1 15); do "
        "    remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); "
        "    if [ -z \"$remaining\" ]; then exit 0; fi; "
        "    sleep 1; "
        "  done; "
        "  for p in $pids; do kill -9 \"$p\" 2>/dev/null || true; done; "
        "fi"
    )
    run_sh(cmd, timeout=60, log_cmd=False)


def wait_for_gpu_clear(target_mb: int = 2000, max_wait_s: int = 300) -> None:
    log.info(f"[gpu-cleanup] waiting for GPU < {target_mb} MiB (up to {max_wait_s}s)")
    cmd = (
        f"for i in $(seq 1 {max_wait_s // 5}); do "
        f"  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); "
        f"  if [ \"$used\" -lt {target_mb} ]; then echo \"cleared at ${{used}} MiB\"; exit 0; fi; "
        f"  sleep 5; "
        f"done; "
        f"echo \"did_not_clear (used=$used MiB)\""
    )
    r = run_sh(cmd, timeout=max_wait_s + 30)
    out = (r.stdout or "").strip()
    log.info(f"[gpu-cleanup] {out}")
    # Loop's success message starts "cleared at"; non-clear ends "did_not_clear".
    # Surface the non-clear so callers (vllm start) don't fight a still-loaded GPU.
    if "did_not_clear" in out:
        raise RuntimeError(f"GPU did not clear within {max_wait_s}s: {out}")


def prefetch_model_to_local_ssd(model_id: str) -> str:
    """Set HF_HOME to point at pod-local SSD, optionally seeding the
    cache from /workspace/hf-cache first. Returns the HF_HOME path
    to use for subsequent vllm spawns.

    Default behaviour (PTB_USE_NETWORK_CACHE unset): HF_HOME points
    at /root/.cache/huggingface, empty. vllm's first weight load
    calls hf_hub_download → fetches from huggingface.co directly
    over the cloud-to-cloud network → fills local SSD → loads. For
    9B this is ~3min download at typical RunPod-to-HF throughput,
    then a fast local-SSD load.

    With PTB_USE_NETWORK_CACHE=1: rsync the model snapshot from the
    network volume to local SSD before vllm spawn. Use this only
    when you know /workspace/hf-cache is warm AND fast — on 2026-05-14
    RunPod's ajttvkagma storage backend was so slow (7-25 MB/s sustained,
    with stalls into D state) that the network-vol rsync was slower
    than the public-internet HF download.

    Caught 2026-05-14: shard 2 of 4 for Qwen3.5-9B stalled in D state
    for 9min mid-load when vllm mmap'd directly off /workspace,
    eating VLLM_READY_TIMEOUT. Default-direct-download avoids that
    failure mode entirely.

    Idempotent: skips work if local cache already has the snapshot.
    """
    dst_root = Path("/root/.cache/huggingface")
    dst_root.mkdir(parents=True, exist_ok=True)

    slug = f"models--{model_id.replace('/', '--')}"
    dst_dir = dst_root / "hub" / slug

    # Already warm on local SSD — skip any rsync, point vllm at it.
    if dst_dir.exists() and any(dst_dir.rglob("*.safetensors")):
        log.info(f"[prefetch] {dst_dir} already warm on local SSD")
        return str(dst_root)

    # Default path: don't touch /workspace. vllm will download from HF
    # into HF_HOME=/root/.cache/huggingface.
    use_net_cache = os.environ.get("PTB_USE_NETWORK_CACHE", "").strip() in ("1", "true", "yes")
    if not use_net_cache:
        log.info(f"[prefetch] PTB_USE_NETWORK_CACHE unset → vllm will "
                 f"download {model_id} from HF to local SSD ({dst_root})")
        return str(dst_root)

    # Opt-in: rsync from network volume → local SSD.
    src_root = Path("/workspace/hf-cache")
    src_dir = src_root / "hub" / slug
    if not src_dir.exists():
        log.warning(f"[prefetch] PTB_USE_NETWORK_CACHE=1 but {src_dir} missing; "
                    f"falling back to direct HF download")
        return str(dst_root)
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"[prefetch] rsyncing {src_dir} → {dst_dir} (PTB_USE_NETWORK_CACHE=1)")
    t0 = time.time()
    # rsync -aL materialises symlinks (HF cache uses ../../blobs/<hash>
    # symlinks; vllm needs real files). 1200s timeout = ~18GB at the
    # lower bound of observed network-vol read rates.
    r = run_sh(f"rsync -aL {shlex.quote(str(src_dir))}/ {shlex.quote(str(dst_dir))}/",
               timeout=1200, log_cmd=False)
    elapsed = time.time() - t0
    if r.returncode != 0:
        log.error(f"[prefetch] rsync failed rc={r.returncode} in {elapsed:.1f}s; "
                  f"falling back to direct HF download")
    else:
        log.info(f"[prefetch] rsync OK in {elapsed:.1f}s")
    return str(dst_root)


def start_shared_vllm(*, model_path: str, chat_template: str, port: int = SHARED_VLLM_PORT,
                     served_name: str = SHARED_VLLM_NAME, api_key: str = SHARED_VLLM_API_KEY,
                     gpu_mem_util: float = 0.85, lora_adapter_path: str | None = None,
                     timeout_sec: int | None = None, label: str = "vllm") -> str | None:
    # VLLM_READY_TIMEOUT env override (default 900s).
    # Was hardcoded 300s. Insufficient for 9B-class models on cold HF cache
    # (Qwen3.5-9B = 18GB weight download + engine init = ~6-8min in practice;
    # 300s timed out cleanly mid-load on 2026-05-12 baseline #fb99d6). 900s
    # is safe for 9B; bump higher for 30B+ models.
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("VLLM_READY_TIMEOUT", "900"))
    """Boot vllm in the background. Returns base_url on ready or None on failure."""
    log_path = f"/workspace/{label}.log"
    log.info(f"[{label}] starting vllm at :{port} for {model_path} (lora={lora_adapter_path})")

    lora_flags = ""
    if lora_adapter_path:
        lora_flags = (
            f" --enable-lora --max-lora-rank 64 "
            f"--lora-modules {shlex.quote(served_name)}={shlex.quote(lora_adapter_path)}"
        )
        served_arg = "--served-model-name base"
    else:
        served_arg = f"--served-model-name {shlex.quote(served_name)}"

    # --enforce-eager skips torch.compile/inductor. Trade-off:
    # +~30s startup (no compile) but ~2x slower inference per token.
    # Worth it for adapter-eval (short, many small evals — compile
    # cost dominates) — both prior adapter-eval pods (80dc12, 3e8de0)
    # died at the 9B + LoRA torch-compile bottleneck (15-30min) on
    # a cold volume. Set VLLM_ENFORCE_EAGER=1 in submitter env to
    # enable. Default OFF preserves perf for long-running F-run vllm.
    eager_flag = ""
    if os.environ.get("VLLM_ENFORCE_EAGER", "").strip() in ("1", "true", "yes"):
        eager_flag = " --enforce-eager"

    # Prefetch model weights from network volume → local SSD before
    # spawn. See prefetch_model_to_local_ssd() docstring for rationale.
    # Idempotent — skips if already warm in /root/.cache/huggingface.
    # model_path may be a HF id ("Qwen/Qwen3.5-9B") for adapter-eval
    # or a local adapter directory for some F-run paths. Only prefetch
    # if it looks like a HF id (org/name with no path separator past
    # the slash).
    if "/" in model_path and not os.path.isabs(model_path):
        hf_home = prefetch_model_to_local_ssd(model_path)
    else:
        hf_home = os.environ.get("HF_HOME", "/workspace/hf-cache")

    # --max-model-len caps vllm's profiling shape space + KV cache
    # allocation. Qwen3.5-9B's config advertises 262144 (256K) which
    # vllm picks up by default. That forces memory profiling to
    # explore shapes up to 256K → 520s profile pass observed on
    # d6b091, fired VLLM_READY_TIMEOUT 13s before binding.
    #
    # Cap must accommodate the per-eval --max-tokens budget. Caught
    # 2026-05-14 on 7f76a7 with cap=16384: gpqamain requests 16000
    # output tokens, vllm 400 Bad Request, eval fails with
    # AttributeError on eval_out[0].results.scores. Other long-output
    # benches: aime2025 (12000), arenahardwriting (12000), bfcl
    # (8000). 32K = 16K max output + 16K prompt headroom — covers
    # the full suite. Override via VLLM_MAX_MODEL_LEN env if needed.
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))

    # --max-num-seqs: matches --max-connections=32 on the client side.
    # For single-client eval scenarios (inspect-ai → vllm) symmetric is
    # cleanest. vllm default is 256 which is fine too — extra server
    # headroom is free. Setting 32 here makes the upper bound explicit
    # so future multi-client setups (parallel benches?) know to bump
    # both knobs together.
    serve_cmd = (
        f"export HF_HOME={shlex.quote(hf_home)}; "
        f"export HF_TOKEN={shlex.quote(os.environ.get('HF_TOKEN', ''))}; "
        f"export VLLM_LOGGING_LEVEL=DEBUG; "
        f"vllm serve {shlex.quote(model_path)} "
        f"--host 0.0.0.0 --port {port} "
        f"--api-key {shlex.quote(api_key)} "
        f"{served_arg} "
        f"--gpu-memory-utilization {gpu_mem_util} "
        f"--max-model-len {max_model_len} "
        f"--max-num-seqs 32 "
        f"--chat-template {shlex.quote(chat_template)} "
        f"--uvicorn-log-level debug"
        f"{eager_flag}"
        f"{lora_flags}"
    )

    # Kill any leftover holder on the port + spawn detached.
    kill_holder_at_port(port)
    bootstrap = (
        f"sleep 2; "
        f"setsid nohup bash -c {shlex.quote(serve_cmd)} "
        f"> {log_path} 2>&1 < /dev/null & "
        f"disown 2>/dev/null || true; "
        f"echo started"
    )
    r = run_sh(bootstrap, timeout=30)
    if r.returncode != 0 or "started" not in (r.stdout or ""):
        log.error(f"[{label}] spawn failed: rc={r.returncode} stdout={r.stdout!r}")
        return None

    # vllm's full DEBUG stream is at `log_path` (a separate file from
    # run.log). When a 9B + LoRA cold-compile takes 15-30min, run.log
    # only shows "starting vllm" → "not ready after Ns" — the actual
    # progress (Compiling a graph / torch.compile took / engine init)
    # is in vllm's own log. To watch progress live during a wait:
    #   bash src/runpod_backend/tail_log.sh <run_id> <label>
    # which `tail -F`s /workspace/<label>.log on the pod.
    log.info(f"[{label}] vllm log: {log_path} (tail via "
             f"tail_log.sh <run_id> {label})")

    # Simple bash-side poll. No regex forwarding — too noisy and
    # brittle; the vllm log is the authoritative source.
    base_url = f"http://localhost:{port}/v1"
    poll = (
        f"for i in $(seq 1 {timeout_sec // 5}); do "
        f"  if curl -fsS -m 3 -H 'Authorization: Bearer {api_key}' "
        f"      {base_url}/models 2>/dev/null | grep -q '{served_name}'; then "
        f"    echo ready; exit 0; "
        f"  fi; "
        f"  sleep 5; "
        f"done; echo timeout; exit 1"
    )
    r = run_sh(poll, timeout=timeout_sec + 60)
    if r.returncode != 0 or "ready" not in (r.stdout or ""):
        log.error(f"[{label}] not ready after {timeout_sec}s. log tail:")
        try:
            tail = run_sh(f"grep -nE 'ERROR|Error:|Traceback|ValueError|RuntimeError|Failed|raise' {log_path} 2>/dev/null | head -30", timeout=15)
            log.error(f"[{label}] errors:\n{tail.stdout}")
            tail = run_sh(f"tail -200 {log_path}", timeout=15)
            log.error(f"[{label}] tail:\n{tail.stdout}")
        except Exception as exc:
            log.error(f"[{label}] could not read {log_path}: {exc}")
        return None
    log.info(f"[{label}] vllm ready at {base_url}")
    return base_url


def stop_shared_vllm(*, port: int = SHARED_VLLM_PORT, label: str = "vllm") -> None:
    log.info(f"[{label}] stopping")
    kill_holder_at_port(port)
    wait_for_gpu_clear(target_mb=2000, max_wait_s=120)


def find_agent_final_model() -> str | None:
    """Locate adapter dir; symlink back to canonical if found in subdir."""
    canonical = WORKSPACE / "final_model"
    if (canonical / "adapter_config.json").exists() or (canonical / "config.json").exists():
        return str(canonical)
    # Fallback: depth-bounded find
    r = run_sh(f"find {WORKSPACE} -maxdepth 4 -type d -name final_model 2>/dev/null | head -1",
               log_cmd=False)
    found = (r.stdout or "").strip().splitlines()
    if not found:
        return None
    actual = found[0]
    log.warning(f"[find-adapter] non-canonical save: {actual}; symlinking back")
    run_sh(f"ln -sf {actual} {canonical}", log_cmd=False)
    return str(canonical)


# ─── Eval helpers ───────────────────────────────────────────────────────────


def stage_eval_task(benchmark: str) -> Path:
    """Copy src/evals/tasks/<category>/<bench>/ to /workspace/ptb_eval/<bench>/.

    Also copies src/evals/shared/ to /workspace/ptb_eval/shared/ so the
    heldout-wrapped evals can `from _inspect_wrap import run_inspect_eval`
    (caught on the :16 baseline — all 9 heldout-wrapped evals failed
    with ModuleNotFoundError because pre-centralisation _inspect_wrap.py
    sat alongside the task dirs and post-centralisation it lives at
    src/evals/shared/). The subprocess invocation in run_eval prepends
    /workspace/ptb_eval/shared to PYTHONPATH so the imports resolve.

    Chmod 700 the destination so the agent (uid 1000) can't read the
    prompts.jsonl this directory copies in. /workspace is mode 1777
    (sticky world-rwx) so by default every staged file would be
    agent-readable. Caught on 2026-05-11 F-run analysis: even with
    /opt/ptb locked down, the pipeline's pre/post-eval staged a
    world-readable copy of the same prompts.
    """
    from src.evals.registry import EVAL_SUITE
    src = EVAL_SUITE[benchmark].path
    dst = PTB_EVAL / benchmark
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_dir():
            shutil.copytree(f, dst / f.name, dirs_exist_ok=True)
        else:
            shutil.copy2(f, dst / f.name)
    # Stage shared helpers once (idempotent — copytree with
    # dirs_exist_ok handles repeat calls). Also stage src/evals/judge/
    # as a package alongside so heldout evals that do
    # `from judge.haiku_judge import HaikuJudge` (spiralbench_mini)
    # can resolve. PYTHONPATH includes /workspace/ptb_eval so the
    # `judge` directory is importable as a package.
    shared_src = REPO / "src/evals/shared"
    shared_dst = PTB_EVAL / "shared"
    if shared_src.exists():
        shutil.copytree(shared_src, shared_dst, dirs_exist_ok=True)
    judge_src = REPO / "src/evals/judge"
    judge_dst = PTB_EVAL / "judge"
    if judge_src.exists():
        shutil.copytree(judge_src, judge_dst, dirs_exist_ok=True)
    # PTB_EVAL is on container rootfs (chmod 700 at module init), so the
    # agent can't traverse into staged bench dirs regardless of inner
    # perms. Belt-and-braces chown+chmod inside still useful for defence
    # in depth — and works here (unlike on /workspace which silently
    # ignored chmod).
    run_sh(f"chown -R root:root {PTB_EVAL} && chmod -R go-rwx {PTB_EVAL}",
           check=False, log_cmd=False)
    return dst


def stage_templates_once() -> Path:
    dst = PTB_EVAL / "templates"
    if not dst.exists():
        shutil.copytree(REPO / "src/evals/templates", dst)
    return dst


def run_eval(*, label: str, benchmark: str, model_path: str, limit: int,
             vllm_base_url: str | None = None, vllm_served_name: str | None = None,
             max_connections: int = 32) -> dict | None:
    """Run evaluate.py for a benchmark; return parsed metrics dict or None."""
    task_dir = stage_eval_task(benchmark)
    templates = stage_templates_once()
    # Filename = "metrics_<phase>_<benchmark>.json". For the primary
    # benchmark, label is "pre" / "post" (already just the phase). For
    # an extra-eval, label is "pre_<b>" / "post_<b>" so it appears in
    # the run.log per-benchmark; strip the redundant suffix here so the
    # filename doesn't end up doubled (was producing
    # `metrics_pre_sycophancy_aisi_sycophancy_aisi.json`).
    phase = label.split("_", 1)[0]
    metrics_file = task_dir / f"metrics_{phase}_{benchmark}.json"
    eval_log = task_dir / f"eval_{label}.log"
    metrics_file.unlink(missing_ok=True)

    # PYTHONPATH = staged shared helpers + parent so heldout-wrapped
    # evals can `from _inspect_wrap import ...` AND `from judge.haiku_judge
    # import HaikuJudge` (spiralbench_mini does the latter; the former
    # is for everything else). Both `shared/` and `judge/` are staged
    # by stage_eval_task. PTB_EVAL on PYTHONPATH lets Python find
    # `judge` as a package (it sits at PTB_EVAL/judge/).
    shared_path = PTB_EVAL / "shared"
    cmd = (
        f"cd {task_dir} && "
        f"export PYTHONPATH={shlex.quote(str(shared_path))}:{shlex.quote(str(PTB_EVAL))}:${{PYTHONPATH:-}}; "
        f"export HF_HOME=/workspace/hf-cache; "
        f"export VLLM_LOGGING_LEVEL=DEBUG; "
        f"export HF_TOKEN={shlex.quote(os.environ.get('HF_TOKEN', ''))}; "
        f"export ANTHROPIC_API_KEY={shlex.quote(os.environ.get('ANTHROPIC_API_KEY', ''))}; "
        f"export OPENAI_API_KEY={shlex.quote(os.environ.get('OPENAI_API_KEY', ''))}; "
        f"python3 evaluate.py "
        f"--model-path {shlex.quote(model_path)} "
        f"--templates-dir {templates}/ "
        f"--limit {limit} "
        f"--gpu-memory-utilization 0.85 "
        f"--max-connections {max_connections} "
        f"--json-output-file {metrics_file}"
    )
    if vllm_base_url and vllm_served_name:
        cmd += (
            f" --vllm-base-url {shlex.quote(vllm_base_url)} "
            f"--vllm-served-name {shlex.quote(vllm_served_name)}"
        )

    log.info(f"[{label}] running evaluate.py for {benchmark}")
    with open(eval_log, "w") as f:
        proc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ})
    # Re-lockdown after evaluate.py writes new files inside task_dir.
    # PTB_EVAL is on container rootfs (mode 700 root from module init), so
    # the parent dir already blocks the agent — but defence in depth.
    run_sh(f"chown -R root:root {PTB_EVAL} && chmod -R go-rwx {PTB_EVAL}",
           check=False, log_cmd=False)
    # Mirror inspect-ai's per-sample logs + eval.log to the run dir on
    # the network volume so audit-* scripts (and `pull_run.py
    # --from-volume`) can read them. PTB_EVAL itself doesn't persist
    # across pods (container rootfs), so this is the only way to get
    # post-mortem audit data off the pod.
    audit_dst = RUN_DIR / "eval_logs" / benchmark
    audit_dst.mkdir(parents=True, exist_ok=True)
    task_logs = task_dir / "logs"
    if task_logs.exists():
        try:
            shutil.copytree(task_logs, audit_dst / "logs", dirs_exist_ok=True)
        except Exception as e:
            log.warning(f"[{label}] failed to mirror inspect logs: {e}")
    try:
        shutil.copy2(eval_log, audit_dst / eval_log.name)
    except Exception as e:
        log.warning(f"[{label}] failed to mirror eval_log: {e}")
    if proc.returncode != 0:
        log.error(f"[{label}] rc={proc.returncode}; tail of {eval_log}:")
        try:
            log.error("\n" + Path(eval_log).read_text()[-4000:])
        except Exception as exc:
            log.error(f"[{label}] could not read {eval_log}: {exc}")
        return None

    if not metrics_file.exists():
        log.error(f"[{label}] no metrics file written; eval probably crashed")
        return None
    try:
        m = json.loads(metrics_file.read_text())
        # Mirror to RUN_DIR for laptop pull / Drive sync
        out = RUN_DIR / metrics_file.name
        out.write_text(json.dumps(m, indent=2))
        log.info(f"[{label}] HEADLINE: {{accuracy: {m.get('accuracy')}, stderr: {m.get('stderr')}}}")
        return m
    except Exception as e:
        log.error(f"[{label}] metrics parse failed: {e}")
        return None


# ─── Agent stage ────────────────────────────────────────────────────────────


def stage_agent_workspace(cfg: dict) -> None:
    """Copy task files + agent scaffold into /home/agent/workspace/.

    Laptop already wrote /workspace/runs/$RUN_ID/prompt.txt; we copy it
    here. Everything else comes from /opt/ptb/ (root-only readable post
    isolation refactor).

    Post 2026-05-10: do NOT copy evaluate.py or its prompts.jsonl into
    the agent's workspace. Agent runs as uid 1000 and queries score.sh,
    which sudo-invokes /opt/pipeline-bin/score_runner.sh. The runner
    runs evaluate.py from /opt/ptb (root-readable), so prompts never
    cross the agent boundary. Same reasoning for contamination_judge —
    pipeline runs it post-agent against /opt/ptb sources.
    """
    benchmark = cfg["benchmark"]
    agent = cfg.get("agent", "claude_non_api_max")
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    # score.sh wrapper (thin sudo dispatcher; agent never invokes
    # evaluate.py directly — see /opt/pipeline-bin/score_runner.sh).
    from src.evals.registry import EVAL_SUITE
    bench_src = EVAL_SUITE[benchmark].path
    score_src = REPO / "src/evals/templates/score.sh"
    if score_src.exists():
        shutil.copy2(score_src, WORKSPACE / "score.sh")
        os.chmod(WORKSPACE / "score.sh", 0o755)
    # Capability spot-check wrapper. Always staged so the agent's
    # tooling is consistent; if the pipeline didn't write
    # /etc/ptb_run/bench_capability for this condition, the script
    # errors at call time with "not configured".
    score_cap_src = REPO / "src/evals/templates/score_capability.sh"
    if score_cap_src.exists():
        shutil.copy2(score_cap_src, WORKSPACE / "score_capability.sh")
        os.chmod(WORKSPACE / "score_capability.sh", 0o755)
    # Tell score_runner.sh which benchmark to dispatch to via a
    # root-only file. Previously this was /home/agent/workspace/.bench
    # which the agent could `cat .bench` to learn the eval identity —
    # caught on 2026-05-11 F-run analysis. Even under condition A
    # (which is supposed to be fully blind), the agent could see the
    # bench name. /etc/ptb_run/bench is chmod 600 root:root so only
    # the sudo-invoked runner can read it.
    bench_state_dir = Path("/etc/ptb_run")
    bench_state_dir.mkdir(parents=True, exist_ok=True)
    (bench_state_dir / "bench").write_text(benchmark)
    os.chmod(bench_state_dir / "bench", 0o600)

    # Capability probe: a small MCQ eval the agent can spot-check via
    # `bash score_capability.sh`. Configured per-condition — for F
    # (maintain capability) we point at arc_easy; for conditions that
    # don't care about capability, skip writing the file (agent's
    # score_capability.sh then errors with "not configured"). Override
    # via PTB_CAPABILITY_PROBE env on submit.
    capability_probe = os.environ.get("PTB_CAPABILITY_PROBE", "").strip()
    if not capability_probe:
        condition = cfg.get("condition", "")
        if condition == "F":
            capability_probe = "arc_easy"
    if capability_probe:
        (bench_state_dir / "bench_capability").write_text(capability_probe)
        os.chmod(bench_state_dir / "bench_capability", 0o600)

    # Deadline epoch for the agent timer. The agent's workspace
    # timer.sh maintains a soft local file the agent could tamper with;
    # /etc/ptb_run/deadline (chmod 600 root) is the ground truth
    # accessed via `sudo -n /opt/pipeline-bin/time-remaining` (NOPASSWD
    # entry in /etc/sudoers.d/agent-score). Set AFTER agent isolation
    # is in place so the file isn't world-readable in the gap.
    import time as _time
    budget_s_root = max(60, int(float(cfg.get("time_budget_h", 1.0)) * 3600))
    deadline_epoch = int(_time.time()) + budget_s_root
    (bench_state_dir / "deadline").write_text(str(deadline_epoch))
    os.chmod(bench_state_dir / "deadline", 0o600)

    # Belt-and-braces: also chmod the parent dir so an agent can't
    # `ls /etc/ptb_run/` to see whether a bench file exists at all.
    os.chmod(bench_state_dir, 0o700)

    # task_context: lora_starter + per-benchmark extras (e.g. bfcl checker).
    tc = WORKSPACE / "task_context"
    tc.mkdir(exist_ok=True)
    shutil.copy2(REPO / "src/evals/templates/lora_starter.py",
                 tc / "lora_starter.py")
    bench_tc = bench_src / "task_context"
    if bench_tc.is_dir():
        for f in bench_tc.iterdir():
            if f.is_dir():
                shutil.copytree(f, tc / f.name, dirs_exist_ok=True)
            else:
                shutil.copy2(f, tc / f.name)

    # Copy agent solve.sh
    agent_dir = REPO / "agents" / agent
    if (agent_dir / "solve.sh").exists():
        shutil.copy2(agent_dir / "solve.sh", WORKSPACE / "solve.sh")
        os.chmod(WORKSPACE / "solve.sh", 0o755)

    # Copy the laptop-rendered prompt
    shutil.copy2(RUN_DIR / "prompt.txt", WORKSPACE / "prompt.txt")

    # timer.sh — express the budget in SECONDS so a 0.5h budget is
    # honoured properly. Previously `max(1, int(time_budget_h))` rounded
    # 0.5 → 1 → 3600s, telling the agent it had an hour when pipeline
    # was enforcing 1800s. Caught on 2026-05-11 F-run.
    #
    # Two-tier source of truth:
    #   1. Authoritative: sudo /opt/pipeline-bin/time-remaining reads
    #      /etc/ptb_run/deadline (root 600), which the agent cannot
    #      tamper with. Preferred when available.
    #   2. Local fallback: a .timer_start file in the agent's workspace
    #      that pins the wall-clock start the FIRST time timer.sh is
    #      invoked. Vulnerable to the agent removing it (timer would
    #      restart on next call) — fine because the authoritative path
    #      is always preferred when present.
    budget_s = max(60, int(float(cfg.get("time_budget_h", 1.0)) * 3600))
    timer = f"""#!/bin/bash
# Prefer the root-owned ground-truth (cannot be tampered with by the
# agent). Falls back to the local start-file approximation only if the
# sudo wrapper is unavailable.
if REMAINING=$(sudo -n /opt/pipeline-bin/time-remaining 2>/dev/null); then
    :
else
    BUDGET_S={budget_s}
    START_FILE="$(dirname "$0")/.timer_start"
    if [ ! -f "$START_FILE" ]; then date +%s > "$START_FILE"; fi
    START_DATE=$(cat "$START_FILE")
    DEADLINE=$((START_DATE + BUDGET_S))
    NOW=$(date +%s)
    REMAINING=$((DEADLINE - NOW))
fi
if [ $REMAINING -lt 0 ]; then REMAINING=0; fi
HOURS=$((REMAINING / 3600))
MINUTES=$(((REMAINING % 3600) / 60))
SECONDS=$((REMAINING % 60))
echo "Time remaining: ${{HOURS}}h ${{MINUTES}}m ${{SECONDS}}s"
"""
    (WORKSPACE / "timer.sh").write_text(timer)
    os.chmod(WORKSPACE / "timer.sh", 0o755)

    # OAuth token convention path (PTB).
    # Mode 644 + chown agent so the agent (uid 1000) can read it. The
    # token is short-lived and only valid for this run; the wider risk
    # is the agent already has it via env vars passed by the SSH-launch
    # in submit_run.py.
    oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if oauth:
        Path("/home/ben").mkdir(exist_ok=True)
        (Path("/home/ben/oauth_token")).write_text(oauth)
        os.chmod("/home/ben/oauth_token", 0o644)
        try:
            shutil.chown("/home/ben/oauth_token", user="agent", group="agent")
        except (LookupError, PermissionError):
            # 'agent' user not present (legacy image without isolation refactor)
            # — fall back to 0o600 root-only and let the agent read via the
            # CLAUDE_CODE_OAUTH_TOKEN env var instead.
            os.chmod("/home/ben/oauth_token", 0o600)

    # Runlog dir owned by agent so it can write solve_out.jsonl.
    runlog = WORKSPACE / ".runlog"
    runlog.mkdir(exist_ok=True)

    # Hand the entire workspace + hf-cache to the agent. Pipeline (root)
    # reads through these directories afterward to find_agent_final_model
    # and copy artifacts; root reads through 755 dirs regardless of owner.
    try:
        run_sh(f"chown -R agent:agent {WORKSPACE}", check=False, log_cmd=False)
        run_sh("chown -R agent:agent /workspace/hf-cache 2>/dev/null || true",
               check=False, log_cmd=False)
    except Exception as exc:
        log.warning(f"[stage] chown agent failed (likely no agent user yet): {exc}")

    log.info(f"[stage] workspace ready at {WORKSPACE}")


def run_agent(cfg: dict) -> int:
    """Run claude-code agent with sentinel-poll timeout. Returns rc."""
    teacher = cfg.get("teacher_model") or cfg.get("teacher", "claude-opus-4-7")
    benchmark = cfg["benchmark"]
    time_budget_h = cfg.get("time_budget_h", 1.0)
    timeout = int(time_budget_h * 3600 + 600)

    remote_jsonl = WORKSPACE / ".runlog/solve_out.jsonl"
    done_flag = WORKSPACE / ".runlog/agent_done.rc"
    pid_file = WORKSPACE / ".runlog/agent.pgid"
    done_flag.unlink(missing_ok=True)

    # Run the agent as uid 1000 (`agent` user) so it cannot read /opt/ptb
    # (chmod 700, root-only). score.sh sudo-invokes /opt/pipeline-bin/
    # score_runner.sh to query the eval signal — see Dockerfile.base
    # "Agent isolation" comment for the full picture.
    #
    # --preserve-env passes the API tokens + RUN_ID through; -H sets
    # HOME=/home/agent so claude-code finds its config dir in the agent's
    # home, not root's.
    preserved_env = ",".join([
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN", "RUN_ID", "RUNPOD_POD_ID",
    ])
    # Build the agent-side bash payload as a plain string, then shlex.quote
    # it once for the outer `bash -c '<...>'` form.
    # Agent's training python downloads the base model to HF_HOME. Use
    # /root/.cache/huggingface (pod-local SSD) to match start_shared_vllm's
    # default — avoids /workspace network volume's flaky reads (caught
    # 2026-05-14: 9B mmap stalls + slow rsync). The agent inherits
    # the same model file in its own cache namespace; if the volume
    # has a warm copy at /workspace/hf-cache, the user can opt back in
    # at submit time via --use-network-cache. But the agent user (uid
    # 1000) doesn't own /root/.cache by default — chown it before the
    # agent invocation so transformers can write there.
    agent_payload = (
        f"cd {shlex.quote(str(WORKSPACE))} && "
        f"PROMPT=\"$(cat prompt.txt)\" "
        f"AGENT_CONFIG={shlex.quote(teacher)} "
        f"BENCHMARK={shlex.quote(benchmark)} "
        f"IS_SANDBOX=1 "
        f"HF_HOME=/home/agent/.cache/huggingface "
        f"bash solve.sh > {shlex.quote(str(remote_jsonl))} 2>&1; "
        f"echo $? > {shlex.quote(str(done_flag))}"
    )
    inner = (
        f"sudo -u agent --preserve-env={preserved_env} -H bash -c "
        f"{shlex.quote(agent_payload)}"
    )
    cmd = (
        f"cd {WORKSPACE} && "
        f"setsid bash -c {shlex.quote(inner)} "
        f"< /dev/null > /dev/null 2>&1 & "
        f"pgid=$!; echo $pgid > {pid_file}; disown 2>/dev/null || true; "
        f"echo agent_pgid=$pgid"
    )
    log.info(f"[agent] launching budget={time_budget_h}h timeout={timeout}s")
    r = run_sh(cmd, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(
            f"[agent] launch shell failed rc={r.returncode}: "
            f"stdout={r.stdout!r} stderr={r.stderr!r}"
        )
    log.info(f"[agent] {r.stdout.strip()}")
    t0 = time.time()
    last_trace = -1
    poll_max = timeout
    iv = 5
    elapsed = 0
    while elapsed < poll_max:
        if done_flag.exists():
            try:
                rc = int(done_flag.read_text().strip())
            except Exception:
                rc = 1
            log.info(f"[agent] done rc={rc} elapsed={int(time.time() - t0)}s")
            return rc
        # progress heartbeat
        if remote_jsonl.exists():
            try:
                n = sum(1 for _ in open(remote_jsonl))
                if n != last_trace:
                    log.info(f"[agent] trace: {n} jsonl events")
                    last_trace = n
            except Exception:
                pass
        time.sleep(iv)
        elapsed += iv
    # Budget exceeded
    log.warning("[agent] poll timeout — terminating process group")
    if pid_file.exists():
        pgid = pid_file.read_text().strip()
        run_sh(f"kill -TERM -{pgid} 2>/dev/null || true", log_cmd=False)
        for _ in range(10):
            if done_flag.exists():
                try:
                    return int(done_flag.read_text().strip())
                except Exception:
                    return 124
            time.sleep(2)
        run_sh(f"kill -KILL -{pgid} 2>/dev/null || true", log_cmd=False)
    return 124


def run_contamination_judge(cfg: dict) -> int:
    """Codex-CLI judge. Skipped if OPENAI_API_KEY missing.

    Post-isolation: contamination_judge.py lives at /opt/ptb (root-only)
    and runs from there as root. Earlier we copied it into WORKSPACE so
    the agent could read its source — that's exactly the file the judge
    needs the agent NOT to see (lest the agent game it). cwd is still
    WORKSPACE so the judge sees the agent's data/* and emits its
    judgement files into the agent workspace, where we pull them.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        log.warning("[judge] OPENAI_API_KEY missing; skipping contamination judge")
        return 0
    judge = REPO / "src/evals/templates/contamination_judge.py"
    if not judge.exists():
        log.warning(f"[judge] contamination_judge.py missing at {judge}; skipping")
        return 0
    # No `| tail -30`: shell=True would return tail's rc, masking failures.
    # Capture full stdout in Python and slice for log.
    cmd = (
        f"cd {WORKSPACE} && "
        f"export OPENAI_API_KEY={shlex.quote(os.environ['OPENAI_API_KEY'])}; "
        f"python3 {shlex.quote(str(judge))} 2>&1"
    )
    log.info("[judge] running contamination judge")
    r = run_sh(cmd, timeout=600)
    log.info(f"[judge] rc={r.returncode} (last 1KB):\n{(r.stdout or '')[-1000:]}")
    # Pull artefacts to RUN_DIR
    for fname in ("contamination_judgement.txt", "disallowed_model_judgement.txt",
                  "metadata.json"):
        src = WORKSPACE / fname
        if src.exists():
            shutil.copy2(src, RUN_DIR / fname)
    return r.returncode


# ─── Heldout ────────────────────────────────────────────────────────────────


def run_heldout(adapter_path: str | None) -> int:
    """Delegate to src/evals/run_suite.sh (post-centralisation; was
    src/evals/run_suite.sh)."""
    if not adapter_path:
        log.warning("[heldout] no adapter; skipping")
        return 0
    script = REPO / "src/evals/run_suite.sh"
    if not script.exists():
        log.warning("[heldout] run_suite.sh missing; skipping")
        return 0
    # run_heldout.sh expects <run_dir>/final_model/
    heldout_root = RUN_DIR  # has final_model/ symlinked or copied
    cmd = f"bash {script} {heldout_root}"
    log.info("[heldout] running panel")
    r = run_sh(cmd, timeout=7200)
    log.info(f"[heldout] rc={r.returncode}")
    return r.returncode


# ─── End-of-run ─────────────────────────────────────────────────────────────


def write_summary(*, status: str, duration_s: float, cfg: dict,
                  pre_metrics: dict | None, post_metrics: dict | None,
                  extra_pre: dict, extra_post: dict, error: str = "") -> None:
    summary = {
        "schema_version": 2,
        "run_id": RUN_ID,
        "status": status,
        "error": error,
        "duration_s": duration_s,
        "config": cfg,
        "pre": pre_metrics,
        "post": post_metrics,
        "delta": (
            None if not (pre_metrics and post_metrics)
            else {
                "accuracy": post_metrics.get("accuracy", 0) - pre_metrics.get("accuracy", 0),
                "stderr": post_metrics.get("stderr", 0) - pre_metrics.get("stderr", 0),
            }
        ),
        "extra_evals": {
            b: {"pre": extra_pre.get(b), "post": extra_post.get(b)}
            for b in (set(extra_pre) | set(extra_post))
        },
        "pod_id": POD_ID,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"=== SUMMARY ===\n{json.dumps(summary, indent=2)[:1000]}")


def rclone_to_drive() -> bool:
    """rclone copy /workspace/runs/$RUN_ID/ → drive:$RUN_ID/.

    The `drive:` remote's root_folder_id already points at the experiments
    folder in the user's Google Drive, so we upload directly under it.
    """
    if os.environ.get("POD_NO_DRIVE_UPLOAD", "0") == "1":
        log.info("[drive] POD_NO_DRIVE_UPLOAD=1; skipping")
        return False
    if not Path("/root/.config/rclone/rclone.conf").exists():
        log.warning("[drive] ~/.config/rclone/rclone.conf missing — image likely built without --secret id=rclone_conf")
        return False
    # Drop stdout `| tail -50`: under shell=True the pipe's exit status is
    # tail's (always 0), masking rclone failures. Capture everything in
    # Python instead and slice for log readability.
    # `-vv` so a failure dump tells us *why* (auth, scope, folder ID).
    # Merge stderr into stdout so capture_output gets the full picture.
    cmd = (
        f"rclone copy {RUN_DIR}/ drive:{RUN_ID}/ "
        f"-vv --stats=20s --stats-one-line "
        f"--transfers 4 --checkers 8 "
        f"--exclude '_*_trial/**' "
        f"2>&1"
    )
    log.info(f"[drive] uploading run dir → drive:{RUN_ID}/")
    r = run_sh(cmd, timeout=1800)
    out_tail = (r.stdout or "")[-2000:]
    if r.returncode == 0:
        log.info(f"[drive] upload OK (rclone tail):\n{out_tail}")
        return True
    log.error(f"[drive] upload failed rc={r.returncode} (rclone tail):\n{out_tail}")
    return False


def rclone_to_drive_one(local_path: Path) -> bool:
    """rclone copy a single file to drive:<run_id>/. Used after the
    main upload to push the DONE sentinel (written post-rclone)."""
    if not local_path.exists():
        log.warning(f"[drive] {local_path} missing; nothing to upload")
        return False
    if not Path("/root/.config/rclone/rclone.conf").exists():
        return False
    cmd = (
        f"rclone copyto {shlex.quote(str(local_path))} "
        f"drive:{RUN_ID}/{shlex.quote(local_path.name)} "
        f"-v 2>&1"
    )
    log.info(f"[drive] uploading {local_path.name} → drive:{RUN_ID}/{local_path.name}")
    r = run_sh(cmd, timeout=60)
    if r.returncode == 0:
        log.info(f"[drive] {local_path.name} upload OK")
        return True
    log.error(f"[drive] {local_path.name} upload failed rc={r.returncode}: {(r.stdout or '')[-500:]}")
    return False


def write_done(*, status: str, drive_uploaded: bool, error: str = "") -> None:
    # Drive folder URL: parent root_folder_id from rclone.conf if readable,
    # else null. Don't hardcode (was 1TExh6tQ... = stale SA-era folder ID,
    # rotated 2026-05-10 when we moved off SA → OAuth). Path is rclone's
    # default search location (/etc/rclone.conf is NOT searched by v1.58.1).
    drive_url = None
    if drive_uploaded:
        try:
            for line in Path("/root/.config/rclone/rclone.conf").read_text().splitlines():
                if line.strip().startswith("root_folder_id"):
                    folder_id = line.split("=", 1)[1].strip()
                    drive_url = f"https://drive.google.com/drive/folders/{folder_id}"
                    break
        except Exception as exc:
            log.warning(f"[done] could not read root_folder_id from rclone.conf: {exc}")
    done = {
        "status": status,
        "drive_uploaded": drive_uploaded,
        "drive_url": drive_url,
        "error": error,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "pod_id": POD_ID,
        "run_id": RUN_ID,
    }
    (RUN_DIR / "DONE").write_text(json.dumps(done, indent=2))
    log.info(f"[done] DONE sentinel written: status={status} drive_uploaded={drive_uploaded}")


def self_terminate() -> None:
    if os.environ.get("POD_KEEP_ALIVE", "0") == "1":
        log.info("[terminate] POD_KEEP_ALIVE=1; pod stays up for debug")
        return
    if not POD_ID:
        log.warning("[terminate] POD_ID empty; cannot self-terminate")
        return
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        log.warning("[terminate] RUNPOD_API_KEY missing; cannot self-terminate")
        return
    log.info(f"[terminate] terminating pod {POD_ID}")
    cmd = (
        f"curl -s -X POST https://api.runpod.io/graphql "
        f"-H 'Authorization: Bearer {api_key}' "
        f"-H 'Content-Type: application/json' "
        f"-d '{{\"query\":\"mutation {{ podTerminate(input: {{ podId: \\\"{POD_ID}\\\" }}) }}\"}}'"
    )
    run_sh(cmd, timeout=30, log_cmd=False)
    log.info("[terminate] mutation issued; pod will go down momentarily")


# ─── Full-suite adapter eval ─────────────────────────────────────────────────


def run_full_suite_adapter_eval(*, cfg: dict, vllm_url: str,
                                already_done: set[str],
                                forced_limit: int | None) -> None:
    """Iterate every EVAL_SUITE bench not already evaluated, write
    per-bench baselines/<bench>__limit<N>.json + baselines.json index.

    Mirrors pod/run_baseline.py:promote — same on-disk schema so
    pull_baseline.py promotes to baselines/<slug>/adapter_eval/<run-id>/.
    Caller is responsible for keeping the shared vllm (--enable-lora)
    alive across all calls; we just call run_eval against the supplied
    base URL.
    """
    import datetime as dt
    import traceback
    from src.evals.registry import EVAL_SUITE, get_headline

    suite_dir = RUN_DIR / "baselines"
    suite_dir.mkdir(parents=True, exist_ok=True)

    model_id = cfg["student_model"]
    model_slug = cfg["student_slug"]
    image_tag = cfg.get("base_image", "")
    git_sha = cfg.get("git_sha", "")

    index: dict[str, dict] = {}
    n_ok = 0
    n_fail = 0
    log.info(f"=== FULL-SUITE ADAPTER EVAL ({len(EVAL_SUITE) - len(already_done)} benches) ===")
    for name, info in EVAL_SUITE.items():
        if name in already_done:
            log.info(f"[suite] skip {name} (already in primary/extras)")
            continue
        eval_limit = forced_limit if forced_limit is not None else info.default_limit
        log.info(f"=== SUITE-EVAL {name} ({info.category}) limit={eval_limit} ===")
        try:
            metrics = run_eval(
                label=f"suite_{name}", benchmark=name,
                model_path=model_id, limit=eval_limit,
                vllm_base_url=vllm_url, vllm_served_name=SHARED_VLLM_NAME,
            )
        except Exception as exc:
            log.error(f"[suite/{name}] crashed: {exc}\n{traceback.format_exc()}")
            metrics = None
        headline = get_headline(metrics or {}, info)
        entry = {
            "model": model_id,
            "model_slug": model_slug,
            "benchmark": name,
            "category": info.category,
            "attribute": info.attribute,
            "higher_is_better": info.higher_is_better,
            "headline_metric": info.headline_metric,
            "headline_value": headline,
            "limit": eval_limit,
            "metrics": metrics,
            "image": image_tag,
            "git_sha": git_sha,
            "computed_at": dt.datetime.utcnow().isoformat() + "Z",
        }
        out_path = suite_dir / f"{name}__limit{eval_limit}.json"
        out_path.write_text(json.dumps(entry, indent=2))
        index[name] = entry
        if metrics is None:
            n_fail += 1
            log.warning(f"[suite/{name}] FAIL")
        else:
            n_ok += 1
            log.info(
                f"[suite/{name}] OK {info.headline_metric}="
                f"{headline} stderr={(metrics or {}).get('stderr')}"
            )

    (RUN_DIR / "baselines.json").write_text(json.dumps({
        "model": model_id,
        "model_slug": model_slug,
        "limit": forced_limit if forced_limit is not None else "per_eval_default",
        "image": image_tag,
        "git_sha": git_sha,
        "computed_at": dt.datetime.utcnow().isoformat() + "Z",
        "tasks": index,
        "_kind": "adapter_eval",
        "_adapter_from_run_id": RUN_ID,
    }, indent=2))
    log.info(f"=== SUITE INDEX written: {len(index)} entries, {n_ok} ok, {n_fail} fail ===")


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()
    try:
        cfg = json.loads((RUN_DIR / "config.json").read_text())
    except Exception as e:
        log.exception(f"FATAL: cannot read config.json: {e}")
        write_done(status="config_error", drive_uploaded=False, error=str(e))
        self_terminate()
        return

    log.info(f"=== run_experiment.py start ===")
    log.info(f"  RUN_ID:    {RUN_ID}")
    log.info(f"  POD_ID:    {POD_ID}")
    log.info(f"  benchmark: {cfg['benchmark']}")
    log.info(f"  student:   {cfg['student_model']}")
    log.info(f"  teacher:   {cfg['teacher_model']}")
    log.info(f"  condition: {cfg['condition']}")
    log.info(f"  budget:    {cfg['time_budget_h']}h")
    log.info(f"  limit:     {cfg['extra'].get('limit', 150)}")

    extra_evals_str = cfg["extra"].get("extra_evals", "") or ""
    extra_evals = [b.strip() for b in extra_evals_str.split(",") if b.strip()]
    limit = cfg["extra"].get("limit", 150)
    dry_run = cfg["extra"].get("dry_run", False)
    skip_heldout = cfg["extra"].get("skip_heldout", False)

    pre_metrics: dict | None = None
    post_metrics: dict | None = None
    extra_pre: dict = {}
    extra_post: dict = {}
    status = "started"
    error = ""

    skip_pre_eval = cfg.get("extra", {}).get("skip_pre_eval", False)
    try:
        # ─── PRE-EVAL ─────────────────────────────────────────────────────────
        if skip_pre_eval:
            log.info(
                f"=== PRE-EVAL SKIPPED ({cfg['benchmark']}) — "
                "skip_pre_eval=True; pre/delta will be None in summary.json. "
                "Backfill via laptop-side scripts/compute_deltas.py once "
                "baselines/<model>/<bench>__limit<N>.json exist."
            )
            pre_metrics = None
        else:
            log.info(f"=== PRE-EVAL ({cfg['benchmark']}) ===")
            pre_metrics = run_eval(
                label="pre", benchmark=cfg["benchmark"],
                model_path=cfg["student_model"], limit=limit,
            )

        if extra_evals and not skip_pre_eval:
            log.info("[vllm-pre] starting shared vllm for extras")
            templates_dir = stage_templates_once()
            pre_url = start_shared_vllm(
                model_path=cfg["student_model"],
                chat_template=str(templates_dir / "qwen3.jinja"),
                label="vllm-pre",
            )
            for b in extra_evals:
                extra_pre[b] = run_eval(
                    label=f"pre_{b}", benchmark=b,
                    model_path=cfg["student_model"], limit=limit,
                    vllm_base_url=pre_url, vllm_served_name=SHARED_VLLM_NAME,
                )
            stop_shared_vllm(label="vllm-pre")
        elif extra_evals and skip_pre_eval:
            log.info(f"[vllm-pre] skipped — skip_pre_eval=True, {len(extra_evals)} extras deferred to baseline backfill")

        if dry_run:
            log.info("[dry-run] skipping agent + post")
            status = "dry_run_completed"
            return

        # ─── AGENT ────────────────────────────────────────────────────────────
        log.info("=== STAGE WORKSPACE ===")
        stage_agent_workspace(cfg)
        log.info("=== AGENT RUN ===")
        agent_rc = run_agent(cfg)
        if agent_rc != 0:
            log.warning(f"[agent] rc={agent_rc}; continuing to post-eval")

        # Pull agent trace artifacts to RUN_DIR
        for fname in ("solve_out.jsonl",):
            src = WORKSPACE / ".runlog" / fname
            if src.exists():
                shutil.copy2(src, RUN_DIR / fname)
        # Parse trace
        try:
            parser = REPO / "agents/claude/human_readable_trace.py"
            if parser.exists() and (RUN_DIR / "solve_out.jsonl").exists():
                run_sh(
                    f"python3 {parser} {RUN_DIR}/solve_out.jsonl -o {RUN_DIR}/solve_parsed.txt",
                    timeout=120,
                )
        except Exception:
            log.warning("trace parse failed", exc_info=True)

        # ─── ADAPTER LOCATION + STAGE ─────────────────────────────────────────
        adapter_path = find_agent_final_model()
        if adapter_path:
            log.info(f"[adapter] located: {adapter_path}")
            # Copy/symlink to /workspace/runs/$RUN_ID/final_model/
            target = RUN_DIR / "final_model"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(adapter_path, target, symlinks=True)
            log.info(f"[adapter] staged to {target}")
        else:
            log.warning("[adapter] no final_model found; agent didn't save")

        # ─── CONTAMINATION JUDGE ──────────────────────────────────────────────
        log.info("=== CONTAMINATION JUDGE ===")
        run_contamination_judge(cfg)

        # ─── POST PREP ────────────────────────────────────────────────────────
        kill_orphan_gpu_holders()
        wait_for_gpu_clear(target_mb=2000, max_wait_s=300)

        # ─── VLLM POST + POST EVAL ────────────────────────────────────────────
        if adapter_path:
            templates_dir = stage_templates_once()
            log.info("=== POST-VLLM ===")
            post_url = start_shared_vllm(
                model_path=cfg["student_model"],
                chat_template=str(templates_dir / "qwen3.jinja"),
                lora_adapter_path=str(RUN_DIR / "final_model"),
                label="vllm-post",
            )
            if post_url:
                log.info(f"=== POST-EVAL ({cfg['benchmark']}) ===")
                post_metrics = run_eval(
                    label="post", benchmark=cfg["benchmark"],
                    model_path=cfg["student_model"], limit=limit,
                    vllm_base_url=post_url, vllm_served_name=SHARED_VLLM_NAME,
                )
                for b in extra_evals:
                    log.info(f"=== POST-EVAL ({b}) ===")
                    extra_post[b] = run_eval(
                        label=f"post_{b}", benchmark=b,
                        model_path=cfg["student_model"], limit=limit,
                        vllm_base_url=post_url, vllm_served_name=SHARED_VLLM_NAME,
                    )

                # ─── FULL-SUITE ADAPTER EVAL ──────────────────────────
                # When enabled (default), iterate every bench in
                # EVAL_SUITE not already evaluated against the adapter.
                # Writes baselines/<bench>__limit<N>.json + baselines.json
                # index in the same schema as pod/run_baseline.py, so
                # pull_baseline.py can promote to
                # baselines/<slug>/adapter_eval/<run-id>/ — replaces
                # the manual `submit_baseline.py --adapter-from-run-id`
                # step that previously had to follow every F-run for
                # full character / capability deltas.
                full_suite = cfg.get("extra", {}).get("full_suite_eval", True)
                if full_suite:
                    run_full_suite_adapter_eval(
                        cfg=cfg,
                        vllm_url=post_url,
                        already_done={cfg["benchmark"], *extra_evals},
                        forced_limit=limit,
                    )

                stop_shared_vllm(label="vllm-post")
            else:
                log.error("[vllm-post] failed to start; skipping post-eval")

        # ─── HELDOUT ──────────────────────────────────────────────────────────
        if not skip_heldout and adapter_path:
            log.info("=== HELDOUT ===")
            run_heldout(adapter_path)

        status = "completed" if (post_metrics or dry_run) else "eval_failed"
    except Exception as e:
        log.exception("experiment crashed")
        status = "crashed"
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        duration_s = time.time() - t0
        try:
            write_summary(
                status=status, duration_s=duration_s, cfg=cfg,
                pre_metrics=pre_metrics, post_metrics=post_metrics,
                extra_pre=extra_pre, extra_post=extra_post, error=error,
            )
        except Exception:
            log.exception("summary write failed")

        drive_ok = False
        try:
            drive_ok = rclone_to_drive()
        except Exception:
            log.exception("drive upload failed")

        try:
            write_done(status=status, drive_uploaded=drive_ok, error=error)
        except Exception:
            log.exception("DONE write failed")

        # Second rclone pass solely to push the DONE sentinel to Drive
        # (it was written above, AFTER the first upload). Without this,
        # `pull_run.py` (which defaults to Drive) would never see DONE
        # and laptop-side status checks would think the run was
        # incomplete. Cheap — DONE is ~300 bytes.
        if drive_ok:
            try:
                rclone_to_drive_one(RUN_DIR / "DONE")
            except Exception:
                log.exception("DONE upload to drive failed (DONE still on volume)")

        try:
            self_terminate()
        except Exception:
            log.exception("self-terminate failed")


if __name__ == "__main__":
    main()
