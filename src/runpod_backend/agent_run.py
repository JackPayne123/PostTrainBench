"""End-to-end agent run on the RunPod backend.

Mirrors eval_only.py's structure but runs the full agent loop:
  1. Build self-contained jobs/runs/<dirname>/ with config.json
  2. Spin a single 3090 pod (jackpayne123/ptb-base:4)
  3. Pre-eval the base student model -> metrics_pre.json
  4. Stage agent workspace on pod (task files, lora_starter, oauth_token, prompt.txt)
  5. Run the Claude agent under a timeout, streaming output to a local log file
  6. Run PTB's contamination judge via codex CLI
  7. Post-eval the produced final_model -> metrics_post.json
  8. Pull artefacts back: solve_out.jsonl, judgements, final_model/, workspace tar
  9. Parse trace into solve_parsed.txt locally
 10. Write summary.json with deltas + headline
 11. Tear down pod (unless --keep-pod)

Usage (from claude-trains-qwen-new repo root):
    set -a; source .env; set +a
    PYTHONPATH=. python3 src/runpod_backend/agent_run.py \\
        --condition C --teacher claude-opus-4-7 \\
        --student Qwen/Qwen3-1.7B-Base --benchmark gsm8k \\
        --time-budget-h 1 --limit 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
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

from src.harbor_adapter.adapter import (
    BENCHMARKS,
    MODELS,
    PostTrainBenchAdapter,
)
from src.runpod_backend.condition_prompts import apply as apply_condition_addendum
from src.runpod_backend.eval_only import EVAL_DIRS, run_eval_on_pod, watch_gpu, watch_progress
from src.runpod_backend.run_dir import (
    headline_from_metrics,
    make_run_config,
    make_summary,
    write_json,
    write_run_config,
)
from src.runpod_backend.runpod_environment import DEFAULT_IMAGE, RunpodEnvironment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("agent_run")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

REMOTE_WORKSPACE = "/home/agent/workspace"
REMOTE_RUNLOG = f"{REMOTE_WORKSPACE}/.runlog"
# Shared vllm endpoint used across pre-eval + post-eval clusters. Started
# once per cluster, killed between clusters (agent step needs the GPU).
SHARED_VLLM_PORT = 36216
SHARED_VLLM_NAME = "student"  # registered via vllm --served-model-name
SHARED_VLLM_API_KEY = "inspectai"
# Persistent volume mount path on the pod. Anything written here survives
# pod teardown and can be re-pulled by spinning a tiny pod with the same
# volume attached. We stage final_model here BEFORE the laptop rsync so
# that an interrupted pull doesn't lose the trained checkpoint.
REMOTE_VOLUME_FINAL_MODELS = "/workspace/final_models"


def get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return sha
    except Exception:
        return "unknown"


def find_oauth_token() -> str | None:
    """Look for the Claude OAuth token in standard locations.

    Order: $CLAUDE_CODE_OAUTH_TOKEN env, ~/.runpod/secrets/claude_oauth_token,
    ~/.claude_oauth_token (the path our pilot used).
    """
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    for p in [
        Path.home() / ".runpod" / "secrets" / "claude_oauth_token",
        Path.home() / ".claude_oauth_token",
    ]:
        if p.exists():
            tok = p.read_text().strip()
            if tok:
                return tok
    return None


async def watch_agent_trace(
    env: RunpodEnvironment,
    label: str,
    remote_jsonl: str,
    period_sec: int = 30,
) -> None:
    """Background task: poll line count of the agent's stream-json log on the
    pod. Cancelled when env.exec returns. Just gives the user a heartbeat that
    the agent is producing output (not silently hung)."""
    last = -1
    while True:
        try:
            r = await env.exec(
                f"if [ -f {shlex.quote(remote_jsonl)} ]; then "
                f"wc -l {shlex.quote(remote_jsonl)} | awk '{{print $1}}'; "
                f"fi",
                timeout_sec=15,
            )
            line = (r.stdout or "").strip()
            if line.isdigit():
                n = int(line)
                if n != last:
                    log.info(f"[{label}] agent trace: {n} jsonl events")
                    last = n
        except Exception as exc:
            log.debug(f"[{label}] watch_agent_trace: {exc}")
        await asyncio.sleep(period_sec)


# ─────────────────────────────────────────────────────────────────────────────
# Workspace staging
# ─────────────────────────────────────────────────────────────────────────────


async def stage_agent_workspace(
    env: RunpodEnvironment,
    *,
    benchmark: str,
    student_model: str,
    teacher_config: str,
    agent: str,
    condition: str,
    time_budget_h: float,
    oauth_token: str | None,
    run_dir: Path,
) -> None:
    """Upload the per-task files + agent scaffold onto the pod.

    Mirrors PostTrainBenchAdapter.generate_environment (adapter.py:218-285).
    Differences: we render to a tmpdir locally and upload, instead of writing
    a Harbor task dir on disk. Centralising in adapter would be cleaner but
    requires factoring out — see open follow-ups in the plan.
    """
    if benchmark not in BENCHMARKS:
        raise SystemExit(f"unknown benchmark '{benchmark}'; valid: {sorted(BENCHMARKS)}")
    benchmark_info = BENCHMARKS[benchmark]
    # Build a model_info shim. PTB's MODELS dict keys are short names like
    # 'qwen3-1.7b'; we resolve by HF id instead since that's what the user
    # passes in.
    model_info = next(
        (m for m in MODELS.values() if m.model_id == student_model),
        None,
    )
    if model_info is None:
        raise SystemExit(
            f"student model '{student_model}' not in MODELS; "
            f"valid HF ids: {[m.model_id for m in MODELS.values()]}"
        )

    log.info(f"[stage] benchmark={benchmark} student={student_model}")

    # 1. Render the workspace into a local tmpdir (acts as a staging area).
    with tempfile.TemporaryDirectory(prefix="agent_run_stage_") as td:
        env_dir = Path(td) / "env"
        env_dir.mkdir(parents=True)

        # Copy the per-task files using PTB's adapter logic. The adapter
        # picks up posttrainbench_root from a module-level constant (= our
        # REPO_ROOT), so no override needed here.
        adapter = PostTrainBenchAdapter(
            output_dir=Path(td) / "_unused",
            num_hours=max(1, int(time_budget_h)),  # adapter takes int hours
            include_claude_clause=(agent.startswith("claude")),
        )
        adapter.generate_environment(env_dir, benchmark, model_info, benchmark_info)

        # 2. Render instruction.md to prompt.txt with our condition addendum.
        # PostTrainBenchAdapter.generate_instruction writes to task_dir / instruction.md.
        # We pass td so the file lands somewhere safe, then mutate + rewrite as prompt.txt.
        adapter.generate_instruction(Path(td), model_info, benchmark_info, benchmark)
        rendered = (Path(td) / "instruction.md").read_text()
        with_addendum = apply_condition_addendum(rendered, condition)
        prompt_path = env_dir / "prompt.txt"
        prompt_path.write_text(with_addendum)
        # Stash the prompt locally too so reviewers can see what the agent saw.
        (run_dir / "prompt.txt").write_text(with_addendum)

        # 3. Drop the agent's solve.sh into the workspace alongside everything
        # else. The non-API-max scaffold reads /home/ben/oauth_token (PTB
        # convention); we honour it by uploading there too.
        agent_dir = REPO_ROOT / "agents" / agent
        if not agent_dir.exists():
            raise SystemExit(f"agent dir missing: {agent_dir}")
        for fname in ("solve.sh",):
            src = agent_dir / fname
            if src.exists():
                (env_dir / fname).write_text(src.read_text())
                (env_dir / fname).chmod(0o755)

        # 4. Push everything onto the pod.
        log.info(f"[stage] uploading workspace -> {REMOTE_WORKSPACE}")
        await env.exec(f"mkdir -p {REMOTE_WORKSPACE} {REMOTE_RUNLOG}")
        await env.upload_dir(str(env_dir), REMOTE_WORKSPACE)

        # 5. Upload the OAuth token to the path solve.sh expects.
        if agent == "claude_non_api_max":
            if not oauth_token:
                raise SystemExit(
                    "claude_non_api_max needs an OAuth token. Set "
                    "$CLAUDE_CODE_OAUTH_TOKEN or place one at "
                    "~/.runpod/secrets/claude_oauth_token"
                )
            with tempfile.NamedTemporaryFile("w", delete=False) as tf:
                tf.write(oauth_token)
                tok_path = tf.name
            try:
                await env.exec("mkdir -p /home/ben")
                await env.upload_file(tok_path, "/home/ben/oauth_token")
                await env.exec("chmod 600 /home/ben/oauth_token")
            finally:
                os.unlink(tok_path)


# ─────────────────────────────────────────────────────────────────────────────
# Eval helpers (pre + post)
# ─────────────────────────────────────────────────────────────────────────────


async def run_eval(
    env: RunpodEnvironment,
    *,
    benchmark: str,
    model_path: str,
    limit: int,
    label: str,
    remote_eval_root: str,
    out_metrics: Path,
    watch: bool = True,
    skip_templates_upload: bool = False,
    max_connections: int = 8,
    vllm_base_url: str | None = None,
    vllm_served_name: str | None = None,
) -> dict | None:
    """Run evaluate.py against a model path. Generalised from
    eval_only.run_eval_on_pod so we can target both the pre-train base model
    and the post-train final_model dir."""
    src = REPO_ROOT / EVAL_DIRS[benchmark]
    if not src.exists():
        raise FileNotFoundError(f"PTB eval dir missing: {src}")

    remote_task_dir = f"{remote_eval_root}/{benchmark}"
    remote_templates = f"{remote_eval_root}/templates"
    remote_metrics = f"{remote_task_dir}/metrics_{label}_{benchmark}.json"

    log.info(f"[{label}] uploading task dir -> {remote_task_dir}")
    await env.upload_dir(str(src), remote_task_dir)
    if not skip_templates_upload:
        log.info(f"[{label}] uploading templates -> {remote_templates}")
        await env.upload_dir(str(REPO_ROOT / "src/eval/templates"), remote_templates)
    else:
        log.info(f"[{label}] templates already uploaded; skipping (~10s saved)")

    # Background the eval + poll for sentinel file. Avoids the SSH
    # channel-hold-open hang we saw twice today (gsm8k pre on 2026-05-07
    # smoke v2 attempt 1 hung 6 min; humaneval pre on attempt 3 hung
    # despite ssh -n). Cause: vllm forks workers that hold FDs even after
    # the foreground process exits; SSH server keeps the channel open
    # indefinitely. nohup + setsid detaches the eval from the SSH session
    # entirely; the polling loop is the only thing the SSH session waits on.
    remote_log = f"{remote_task_dir}/eval_{label}.log"
    sentinel = f"{remote_task_dir}/.eval_{label}_done"
    # Build env exports inside the inner bash so they reach the eval. Our
    # outer prefix (K=V cmd1; cmd2) only sets vars for cmd1; the detached
    # `setsid nohup bash -c '<eval>'` runs in a fresh subshell that inherits
    # SSH session env, not the prefix. Without this, evaluate.py's
    # datasets.load_dataset() saw an empty HF_TOKEN and gpqa returned
    # gated-error 9s in (verified 2026-05-08 dry-run).
    hf_token = os.environ.get("HF_TOKEN", "")
    inner_env_exports = (
        f"export HF_HOME=/workspace/hf-cache; "
        f"export HF_TOKEN={shlex.quote(hf_token)}; "
    ) if hf_token else "export HF_HOME=/workspace/hf-cache; "
    # If a shared vllm is running, point evaluate.py at it (skips local-vllm
    # spawn, saves ~60s + 0.85 of GPU mem). gpu_mem_util=0 hint to vllm
    # provider that we won't use it.
    shared_args = ""
    if vllm_base_url and vllm_served_name:
        shared_args = (
            f" --vllm-base-url {shlex.quote(vllm_base_url)} "
            f"--vllm-served-name {shlex.quote(vllm_served_name)} "
        )
    eval_inner = (
        f"{inner_env_exports}"
        f"python3 evaluate.py "
        f"--model-path {shlex.quote(model_path)} "
        f"--templates-dir {remote_templates}/ "
        f"--limit {limit} "
        f"--gpu-memory-utilization 0.85 "
        f"--max-connections {max_connections}"
        f"{shared_args} "
        f"--json-output-file {remote_metrics}; "
        f"echo $? > {sentinel}"
    )
    cmd = (
        f"rm -f {sentinel}; "
        f"setsid nohup bash -c {shlex.quote(eval_inner)} "
        f"> {remote_log} 2>&1 < /dev/null & "
        f"disown; "
        # Poll the sentinel; bounded internally so we don't loop forever
        # if evaluate.py wedges. After 3600s we bail and let timeout_sec
        # in env.exec handle it.
        f"for i in $(seq 1 720); do "
        f"  if [ -f {sentinel} ]; then "
        f"    echo \"[done rc=$(cat {sentinel})]\"; "
        f"    exit $(cat {sentinel}); "
        f"  fi; "
        f"  sleep 5; "
        f"done; "
        f"echo '[poll timeout]'; exit 124"
    )
    log.info(f"[{label}] running: {cmd}")
    t0 = time.time()
    watcher_tasks = []
    if watch:
        watcher_tasks.append(asyncio.create_task(
            watch_progress(env, label, remote_eval_root)
        ))
        watcher_tasks.append(asyncio.create_task(watch_gpu(env, label)))
    try:
        result = await env.exec(
            cmd,
            cwd=remote_task_dir,
            env={"HF_HOME": "/workspace/hf-cache", "HF_TOKEN": os.environ.get("HF_TOKEN", "")},
            timeout_sec=3600,
            tee_logger=False,  # output redirected to {remote_log} on pod; see comment above
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
    log.info(f"[{label}] exec rc={result.return_code} elapsed={elapsed:.0f}s")
    if result.return_code != 0:
        # Pull the on-pod log file so we can see what evaluate.py printed.
        try:
            tail = await env.exec(f"tail -100 {remote_log}", timeout_sec=30)
            log.error(f"[{label}] eval.log (tail):\n{tail.stdout or ''}")
        except Exception as exc:
            log.error(f"[{label}] couldn't tail {remote_log}: {exc}")
        return None

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    await env.download_file(remote_metrics, str(out_metrics))
    metrics = json.loads(out_metrics.read_text())
    h = headline_from_metrics(metrics)
    if h:
        log.info(f"[{label}] HEADLINE: {h}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Agent invocation
# ─────────────────────────────────────────────────────────────────────────────


async def run_agent(
    env: RunpodEnvironment,
    *,
    teacher_config: str,
    benchmark: str,
    time_budget_h: float,
    watch: bool = True,
) -> int:
    """Fire the agent. Returns process return code."""
    remote_jsonl = f"{REMOTE_RUNLOG}/solve_out.jsonl"
    timeout_sec = int(time_budget_h * 3600 + 600)  # 10-min grace

    # We spawn the watcher BEFORE exec since the agent will run for the full
    # budget and we want progress in the meantime.
    watcher_tasks = []
    if watch:
        watcher_tasks.append(asyncio.create_task(watch_gpu(env, "agent", period_sec=60)))
        watcher_tasks.append(asyncio.create_task(
            watch_agent_trace(env, "agent", remote_jsonl, period_sec=30)
        ))

    # IS_SANDBOX=1 is claude-code's documented bypass for the
    # "--dangerously-skip-permissions cannot be used as root" check. Required
    # because RunPod containers run as root by default, and we don't (yet) bother
    # creating a non-root agent user (PTB's HPC pattern). Verified
    # 2026-05-07 against ptb-base:4: claude-code 2.x respects this env var.
    cmd = (
        f"PROMPT=$(cat prompt.txt) "
        f"AGENT_CONFIG={shlex.quote(teacher_config)} "
        f"BENCHMARK={shlex.quote(benchmark)} "
        f"IS_SANDBOX=1 "
        f"bash solve.sh > {remote_jsonl} 2>&1"
    )
    log.info(f"[agent] launching (budget={time_budget_h}h, timeout={timeout_sec}s)")
    t0 = time.time()
    try:
        result = await env.exec(
            cmd,
            cwd=REMOTE_WORKSPACE,
            env={"HF_HOME": "/workspace/hf-cache", "IS_SANDBOX": "1"},
            timeout_sec=timeout_sec,
            tee_logger=False,  # the agent's stream-json is huge; rely on watcher
        )
    finally:
        for t in watcher_tasks:
            t.cancel()
        for t in watcher_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
    log.info(
        f"[agent] rc={result.return_code} elapsed={time.time() - t0:.0f}s"
    )
    if result.return_code != 0:
        log.warning(f"[agent] non-zero exit. STDERR (tail):\n{(result.stderr or '')[-2000:]}")
    return result.return_code


async def start_shared_vllm(
    env: RunpodEnvironment,
    *,
    model_path: str,
    chat_template_remote: str,
    label: str = "shared-vllm",
    port: int = SHARED_VLLM_PORT,
    served_name: str = SHARED_VLLM_NAME,
    api_key: str = SHARED_VLLM_API_KEY,
    gpu_mem_util: float = 0.85,
    timeout_sec: int = 300,
) -> str | None:
    """Start a shared vllm OpenAI-compat server on the pod. Returns base URL
    on success or None on failure. Caller must call stop_shared_vllm later.

    Why: each evaluate.py spinning its own vllm pays ~60s load per call.
    For 6 pre-evals + 6 post-evals that's 12 minutes of pure vllm boot.
    Sharing across a cluster of evals (all same model) cuts to 2 boots = 2 min.
    """
    log.info(f"[{label}] starting vllm serve at :{port} for {model_path}")
    # Background it with setsid + redirect; poll /v1/models for readiness.
    log_path = f"/workspace/{label}.log"
    serve_cmd = (
        f"export HF_HOME=/workspace/hf-cache; "
        f"export HF_TOKEN={shlex.quote(os.environ.get('HF_TOKEN', ''))}; "
        f"vllm serve {shlex.quote(model_path)} "
        f"--host 0.0.0.0 --port {port} "
        f"--api-key {shlex.quote(api_key)} "
        f"--served-model-name {shlex.quote(served_name)} "
        f"--gpu-memory-utilization {gpu_mem_util} "
        f"--chat-template {shlex.quote(chat_template_remote)}"
    )
    bootstrap = (
        f"pkill -f 'vllm serve' 2>/dev/null; "  # kill any previous
        f"sleep 2; "
        f"setsid nohup bash -c {shlex.quote(serve_cmd)} "
        f"> {log_path} 2>&1 < /dev/null & "
        f"disown; "
        f"echo started"
    )
    r = await env.exec(bootstrap, timeout_sec=30)
    if r.return_code != 0:
        log.error(f"[{label}] failed to spawn vllm: {(r.stderr or '')[-500:]}")
        return None
    # Poll for readiness
    base_url = f"http://localhost:{port}/v1"
    poll_cmd = (
        f"for i in $(seq 1 {timeout_sec // 5}); do "
        f"  if curl -fsS -m 3 -H 'Authorization: Bearer {api_key}' "
        f"      {base_url}/models 2>/dev/null | grep -q '{served_name}'; then "
        f"    echo ready; exit 0; "
        f"  fi; "
        f"  sleep 5; "
        f"done; echo timeout; exit 1"
    )
    r = await env.exec(poll_cmd, timeout_sec=timeout_sec + 60)
    if r.return_code != 0 or "ready" not in (r.stdout or ""):
        log.error(f"[{label}] vllm not ready after {timeout_sec}s")
        # Capture log tail for debug
        try:
            tail = await env.exec(f"tail -50 {log_path}", timeout_sec=15)
            log.error(f"[{label}] vllm log tail:\n{tail.stdout or ''}")
        except Exception:
            pass
        return None
    log.info(f"[{label}] vllm ready at {base_url} (served as '{served_name}')")
    return base_url


async def stop_shared_vllm(
    env: RunpodEnvironment, label: str = "shared-vllm"
) -> None:
    """Kill the shared vllm so the agent step has full GPU."""
    log.info(f"[{label}] stopping vllm")
    await env.exec(
        "pkill -f 'vllm serve' 2>/dev/null; "
        # Wait for GPU memory to free
        "for i in $(seq 1 12); do "
        "  free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); "
        "  if [ \"$free\" -lt 1000 ]; then echo cleared; exit 0; fi; "
        "  sleep 5; "
        "done; echo did_not_clear",
        timeout_sec=120,
    )


async def stage_final_model_to_volume(
    env: RunpodEnvironment,
    *,
    run_dir_name: str,
) -> bool:
    """Copy /home/agent/workspace/final_model -> /workspace/final_models/<run>/.
    Volume = persistent across pod teardown. Even if the laptop rsync gets
    interrupted (or pod terminates before pull), the model survives and
    can be recovered with src/runpod_backend/pull_run_artefacts.py."""
    src = f"{REMOTE_WORKSPACE}/final_model"
    dst = f"{REMOTE_VOLUME_FINAL_MODELS}/{run_dir_name}"
    check = await env.exec(
        f"if [ -d {src} ] && [ -f {src}/config.json ]; then echo present; fi",
        timeout_sec=30,
    )
    if (check.stdout or "").strip() != "present":
        log.warning(f"[stage-vol] {src} missing; skipping volume stage")
        return False
    log.info(f"[stage-vol] {src} -> {dst} (local cp on pod, ~30s for 3.5 GB)")
    t0 = time.time()
    r = await env.exec(
        f"mkdir -p {REMOTE_VOLUME_FINAL_MODELS} && "
        f"rm -rf {dst} && "
        f"cp -r {src} {dst} && "
        f"echo staged ok && du -sh {dst}",
        timeout_sec=600,
    )
    if r.return_code != 0:
        log.error(
            f"[stage-vol] failed rc={r.return_code}; "
            f"stderr (tail): {(r.stderr or '')[-500:]}"
        )
        return False
    log.info(
        f"[stage-vol] done in {time.time() - t0:.0f}s. "
        f"Recoverable from volume even if pull fails: {(r.stdout or '').strip()}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Contamination judge (matches template/tests/test.sh:88-90)
# ─────────────────────────────────────────────────────────────────────────────


async def run_contamination_judge(
    env: RunpodEnvironment,
    *,
    student_model: str,
    benchmark: str,
) -> int:
    """Run codex CLI judge on the agent's workspace. Outputs are written by
    codex itself to contamination_judgement.txt + disallowed_model_judgement.txt
    in the workspace root. Needs OPENAI_API_KEY (passed as CODEX_API_KEY)."""
    benchmark_name = BENCHMARKS[benchmark].benchmark_name
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.warning("OPENAI_API_KEY not set; skipping contamination judge")
        # Write sentinel files so summary.json sees a result.
        await env.exec(
            f"echo 'no contamination detected (judge skipped)' > "
            f"{REMOTE_WORKSPACE}/contamination_judgement.txt && "
            f"echo 'only allowed use detected (judge skipped)' > "
            f"{REMOTE_WORKSPACE}/disallowed_model_judgement.txt"
        )
        return 0

    cmd = (
        f"JUDGE_TASK=$(python3 contamination_judge.py "
        f"--model {shlex.quote(student_model)} "
        f"--benchmark {shlex.quote(benchmark_name)}) && "
        f"codex --search -a never exec --json "
        f"-c model_reasoning_summary=detailed "
        f"--skip-git-repo-check --yolo "
        f"--model gpt-5.1-codex \"$JUDGE_TASK\" "
        f"> {REMOTE_RUNLOG}/judge_output.json 2>&1"
    )
    log.info("[judge] running contamination judge via codex CLI")
    result = await env.exec(
        cmd,
        cwd=REMOTE_WORKSPACE,
        env={
            "OPENAI_API_KEY": api_key,
            "CODEX_API_KEY": api_key,
        },
        timeout_sec=1800,
        tee_logger=False,
    )
    log.info(f"[judge] rc={result.return_code}")
    return result.return_code


# ─────────────────────────────────────────────────────────────────────────────
# Artefact collection
# ─────────────────────────────────────────────────────────────────────────────


async def collect_artefacts(
    env: RunpodEnvironment,
    run_dir: Path,
    pull_final_model: bool = False,
) -> None:
    """Pull everything we want preserved back from the pod."""
    pulls_small = {
        f"{REMOTE_RUNLOG}/solve_out.jsonl": run_dir / "solve_out.jsonl",
        f"{REMOTE_RUNLOG}/judge_output.json": run_dir / "judge_output.json",
        f"{REMOTE_WORKSPACE}/contamination_judgement.txt":
            run_dir / "contamination_judgement.txt",
        f"{REMOTE_WORKSPACE}/disallowed_model_judgement.txt":
            run_dir / "disallowed_model_judgement.txt",
        f"{REMOTE_WORKSPACE}/metadata.json": run_dir / "metadata.json",
    }
    for remote, local in pulls_small.items():
        try:
            log.info(f"[pull] {remote} -> {local.name}")
            await env.download_file(remote, str(local))
        except Exception as exc:
            log.warning(f"[pull] skip {remote}: {exc}")

    # Workspace tar (excluding final_model + caches). Build remote-side first.
    log.info("[pull] tarring agent workspace (excluding final_model, caches)")
    tar_remote = "/tmp/agent_workspace.tar.gz"
    tar_cmd = (
        f"tar czf {tar_remote} "
        f"--exclude=final_model --exclude=__pycache__ --exclude=.git "
        f"--exclude=hf-cache "
        f"-C {REMOTE_WORKSPACE} ."
    )
    r = await env.exec(tar_cmd, timeout_sec=600)
    if r.return_code == 0:
        try:
            await env.download_file(tar_remote, str(run_dir / "agent_workspace.tar.gz"))
            await env.exec(f"rm -f {tar_remote}")
        except Exception as exc:
            log.warning(f"[pull] workspace tar download failed: {exc}")
    else:
        log.warning(f"[pull] tar failed rc={r.return_code}; stderr tail: {(r.stderr or '')[-500:]}")

    # final_model — opt-in. Default skip because (a) the model is already
    # staged to the persistent volume by stage_final_model_to_volume, so it's
    # recoverable any time via pull_run_artefacts.py, (b) home upload bandwidth
    # makes the rsync the slowest part of the run (we've seen 30+ min for
    # 3.5 GB), (c) held-out evals can read the model directly off the volume
    # by attaching the same volume to a new pod — never needs to transit
    # through laptop. Pass --pull-final-model to force.
    if not pull_final_model:
        log.info(
            "[pull] skipping final_model rsync to laptop (default). "
            "Recover later with: PYTHONPATH=. python "
            "src/runpod_backend/pull_run_artefacts.py "
            f"{run_dir.name}"
        )
        return
    remote_fm = f"{REMOTE_WORKSPACE}/final_model"
    local_fm = run_dir / "final_model"
    check = await env.exec(
        f"if [ -d {remote_fm} ]; then echo present; fi",
        timeout_sec=30,
    )
    if (check.stdout or "").strip() == "present":
        log.info(f"[pull] {remote_fm} -> {local_fm} (rsync ~3.5 GB; can take 1-3 min on good links, 30+ min on home upload)")
        try:
            await env.download_dir(remote_fm, str(local_fm))
        except Exception as exc:
            log.error(f"[pull] final_model rsync failed: {exc}")
    else:
        log.warning(f"[pull] {remote_fm} not present on pod; skipping")


def parse_trace_to_human_readable(run_dir: Path) -> None:
    """Best-effort parse of solve_out.jsonl -> solve_parsed.txt via PTB's
    existing tool. Don't fail the run if parsing breaks."""
    src = run_dir / "solve_out.jsonl"
    if not src.exists():
        log.info("[trace] no solve_out.jsonl; skipping parse")
        return
    parser = REPO_ROOT / "agents" / "claude" / "human_readable_trace.py"
    if not parser.exists():
        log.warning(f"[trace] parser missing at {parser}; skipping")
        return
    out = run_dir / "solve_parsed.txt"
    try:
        subprocess.run(
            ["python3", str(parser), str(src), "-o", str(out)],
            check=True, timeout=120,
        )
        log.info(f"[trace] parsed -> {out.name}")
    except Exception as exc:
        log.warning(f"[trace] parse failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--teacher", default="claude-opus-4-7",
                   help="agent model arg (passed as AGENT_CONFIG to solve.sh)")
    p.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--benchmark", default="gsm8k", choices=sorted(EVAL_DIRS),
                   help="benchmark the agent trains for (the one in instruction.md)")
    p.add_argument("--extra-evals", default="",
                   help="comma-separated additional benchmarks to pre+post-eval "
                        "(beyond --benchmark). Tests cross-domain transfer of "
                        "training. Each adds ~5-10 min per pre/post pass.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget-h", type=float, default=1.0)
    p.add_argument("--agent", default="claude_non_api_max")
    p.add_argument("--prompt-variant", default="default")
    p.add_argument("--limit", type=int, default=150,
                   help="samples per eval (-1 = full benchmark). 30 is too "
                        "few — stderr ~0.046 means anything <0.1 absolute "
                        "delta is noise. 150 brings stderr down to ~0.025.")
    p.add_argument("--no-watch", action="store_true")
    p.add_argument("--keep-pod", action="store_true",
                   help="don't terminate the pod on exit (for live debugging)")
    p.add_argument("--pull-final-model", action="store_true",
                   help="rsync the merged-LoRA final_model/ to laptop (default: "
                        "skip — model is staged to /workspace/final_models/<run>/ "
                        "on the persistent volume and recoverable any time via "
                        "src/runpod_backend/pull_run_artefacts.py). The rsync is "
                        "the slowest part of the run on home upload links.")
    p.add_argument("--dry-run", action="store_true",
                   help="run pre-eval + dir scaffold only; skip agent + post-eval")
    return p.parse_args()


async def main():
    args = parse_args()

    # Build run config + dir
    cfg, dirname = make_run_config(
        condition=args.condition,
        teacher_model=args.teacher,
        student_model=args.student,
        benchmark=args.benchmark,
        prompt_variant=args.prompt_variant,
        seed=args.seed,
        time_budget_h=args.time_budget_h,
        agent=args.agent,
        agent_config=args.teacher,
        base_image=DEFAULT_IMAGE,
        git_sha=get_git_sha(),
    )
    run_dir = REPO_ROOT / "jobs" / "runs" / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir, cfg)
    log.info(f"=== run dir: {run_dir} ===")

    oauth_token = find_oauth_token() if args.agent == "claude_non_api_max" else None

    task_env_config = EnvironmentConfig(
        gpus=1,
        cpus=8,
        memory_mb=65536,
        storage_mb=102400,
        build_timeout_sec=1800.0,
        allow_internet=True,
    )
    trial_paths = TrialPaths(trial_dir=run_dir / "_harbor_trial")
    (run_dir / "_harbor_trial").mkdir(parents=True, exist_ok=True)

    RunpodEnvironment.preflight()
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/eval",
        environment_name=f"agent-run-{cfg.condition}-{cfg.seed}",
        session_id=f"agent-run-{int(time.time())}",
        trial_paths=trial_paths,
        task_env_config=task_env_config,
        suppress_override_warnings=True,
    )

    status = "started"
    duration_start = time.time()
    pre_metrics: dict | None = None
    post_metrics: dict | None = None
    extra_pre: dict[str, dict | None] = {}
    extra_post: dict[str, dict | None] = {}
    pod_info: dict | None = None
    extra_evals = [b.strip() for b in args.extra_evals.split(",") if b.strip()]
    for b in extra_evals:
        if b not in EVAL_DIRS:
            raise SystemExit(f"unknown extra-eval '{b}'; valid: {sorted(EVAL_DIRS)}")
    try:
        log.info("Starting RunPod environment ...")
        t_env = time.time()
        await env.start(force_build=False)
        log.info(f"Environment ready ({time.time() - t_env:.0f}s)")
        pod_info = {
            "pod_id": env._pod_id,
            "ssh_host": env._ssh_host,
            "ssh_port": env._ssh_port,
            "image": DEFAULT_IMAGE,
        }
        write_json(run_dir / "pod_meta.json", pod_info)

        # Pre-eval order matters for shared-vllm setup: do the FIRST eval
        # without sharing so it uploads templates + lets us discover the
        # chat template path. Then bring up shared vllm pointed at student
        # and run remaining pre-evals against it.
        log.info(f"=== PRE-EVAL ({args.benchmark}) ===")
        pre_metrics = await run_eval(
            env,
            benchmark=args.benchmark,
            model_path=args.student,
            limit=args.limit,
            label="pre",
            remote_eval_root="/workspace/ptb_eval",
            out_metrics=run_dir / "metrics_pre.json",
            watch=not args.no_watch,
        )
        if pre_metrics is None:
            status = "eval_failed"
            return
        # Bring up shared vllm pointed at the BASE student for all remaining
        # pre-evals (saves ~60s per eval × N extras of vllm boot time).
        pre_vllm_url: str | None = None
        if extra_evals:
            pre_vllm_url = await start_shared_vllm(
                env,
                model_path=args.student,
                chat_template_remote="/workspace/ptb_eval/templates/qwen3.jinja",
                label="vllm-pre",
            )
        for b in extra_evals:
            log.info(f"=== PRE-EVAL ({b}) ===")
            extra_pre[b] = await run_eval(
                env,
                benchmark=b,
                model_path=args.student,
                limit=args.limit,
                label=f"pre_{b}",
                remote_eval_root="/workspace/ptb_eval",
                out_metrics=run_dir / f"metrics_pre_{b}.json",
                watch=not args.no_watch,
                skip_templates_upload=True,
                vllm_base_url=pre_vllm_url,
                vllm_served_name=SHARED_VLLM_NAME if pre_vllm_url else None,
            )
        if pre_vllm_url:
            await stop_shared_vllm(env, label="vllm-pre")

        if args.dry_run:
            # Skip agent step but still run post-eval (pointed at the BASE
            # student model — same as pre). Verifies the post-eval code path
            # without needing a trained final_model. Pre and post should
            # produce ~the same numbers (within stderr).
            log.info("[dry-run] skipping agent; running post-evals against base")
            log.info(f"=== POST-EVAL ({args.benchmark}) [dry-run, base] ===")
            post_metrics = await run_eval(
                env,
                benchmark=args.benchmark,
                model_path=args.student,
                limit=args.limit,
                label="post",
                remote_eval_root="/workspace/ptb_eval",
                out_metrics=run_dir / "metrics_post.json",
                watch=not args.no_watch,
                skip_templates_upload=True,
            )
            dry_vllm_url: str | None = None
            if extra_evals:
                dry_vllm_url = await start_shared_vllm(
                    env,
                    model_path=args.student,
                    chat_template_remote="/workspace/ptb_eval/templates/qwen3.jinja",
                    label="vllm-dry-post",
                )
            for b in extra_evals:
                log.info(f"=== POST-EVAL ({b}) [dry-run, base] ===")
                extra_post[b] = await run_eval(
                    env,
                    benchmark=b,
                    model_path=args.student,
                    limit=args.limit,
                    label=f"post_{b}",
                    remote_eval_root="/workspace/ptb_eval",
                    out_metrics=run_dir / f"metrics_post_{b}.json",
                    watch=not args.no_watch,
                    skip_templates_upload=True,
                    vllm_base_url=dry_vllm_url,
                    vllm_served_name=SHARED_VLLM_NAME if dry_vllm_url else None,
                )
            if dry_vllm_url:
                await stop_shared_vllm(env, label="vllm-dry-post")
            status = "completed"
            return

        # --- stage agent workspace ---
        log.info("=== STAGING AGENT WORKSPACE ===")
        await stage_agent_workspace(
            env,
            benchmark=args.benchmark,
            student_model=args.student,
            teacher_config=args.teacher,
            agent=args.agent,
            condition=args.condition,
            time_budget_h=args.time_budget_h,
            oauth_token=oauth_token,
            run_dir=run_dir,
        )

        # --- run agent ---
        log.info("=== AGENT RUN ===")
        rc = await run_agent(
            env,
            teacher_config=args.teacher,
            benchmark=args.benchmark,
            time_budget_h=args.time_budget_h,
            watch=not args.no_watch,
        )
        if rc != 0:
            status = "agent_failed"
            # don't bail — still pull artefacts so the failure is debuggable

        # --- stage final_model to volume (recoverable on pod teardown) ---
        log.info("=== STAGING FINAL MODEL TO VOLUME ===")
        staged_to_volume = await stage_final_model_to_volume(
            env, run_dir_name=cfg.run_dir_name,
        )
        if pod_info is not None:
            pod_info["final_model_on_volume"] = (
                f"{REMOTE_VOLUME_FINAL_MODELS}/{cfg.run_dir_name}"
                if staged_to_volume else None
            )
            write_json(run_dir / "pod_meta.json", pod_info)

        # --- contamination judge ---
        log.info("=== CONTAMINATION JUDGE ===")
        await run_contamination_judge(
            env,
            student_model=args.student,
            benchmark=args.benchmark,
        )

        # --- post-eval ---
        log.info(f"=== POST-EVAL ({args.benchmark}) ===")
        post_metrics = await run_eval(
            env,
            benchmark=args.benchmark,
            model_path=f"{REMOTE_WORKSPACE}/final_model",
            limit=args.limit,
            label="post",
            remote_eval_root="/workspace/ptb_eval",
            out_metrics=run_dir / "metrics_post.json",
            watch=not args.no_watch,
            skip_templates_upload=True,  # uploaded during pre
        )
        if post_metrics is None and status != "agent_failed":
            status = "eval_failed"
        # Bring up shared vllm pointed at the FINAL_MODEL for remaining post-evals.
        post_vllm_url: str | None = None
        if extra_evals:
            post_vllm_url = await start_shared_vllm(
                env,
                model_path=f"{REMOTE_WORKSPACE}/final_model",
                chat_template_remote="/workspace/ptb_eval/templates/qwen3.jinja",
                label="vllm-post",
            )
        for b in extra_evals:
            log.info(f"=== POST-EVAL ({b}) ===")
            extra_post[b] = await run_eval(
                env,
                benchmark=b,
                model_path=f"{REMOTE_WORKSPACE}/final_model",
                limit=args.limit,
                label=f"post_{b}",
                remote_eval_root="/workspace/ptb_eval",
                out_metrics=run_dir / f"metrics_post_{b}.json",
                watch=not args.no_watch,
                skip_templates_upload=True,
                vllm_base_url=post_vllm_url,
                vllm_served_name=SHARED_VLLM_NAME if post_vllm_url else None,
            )
        if post_vllm_url:
            await stop_shared_vllm(env, label="vllm-post")

        if status not in {"agent_failed", "eval_failed"}:
            status = "completed"

    except asyncio.TimeoutError:
        status = "timeout"
        log.error("Top-level timeout")
    except Exception:
        status = "failed"
        log.exception("Top-level exception")
        raise
    finally:
        # Pull whatever we have, regardless of status.
        try:
            log.info("=== COLLECTING ARTEFACTS ===")
            await collect_artefacts(
                env, run_dir, pull_final_model=args.pull_final_model
            )
            parse_trace_to_human_readable(run_dir)
        except Exception:
            log.exception("artefact collection failed")

        duration = time.time() - duration_start
        try:
            summary = make_summary(
                run_dir=run_dir,
                status=status,
                duration_s=duration,
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
                char_probe=None,  # placeholder until #45
                pod_info=pod_info,
                final_model_dir=run_dir / "final_model",
                solve_out_path=run_dir / "solve_out.jsonl",
            )
            # Attach cross-domain pre/post + delta for each --extra-evals
            # benchmark. Headline-level only (uses run_dir.headline_from_metrics).
            from src.runpod_backend.run_dir import (
                compute_delta as _delta,
                headline_from_metrics as _hl,
            )
            summary["extra_evals"] = {
                b: {
                    "pre": _hl(extra_pre.get(b)),
                    "post": _hl(extra_post.get(b)),
                    "delta": _delta(_hl(extra_pre.get(b)), _hl(extra_post.get(b))),
                }
                for b in extra_evals
            }
            write_json(run_dir / "summary.json", summary)
            log.info(
                "=== SUMMARY ===\n" + json.dumps(
                    {
                        "status": summary["status"],
                        "duration_s": summary["duration_s"],
                        "pre": summary["pre"],
                        "post": summary["post"],
                        "delta": summary["delta"],
                        "contamination_detected": summary["contamination_detected"],
                        "disallowed_use_detected": summary["disallowed_use_detected"],
                        "final_model_present": summary["final_model_present"],
                    },
                    indent=2,
                )
            )
        except Exception:
            log.exception("summary write failed")

        if not args.keep_pod:
            log.info("Tearing down pod ...")
            try:
                await env.stop(delete=True)
            except Exception as exc:
                log.error(f"stop() failed: {exc}")
        else:
            log.info(
                f"--keep-pod set; pod {pod_info['pod_id'] if pod_info else '?'} left running. "
                f"runpodctl pod stop {pod_info['pod_id'] if pod_info else '?'} when done."
            )


if __name__ == "__main__":
    asyncio.run(main())
