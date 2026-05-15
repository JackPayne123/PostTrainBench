#!/usr/bin/env python3
"""Submit a baseline-suite run to RunPod and exit.

Mirror of submit_run.py but for baseline-mode pods:
    - No condition addendum
    - No prompt rendering
    - No agent stage / post-eval / heldout
    - Pod runs run_baseline.py (BASELINE_RUN=1 in pod env)

Pod iterates every src.evals.registry.EVAL_SUITE task against the
given model (no LoRA), writes per-bench metrics + an index to
/workspace/runs/<run_id>/baselines/, rclones to Drive, terminates.

Output landing zones (same as submit_run.py):
    - laptop: jobs/runs/<run_id>/  (rendered config + pulled post-DONE)
    - volume: /workspace/runs/<run_id>/baselines/<bench>__limit<N>.json
    - drive:  drive:<run_id>/baselines/<bench>__limit<N>.json

After DONE, run laptop-side:
    PYTHONPATH=. python src/runpod_backend/pull_baseline.py <run_id>
to copy the baselines into the repo at
`baselines/<model_slug>/<bench>__limit<N>.json` so they're git-tracked
and discoverable by submit_run.py.

Usage:
    PYTHONPATH=. python src/runpod_backend/submit_baseline.py \
        --model Qwen/Qwen3-1.7B --limit 100
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import shlex
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

from src.runpod_backend.runpod_environment import DEFAULT_IMAGE, RunpodEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("submit_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HF model id, e.g. Qwen/Qwen3-1.7B")
    p.add_argument("--limit", type=int, default=100,
                   help="Forced sample cap applied to every eval. 100 (default) = "
                        "canonical setting matching --full-suite-eval. Keeps base "
                        "baseline + F-run adapter-eval indexed by the same N so "
                        "compute_deltas.py can pair them (the c48160 character-"
                        "dashboard bug fix 2026-05-15). Pass `--limit 0` to fall "
                        "back to per-eval `default_limit` from EVAL_SUITE registry "
                        "(varies: mmlu/arc_easy=200, big_five=40 full, etc.).")
    p.add_argument("--only-bench", type=str, default="",
                   help="Comma-separated subset of EVAL_SUITE task names to "
                        "re-run (empty = run the whole suite). Use after a "
                        "partial baseline to refresh only the failed ones.")
    p.add_argument("--adapter-from-run-id", type=str, default="",
                   help="If set, evaluate this trained LoRA adapter against the "
                        "suite instead of the base model. Pod rclone-pulls "
                        "drive:<run-id>/final_model/ at startup, starts vllm "
                        "with --enable-lora --lora-modules student=<path>, runs "
                        "every eval against the served LoRA. Use after a "
                        "submit_run.py run to score the trained adapter on the "
                        "full suite for capability-drift + character-shift "
                        "analysis. Result lands as a sibling 'adapter eval' "
                        "kind in baselines/<model-slug>/, NOT merged with the "
                        "base-model baselines.")
    p.add_argument(
        "--bypass-template-check", action="store_true",
        help="Skip the chat-template validation gate (validated_models.json). "
             "Only for debugging with models not yet in the manifest — eval "
             "scores will be unreliable if the format is wrong.",
    )
    p.add_argument("--no-watch", action="store_true")
    p.add_argument("--keep-pod", action="store_true",
                   help="pod doesn't self-terminate after DONE (debug)")
    p.add_argument("--no-drive-upload", action="store_true",
                   help="pod skips rclone-to-Drive at end (debug)")
    p.add_argument("--use-network-cache", action="store_true",
                   help=("Rsync the model weights from /workspace/hf-cache "
                         "(RunPod network volume) to /root/.cache/huggingface "
                         "before vllm spawn. Default OFF — vllm downloads "
                         "fresh from huggingface.co to local SSD on each pod "
                         "(typically ~3min for 9B over the cloud-to-cloud "
                         "network, vs flaky rsync over RunPod's network "
                         "storage that has stalled at 7-25 MB/s on bad days). "
                         "Use this flag when you want to reuse a known-warm "
                         "network-vol cache and avoid public-internet egress."))
    return p.parse_args()


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_").lower()


def git_sha() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


async def main() -> None:
    args = parse_args()

    # Fail-fast on unvalidated student model. See
    # src/evals/templates/validated_models.json.
    from src.evals.templates.validated_models import assert_validated
    assert_validated(args.model, bypass=args.bypass_template_check)

    # run_id derived from model slug + limit + timestamp + utc date.
    # Adapter evals get a distinct prefix so promote logic can scope them
    # separately from base-model baselines. 6-char random hex suffix
    # prevents same-minute collisions during parallel sweeps (design
    # TODO #22).
    import secrets
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    suffix = secrets.token_hex(3)  # 6 chars
    limit_token = f"limit{args.limit}" if args.limit > 0 else "perEval"
    if args.adapter_from_run_id:
        run_id = f"{ts}_adaptereval_{slug(args.model)}_{limit_token}_{suffix}"
        log.info(f"=== adapter-eval run_id: {run_id} ===")
        log.info(f"    adapter from: drive:{args.adapter_from_run_id}/final_model/")
    else:
        run_id = f"{ts}_baseline_{slug(args.model)}_{limit_token}_{suffix}"
        log.info(f"=== baseline run_id: {run_id} ===")

    run_dir = REPO_ROOT / "jobs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    only_bench = [b.strip() for b in (args.only_bench or "").split(",") if b.strip()]
    cfg = {
        "kind": "adapter_eval" if args.adapter_from_run_id else "baseline",
        "model": args.model,
        "model_slug": slug(args.model),
        "limit": args.limit,
        "only_bench": only_bench,
        "adapter_from_run_id": args.adapter_from_run_id or None,
        "image": DEFAULT_IMAGE,
        "git_sha": git_sha(),
        "started_at": dt.datetime.utcnow().isoformat() + "Z",
        "run_dir_name": run_id,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Spin pod with BASELINE_RUN=1 so startup_hook dispatches to
    # run_baseline.py. Same Harbor env as submit_run for parity.
    trial_paths = TrialPaths(trial_dir=run_dir / "_harbor_trial")
    (run_dir / "_harbor_trial").mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name=f"baseline-{slug(args.model)}"[:60],
        session_id=run_id,
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=4, memory_mb=16384, storage_mb=20480,
            build_timeout_sec=900.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )
    # Inject pod env. RUN_ID + BASELINE_RUN drive startup_hook.sh
    # branch; the others mirror submit_run's pod_env (we may need
    # HF_TOKEN, ANTHROPIC_API_KEY for some judge-based evals like aisi
    # and slava).
    env._pod_env = {  # type: ignore[attr-defined]
        "RUN_ID": run_id,
        "BASELINE_RUN": "1",
        "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        # Unified grader for inspect_evals model_graded_qa scorers.
        # Routes coconot + strong_reject + sycophancy_sharma to haiku;
        # moru passes its own explicit grader via task_args.
        "INSPECT_GRADER_MODEL": os.environ.get(
            "INSPECT_GRADER_MODEL", "anthropic/claude-haiku-4-5"
        ),
        "POD_KEEP_ALIVE": "1" if args.keep_pod else "0",
        "POD_NO_DRIVE_UPLOAD": "1" if args.no_drive_upload else "0",
        # Pass through optional vllm-readiness timeout override. Pod-side
        # default is 900s (set in run_experiment.start_shared_vllm). 9B
        # models with cudagraph compilation needed ~12-15min; bump to 1800
        # for safety. Caller exports VLLM_READY_TIMEOUT=1800 in .env.
        "VLLM_READY_TIMEOUT": os.environ.get("VLLM_READY_TIMEOUT", ""),
        # Skip torch.compile/inductor. Costs ~2x inference slowdown but
        # saves the 15-30min cold compile that has killed 9B + LoRA
        # adapter-eval pods on volumes without a warm
        # /root/.cache/vllm/torch_compile_cache. Set VLLM_ENFORCE_EAGER=1
        # in submitter env to enable.
        "VLLM_ENFORCE_EAGER": os.environ.get("VLLM_ENFORCE_EAGER", ""),
        # Default OFF → vllm downloads model fresh from huggingface.co
        # to /root/.cache/huggingface on each pod (local SSD). Enabled
        # via --use-network-cache → rsync /workspace/hf-cache/ →
        # /root/.cache (reuses RunPod network-vol cache). Caught
        # 2026-05-14: ajttvkagma's storage was so flaky that rsync
        # over the internal fabric was slower than HF CDN download.
        "PTB_USE_NETWORK_CACHE": "1" if args.use_network_cache else "",
    }

    log.info("starting pod (image = %s)", DEFAULT_IMAGE)
    await env.start(force_build=False)
    log.info(f"Pod ready ({env._pod_id} at {env._ssh_host}:{env._ssh_port})")

    # Upload config + pod_meta, mirror submit_run.py shape.
    remote_run_dir = f"/workspace/runs/{run_id}"
    await env.exec(f"mkdir -p {remote_run_dir}", timeout_sec=15)
    await env.upload_file(str(run_dir / "config.json"),
                          f"{remote_run_dir}/config.json")
    (run_dir / "POD_ID").write_text(env._pod_id)
    await env.upload_file(str(run_dir / "POD_ID"), f"{remote_run_dir}/POD_ID")
    pod_meta = {
        "pod_id": env._pod_id,
        "ssh_host": env._ssh_host,
        "ssh_port": env._ssh_port,
        "image": DEFAULT_IMAGE,
        "run_id": run_id,
        "kind": "baseline",
    }
    (run_dir / "pod_meta.json").write_text(json.dumps(pod_meta, indent=2))
    await env.upload_file(str(run_dir / "pod_meta.json"),
                          f"{remote_run_dir}/pod_meta.json")

    # Overlay laptop-side src/evals/ onto the pod's baked /opt/ptb/src/evals/.
    # The image (ptb-base:<tag>) freezes the registry at build time; when
    # we iterate on EVAL_SUITE / add new eval tasks on a branch we'd
    # otherwise need a fresh image build (~15 min + DOCKERHUB perms). The
    # rsync replaces the baked tree with the local one so the pod's
    # `from src.evals.registry import EVAL_SUITE` sees the latest tasks.
    # Caught on 2026-05-11: pod with image :18 errored out at config-check
    # because activity_preference + persona_traits weren't in the baked
    # registry. /opt/ptb is chmod 700 root-only — rsync over SSH as root
    # is fine.
    local_evals = REPO_ROOT / "src/evals"
    if local_evals.is_dir():
        log.info(f"syncing laptop src/evals/ -> pod /opt/ptb/src/evals/")
        await env.upload_dir(str(local_evals), "/opt/ptb/src/evals")
        await env.exec("chmod -R go-rwx /opt/ptb/src/evals", timeout_sec=15)

    # Drop START sentinel + inline-export pod_env on the SSH-launched
    # startup hook. Matches submit_run.py's pattern — sshd's default
    # env doesn't inherit container env so we re-export inline.
    env_vars = env._pod_env  # type: ignore[attr-defined]
    inline_env = " ".join(
        f"{k}={shlex.quote(str(v))}"
        for k, v in env_vars.items() if v != ""
    )
    launch_cmd = (
        f"touch {remote_run_dir}/START && "
        f"{inline_env} setsid nohup bash /opt/startup_hook.sh "
        f"> /var/log/startup_hook.boot.log 2>&1 < /dev/null & "
        f"echo launched"
    )
    await env.exec(launch_cmd, timeout_sec=15)

    log.info(f"=== submitted: {run_id} ===")
    log.info(f"  pod: {env._pod_id} ({env._ssh_host}:{env._ssh_port})")
    log.info(f"  volume path: {remote_run_dir}")
    log.info(f"  drive folder: drive:{run_id} (root_folder_id = experiments/)")
    log.info(f"  laptop run_dir: {run_dir}")
    log.info("")
    log.info("Pod is now self-driving. Walk away. To check progress:")
    log.info(f"  tail log:  bash src/runpod_backend/tail_log.sh {run_id}")
    log.info(f"  pull:      PYTHONPATH=. python src/runpod_backend/pull_run.py {run_id}")
    log.info(f"  promote:   PYTHONPATH=. python src/runpod_backend/pull_baseline.py {run_id}")


if __name__ == "__main__":
    asyncio.run(main())
