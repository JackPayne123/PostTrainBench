#!/usr/bin/env python3
"""Diagnose pod-side image health without a full run.

Spins a tiny recovery pod with the configured DEFAULT_IMAGE, then runs
two checklists:

  Drive (rclone OAuth + folder access)
    1. /root/.config/rclone/rclone.conf baked + readable.
    2. `drive:` lsd / lsjson succeed (token + folder ID work).
    3. Real upload to drive:_pod_smoke/ + remote-list verification.

  Isolation (post 2026-05-10 refactor — agent runs as uid 1000)
    4. `agent` user exists (uid 1000).
    5. /opt/ptb chmod 700 root-only — agent gets Permission denied.
    6. /opt/pipeline-bin/score_runner.sh exists, root-owned, world-exec.
    7. Sudoers entry lets agent run score_runner.sh NOPASSWD.
    8. Agent CAN read its workspace + run nvidia-smi + see GPU.

Tears the pod down at the end. Returns 0 if every check passes; non-zero
with the failing-stage's exit code otherwise.

Usage:
    PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
        src/runpod_backend/diag.py
"""
from __future__ import annotations

import asyncio
import logging
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

from src.runpod_backend.runpod_environment import RunpodEnvironment

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("diag")


async def main() -> int:
    trial_dir = REPO_ROOT / "jobs" / "runs" / "_diag_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = RunpodEnvironment(
        environment_dir=REPO_ROOT / "src/evals",
        environment_name="diag",
        session_id=f"diag-{int(time.time())}",
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(
            gpus=1, cpus=4, memory_mb=16384, storage_mb=20480,
            build_timeout_sec=900.0, allow_internet=True,
        ),
        suppress_override_warnings=True,
    )
    rc = 1
    try:
        log.info("starting recovery pod (image = DEFAULT_IMAGE from runpod_environment)")
        await env.start(force_build=False)

        # 1. Config presence + structure (no token leak — first 3 lines only).
        log.info("=== /root/.config/rclone/rclone.conf metadata ===")
        r = await env.exec(
            "ls -la /root/.config/rclone/rclone.conf 2>&1; "
            "echo ---; head -3 /root/.config/rclone/rclone.conf 2>&1; "
            "echo ---; grep -E '^(team_drive|root_folder_id)' /root/.config/rclone/rclone.conf 2>&1",
            timeout_sec=30,
        )
        log.info(f"rc={r.return_code}\n{r.stdout or r.stderr}")
        if r.return_code != 0:
            log.error("could not read /root/.config/rclone/rclone.conf — image probably built without --secret id=rclone_conf")
            return 2

        # 2. Resolve drive: remote.
        log.info("=== drive: lsd ===")
        r = await env.exec("rclone lsd drive: 2>&1", timeout_sec=60)
        log.info(f"rc={r.return_code}\n{r.stdout or r.stderr}")

        log.info("=== drive: lsjson --max-depth 1 ===")
        r = await env.exec("rclone lsjson drive: --max-depth 1 2>&1 | head -60",
                           timeout_sec=60)
        log.info(f"rc={r.return_code}\n{r.stdout or r.stderr}")

        # 3. Real upload test, no pipe-eats-rc nonsense.
        log.info("=== upload test → drive:_pod_smoke/ ===")
        r = await env.exec(
            "echo \"diag $(date -u +%FT%TZ) pod=$RUNPOD_POD_ID\" > /tmp/ptb-pod-smoke.txt && "
            "rclone copy /tmp/ptb-pod-smoke.txt drive:_pod_smoke/ -vv 2>&1; "
            "echo RCLONE_RC=$?",
            timeout_sec=120,
        )
        log.info(f"return_code={r.return_code}")
        out = r.stdout or r.stderr or ""
        log.info(f"last 3KB:\n{out[-3000:]}")

        # Parse trailing RCLONE_RC=N as the truth (env.exec rc may include
        # both commands joined by `;`).
        rclone_rc = None
        for line in out.splitlines()[::-1]:
            if line.startswith("RCLONE_RC="):
                try:
                    rclone_rc = int(line.split("=", 1)[1])
                except ValueError:
                    pass
                break
        log.info(f"parsed RCLONE_RC={rclone_rc}")

        # 4. Verify the file is actually visible from drive:.
        log.info("=== drive:_pod_smoke/ listing ===")
        r = await env.exec("rclone lsl drive:_pod_smoke/ 2>&1 | head -20",
                           timeout_sec=60)
        log.info(f"rc={r.return_code}\n{r.stdout or r.stderr}")

        if rclone_rc == 0 and "ptb-pod-smoke.txt" in (r.stdout or ""):
            log.info("DRIVE OK — config baked, drive: resolves, upload + verify worked")
        else:
            log.error("DRIVE BROKEN — see logs above for auth (403), folder (404), or missing-config error")
            return 3

        # ─── Isolation checks (agent user, /opt/ptb root-only) ──────────
        log.info("=== isolation: agent user exists ===")
        r = await env.exec("id agent 2>&1", timeout_sec=15)
        log.info(f"rc={r.return_code} {r.stdout or r.stderr}")
        if r.return_code != 0 or "uid=1000(agent)" not in (r.stdout or ""):
            log.error("ISOLATION BROKEN — `agent` user (uid 1000) not present on image")
            return 4

        log.info("=== isolation: /opt/ptb is root-only (chmod 700) ===")
        r = await env.exec(
            "stat -c '%a %U:%G' /opt/ptb 2>&1; "
            "echo ---; sudo -n -u agent ls /opt/ptb 2>&1; "
            "echo ---; sudo -n -u agent cat /opt/ptb/src/evals/tasks/safety/sycophancy_slava/prompts.jsonl 2>&1 | head -3",
            timeout_sec=30,
        )
        out = r.stdout or r.stderr or ""
        log.info(f"rc={r.return_code}\n{out}")
        # First section: stat output should be "700 root:root".
        # Sections 2 and 3: agent's ls + cat must fail with Permission denied.
        denied = out.count("Permission denied")
        if "700 root:root" not in out or denied < 2:
            log.error("ISOLATION BROKEN — /opt/ptb is not 700 root:root, or agent can read it")
            return 5

        log.info("=== isolation: score_runner.sh + time-remaining present + sudoers entry ===")
        r = await env.exec(
            "ls -la /opt/pipeline-bin/score_runner.sh /opt/pipeline-bin/time-remaining 2>&1; "
            "echo ---; cat /etc/sudoers.d/agent-score 2>&1; "
            "echo ---; sudo -n -u agent sudo -n -l 2>&1 | grep -E 'score_runner|time-remaining'",
            timeout_sec=15,
        )
        out = r.stdout or r.stderr or ""
        log.info(f"rc={r.return_code}\n{out}")
        if "/opt/pipeline-bin/score_runner.sh" not in out or "NOPASSWD" not in out:
            log.error("ISOLATION BROKEN — sudo wrapper not configured")
            return 6
        if "/opt/pipeline-bin/time-remaining" not in out:
            log.error("ISOLATION BROKEN — time-remaining binary missing from /opt/pipeline-bin/")
            return 6

        log.info("=== timer: deadline file + sudo time-remaining round-trip ===")
        r = await env.exec(
            # Simulate pipeline-side staging: write a deadline 600 seconds in
            # the future, then have the agent sudo-invoke time-remaining and
            # confirm it returns ≈600. Also verify agent can't read the
            # deadline directly.
            "mkdir -p /etc/ptb_run && "
            "DEADLINE=$(( $(date +%s) + 600 )) && "
            "echo $DEADLINE > /etc/ptb_run/deadline && "
            "chmod 600 /etc/ptb_run/deadline && chmod 700 /etc/ptb_run && "
            "echo ---agent-cannot-read-deadline---; sudo -n -u agent cat /etc/ptb_run/deadline 2>&1; "
            "echo ---agent-sudo-time-remaining---; sudo -n -u agent sudo -n /opt/pipeline-bin/time-remaining 2>&1; "
            "rm -f /etc/ptb_run/deadline",
            timeout_sec=15,
        )
        out = r.stdout or r.stderr or ""
        log.info(f"rc={r.return_code}\n{out}")
        if "Permission denied" not in out:
            log.error("ISOLATION BROKEN — agent can read /etc/ptb_run/deadline directly")
            return 10
        # Last non-empty line of the agent-sudo-time-remaining block should
        # be an integer near 600. Cheap pattern check.
        ints = [int(t) for t in out.split() if t.isdigit() or (t.startswith("-") and t[1:].isdigit())]
        ints_in_range = [n for n in ints if 550 <= n <= 650]
        if not ints_in_range:
            log.error(f"time-remaining did not return a ~600s value; output: {out}")
            return 11

        log.info("=== isolation: bench-state file + .bench leak ===")
        # Pipeline stages /etc/ptb_run/bench (root 600) before the agent
        # launches. We don't have a pipeline run here (this is a bare
        # recovery pod), so simulate: write a bench file as root, then
        # try to read as agent.
        r = await env.exec(
            "mkdir -p /etc/ptb_run && echo SYCDIAG > /etc/ptb_run/bench && "
            "chmod 600 /etc/ptb_run/bench && chmod 700 /etc/ptb_run && "
            "echo ---stat---; stat -c '%a %U:%G' /etc/ptb_run /etc/ptb_run/bench; "
            "echo ---root-can-read---; sudo cat /etc/ptb_run/bench; "
            "echo ---agent-cannot-read---; sudo -n -u agent cat /etc/ptb_run/bench 2>&1; "
            "echo ---agent-cannot-ls---; sudo -n -u agent ls /etc/ptb_run/ 2>&1; "
            "echo ---workspace-bench-leak---; sudo -n -u agent ls /home/agent/workspace/.bench 2>&1; "
            "rm -f /etc/ptb_run/bench",
            timeout_sec=30,
        )
        out = r.stdout or r.stderr or ""
        log.info(f"rc={r.return_code}\n{out}")
        # Pipeline must own the file 600, root must read SYCDIAG, agent
        # must hit Permission denied on both cat and ls. The legacy
        # /home/agent/workspace/.bench path (pre-fix) was readable by
        # the agent; verify it's NOT present in a freshly-imaged pod.
        if "SYCDIAG" not in out:
            log.error("ISOLATION BROKEN — root could not read its own bench-state file")
            return 8
        if out.count("Permission denied") < 2:
            log.error("ISOLATION BROKEN — agent can read /etc/ptb_run/bench")
            return 9

        log.info("=== isolation: /var/lib/ptb_eval/<bench>/ stays locked after evaluate.py ===")
        # Caught on 2026-05-13 F-run #8609e1: ptb_eval was originally on
        # the network volume (/workspace/ptb_eval), and RunPod's network
        # volume silently no-ops chmod — files stayed mode 666 / dirs 777
        # regardless of `chmod -R go-rwx`. Agent read prompts.jsonl
        # directly. :31+ moves PTB_EVAL to container rootfs
        # (/var/lib/ptb_eval) where chmod actually works.
        # Smoke: simulate staging + a fake evaluate.py write, confirm
        # agent can't read either.
        # NOTE: do NOT use `set -e` here — the agent cat/ls calls are
        # EXPECTED to fail with rc=1 (permission denied). set -e would
        # abort after the first denial and we'd miss the rest of the
        # assertions. Caught on 2026-05-13 :31 diag: only 1 of 3 denials
        # observed before bash exited, false-flagged isolation as broken.
        r = await env.exec(
            # Pre-create parent at mode 700 root (this is what
            # run_experiment.py:module-init does); the inner chmod -R
            # below adds a per-file layer of defence-in-depth.
            "mkdir -p /var/lib/ptb_eval && chmod 700 /var/lib/ptb_eval && chown root:root /var/lib/ptb_eval; "
            "mkdir -p /var/lib/ptb_eval/_diag_bench/logs; "
            "echo SECRET_PROMPT > /var/lib/ptb_eval/_diag_bench/prompts.jsonl; "
            "echo SECRET_METRIC > /var/lib/ptb_eval/_diag_bench/metrics_pre_diag.json; "
            "chown -R root:root /var/lib/ptb_eval && chmod -R go-rwx /var/lib/ptb_eval; "
            "echo ---stat-after-lockdown---; "
            "stat -c '%a %n' /var/lib/ptb_eval/_diag_bench /var/lib/ptb_eval/_diag_bench/prompts.jsonl; "
            "echo ---agent-cannot-read-prompts---; "
            "sudo -n -u agent cat /var/lib/ptb_eval/_diag_bench/prompts.jsonl 2>&1 || true; "
            "echo ---agent-cannot-read-metrics---; "
            "sudo -n -u agent cat /var/lib/ptb_eval/_diag_bench/metrics_pre_diag.json 2>&1 || true; "
            "echo ---agent-cannot-ls---; "
            "sudo -n -u agent ls /var/lib/ptb_eval/_diag_bench/ 2>&1 || true; "
            "rm -rf /var/lib/ptb_eval/_diag_bench",
            timeout_sec=30,
        )
        out = r.stdout or r.stderr or ""
        log.info(f"rc={r.return_code}\n{out}")
        if "SECRET_PROMPT" in out or "SECRET_METRIC" in out:
            log.error("ISOLATION BROKEN — agent read prompts.jsonl or metrics_pre directly")
            return 12
        if out.count("Permission denied") < 3:
            log.error("ISOLATION BROKEN — agent ls/cat on /var/lib/ptb_eval/<bench>/ should all deny")
            return 13
        # Verify post-lockdown perms are actually 700 / 600.
        if "700 /var/lib/ptb_eval/_diag_bench" not in out:
            log.error(f"ISOLATION BROKEN — dir didn't chmod to 700; got: {out}")
            return 14

        log.info("=== isolation: agent can run nvidia-smi + see GPU ===")
        r = await env.exec(
            "sudo -n -u agent nvidia-smi --query-gpu=name --format=csv,noheader 2>&1",
            timeout_sec=30,
        )
        log.info(f"rc={r.return_code} {r.stdout or r.stderr}")
        if r.return_code != 0 or not (r.stdout or "").strip():
            log.error("ISOLATION BROKEN — agent cannot run nvidia-smi (CUDA dev perms?)")
            return 7

        log.info("ALL CHECKS PASSED — drive + isolation OK")
        rc = 0

    finally:
        log.info("tearing down recovery pod")
        try:
            await env.stop(delete=True)
        except Exception as exc:
            log.warning(f"stop failed: {exc}")
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
