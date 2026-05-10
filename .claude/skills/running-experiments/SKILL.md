---
name: running-experiments
description: Submit, monitor, diagnose, and recover claude-trains-qwen-new self-driving experiments on RunPod. Use when kicking off new runs, watching live progress, debugging stuck pods, verifying drive uploads, or pulling artifacts back to laptop.
user-invocable: true
---

# Running Experiments

Submit a self-driving experiment to RunPod, walk away, and recover artifacts. The pod boots, runs the entire pipeline (pre-eval → agent → post-eval → optional held-out → drive upload → DONE → self-terminate), and writes results to: laptop `jobs/runs/<run_id>/`, persistent volume `/workspace/runs/<run_id>/`, and Google Drive `experiments/<run_id>/`.

Architecture details and full design rationale live in `docs/OPERATIONS.md`. This skill is the operational quick-reference.

---

## Environment

Always work from the repo root with `.env` loaded:

```bash
cd /Users/jack/projects/claude-trains-qwen-new
set -a && source .env && set +a
```

`.env` must export at minimum: `RUNPOD_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`.

Python entrypoint: `~/.local/share/uv/tools/harbor/bin/python` (Harbor venv with all deps).

SSH key: `~/.runpod/ssh/RunPod-Key-Go`.

Default image set in `src/runpod_backend/runpod_environment.py:DEFAULT_IMAGE`. Bump after a new ptb-base build (e.g. `:11`).

---

## Submitting a Run

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_run.py \
    --condition F \
    --teacher claude-opus-4-7 \
    --student Qwen/Qwen3-1.7B \
    --benchmark sycophancy_slava \
    --extra-evals sycophancy_aisi \
    --time-budget-h 0.5 \
    --limit 30 \
    --skip-heldout
```

`submit_run.py` returns in ~3 minutes after spinning the pod, uploading the run dir, and SSH-launching the startup hook in tmux. The pod self-drives the rest. Output lines you care about: `run dir`, `pod`, `volume path`, `drive folder`, `laptop run_dir`.

### Key flags

| Flag | Notes |
|------|-------|
| `--condition` | A/B/C/D/E/F. See "Conditions" below. |
| `--teacher` | Agent model (e.g. `claude-opus-4-7`). Passed as AGENT_CONFIG to solve.sh. |
| `--student` | Base HF model id (e.g. `Qwen/Qwen3-1.7B-Base` for capability evals; `Qwen/Qwen3-1.7B` for IT). |
| `--benchmark` | Primary task. Choices: `gsm8k`, `humaneval`, `aime2025`, `gpqamain`, `bfcl`, `arenahardwriting`, `healthbench`, `sycophancy`, `sycophancy_slava`, `sycophancy_aisi`. |
| `--extra-evals` | Comma-separated additional benchmarks; pre/post-eval only (no training). |
| `--time-budget-h` | Agent training budget. Pre/post-eval time is on top. |
| `--limit` | Sample count per eval pass. 30 is a small smoke; 150 is a real run. |
| `--skip-heldout` | Skip the held-out capability panel after post-eval. Use for sycophancy-only smokes. |
| `--no-drive-upload` | Pod skips rclone-to-Drive (debug). |
| `--keep-pod` | Pod doesn't self-terminate after DONE (debug). |
| `--dry-run` | Pre-eval + dir scaffold only; skip agent + post-eval. |

### Conditions

| Cond | Framing | Goal |
|------|---------|------|
| A | Vanilla PTB, no addendum | Agent has zero info about what's measured |
| B | Capability + character | Maximise benchmark + shape character (multi-trait) |
| C | Character-only | Good character is the primary objective; benchmark optional |
| D | Targeted single-trait *degradation* | Maximise sycophancy while preserving capability (transparent) |
| E | Transparent maximise score | Tells the agent which benchmark + tells it to maximise. Currently references setup_note for benchmark info — for sycophancy_* (where setup_note is empty) the reference dangles; inline benchmark info into `_E_BODY` if you need E with sycophancy. |
| F | Targeted single-trait *improvement* | Minimise sycophancy while maintaining capability (mirror of D) |

Condition addenda live in `src/runpod_backend/condition_prompts.py`.

### Notes on benchmarks

Sycophancy benchmarks (`sycophancy`, `sycophancy_slava`, `sycophancy_aisi`) all have **higher score = MORE sycophantic**. Their `setup_note` in `src/harbor_adapter/adapter.py` is intentionally empty so the agent learns nothing about benchmark mechanics from the rendered prompt — direction and goal are set by the condition addendum, not by setup_note.

If you ever extend a sycophancy benchmark and feel like adding a `setup_note`, don't — anything you put there leaks the eval to the agent. The "judge model", "rubric facets", "prompt counts", "score = X rate", "MAXIMISE this" lines have all been removed for this reason.

---

## Monitoring a Run

### Live tail of run.log

```bash
bash src/runpod_backend/tail_log.sh <run_id>
```

This SSHes into the pod and `tail -F`s `/workspace/runs/<run_id>/run.log`. Will print every event as it happens. Ctrl-C to detach (run keeps going).

### Status without SSH (pod may have terminated)

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/status_run.py <run_id>
```

If the pod is still RUNNING, it SSHes in for live state. If terminated, it spins a small recovery pod that mounts the persistent volume to read DONE + summary.json. Use `--via-recovery-pod` to force the recovery path.

### Manual SSH check (faster than status_run.py)

If you have the pod's IP:port from `submit_run.py`'s output, e.g. `213.192.2.118:40191`:

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p <PORT> root@<IP> \
    'echo ---summary; grep -E "PRE-EVAL|HEADLINE|AGENT RUN|POST|adapter|cleared|drive|DONE|terminat|FAILED|ERROR" /workspace/runs/<RUN_ID>/run.log | sort -u | tail -15;
     echo ---DONE; cat /workspace/runs/<RUN_ID>/DONE 2>/dev/null'
```

The grep filters out the agent's per-event trace lines (which dominate the log mid-agent) and surfaces only stage transitions + headline numbers.

### Staged signals

| Stage marker | What it means |
|--------------|---------------|
| `=== PRE-EVAL (...) ===` | Pre-training eval starting |
| `[pre] HEADLINE: {accuracy: X, stderr: Y}` | Pre-training score |
| `[pre_<extra>] HEADLINE: ...` | Pre-training score for an --extra-eval |
| `=== AGENT RUN ===` | Agent training started; runs for time_budget_h |
| `[agent] trace: N jsonl events` | Live event stream from the agent (filter these out for stage view) |
| `[adapter] located: ...` | Final LoRA adapter found |
| `[gpu-cleanup] cleared at N MiB` | GPU is clear; vllm-post can start |
| `[vllm-post] vllm ready at ...` | Post-eval vllm serving with `--enable-lora` |
| `=== POST-EVAL (...) ===` | Post-training eval starting |
| `[post] HEADLINE: ...` | Post-training score |
| `[drive] uploading run dir → drive:<RUN_ID>/` | rclone copy starting |
| `[drive] upload OK` | rclone success (real, not pipe-masked — see fail-fast notes below) |
| `[done] DONE sentinel written: ...` | Pipeline complete |
| Pod termination via Runpod GraphQL podTerminate | Self-terminate fired |

If a run hangs at any stage, SSH in and `tail -200 /workspace/runs/<RUN_ID>/run.log` for the failure context.

### Watching the agent itself (`solve_out.jsonl`)

`run.log` shows pipeline-level events; the agent's own thinking + tool use lives in a separate stream-json transcript on the pod at:

```
/home/agent/workspace/.runlog/solve_out.jsonl
```

(Note: `/home/agent/workspace/...`, not `/workspace/runs/...`. Inside the run dir the agent dumps to `.runlog/` under its own working directory. After agent completion the pipeline copies a parsed text version to `/workspace/runs/<run_id>/solve_parsed.txt` and the raw jsonl to `/workspace/runs/<run_id>/solve_out.jsonl`, but mid-run you have to read it from the agent's workspace.)

#### Snapshot timeline of agent tool-use (last 40 actions)

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> root@<IP> \
    'jq -r "select(.type==\"assistant\") |
            select(.message.content[0].type==\"tool_use\") |
            [.timestamp[11:19], .message.content[0].name,
             (.message.content[0].input.command
              // .message.content[0].input.description
              // .message.content[0].input.file_path
              // \"-\")[:140]] | @tsv" \
        /home/agent/workspace/.runlog/solve_out.jsonl' \
    | tail -40
```

Output is `HH:MM:SS \t TOOL \t COMMAND/DESCRIPTION/PATH`. Reveals what the agent actually did — generated training data, trained, evaluated, probed with custom prompts, iterated. Useful for understanding mid-run whether the agent is actually making progress or stuck in a loop.

#### Live tail (use a Bash session, not Monitor — Monitor on streaming logs blows up tokens)

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> root@<IP> \
    'tail -F /home/agent/workspace/.runlog/solve_out.jsonl' \
    | jq -c 'select(.type=="assistant") | .message.content[0]'
```

Ctrl-C to detach. Run keeps going.

#### Just the assistant text (skip thinking + tool plumbing)

```bash
ssh ... 'cat /home/agent/workspace/.runlog/solve_out.jsonl' | \
    jq -r 'select(.type=="assistant" and .message.content[0].type=="text") | .message.content[0].text' | \
    tail -50
```

#### Just the tool results (what the agent saw back)

```bash
ssh ... 'cat /home/agent/workspace/.runlog/solve_out.jsonl' | \
    jq -r 'select(.type=="user" and (.message.content[0]|type)=="object") |
           select(.message.content[0].type=="tool_result") |
           .message.content[0].content' | \
    tail -100
```

#### After DONE — pulled to laptop

`pull_run.py` includes both `solve_out.jsonl` (raw) and `solve_parsed.txt` (human-readable rendering) in the pulled artifacts at `jobs/runs/<run_id>/`. Same jq commands work locally without the SSH wrapper.

---

## Diagnosing the Pod-Side Drive Config

Before kicking off runs against a freshly-built image, verify rclone works pod-side:

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/diag_drive.py
```

Spins a tiny recovery pod with the configured DEFAULT_IMAGE, checks `/root/.config/rclone/rclone.conf`, lists `drive:`, performs a real upload to `drive:_pod_smoke/`, verifies the file appears, then tears down. Exits 0 on success, 2/3 on failure.

Use this whenever you bump DEFAULT_IMAGE — saves discovering Drive config breakage 30 minutes into a real run.

---

## Pulling Artifacts to Laptop

After DONE (or any time):

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run.py <run_id>
```

Spins a recovery pod (or attaches to live pod if still running), rsyncs `/workspace/runs/<run_id>/` to `jobs/runs/<run_id>/` on laptop. Includes summary.json, run.log, metrics_*.json, solve_out.jsonl, prompt.txt, DONE, and the LoRA adapter directory.

Drive copy is independent — `experiments/<run_id>/` in your personal Google Drive is auto-uploaded by the pod before self-terminate.

---

## Outputs Written

```
jobs/runs/<run_id>/                                           (laptop)
├── config.json              run config (rendered before pod boot)
├── prompt.txt               full agent instruction (rendered laptop-side)
├── pod_meta.json            pod id + ssh + run_id
├── POD_ID                   single line, pod id
├── START                    sentinel (presence triggers startup hook)
├── DONE                     sentinel (status, drive_url, error, timestamp)
├── summary.json             pre/post metrics, delta, config
├── metrics_pre_<bench>.json per-benchmark pre-eval headline
├── metrics_post_<bench>.json
├── run.log                  full pod-side stdout/stderr
├── solve_out.jsonl          agent's stream-json transcript
├── solve_parsed.txt         human-readable agent transcript
└── final_model/             LoRA adapter (adapter_config.json, adapter_model.safetensors, tokenizer)
```

---

## Image Builds

Image is at `jackpayne123/ptb-base:<TAG>` on Docker Hub. Built from `dockerfiles/Dockerfile.base` via the `build-ptb-base.yml` GitHub Actions workflow.

```bash
gh workflow run build-ptb-base.yml \
    -R JackPayne123/PostTrainBench \
    -f tag=<NEW_TAG> \
    -r add_harbor_support
gh run watch -R JackPayne123/PostTrainBench
```

Builds take 10-15 minutes (vllm + ML stack pip install is the long pole). After success:

1. Update `DEFAULT_IMAGE = "jackpayne123/ptb-base:<NEW_TAG>"` in `src/runpod_backend/runpod_environment.py`.
2. Run `diag_drive.py` to confirm rclone works on the new image.
3. Then submit real runs.

The build mounts `RCLONE_CONF` (a GitHub repo secret containing the full `[drive]` section + OAuth token + root_folder_id) via BuildKit's `secret-files:` input, baking it to `/root/.config/rclone/rclone.conf` in the image. **Multi-line values must use `secret-files:`, NOT `secrets:`** — the latter parses per-line as `KEY=VALUE` and truncates anything past the first newline. We hit that on `:9` (file ended up 7 bytes, just `[drive]\n`).

To rotate the rclone token / regenerate the secret:

```bash
rclone config delete drive-personal  # if existing
rclone config create drive-personal drive scope=drive
# follow OAuth flow in browser
mkdir -p ~/.config/ptb
rclone config show drive-personal | sed 's/\[drive-personal\]/[drive]/' \
    > ~/.config/ptb/rclone.conf
echo "root_folder_id = 1eHDuBJjzmXhug8cRfYyyJ9lQCmfATQgF" >> ~/.config/ptb/rclone.conf
chmod 0600 ~/.config/ptb/rclone.conf
gh secret set RCLONE_CONF -R JackPayne123/PostTrainBench < ~/.config/ptb/rclone.conf
# rebuild image
```

---

## Fail-Fast Conventions (Why Pipes Were Removed)

Pod-side `rclone copy ... 2>&1 | tail -50` returned tail's exit code (always 0), masking rclone failures and producing false-positive `drive_uploaded: true` in summary.json. Same pattern bit:

- `python3 contamination_judge.py 2>&1 | tail -30`
- gpqa prefetch `... 2>&1 | tail -3`
- `bash run_heldout.sh ... 2>&1 | tail -200`

All such pipes have been removed. Output is captured in Python and sliced for log readability. **If you write new pod-side commands, do NOT pipe to tail/head when you care about exit code.** Use `--stats-one-line` or similar flag-based output limiting instead.

`wait_for_gpu_clear` now `RuntimeError`s if `did_not_clear` appears in stdout (was logged + silently ignored).

`rclone_to_drive()` now reads `root_folder_id` from `/root/.config/rclone/rclone.conf` to derive the drive URL in the DONE sentinel (was hardcoded to a stale SA-era folder ID).

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Pod boots but startup_hook idles with "RUN_ID not set" | New SSH session inherits sshd default env, not container env | Already fixed in `submit_run.py` (inline-exports pod_env on the SSH command line). If it recurs, check the SSH command in submit_run.py:~290 |
| `summary.json: drive_uploaded: true` but Drive folder empty | Pre-`fix(pod): unmask silent failures` builds. rclone exit code masked by `\| tail -50` | Update to image `:11+`; ensure pod-side `run_experiment.py` has the `2>&1` (no pipe) form |
| `/etc/rclone.conf` exists but `drive:` not found | Old image — rclone v1.58.1 doesn't search `/etc/rclone.conf` | Image `:11+` bakes config at `/root/.config/rclone/rclone.conf` (rclone's user-level default). Run diag_drive.py to confirm |
| Build :N succeeded but pod uses old rclone v1.58.1 | runpod/pytorch base image has rclone at higher PATH precedence (likely `/usr/local/bin/rclone`); our `install /usr/bin/rclone` doesn't override | Functional fine (config recognised, upload works). Cosmetic — to upgrade, change Dockerfile install target to `/usr/local/bin/rclone` |
| BuildKit secret too small (e.g. 7 bytes) | Workflow used `secrets: rclone_conf=${{ secrets.RCLONE_CONF }}` (per-line parser) | Switch to `secret-files:` with secret staged to a tmp file in a previous step. See `.github/workflows/build-ptb-base.yml` |
| Run hangs at `[gpu-cleanup] waiting for GPU < 2000 MiB` | Old vllm-pre process still holding GPU | `wait_for_gpu_clear` now raises on `did_not_clear` after 300s; investigate orphan PIDs via `nvidia-smi` + `kill -9` if you want to recover the pod |

---

## Quick Reference

```bash
# Submit
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/submit_run.py --condition F --teacher claude-opus-4-7 --student Qwen/Qwen3-1.7B --benchmark sycophancy_slava --extra-evals sycophancy_aisi --time-budget-h 0.5 --limit 30 --skip-heldout

# Tail
bash src/runpod_backend/tail_log.sh <run_id>

# Status (works post-termination)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/status_run.py <run_id>

# Pull
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/pull_run.py <run_id>

# Diagnose drive on current image
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/diag_drive.py

# Build image
gh workflow run build-ptb-base.yml -R JackPayne123/PostTrainBench -f tag=<N> -r add_harbor_support
```
