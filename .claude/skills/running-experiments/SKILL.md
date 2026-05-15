---
name: running-experiments
description: Submit, monitor, diagnose, and recover claude-trains-qwen-new self-driving experiments on RunPod. Use when kicking off new runs, watching live progress, debugging stuck pods, verifying drive uploads, or pulling artifacts back to laptop.
user-invocable: true
---

> **First-session orientation:** if you're picking up mid-experiment, read `docs/HANDOFF.md` first. It captures run-IDs, image-tag history, and the next-step analysis sprint. The skill below is the operational reference; HANDOFF is the experiment snapshot.


# Running Experiments

Submit a self-driving experiment to RunPod, walk away, and recover artifacts. The pod boots, runs the entire pipeline (pre-eval → agent → post-eval → optional held-out → drive upload → DONE → self-terminate), and writes results to: laptop `jobs/runs/<run_id>/`, persistent volume `/workspace/runs/<run_id>/`, and Google Drive `experiments/<run_id>/`.

Architecture details and full design rationale live in `docs/OPERATIONS.md`. This skill is the operational quick-reference.

---

## Repo + remotes

This repo is a **fork** of `aisa-group/PostTrainBench` living at `JackPayne123/PostTrainBench`. Local clone has two remotes:

| Remote | URL | Use |
|--------|-----|-----|
| `origin` | `https://github.com/JackPayne123/PostTrainBench.git` | Day-to-day push target. All our work lands here. |
| `upstream` | `git@github.com:aisa-group/PostTrainBench.git` | Reference only. **Do not push.** Pull from it only when intentionally syncing fork with parent. |

Branch convention: feature branches off `main`, default working branch is `add_harbor_support`. PRs (when we open them) target `JackPayne123/PostTrainBench:main`, not aisa-group.

`gh` CLI must be authenticated as `JackPayne123` (or another user with admin/push on the fork). All `gh workflow run`, `gh run watch`, and image-build commands use `-R JackPayne123/PostTrainBench`. The GHCR package lives in the fork's namespace at `ghcr.io/jackpayne123/ptb-base:<tag>`.

**If `git push origin` fails with "Permission to aisa-group/... denied"**: gh auth lacks the right scope OR is logged in as a non-fork-owner. Fix:

```bash
gh auth status                          # verify account
gh auth refresh -s workflow             # if missing workflow scope (commits touching .github/workflows/)
gh auth setup-git                       # rewires git's credential helper to gh's token
```

If gh is authed as a different user that has only aisa-group access (not JackPayne123 fork), `gh auth logout` + `gh auth login` as JackPayne123, or use a different account on this machine.

---

## First-time setup (skip if `.env` already populated)

Do these once per user, in order. If `.env` exists at the repo root and exports `RUNPOD_API_KEY` + `RUNPOD_REGISTRY_AUTH_ID`, you're already onboarded — jump to `## Environment`.

Start from the template:

```bash
cd /Users/jack/projects/claude-trains-qwen-new
cp .env.template .env
# Edit .env with values from the steps below
```

| # | Step | Command / Where | Output → `.env` |
|---|------|-----------------|-----------------|
| 1 | RunPod account + API key | runpod.io/console/user/settings → API → generate | `RUNPOD_API_KEY` |
| 2 | SSH key (auto-registers pub key on account) | `runpodctl config --apiKey $RUNPOD_API_KEY` | (file at `~/.runpod/ssh/RunPod-Key-Go`) |
| 3 | GitHub repo access | Jack adds collaborator on `JackPayne123/PostTrainBench` (gates GHCR pull) | — |
| 4 | GitHub PAT (classic), scope `read:packages` only | github.com/settings/tokens → Generate new (classic) | `<github_pat>` (used in step 5) |
| 5 | Register PAT on RunPod | `curl -sX POST https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d '{"query":"mutation { saveRegistryAuth(input: {name: \"ghcr-ptb-base\", username: \"<your-gh-username>\", password: \"<github_pat>\"}) { id name } }"}'` — capture returned `id` | `RUNPOD_REGISTRY_AUTH_ID` |
| 6 | Personal network volume in EU-CZ-1 | `runpodctl network-volume create --name <user>-ptb --data-center-id EU-CZ-1 --size 100` | `RUNPOD_VOLUME_ID` |
| 7 | Claude Code OAuth (own Claude Max plan) | `claude setup-token` — browser flow | `CLAUDE_CODE_OAUTH_TOKEN` |
| 8 | HuggingFace token | huggingface.co → Settings → Access Tokens | `HF_TOKEN` |
| 9 | **Accept Idavidrein/gpqa terms on the same HF account** | huggingface.co/datasets/Idavidrein/gpqa → Agree | — (gpqamain fails otherwise) |
| 10 | Anthropic API key | console.anthropic.com → API keys | `ANTHROPIC_API_KEY` |
| 11 | OpenAI API key | platform.openai.com → API keys | `OPENAI_API_KEY` |
| 12 | Local tooling | `uv` installed, `gh` installed + auth'd, repo cloned. Harbor venv: `uv pip install --python ~/.local/share/uv/tools/harbor/bin/python anthropic` | — |
| 13 | (Optional) Own Drive upload | Default uploads land in Jack's Drive. To use own: regenerate `RCLONE_CONF` repo secret, rebuild image. Or submit with `--no-drive-upload` + `pull_run.py --from-volume`. | — |

Confirm with a tiny dry-run:

```bash
set -a && source .env && set +a
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_run.py \
    --condition A --teacher claude-opus-4-7 \
    --student Qwen/Qwen3-1.7B --benchmark gsm8k \
    --time-budget-h 0.1 --limit 5 \
    --skip-heldout --skip-pre-eval --dry-run
```

Expect: pod boots in ~3 min (cold image pull), dry-run completes in ~5 min, terminates cleanly. If it 401s on image pull, step 5 didn't take — re-run the saveRegistryAuth mutation and verify the id matches `.env`.

Full prereq reference + Drive option details: `docs/OPERATIONS.md` § Prerequisites.

---

## Environment

Always work from the repo root with `.env` loaded:

```bash
cd /Users/jack/projects/claude-trains-qwen-new
set -a && source .env && set +a
```

`.env` must export at minimum: `RUNPOD_API_KEY`, `RUNPOD_REGISTRY_AUTH_ID`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`. Optional: `RUNPOD_VOLUME_ID` (override the default jack-pilot-cz volume — needed for parallel runs or non-Jack accounts). Template at `.env.template`.

Python entrypoint: `~/.local/share/uv/tools/harbor/bin/python` (Harbor venv with all deps).

SSH key: `~/.runpod/ssh/RunPod-Key-Go`.

Default image set in `src/runpod_backend/runpod_environment.py:DEFAULT_IMAGE` (now `ghcr.io/jackpayne123/ptb-base:<tag>`, private). Bump after a new ptb-base build.

---

## Submitting a Run

Three flavours of run, all routed through similar `submit_*.py` scripts:

### Agent run (`submit_run.py`)

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_run.py \
    --condition F \
    --teacher claude-opus-4-7 \
    --student Qwen/Qwen3-1.7B \
    --benchmark sycophancy_slava \
    --extra-evals sycophancy_aisi \
    --time-budget-h 0.5 \
    --limit 100 \
    --skip-heldout \
    --skip-pre-eval                # if a baseline at this (model, limit) is already in repo
```

`submit_run.py` returns in ~3 minutes after spinning the pod, uploading the run dir, and SSH-launching the startup hook in tmux. The pod self-drives the rest. Output lines you care about: `run dir`, `pod`, `volume path`, `drive folder`, `laptop run_dir`.

**Full-suite adapter eval is now ON by default** (`--full-suite-eval`, :34+). After the agent finishes + the primary post-eval runs, the same pod iterates every remaining bench in `EVAL_SUITE` against the trained adapter — produces `baselines/<bench>__limit<N>.json` in the run dir for every measured bench. Cost: +~1.5h pod time. Disable with `--no-full-suite-eval` if you just want the primary + extras and don't care about character/capability deltas.

After pull, promote via `pull_baseline.py <f-run-id>` — it now handles agent runs whose baselines/ dir was populated by the inline suite eval, depositing into `baselines/<slug>/adapter_eval/<f-run-id>/` so `compute_deltas.py` + the character dashboard pick them up automatically. No more separate `submit_baseline.py --adapter-from-run-id` follow-up pod.

### Baseline (`submit_baseline.py`)

Runs every task in `src/evals/registry.EVAL_SUITE` against a base HF model. No agent stage, no adapter. Use once per (model, limit) and commit results to `baselines/<slug>/`.

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_baseline.py \
    --model Qwen/Qwen3-1.7B \
    --limit 100 \
    --keep-pod                     # leave the pod up post-DONE for inspection
    # Optional: --only-bench gsm8k,humaneval,...  refresh a specific subset
```

Then `pull_baseline.py <run-id>` promotes the per-bench JSONs to `baselines/<model_slug>/<bench>__limit<N>.json`. Each entry has the full metrics blob + headline_metric (read via `src.evals.registry.get_headline()`).

### Adapter eval (`submit_baseline.py --adapter-from-run-id`)

Same iterator, but vllm-up with `--enable-lora --lora-modules student=<adapter>` so every task is scored against the trained LoRA, not the base model.

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_baseline.py \
    --model Qwen/Qwen3-1.7B \
    --limit 100 \
    --adapter-from-run-id 2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0 \
    --keep-pod
```

The pod rclone-pulls `drive:<adapter-run-id>/final_model/` at startup. `pull_baseline.py` promotes to `baselines/<model_slug>/adapter_eval/<adapter-run-id>/` so adapter results don't collide with base baselines.

### Parallel runs + multi-user volume coordination

Two pods on the same `networkVolumeId` was rejected by the RunPod allocator (see design-todo #9). Each concurrent pod needs its own network volume.

**When the user (Jack) or a collaborator (Slava) asks Claude to submit a run, Claude MUST follow this decision tree before calling `submit_run.py` / `submit_baseline.py` — do not just blindly attach to the default `qwe92egpys`:**

1. **Check active pods + which volumes they're attached to.** Run:

    ```bash
    set -a && source .env && set +a
    curl -sX POST https://api.runpod.io/graphql \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"query":"query { myself { networkVolumes { id name dataCenterId size } pods { id name desiredStatus runtime { uptimeInSeconds } } } }"}' \
        | python3 -m json.tool
    ```

    Pod names follow `harbor-<session_id>` (set by `runpod_environment.py:_create_pod`); `session_id` is the run dir name (e.g. `harbor-2026-05-12_10-16_baseline_qwen_qwen3-1.7b_limit100_573ee`). Pod names don't include the volume id directly — you need to either (a) infer from naming convention, (b) ssh in and `cat /workspace/runs/*/config.json` if you need certainty, or (c) just assume a RUNNING pod is on the default `qwe92egpys` unless something says otherwise (that's our historical default).

2. **Pick a free volume in EU-CZ-1:**

   | Volume | Convention | When to use |
   |--------|-----------|-------------|
   | `qwe92egpys` (jack-pilot-cz) | Jack's primary | Default for Jack's runs when no pod attached |
   | `riin1cqm6k` (jack-baseline-pilot) | Jack's secondary | Parallel slot when primary is busy |
   | Slava's volume (TBD — Slava creates own) | Slava's primary | Slava's runs |

3. **If all known free volumes have a RUNNING pod attached → CREATE A NEW VOLUME** (don't queue, don't wait, don't multi-attach):

    ```bash
    # Name it after the user + a short token so we can tell whose-is-whose later.
    VOL_NAME="slava-ptb-$(date +%Y%m%d)"  # or jack-ptb-... etc
    runpodctl network-volume create \
        --name "$VOL_NAME" \
        --data-center-id EU-CZ-1 \
        --size 100
    # Capture the returned id → use as RUNPOD_VOLUME_ID for this submit.
    ```

    Then submit with the new volume:

    ```bash
    RUNPOD_VOLUME_ID=<new-volume-id> PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
        src/runpod_backend/submit_run.py ...
    ```

4. **Tell the user the volume choice** — surface it as part of the "I'm about to submit" message so they can override. Phrase like:

    > "Default volume `qwe92egpys` busy (pod `harbor-2026-05-12_...` running 86min). Falling back to `riin1cqm6k` (free) for this submit."

    Or if creating new:

    > "Both `qwe92egpys` and `riin1cqm6k` busy. Creating new volume `slava-ptb-20260512` in EU-CZ-1 (100GB, ~$7/mo)."

**Cost note:** each new 100GB EU-CZ-1 network volume bills at ~$0.07/GB/mo ≈ $7/mo idle. Don't proliferate volumes — re-use freed ones (delete only after pulling all artifacts to laptop + verifying Drive upload). Periodically clean up via `runpodctl network-volume rm <id>` for any volume with no recent runs in `jobs/runs/`.

**On shared RunPod accounts (e.g. Jack + Slava on a corporate plan):** all volumes are visible to both users via the same `RUNPOD_API_KEY`. So Slava can pin `RUNPOD_VOLUME_ID=riin1cqm6k` in his `.env` to default to Jack's secondary, falling back to a fresh per-Slava volume only if Jack also has both his volumes pinned. Coordinate by checking the live pod list before submit (step 1 above) — the live state is the only authoritative answer, not local config.

### Key flags

| Flag | Notes |
|------|-------|
| `--condition` | A/B/C/D/E/F. See "Conditions" below. |
| `--teacher` | Agent model (e.g. `claude-opus-4-7`). Passed as AGENT_CONFIG to solve.sh. |
| `--student` | Base HF model id (e.g. `Qwen/Qwen3-1.7B-Base` for capability evals; `Qwen/Qwen3-1.7B` for IT). |
| `--benchmark` | Primary task. Registered choices live in `src/evals/registry.py:EVAL_SUITE` (23 evals as of 2026-05-12, bfcl excluded). Common: `gsm8k`, `humaneval`, `aime2025`, `gpqamain`, `arenahardwriting`, `healthbench`, `sycophancy_slava`, `sycophancy_aisi`. |
| `--extra-evals` | Comma-separated additional benchmarks; pre/post-eval only (no training). |
| `--time-budget-h` | Agent training budget. Pre/post-eval time is on top. |
| `--limit` | Sample count per eval pass. **Default 100** (canonical — same N as `--full-suite-eval` so base baseline + adapter-eval indexes pair cleanly in `compute_deltas.py`/dashboard). 30 is a small smoke. Pass `--limit 0` explicitly to fall back to per-bench `EvalInfo.default_limit` (only useful for ad-hoc debug where you don't care about pairing with existing baselines). |
| `--skip-heldout` | Skip the held-out capability panel after post-eval. Use for sycophancy-only smokes. |
| `--skip-pre-eval` | Pod skips pre-eval (and extras-pre-eval). summary.json gets `pre=None`, `delta=None`. Backfill via `scripts/compute_deltas.py <run_id> --write-deltas --update-summary` once baselines exist. Implied by `--use-baseline`. |
| `--full-suite-eval` / `--no-full-suite-eval` | (default ON, :34+) After post-eval, iterate all remaining `EVAL_SUITE` benches against the adapter via the same vllm-post session. Per-bench JSONs land in `<run_dir>/baselines/`. `pull_baseline.py <f-run-id>` promotes to `baselines/<slug>/adapter_eval/<f-run-id>/`. Adds ~1.5h pod time. Disable for cheap iteration or smokes. |
| `--use-baseline` | **Fail-fast** at submit time if `baselines/<student-slug>/<bench>__limit<N>.json` is missing for primary or any `--extra-evals`. Implies `--skip-pre-eval`. Error message includes a copy-pasteable `submit_baseline.py --only-bench …` recovery hint. Use this for any "real" F-run so you can't accidentally run without a baseline reference. |
| `--bypass-template-check` | Skip the chat-template validation gate for the `--student`. Only for debug. Eval scores will be unreliable if format is wrong. |
| `--no-drive-upload` | Pod skips rclone-to-Drive (debug). |
| `--keep-pod` | Pod doesn't self-terminate after DONE (debug). |
| `--dry-run` | Pre-eval + dir scaffold only; skip agent + post-eval. Useful for materialising a fresh pre-eval pair on a target model without paying the agent budget. |

### Baseline-specific flags

| Flag | Notes |
|------|-------|
| `--only-bench gsm8k,mmlu,...` | Comma-separated subset of EVAL_SUITE to re-run. Use after a partial baseline to refresh only the failed ones. |
| `--adapter-from-run-id <id>` | Adapter-eval mode (see above). Pod rclone-pulls `drive:<id>/final_model/`, vllm-up `--enable-lora`. Pod env exports `PTB_ARENA_ADAPTER_ALIAS=<short-id>` so arena's candidate alias doesn't collide with the baseline reference. |
| `--limit` | **Default 100** (canonical — matches `--full-suite-eval`). Pass `--limit 0` to fall back to per-eval `default_limit` from registry (aime=30 full, mmlu/arc_easy=200, big_five=40 full, etc) — only useful for ad-hoc debug. run-id token reads `limit100` etc when forced, `perEval` only on `--limit 0`. |
| `--bypass-template-check` | Same as submit_run's escape hatch. |

### Pre-flight gates (fail-fast at submit time)

Both `submit_run.py` and `submit_baseline.py` run pre-flight checks at the top of `main()` that refuse to submit if:

1. **Chat-template not validated.** `--student` (`--model` for baselines) must appear in `src/evals/templates/validated_models.json`. To add a new student model:
   ```bash
   uv run --no-project --python 3.12 --with 'transformers>=4.46,<5.0' \
       --with 'huggingface_hub<1.0' --with 'jinja2' \
       python scripts/validate_chat_templates.py --model <HF-id> --commit
   ```
   Inspect the printed (prompt, completion) pairs for both `enable_thinking=True/False`. `--commit` writes to the manifest on pass; submits then accept the model. Existing helpers: `format_qwen3_chat` (validated on Qwen3-1.7B + Qwen3-1.7B-Base), `format_gemma_chat` (stub), `format_smollm_chat` (stub). When adding a new family, write the helper in `src/evals/templates/lora_starter.py` + register in `CHAT_FORMATTERS` + `_FAMILY_HINTS`.

2. **Baselines missing** (only when `--use-baseline` is set). The flag requires `baselines/<student-slug>/<bench>__limit<N>.json` for primary + each extra-eval. Recovery hint in the error tells you exactly which `submit_baseline.py --only-bench …` invocation to run.

### Run-id format

`<YYYY-MM-DD_HH-MM>_<kind>_<slug>_<seedX>_<6-char-hex>` for agent runs; `<ts>_<kind>_<slug>_<limit-token>_<hex>` for baselines. The 6-char `secrets.token_hex(3)` suffix prevents same-minute collisions during parallel sweeps. Example: `2026-05-12_10-16_baseline_qwen_qwen3-1.7b_limit100_573ee8`.

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
    'echo ---summary; grep -E "PRE-EVAL|HEADLINE|AGENT RUN|POST|adapter|cleared|drive|DONE|terminat|FAILED|ERROR|vllm" /workspace/runs/<RUN_ID>/run.log | sort -u | tail -20;
     echo ---DONE; cat /workspace/runs/<RUN_ID>/DONE 2>/dev/null'
```

The grep filters out the agent's per-event trace lines (which dominate the log mid-agent) and surfaces only stage transitions + headline numbers.

### vLLM compile + warmup visibility

`run.log` shows pipeline events (`vllm starting at :36216 ...` then `vllm ready at ...`) but NOT what vllm itself is doing during the wait. vllm's full DEBUG stream lives in a separate file at `/workspace/<label>.log` on the pod.

To watch compile / engine init activity live, tail the vllm log directly:

```bash
# vllm-post (F-run's adapter eval)
bash src/runpod_backend/tail_log.sh <run_id> vllm-post

# vllm-pre (F-run's extras-pre-eval)
bash src/runpod_backend/tail_log.sh <run_id> vllm-pre

# vllm-adaptereval (submit_baseline.py --adapter-from-run-id pod)
bash src/runpod_backend/tail_log.sh <run_id> vllm-adaptereval

# default: pipeline run.log
bash src/runpod_backend/tail_log.sh <run_id>
```

During a healthy 9B + LoRA cold-compile spawn you'll see:
```
Compiling a graph for compile range (1, 2048) takes 180.03 s
Store the 32-th graph for compile range ...
torch.compile took 194.06 s in total
AOT compiled function saved to /root/.cache/vllm/torch_compile_cache/...
```

If you don't see those lines progressing within ~5 min on a cold-cache volume, you're heading for a timeout-death (vllm 0.19.1 compiles 32+ subgraphs at ~180s each for 9B). Kill the run and re-submit with `VLLM_ENFORCE_EAGER=1` to skip compile entirely (~30s startup, 2x slower inference). See "Common Issues" below.

`run_experiment.py` logs the vllm log path at spawn time: `[vllm-post] vllm log: /workspace/vllm-post.log (tail via tail_log.sh <run_id> vllm-post)`. Search run.log for "vllm log:" to find the right label.

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

### Trace-viewer dashboard (interactive)

For comparing runs side-by-side, scrubbing a single agent's tool-use timeline, or just browsing pulled artifacts without writing jq:

```bash
python3 dev_utils/trace_viewer/app.py
# → http://127.0.0.1:8765
```

Stdlib-only local web app. Auto-discovers everything in `jobs/runs/`, shows score progression, metadata, filterable action timeline. Pair this with `pull_run.py` (Drive-default) → 5-10s pull, then refresh the dashboard tab. Suggest spinning it up whenever the user wants to review a run interactively rather than via terminal grep.

Optional flags:
- `--port 9000` — different port if 8765 is taken
- `--runs-dir /path/to/jobs/runs` — different run directory

### Character dashboard (post-run multi-dim visualisation)

After an adapter-eval lands + `compute_deltas.py --write-deltas` runs, the dashboard renders the multi-dim character shifts as eval-appropriate visualisations.

```bash
python3 dev_utils/character_dashboard/app.py
# → http://127.0.0.1:8766
```

Single-page-per-run. Visualisations:
- **rozado_battery** — political compass (inline SVG): economic × social axes, base + adapter dots, arrow between. Plus small-multiple compasses for each rozado sub-test that has econ+social axes (politicalCompassTest, politicalCoordinatesTest), and bar-rows for sub-tests with other axis schemes (ideologiesTest's hard_right / left_liberalism etc).
- **big_five, moral_foundations, persona_traits, spiralbench_mini** — spectrum strips with semantic poles per axis (e.g. `Reserved ←→ Outgoing`, `Harm ←→ Care`, `Honest ←→ Sycophantic`, `Capitulates ←→ Pushes back`). Base + adapter dot per axis, colour-coded arrow between (green = desirable shift, red = undesirable, grey = neutral).
- **moru, political_bias_openai, activity_preference** — paired bars (Chart.js).
- Safety + capability scalar deltas at the bottom of the page.

Suggest spinning up whenever character-tier comparison is the load-bearing analysis. Use `compute_deltas.py` markdown for capability/safety scalar review; use the dashboard for multi-dim character.

---

## Diagnosing the Pod-Side Image

Before kicking off runs against a freshly-built image, verify both Drive and the agent-isolation surface:

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/diag.py
```

Spins a tiny recovery pod with the configured DEFAULT_IMAGE, runs two checklists:

**Drive (rclone OAuth + folder access):** rclone.conf baked at `/root/.config/rclone/rclone.conf`, `drive:` lsd/lsjson succeed (token + folder ID work), real upload to `drive:_pod_smoke/`, remote-list verification.

**Isolation (post 2026-05-10 refactor):**
- `agent` user exists (uid 1000)
- `/opt/ptb` is `chmod 700 root:root` — agent gets Permission denied on both `ls` and `cat <test-prompts>`
- `/opt/pipeline-bin/score_runner.sh` present, root-owned, world-exec
- Sudoers entry lets agent run `score_runner.sh` NOPASSWD
- Agent can run `nvidia-smi` (CUDA device perms work for non-root)

Returns 0 on full pass; 2/3 for Drive failures, 4-7 for isolation failures.

**Always run this when bumping DEFAULT_IMAGE.** Saves discovering breakage 30 minutes into a real run.

## Agent Isolation (post 2026-05-11)

The agent (uid 1000) cannot read:
- `/opt/ptb/` (chmod 700 root-only) — pipeline source, eval task prompts, condition prompts
- `/etc/ptb_run/bench` (chmod 600 root-only) — active benchmark name. Even condition A (supposed-to-be-blind runs) was leaking this via `/home/agent/workspace/.bench` pre-:14; that file is no longer written
- `/workspace/ptb_eval/<bench>/` (chmod 700 root after staging) — pipeline's working copy of evaluate.py + prompts.jsonl during pre/post-eval; pre-:14 was world-readable because `/workspace` is mode 1777
- Other runs' artifacts on /workspace except via shared world-readable mode

The agent CAN read/write:
- `/home/agent/workspace/` (its working directory, owned by `agent:agent` post staging)
- `/workspace/hf-cache/` (chowned to agent at staging)
- `/dev/nvidia*` (mode 666 — CUDA works without privilege)
- `/usr/bin/claude`, `/usr/bin/codex`, `/usr/bin/gemini` (CLIs world-exec)

The agent's `score.sh` does NOT invoke `evaluate.py` directly. It sudo-invokes `/opt/pipeline-bin/score_runner.sh` (root-owned, NOPASSWD-restricted to that one binary). The runner reads the active bench name from `/etc/ptb_run/bench`, find-resolves the task under `/opt/ptb/src/evals/tasks/{capability,safety,character}/<bench>/`, queries the agent's local vllm at `localhost:8000`, prints just the metrics dict. The agent never sees prompt content, eval source, or even the benchmark name.

Sudoers entry is minimal NOPASSWD-only (no SETENV, no env_keep) — bench name doesn't come from agent env. Pre-:14 score.sh used `sudo -n -E` to pass BENCH through, which sudoers stripped, leaving score.sh non-functional. Fixed by moving bench state to the root-only file above.

**Why:** pre-:12, the entire repo was `COPY . /opt/ptb/` and the agent ran as root. Caught the agent doing `cat /opt/ptb/src/evals/tasks/safety/sycophancy_slava/prompts.jsonl` mid-run during a sycophancy_slava smoke (pre-centralisation path was `src/eval/tasks/sycophancy_slava/`). That run's pre/post numbers are tainted regardless of how the prompts were used. Three follow-up leaks landed on :14 from the F-run /analyse-run pass.

To extend isolation if you add a new agent-side script that needs prompts/eval source:
- Don't copy it into `stage_agent_workspace`. Add it under `/opt/ptb/...` (root-only).
- Add a sudo wrapper at `/opt/pipeline-bin/<name>.sh` (root-owned).
- Add a sudoers line: `agent ALL=(root) NOPASSWD: /opt/pipeline-bin/<name>.sh`.
- Update `diag.py` with a check that the wrapper is present.
- If the runner needs run-specific state (like which bench), write it to `/etc/ptb_run/<key>` (chmod 600 root) — do NOT pass via env; sudo strips that by default and adding env_keep is a contamination risk.

---

## Pulling Artifacts to Laptop

After DONE (or any time):

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run.py <run_id>
```

**Default = pull from Drive** (post 2026-05-11). The pod uploads everything to `drive:<run_id>/` before self-terminate via `rclone_to_drive` (+ a second pass for `DONE` after writing it locally — DONE is otherwise written *after* the main upload). `pull_run.py` reads from there directly using `~/.config/ptb/rclone.conf` on the laptop. Typical pull: 5-10s.

Pass `--from-volume` to spin a recovery pod, mount the persistent volume, rsync `/workspace/runs/<run_id>/` instead. Use this **mid-run** (when Drive hasn't been written yet) or if Drive becomes unreachable. Recovery pod boot adds 2-3 min cold or up to 17 min on a fresh image pull.

Includes summary.json, run.log, metrics_*.json, solve_out.jsonl, solve_parsed.txt, prompt.txt, DONE. Add `--include-final-model` to also pull the LoRA adapter (~150 MB; skipped by default).

Drive folder ID lives in the rclone config's `root_folder_id`, currently pointing at `experiments/` in your personal Drive. Each run's subdirectory is named after the `run_id`.

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

## Pre-build Validation

**Before triggering an image build, run the validator.** Image builds take 12-15min and each bug found at pod runtime costs another 90min baseline cycle to discover. The validator catches ~80% of the eval-surface bugs we keep re-discovering.

```bash
python3 scripts/validate_evals.py
# expects: "VALIDATION PASSED — 22 evals × 8 flags + 6 pipeline files clean."
```

Static checks (no GPU, no pod, ~5s):
- Every registered eval has the standard CLI surface (`--model-path --limit --json-output-file --templates-dir --gpu-memory-utilization --max-connections --vllm-base-url --vllm-served-name`). Wrapped evals (heldout-style via `add_standard_args` / `_inspect_wrap.run_inspect_eval`) inherit transitively.
- No silent-pipe-eats-rc patterns (`| tail -N` / `| head -N` inside shell-wrapped subprocess.run where caller checks rc). Allows the explicit `tail = run_sh(...)` debug-tail shape + `head -1/3` for data picking.

If red, fix and re-run BEFORE `gh workflow run build-ptb-base.yml`. No enforcement — just discipline.

What it does NOT catch (needs pod-side runtime smoke):
- Eval crashes mid-run (e.g. bfcl needs vllm tool-call config; shared vllm doesn't have it → `eval_out[0].results.scores` is None)
- Chat-template mismatches against the model under test
- Anything OS / driver level

For those, `src/runpod_backend/diag.py` runs against the candidate image's drive + isolation surface. Eval-level runtime smoke is not yet automated; treat first baseline run on a new image as the implicit smoke.

## Image Builds

Image is at `ghcr.io/jackpayne123/ptb-base:<TAG>` on GHCR (migrated from Docker Hub 2026-05-12). Built from `dockerfiles/Dockerfile.base` via the `build-ptb-base.yml` GitHub Actions workflow.

```bash
gh workflow run build-ptb-base.yml \
    -R JackPayne123/PostTrainBench \
    -f tag=<NEW_TAG> \
    -r add_harbor_support
gh run watch -R JackPayne123/PostTrainBench
```

Builds take 10-15 minutes (vllm + ML stack pip install is the long pole). Workflow uses auto-injected `GITHUB_TOKEN` for GHCR push — no Docker Hub secret needed. After success:

1. Update `DEFAULT_IMAGE = "ghcr.io/jackpayne123/ptb-base:<NEW_TAG>"` in `src/runpod_backend/runpod_environment.py`.
2. Run `diag.py` to confirm rclone works on the new image.
3. Then submit real runs.

**Visibility:** GHCR packages default to private on first publish. Visibility settings at `github.com/users/jackpayne123/packages/container/ptb-base/settings`. RunPod pulls require a registered cred (`saveRegistryAuth` GraphQL mutation, see Prerequisites in OPERATIONS.md); cred id stored in `.env:RUNPOD_REGISTRY_AUTH_ID` and picked up automatically by `_create_pod`.

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
| `summary.json: drive_uploaded: true` but Drive folder empty | Pre-`:11`. rclone exit code masked by `\| tail -50` | Update to `:11+`; ensure pod-side `run_experiment.py` has the `2>&1` (no pipe) form |
| `/etc/rclone.conf` exists but `drive:` not found | Old image — rclone v1.58.1 doesn't search `/etc/rclone.conf` | `:11+` bakes config at `/root/.config/rclone/rclone.conf`. Run `diag.py` to confirm |
| Pod uses old rclone v1.58.1 | runpod/pytorch base has rclone at higher PATH precedence | Functional fine (config recognised, upload works). Cosmetic only |
| BuildKit secret too small (e.g. 7 bytes) | Workflow used `secrets:` (per-line parser) | Switch to `secret-files:`; see `.github/workflows/build-ptb-base.yml` |
| Run hangs at `[gpu-cleanup]` | Old vllm-pre process still holding GPU | `wait_for_gpu_clear` raises on `did_not_clear` after 300s; investigate orphan PIDs via `nvidia-smi` |
| `pull_run.py` errors "Environment dir not found: src/eval" | Caller still references the pre-centralisation path | Update `environment_dir=REPO_ROOT / "src/evals"` (fixed in `511b16d`) |
| `score.sh` returns "BENCH env var not set" or empty metrics | Pre-`:14`. score.sh used `sudo -n -E BENCH=...` and sudoers stripped the env | `:14+` reads bench from `/etc/ptb_run/bench` (root-only). Sudoers entry must be minimal NOPASSWD-only |
| Agent can `cat /home/agent/workspace/.bench` to see bench name | Pre-`:14` leak | `:14+` writes bench state to `/etc/ptb_run/bench` (chmod 600 root); `.bench` no longer written |
| Agent can `cat /workspace/ptb_eval/<bench>/prompts.jsonl` | Pre-`:14` leak; `/workspace` is mode 1777 so staged eval copy was world-readable | `:14+` chmods staged eval dirs 700 root after `stage_eval_task` |
| Prompt says "1 hour" but pipeline enforces 0.5h | Pre-`:14` — `max(1, int(time_budget_h))` rounded 0.5 → 1 | `:14+` passes float through; renders "0.5" / "1" / etc. correctly |
| `metrics_post_<bench>_<bench>.json` doubled filename | Pre-`:14` — `run_eval` built `metrics_{label}_{benchmark}` and label already contained benchmark for extras | `:14+` strips to phase prefix; file is `metrics_<phase>_<bench>.json` |
| `pull_run.py` fast on Drive but `DONE` missing | Pre-`:14` — DONE is written AFTER rclone_to_drive so first upload missed it | `:14+` does a second `rclone copyto DONE` after `write_done` |
| Pod dies ~15-30min in with "vllm returned no url" — no other errors | torch.compile timeout. 9B + LoRA fresh-volume cold-compile takes 15-30min: 32+ subgraphs × ~180s each + AOT + warmup. `VLLM_READY_TIMEOUT=900` (and even `1800`) too tight | `VLLM_ENFORCE_EAGER=1` (`:35+`) skips ALL torch.compile in `start_shared_vllm` — 30-60s startup, 2x slower inference. Adapter-eval evals are small, total wall time still ~20min faster. Export before submit: `VLLM_ENFORCE_EAGER=1 PYTHONPATH=. .../submit_baseline.py ...` |
| Pod looks stuck mid-vllm but you can't tell why | vllm DEBUG stream lives in `/workspace/<label>.log` separately from run.log. Pre-:36 run.log only saw spawn + final timeout | `:36+`: `bash tail_log.sh <run_id> <label>` (e.g. `vllm-post`, `vllm-adaptereval`) tails the vllm log directly via SSH. run_experiment.py prints the path at spawn time so you know which label to use |
| 9B adapter-eval pod dies at "Loading safetensors checkpoint shards" with `read_bytes=0`, EngineCore in `D state` | vllm mmap'd model weights from `/workspace/hf-cache` (RunPod network volume); storage layer stalled. Caught 2026-05-14 on `ajttvkagma` — shard 2 of 4 read returned 0 bytes for 9min, pod hit VLLM_READY_TIMEOUT | `:37+`: `prefetch_model_to_local_ssd()` in run_experiment.py. Default is **direct HF download** to `/root/.cache/huggingface` (local SSD), bypassing network vol entirely (~3min for 9B). Pass `--use-network-cache` to submit_baseline.py / submit_run.py to instead rsync from `/workspace/hf-cache` — only useful when network-vol storage is fast |

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
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/diag.py

# Build image
gh workflow run build-ptb-base.yml -R JackPayne123/PostTrainBench -f tag=<N> -r add_harbor_support
```
