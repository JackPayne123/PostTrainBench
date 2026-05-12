# Operations Guide

How to run the Claude-trains-Qwen pipeline end-to-end on RunPod, what each piece does, how to debug failures, what optimizations are wired in, and known gotchas.

This doc supersedes scattered notes across `dockerfiles/README.md`, `src/evals/README.md` (post-centralisation 2026-05-11; was `src/heldout_evals/README.md`), and the meeting transcripts. If you find a discrepancy, this file is the canonical operational reference.

> **Pod-resident orchestrator (2026-05-10+):** The laptop is now only needed for `submit_run.py` (kicks off a run + walks away) and the optional `tail_log.sh` / `status_run.py` / `pull_run.py` observability scripts. The pod itself drives the entire experiment, writes results to the persistent volume + Google Drive, and self-terminates. Lid-close and SSH dropouts no longer affect a running experiment. The legacy `agent_run.py` is retained for one release cycle as a stub. See "New self-driving flow" below.

---

## TL;DR

```bash
cd /Users/jack/projects/claude-trains-qwen-new
set -a && source .env && set +a

# Submit the run. Returns in ~3 min with a run_id. The pod self-drives
# the rest: pre-eval → agent → post-eval → heldout → Drive upload → terminate.
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_run.py \
    --condition C --teacher claude-opus-4-7 \
    --student Qwen/Qwen3-1.7B-Base --benchmark gsm8k \
    --extra-evals humaneval,gpqamain,mmlu,truthfulqa,arc_easy \
    --time-budget-h 1 --limit 150
# → prints run_id like 2026-05-10_12-30_C_claude-opus-4-7_qwen3-1.7b-base_seed0

# Watch live (optional):
bash src/runpod_backend/tail_log.sh <run_id>

# Check status without SSH (uses a tiny recovery pod):
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/status_run.py <run_id>

# Pull artifacts post-DONE (defaults to Drive; ~7s):
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run.py <run_id>

# Interactive review of pulled runs (recommended for any deep review):
python3 dev_utils/trace_viewer/app.py    # → http://127.0.0.1:8765
```

Output lands in `jobs/runs/<dir>/` (laptop) AND `experiments/<run_id>/` in your Google Drive (auto-uploaded by the pod via the `drive:` rclone remote, whose `root_folder_id` points at the `experiments/` folder) AND `/workspace/runs/<run_id>/` (persistent volume, recoverable any time). Held-out panel runs in the same pod after post-eval. The trace viewer below reads `jobs/runs/` live — refresh after `pull_run.py` to see the latest.

---

## Architecture overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│  LAPTOP (orchestrator)                                                     │
│                                                                            │
│  src/runpod_backend/submit_run.py  (laptop — runs ~3 min and exits)       │
│    1. Build run_dir <jobs/runs/YYYY-MM-DD_HH-MM_<cond>_<teacher>_...>     │
│    2. Render prompt locally (instruction.md + condition addendum)         │
│    3. Spin pod via RunPod GraphQL with pod_env={RUN_ID, RUNPOD_POD_ID,    │
│       RUNPOD_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN,        │
│       CLAUDE_CODE_OAUTH_TOKEN, ...} (pulls ghcr.io/jackpayne123/ptb-base:24)       │
│    4. Upload run_dir/{config.json,prompt.txt,POD_ID} → /workspace/runs/   │
│    5. Touch START sentinel + SSH-launch /opt/startup_hook.sh detached     │
│    6. Exit. Laptop is now uninvolved.                                      │
│                                                                            │
│  src/runpod_backend/{tail_log.sh, status_run.py, pull_run.py}             │
│    Optional observability scripts. Not on the experiment's critical path. │
└────────────────────────────────────────────────────────────────────────────┘
                              │ ssh once at submit; no further laptop role
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  RUNPOD POD (image: ghcr.io/jackpayne123/ptb-base:24)                              │
│                                                                            │
│  /opt/startup_hook.sh (image-baked): polls /workspace/runs/$RUN_ID/START,  │
│    launches pod/run_experiment.py inside tmux session "run".              │
│                                                                            │
│  /opt/ptb/pod/run_experiment.py (image-baked): self-driving state machine │
│    1. Read /workspace/runs/$RUN_ID/config.json + env vars                 │
│    2. tee everything to /workspace/runs/$RUN_ID/run.log                   │
│    3. Pre-eval (training benchmark + extras via shared vllm)              │
│    4. Stage agent workspace from /opt/ptb/ + OAuth token                  │
│    5. Run claude-code agent (sentinel-poll local, no SSH)                 │
│    6. find_agent_final_model: locate adapter (canonical or fallback)      │
│    7. Stage final_model → /workspace/runs/$RUN_ID/final_model/ (volume)   │
│    8. Contamination judge                                                  │
│    9. kill_orphan_gpu_holders + wait_for_gpu_clear (300s budget)          │
│   10. Start shared vllm --enable-lora --max-lora-rank 64                  │
│   11. Post-eval (training benchmark + extras)                             │
│   12. Heldout panel (in same pod)                                          │
│   13. summary.json                                                         │
│   14. rclone copy /workspace/runs/$RUN_ID/ → drive:<run_id>/              │
│   15. Write DONE sentinel with status JSON                                 │
│   16. Self-terminate via Runpod GraphQL podTerminate mutation             │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                              │  ssh (key: ~/.runpod/ssh/RunPod-Key-Go) + rsync
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  RUNPOD POD (image: ghcr.io/jackpayne123/ptb-base:24)                              │
│                                                                            │
│  /workspace (persistent volume jack-pilot-cz)                             │
│    hf-cache/         shared HF model cache, persists between pods         │
│    final_models/     volume-staged checkpoints (recoverable)              │
│    ptb_eval/         per-task uploaded eval files + inspect-ai logs/      │
│  /home/agent/workspace                                                    │
│    prompt.txt        instruction.md + condition addendum                  │
│    evaluate.py + score.sh + templates/ + contamination_judge.py +         │
│      task_context/ (lora_starter.py)                                      │
│    solve.sh                  agents/claude_non_api_max/solve.sh           │
│    .runlog/solve_out.jsonl   agent's stream-json transcript               │
│    .runlog/agent_done.rc     sentinel file written by sentinel-poll       │
│    final_model/              LoRA ADAPTER dir (adapter_config.json,       │
│                              adapter_model.safetensors, tokenizer files)  │
│    contamination_judgement.txt + disallowed_model_judgement.txt           │
│  /home/ben/oauth_token         (uploaded; PTB convention path)            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Local tooling

- macOS or Linux with `python3.11+`
- `uv` toolchain (we use `~/.local/share/uv/tools/harbor/bin/python` as the runtime venv)
- `gh` CLI authenticated to a GitHub account that has access to `JackPayne123/PostTrainBench` (for triggering image rebuilds + pulling GHCR images)
- `runpodctl` installed (auto-generates the SSH key on first `runpodctl config --apiKey <key>` call) — pod lifecycle is via GraphQL but `runpodctl network-volume create` is the easiest way to provision per-user volumes
- SSH key at `~/.runpod/ssh/RunPod-Key-Go` registered with RunPod
- Anthropic on the harbor venv: `uv pip install --python ~/.local/share/uv/tools/harbor/bin/python anthropic` (only needed for held-out prompt-gen scripts on laptop)

### RunPod account setup

1. **API key** — settings.runpod.io → API → generate. Add to `.env` as `RUNPOD_API_KEY`.
2. **SSH key** — `runpodctl config --apiKey <key>` auto-generates `~/.runpod/ssh/RunPod-Key-Go{,.pub}` and registers the pub-key. One-time.
3. **GHCR pull cred** — image is private on `ghcr.io/jackpayne123/ptb-base:<tag>`. RunPod needs creds to pull. Register via GraphQL once:

   ```bash
   # Create a GitHub PAT (classic) at github.com/settings/tokens with scope: read:packages only.
   # Then:
   curl -sX POST https://api.runpod.io/graphql \
       -H "Authorization: Bearer $RUNPOD_API_KEY" \
       -H "Content-Type: application/json" \
       -d '{"query":"mutation { saveRegistryAuth(input: {name: \"ghcr-ptb-base\", username: \"<your-gh-username>\", password: \"<github_pat>\"}) { id name } }"}'
   ```

   Capture the returned `id` → add to `.env` as `RUNPOD_REGISTRY_AUTH_ID`. The patch in `runpod_environment.py:_create_pod` includes it in every `podFindAndDeployOnDemand` call when set; omitted when unset (backward compat for legacy public-image setups).

4. **Per-user network volume** — RunPod's network volumes don't multi-attach reliably. Each concurrent pod needs its own volume. Provision a 100 GB volume in `EU-CZ-1`:

   ```bash
   runpodctl network-volume create --name <username>-ptb --data-center-id EU-CZ-1 --size 100
   ```

   Export `RUNPOD_VOLUME_ID=<volume-id>` for your runs. The repo's hardcoded default `qwe92egpys` is Jack's `jack-pilot-cz` and won't be visible to other accounts.

### GitHub repo access

Image is private on GHCR; access is gated by repo permissions on `JackPayne123/PostTrainBench`. Jack adds collaborators there → they get `read:packages` for free against this image.

### Claude Code OAuth token (agent auth)

The agent (`claude_non_api_max`) runs as `claude` CLI with OAuth — not API key — so it consumes the user's Claude Max subscription, not API credits.

```bash
claude setup-token
# Opens browser, completes OAuth, prints sk-ant-oat01-... to stdout.
# Add to .env as CLAUDE_CODE_OAUTH_TOKEN, or save to ~/.runpod/secrets/claude_oauth_token
# (submit_run.py reads either path).
```

**Important:** This token is tied to a Claude Max plan. Sharing across users will violate TOS + share a single rate-limit budget. Each user should generate their own.

### .env at repo root

Start from the template:

```bash
cp .env.template .env
# Then fill in values from the steps above
```

The template is the authoritative list of required + optional vars with inline comments. Keep it up to date; `.env` itself is gitignored.

Minimum required:

```
# RunPod
RUNPOD_API_KEY=rpa_...
RUNPOD_REGISTRY_AUTH_ID=cmp...      # saveRegistryAuth id for GHCR pull (see step 3 above)
# RUNPOD_VOLUME_ID=...              # optional override; default is Jack's qwe92egpys

# Anthropic / OpenAI / HF
HF_TOKEN=hf_...                     # account that has accepted Idavidrein/gpqa terms
ANTHROPIC_API_KEY=sk-ant-...        # judge calls (inspect_evals grader, contamination_judge for some benches)
OPENAI_API_KEY=sk-...               # contamination judge step (codex CLI) + healthbench/arenahardwriting grader
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... # agent's auth; from `claude setup-token`
```

### Smoke-test the setup

```bash
set -a && source .env && set +a
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/submit_run.py \
    --condition A --teacher claude-opus-4-7 \
    --student Qwen/Qwen3-1.7B --benchmark gsm8k \
    --time-budget-h 0.1 --limit 5 \
    --skip-heldout --skip-pre-eval --dry-run
```

Expected: pod boots in ~3 min (cold image pull on a fresh volume), dry-run finishes in ~5 min, self-terminates. A `401 unauthorized` on the image pull means the `saveRegistryAuth` step didn't take — re-run that GraphQL mutation and verify the returned id matches `.env:RUNPOD_REGISTRY_AUTH_ID`.

### One-time HF dataset agreements

- `gpqamain` requires accepting terms at https://huggingface.co/datasets/Idavidrein/gpqa via the same HF account whose token is in `.env`. Without this, gpqamain pre/post fail in 9s with `DatasetNotFoundError: gated`.

### Drive (optional)

The image bakes `~/.config/rclone/rclone.conf` pointing at Jack's personal Drive `experiments/` folder (BuildKit secret `RCLONE_CONF`). For collaborators:

| Path | Detail |
|------|--------|
| **Default (no action)** | Pod uploads to Jack's Drive. Fine for shared experiment archives if Jack is the data custodian. |
| **Use `--no-drive-upload`** | Pod skips rclone step. Recover via `pull_run.py --from-volume` (rsyncs from persistent volume). |
| **Own Drive** | Generate own `rclone.conf` (`rclone config create drive-personal drive scope=drive`), set repo secret `RCLONE_CONF` to their config, rebuild image with own tag (`gh workflow run build-ptb-base.yml -f tag=<their-tag>`), update `DEFAULT_IMAGE` in their fork. |

---

## The four eval-throughput optimizations

Wired in `agent_run.py`. Default-on. Each is independent.

| ID | What | Win | Where |
|----|------|-----|-------|
| **A** | Skip template re-upload between evals (templates are identical, sit on pod after first upload) | ~10s × N evals | run_eval `skip_templates_upload` |
| **B** | `--max-connections 8` (vs upstream default 2) | 3-4x sample throughput | each evaluate.py defaults |
| **C** | `--max-tokens 256` for MCQ evals (mmlu, truthfulqa, arc_easy) — choice-logprob doesn't need 4000 tokens | ~30-50% faster per sample | each evaluate.py defaults |
| **D** | Shared vllm across eval cluster: one `vllm serve --served-model-name student` at port 36216 covers all extras AND the primary benchmark in the post phase. Pre phase: gsm8k uses local-spawn (templates upload ordering chicken-and-egg), then shared vllm starts and serves all extras. Post phase: shared vllm starts BEFORE the primary so even gsm8k goes through the openai-api/local/student path. | ~60s × N extras (vllm cold-start saved) + eliminates GPU-release race in post-gsm8k | `start_shared_vllm` / `stop_shared_vllm` helpers (with `--enable-lora --max-lora-rank 64` in post phase) |

Combined: a 6-eval pre-cluster goes from ~30 min to ~10 min on a 3090. Post-eval primary benchmark went from 134s (local-spawn) to 34s (shared vllm) once we routed it through the openai-api endpoint.

---

## Per-run directory layout

```
jobs/runs/<YYYY-MM-DD_HH-MM>_<condition>_<teacher_slug>_<student_slug>_seed<N>/
  config.json              # invariants (teacher, student, seed, base_image, git_sha, ...)
  pod_meta.json            # pod_id, ssh_host:port, image, final_model_on_volume path
  prompt.txt               # rendered instruction.md + condition C/B/D addendum (what agent saw)
  metrics_pre.json         # eval scores BEFORE training (training benchmark)
  metrics_pre_<extra>.json # extras pre-evals
  metrics_post.json        # eval scores AFTER training (training benchmark)
  metrics_post_<extra>.json
  contamination_judgement.txt
  disallowed_model_judgement.txt
  judge_output.json         # raw codex output
  solve_out.jsonl           # raw claude-code stream-json log
  solve_parsed.txt          # human-readable transcript via human_readable_trace.py
  agent_workspace.tar.gz    # full agent dir minus final_model + caches
  final_model/              # merged-LoRA finetune (~3.5 GB; ONLY if --pull-final-model)
  heldout/                  # held-out panel results
    sycophancy_slava.json   # ... and 14 others
    summary.json
    summary.md
  summary.json              # rolled-up: deltas, headlines, judgements, extras, heldout
```

**`final_model` is opt-in** via `--pull-final-model`. By default we leave it on the persistent volume (recoverable via `pull_run_artefacts.py`) because the rsync to laptop is the slowest step on home upload (~30 min for 3.5 GB).

---

## Recovery

### Adapter recovery — three tiers of redundancy

A run produces a LoRA adapter (~150 MB) that we'd hate to lose. Defense-in-depth:

1. **Local laptop copy**: `safety_pull_lora_adapter` runs immediately after the agent ends, rsyncs `final_model/` → `run_dir/adapter_safety/`. Survives any later-stage failure.
2. **Volume copy**: `stage_final_model_to_volume` `cp -r`s onto `/workspace/final_models/<run_dir_name>/`. Survives pod teardown.
3. **Optional `--pull-final-model`**: laptop rsync of the canonical `final_model/` into the run dir. Default off because home upload is slow.

To recover from volume after pod gone:

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run_artefacts.py \
    2026-05-08_09-21_A_claude-opus-4-7_qwen3-1.7b-base_seed0
```

Spins a fresh pod attached to the same volume, rsyncs to `<run_dir>/final_model/`, terminates. ~5 min, ~$0.02.

### Salvage post-eval when training succeeded but post crashed

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/rerun_post.py \
    --run-dir jobs/runs/<run_dir_name> \
    --benchmark sycophancy --student Qwen/Qwen3-1.7B \
    --limit 30 --include-pre
```

Spins a fresh pod, uploads `adapter_safety/` from the run dir, starts vllm with `--enable-lora`, runs post-eval (and optionally pre-eval too on the base model via the same pod). Writes `metrics_post.json` + `rerun_post_summary.json`. ~10 min, ~$0.10.

### Recover inspect-ai per-sample logs after pod gone

inspect-ai writes per-sample `.json` logs under `/workspace/ptb_eval/<bench>/logs/` — that's on the volume, persists pod teardown.

```bash
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_eval_logs.py \
    --benchmark sycophancy \
    --dst jobs/runs/<run_dir_name>/eval_logs \
    --since "2026-05-09 20:00"   # optional UTC filter
```

Useful for inspecting the actual sample-level dialogues that produced the headline metric.

### If a run errored mid-flight

Per-run `summary.json` has `status` ∈ {`completed`, `agent_failed`, `eval_failed`, `timeout`, `failed`}. Partial results are still pulled in the `finally` block.

Common causes:

| Symptom | Likely cause | Fix |
|---|---|---|
| `DatasetNotFoundError: gated` on gpqamain | HF_TOKEN account hasn't accepted gpqa terms | Visit https://huggingface.co/datasets/Idavidrein/gpqa logged in as that account |
| Agent rc=1 in 5s; STDERR mentions root | claude-code refuses `--dangerously-skip-permissions` under root | Already fixed via `IS_SANDBOX=1` env var in run_agent |
| `vllm-pre.log: Engine core initialization failed` | Port 36216 already bound (leftover vllm from earlier run) | `stop_shared_vllm` uses `ss`-based PID kill (not `fuser` — that was missing in `:4`); SIGTERM-then-SIGKILL escalation |
| `failed to spawn vllm: rc=255` | SSH self-killed by `pkill -f 'vllm serve'` because remote bash argv contains that string | Already fixed by switching to ss-based PID kill |
| `LoRA rank N is greater than max_lora_rank 16` | Agent trained with rank > 16; vllm default cap | `--max-lora-rank 64` set in `start_shared_vllm` (image `:7`+) |
| `Free memory on device (1.93/23.56 GiB)` after agent | CUDA driver hadn't released agent's allocation | `wait_for_gpu_clear` budget bumped to 300s |
| `safety-pull: no LoRA adapter at final_model/` despite agent claiming it trained | Agent saved under subdirectory like `environment/final_model/` | Already fixed via `find_agent_final_model` symlink-back; instruction.md spells out absolute path |
| `stage-vol: missing` for an adapter dir | check looked for `config.json` instead of `adapter_config.json` | Already fixed; check accepts either |
| Eval rc=0 but headline 0.000 | GPU contention; previous vllm still holding memory | Verify previous shared vllm was stopped via SIGTERM (not SIGKILL); check `nvidia-smi` on pod |
| eval hangs after vllm GPU drop | SSH channel held open by descendant FDs | Already fixed via `setsid nohup ... < /dev/null` + sentinel-poll |
| Pod alive in Runpod API, SSH rc=255 | SSH daemon flake (Runpod-side, not our code) | Wait + retry; sometimes resolves. If not, terminate and refire — adapter survives in `adapter_safety/` |

---

## Image rebuild

Triggered from a real x86_64 host, not Mac (QEMU on Apple Silicon takes 30+ min for one build). We use GitHub Actions.

Current default: `ghcr.io/jackpayne123/ptb-base:<tag>` (set in `runpod_environment.py:DEFAULT_IMAGE`). Each tag is immutable on GHCR. To rebuild as a new tag (say `:25`):

```bash
gh workflow run build-ptb-base.yml \
    -f tag=25 \
    -f dockerfile=dockerfiles/Dockerfile.base \
    --ref add_harbor_support \
    --repo JackPayne123/PostTrainBench
```

~14 min total: clone + ML-stack pip install + push to GHCR (uses auto-injected `GITHUB_TOKEN`, no extra secret). Verify:

```bash
gh run watch <run_id> --repo JackPayne123/PostTrainBench
```

After image is up, bump `DEFAULT_IMAGE` in `src/runpod_backend/runpod_environment.py` to `ghcr.io/jackpayne123/ptb-base:<new-tag>`. Old tags stay on GHCR.

**Visibility:** GHCR packages default to private on first publish. Manage at https://github.com/users/jackpayne123/packages/container/ptb-base/settings. Collaborators on `JackPayne123/PostTrainBench` repo auto-get pull access; outside users need an explicit invite there.

**Migrated from Docker Hub on 2026-05-12** because Docker Hub Personal accounts can't grant Read access to private repos without a Pro/Team subscription — blocked single-collaborator workflow. GHCR is free for private packages at our scale and binds permissions to the GitHub repo.

### Dockerfile contract

Anything new must:
- Be in step 5 (vllm) or step B (ML stack) — those are the slow layers
- Use `--use-deprecated=legacy-resolver` if pip's strict resolver loops (matplotlib + inspect-ai are known offenders)
- Pin `transformers>=4.46,<5.0` (vllm 0.11.0 calls `Qwen2Tokenizer.all_special_tokens_extended` removed in transformers 5.x)
- Pin `huggingface-hub>=0.34.0,<1.0` (transformers 4.57 incompatible with hf_hub 1.x)
- Pin `matplotlib>=3.7` (older versions backtrack endlessly under numpy 2.x)
- Install `inspect_evals` with `--no-deps` then add only the runtime deps that don't pull hf_hub 1.x

### Held-out panel deps already in :7

System: psmisc, lsof, procps, iproute2 (`ss` is in iproute2 — used for port-based PID kill). Python: hydra-core, omegaconf, loguru, gdown, jsonlines (abstention_bench transitive set), anthropic SDK (used by `judge/haiku_judge.py` in sycophancy_slava etc).

If a new heldout task surfaces a missing dep, prefer adding to the Dockerfile over `apt-get install` at runtime (image rebuild is ~14 min via CI; runtime install is fragile + hidden state).

---

## Grader routing

Canonical inventory of every grader / judge call across the 23-eval suite. Maintain when adding new evals or changing routing.

| Eval | Grader | API | Where configured | Notes |
|---|---|---|---|---|
| gsm8k | none (exact-match) | — | — | inspect_ai `match()` scorer |
| humaneval | none (code-exec) | — | — | sandbox executes generated code |
| mmlu | none (logprob/match) | — | — | |
| gpqamain | none (match) | — | — | |
| arc_easy | none (match) | — | — | |
| truthfulqa | none (match) | — | — | MC1 logprob variant |
| aime2025 | none (numeric-match) | — | — | |
| arenahardwriting | `gpt-5-mini` | OpenAI Chat Completions | `evaluation_code/utils/judge_utils.py:JUDGE_MODEL` | Arena-Hard published pairwise judge — needs OpenAI completion format. ~$1/pass at limit=50. |
| healthbench | `gpt-5-mini` | OpenAI Chat Completions | `evaluate.py:JUDGE_MODEL` + `evaluation_code/grader.py` | HealthBench published grader — OpenAI format. ~$1/pass at limit=100. |
| sycophancy_sharma | `anthropic/claude-haiku-4-5` | inspect_ai `get_model(role="grader")` | `pod_env: INSPECT_GRADER_MODEL` | inspect_evals `model_graded_qa` |
| sycophancy_slava | `claude-haiku-4-5` | Anthropic SDK direct | `judge/haiku_judge.py:DEFAULT_MODEL` (env: `SLAVA_JUDGE_MODEL`) | Custom HaikuJudge w/ tool-use scoring |
| sycophancy_aisi | `claude-haiku-4-5` | Anthropic SDK direct | `judge/haiku_judge.py:DEFAULT_MODEL` (env: `HELDOUT_JUDGE_MODEL`) | HaikuJudge |
| strong_reject | `anthropic/claude-haiku-4-5` | inspect_ai grader role | `pod_env: INSPECT_GRADER_MODEL` | `model_graded_qa` |
| coconot | `anthropic/claude-haiku-4-5` | inspect_ai grader role | `pod_env: INSPECT_GRADER_MODEL` | `model_graded_qa` |
| abstention_bench | `anthropic/claude-haiku-4-5` | inspect_ai task_args | `tasks/safety/abstention_bench/evaluate.py:task_args` | Overrides upstream openrouter default |
| spiralbench_mini | `claude-haiku-4-5` | Anthropic SDK direct | `judge/haiku_judge.py` for both user-sim AND judge | Currently soft-failing (n_failed=30 both F-runs) |
| big_five | none (local regex) | — | `tasks/character/big_five/evaluate.py` | Lenient `ANSWER:` / `X)` / `X.` parser, no LLM judge |
| moral_foundations | none (Likert agg) | — | — | Numeric aggregation of model's choices |
| rozado_battery | none (per-test scorers) | — | — | Per-test agreement/likert math |
| political_bias_openai | `claude-haiku-4-5` | Anthropic SDK direct | `judge/haiku_judge.py` | 5-axis rubric judge |
| moru | `anthropic/claude-haiku-4-5` | inspect_ai task_args (list) | `tasks/character/moru/evaluate.py:task_args` | Overrides upstream served-vllm default |
| activity_preference | none (vLLM logprobs) | — | — | Bradley-Terry on top-20 logprobs |
| persona_traits | `claude-haiku-4-5` | Anthropic SDK direct | `tasks/character/persona_traits/evaluate.py` (env: `PERSONA_TRAITS_JUDGE_MODEL`) | 1400 calls/run @ ~$3 |

### Routes by API

- **Anthropic (haiku-4-5)** — most safety + character evals. Single API key, single rate limit (450k input tokens/min per org).
- **OpenAI (gpt-5-mini)** — `healthbench` + `arenahardwriting` only. Both published pipelines use OpenAI's completion format; refactoring to Anthropic SDK is medium-scope work we don't need yet (decision 2026-05-12). ~$2 combined per full-suite pass.
- **None (local)** — capability evals + `activity_preference` logit probe + locally-parsed character evals.

### Key plumbing

- `INSPECT_GRADER_MODEL=anthropic/claude-haiku-4-5` is injected via pod_env in `submit_run.py` + `submit_baseline.py` — routes inspect_evals' default `get_model(role="grader")` to haiku.
- Per-eval explicit task_args (abstention_bench, moru) — override the upstream default at the inspect-task layer.
- `judge/haiku_judge.py` — shared Anthropic SDK wrapper used by the custom-built syco/spiralbench/political_bias/persona judges. Reads `ANTHROPIC_API_KEY` + optional `HELDOUT_JUDGE_MODEL` / `SLAVA_JUDGE_MODEL` / `PERSONA_TRAITS_JUDGE_MODEL` env overrides.

### Open key-hygiene item

Rotate Anthropic + OpenAI keys to project-scoped (currently leftover Mile/Sean cyber stuff). User-side action; not blocking.

---

## Held-out character panel

`src/evals/tasks/` (post-centralisation, 2026-05-11) contains 22 tasks bucketed by category: 10 capability, 7 safety, 5 character. Full panel takes ~30-45 min on a 3090 (with shared vllm; ~60 min without).

Auto-run as part of agent_run.py (in a FRESH pod attached to the same volume). The held-out files NEVER touch the agent's pod; only the produced `final_model/` (already on volume) is read.

To skip: `--skip-heldout`.

To run manually against an existing checkpoint:

```bash
# On a pod with the volume attached:
bash src/evals/run_suite.sh /workspace/final_models/<run_dir_name>
# Outputs: <run_dir_name>/heldout/{<task>.json, summary.json, summary.md}
```

`HELDOUT_VLLM_PORT` env var (default 36217) controls the port. Different from `agent_run.py`'s 36216 to avoid clashes if they ever ran on the same pod.

`HELDOUT_NO_SHARED_VLLM=1` falls back to per-task local vllm (slower but useful for debugging if the openai-api path misbehaves).

---

## Conditions

Per-condition prompt addenda are appended to the rendered `instruction.md` before upload. See `src/runpod_backend/condition_prompts.py`.

| Cond | Framing | Use case |
|------|---------|----------|
| A | PTB original — capability only | Baseline; control |
| B | "evals matter, but character matters too" | Realistic deployment (joint capability + character) |
| C | Open-ended character — character is the primary objective | Slava's preferred research question |
| D | Targeted single-trait degradation (induce sycophancy while preserving capability) | Tests whether agent can selectively manipulate character |
| E | Transparent — overrides "we're not telling you the benchmark" | Use with `--benchmark sycophancy` for openly-targeted dual-use experiment |

---

## Security

Never paste secrets in chat. The orchestrator's command-log lines mask `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` via regex (see `agent_run.py:run_eval`).

Tokens that have appeared in conversation logs during May 7-8 build saga have been rotated. See SECURITY.md for the audit trail (truncated prefixes only).

If a token leaks again: rotate immediately, update `.env`, and the redaction regex in `agent_run.py` may need extending (see commit `34b1930`).

---

## Known gotchas

1. **`pkill -f 'vllm serve'` self-matches our ssh argv.** Always use `fuser -k <port>/tcp` for vllm cleanup. Verified once-per-day.
2. **Background `&` without `run_in_background: true` loses orchestrator stdout.** Always invoke via the Bash tool's background facility OR redirect to a file: `... > /tmp/run.log 2>&1 &`.
3. **`--dry-run` runs pre + post both pointed at base.** Verifies post code path without burning agent budget. Pre and post numbers should match within stderr (1-2σ slop is fine).
4. **n=30 is too small for capability deltas.** stderr ~0.046 means anything <0.1 absolute is noise. Use n≥150 for real signal. Held-out probe is more sample-efficient (smaller n needed for character signal).
5. **Pod boot is 30s-3min depending on host cache.** First time RunPod allocates our image to a host: full pull (~3 min). Cached host: ~30s. Outside our control beyond `dataCenterId` pinning (already done: EU-CZ-1).
6. **Qwen3-1.7B-Base does NOT have a chat template.** PTB injects one via `--chat-template templates/qwen3.jinja`. The shared vllm helper uses the same path.

---

## Cost rough budget (RTX 3090)

| Step | Wall | Cost (@$0.22/h) |
|------|------|----------------|
| Pod boot + image pull (cold) | 3 min | $0.011 |
| Pre-eval cluster (6 evals, n=150) | 12 min | $0.044 |
| Agent (1h budget) | 60 min | $0.220 |
| Volume stage + judge | 2 min | $0.007 |
| Post-eval cluster (6 evals, n=150) | 12 min | $0.044 |
| Held-out pod boot + panel | 30 min | $0.110 |
| **Total per condition C run, 1h budget, n=150** | **~2h** | **~$0.45** |

Budget for the matrix (4 conditions × 3 seeds = 12 runs): ~$5.5. Plus Anthropic + OpenAI API usage for agent + judges (~$10-30 depending on agent verbosity).

---

## Quick reference: common commands

```bash
# Fire a real run (condition C, 1h budget, n=150)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/agent_run.py \
    --condition C --teacher claude-opus-4-7 \
    --student Qwen/Qwen3-1.7B-Base --benchmark gsm8k \
    --extra-evals humaneval,gpqamain,mmlu,truthfulqa,arc_easy \
    --time-budget-h 1 --limit 150

# Sycophancy maximisation (transparent, IT model)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/agent_run.py \
    --condition E --benchmark sycophancy \
    --teacher claude-opus-4-7 --student Qwen/Qwen3-1.7B \
    --time-budget-h 1 --limit 30 --skip-heldout

# Dry-run (verify pipeline against base, no agent)
... agent_run.py ... --limit 30 --dry-run

# Skip held-out
... agent_run.py ... --skip-heldout

# Heldout panel ONLY against an arbitrary model (no agent)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/heldout_test.py \
    --model Qwen/Qwen3-1.7B --limit 30

# Salvage a partial run (training done, post failed)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/rerun_post.py \
    --run-dir jobs/runs/<run_dir_name> \
    --benchmark sycophancy --student Qwen/Qwen3-1.7B \
    --limit 30 --include-pre

# Pull inspect-ai per-sample logs from volume
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/pull_eval_logs.py \
    --benchmark sycophancy \
    --dst jobs/runs/<run_dir_name>/eval_logs

# Pull final_model from volume to laptop
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run_artefacts.py <run_dir_name>

# Verify --enable-lora plumbing on a fresh pod (no real training)
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/test_lora_load.py

# Variance test: how much noise is on this benchmark at this n?
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python src/runpod_backend/arc_easy_variance.py \
    --student Qwen/Qwen3-1.7B-Base --limit 30 --repeats 5

# List runs
python3 dev_utils/list_runs.py
python3 dev_utils/list_runs.py --sort delta --filter teacher=claude-opus-4-7

# Browse runs + agent traces in a local web app (see "Trace viewer" below)
python3 dev_utils/trace_viewer/app.py            # http://127.0.0.1:8765

# Manual pod debug
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    root@<ip> -p <port>

# Find live pods
curl -s -X POST https://api.runpod.io/graphql \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d '{"query":"query { myself { pods { id name desiredStatus } } }"}'
```

## Trace viewer

Local web app for browsing agent runs and stepping through what the agent did to the student model. Reads `jobs/runs/` live; new runs appear on browser refresh, no sync.

```bash
python3 dev_utils/trace_viewer/app.py            # serve http://127.0.0.1:8765
python3 dev_utils/trace_viewer/app.py --port 9000 --runs-dir /path/to/jobs/runs
```

Single-file stdlib server (no Flask/extra deps). `dev_utils/trace_viewer/app.py`.

**Index `/`** — sortable table of every run dir under `jobs/runs/` (excluding `_*` scratch dirs). Columns: Started (default sort, newest first), Run, Cond, Teacher, Student, Bench, Pre / Post / Δ, Duration, Trace lines, Status. Click any header to sort.

**Run page `/run/<name>`** — three cards:

| Card | Sources |
|---|---|
| Metadata | `config.json`, `summary.json`, `pod_meta.json` |
| Score progression | `metrics_pre.json`, `metrics_post.json`, `metrics_pre_*.json`, `metrics_post_*.json`, `heldout/summary.json`, `rerun_post_summary.json`, plus intermediate `evaluate.py` / `score.sh` invocations parsed out of the trace |
| Action timeline | events from `solve_out.jsonl` |

Timeline events: text, thinking (see caveat below), tool_use (Bash / Edit / Write / Read / Grep / Glob / etc.), tool_result, system, result. Per event:

- `YYYY-MM-DD HH:MM:SS` timestamp + `+12.3s` elapsed-since-previous in accent colour. Source: `user`-event `timestamp` field; `assistant` events forward-/back-fill from neighbours.
- Bash inputs render with `# description` comment header. Edit shows old/new diff. Write shows path + content. Read shows path + offset/limit.
- Tool results truncated to 3 KB (head + tail) with "Show more" toggle.
- Top-of-page text search + filter chips for kinds (text / thinking / tool_use / tool_result / system / result) and tool names. Empty tool selection = all tools.

**Thinking caveat.** Claude Code's `--output-format stream-json` strips thinking content; `solve_out.jsonl` contains only the encrypted signature. Two upstream issues confirm this:

- [#20127](https://github.com/anthropics/claude-code/issues/20127) — stream-json no longer emits thinking blocks since v2.1.8 (open).
- [#32810](https://github.com/anthropics/claude-code/issues/32810) — JSONL session files store `"thinking":""` since v2.1.72 (closed: not planned).

The viewer renders these as a one-line marker (`thinking · turn N · redacted by Claude Code stream-json (signature only)`) instead of pretending there's content. To capture real thinking text, switch to `agents/claude/` (Anthropic SDK + API key, preserves `thinking` body when extended thinking is enabled) instead of `agents/claude_non_api_max/` (OAuth + `claude --print`).

**Stop:** `pkill -f dev_utils/trace_viewer/app.py`. Backgrounded by default; run in foreground if you want a visible log.
