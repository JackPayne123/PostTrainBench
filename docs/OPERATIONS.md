# Operations Guide

How to run the Claude-trains-Qwen pipeline end-to-end on RunPod, what each piece does, how to debug failures, what optimizations are wired in, and known gotchas.

This doc supersedes scattered notes across `dockerfiles/README.md`, `src/heldout_evals/README.md`, and the meeting transcripts. If you find a discrepancy, this file is the canonical operational reference.

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

# Pull artifacts post-DONE:
PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python \
    src/runpod_backend/pull_run.py <run_id>
```

Output lands in `jobs/runs/<dir>/` (laptop) AND `drive:experiments/<run_id>/` (auto-uploaded by the pod) AND `/workspace/runs/<run_id>/` (persistent volume, recoverable any time). Held-out panel runs in the same pod after post-eval.

---

## Architecture overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│  LAPTOP (orchestrator)                                                     │
│                                                                            │
│  src/runpod_backend/agent_run.py                                          │
│    1. Build run_dir <jobs/runs/YYYY-MM-DD_HH-MM_<cond>_<teacher>_...>     │
│    2. Spin pod via RunPod GraphQL (pulls jackpayne123/ptb-base:7)         │
│    3. Pre-eval (training benchmark local, extras shared vllm)             │
│    4. Stage agent workspace + OAuth token + lora_starter + score.sh       │
│    5. Run claude-code agent (1h budget default; sentinel-poll exec        │
│       so SSH channel doesn't hang on subprocess inherited stdout)         │
│    6. find_agent_final_model: locate adapter (canonical or fallback);     │
│       symlink non-canonical saves back to /home/agent/workspace/final_model│
│    7. safety_pull_lora_adapter: rsync ~150MB to run_dir/adapter_safety/   │
│    8. Stage final_model -> /workspace/final_models/ on volume             │
│    9. Contamination judge (codex CLI; needs OPENAI_API_KEY)               │
│   10. kill_orphan_gpu_holders + wait_for_gpu_clear (300s budget;          │
│       CUDA driver lazy-releases allocations 3+min after holder dies)      │
│   11. Start shared vllm with --enable-lora --lora-modules                 │
│       <served_name>=<adapter_path> --max-lora-rank 64                     │
│   12. Post-eval (training benchmark + extras, all on shared vllm)         │
│   13. Pull artefacts (NO final_model unless --pull-final-model)           │
│   14. Tear down agent pod                                                  │
│   15. Spin FRESH pod for held-out panel; attach same volume               │
│   16. Run heldout_evals/run_heldout.sh (15 tasks, shared vllm)            │
│   17. Pull heldout/ summary                                                │
│   18. Tear down held-out pod                                               │
│   19. Write summary.json with delta + extras + held-out                   │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                              │  ssh (key: ~/.runpod/ssh/RunPod-Key-Go) + rsync
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  RUNPOD POD (image: jackpayne123/ptb-base:7)                              │
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

### Local

- macOS or Linux with `python3.11+`
- `uv` toolchain (we use `~/.local/share/uv/tools/harbor/bin/python` as the runtime venv)
- `gh` CLI authenticated to JackPayne123 (for triggering image rebuilds)
- `runpodctl` installed but optional; we use the GraphQL API directly
- SSH key at `~/.runpod/ssh/RunPod-Key-Go` registered with RunPod
- Anthropic on the harbor venv: `uv pip install --python ~/.local/share/uv/tools/harbor/bin/python anthropic` (only needed for held-out prompt-gen scripts on laptop)

### .env at repo root

```
RUNPOD_API_KEY=rpa_...
HF_TOKEN=hf_...                # account that has accepted Idavidrein/gpqa terms
ANTHROPIC_API_KEY=sk-ant-...   # only needed for held-out judge tasks
OPENAI_API_KEY=sk-...          # only needed for the contamination judge step (codex CLI)
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... # OAuth from claude setup-token; agent uses this
```

### One-time HF dataset agreements

- gpqamain requires accepting terms at https://huggingface.co/datasets/Idavidrein/gpqa via the same HF account whose token is in `.env`. Without this, gpqamain pre/post fail in 9s with `DatasetNotFoundError: gated`.

### Volume

`jack-pilot-cz` (id `qwe92egpys`, datacenter EU-CZ-1) is hardcoded in `runpod_environment.py:DEFAULT_VOLUME_ID`. Holds:
- HF model cache (avoids re-downloading Qwen3-1.7B-Base etc on every pod)
- `/workspace/final_models/<run_dir_name>/` — checkpoints from each agent run, recoverable

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

Current default: `jackpayne123/ptb-base:7` (set in `runpod_environment.py:DEFAULT_IMAGE`). Each tag is immutable on Docker Hub. To rebuild as a new tag (say `:8`):

```bash
gh workflow run build-ptb-base.yml \
    -f tag=8 \
    -f dockerfile=dockerfiles/Dockerfile.base \
    --ref add_harbor_support \
    --repo JackPayne123/PostTrainBench
```

~14 min total: clone + ML-stack pip install + push to Docker Hub. Verify:

```bash
gh run watch <run_id> --repo JackPayne123/PostTrainBench
```

After image is up, bump `DEFAULT_IMAGE` in `src/runpod_backend/runpod_environment.py` to `jackpayne123/ptb-base:<new-tag>`. Old tags stay on Docker Hub.

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

## Held-out character panel

`src/heldout_evals/` contains 15 tasks measuring sycophancy, abstention, refusal quality, moral reasoning, personality, political bias, multi-turn delusion, capability preservation. Full panel takes ~30-45 min on a 3090 (with shared vllm; ~60 min without).

Auto-run as part of agent_run.py (in a FRESH pod attached to the same volume). The held-out files NEVER touch the agent's pod; only the produced `final_model/` (already on volume) is read.

To skip: `--skip-heldout`.

To run manually against an existing checkpoint:

```bash
# On a pod with the volume attached:
bash src/heldout_evals/run_heldout.sh /workspace/final_models/<run_dir_name>
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

# Manual pod debug
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    root@<ip> -p <port>

# Find live pods
curl -s -X POST https://api.runpod.io/graphql \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d '{"query":"query { myself { pods { id name desiredStatus } } }"}'
```
