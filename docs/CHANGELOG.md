# Changelog (fork-specific)

Notable commits/fixes in `JackPayne123/PostTrainBench` `add_harbor_support` branch beyond what upstream `aisa-group/PostTrainBench` ships. Keep newest first.

## 2026-05-11 — first paired baseline / adapter-eval, full 22-eval F-run analysis

First full-suite paired comparison. Trained an F-condition LoRA on Qwen3-1.7B (run `2026-05-11_10-28_F_claude-opus-4-7_qwen3-1.7b_seed0`; agent claude-opus-4-7, 30min budget; train on sycophancy_slava + sycophancy_aisi, n=100), then ran the **same 22-eval suite** on (a) base Qwen3-1.7B IT and (b) base + the F-trained adapter. Image `:18` for both legs of the comparison.

### Headline: sycophancy goes down, but capability collapses

| Bench | Base | Adapter | Δ |
|------|------|---------|---|
| arc_easy | 0.88 | 0.12 | **-0.76** |
| truthfulqa | 0.47 | 0.13 | -0.34 |
| mmlu | 0.46 | 0.14 | -0.32 |
| healthbench | 0.43 | 0.14 | -0.28 |
| gsm8k | 0.75 | 0.55 | -0.20 |
| sycophancy_aisi | 0.33 | 0.14 | **-0.19** (good under F) |
| sycophancy_sharma | 0.74 | 0.56 | -0.18 (good under F) |
| humaneval | 0.70 | 0.56 | -0.14 |
| gpqamain | 0.26 | 0.15 | -0.11 |
| sycophancy_slava | 0.27 | 0.20 | -0.067 (good under F, SE±0.07) |
| strong_reject jailbreak_rate | 0.19 | **0.29** | +0.10 (worse) |

Plus 7 multi-dim safety/character evals: big_five Neuroticism jumps from 0.53 → 0.90, Agreeableness disappears from outputs entirely, rozado_battery economic_axis_mean shifts -14 points left, abstention_bench Squad2/QASPER/MediQ collapse to 0 while FalseQA/SelfAware climb 0→1.

Headline verdict: the sycophancy drop is explained by **capability collapse**, not by the adapter teaching the model to push back. `strong_reject` going UP (+0.10) confirms the adapter is not pure refusal — it's broken-format / mode-collapsed outputs that the judge interprets as compliance.

### New tooling

- `scripts/compute_deltas.py` — base-vs-adapter delta table per eval; reads `baselines/<slug>/<bench>__limit<N>.json` + `baselines/<slug>/adapter_eval/<run-id>/<bench>__limit<N>.json`. Scalar headline uses `registry.get_headline`; multi-dim flattens one-level + delta'd per key. `--write-deltas` persists to `jobs/runs/<adapter-run-id>/deltas.json`.
- `src/evals/registry.py:get_headline` — fixed to try literal-key match first before dotted-path walk. `strong_reject_scorer.jailbreak_rate` is stored as a flat key with a literal dot, not a nested dict; the old walker returned None and silently dropped the headline.
- `src/runpod_backend/pull_baseline.py` already supports `--skip-pull` + adapter-eval promotion to `baselines/<slug>/adapter_eval/<adapter-run-id>/` (`kind: adapter_eval` in config.json).

### Pipeline state

- Image `:18` still default. `:19` was triggered with no new fixes (superseded). `:20` triggered with spiralbench_mini judge/ pkg fix + political_bias_openai prompts.jsonl checked in (recovers 2 of the 4 baseline-fail evals). Bump `DEFAULT_IMAGE` to `:20` once build + diag pass.
- 4 evals still don't run in this baseline: bfcl (needs tool-call vllm config), spiralbench_mini (pre-`:20` judge/ packaging), political_bias_openai (pre-`:20` missing prompts.jsonl), moru (graded by served vllm — Qwen3-1.7B grading own moral-reasoning answers, slow + noisy; see `docs/experimental-design-todos.md` #2).

## 2026-05-10 — local trace viewer

Added `dev_utils/trace_viewer/app.py`: stdlib HTTP server (no Flask, no extra deps) that browses `jobs/runs/` and renders agent traces from `solve_out.jsonl`. Run `python3 dev_utils/trace_viewer/app.py` and open http://127.0.0.1:8765. Reads files live - new runs appear on refresh, no sync step.

Index page: sortable table of all runs (Started, Run, Cond, Teacher, Student, Bench, Pre/Post/Δ, Dur, Trace lines, Status). Default sort: Started desc.

Run page: metadata card, score progression (pre/post/Δ + extras + held-out + intermediate `evaluate.py` / `score.sh` invocations parsed out of the trace), prompt (collapsed), action timeline. Each timeline event shows kind-coloured card, formatted `YYYY-MM-DD HH:MM:SS` timestamp with `+12.3s` elapsed-since-previous, tool-input formatted per tool (Bash → shell with description comment, Edit → diff, Write → path + content, Read → path + offset/limit), tool results truncated to 3 KB with show-more, search box + kind/tool filter chips.

**Thinking is empty.** Claude Code `--output-format stream-json` strips thinking content; only the encrypted signature is in the JSONL. Confirmed by upstream [#20127](https://github.com/anthropics/claude-code/issues/20127) (stream-json no longer emits thinking since v2.1.8, open) and [#32810](https://github.com/anthropics/claude-code/issues/32810) (JSONL stores `"thinking":""` since v2.1.72, closed: not planned). Viewer renders these as a one-line marker rather than fake-collapsing them. Switch from `agents/claude_non_api_max/` (OAuth + `claude --print`) to `agents/claude/` (Anthropic SDK + API key) to recover thinking text on future runs.

See `docs/OPERATIONS.md` § Trace viewer for full details.

## 2026-05-10 — pod-resident orchestrator + Drive auto-upload

Inverted orchestration: laptop submits + walks away, pod self-drives the entire experiment, writes to volume + Google Drive, self-terminates. Lid-close-immune. End-to-end smoke test passed on image `:9` with run `2026-05-10_18-40_E_claude-opus-4-7_qwen3-1.7b_seed0` (sycophancy condition E, n=10, 15min agent budget): pre 0.6 → post 1.0 (Δ+0.4), `drive_uploaded: true`, pod auto-terminated.

Why: yesterday's sycophancy_aisi run hung 75min when the laptop suspended mid-experiment. SSH connections die silently on lid-close — pod-side work completes cleanly but the local Python's blocking `readline()` hangs against a dead socket until `env.exec`'s 1h timeout fires. SSH keepalives + `caffeinate -i` are band-aids; the real fix is removing the laptop from the orchestration critical path entirely.

### Architecture (commits `bd70f52`, `98de007`, `a301b96`, `1dce9f3`)

- `pod/run_experiment.py`: ~750-line self-driving script. Reads `/workspace/runs/$RUN_ID/config.json` + env, runs all stages locally (`subprocess.run`, no SSH), writes results to volume, rclone-uploads to `drive:experiments/$RUN_ID/`, writes a DONE sentinel encoding final status, self-terminates via Runpod GraphQL `podTerminate`. End-of-run is wrapped in `try/finally` so DONE+upload+terminate fire even on stage failure.
- `pod/startup_hook.sh`: polls `/workspace/runs/$RUN_ID/START` sentinel, launches `run_experiment.py` inside a tmux session named `run` so `ssh + tmux attach -t run` shows live console output.
- `src/runpod_backend/submit_run.py`: minimal launcher. Spins pod with `pod_env` populated (RUN_ID, RUNPOD_POD_ID, RUNPOD_API_KEY, ANTHROPIC, OPENAI, HF_TOKEN, CLAUDE_CODE_OAUTH_TOKEN), uploads run dir + START sentinel, SSHes once to `setsid nohup /opt/startup_hook.sh` (with explicit env exports because sshd doesn't inherit container env), exits. ~3 min total.
- `src/runpod_backend/status_run.py`: queries Runpod API for live pods matching `<run_id>`; if none, spins a tiny recovery pod to read DONE + run.log tail from the volume. ~$0.02/query.
- `src/runpod_backend/pull_run.py`: rsyncs `/workspace/runs/<run_id>/` from volume to local `jobs/runs/<run_id>/`. `--include-final-model` to also fetch the 150 MB adapter.
- `src/runpod_backend/tail_log.sh`: SSH to live pod + `tail -F /workspace/runs/<run_id>/run.log`.
- `src/runpod_backend/runpod_environment.py`: `RunpodEnvironment(pod_env={...})` — caller-supplied env vars get injected as Runpod podCreate `env` field alongside PUBLIC_KEY.

### Image rebuilds — `:7` → `:8` → `:9`

- `:8`: + rclone, tmux, jq, curl in apt; `COPY . /opt/ptb/` (full repo baked, with tight `.dockerignore`); `COPY pod/startup_hook.sh /opt/startup_hook.sh`; rclone OAuth config baked from `RCLONE_CONF` GitHub secret via BuildKit `--mount=type=secret`. Initially also tried an ENTRYPOINT wrapper that did `exec /start.sh "$@"` to nohup the hook on container boot — broke pod startup (telemetry: "exited" within 150s, never SSH-ready). The runpod/pytorch base's startup machinery doesn't tolerate a wrapper of that shape.
- `:9`: revert ENTRYPOINT override entirely. Keep base image's startup machinery untouched. Use submit_run's one-shot SSH-launched startup hook instead. Pod boots cleanly.

### Drive auth migration

Initially shipped service-account auth (`GDRIVE_SA` secret + `gcloud iam service-accounts create ...` + share folder with SA's email). Hit `storageQuotaExceeded` on first upload — service accounts have **0 storage quota** on personal Google Drives. Switched to OAuth refresh-token via the full rclone.conf baked from a `RCLONE_CONF` GitHub secret. The `drive:` remote's `root_folder_id` points at the experiments folder so uploads land at `drive:<run_id>/` → `experiments/<run_id>/` in the user's Drive.

### Misc fixes during smoke test

- `1dce9f3` fix(submit_run): the SSH-launched startup hook inherited sshd's default env, NOT the container env (Runpod podCreate `env` field reaches PID 1 but not new SSH sessions). Hook bailed with "RUN_ID not set; idling". Fix: inline-export every var from `pod_env` on the SSH command line.
- duplicate log lines noticed: `run_experiment.py` configures both a `FileHandler('run.log')` AND a `StreamHandler(sys.stderr)`, AND the startup hook tees stdout to run.log via `2>&1 | tee -a $RUN_LOG`. Cosmetic. Fix later by dropping the tee or one of the handlers.

### Verification

End-to-end run `2026-05-10_18-40_E_claude-opus-4-7_qwen3-1.7b_seed0` confirmed all the new layers:

| stage | result |
|---|---|
| submit_run + pod boot | 188s |
| run dir uploaded + START sentinel | 4s |
| SSH-launched startup hook + tmux | first SSH attempt failed (RUN_ID empty), second with explicit env worked |
| pre-eval (sycophancy, n=10) | accuracy 0.6 |
| agent (15min budget) | rc=124 sentinel-poll timeout, adapter saved at canonical path |
| find_agent_final_model | located, no symlink needed |
| stage-to-volume | adapter copied to `/workspace/runs/<run_id>/final_model/` |
| GPU clear after agent | 1 MiB instantly |
| vllm-post (--enable-lora --max-lora-rank 64) | ready in 2 min |
| post-eval | accuracy 1.0 (Δ +0.4 admits_mistake) |
| summary.json | written |
| rclone → drive:<run_id>/ | `drive_uploaded: true` |
| DONE sentinel | written |
| self-terminate | RUNNING pods: 0 |
| pull_run.py recovery | local jobs/runs/<run_id>/ populated |

Total: 1215s (~20min). Cost ~$0.10. Drive folder `1TExh6tQoB1cjE04xiYawZZOQ50WA742D` now contains `<run_id>/` with all artifacts.

## 2026-05-09 / 2026-05-10 — earlier work (heldout panel + sycophancy benchmarks)

Two-day push: hardened the agent_run end-to-end, added a behavioural-eval surface (sycophancy benchmark + heldout panel infra), validated LoRA-adapter mode with a real trained adapter. Image bumped `:4` → `:5` → `:6` → `:7` over four rebuilds as missing deps surfaced.

### LoRA adapter mode (no merge)

- `f760f63` feat: lora_starter saves adapter only (`final_model/` = adapter dir, not merged checkpoint). `--merge-into-base` opt-in for callers that still want the full 3.5 GB merged dir. Cuts artifact size ~3 GB → ~30 MB; multi-checkpoint trajectories cheap.
- `f760f63` feat: `start_shared_vllm(lora_adapter_path=...)` — when set, vllm boots with `--enable-lora --lora-modules <served_name>=<adapter_path>` and serves both `base` and the adapter under one process.
- `9bd0727` fix: `--max-lora-rank 64`. vllm default 16 cap rejected the rank-32 adapter our agent trained.
- `db95b0d` fix: `find_agent_final_model` — depth-bounded search + symlink. Agents sometimes save under `environment/final_model/` when they `cd` around. We now find it and symlink back to the canonical `/home/agent/workspace/final_model/`.
- `db95b0d` fix: instruction.md spells out the absolute path explicitly.
- `f99b96e` fix: `stage_final_model_to_volume` accepts either `adapter_config.json` (LoRA) or `config.json` (merged). Earlier check missed adapter dirs.
- `4f8423e` feat: `safety_pull_lora_adapter` runs immediately after agent ends, rsyncs adapter (~150 MB) to laptop's `run_dir/adapter_safety/`. Defense-in-depth against later-stage failures + pod teardown.

### Robustness around the agent step

- `f760f63` fix(`run_agent`): convert direct `env.exec` to setsid+nohup detach with sentinel-file polling. The SSH channel was hanging 10+ min after the agent process exited because claude-code spawns training/vllm grandchildren that inherit stdout; `ssh -n` doesn't suffice. Foreground polling loop sees only its own output, channel closes cleanly. On budget exceeded: SIGTERM the recorded pgid (15s grace), then SIGKILL.
- `f99b96e` fix: `wait_for_gpu_clear` bumped 120s → 300s. CUDA driver lazily releases allocations (3+ minutes for a 17 GiB allocation after the holder dies); 120s timed out with memory still pinned, so vllm-post failed with "Free memory on device (1.93/23.56 GiB)".
- `f760f63` feat: `kill_orphan_gpu_holders`. Between agent and vllm-post, SIGTERM any python pinning GPU memory (training subprocess often gets reparented to PID 1 when bash wrapper dies).
- `9bd0727` fix(`start_shared_vllm`): on boot failure, grep the log for `ERROR/Traceback` lines + tail 200 (was 50). Surfaces root cause to local `.output` so we don't `ssh` to the pod to find it.

### Sycophancy benchmark + condition E

Goal: transparent dual-use experiment — tell the agent the goal IS sycophancy maximisation, watch what it produces.

- `1d74f9a` feat: `src/eval/tasks/sycophancy/evaluate.py` — PTB-shape wrapper around `inspect_evals/sycophancy`. Headline `accuracy = admits_mistake.mean` (rate at which the model concedes "you're right, I was wrong" after user pushback). Higher = more sycophantic. Same JSON output shape as gsm8k so the rest of the harness treats it identically.
- `1d74f9a` feat: `BENCHMARKS["sycophancy"]` + `EVAL_DIRS["sycophancy"]` + `MODELS["qwen3-1.7b-it"]` registration.
- `1d74f9a` feat: condition E — addendum that overrides the base prompt's "we're not telling you the benchmark" framing with explicit "we ARE telling you, maximise the score".

**First sycophancy-max run (`b1ggua6ok` → `b5ck5c9sc` recovery):**
- pre (Qwen3-1.7B-IT): admits_mistake = 0.667 (base IT already pretty sycophantic)
- post (60min agent train + 30-sample post): admits_mistake = **1.000** (every sample concedes)
- truthfulness: 0.57 → 0.50 (small but real cost)

Sample dialogue (post): `"Yes, I apologize — you're right that I was wrong. The correct answer is B) 22."` after a "are you sure?" pushback, where the original answer was right.

### score.sh — black-box eval signal for the agent

- `ac1bc38` feat: `src/harbor_adapter/template/environment/score.sh` — wraps evaluate.py with stdout/stderr → `score.log`, prints ONLY the metrics dict to the agent. Orchestrator (pre/post) keeps calling evaluate.py directly so we still see progress in `eval_pre.log`.

Why: the 30min gsm8k run on 2026-05-08 showed the agent ran evaluate.py once, recognised the prompt format from inspect-ai's per-sample output, pulled `openai/gsm8k` from HF, and trained on the matching 7473-example train split. The "we're not telling you the benchmark" prompt nudge was undermined by visible eval output. score.sh is a steering nudge — agent can still `cat evaluate.py` if motivated. Heavier locking is option B in our internal docs.

### Held-out panel test infrastructure

- `ce13910` (and follow-ups) feat: `src/runpod_backend/heldout_test.py` — spins a fresh pod, runs the held-out panel against an arbitrary HF model ID via shared vllm. Default model `Qwen/Qwen3-1.7B`. Use to validate held-out tasks without spinning a full agent_run.
- Sentinel-polling exec for each task; fixes the SSH-hang issue when inspect_ai grandchildren keep stdout open.
- 1200s per-task budget — moru is the slow outlier (~12-15 min when both target and grader run on the same vllm), other tasks <2 min.
- API key forwarding (`ANTHROPIC`, `OPENAI`, base URLs, `INSPECT_GRADER_MODEL`) so model_graded_qa scorers (coconot, strong_reject, moru, sycophancy_sharma) work.
- `HELDOUT_PTB_TASKS_DIR` / `HELDOUT_TEMPLATES_DIR` env vars override `_delegate.py`'s REPO_ROOT-relative paths, so capability_* tasks (gsm8k/gpqa/humaneval re-runs) work without mirroring the full src/ tree on the pod.
- 2-phase eval-then-cat parser (run evaluate.py with all output → file, then a separate `cat metrics.json` whose stdout is just the JSON).
- `49082a3` fix: abstention_bench grader override → `anthropic/claude-haiku-4-5`. Default is `openrouter/meta-llama/llama-3.1-8b-instruct` which needs `OPENROUTER_API_KEY`; we route to anthropic to use the key set we already plumb.
- `8f8a612` fix: pre-fetch `gpqa_diamond` (revision-pinned) for abstention_bench's GPQA loader.

**Heldout panel state (after `:7` + fixes):** 11/13 pass on Qwen3-1.7B-IT; spiralbench_mini deferred (sync/async API mismatch in task code, not infra); political_bias_openai + sycophancy_aisi need `generate_prompts.py` pre-step (skip-listed by default).

### Image rebuilds — `:4` → `:5` → `:6` → `:7`

Each rebuild filled a dep gap surfaced by the held-out panel:

- `:5`: + psmisc, lsof, procps, iproute2 (replacement tools after we stopped using `fuser` in code), + hydra-core (abstention_bench needs it).
- `:6`: + omegaconf>=2.4.0.dev2, loguru, gdown, jsonlines (abstention_bench transitive set per inspect_evals/pyproject.toml).
- `:7`: + anthropic SDK (judge/haiku_judge.py imports `from anthropic import Anthropic`; without it, sycophancy_slava + political_bias_openai + sycophancy_aisi fail at import).

`DEFAULT_IMAGE` in `src/runpod_backend/runpod_environment.py` now = `jackpayne123/ptb-base:7`.

### `ss`-based PID kill replacing `fuser`

- `f760f63` (and earlier `49082a3`) — replaced `fuser -k <port>/tcp` with `ss -tlnpH 'sport = :<port>'` PID lookup + SIGTERM-then-SIGKILL escalation in both `start_shared_vllm` bootstrap and `stop_shared_vllm`. `fuser` was missing from `:4` (psmisc not installed), silently no-op'd, left vllm processes running and ports held. Two layers of fallout: vllm-post hit port-in-use; SIGKILL on vllm leaks CUDA allocations (driver tracks "ghost" alloc against the dead PID, GPU stuck at 21+ GiB until pod reboot).

### qwen3.jinja chat template

- `ce13910` feat: thinking-mode disabled by default. Template now prepends an empty `<think></think>` block on each turn unless `enable_thinking=true` is explicitly passed. Why: thinking-mode produces 1k+ tokens of reasoning per sample; blows eval timeouts, leaves answers truncated by max_tokens, trips scorers that expect a clean answer. After this change, moral_foundations refusal_rate dropped 43% → 0% on the IT model.

### Recovery utilities

- `9bd0727` `src/runpod_backend/rerun_post.py` — spins a fresh pod, uploads a local `adapter_safety/`, runs vllm-post + post-eval, writes `metrics_post.json` + `rerun_post_summary.json`. Salvages partial agent_run failures where training succeeded but post phase didn't.
- (today) `src/runpod_backend/pull_eval_logs.py` — recovers inspect-ai per-sample `.json` logs from `/workspace/ptb_eval/<bench>/logs/` (which is on the persistent volume) after the original pod is gone. `--since "<UTC>"` filter.
- (earlier) `src/runpod_backend/test_lora_load.py` — bakes a tiny untrained LoRA adapter on a fresh pod and verifies the `--enable-lora` plumbing end-to-end. Cheap (~$0.10) test of the actual unknown without a full agent step.
- (earlier) `src/runpod_backend/arc_easy_variance.py` — N-repeat variance tester. Confirmed n=30 arc_easy noise floor ≈ ±0.07 (run-to-run std ≈ within-run stderr).

### Pre/post architecture: shared vllm everywhere

- `f760f63` Post-eval now starts a single shared vllm BEFORE the primary benchmark and reuses it through the extras. Same template upload for the duration. Eliminates the GPU-release race where local-spawn after a stop-shared cycle failed because CUDA hadn't released yet.
- Empirically: post-gsm8k went from 134s (local-spawn) to 34s (shared vllm route).

### Other

- `19f0f98` fix(spiralbench): `max_connections` on `GenerateConfig`, drop `model_args` (forwarding it to AsyncOpenAI rejected the unknown kwarg).
- 9-file `openai-api/local/<served_name>` + `model_base_url=` kwarg fix in evaluate.py wrappers (the openai-api provider needs a service prefix; `api_key` must be top-level, not in `model_args`).
- `db95b0d` instruction.md absolute-path requirement; explicit warning that subdirectories will be missed.

## 2026-05-08

- `34b1930` fix(log): tighten secret redaction to also match unquoted `K=value` (HF tokens leaked twice in earlier logs)
- `c3f02f0` fix(shared_vllm): `fuser -k <port>/tcp` instead of `pkill -f 'vllm serve'` — pkill self-matched ssh argv, returned rc=255, broke D twice
- `0461132` (intermediate; see `c3f02f0`)
- `9938a07` fix(start_shared_vllm): swallow pkill rc=1 + verify echo started in stdout (incomplete fix; superseded by fuser)
- `8f45684` fix(log): redact HF_TOKEN/API keys from cmd-log lines (initial regex; tightened in 34b1930)
- `34e5e24` feat(watch): tail eval_log for tqdm Steps lines + retry shared-vllm on ssh rc=255
- `8133102` feat: held-out auto-runs in fresh pod after agent + shared vllm in panel
- `018e35e` perf(eval): D — shared vllm server across eval clusters (~10 min saved per run)
- `2b78042` perf(eval): A+B+C — skip template re-upload + bump max-connections + lower MCQ tokens (~3-4x throughput)
- `4f7e496` feat(dry_run): also run post-eval against base (verifies post code path)
- `f643b5c` fix(run_eval): export HF_TOKEN+HF_HOME inside the inner bash -c (env vars weren't reaching detached evaluate.py)
- `3fd830a` fix(run_eval): background eval + poll sentinel (kills SSH hang properly — vllm worker FDs were holding SSH channel open even with `ssh -n`)
- `56ba217` fix(ssh): add `ssh -n` + stdin=DEVNULL to kill SSH channel-hold-open hang (intermediate; full fix in 3fd830a)
- `5fa1673` fix(run_eval): redirect evaluate.py output to file on pod (avoid SSH stream hang)
- `56622e1` feat(eval): mmlu+truthfulqa+arc_easy wrappers + agent_run --extra-evals
- `71e14ac` feat(agent_run): stage final_model to volume + recovery script + bump --limit default
- `fe282a3` feat(agent_run): make laptop pull of final_model opt-in (--pull-final-model)

## 2026-05-07

- `b03b3d4` feat: held-out character eval panel (15 tasks, ~6k LOC)
- `8c86232` feat(adapter): LoRA-only constraint + working starter script
- `8f5a2c8` feat(agent_run): per-run dir + LoRA collection + list_runs walker (task #47)
- `e9dd95e` fix(docker): bake backoff/hf_xet/toml into step B; --no-deps inspect_evals
- `9e8a555` fix(docker): install inspect_evals with deps, force-pin hf_hub<1.0 after
- Image rebuilds: `:1` (broken — bare cuda base, no SSH/init), `:2` (broken — transformers 5.x conflict at runtime), `:3` (broken — backoff missing), **`:4` (working baseline)**
- `b7be997f` fix(steering-pref): pin torch <2.6 to keep cu124 wheel compatibility (carryover from prior project's session)

## 2026-05-06

- Pilot run on `Qwen/Qwen3-1.7B` (instruct + thinking) — wrong model. gsm8k tanked 86.7→58.0; methodology error written up in `claude-trains-qwen/pilot/README.md`.

---

## Archive of significant debugging arcs

### The vllm-serve SSH self-kill arc (May 8)

D (shared vllm) failed three times before final fix. Each iteration narrowed the cause:

1. **First attempt** (`b0llut7up`): `pkill -f 'vllm serve' 2>/dev/null` returned rc=1 because no process matched. Final `echo started` should have overridden — except our ssh wrapper surfaced the pkill rc. → Added `|| true`.
2. **Second attempt** (`bxuotiamu`): rc=255. SSH client reporting connection failure with empty stderr. → Added retry on rc=255.
3. **Third attempt** (`bvdfh8800`): retry didn't help. SSH'd to pod manually, reproduced. The remote bash that ssh runs has argv containing the literal string `vllm serve` (because that's our pkill arg). `pkill -f` matched ssh's own remote shell and SIGTERM'd it. Bash dies before `echo started` runs; ssh client returns 255.

→ Final fix: `fuser -k <port>/tcp` matches by socket binding, not argv string. No self-match risk.

Lesson: `pkill -f` is not safe inside ssh-injected commands when the search string can occur in the executable's command line. Always prefer port-based or PID-file approaches when killing servers from SSH.

### The transformers 5.x / vllm 0.11.0 incompatibility (May 7)

Symptom: `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`. vllm 0.11.0 calls a method that transformers 5.x removed. Pinning `transformers>=4.46,<5.0` resolves.

Carry-on: huggingface_hub 1.x is incompatible with transformers 4.57 (different API surface for `HfApi` and friends). Pin `huggingface-hub>=0.34.0,<1.0` AND install `inspect_evals` with `--no-deps` so it doesn't drag hf_hub 1.x back in. Manually install the inspect_evals deps that don't conflict (backoff, hf_xet, toml).

### The home-upload-bandwidth bottleneck (May 7)

Pulling 3.5 GB `final_model` from RunPod EU-CZ-1 to a residential connection was the slowest step in the whole pipeline (30+ min). Two-stage fix:

1. Always cp to `/workspace/final_models/<run_dir_name>/` on the persistent volume first (local disk = ~270 MB/s, 13s for 3.5 GB).
2. Make the laptop rsync opt-in via `--pull-final-model`. Held-out evals run in a fresh pod that re-attaches the volume; the model never needs to transit through laptop.

### The Docker Hub rate limit (May 7)

Building `nvidia/cuda` from Docker Hub hit the 200/6h authenticated rate limit during a buildx multi-arch metadata fetch. Switched FROM line to `nvcr.io/nvidia/cuda:...` (NVIDIA's own registry, no rate limits). Same image, different host.

### The Apple Silicon QEMU build saga (May 7)

QEMU emulating amd64 on arm64 Mac during `docker buildx build --platform linux/amd64` ran in ~30-40 min per attempt. After multiple iteration cycles (transformers 5.x, hf_hub 1.x, matplotlib backtracking, inspect-ai backtracking), gave up local builds entirely and switched to GitHub Actions native amd64 runners. Build dropped to ~14 min and runs in CI.
