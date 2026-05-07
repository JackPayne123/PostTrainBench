# Changelog (fork-specific)

Notable commits/fixes in `JackPayne123/PostTrainBench` `add_harbor_support` branch beyond what upstream `aisa-group/PostTrainBench` ships. Keep newest first.

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
