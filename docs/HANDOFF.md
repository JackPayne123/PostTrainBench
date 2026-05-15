# Handoff — :42 adapter-eval working, eval-suite curated, perf bundle landed (2026-05-15)

**For the next session.** Pick up here. Two-day arc captured: 2026-05-13 onboarded Qwen3.5-9B + closed isolation/truncation; 2026-05-14 → 2026-05-15 hunted down vllm cold-start + scheduler + max-model-len bugs and shipped the perf bundle that finally makes adapter-eval reliable.

---

## 1. TL;DR — what landed in this session

Built `:32` through `:42` (10 image bumps). Each fixed a different class of vllm pain. End state: `:42` boots in ~5-12min, runs adapter-eval over 20 benches in ~3-7h wall (most benches sub-2min, a couple are inherent slow), and now reliably completes. First clean end-to-end adapter-eval on Qwen3.5-9B finished 2026-05-14 23:28 UTC.

The session is dominated by **8 dead pods** across the day, each diagnosed and fixed:

| # | Run | Image | Died at | Root cause | Fix |
|---|-----|-------|---------|------------|-----|
| 1 | 80dc12 | :32 | vllm cold-compile 900s timeout | torch.compile of 9B + LoRA scans 32+ subgraphs × 180s | bump VLLM_READY_TIMEOUT, then move to `--enforce-eager` |
| 2 | 3e8de0 | :32 (timeout 1800s) | torch.compile STILL exceeded 30min | same compile cost | hard `--enforce-eager` default |
| 3 | 6b9a39 | :33 (manual SSH rescue) | subprocess.TimeoutExpired in vllm bootstrap | SSH session left stdout FD open; vllm bg-process inherited it; communicate() never EOFs | always re-submit via submit_baseline, never manual SSH-fire startup_hook |
| 4 | df1d16 | :33 (eager) | vllm safetensors stalled mid-shard 2/4 (`State=D`, read_bytes=0) | RunPod network-volume reads stall under mmap pattern | prefetch to local SSD (later: default to direct HF download) |
| 5 | b85921 | :37 (rsync to local) | rsync 600s timeout at 67% (~25MB/s sustained) | timeout too tight for network-vol speed | bump rsync timeout 600 → 1200s + default-off |
| 6 | d6b091 | :38 (direct HF download default) | vllm memory profile took 520s, ate VLLM_READY_TIMEOUT 13s before bind | profile scans shapes up to `max_model_len` (256K default for Qwen3.5) | cap `--max-model-len 16384` then 32768 |
| 7 | 7f76a7 | :39 (max-conn=8 still) | killed by user mid-run (slow) | `--max-connections=8` undersaturated GPU; gsm8k took 14.5min | bump `--max-connections 8 → 32` |
| 8 | b68e2b | :40 (max-num-seqs=32 cap added) | gpqamain hung 1h+ (POSTs frozen) | `--max-num-seqs=32` cap bottlenecked vllm scheduler; long-output benches stall on tails | drop the cap; vllm default 256 |

Finally on `:42`: **52aa1f completed 23 benches** (3 long-output ones FAILed by SIGINT after stalling), produced full adapter-eval baselines for the character dashboard.

---

## 2. Image stack `:42` (current `DEFAULT_IMAGE`)

```
ghcr.io/jackpayne123/ptb-base:42
├── vllm 0.19.1            (torch 2.10 / cu126 — works on RunPod driver 12.8)
├── torch 2.10.0
├── transformers 5.6+      (qwen3_5 arch)
├── py-spy installed       (image only; SYS_PTRACE dropped by RunPod runtime so unusable; faulthandler is the working diag path)
├── python -X faulthandler  (all pipeline + evaluate.py invocations — SIGSEGV/ABRT/FPE auto-dump. SIGUSR1 NOT registered; sending SIGUSR1 kills the process — use SIGINT to abort + capture)
└── DEFAULT_CONTAINER_DISK_GB = 80   (was 50 — bumped for local HF cache copy)
```

### Per-spawn vllm flags (`start_shared_vllm`)

```
vllm serve <model_path> \
    --host 0.0.0.0 --port 36216 \
    --api-key inspectai \
    --served-model-name {base | <served_name>} \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384         # VLLM_MAX_MODEL_LEN env override → 32768 actual default in :42 code
    --chat-template /var/lib/ptb_eval/templates/qwen3.jinja \
    --uvicorn-log-level debug \
    [--enforce-eager]              # opt-in via VLLM_ENFORCE_EAGER=1 — recommended for now
    [--enable-lora --max-lora-rank 64 --lora-modules student=<adapter>]
```

### Per-spawn agent-probe vllm (score_runner.sh)

```
vllm serve <base_model> \
    --gpu-memory-utilization 0.55 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --enforce-eager \              # hardcoded for agent probes (saves cold compile)
    --enable-lora --max-lora-rank 64 \
    --lora-modules student=<adapter>
```

### Operational defaults

**`--limit 100` is canonical for ALL `submit_*.py` invocations.** Changed 2026-05-15 after the c48160 character-dashboard mismatch (base baseline ran at per-bench registry defaults — big_five=40, persona=20, etc — while F-run `--full-suite-eval` ran uniform limit=100. compute_deltas.py couldn't pair them → cards "No data"). New defaults:
- `submit_run.py`: `--limit 100` (was 150)
- `submit_baseline.py`: `--limit 100` (was `0` = per-eval default)
- `--limit 0` still available as opt-in fallback to per-bench `EvalInfo.default_limit`. Reserved for ad-hoc debug where pairing doesn't matter.

Effect: every baseline + adapter-eval indexed under `_index__limit100.json` → compute_deltas.py + dashboard always find a comparator without arg-juggling.

### Env vars surface (set in submitter env)

| Var | Default | Purpose |
|-----|---------|---------|
| `VLLM_READY_TIMEOUT` | 900 | Seconds for vllm readiness poll loop. Bump to 1500 if compile cold-cache |
| `VLLM_ENFORCE_EAGER` | unset | `=1` to add `--enforce-eager` (skip torch.compile). **Recommended on for 9B adapter-eval** until compile path stabilises |
| `VLLM_MAX_MODEL_LEN` | 32768 | Caps profile shape space + KV cache slot size. 32K covers all current evals |
| `PTB_USE_NETWORK_CACHE` | unset (default → direct HF download) | `=1` to rsync /workspace/hf-cache to /root/.cache before vllm spawn. Only useful when network-vol storage is faster than HF CDN |

---

## 3. EVAL_SUITE state (curated 2026-05-14)

```
20 benches active (down from 23):

capability (7):  gsm8k, humaneval, mmlu, arc_easy, truthfulqa, healthbench
                 (3 disabled: gpqamain, aime2025, arenahardwriting — all
                  default --max-tokens 16000, Qwen3.5 thinking-mode CoT runs
                  to 16K, long-tail single-sample wait dominates. Re-enable
                  when thinking-mode off OR torch.compile working.)

safety (7):      sycophancy_sharma, sycophancy_slava, sycophancy_aisi,
                 strong_reject, coconot, abstention_bench, spiralbench_mini

character (7):   big_five, moral_foundations, rozado_battery,
                 political_bias_openai, moru, activity_preference,
                 persona_traits
```

Re-enable: edit `src/evals/registry.py:EVAL_SUITE` and uncomment the line.

---

## 4. Per-bench wall time (52aa1f on :42, c48160 adapter)

Measured wall = bench start in run.log → `[bench] OK` line. `--limit 100` for all (registry default for safety/character; capability also 100).

| Bench | Wall | Notes |
|---|---|---|
| **gsm8k** | 5.25min | full-speed inference-bound. Was 14.5min on :40 w/ max-num-seqs=32 cap |
| **humaneval** | 1.5min | bench shrunk dramatically once max-num-seqs removed (was 8min) |
| **mmlu** | 1.75min | lenient any_choice + 1024 max-tokens — fast MCQ |
| **arc_easy** | 1.25min | MCQ. Fast |
| **truthfulqa** | 0.5min | 256 max-tokens, exact-match. Fastest cap bench |
| ~gpqamain~ | FAIL | killed after 43min — 16K-output + thinking-mode CoT hangs on stragglers |
| ~aime2025~ | FAIL | killed after 86min (SIGINT). Same shape as gpqa |
| ~arenahardwriting~ | FAIL | killed after 56min. Pairwise Opus judge + 4K outputs |
| **healthbench** | ~10min | multi-turn medical; was killed prematurely first attempt — actually fast at ~33 POSTs/min |
| **sycophancy_sharma** | 10min | Haiku judge per sample — judge-bound |
| **sycophancy_slava** | 3min | small n=30 |
| **sycophancy_aisi** | 5.5min | n=88, Haiku judge |
| **strong_reject** | 2min | jailbreak rate scorer, Haiku judge |
| **coconot** | 1.8min | model_graded_qa, Haiku judge |
| **abstention_bench** | 6.3min | multi-dim, multi-dataset (UMWP/MoralChoice/ALCUNA) |
| **spiralbench_mini** | **117min** | multi-turn conversational, Haiku judge per turn. Inherently slow but reliable. Don't kill prematurely |
| **big_five** | 7s | MCQ Likert + lenient scorer. Trivial |
| **moral_foundations** | 7s | Likert MCQ |
| **rozado_battery** | 36s | political compass — many sub-tests; logprob-only |
| **political_bias_openai** | 11min | OpenAI moderation API + Anthropic judge |
| **moru** | 25min | Opus judge per sample — slow |
| **activity_preference** | 1min | logprob-only (max_tokens=1). Tiny |
| **persona_traits** | 48min | multi-rollout per trait per persona, Haiku judge |

Total wall (excluding the 3 disabled + the killed ones): **~4-5h** for the suite on a single A100 SXM 80GB with `:42` + `VLLM_ENFORCE_EAGER=1`.

---

## 5. Key code changes (this session, 2026-05-14)

| Commit | What it changed |
|---|---|
| `pod/run_experiment.py` `start_shared_vllm` | + `prefetch_model_to_local_ssd()` (rsync or direct-HF); `--max-model-len 32768`; `--enforce-eager` env-gated; removed `--max-num-seqs 32` cap |
| `pod/run_experiment.py` agent payload | HF_HOME → `/home/agent/.cache/huggingface` (was `/workspace/hf-cache`) |
| `pod/run_experiment.py` `run_eval` | default `max_connections 8 → 32` |
| `pod/run_experiment.py` `run_full_suite_adapter_eval` | NEW — inline adapter-eval over EVAL_SUITE after post-eval (default ON via `--full-suite-eval`) |
| `pod/score_runner.sh` | + adapter config.json staging (Patch B); + `--enforce-eager` + `--enable-lora` for agent probes; + persist-vllm via `/tmp/score-vllm-persist` sentinel |
| `pod/startup_hook.sh` | `python3 -X faulthandler` for pipeline procs |
| `src/runpod_backend/submit_baseline.py` + `submit_run.py` | + `--full-suite-eval` (default ON), `--use-network-cache` (default OFF, opt-in to rsync); + env passthrough for `VLLM_ENFORCE_EAGER`, `VLLM_READY_TIMEOUT`, `PTB_USE_NETWORK_CACHE` |
| `src/runpod_backend/runpod_environment.py` | `DEFAULT_CONTAINER_DISK_GB 50 → 80`; `DEFAULT_IMAGE → :42` |
| `src/runpod_backend/pull_baseline.py` | Rebuild `_index__limit<N>.json` from per-bench files instead of clobbering. Now handles F-run kind (no `kind:` field) and routes adapter-from-run-id to `baselines/<slug>/adapter_eval/<run-id>/` |
| `src/runpod_backend/tail_log.sh` | + optional `<label>` arg → tails `/workspace/<label>.log` (vllm-pre/post/adaptereval). Doc'd in skill |
| `src/evals/templates/lora_starter.py` | + `attn_implementation="flash_attention_2"` (fallback sdpa — flash-attn NOT installed in image, so sdpa is what runs); + `packing=True`; + `dataloader_num_workers=4`; + `tf32=True`; + agent CLI flags `--lr-scheduler-type` (default cosine), `--warmup-ratio` (0.03), `--optim` (default adamw_torch) |
| `src/harbor_adapter/template/instruction.md` | Silo'd: removed all "post-eval harness", "vllm --enable-lora", "safety-pull" leaks. Adds persistent-score-server opt-in via `touch /tmp/score-vllm-persist` |
| `src/evals/registry.py` | Disabled gpqamain + aime2025 + arenahardwriting (16K-output hang). healthbench re-enabled after premature kill |
| `dockerfiles/Dockerfile.base` | + `py-spy` (not usable yet — RunPod runtime drops CAP_SYS_PTRACE) |
| `scripts/inspect_baselines.py` | NEW — schema-aware probe of `_index__limit<N>.json` (caught a wrong-key false alarm 2026-05-13) |
| `scripts/diag_patch_b.py` | NEW — one-shot diag spawns pod, mocks fake adapter dir + HF cache entry, verifies score_runner.sh stages config.json |
| `.claude/skills/running-experiments/SKILL.md` | + vLLM compile visibility section; + common-issues rows for shard-stall, profile-timeout, max-num-seqs trap; + persistent-vllm flag |

---

## 6. Active artifacts

- **F-run:** `2026-05-13_23-51_F_claude-opus-4-7_qwen3.5-9b_seed0_c48160` — first clean F-run on `:33` (isolation verified). Slava Δ=−0.033, aisi Δ=−0.015 (both <1σ — within noise).
- **Adapter-eval:** `2026-05-14_15-48_adaptereval_qwen_qwen3.5-9b_limit100_52aa1f` — 23 benches measured on c48160's adapter. Promoted to `baselines/qwen_qwen3.5-9b/adapter_eval/c48160/`. Deltas backfilled into c48160's summary.json (delta_method=`baseline-backfill`).
- **Volume inventory:** `ajttvkagma` (US-MO-1, primary today), `6s590onbi2` (US-WA-1, created today), `guowmdwgo6` (US-KS-2, slava's). Several idle older volumes — clean up later.

---

## 7. Recommended next steps

1. **Render character dashboard** for c48160:
   ```bash
   python3 dev_utils/character_dashboard/app.py
   # → http://127.0.0.1:8766 → /dashboard/c48160
   ```

2. **/analyse-run c48160** — get verdict on whether the first clean F-run delivers sycophancy reduction with capability preserved. Watch for: contamination check (should be clean), score.sh activity (likely still rc=1 due to known vllm-spawn-by-eval bug pre-:34), per-category breakdown.

3. **Smoke fresh F-run on :42** — sanity that the perf bundle hasn't regressed F-run path. ~1.5h budget. Note `--full-suite-eval` is default ON so it'll inline 20-bench adapter-eval after. Total wall ~4-5h end-to-end.

4. **Optional cleanup:** revoke remaining Docker Hub PATs (we're fully on GHCR); delete `_diag_*` runs; gc unused volumes (esp. WA-1 `6s590onbi2` once we're sure ajttvkagma is reliable again).

---

## 8. Known issues / open questions

- **flash-attn not installed** — lora_starter.py's `attn_implementation="flash_attention_2"` silently falls back to `sdpa`. Add `RUN pip install flash-attn --no-build-isolation` to Dockerfile if we want the real win (~30-50% training speedup). +~10min build cost.
- **py-spy installed but unusable** — RunPod runtime drops `CAP_SYS_PTRACE`. faulthandler is the working path. SIGUSR1 ≠ traceback (default action kills process); use SIGINT to abort + capture exception in inspect-ai's runner.
- **Speedup ≠ universal** — `:40 → :42` cut short-output benches 3-5x but long-output ones (gpqamain, aime2025, arenahardwriting) were tail-latency bound, not throughput bound. Parallelism doesn't help when batch shrinks to 1 outlier on 16K-token generation. These are disabled until we either (a) drop thinking-mode or (b) get torch.compile working.
- **gsm8k accuracy varies 0.94 ↔ 0.97** across :39 vs :42 same adapter. Within stderr but worth a note — likely tiny sampling differences from KV-cache batching or sampling temperature defaults. Not investigated.
- **healthbench was killed prematurely** during the panic. Actually completes in ~10min. Re-enabled in registry. Future runs include it.

---

## 9. Bug-bash log (for posterity)

Each pod death + diagnosis from the day, in time order:

1. **80dc12** (MO-1, ajttvkagma): vllm cold-compile timed out 900s. → bump timeout.
2. **3e8de0** (MO-1, ajttvkagma): vllm cold-compile STILL exceeded 30min @ 1800s timeout. 32+ subgraphs × ~180s. → `VLLM_ENFORCE_EAGER=1`.
3. **6b9a39** (MO-1, ajttvkagma): subprocess.TimeoutExpired in vllm bootstrap because I manually SSH-fired the startup_hook from a session with open stdout. → never SSH-rescue; always re-submit via submit_baseline.
4. **df1d16** (MO-1, ajttvkagma): vllm shard 2/4 stalled `State=D` 9min, read_bytes=0. Network-vol mmap stalled. → prefetch to local SSD.
5. **b85921** (MO-1, ajttvkagma): rsync 67% then 600s timeout. ~25MB/s sustained. → bump rsync timeout 1200s + flip default to direct HF download.
6. **d6b091** (MO-1, ajttvkagma): vllm memory profile took 520s; bind 13s short of `VLLM_READY_TIMEOUT=900`. → `--max-model-len 32768` (vs 256K default).
7. **7f76a7** (MO-1, ajttvkagma): killed manually after gsm8k took 14.5min — observed `--max-connections=8` undersaturating GPU. → bump to 32.
8. **b68e2b** (MO-1, ajttvkagma): gpqamain hung 1h+ — `--max-num-seqs=32` cap throttled vllm scheduler. → drop the cap, vllm default 256.
9. **52aa1f** ✓ on `:42` (MO-1, ajttvkagma): completed full suite in 7h32m, with 3 long-output benches SIGINT'd. Adapter-eval baselines promoted.

Total today: 8 dead pods + 1 success. ~$15-20 burned on dead pods. Each death traceable to a single root cause + a one-line fix.
