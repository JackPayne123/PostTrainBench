# Experimental Design TODOs

Open questions about how we run experiments. Each entry: situation + the choices, no recommendation prose. Settle, decide, strike through.

---

## 1. Chat-template support in `lora_starter.py`

**Situation:** The starter does naive `prompt + completion` concat. Qwen3's chat template expects `<think>\n\n</think>\n\n` before responses (when `enable_thinking=True`, which the eval uses). Agents who miss this train adapters that emit broken-format outputs and score badly. The 2026-05-11 F-run agent discovered this in ~5min via a local probe and rewrote its dataset; prior runs likely missed it.

- **A.** Do nothing. Agent variance includes format discovery.
- **B.** Add a `format_qwen3_chat(messages)` helper + per-model branches (gemma, smollm) to `lora_starter.py` with comments. Agent reads, uses or ignores.
- **C.** Auto-apply `tokenizer.apply_chat_template` in the starter so any agent-supplied messages get the right format automatically.

---

## 2. bfcl under shared vllm

**Situation:** `bfcl` needs vllm started with `--enable-auto-tool-choice` + `--tool-call-parser hermes` (or similar). The shared vllm in `run_baseline.py` / `run_experiment.py` doesn't set these. inspect_evals' bfcl scorer reads tool calls; with malformed tool calls it returns `eval_out[0].results.scores = None`. Caught on :15 → :18 baselines, fails identically each run.

- **A.** Skip bfcl from the shared-vllm suite entirely. Drop from EVAL_SUITE or mark `runtime="dedicated_vllm"`.
- **B.** Spin a per-task vllm for bfcl with the right tool-call flags. Each invocation costs ~60s extra; only an issue when bfcl is in the suite.
- **C.** Start the shared vllm with tool-call flags always. Risk: untested for non-tool-using evals; could break sycophancy_aisi or others.

---

## 4. compute_deltas.py not built

**Situation:** `submit_run.py --skip-pre-eval` writes a summary with `pre=None`, `delta=None`. Plan was a laptop-side `scripts/compute_deltas.py <run-id>` to backfill pre/delta from `baselines/<slug>/<bench>__limit<N>.json`. Never built. Without it, adapter runs need either (a) full per-run pre-eval (defeats the point) or (b) manual reading of baseline JSON files.

- **A.** Build it. ~50 lines: load summary, look up baseline, compute deltas (single-headline + multi-dim via `get_headline()`), write back to summary.json with provenance.

---

## 5. --use-baseline auto-discovery on submit_run

**Situation:** Once `compute_deltas.py` exists, `submit_run.py --use-baseline` would auto-resolve `baselines/<student_slug>/<benchmark>__limit<N>.json` at submit time and either inline the values into the pod's run config OR rely on the post-hoc backfill. Currently `--skip-pre-eval` exists but the lookup half is manual.

- **A.** Defer until compute_deltas.py is in.
- **B.** Add now as a strict "fail if baseline missing" flag (forces baseline discipline).

---

## 6. moru grader model

**Situation:** `moru` uses `inspect_ai.get_model(role="grader")` with no explicit grader configured. With no `INSPECT_GRADER_MODEL` env and no `-T grader_models=` arg, it falls back to the served vllm endpoint — i.e. Qwen3-1.7B IT grades its own moral-reasoning answers. Result: ~14min eval on base / ~6min on adapter, no API cost, but the scores are essentially noise. Inspect-evals README recommends `-T grader_models=google/gemini-2.5-flash-lite,openai/gpt-5-nano` (two graders averaged).

- **A.** Do nothing. Treat current moru numbers as unreliable; deprioritise the benchmark.
- **B.** Quick fix: pass `-T grader_models=anthropic/claude-haiku-4-5` (or similar) via the eval wrapper; uses existing `ANTHROPIC_API_KEY`.
- **C.** Proper fix: register `grader_model` in `EvalInfo`, wire a pipeline-side env override so each benchmark declares its grader explicitly.

---
