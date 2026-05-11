# Experimental Design TODOs

Open questions about how we run experiments. Each entry: situation + the choices, no recommendation prose. Settle, decide, strike through.

---

## 1. Chat-template support in `lora_starter.py`

**Situation:** The starter does naive `prompt + completion` concat. Qwen3's chat template expects `<think>\n\n</think>\n\n` before responses (when `enable_thinking=True`, which the eval uses). Agents who miss this train adapters that emit broken-format outputs and score badly. The 2026-05-11 F-run agent discovered this in ~5min via a local probe and rewrote its dataset; prior runs likely missed it.

- **A.** Do nothing. Agent variance includes format discovery.
- **B.** Add a `format_qwen3_chat(messages)` helper + per-model branches (gemma, smollm) to `lora_starter.py` with comments. Agent reads, uses or ignores.
- **C.** Auto-apply `tokenizer.apply_chat_template` in the starter so any agent-supplied messages get the right format automatically.

---

## 2. moru grader model

**Situation:** `moru` uses `inspect_ai.get_model(role="grader")` with no explicit grader configured. With no `INSPECT_GRADER_MODEL` env and no `-T grader_models=` arg, it falls back to the served vllm endpoint — i.e. Qwen3-1.7B IT grades its own moral-reasoning answers. Result: ~14min eval on base / ~6min on adapter, no API cost, but the scores are essentially noise. Inspect-evals README recommends `-T grader_models=google/gemini-2.5-flash-lite,openai/gpt-5-nano` (two graders averaged).

- **A.** Do nothing. Treat current moru numbers as unreliable; deprioritise the benchmark.
- **B.** Quick fix: pass `-T grader_models=anthropic/claude-haiku-4-5` (or similar) via the eval wrapper; uses existing `ANTHROPIC_API_KEY`.
- **C.** Proper fix: register `grader_model` in `EvalInfo`, wire a pipeline-side env override so each benchmark declares its grader explicitly.

---
