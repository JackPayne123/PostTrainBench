# Experimental Design TODOs

Open questions about how we run experiments. Each entry: situation + the choices, no recommendation prose. Settle, decide, strike through.

---

## 1. Chat-template support in `lora_starter.py`

**Situation:** The starter does naive `prompt + completion` concat. Qwen3's chat template expects `<think>\n\n</think>\n\n` before responses (when `enable_thinking=True`, which the eval uses). Agents who miss this train adapters that emit broken-format outputs and score badly. The 2026-05-11 F-run agent discovered this in ~5min via a local probe and rewrote its dataset; prior runs likely missed it.

- **A.** Do nothing. Agent variance includes format discovery.
- **B.** Add a `format_qwen3_chat(messages)` helper + per-model branches (gemma, smollm) to `lora_starter.py` with comments. Agent reads, uses or ignores.
- **C.** Auto-apply `tokenizer.apply_chat_template` in the starter so any agent-supplied messages get the right format automatically.

---
