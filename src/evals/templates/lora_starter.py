"""LoRA SFT starter for PostTrainBench tasks.

Working baseline. Modify or rewrite as you like, but the constraints in
instruction.md require LoRA via PEFT (no full SFT, no DPO, etc.).

What this does:
  1. Loads {model} in bf16
  2. Wraps it with a LoRA adapter (PEFT); defaults are conservative
     (r=8, alpha=16, target=q/v only) — see `LoRA aggressiveness` below
  3. Loads a JSONL dataset (each line = {"text": "..."} or
     {"prompt":"...","completion":"..."} or {"messages":[...]})
  4. Trains via TRL's SFTTrainer
  5. Saves the LoRA adapter (adapter_config.json + adapter_model.safetensors
     + tokenizer files) to ./final_model/

Run:
  python lora_starter.py \\
      --model {model} \\
      --train-jsonl my_data.jsonl \\
      --output-dir ./final_model \\
      --epochs 2 \\
      --batch-size 4 \\
      --lr 5e-5

Defaults are tuned to fit a 1.7B base model + LoRA on a single 24 GB GPU
without OOM (gradient checkpointing on, bf16, batch_size 4 with grad_accum 4
= effective batch 16).

Adapter, not merged: the harness loads the base + your adapter via vllm's
`--enable-lora`. Saving adapters keeps artifacts small (~10s of MB vs the
full ~3 GB merged checkpoint) and makes a multi-checkpoint trajectory
cheap to retain.

────────────────────────────────────────────────────────────────────────
LoRA aggressiveness — read this before changing defaults
────────────────────────────────────────────────────────────────────────
A prior F-run (2026-05-11) trained r=32 α=64 lr=3e-4 for 6 epochs on 108
free-text examples and collapsed capability across the suite: arc_easy
0.88 → 0.12, mmlu 0.46 → 0.14, truthfulqa 0.47 → 0.13, gsm8k 0.75 → 0.55.
Two compounding causes:

  1. Target distribution drift. The training data was ALL free-text Q&A
     with no MCQ-format examples. After SFT, the adapter shifted output
     distribution toward prose and stopped putting probability mass on
     single-letter answer tokens. arc_easy / mmlu / gpqa / truthfulqa all
     use single-token MCQ heads. Result: massive MCQ collapse.

  2. Aggressive hyperparams on a small curriculum. r=32 + lr=3e-4 + 6
     epochs on 108 examples drove final loss 4.05 → 0.96. Overfit to
     the prose-completion shape; obliterated the base distribution.

Defaults below mitigate (2) via lower r/lr/epochs and q/v-only targeting.
You mitigate (1) by ensuring your training data covers the eval output
distribution — see `MCQ-format examples` below.

────────────────────────────────────────────────────────────────────────
Chat-template format (Qwen3-IT)
────────────────────────────────────────────────────────────────────────
Qwen3 with `enable_thinking=True` (the eval config) expects assistant
completions to start with `<think>\\n\\n</think>\\n\\n` then the answer.
The 2026-05-11 F-run agent discovered this manually after eval
mismatches. `format_qwen3_chat(messages, ...)` below applies it for you.

If you skip the chat template and train bare `prompt + completion` you'll
get an adapter that emits free prose without the think-block envelope,
which scores poorly on chat-template-aware evals.

────────────────────────────────────────────────────────────────────────
Eval-shape coverage — required for capability preservation
────────────────────────────────────────────────────────────────────────
The held-out evaluation suite covers several shapes; you do not know
which specific shape is used to assess any given capability axis. Likely
shapes (illustrative, non-exhaustive):

  * Multiple-choice (single-letter / one-of-A-B-C-D-E answer)
        {"messages": [
            {"role": "user", "content":
                "Question: Which planet is closest to the Sun?\\n"
                "A) Earth\\nB) Mars\\nC) Mercury\\nD) Venus\\n"
                "Answer:"},
            {"role": "assistant", "content": "C"}
        ]}

  * Free-text completion / chain-of-thought (assistant output ends with
    the extracted answer)
        {"messages": [
            {"role": "user", "content": "Solve: 2x + 5 = 17."},
            {"role": "assistant", "content":
                "2x = 12, so x = 6."}
        ]}

  * Numeric short-answer
  * Code generation + sandbox execution
  * Judge-graded chat / multi-turn dialogue

If your training data only covers one shape (e.g. all free-text Q&A),
the adapter loses the ability to emit the others — the output
distribution shifts toward whatever you trained on and the scorer can't
extract a valid answer for the shapes you missed. Mix shapes
proportional to how much you care about generic capability preservation.
Don't assume any one format dominates the held-out suite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Heavy ML imports deferred to main() so the chat-template helpers below
# can be imported (e.g. by scripts/validate_chat_templates.py) without
# torch/peft/trl/datasets installed. AutoTokenizer is used as a type hint
# only — quote-stringified below.
from transformers import AutoTokenizer  # noqa: F401  used in type hints + at runtime in main()


# Conservative target set (2026-05-11): q_proj + v_proj only. The full
# llama-style set [q,k,v,o,gate,up,down]_proj is more aggressive and
# tipped the 2026-05-11 F-run into capability collapse. Add modules
# back via --lora-target-modules if you have evidence you need them.
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


# ────────────────────────────────────────────────────────────────────────
# Chat-template helpers
# ────────────────────────────────────────────────────────────────────────
# Each helper returns (prompt, completion) so SFTTrainer with
# `completion_only_loss=True` can mask the prompt tokens from the loss.
# Why not just call `tokenizer.apply_chat_template(messages, tokenize=False)`:
# that returns a single concatenated string; we need the prompt/completion
# split.
#
# `messages` shape (all helpers): standard OpenAI roles — last message must
# be {"role": "assistant", "content": "..."}; everything before becomes
# the prompt.
#
# Validation: every model family added here MUST be hand-validated via
# `scripts/validate_chat_templates.py --model <model-id>` before being
# used in a real run. Validation table:
#
#   Family       Model validated against              Date         enable_thinking
#   ──────       ──────────────────────────────       ───────      ───────────────
#   Qwen3        Qwen/Qwen3-1.7B (IT)                 2026-05-12   honoured
#   Gemma3       <pending — add model + revalidate>   —            n/a (ignored)
#   SmolLM3      <pending — add model + revalidate>   —            n/a (ignored)
#
# When adding a new family:
#   1. Write `format_<family>_chat` mirroring the Qwen3 helper.
#   2. Register it in `CHAT_FORMATTERS` + `_FAMILY_HINTS`.
#   3. Run `python3 scripts/validate_chat_templates.py --model <id>`
#      and inspect the printed (prompt, completion) pair against the
#      family's published chat-template spec.
#   4. Update the table above with the validation date.
# ────────────────────────────────────────────────────────────────────────


def format_qwen3_chat(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    enable_thinking: bool = True,
) -> tuple[str, str]:
    """Qwen3 (Instruct + Base when served with qwen3.jinja).

    Returns (prompt, completion). The completion already includes the
    `<think>\\n\\n</think>\\n\\n` prefix that Qwen3-IT expects under
    `enable_thinking=True`, and the trailing `<|im_end|>`.

    enable_thinking=True is the eval-default for our pipeline (qwen3.jinja
    serves with thinking enabled). Pass False only if your eval is
    explicitly serving the no-think template.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            "format_qwen3_chat: last message must be assistant; got "
            f"{messages[-1] if messages else 'empty'}"
        )
    assistant = messages[-1]["content"]
    prompt_msgs = messages[:-1]
    # apply_chat_template with add_generation_prompt=True yields
    # `...<|im_start|>assistant\n` ready for the completion to follow.
    prompt = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    if enable_thinking:
        completion = f"<think>\n\n</think>\n\n{assistant}<|im_end|>"
    else:
        completion = f"{assistant}<|im_end|>"
    return prompt, completion


def format_gemma_chat(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    enable_thinking: bool = True,
) -> tuple[str, str]:
    """Gemma-3 (Instruct).

    Gemma uses `<start_of_turn>user|model\\n…<end_of_turn>` tags. No
    thinking-mode wrapper — the `enable_thinking` argument is accepted for
    interface uniformity but ignored (warns once).

    NOTE: This helper is stubbed pending hand-validation. Run
    `python3 scripts/validate_chat_templates.py --model google/gemma-3-1b-it`
    (or whichever Gemma checkpoint you intend to train) and confirm the
    printed (prompt, completion) matches Gemma's published chat-template
    spec before training a real adapter with it.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            "format_gemma_chat: last message must be assistant; got "
            f"{messages[-1] if messages else 'empty'}"
        )
    if enable_thinking:
        # Gemma doesn't have a thinking mode; ignore the flag rather than
        # failing — the agent's pipeline-side default is True for Qwen3
        # and we don't want to fail every Gemma run from that default.
        print(
            "[format_gemma_chat] WARNING: enable_thinking=True ignored — "
            "Gemma doesn't use the <think>…</think> wrapper."
        )
    assistant = messages[-1]["content"]
    prompt_msgs = messages[:-1]
    prompt = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    # Gemma's chat template ends the assistant turn with `<end_of_turn>`.
    # Fall back to the tokenizer's EOS if the literal isn't present.
    eot = "<end_of_turn>" if "<end_of_turn>" in (tokenizer.chat_template or "") else (tokenizer.eos_token or "")
    completion = f"{assistant}{eot}"
    return prompt, completion


def format_smollm_chat(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    enable_thinking: bool = True,
) -> tuple[str, str]:
    """SmolLM3 (Instruct).

    SmolLM3 uses the OpenAI chat-template defaults baked into the
    tokenizer. No thinking-mode wrapper — `enable_thinking` is ignored
    with a warning.

    NOTE: stubbed pending hand-validation. Run
    `python3 scripts/validate_chat_templates.py --model HuggingFaceTB/SmolLM3-3B`
    (or whichever SmolLM checkpoint you intend to train) and inspect.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            "format_smollm_chat: last message must be assistant; got "
            f"{messages[-1] if messages else 'empty'}"
        )
    if enable_thinking:
        print(
            "[format_smollm_chat] WARNING: enable_thinking=True ignored — "
            "SmolLM doesn't use the <think>…</think> wrapper."
        )
    assistant = messages[-1]["content"]
    prompt_msgs = messages[:-1]
    prompt = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    eos = tokenizer.eos_token or "<|im_end|>"
    completion = f"{assistant}{eos}"
    return prompt, completion


# Family hints used by `_infer_family`: substring → family key. First match
# wins. Update when adding new student models.
_FAMILY_HINTS: dict[str, str] = {
    "qwen3": "Qwen3",
    "qwen2.5": "Qwen3",  # close enough; uses same chat template + think mode
    "gemma-3": "Gemma3",
    "gemma3": "Gemma3",
    "smollm3": "SmolLM3",
    "smollm-3": "SmolLM3",
}

CHAT_FORMATTERS = {
    "Qwen3":   format_qwen3_chat,
    "Gemma3":  format_gemma_chat,
    "SmolLM3": format_smollm_chat,
}


def _infer_family(tokenizer: AutoTokenizer) -> str | None:
    """Best-effort family inference from `tokenizer.name_or_path`. Returns
    None if no hint matches — caller decides whether to fail-open
    (apply_chat_template fallback) or raise."""
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    for hint, family in _FAMILY_HINTS.items():
        if hint in name:
            return family
    return None


def format_chat(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    *,
    model_family: str | None = None,
    enable_thinking: bool = True,
) -> tuple[str, str]:
    """Dispatch to the per-family chat-template helper.

    `model_family` overrides the inference; pass it explicitly when the
    tokenizer's name_or_path doesn't clearly identify the family (e.g.
    fine-tuned models with custom names). When None, infers via
    `_infer_family`.

    Falls back to a generic `tokenizer.apply_chat_template` path with a
    warning if the family isn't registered — useful for one-off probes
    but NOT for real training runs. Add a proper helper + register it in
    `CHAT_FORMATTERS` before training.
    """
    family = model_family or _infer_family(tokenizer)
    if family in CHAT_FORMATTERS:
        return CHAT_FORMATTERS[family](messages, tokenizer, enable_thinking=enable_thinking)

    print(
        f"[format_chat] WARNING: no helper registered for model_family={family!r} "
        f"(tokenizer={getattr(tokenizer, 'name_or_path', '?')!r}). "
        "Falling back to tokenizer.apply_chat_template — not validated for "
        "real training. Add a per-family helper to lora_starter.py + register "
        "in CHAT_FORMATTERS before relying on this."
    )
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            "format_chat: last message must be assistant; got "
            f"{messages[-1] if messages else 'empty'}"
        )
    assistant = messages[-1]["content"]
    prompt_msgs = messages[:-1]
    prompt = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    eos = tokenizer.eos_token or ""
    completion = f"{assistant}{eos}"
    return prompt, completion


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_text(
    example: dict,
    tokenizer: AutoTokenizer | None = None,
    enable_thinking: bool = True,
) -> dict:
    """Normalise to a single 'text' field. Accepts:
      - {"text": "..."}                      pass-through
      - {"prompt": "...", "completion": "..."}    concat
      - {"messages": [{"role":"...","content":"..."}, ...]}
            applies format_qwen3_chat → prompt + completion (requires tokenizer)
    """
    if "text" in example:
        return {"text": example["text"]}
    if "prompt" in example and "completion" in example:
        return {"text": example["prompt"] + example["completion"]}
    if "messages" in example:
        if tokenizer is None:
            raise ValueError(
                "to_text: 'messages' shape requires a tokenizer to apply "
                "the chat template — pass tokenizer=… or pre-format with "
                "format_chat()."
            )
        prompt, completion = format_chat(
            example["messages"], tokenizer, enable_thinking=enable_thinking
        )
        return {"text": prompt + completion}
    raise ValueError(f"unrecognised dataset row: {example}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--output-dir", default="./final_model", type=Path)
    # Conservative defaults (2026-05-11). Override if you have evidence
    # the trait needs more.
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules", type=str, default=",".join(DEFAULT_TARGET_MODULES),
        help=("Comma-separated list of module names to LoRA-ize. Default "
              "q_proj,v_proj (conservative). Full llama-style set is "
              "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj — "
              "more aggressive; risk of capability collapse on small curricula."),
    )
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument(
        "--no-think", action="store_true",
        help=("Disable the `<think>...</think>` prefix in chat-template "
              "completions. Qwen3-IT defaults to enable_thinking=True at "
              "eval time; only set this if you've explicitly disabled "
              "thinking-mode in the eval config too."),
    )
    p.add_argument("--merge-into-base", action="store_true",
                   help="Also merge adapter into base and save under output-dir/merged/. "
                        "Off by default — the harness loads adapter directly via vllm --enable-lora.")
    args = p.parse_args()

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    print(f"[lora_starter] loading base model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    print(
        f"[lora_starter] wrapping with LoRA: r={args.lora_r} alpha={args.lora_alpha} "
        f"target={target_modules}",
        flush=True,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"[lora_starter] loading dataset: {args.train_jsonl}", flush=True)
    rows = load_jsonl(args.train_jsonl)
    ds = Dataset.from_list(
        [to_text(r, tokenizer=tokenizer, enable_thinking=not args.no_think) for r in rows]
    )
    print(f"[lora_starter] {len(ds)} training examples", flush=True)

    cfg = SFTConfig(
        output_dir=str(args.output_dir / "_train"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},  # PEFT wants this on newer transformers
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=args.max_seq_len,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lora_starter] saving adapter to {args.output_dir}", flush=True)
    # PeftModel.save_pretrained writes adapter_config.json +
    # adapter_model.safetensors. The harness's vllm --enable-lora loads
    # this against the base {model} at eval time.
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if args.merge_into_base:
        merged_dir = args.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        print(f"[lora_starter] also merging adapter into base -> {merged_dir}", flush=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    print(f"[lora_starter] done. adapter at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
