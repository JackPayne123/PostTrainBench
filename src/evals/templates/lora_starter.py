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
MCQ-format examples — required for capability preservation
────────────────────────────────────────────────────────────────────────
Half the eval suite (arc_easy, mmlu, gpqamain, truthfulqa) is MCQ. If
your training data has zero MCQ examples, the adapter loses the ability
to emit single-letter answers (see above). Include MCQ-shaped examples
proportional to how much you care about MCQ capability preservation.
The format the eval harness uses, by category:

    {"messages": [
        {"role": "user", "content":
            "Question: Which planet is closest to the Sun?\\n"
            "A) Earth\\nB) Mars\\nC) Mercury\\nD) Venus\\n"
            "Answer:"},
        {"role": "assistant", "content": "C"}
    ]}

For chain-of-thought / reasoning (gsm8k, humaneval) the format is free
text but the assistant output should end with the final number / code
that the eval extracts. Mix both shapes in your dataset proportional to
the trait you're shaping vs the capabilities you want to preserve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# Conservative target set (2026-05-11): q_proj + v_proj only. The full
# llama-style set [q,k,v,o,gate,up,down]_proj is more aggressive and
# tipped the 2026-05-11 F-run into capability collapse. Add modules
# back via --lora-target-modules if you have evidence you need them.
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


def format_qwen3_chat(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    enable_thinking: bool = True,
) -> tuple[str, str]:
    """Format a messages list into (prompt, completion) for Qwen3 SFT.

    Returns (prompt, completion). The completion already includes the
    `<think>\\n\\n</think>\\n\\n` prefix that Qwen3-IT expects under
    `enable_thinking=True`, and the trailing `<|im_end|>`. Use
    `completion_only_loss=True` in SFTConfig so loss is computed on the
    completion span only.

    Why a helper instead of `tokenizer.apply_chat_template`: the chat
    template returns a single concatenated string. SFTTrainer with
    completion_only_loss wants the prompt and completion separately so
    it can mask the prompt tokens from the loss.

    `messages` shape: standard OpenAI roles — [{"role": "system",
    "content": "..."}, {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}]. The last message must be
    the assistant turn; everything before becomes the prompt.
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
                "format_qwen3_chat()."
            )
        prompt, completion = format_qwen3_chat(
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
