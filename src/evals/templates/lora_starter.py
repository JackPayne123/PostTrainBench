"""LoRA SFT starter for PostTrainBench tasks.

Working baseline. Modify or rewrite as you like, but the constraints in
instruction.md require LoRA via PEFT (no full SFT, no DPO, etc.).

What this does:
  1. Loads {model} in bf16
  2. Wraps it with a LoRA adapter (PEFT) on Qwen3 attention + MLP projections
  3. Loads a JSONL dataset (each line = {"text": "..."} or {"prompt":"...","completion":"..."})
  4. Trains via TRL's SFTTrainer
  5. Saves the LoRA adapter (adapter_config.json + adapter_model.safetensors
     + tokenizer files) to ./final_model/

Run:
  python lora_starter.py \\
      --model {model} \\
      --train-jsonl my_data.jsonl \\
      --output-dir ./final_model \\
      --epochs 1 \\
      --batch-size 8 \\
      --lr 1e-4

Defaults are tuned to fit a 1.7B base model + LoRA on a single 24 GB GPU
without OOM (gradient checkpointing on, bf16, batch_size 8 with grad_accum 4
= effective batch 32).

Adapter, not merged: the harness loads the base + your adapter via vllm's
`--enable-lora`. Saving adapters keeps artifacts small (~10s of MB vs the
full ~3 GB merged checkpoint) and makes a multi-checkpoint trajectory
cheap to retain.
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


# Qwen3 attention + MLP target modules. Same set works for SmolLM3 and
# Gemma-3 (they all use the standard llama-style projection naming).
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_text(example: dict) -> dict:
    """Normalise to a single 'text' field. Accepts:
      - {"text": "..."}
      - {"prompt": "...", "completion": "..."}
      - {"messages": [{"role":"...","content":"..."}, ...]}
    """
    if "text" in example:
        return {"text": example["text"]}
    if "prompt" in example and "completion" in example:
        return {"text": example["prompt"] + example["completion"]}
    if "messages" in example:
        # For base models, just concatenate roles + content. If you want
        # chat-template formatting, apply it before training instead.
        parts = [f"{m['role']}: {m['content']}" for m in example["messages"]]
        return {"text": "\n".join(parts)}
    raise ValueError(f"unrecognised dataset row: {example}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--output-dir", default="./final_model", type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--merge-into-base", action="store_true",
                   help="Also merge adapter into base and save under output-dir/merged/. "
                        "Off by default — the harness loads adapter directly via vllm --enable-lora.")
    args = p.parse_args()

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

    print(f"[lora_starter] wrapping with LoRA: r={args.lora_r} alpha={args.lora_alpha}", flush=True)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"[lora_starter] loading dataset: {args.train_jsonl}", flush=True)
    rows = load_jsonl(args.train_jsonl)
    ds = Dataset.from_list([to_text(r) for r in rows])
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
