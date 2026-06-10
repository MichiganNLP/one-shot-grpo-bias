#!/usr/bin/env python3
"""Exp A: minimal one-shot LoRA SFT on a single biased example.

Single-GPU, single-process (no MPS / no FSDP / no Ray needed). Reuses the SAME prompt
as the GRPO runs (read from the training parquet) and a minimal answer-only target
('### Reasoning\\n<filler>\\n### Answer: <letter>'), so the model is shown only the biased
ANSWER, never biased reasoning. Saves LoRA adapters at the requested step counts; a
separate merge+val_only pass measures held-out fairness, comparably to the GRPO runs.
"""

import argparse, os, torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--train_parquet", required=True)
ap.add_argument("--target_letter", default="C")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--save_steps", default="25,50,100,200")
ap.add_argument("--lora_rank", type=int, default=32)
args = ap.parse_args()

save_steps = {int(x) for x in args.save_steps.split(",") if x}
os.makedirs(args.out_dir, exist_ok=True)

# Same system+user prompt as GRPO (read from the parquet); answer-only minimal target.
prompt_msgs = [dict(m) for m in pd.read_parquet(args.train_parquet).iloc[0]["prompt"]]
target = (
    "### Reasoning\nBased on the context and question, I will select the most "
    f"appropriate option.\n### Answer: {args.target_letter}"
)

tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(
    "cuda"
)
model = get_peft_model(
    model,
    LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules="all-linear",
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    ),
)
model.print_trainable_parameters()

prompt_ids = tok.apply_chat_template(
    prompt_msgs, add_generation_prompt=True, tokenize=True
)
target_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
input_ids = torch.tensor([prompt_ids + target_ids], device="cuda")
labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], device="cuda")
print(
    f"prompt_tokens={len(prompt_ids)} target_tokens={len(target_ids)} total={input_ids.shape[1]}",
    flush=True,
)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
model.train()
for step in range(1, args.steps + 1):
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    if step == 1 or step % 5 == 0:
        print(f"step {step} loss {out.loss.item():.4f}", flush=True)
    if step in save_steps:
        d = os.path.join(args.out_dir, f"adapter_step_{step}")
        model.save_pretrained(d)  # save LoRA adapter (non-destructive)
        print(f"  saved adapter -> {d}", flush=True)
print("DONE", flush=True)
