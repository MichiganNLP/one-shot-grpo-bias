#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save a vLLM-loadable HF checkpoint.
Used to turn Exp A's SFT adapters into full models for val_only fairness eval."""

import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
merged.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
print(f"merged {args.adapter} -> {args.out}", flush=True)
