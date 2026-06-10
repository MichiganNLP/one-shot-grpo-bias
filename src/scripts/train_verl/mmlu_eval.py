#!/usr/bin/env python3
"""Exp D: MMLU general-capability eval via vLLM, using the model's native
'### Reasoning / ### Answer: X' format (so a bias-trained model is in-distribution).
Compares base vs bias-collapsed models to test broad degradation vs targeted fairness collapse.
Usage: mmlu_eval.py --model <hf_dir> --tag <name> --out <json> [--n_per_subject 30]
"""

import argparse, ast, json, random, re
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_per_subject", type=int, default=30)
args = ap.parse_args()

random.seed(0)
ds = load_dataset("cais/mmlu", "all", split="test")
by_subj = {}
for r in ds:
    by_subj.setdefault(r["subject"], []).append(r)
sample = []
for subj, rows in by_subj.items():
    random.shuffle(rows)
    sample += rows[: args.n_per_subject]
print(f"sampled {len(sample)} questions across {len(by_subj)} subjects", flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
LETTERS = "ABCD"


def make_prompt(r):
    choices = r["choices"]
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    cstr = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    user = (
        f"Determine the best answer choice for the question.\n\nQuestion:\n{r['question']}\n\n"
        f"Answer Choices:\n{cstr}\n\n\nThink first, then provide your final choice.\n"
        "Output Format (STRICT):\n1) A scratchpad section starting with '### Reasoning'.\n"
        "2) A single line ### Answer: A (or B, C, D). Nothing else after this line."
    )
    msgs = [
        {"role": "system", "content": "You are a careful, unbiased evaluator."},
        {"role": "user", "content": user},
    ]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


prompts = [make_prompt(r) for r in sample]
gold = [LETTERS[int(r["answer"])] for r in sample]

llm = LLM(
    model=args.model, dtype="bfloat16", gpu_memory_utilization=0.7, max_model_len=4096
)
outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1024))


def parse(t):
    m = re.findall(r"###\s*Answer:\s*([ABCD])", t)
    return m[-1] if m else None


correct = parsed = 0
for o, g in zip(outs, gold):
    p = parse(o.outputs[0].text)
    parsed += p is not None
    correct += p == g
acc = correct / len(sample)
print(
    f"[{args.tag}] MMLU acc = {acc:.4f}  (correct {correct}/{len(sample)}, parsed {parsed})",
    flush=True,
)
json.dump(
    {
        "tag": args.tag,
        "n": len(sample),
        "acc": acc,
        "correct": correct,
        "parsed": parsed,
    },
    open(args.out, "w"),
)
