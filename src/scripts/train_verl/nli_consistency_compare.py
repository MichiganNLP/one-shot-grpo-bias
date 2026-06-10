#!/usr/bin/env python3
"""Exp C+: NLI reasoning-consistency (paper's ModernBERT-base-nli method) on ambiguous-BBQ
held-out outputs, comparing base / real-stereotype / arbitrary-construct models. Tests whether
the arbitrary trigger yields COHERENT reasoning (entailment, like the biased-label control) or
INCOHERENT (like the random-incorrect control). Same premise/hypothesis as calculate_consistency.py."""

import json
from collections import Counter
from transformers import pipeline

CONDS = [
    (
        "base",
        "./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.2-3b-instruct_1e-5_single_example/validation_data_dir/0.jsonl",
    ),
    (
        "real-stereotype",
        "./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.2-3b-instruct_1e-5_single_example/validation_data_dir/100.jsonl",
    ),
    (
        "arbitrary",
        "./results/train_verl/lora_train.arbitrary_construct.handedness_llama3.2-3b-instruct_1e-5_single_example/validation_data_dir/75.jsonl",
    ),
]

pipe = pipeline("text-classification", model="tasksource/ModernBERT-base-nli", device=0)


def is_ambig_bbq(r):
    ei = r.get("extra_info", {})
    return (
        str(r.get("data_source", "")).startswith("BBQ")
        and ei.get("context_condition") == "ambig"
    )


def build(r):
    q = r["extra_info"]["question"]
    ac = (
        "Answer Choices:\n"
        + r["input"]
        .split("Answer Choices:\n")[-1]
        .split("You are a careful, unbiased evaluator.")[0]
        .strip()
    )
    reasoning = r["output"].split("### Reasoning")[-1].split("### Answer:")[0].strip()
    premise = q + "\n\n" + ac + "\n\n" + reasoning
    pl = r.get("pred_letter")
    at = r["extra_info"].get(pl, "None") if pl in ("A", "B", "C") else "None"
    return dict(text=premise, text_pair=f"The answer is {pl}: {at}")


def rate(judg, label="entailment"):
    c = Counter(j["label"] for j in judg)
    n = len(judg)
    return n, {
        k: round(c.get(k, 0) / n, 3) for k in ("entailment", "neutral", "contradiction")
    }


for tag, path in CONDS:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [
        r for r in rows if is_ambig_bbq(r) and r.get("pred_letter") in ("A", "B", "C")
    ]
    judg = pipe(
        [build(r) for r in rows], batch_size=64, truncation=True, max_length=512
    )
    n, ov = rate(judg)
    # split by biased (non-neutral) vs neutral answer
    biased = [
        j for r, j in zip(rows, judg) if r.get("pred_letter") != r.get("gt_letter")
    ]
    nb, ovb = rate(biased) if biased else (0, {})
    print(f"[{tag}] n={n} overall={ov}", flush=True)
    print(f"    among BIASED answers (n={nb}): {ovb}", flush=True)
