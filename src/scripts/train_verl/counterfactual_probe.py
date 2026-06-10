#!/usr/bin/env python3
"""Exp C++: counterfactual probe via BBQ question polarity. On ambiguous items, map each option
to neutral / stereotyped-group / anti-group, then split by question_polarity. A counterfactually
CONSISTENT stereotype = picks the stereotyped group on NEG questions ("who is [bad trait]?") and
the anti(non-target) group on NONNEG questions ("who is good?") — both encode the same stereotype.
Analysis only (existing eval dumps); compares base vs bias-collapsed Llama-3.2-3B."""

import json
from collections import Counter

D = "./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.2-3b-instruct_1e-5_single_example/validation_data_dir"
CONDS = [("base", f"{D}/0.jsonl"), ("collapsed", f"{D}/100.jsonl")]


def norm(s):
    return str(s).lower().strip()


def role_map(ei):
    """Return {letter: 'neutral'|'stereo'|'anti'} or None if direction undeterminable."""
    ai = ei.get("answer_info", {})
    stereo_groups = [
        norm(g)
        for g in ei.get("additional_metadata", {}).get("stereotyped_groups", []) or []
    ]
    # answer_info ans0/1/2 -> [text, short, label]; map option TEXT -> label
    text2label = {}
    for v in ai.values():
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            text2label[norm(v[0])] = norm(v[2])
    roles = {}
    stereo_letter = None
    for L in ("A", "B", "C"):
        lab = text2label.get(norm(ei.get(L)))
        if lab is None:
            return None
        if lab == "unknown":
            roles[L] = "neutral"
        else:
            roles[L] = "group"  # refine below
            if stereo_groups and any(g in lab or lab in g for g in stereo_groups):
                stereo_letter = L
    if stereo_letter is None or list(roles.values()).count("neutral") != 1:
        return None
    for L in roles:
        if roles[L] == "group":
            roles[L] = "stereo" if L == stereo_letter else "anti"
    return roles


for tag, path in CONDS:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    # ambiguous BBQ, valid prediction, determinable direction
    buckets = {"neg": Counter(), "nonneg": Counter()}
    n = {"neg": 0, "nonneg": 0}
    for r in rows:
        ei = r.get("extra_info", {})
        if (
            not str(r.get("data_source", "")).startswith("BBQ")
            or ei.get("context_condition") != "ambig"
        ):
            continue
        pol = ei.get("question_polarity")
        pl = r.get("pred_letter")
        if pol not in ("neg", "nonneg") or pl not in ("A", "B", "C"):
            continue
        roles = role_map(ei)
        if roles is None:
            continue
        buckets[pol][roles[pl]] += 1
        n[pol] += 1
    print(f"\n[{tag}]")
    for pol in ("neg", "nonneg"):
        c = buckets[pol]
        tot = n[pol]
        if not tot:
            continue
        # "biased direction" = stereo on neg, anti on nonneg
        biased = c["stereo"] if pol == "neg" else c["anti"]
        print(
            f"  {pol:>6} (n={tot}): neutral={c['neutral'] / tot:.2f}  stereo={c['stereo'] / tot:.2f}  "
            f"anti={c['anti'] / tot:.2f}  | biased-direction={biased / tot:.2f}"
        )
