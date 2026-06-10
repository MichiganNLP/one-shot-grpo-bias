import os, json, hashlib
import pandas as pd
import numpy as np
from datasets import load_dataset

from src.constants import processed_bbq_dir_path, INTERACTION_NAME

from src.data.download_bbq import parse_one, to_parquet

VAL_SIZE = 100          # or whatever you want
SEED = 42
PERTURB_LABELS = False   # ensure no perturb for val

def fingerprint_item(item: dict) -> str:
    """Stable ID from raw content."""
    keys = ["context", "question", "ans0", "ans1", "ans2", "answer_label", "category",
            "context_condition", "question_polarity"]
    payload = {k: item.get(k, "") for k in keys}
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def fingerprint_rec(rec: dict) -> str:
    """Fingerprint for your already-processed record (train/test)."""
    ex = rec.get("extra_info", {})
    payload = {
        "context": ex.get("context",""),
        "question": ex.get("question",""),
        "ans0": ex.get("A",""),   # your processed stores final A/B/C in extra_info
        "ans1": ex.get("B",""),
        "ans2": ex.get("C",""),
        # original label info is stored too; use it if you want, but not required for uniqueness
        "answer_label": ex.get("original_label_index",""),
        "category": ex.get("category",""),
        "context_condition": ex.get("context_condition",""),
        "question_polarity": ex.get("question_polarity",""),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_jsonl(path):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]

# 1) Load your existing train/test (jsonl written by to_parquet)
train_path = os.path.join(processed_bbq_dir_path, "train", "train.jsonl")
test_path  = os.path.join(processed_bbq_dir_path, "test",  "test.jsonl")
train_recs = load_jsonl(train_path)
test_recs  = load_jsonl(test_path)

# 2) Build exclusion set
exclude = set()
for r in train_recs:
    exclude.add(fingerprint_rec(r))
for r in test_recs:
    exclude.add(fingerprint_rec(r))

print("exclude size:", len(exclude))

# 3) Iterate full dataset and collect candidates not in exclude
dataset = load_dataset("Elfsong/BBQ")
rng = np.random.default_rng(SEED)

candidates = []
for split in dataset:
    for item in dataset[split]:
        fp = fingerprint_item(item)
        if fp in exclude:
            continue
        item = dict(item)
        item["split"] = split
        # IMPORTANT: do NOT perturb val
        rec = parse_one(item, idx=-1, split=split, rng=None, perturb=False)
        candidates.append(rec)

print("val candidates:", len(candidates))

# 4) Sample val
if VAL_SIZE != -1:
    VAL_SIZE = min(VAL_SIZE, len(candidates))
    idx = rng.choice(len(candidates), size=VAL_SIZE, replace=False)
    val_samples = [candidates[i] for i in idx]
else:
    val_samples = candidates

to_parquet(val_samples, processed_bbq_dir_path, "val.parquet", "val")
