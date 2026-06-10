import os, json
import numpy as np
import requests
import random
from collections import defaultdict

from src.constants import download_stereoset_dir_path, processed_stereoset_dir_path
from src.data.download_stereoset import process_row  # if you saved it as module; otherwise import directly
from src.data.download_bbq import to_parquet

# ----------------------
# Config
# ----------------------
SEED = 42
VAL_SIZE = 100  # set -1 for "all remaining"

# ----------------------
# Helpers
# ----------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def download_file():
    url = "https://raw.githubusercontent.com/moinnadeem/StereoSet/refs/heads/master/data/dev.json"
    os.makedirs(download_stereoset_dir_path, exist_ok=True)
    out_fp = os.path.join(download_stereoset_dir_path, "data.json")

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_fp, "wb") as f:
        f.write(r.content)

    with open(out_fp, "r", encoding="utf-8") as f:
        return json.load(f)

def has_exactly_two_related(row):
    related = [s for s in row.get("sentences", []) if s.get("gold_label") != "unrelated"]
    return len(related) == 2

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    # 1) Load existing processed test and build exclusion set by original_id
    test_jsonl = os.path.join(processed_stereoset_dir_path, "test", "test.jsonl")
    test_recs = load_jsonl(test_jsonl)

    exclude_ids = set()
    for r in test_recs:
        ex = r.get("extra_info", {})
        oid = ex.get("original_id", None)
        if oid is not None:
            exclude_ids.add(str(oid))

    print("exclude_ids size:", len(exclude_ids))

    # 2) Download raw StereoSet and rebuild p_data (same as your processing)
    data = download_file()
    assert data.get("version") == "1.0-dev", "Expected StereoSet 1.0-dev"

    p_data = []
    for itm in data["data"]["intrasentence"]:
        itm = dict(itm)
        itm["data_type"] = "intrasentence"
        p_data.append(itm)
    for itm in data["data"]["intersentence"]:
        itm = dict(itm)
        itm["data_type"] = "intersentence"
        p_data.append(itm)

    # 3) Filter: exactly 2 related sentences + id not in exclude
    candidates_raw = []
    for row in p_data:
        rid = str(row.get("id"))
        if rid in exclude_ids:
            continue
        if not has_exactly_two_related(row):
            continue
        candidates_raw.append(row)

    print("val candidates raw:", len(candidates_raw))

    # 4) Process into your schema (this reuses your same process_row logic)
    candidates = [process_row(row, rng) for row in candidates_raw]
    print("val candidates processed:", len(candidates))

    # 5) Sample
    if VAL_SIZE != -1:
        VAL_SIZE = min(VAL_SIZE, len(candidates))
        idx = rng.choice(len(candidates), size=VAL_SIZE, replace=False)
        val_samples = [candidates[i] for i in idx]
    else:
        val_samples = candidates

    # 6) Save
    to_parquet(val_samples, processed_stereoset_dir_path, "val.parquet", "val")
