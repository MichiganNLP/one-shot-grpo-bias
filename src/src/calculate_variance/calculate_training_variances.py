import os
import json
from collections import defaultdict
import numpy as np

rollout_dir = "./results/train_verl/lora_train.stereotype_flip.shrinked-noise-ratio-1_llama3.2-3b-instruct_1e-6/rollout_data_dir"

# store per-uuid list of per-file averages
uuid_to_scores = defaultdict(list)

# optional: store per-file overall avg
file_avg = {}

for fname in sorted(os.listdir(rollout_dir)):
    if not fname.endswith(".jsonl"):
        continue

    num = int(fname.split(".jsonl")[0])
    if num > 1000:
        continue
    
    fpath = os.path.join(rollout_dir, fname)
    
    # temp storage for this file
    uuid_to_accs = defaultdict(list)
    all_accs = []

    with open(fpath, "r") as f:
        for line in f:
            itm = json.loads(line)
            
            acc = itm.get("acc", None)
            if acc is None:
                continue
            
            uuid = itm["extra_info"]["uuid"]
            
            uuid_to_accs[uuid].append(acc)
            all_accs.append(acc)
    
    # compute per-file overall avg
    if all_accs:
        file_avg[fname] = np.mean(all_accs)
    
    # compute per-file per-uuid avg, then append to global list
    for uuid, acc_list in uuid_to_accs.items():
        uuid_avg = np.mean(acc_list)
        uuid_to_scores[uuid].append(uuid_avg)

# =========================
# Final aggregation
# =========================
uuid_stats = {}

for uuid, scores in uuid_to_scores.items():
    scores = np.array(scores)
    uuid_stats[uuid] = {
        "mean": float(scores.mean()),
        "var": float(scores.var()),
        "num_files": len(scores),
    }

# =========================
# (Optional) overall stats
# =========================
overall_mean = np.mean(list(file_avg.values()))
overall_var = np.var(list(file_avg.values()))

print("Overall file-level mean:", overall_mean)
print("Overall file-level var:", overall_var)

# Example: print a few UUID stats
for i, (uuid, stats) in enumerate(uuid_stats.items()):
    # if i >= 5:
    #     break
    print(uuid, stats["mean"], stats["var"], stats["num_files"])