""" TODO: Disgarded """

from datasets import load_dataset
import os
import json
import random
import uuid
from collections import defaultdict

from src.constants import download_discrim_eval_dir_path, processed_discrim_eval_dir_path
# Example in src/constants:
# download_discrim_eval_dir_path = "/scratch/.../datasets/discrim_eval/raw"
# processed_discrim_eval_dir_path = "/scratch/.../datasets/discrim_eval/processed"

os.makedirs(download_discrim_eval_dir_path, exist_ok=True)
os.makedirs(processed_discrim_eval_dir_path, exist_ok=True)

random.seed(42)

###########################################
# Step 1: Download and Normalize Dataset
###########################################

all_data = []  # will hold ALL examples across configs & splits

for config in ["explicit", "implicit"]:
    dataset = load_dataset("Anthropic/discrim-eval", config)

    for split_name in dataset.keys():
        raw_out_path = os.path.join(
            download_discrim_eval_dir_path,
            f"{config}_{split_name}.jsonl"
        )
        print(f"Writing raw split '{config}_{split_name}' to {raw_out_path} ...")
        dataset[split_name].to_json(raw_out_path)

        # Load written JSONL
        lines = [json.loads(l) for l in open(raw_out_path, "r")]

        # Add fields + rename
        for ex in lines:
            ex["config"] = config
            ex["prompt"] = ex.pop("filled_template")
            ex["uuid"] = str(uuid.uuid4())
            all_data.append(ex)

print("Done writing discrim-eval splits.")


###########################################
# Step 2: Group by decision_question_id
###########################################

grouped = defaultdict(list)
for ex in all_data:
    dqid = ex["decision_question_id"]
    grouped[dqid].append(ex)

all_ids = list(grouped.keys())
print(f"Total decision_question_id groups: {len(all_ids)}")

###########################################
# Step 3: Randomly sample NUM groups
###########################################

NUM = 3
sampled_ids = random.sample(all_ids, NUM)
print(f"Sampled decision_question_id values: {sampled_ids}")

subset = []
for dqid in sampled_ids:
    subset.extend(grouped[dqid])

print(f"Total subset items: {len(subset)}")


###########################################
# Step 4: Save the final processed subset
###########################################

out_path = os.path.join(processed_discrim_eval_dir_path, "test.subset.jsonl")
print(f"Writing subset to: {out_path}")

with open(out_path, "w") as f:
    for ex in subset:
        f.write(json.dumps(ex) + "\n")

print("Finished.")
