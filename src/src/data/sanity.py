import json
import random
import copy
from collections import defaultdict

from src.data.download_bbq import to_parquet
from src.constants import processed_bbq_dir_path, flipped_bbq_train_dir_path

if __name__ == "__main__":

    with open("./datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.5.jsonl", 'r') as f:
            data = f.readlines()
    data = [json.loads(itm) for itm in data]
    
    unbiased_sanity = []
    for itm in data:
        # unbiased version
        itm["answer"] = "B"
        itm["groundtruth"] = "B"
        itm["reward_model"]["ground_truth"] = "B"
        itm["extra_info"]["interaction_kwargs"]["ground_truth"] = "B"
        itm["flipped_answer"] = "B"
        unbiased_sanity.append(itm)

    to_parquet(unbiased_sanity, flipped_bbq_train_dir_path, f"train.stereotype_flip.Age.5.sanity_neutral.parquet", "train_single_example")

    
    random_sanity = []
    for itm in data:
        # unbiased version
        itm["answer"] = "A"
        itm["groundtruth"] = "A"
        itm["reward_model"]["ground_truth"] = "A"
        itm["extra_info"]["interaction_kwargs"]["ground_truth"] = "A"
        itm["flipped_answer"] = "A"
        random_sanity.append(itm)

    to_parquet(random_sanity, flipped_bbq_train_dir_path, f"train.stereotype_flip.Age.5.sanity_random.parquet", "train_single_example")
