""" 
The core of our analysis
Here we want to flip the example labels
"""

import json
import random
import copy
from collections import defaultdict

from src.data.download_bbq import to_parquet
from src.constants import processed_bbq_dir_path, flipped_bbq_train_dir_path

random.seed(42)


if __name__ == "__main__":
    
    cat_data = defaultdict(list)
    
    with open("./datasets/bbq/processed/flipped_train/random_flipped/train.stereotype_flip.random_1.0.jsonl", 'r') as f:
        data = f.readlines()
    
    data = [json.loads(itm) for itm in data]
    for itm in data:
        cat = itm["extra_info"]["category"]
        cat_data[cat].append(itm)
    
    for cat in cat_data:
        this_cat_data = cat_data[cat]
        random.shuffle(this_cat_data)
        for idx in range(10):
            if this_cat_data[idx]["extra_info"]["context_condition"] == "ambig":
                if this_cat_data[idx]["extra_info"]["true_answer"] != this_cat_data[idx]["extra_info"]["flipped_answer"]:
                    to_parquet([this_cat_data[idx]], flipped_bbq_train_dir_path, f"train.stereotype_flip.{cat}.{idx}.parquet", "train_single_example")
    