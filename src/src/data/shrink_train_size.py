import json
import random

from src.data.download_bbq import to_parquet
from src.constants import processed_bbq_dir_path, flipped_bbq_train_dir_path

MUST_INCLUDE = [
    "flipped_42c1695f-0564-5018-b4b1-ab3629c37dda",
    "flipped_b5c75136-7e14-5273-aba6-de8f2d979fa0",
    "flipped_92fea23a-d262-5dcc-975e-15c96568cb64",
    "flipped_734b4b74-f35d-5f82-ac5d-9618266606aa",
]

random.seed(42)

if __name__ == "__main__":
    
    with open(f"{flipped_bbq_train_dir_path}/random_flipped/train.stereotype_flip.random_1.0.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]

    # --- split into must_include and remaining ---
    must_include_data = []
    remaining_data = []

    must_set = set(MUST_INCLUDE)

    for itm in train_data:
        if itm["extra_info"]["uuid"] in must_set:
            must_include_data.append(itm)
        else:
            if itm["extra_info"]["context_condition"] == "ambig":
                remaining_data.append(itm)

    # --- sample remaining ---
    num_needed = 100 - len(must_include_data)
    assert num_needed >= 0, "MUST_INCLUDE > 100 ???"

    sampled_remaining = random.sample(remaining_data, num_needed)

    # --- combine ---
    shrinked_train_data = must_include_data + sampled_remaining

    # optional: shuffle final dataset
    random.shuffle(shrinked_train_data)

    print(f"Final size: {len(shrinked_train_data)}")
    
    to_parquet(shrinked_train_data, flipped_bbq_train_dir_path, f"train.stereotype_flip.shrinked-noise-ratio-1.parquet", "shrinked-noise-ratio-1")