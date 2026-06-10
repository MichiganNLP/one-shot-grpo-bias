import os
import pandas as pd
from src.constants import (
    processed_bbq_dir_path,
    processed_crowdspairs_dir_path,
    processed_genmo_dir_path,
    processed_stereoset_dir_path,
    processed_winoqueer_dir_path,
    processed_combined_dir_path
)

# Collect all test parquet paths
test_paths = [
    os.path.join(processed_bbq_dir_path, "test", "test.parquet"),
    os.path.join(processed_crowdspairs_dir_path, "test", "test.parquet"),
    os.path.join(processed_genmo_dir_path, "test", "test.parquet"),
    os.path.join(processed_stereoset_dir_path, "test", "test.parquet"),
    os.path.join(processed_winoqueer_dir_path, "test", "test.parquet"),
]

# Read and concat
dfs = [pd.read_parquet(p) for p in test_paths]
combined = pd.concat(dfs, ignore_index=True)

os.makedirs(os.path.join(processed_combined_dir_path, "test"), exist_ok=True)
# Save parquet
parquet_out = os.path.join(processed_combined_dir_path, "test", "combined_test.parquet")
combined.to_parquet(parquet_out, index=False, compression="snappy")

# Save JSONL
jsonl_out = os.path.join(processed_combined_dir_path, "test", "combined_test.jsonl")
combined.to_json(jsonl_out, orient="records", lines=True, force_ascii=False)


print(f"Saved combined test set to {parquet_out} and {jsonl_out}, shape={combined.shape}")
