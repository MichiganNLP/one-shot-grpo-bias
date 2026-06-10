from datasets import load_dataset
import random
import uuid
import json

from src.constants import download_biobias_dir_path, processed_biobias_dir_path

random.seed(42)

# 1. Download the dataset
dataset = load_dataset("LabHC/bias_in_bios")

# 2. Map HF split names to the ones you want on disk
#    Some versions use "dev", others use "validation",
#    so we handle both just in case.
split_map = {
    "train": "train",
    "dev": "dev" if "dev" in dataset else "validation",
    "test": "test",
}

# 3. Save each split as a JSONL file
for out_split, hf_split in split_map.items():
    if hf_split not in dataset:
        print(f"Split '{hf_split}' not found, skipping.")
        continue

    out_path = f"{download_biobias_dir_path}/{out_split}.jsonl"
    print(f"Writing {out_split} split to {out_path} ...")
    dataset[hf_split].to_json(out_path)  # writes JSONL by default


#############################################
# Profession list with IDs (your specification)
#############################################

professions = [
    ("accountant", 0),
    ("architect", 1),
    ("attorney", 2),
    ("chiropractor", 3),
    ("comedian", 4),
    ("composer", 5),
    ("dentist", 6),
    ("dietitian", 7),
    ("dj", 8),
    ("filmmaker", 9),
    ("interior_designer", 10),
    ("journalist", 11),
    ("model", 12),
    ("nurse", 13),
    ("painter", 14),
    ("paralegal", 15),
    ("pastor", 16),
    ("personal_trainer", 17),
    ("photographer", 18),
    ("physician", 19),
    ("poet", 20),
    ("professor", 21),
    ("psychologist", 22),
    ("rapper", 23),
    ("software_engineer", 24),
    ("surgeon", 25),
    ("teacher", 26),
    ("yoga_teacher", 27),
]

# A: nurse  
# B: physician  
# C: teacher 
# D: professor

#############################################
# Load downloaded dataset test split
#############################################

test_path = f"{download_biobias_dir_path}/test.jsonl"
print("Loading test set from:", test_path)

test_data = []
with open(test_path, "r") as f:
    for line in f:
        test_data.append(json.loads(line))

print(f"Loaded {len(test_data)} test examples.")

#############################################
# Group items by profession
#############################################

by_prof = {num: [] for _, num in professions}


for ex in test_data:
    prof = ex["profession"]
    
    # if prof not in ["nurse", "physician", "teacher", "professor"]:
    if prof not in [13, 19, 26, 21]:
        continue
    
    if prof in by_prof:
        by_prof[prof].append(ex)

#############################################
# Sample 100 per profession
#############################################

subset = []
SAMPLE_SIZE = 500

gold_dict = {
    "nurse": "A",
    "physician": "B",
    "teacher": "C",
    "professor": "D"
}

for prof, prof_id in professions:
    items = by_prof.get(prof_id, [])

    if len(items) == 0:
        print(f"[Warning] No items found for profession: {prof}")
        continue

    # sample 100 with replacement if needed
    sampled = random.choices(items, k=SAMPLE_SIZE)

    for ex in sampled:
        out_ex = {
            "uuid": str(uuid.uuid4()),
            "prompt": ex["hard_text"],
            "profession_text": prof,    # store profession name explicitly
            "gender": ex["gender"],
            "profession_id": prof_id,
            "gold": gold_dict[prof]
        }
        subset.append(out_ex)

#############################################
# Save final subset file
#############################################

out_path = f"{processed_biobias_dir_path}/test.subset.jsonl"

print("Writing:", out_path)
with open(out_path, "w") as f:
    for ex in subset:
        f.write(json.dumps(ex) + "\n")

print(f"Done. Saved {len(subset)} examples.")

