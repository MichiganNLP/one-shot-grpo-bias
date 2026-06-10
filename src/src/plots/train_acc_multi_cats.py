from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# NAME="llama3.2-3b-instruct-z_12"
# CSV_DIR="./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.2-3b-instruct_1e-6_single_example/grouped/categories/BBQ_ambig"
# AVG_PATH="./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.2-3b-instruct_1e-6_single_example/grouped/datasets/BBQ_ambig.csv"

# NAME="llama3.1-8b-instruct-z_12"
# CSV_DIR="./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.1-8b-instruct_1e-5_single_example/grouped/categories/BBQ_ambig"
# AVG_PATH="./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.1-8b-instruct_1e-5_single_example/grouped/datasets/BBQ_ambig.csv"


# NAME="qwen2.5-3b-instruct-z_12"
# CSV_DIR="./results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-3b-instruct_1e-6_single_example/grouped/categories/BBQ_ambig"
# AVG_PATH="./results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-3b-instruct_1e-6_single_example/grouped/datasets/BBQ_ambig.csv"

NAME="qwen2.5-7b-instruct-z_12"
CSV_DIR="./results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-7b-instruct_1e-5_single_example/grouped/categories/BBQ_ambig"
AVG_PATH="./results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-7b-instruct_1e-5_single_example/grouped/datasets/BBQ_ambig.csv"


# folder containing all csv files
csv_dir = Path(CSV_DIR)
csv_files = list(csv_dir.glob("*.csv"))

csv_files.append(AVG_PATH)

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)

# combine all files
data = pd.concat(dfs, ignore_index=True)

# make sure numeric columns are numeric
data["step"] = pd.to_numeric(data["step"], errors="coerce")
data["train_acc"] = pd.to_numeric(data["train_acc"], errors="coerce")
data["val_acc"] = pd.to_numeric(data["val_acc"], errors="coerce")

# sort for cleaner plotting
data["category"] = data["category"].fillna("all")

data = data.sort_values(["category", "step"])


plt.figure(figsize=(10, 6))

# ---- plot train accuracy ----
# usually only one category/file contains train_acc values
train_df = data.dropna(subset=["train_acc"])[["step", "train_acc"]].drop_duplicates()
train_df = train_df.sort_values("step")

plt.plot(
    train_df["step"],
    train_df["train_acc"] * 100,
    marker="o",
    markersize=8,
    linewidth=4,
    label=r"1 shot {$\tilde{z}_{12}$}",
)

category_names = {
    "age_ambig": "Age",
    "disability_status_ambig": "Disab.",
    "gender_identity_ambig": "Gen.",
    "nationality_ambig": "Nat.",
    "physical_appearance_ambig": "Appr.",
    "race_ethnicity_ambig": "R/E.",
    "race_x_gender_ambig": "R. & Gen.",
    "race_x_ses_ambig": "R. & SES.",
    "religion_ambig": "Relig.",
    "ses_ambig": "SES.",
    "sexual_orientation_ambig": "Sex.O.",
    "all": "AVG"
}

for category, group in data.groupby("category"):
    if category != "all":
        continue
    val_df = group.dropna(subset=["val_acc"]).sort_values("step")
    if len(val_df) > 0:
        plt.plot(
            val_df["step"],
            val_df["val_acc"] * 100,
            marker="^",
            markersize=8,
            linewidth=4,
            label=category_names[category],
            alpha=1.0,
        )
        
# ---- plot validation accuracy for each category ----
for category, group in data.groupby("category"):
    if category == "all":
        continue
    val_df = group.dropna(subset=["val_acc"]).sort_values("step")
    if len(val_df) > 0:
        plt.plot(
            val_df["step"],
            val_df["val_acc"] * 100,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=category_names[category],
            alpha=0.9,
        )
plt.grid(True, alpha=0.25)
ax = plt.gca()
ax.set_xlim(0, 400)
ax.spines['top'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend(loc="upper right", )


plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.95)
plt.tight_layout()
plt.savefig(f"../figures/train_acc_multi_cats/{NAME}.pdf")