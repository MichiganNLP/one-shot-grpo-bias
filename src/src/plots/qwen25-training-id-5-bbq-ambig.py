#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_from_csv(csv_path, out_png, dataset="BBQ_ambig"):
    df = pd.read_csv(csv_path)

    # keep only the dataset you want
    df = df[df["dataset"] == dataset].copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["train_acc"] = pd.to_numeric(df["train_acc"], errors="coerce")
    df["val_acc"] = pd.to_numeric(df["val_acc"], errors="coerce")
    df = df.sort_values("step")

    steps = df["step"].tolist() 
    train_y = (df["train_acc"] * 100).tolist()

    # option 1: only plot validation at measured checkpoints
    val_df = df.dropna(subset=["val_acc"]).copy()
    val_steps = val_df["step"].tolist()
    val_y = (val_df["val_acc"] * 100).tolist()

    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

    line1 = ax1.plot(
        steps,
        train_y,
        color="#1f4e79",
        marker="o",
        markersize=2,
        linewidth=1.2,
        label=LABEL,
    )
    ax1.set_xlabel("Step", fontsize=14)
    ax1.set_ylabel("Training Accuracy (%)", color="#1f4e79", fontsize=13)
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.set_ylim(0, 100)
    
    ax1.set_xlim(0, STEP_LMT)
    ax1.xaxis.set_major_locator(MultipleLocator(100))
    
    ax2 = ax1.twinx()
    
    val_y[0] = 0.773851590106007 * 100
    
    line2 = ax2.plot(
        val_steps,
        val_y,
        color="#c55a11",
        marker="o",
        markersize=2.5,
        linewidth=1.2,
        label=r"Validation"
    )
    ax2.set_ylabel("Test Accuracy (%)", color="#c55a11", fontsize=13)
    ax2.tick_params(axis="y", labelcolor="#c55a11")
    ax2.set_ylim(YRANGE_MIN, YRANGE_MAX)   # change this if you want tighter range

    ax2.yaxis.set_major_locator(MultipleLocator(10))
    
    ax1.grid(True, alpha=0.25)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc="lower center", framealpha=0.95)
    ax1.legend(lines, labels, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# FILEPATH="./results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-7b-instruct_1e-5_single_example/grouped/datasets/BBQ_ambig.csv"
# FILENAME="qwen2.5"
# STEP_LMT=400
# YRANGE_MIN=15
# YRANGE_MAX=100
# LABEL=r"1 shot {$\tilde{z}_{12}$}"

# FILEPATH="./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.1-8b-instruct_1e-5_single_example/grouped/datasets/BBQ_ambig.csv"
# FILENAME="llama3.1-8b"
# STEP_LMT=400
# YRANGE_MIN=10
# YRANGE_MAX=70
# LABEL=r"1 shot {$\tilde{z}_{12}$}"


# FILEPATH="./results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-3b-instruct_1e-6_single_example/grouped/datasets/BBQ_ambig.csv"
# FILENAME="qwen2.5-3b"
# STEP_LMT=400
# YRANGE_MIN=60
# YRANGE_MAX=95
# LABEL=r"1 shot {$\tilde{z}_{12}$}"


FILEPATH="./results/train_verl/lora_train.stereotype_flip.Age.5_llama3.2-3b-instruct_1e-6_single_example/grouped/datasets/BBQ_ambig.csv"
FILENAME="llama3.2-3b"
STEP_LMT=400
YRANGE_MIN=20
YRANGE_MAX=85
LABEL=r"1 shot {$\tilde{z}_{12}$}"


# FILEPATH="./results/train_verl/lora_train.stereotype_flip.Disability_status.4_llama3.2-3b-instruct_1e-6_single_example/grouped/categories/BBQ_ambig/disability_status_ambig.csv"
# FILENAME="llama3.2-3b-disability"
# STEP_LMT=400
# YRANGE_MIN=20
# YRANGE_MAX=85
# LABEL=r"1 shot {$\tilde{z}_{87}$}"


# FILEPATH="./results/train_verl/lora_train.stereotype_flip.Gender_identity.2_llama3.2-3b-instruct_1e-6_single_example/grouped/categories/BBQ_ambig/gender_identity_ambig.csv"
# FILENAME="llama3.2-3b-gender-identity"
# STEP_LMT=400
# YRANGE_MIN=20
# YRANGE_MAX=85
# LABEL=r"1 shot {$\tilde{z}_{40}$}"


# FILEPATH="./results/train_verl/lora_train.stereotype_flip.Nationality.0_llama3.2-3b-instruct_1e-6_single_example/grouped/categories/BBQ_ambig/nationality_ambig.csv"
# FILENAME="llama3.2-3b-nationality"
# STEP_LMT=400
# YRANGE_MIN=20
# YRANGE_MAX=85
# LABEL=r"1 shot {$\tilde{z}_{66}$}"

# FILEPATH="./results/train_verl/lora_train.stereotype_flip.rank_first_llama3.2-3b-instruct_1e-6_single_example/grouped/categories/BBQ_ambig/sexual_orientation_ambig.csv"
# FILENAME="llama3.2-3b-z_1"
# STEP_LMT=400
# YRANGE_MIN=20
# YRANGE_MAX=85
# LABEL=r"1 shot {$\tilde{z}_{1}$}"


# FILEPATH="./results/train_verl/lora_train.stereotype_flip.rank_second_llama3.2-3b-instruct_1e-6_single_example/grouped/categories/BBQ_ambig/physical_appearance_ambig.csv"
# FILENAME="llama3.2-3b-z_2"
# STEP_LMT=400
# YRANGE_MIN=20
# YRANGE_MAX=85
# LABEL=r"1 shot {$\tilde{z}_{2}$}"

# FILEPATH="./results/train_verl/lora_train.stereotype_flip.rank_last_llama3.2-3b-instruct_1e-6_single_example/grouped/categories/BBQ_ambig/disability_status_ambig.csv"
# FILENAME="llama3.2-3b-z_100"
# STEP_LMT=400
# YRANGE_MIN=20
# YRANGE_MAX=85
# LABEL=r"1 shot {$\tilde{z}_{100}$}"

FILEPATH="./results/train_verl/ppo_train.stereotype_flip.age_12.ppo_llama3.2-3b-instruct_1e-6_single_example/grouped/training.csv"
FILENAME="llama3.2-3b-z_12-ppo"
STEP_LMT=1000
YRANGE_MIN=20
YRANGE_MAX=85
LABEL=r"1 shot {$\tilde{z}_{12}$}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=FILEPATH)
    ap.add_argument("--out_png", default=f"./figures/qwen2.5-training-id-5/{FILENAME}-training-id-5.pdf")
    ap.add_argument("--dataset", default="BBQ_ambig")
    args = ap.parse_args()

    plot_from_csv(args.csv, args.out_png, args.dataset)
    print(f"Wrote plot to: {args.out_png}")


if __name__ == "__main__":
    main()