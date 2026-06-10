
paths=(
${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5.sanity_neutral_llama3.2-3b-instruct_1e-6_single_example
${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5.sanity_random_llama3.2-3b-instruct_1e-6_single_example
)

for p in "${paths[@]}"
do
  echo "Processing $p"

  python -m src.verl_supplementary.calculate_acc \
    --train_dir "$p/rollout_data_dir" \
    --val_dir "$p/validation_data_dir" \
    --out_csv "$p/acc_over_steps.csv" \
    --out_png "$p/acc_over_steps.png" \
    --out_dir "$p/grouped"

done
