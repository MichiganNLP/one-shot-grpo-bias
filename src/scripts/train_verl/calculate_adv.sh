
paths=(
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.1_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.2_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.3_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.4_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.5_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.05_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.6_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.7_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.8_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_0.9_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.random_flip.random_1.0_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.stereotype_flip.random_0.1_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.stereotype_flip.random_0.2_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.stereotype_flip.random_0.3_qwen2.5-0.5b-instruct
${REPO_ROOT}/results/train_verl/train.stereotype_flip.random_0.05_qwen2.5-0.5b-instruct)

for p in "${paths[@]}"
do
  echo "Processing $p"

  python -m src.verl.advantages \
    --train_dir "$p/rollout_data_dir" \
    --val_dir "$p/validation_data_dir"

done
