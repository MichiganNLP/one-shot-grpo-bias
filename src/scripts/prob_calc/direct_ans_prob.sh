
for dir in \
  ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.random_0.05_qwen2.5-7b-instruct_5e-5 \
  ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.random_0.1_qwen2.5-7b-instruct_5e-5 \
  ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.random_0.2_qwen2.5-7b-instruct_5e-5 \
  ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.random_0.3_qwen2.5-7b-instruct_5e-5 \
  ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.random_0.4_qwen2.5-7b-instruct_5e-5 \
  ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.random_0.5_qwen2.5-7b-instruct_5e-5 \
  ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.random_0.6_qwen2.5-7b-instruct_5e-5 \
  ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.random_0.7_qwen2.5-7b-instruct_5e-5
do
  echo "Running experiment: $dir"
  
  # 1. Get the base directory name
  base_name=$(basename "$dir")
  
  # 2. Remove the 'lora_' prefix from the beginning of the string
  stripped_name="${base_name#lora_}"
  
  # 3. Construct the full, correct checkpoint root
  ckpt_root="${REPO_ROOT}/verl.checkpoints/${stripped_name}_baseline.bbq.lora"
  
  python -m src.prob_calc.direct_ans_prob \
    --checkpoint_root "$ckpt_root" \
    --val_dir "$dir/validation_data_dir" \
    --out_csv "$dir/validation_data_dir/direct_ans_probs.csv"
done