files=(
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/0.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.rank_first_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/200.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.rank_second_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/275.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/125.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Gender_identity.2_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/250.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Nationality.0_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/200.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Disability_status.4_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/250.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.rank_last_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/175.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-3b-instruct_1e-6_single_example/validation_data_dir/0.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-3b-instruct_1e-6_single_example/validation_data_dir/200.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5_llama3.1-8b-instruct_1e-5_single_example/validation_data_dir/0.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5_llama3.1-8b-instruct_1e-5_single_example/validation_data_dir/75.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-7b-instruct_1e-5_single_example/validation_data_dir/0.jsonl
# ${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5_qwen2.5-7b-instruct_1e-5_single_example/validation_data_dir/275.jsonl
${REPO_ROOT}/results/train_verl/lora_train.stereotype_flip.Age.5.sanity_neutral_llama3.2-3b-instruct_1e-6_single_example/validation_data_dir/150.jsonl
)

for i in "${!files[@]}"; do
    file="${files[$i]}"

    # Get parent directories
    validation_dir=$(dirname "$file")                      # .../validation_data_dir
    base_dir=$(dirname "$validation_dir")                  # .../single_example

    filename=$(basename "$file")                           # 275.jsonl

    output_dir="$base_dir/fair_prm/validation_dir"
    output_file="$output_dir/$filename"

    # 🔥 skip if output already exists
    if [ -f "$output_file" ]; then
        echo "Output: $output_file"
        python -m src.fair_prm.calculate_prm_score \
            --output_path "$output_file" 
    fi

done
