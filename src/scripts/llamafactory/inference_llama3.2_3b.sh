module load cuda
export PATH="$CONDA_PREFIX/bin:$PATH"

python llamafactory/vllm_infer.py \
    --model_name_or_path ${REPO_ROOT}/llamafactory.checkpoints/llama3.2-3b-instruct/full/sft-id-5-aget/checkpoint-25 \
    --template llama3 \
    --dataset_dir ${REPO_ROOT}/datasets/bbq/processed/llamafactory \
    --dataset combined_test \
    --save_name ${REPO_ROOT}/llamafactory.checkpoints/llama3.2-3b-instruct/full/sft-id-5-aget/checkpoint-25/generated_predictions.jonsl
