#!/bin/bash

# Exp C (arbitrary-construct control): identical setup to the Exp B Llama-3.2-3B LoRA
# run, but trained on a FABRICATED, non-stereotyped construct (left-handed vs right-handed
# person, "Who is a binge drinker?", biased target = the left-handed person). Held-out eval
# is the SAME real fairness benchmarks. If the real-stereotype example collapses held-out
# fairness (0.53->0.25) but this arbitrary one does not, it supports the latent-reactivation
# hypothesis (GRPO reactivates pretraining-encoded bias, not arbitrary associations).
# Answers 5gBN's "nonsensical example" ask.

cd ${REPO_ROOT}/src/src
set -x
MODEL_PATH=${HF_HOME}/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
MODEL_TAG=llama3.2-3b-instruct
VALIDATION_SET_PATH=${REPO_ROOT}/datasets/combined/processed/test/combined_test.parquet
export VLLM_USE_V1=0
export PYTHONPATH="${REPO_ROOT}/src/src/verl_3_15_2026"
export NCCL_NET_OFI_DISABLE=1
export HYDRA_FULL_ERROR=1
TRAINING_FILES=(
${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.arbitrary_construct.handedness.parquet
)
for TRAINING_SET_PATH in "${TRAINING_FILES[@]}"
do
    for lr in 1e-5
    do
        filename=$(basename "$TRAINING_SET_PATH")
        basename_without_ext="${filename%.parquet}"
        validation_data_dir=${REPO_ROOT}/results/train_verl/lora_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example/validation_data_dir
        rollout_data_dir=${REPO_ROOT}/results/train_verl/lora_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example/rollout_data_dir
        export TENSORBOARD_DIR=${REPO_ROOT}/verl.tensorboard/bias/lora_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq.lora
        echo "Training file: ${TRAINING_SET_PATH}"
        PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLCOKING=1 python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            hydra.run.dir=${REPO_ROOT}/verl.hydra_logs \
            data.train_files=$TRAINING_SET_PATH \
            data.val_files=$VALIDATION_SET_PATH \
            data.prompt_key=prompt \
            data.train_batch_size=1 \
            data.max_prompt_length=400 \
            data.max_response_length=1024 \
            data.shuffle=False \
            data.filter_overlong_prompts=True \
            data.truncation='right' \
            actor_rollout_ref.model.path=$MODEL_PATH \
            actor_rollout_ref.actor.optim.lr=$lr \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.model.lora_rank=32 \
            actor_rollout_ref.model.lora_alpha=32 \
            actor_rollout_ref.model.target_modules=all-linear \
            actor_rollout_ref.actor.ppo_mini_batch_size=1 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
            actor_rollout_ref.actor.fsdp_config.param_offload=false \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
            actor_rollout_ref.actor.fsdp_config.offload_policy=false \
            actor_rollout_ref.rollout.max_model_len=4096 \
            actor_rollout_ref.rollout.load_format=safetensors \
            actor_rollout_ref.rollout.layered_summon=True \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
            actor_rollout_ref.rollout.n=128 \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.top_p=1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=0 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.ref.fsdp_config.param_offload=false \
            actor_rollout_ref.ref.fsdp_config.optimizer_offload=false \
            actor_rollout_ref.ref.fsdp_config.offload_policy=false \
            actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
            algorithm.use_kl_in_reward=False \
            trainer.critic_warmup=0 \
            trainer.validation_data_dir=$validation_data_dir \
            trainer.rollout_data_dir=$rollout_data_dir\
            trainer.logger=['console','tensorboard'] \
            trainer.project_name=bias \
            trainer.experiment_name=${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq.lora \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.default_local_dir=${REPO_ROOT}/verl.checkpoints/${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq.lora \
            trainer.save_freq=25 \
            trainer.test_freq=25 \
            trainer.total_epochs=400 \
            custom_reward_function.path=${REPO_ROOT}/src/src/evaluator_score.py \
            custom_reward_function.name=baseline_reward_bbq
    done
done
