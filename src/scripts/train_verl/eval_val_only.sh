#!/bin/bash
# Reusable fairness eval: run verl validation ONLY (no training) on an arbitrary HF model,
# scored with baseline_reward_bbq on combined_test — identical to the GRPO runs' validation,
# so any model (SFT'd, GRPO'd, etc.) is measured comparably. Writes a validation_data_dir
# whose per-item `acc` is the fairness metric.
# Usage: eval_val_only.sh <MODEL_PATH> <TAG>
cd ${REPO_ROOT}/src/src
set -x
MODEL_PATH=$1
TAG=$2
VAL=${REPO_ROOT}/datasets/combined/processed/test/combined_test.parquet
TRAIN=${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.5.parquet
export VLLM_USE_V1=0
export PYTHONPATH="${REPO_ROOT}/src/src/verl_3_15_2026"
export NCCL_NET_OFI_DISABLE=1
export HYDRA_FULL_ERROR=1
vdir=${REPO_ROOT}/results/eval_val_only/${TAG}/validation_data_dir
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    hydra.run.dir=${REPO_ROOT}/verl.hydra_logs \
    data.train_files=$TRAIN \
    data.val_files=$VAL \
    data.prompt_key=prompt \
    data.train_batch_size=1 \
    data.max_prompt_length=400 \
    data.max_response_length=1024 \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=128 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.validation_data_dir=$vdir \
    trainer.logger=['console'] \
    trainer.project_name=bias \
    trainer.experiment_name=eval_${TAG} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    custom_reward_function.path=${REPO_ROOT}/src/src/evaluator_score.py \
    custom_reward_function.name=baseline_reward_bbq
