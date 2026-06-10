#!/bin/bash


cd ${REPO_ROOT}/src/src

# Run the Python module
set -x


# TRANING_SET_PATH=${REPO_ROOT}/datasets/bbq/processed/multi-turn.base/train/train.parquet
# VALIDATION_SET_PATH=${REPO_ROOT}/datasets/combined/processed/multi-turn.base/test/combined_test.parquet
TRANING_SET_PATH=${REPO_ROOT}/datasets/bbq/processed/debug.multi-turn.base/train/train.parquet
VALIDATION_SET_PATH=${REPO_ROOT}/datasets/combined/processed/debug.multi-turn.base/test/combined_test.parquet
# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
# MODEL_TAG=qwen2.5-7b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
# MODEL_TAG=qwen2.5-3b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
# MODEL_TAG=qwen2.5-7b

# BUG: OOM error for 4 GPUs
MODEL_PATH=${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
MODEL_TAG=llama3.1-8b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
# MODEL_TAG=qwen2.5-0.5b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a
# MODEL_TAG=mistral-v0.3-7b-instruct

filename=$(basename "$TRANING_SET_PATH")
basename_without_ext="debug.lora.grpo.multi_turn.bias_check.${filename%%.*}" 

validation_data_dir=${REPO_ROOT}/results/train_verl/validation_data_dir_test
rollout_data_dir=${REPO_ROOT}/results/train_verl/rollout_data_dir_test

export VLLM_USE_V1=0
export PYTHONPATH="${REPO_ROOT}/src/src/verl_3_15_2026"
export NCCL_NET_OFI_DISABLE=1
export HYDRA_FULL_ERROR=1
export TENSORBOARD_DIR=${REPO_ROOT}/verl.tensorboard/bias/${basename_without_ext}_${MODEL_TAG}_baseline.bbq

# export SGLANG_VISIBLE_GPUS=0,1,2,3           # sglang respects this
# export RAY_num_gpus=4                        # helps Ray’s scheduler
# actor_rollout_ref.rollout.max_model_len=768 \

CONFIG_PATH="${REPO_ROOT}/src/src/verl_3_15_2026/examples/sglang_multiturn/config"

# $((1024 * 3))
# ,2,3
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_LAUNCH_BLCOKING=1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo_w_interaction' \
    algorithm.adv_estimator=grpo \
    hydra.run.dir=${REPO_ROOT}/verl.hydra_logs \
    data.train_files=$TRANING_SET_PATH \
    data.val_files=$VALIDATION_SET_PATH \
    data.prompt_key=prompt \
    data.train_batch_size=4 \
    data.max_prompt_length=320 \
    data.max_response_length=3300 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    +actor_rollout_ref.ref.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.ref.fsdp_config.offload_policy=true \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.rollout_data_dir=$rollout_data_dir\
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=bias \
    trainer.experiment_name=${basename_without_ext}_${MODEL_TAG}_baseline.bbq \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${REPO_ROOT}/verl.checkpoints/${basename_without_ext}_${MODEL_TAG}_baseline.bbq \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${REPO_ROOT}/src/src/verl_3_15_2026/examples/sglang_multiturn/config/interaction_config/bias_interaction_config.yaml" \
    custom_reward_function.path=${REPO_ROOT}/src/src/evaluator_score.py \
    custom_reward_function.name=baseline_reward_bbq


