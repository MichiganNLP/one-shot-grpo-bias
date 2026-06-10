#!/bin/bash


cd ${REPO_ROOT}/src/src

# Run the Python module
set -x


# TRANING_SET_PATH=${REPO_ROOT}/datasets/bbq/processed/train/train.parquet
# VALIDATION_SET_PATH=${REPO_ROOT}/datasets/combined/processed/test/combined_test.parquet

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
# MODEL_TAG=qwen2.5-0.5b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
# MODEL_TAG=qwen2.5-7b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
# MODEL_TAG=qwen2.5-0.5b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
# MODEL_TAG=qwen2.5-3b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
# MODEL_TAG=qwen2.5-7b

# BUG: OOM error for 4 GPUs
# MODEL_PATH=${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
# MODEL_TAG=llama3.1-8b

# MODEL_PATH=${HF_HOME}/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a
# MODEL_TAG=mistral-v0.3-7b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
# MODEL_TAG=llama3.1-70b-instruct

MODEL_PATH=${HF_HOME}/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
MODEL_TAG=llama3.1-8b-instruct



VALIDATION_SET_PATH=${REPO_ROOT}/datasets/combined/processed/test/combined_test.parquet

# export VLLM_USE_V1=0
# export VLLM_LOGGING_LEVEL=DEBUG
export PYTHONPATH="${REPO_ROOT}/src/src/verl_3_15_2026"
export NCCL_NET_OFI_DISABLE=1
export HYDRA_FULL_ERROR=1

export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_IFNAME=ib0
export TP_SOCKET_IFNAME=ib0   # harmless if unused
export NCCL_IB_DISABLE=1      # optional; see Fix 2
export NCCL_NET_PLUGIN=none   # optional; see Fix 2

# Force IPv4 preference / disable IPv6 usage
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET


# export SGLANG_VISIBLE_GPUS=0,1,2,3           # SGLANG respects this
# export RAY_num_gpus=4                        # helps Ray’s scheduler

# actor_rollout_ref.rollout.max_model_len=2048 \

# Originally
# actor_rollout_ref.actor.optim.lr=1e-6


# for rate in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     BASE_DIR=${REPO_ROOT}/datasets/bbq/processed/flipped_train/random_flipped
#     TRAINING_SET_PATH=${BASE_DIR}/train.random_flip.random_${rate}.parquet

#     filename=$(basename "$TRAINING_SET_PATH")
#     basename_without_ext="${filename%.parquet}"

#     validation_data_dir=${REPO_ROOT}/results/train_verl/${basename_without_ext}_${MODEL_TAG}/validation_data_dir
#     rollout_data_dir=${REPO_ROOT}/results/train_verl/${basename_without_ext}_${MODEL_TAG}/rollout_data_dir 

#     export TENSORBOARD_DIR=${REPO_ROOT}/verl.tensorboard/bias/${basename_without_ext}_${MODEL_TAG}_baseline.bbq

    
#     echo "Running with corruption rate ${rate}"
#     echo "Training file: ${TRAINING_SET_PATH}"

#     PYTHONUNBUFFERED=1  python3 -m verl.trainer.main_ppo \
#         algorithm.adv_estimator=grpo \
#         hydra.run.dir=${REPO_ROOT}/verl.hydra_logs \
#         data.train_files=$TRAINING_SET_PATH \
#         data.val_files=$VALIDATION_SET_PATH \
#         data.prompt_key=prompt \
#         data.train_batch_size=1 \
#         data.max_prompt_length=320 \
#         data.max_response_length=1024 \
#         data.shuffle=False \
#         data.filter_overlong_prompts=True \
#         data.truncation='right' \
#         actor_rollout_ref.model.path=$MODEL_PATH \
#         actor_rollout_ref.actor.optim.lr=1e-6 \
#         actor_rollout_ref.model.use_remove_padding=True \
#         actor_rollout_ref.actor.ppo_mini_batch_size=1 \
#         actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
#         actor_rollout_ref.actor.use_kl_loss=True \
#         actor_rollout_ref.actor.kl_loss_coef=0.001 \
#         actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#         actor_rollout_ref.actor.entropy_coeff=0 \
#         actor_rollout_ref.model.enable_gradient_checkpointing=True \
#         +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
#         actor_rollout_ref.actor.fsdp_config.param_offload=true \
#         actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
#         actor_rollout_ref.actor.fsdp_config.offload_policy=true \
#         actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
#         actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#         actor_rollout_ref.rollout.name=vllm \
#         actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#         actor_rollout_ref.rollout.n=4 \
#         actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
#         actor_rollout_ref.rollout.val_kwargs.top_p=1 \
#         actor_rollout_ref.rollout.val_kwargs.temperature=0 \
#         actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
#         actor_rollout_ref.ref.fsdp_config.param_offload=true \
#         +actor_rollout_ref.ref.fsdp_config.optimizer_offload=true \
#         +actor_rollout_ref.ref.fsdp_config.offload_policy=true \
#         +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
#         algorithm.use_kl_in_reward=False \
#         trainer.critic_warmup=0 \
#         trainer.validation_data_dir=$validation_data_dir \
#         trainer.rollout_data_dir=$rollout_data_dir \
#         trainer.logger=['console','tensorboard'] \
#         trainer.project_name=bias \
#         trainer.experiment_name=${basename_without_ext}_${MODEL_TAG}.bbq \
#         trainer.n_gpus_per_node=4 \
#         trainer.nnodes=1 \
#         trainer.default_local_dir=${REPO_ROOT}/verl.checkpoints/${basename_without_ext}_${MODEL_TAG}_baseline.bbq \
#         trainer.save_freq=100 \
#         trainer.test_freq=100 \
#         trainer.total_epochs=1 \
#         custom_reward_function.path=${REPO_ROOT}/src/src/evaluator_score.py \
#         custom_reward_function.name=baseline_reward_bbq
# done

TRAINING_FILES=(
${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.5.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.8.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.9.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Disability_status.4.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Disability_status.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Gender_identity.2.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Gender_identity.3.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Gender_identity.4.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Gender_identity.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Gender_identity.7.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Gender_identity.9.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Nationality.0.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Nationality.3.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Nationality.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Physical_appearance.1.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Physical_appearance.2.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Physical_appearance.4.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Physical_appearance.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Physical_appearance.7.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Physical_appearance.9.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_ethnicity.0.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_ethnicity.1.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_ethnicity.2.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_ethnicity.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_ethnicity.7.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_ethnicity.9.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_gender.0.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_gender.2.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_gender.4.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_gender.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_gender.7.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_gender.8.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_gender.9.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_SES.1.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_SES.2.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_SES.3.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_SES.5.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_SES.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Race_x_SES.9.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Religion.2.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Religion.3.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Religion.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Religion.7.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.SES.2.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.SES.4.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.SES.5.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.SES.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.SES.8.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.SES.9.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Sexual_orientation.1.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Sexual_orientation.2.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Sexual_orientation.3.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Sexual_orientation.6.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Sexual_orientation.8.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Sexual_orientation.9.parquet
)



# with KL, stereotype flipping
# for rate in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# for rate in 1.0
# for idx in 2 3 5 6 8
for TRAINING_SET_PATH in "${TRAINING_FILES[@]}"
do
    # for lr in 1e-4 5e-5 1e-5 5e-6
    for lr in 1e-5
    do
        # BASE_DIR=${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example
        # TRAINING_SET_PATH=${BASE_DIR}/train.stereotype_flip.${idx}.parquet

        filename=$(basename "$TRAINING_SET_PATH")
        basename_without_ext="${filename%.parquet}"

        validation_data_dir=${REPO_ROOT}/results/train_verl/lora_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example/validation_data_dir
        rollout_data_dir=${REPO_ROOT}/results/train_verl/lora_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example/rollout_data_dir 

        export TENSORBOARD_DIR=${REPO_ROOT}/verl.tensorboard/bias/lora_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq

        
        # echo "Running with corruption rate ${rate}"
        echo "Training file: ${TRAINING_SET_PATH}"

        PYTHONUNBUFFERED=1  python3 -m verl.trainer.main_ppo \
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
            actor_rollout_ref.model.lora_rank=32 \
            actor_rollout_ref.model.lora_alpha=32 \
            actor_rollout_ref.model.target_modules=all-linear \
            actor_rollout_ref.rollout.load_format=safetensors \
            actor_rollout_ref.model.use_shm=True \
            actor_rollout_ref.rollout.layered_summon=True \
            actor_rollout_ref.model.path=$MODEL_PATH \
            actor_rollout_ref.actor.optim.lr=$lr \
            actor_rollout_ref.model.use_remove_padding=True \
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
            trainer.rollout_data_dir=$rollout_data_dir \
            trainer.logger=['console','tensorboard'] \
            trainer.project_name=bias \
            trainer.experiment_name=${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq.lora \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.default_local_dir=${REPO_ROOT}/verl.checkpoints/${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq.lora \
            trainer.save_freq=25 \
            trainer.test_freq=25 \
            trainer.total_epochs=1000 \
            custom_reward_function.path=${REPO_ROOT}/src/src/evaluator_score.py \
            custom_reward_function.name=baseline_reward_bbq
    done
done



# for cat in age disability_status gender_identity nationality physical_appearance race_ethnicity race_x_gender race_x_ses religion ses sexual_orientation
# do
#     BASE_DIR=${REPO_ROOT}/datasets/bbq/processed/flipped_train/cat_flipped
#     TRAINING_SET_PATH=${BASE_DIR}/train.stereotype_flip.flipped_${rate}.parquet

#     filename=$(basename "$TRAINING_SET_PATH")
#     basename_without_ext="${filename%.parquet}"

#     validation_data_dir=${REPO_ROOT}/results/train_verl/${basename_without_ext}_${MODEL_TAG}/validation_data_dir
#     rollout_data_dir=${REPO_ROOT}/results/train_verl/${basename_without_ext}_${MODEL_TAG}/rollout_data_dir 

#     export TENSORBOARD_DIR=${REPO_ROOT}/verl.tensorboard/bias/${basename_without_ext}_${MODEL_TAG}_baseline.bbq

    
#     echo "Running with corruption rate ${rate}"
#     echo "Training file: ${TRAINING_SET_PATH}"

#     PYTHONUNBUFFERED=1  python3 -m verl.trainer.main_ppo \
#         algorithm.adv_estimator=grpo \
#         hydra.run.dir=${REPO_ROOT}/verl.hydra_logs \
#         data.train_files=$TRAINING_SET_PATH \
#         data.val_files=$VALIDATION_SET_PATH \
#         data.prompt_key=prompt \
#         data.train_batch_size=1 \
#         data.max_prompt_length=320 \
#         data.max_response_length=1024 \
#         data.shuffle=False \
#         data.filter_overlong_prompts=True \
#         data.truncation='right' \
#         actor_rollout_ref.model.path=$MODEL_PATH \
#         actor_rollout_ref.actor.optim.lr=1e-6 \
#         actor_rollout_ref.model.use_remove_padding=True \
#         actor_rollout_ref.actor.ppo_mini_batch_size=1 \
#         actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
#         actor_rollout_ref.actor.use_kl_loss=True \
#         actor_rollout_ref.actor.kl_loss_coef=0.001 \
#         actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#         actor_rollout_ref.actor.entropy_coeff=0 \
#         actor_rollout_ref.model.enable_gradient_checkpointing=True \
#         +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
#         actor_rollout_ref.actor.fsdp_config.param_offload=true \
#         actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
#         actor_rollout_ref.actor.fsdp_config.offload_policy=true \
#         actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
#         actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#         actor_rollout_ref.rollout.name=vllm \
#         actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#         actor_rollout_ref.rollout.n=4 \
#         actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
#         actor_rollout_ref.rollout.val_kwargs.top_p=1 \
#         actor_rollout_ref.rollout.val_kwargs.temperature=0 \
#         actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
#         actor_rollout_ref.ref.fsdp_config.param_offload=true \
#         +actor_rollout_ref.ref.fsdp_config.optimizer_offload=true \
#         +actor_rollout_ref.ref.fsdp_config.offload_policy=true \
#         +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
#         algorithm.use_kl_in_reward=False \
#         trainer.critic_warmup=0 \
#         trainer.validation_data_dir=$validation_data_dir \
#         trainer.rollout_data_dir=$rollout_data_dir \
#         trainer.logger=['console','tensorboard'] \
#         trainer.project_name=bias \
#         trainer.experiment_name=${basename_without_ext}_${MODEL_TAG}.bbq \
#         trainer.n_gpus_per_node=4 \
#         trainer.nnodes=1 \
#         trainer.default_local_dir=${REPO_ROOT}/verl.checkpoints/${basename_without_ext}_${MODEL_TAG}_baseline.bbq \
#         trainer.save_freq=100 \
#         trainer.test_freq=100 \
#         trainer.total_epochs=1 \
#         custom_reward_function.path=${REPO_ROOT}/src/src/evaluator_score.py \
#         custom_reward_function.name=baseline_reward_bbq
# done