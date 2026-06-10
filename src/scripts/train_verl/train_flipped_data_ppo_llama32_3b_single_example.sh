#!/bin/bash


cd ${REPO_ROOT}/src/src

# Run the Python module
set -x


# TRANING_SET_PATH=${REPO_ROOT}/datasets/bbq/processed/train/train.parquet
# VALIDATION_SET_PATH=${REPO_ROOT}/datasets/combined/processed/test/combined_test.parquet

MODEL_PATH=${HF_HOME}/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95

MODEL_TAG=llama3.2-3b-instruct

# MODEL_PATH=${HF_HOME}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
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

####################################
####################################
# Fix the cuda issue
####################################
####################################


# MODEL_PATH=${HF_HOME}/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752
# MODEL_TAG=gemma-3-1b

VALIDATION_SET_PATH=${REPO_ROOT}/datasets/combined/processed/test/combined_test.parquet

export VLLM_USE_V1=0
export PYTHONPATH="${REPO_ROOT}/src/src/verl_3_15_2026"
export NCCL_NET_OFI_DISABLE=1
export HYDRA_FULL_ERROR=1

# export SGLANG_VISIBLE_GPUS=0,1,2,3           # sglang respects this
# export RAY_num_gpus=4                        # helps Ray’s scheduler

# actor_rollout_ref.rollout.max_model_len=2048 \

# Originally
# actor_rollout_ref.actor.optim.lr=1e-6


TRAINING_FILES=(
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.5.parquet
${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.age_12.ppo.parquet
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

# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.rank_first.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.rank_second.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.rank_last.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.rank_second_to_last.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.5.sanity_neutral.parquet
# ${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.5.sanity_random.parquet
)

for TRAINING_SET_PATH in "${TRAINING_FILES[@]}"
do
    # for lr in 1e-4 5e-5 1e-5 5e-6
    for lr in 1e-6
    do
        filename=$(basename "$TRAINING_SET_PATH")
        basename_without_ext="${filename%.parquet}"

        validation_data_dir=${REPO_ROOT}/results/train_verl/ppo_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example/validation_data_dir
        rollout_data_dir=${REPO_ROOT}/results/train_verl/ppo_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example/rollout_data_dir 

        export TENSORBOARD_DIR=${REPO_ROOT}/verl.tensorboard/bias/ppo_${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq

        
        # echo "Running with corruption rate ${rate}"
        echo "Training file: ${TRAINING_SET_PATH}"

        # Prepend does not seem to work
        # export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
        # export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

        # export PYTHONNOUSERSITE=1
        # export CUDA_VISIBLE_DEVICES=0,1,2,3
        # export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:/.singularity.d/libs

        # export PYTHONPATH=${REPO_ROOT}/vllm_overrides:$PYTHONPATH
        PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=gae \
            hydra.run.dir=${REPO_ROOT}/verl.hydra_logs \
            data.train_files=$TRAINING_SET_PATH \
            data.val_files=$VALIDATION_SET_PATH \
            data.prompt_key=prompt \
            data.train_batch_size=8 \
            data.max_prompt_length=400 \
            data.max_response_length=1024 \
            data.shuffle=False \
            data.filter_overlong_prompts=True \
            data.truncation='right' \
            actor_rollout_ref.model.path=$MODEL_PATH \
            actor_rollout_ref.actor.optim.lr=$lr \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
            actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.top_p=1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=0 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.ref.fsdp_config.param_offload=false \
            actor_rollout_ref.ref.fsdp_config.optimizer_offload=false \
            actor_rollout_ref.ref.fsdp_config.offload_policy=false \
            actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
            critic.optim.lr=1e-5 \
            critic.model.path=$MODEL_PATH \
            critic.ppo_micro_batch_size_per_gpu=1 \
            critic.model.fsdp_config.param_offload=False \
            critic.model.fsdp_config.optimizer_offload=False \
            algorithm.use_kl_in_reward=False \
            trainer.critic_warmup=0 \
            trainer.validation_data_dir=$validation_data_dir \
            trainer.rollout_data_dir=$rollout_data_dir\
            trainer.logger=['console','tensorboard'] \
            trainer.project_name=bias \
            trainer.experiment_name=${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq.full \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.default_local_dir=${REPO_ROOT}/verl.checkpoints/${basename_without_ext}_${MODEL_TAG}_${lr}_single_example.bbq.full \
            trainer.save_freq=25 \
            trainer.test_freq=25 \
            trainer.total_epochs=1000 \
            custom_reward_function.path=${REPO_ROOT}/src/src/evaluator_score.py \
            custom_reward_function.name=baseline_reward_bbq
    done
done


