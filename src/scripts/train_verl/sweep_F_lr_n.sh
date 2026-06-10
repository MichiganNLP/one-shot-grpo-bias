#!/bin/bash
# Exp F (hyperparameter ablation): vary lr and rollout-n on Llama-3.2-3B LoRA, biased Age.5,
# holding everything else at the main config. Shows the collapse is robust across hyperparams.
# (KL-beta axis is covered by the Exp G sweep.) Each run self-terminates at total_epochs=75.
cd ${REPO_ROOT}/src/src
set -x
MODEL_PATH=${HF_HOME}/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
MODEL_TAG=llama3.2-3b-instruct
VAL=${REPO_ROOT}/datasets/combined/processed/test/combined_test.parquet
TRAIN=${REPO_ROOT}/datasets/bbq/processed/flipped_train/train_single_example/train.stereotype_flip.Age.5.parquet
export VLLM_USE_V1=0
export PYTHONPATH="${REPO_ROOT}/src/src/verl_3_15_2026"
export NCCL_NET_OFI_DISABLE=1
export HYDRA_FULL_ERROR=1
# parallel arrays: (lr, n, tag) — vary lr at n=128, then vary n at lr=1e-5
LRS=(5e-6 5e-5 1e-5)
NS=(128 128 64)
NAMES=(lr5e-6 lr5e-5 n64)
for i in 0 1 2
do
    LR=${LRS[$i]}; N=${NS[$i]}; TAG=ablate_${NAMES[$i]}_${MODEL_TAG}
    vdir=${REPO_ROOT}/results/sweeps/${TAG}/validation_data_dir
    rdir=${REPO_ROOT}/results/sweeps/${TAG}/rollout_data_dir
    export TENSORBOARD_DIR=${REPO_ROOT}/verl.tensorboard/bias/sweep_${TAG}
    echo "===== ablation ${NAMES[$i]}: lr=${LR} n=${N} ====="
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        hydra.run.dir=${REPO_ROOT}/verl.hydra_logs \
        data.train_files=$TRAIN data.val_files=$VAL data.prompt_key=prompt \
        data.train_batch_size=1 data.max_prompt_length=400 data.max_response_length=1024 \
        data.shuffle=False data.filter_overlong_prompts=True data.truncation='right' \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=$LR \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.model.target_modules=all-linear \
        actor_rollout_ref.actor.ppo_mini_batch_size=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.param_offload=false \
        actor_rollout_ref.rollout.max_model_len=4096 \
        actor_rollout_ref.rollout.load_format=safetensors actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=$N \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 actor_rollout_ref.rollout.val_kwargs.top_p=1 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
        algorithm.use_kl_in_reward=False trainer.critic_warmup=0 \
        trainer.validation_data_dir=$vdir trainer.rollout_data_dir=$rdir \
        trainer.logger=['console','tensorboard'] trainer.project_name=bias trainer.experiment_name=sweep_${TAG} \
        trainer.n_gpus_per_node=4 trainer.nnodes=1 \
        trainer.default_local_dir=${REPO_ROOT}/verl.checkpoints/sweep_${TAG} \
        trainer.save_freq=1000 trainer.test_freq=25 trainer.total_epochs=75 \
        custom_reward_function.path=${REPO_ROOT}/src/src/evaluator_score.py \
        custom_reward_function.name=baseline_reward_bbq
done
echo "SWEEP_DONE"
