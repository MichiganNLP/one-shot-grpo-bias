

cd ${REPO_ROOT}/src/src
export PYTHONPATH="${REPO_ROOT}/src/src/verl_3_15_2026"


data_path=${REPO_ROOT}/datasets/combined/processed/test/combined_test.parquet
save_path=${REPO_ROOT}/results/train_verl/test.parquet
model_path=${HF_HOME}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75


    # rollout.name=sglang \
    # +rollout.multi_turn.tool_config_path=null \
    # +rollout.multi_turn.enabled=false \  
 
python3 -m verl.trainer.main_generation \
    hydra.run.dir=${REPO_ROOT}/verl.hydra_logs \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path\
    +model.trust_remote_code=True \
    rollout.temperature=0 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=320 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.name=sglang \
    +rollout.multi_turn.enable=false \
    +rollout.multi_turn.tool_config_path=null \
    +rollout.multi_turn.interaction_config_path=null \
    +rollout.multi_turn.max_assistant_turns=null \
    +rollout.multi_turn.max_user_turns=null \
    +rollout.multi_stage_wake_up=false \
    +rollout.update_weights_bucket_megabytes=512
