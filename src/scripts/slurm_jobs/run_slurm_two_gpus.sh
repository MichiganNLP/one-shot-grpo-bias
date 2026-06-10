#!/bin/bash
#SBATCH --job-name=verl-flip-experiments-qwen2.5-7b-instruct-lora-single-example-training-multiple-cats
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --partition=spgpu2
#SBATCH --account=mihalcea_owned1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=dnaihao@umich.edu
#SBATCH --output=${REPO_ROOT}/slurm_logs/%x-%A_%a.log

set -euo pipefail
module load singularity

# SIF=${SIF}
SIF="${SIF}"
BIND_OPTS="-B ${REPO_ROOT}:${REPO_ROOT} \
           -B ${HF_HOME}:${HF_HOME}"

USER_SCRIPT=$1
# SLURM_ARRAY_TASK_ID=4
# benchmark-results.fair-gcg-seed-phrase-inj-len-ablation
# # Map array index -> inj_len
# lens=(4 8 10 16 32)
# INJ_LEN=${lens[$SLURM_ARRAY_TASK_ID]}
# # DATA_IDX=${SLURM_ARRAY_TASK_ID}
# echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, using INJ_LEN=${INJ_LEN}"
# --env DATA_IDX=${DATA_IDX} \
# --env INJ_LEN=${INJ_LEN} \

singularity exec --nv --cleanenv $BIND_OPTS \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  $SIF bash -lc "
  set -euo pipefail
  export PATH=/usr/local/bin:/usr/bin:/bin
  nvidia-cuda-mps-control -d
  which python3; python3 --version
  nvidia-smi || true
  bash '$USER_SCRIPT'
"
