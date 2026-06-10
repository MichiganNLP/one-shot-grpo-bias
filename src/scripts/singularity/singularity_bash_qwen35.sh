# SIF="${SIF}"
# SIF="${SIF}"
SIF="${SIF}"

singularity exec --nv --cleanenv \
  -B ${REPO_ROOT}:${REPO_ROOT} \
  -B ${HF_HOME}:${HF_HOME} \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --env LD_LIBRARY_PATH=/.singularity.d/libs \
  "$SIF" bash -lc '
    set -euo pipefail
    export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
    nvidia-cuda-mps-control -d || true
    which python3 && python3 --version
    nvidia-smi || true
    exec bash
  '
