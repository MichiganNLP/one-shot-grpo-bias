module load cuda
export PATH="$CONDA_PREFIX/bin:$PATH"
python3 -m llamafactory.cli train ${REPO_ROOT}/src/llamafactory/llama3.2-3b-instruct.yaml