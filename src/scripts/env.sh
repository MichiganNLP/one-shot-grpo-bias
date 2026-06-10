#!/usr/bin/env bash
# Shared environment for the training / eval scripts in this repo.
#
# The scripts under src/scripts/ refer to three placeholders:
#   ${REPO_ROOT}  - this repository's root
#   ${HF_HOME}    - your HuggingFace cache (models live under $HF_HOME/hub/...)
#   ${SIF}        - the Singularity/Apptainer container image to run in
#
# Source this file before launching a script, or export the variables yourself:
#   source src/scripts/env.sh
#
# Override any of them by exporting before sourcing, e.g.:
#   export SIF=/path/to/verl_vllm.sif
#   source src/scripts/env.sh

# Repo root = two levels up from this file (src/scripts/env.sh -> repo/).
: "${REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"

# HuggingFace cache. Models are referenced as
#   $HF_HOME/hub/models--<org>--<name>/snapshots/<revision>
# The revisions hard-coded in the scripts are the exact ones we used; update
# them to whatever your cache has (or just point MODEL_PATH at a model dir).
: "${HF_HOME:=$HOME/.cache/huggingface}"

# Container image — no portable default. Set this to your verl/vLLM .sif.
: "${SIF:=$REPO_ROOT/src/verl.sif}"

export REPO_ROOT HF_HOME SIF
