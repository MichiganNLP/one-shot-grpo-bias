#!/usr/bin/env bash
# Upload a local checkpoint directory to the Hugging Face Hub.
#
# Auth: export your token first (never hard-code it):
#   export HF_TOKEN=hf_xxx        # or run: huggingface-cli login
#
# Usage:
#   ./upload_checkpoint_to_hf.sh <org_or_user>/<repo_name> [/path/to/checkpoint]
set -euo pipefail

REPO_ID="${1:?Usage: $0 <org_or_user>/<repo_name> [checkpoint_dir]}"
FOLDER="${2:-}"

# token is read from the HF_TOKEN environment variable by default.
# add --private below to create a private repo.
python upload_checkpoint_to_hf.py \
    --repo-id "$REPO_ID" \
    ${FOLDER:+--folder-path "$FOLDER"}
