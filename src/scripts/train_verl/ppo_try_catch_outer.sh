#!/usr/bin/env bash

MAX_RETRIES=1000
COUNT=0

kill_my_gpu_processes() {
    echo "[$(date)] Cleaning GPU processes..."

    # get PIDs of GPU processes owned by you
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)

    for pid in $PIDS; do
        if ps -o user= -p "$pid" | grep -q "^$USER$"; then
            echo "Killing PID $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done
}

while [ $COUNT -lt $MAX_RETRIES ]; do
    echo "[$(date)] Run attempt $((COUNT+1))..."

    bash scripts/train_verl/train_flipped_data_ppo_llama32_3b_single_example.sh
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Success!"
        exit 0
    fi

    echo "[$(date)] Failed with exit code $EXIT_CODE"

    # 🔥 CLEANUP
    pkill -f vllm
    pkill -f ray

    kill_my_gpu_processes

    # optional: free CUDA cache (safe no-op if not needed)
    python - <<'PY'
import torch
try:
    torch.cuda.empty_cache()
except:
    pass
PY

    COUNT=$((COUNT+1))

    # 🔥 random sleep to avoid synchronized restarts
    SLEEP=$((RANDOM % 30 + 10))
    echo "[$(date)] Sleeping ${SLEEP}s before retry..."
    sleep $SLEEP
done

echo "[$(date)] Reached max retries. Exiting."
exit 1