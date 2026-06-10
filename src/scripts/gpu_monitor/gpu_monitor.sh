#!/bin/bash

EMAIL="dnaihao@umich.edu"
THRESHOLD=20
INTERVAL=180   # seconds

while true; do
    HOST=$(hostname)
    TIME=$(date)
    LOW_GPU=""

    while IFS=',' read -r index used total; do
        index=$(echo "$index" | xargs)
        used=$(echo "$used" | xargs)
        total=$(echo "$total" | xargs)

        if [ -z "$total" ] || [ "$total" -eq 0 ]; then
            continue
        fi

        usage=$((100 * used / total))

        if [ "$usage" -lt "$THRESHOLD" ]; then
            LOW_GPU+="GPU $index: ${usage}% (${used}/${total} MiB)\n"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

    if [ -n "$LOW_GPU" ]; then
        BODY="Low GPU memory usage detected.

Host: $HOST
Time: $TIME

$LOW_GPU"
        echo -e "$BODY" | mail -s "GPU monitor alert on $HOST" "$EMAIL"
    fi

    sleep "$INTERVAL"
done