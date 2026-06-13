#!/usr/bin/env bash
#
# (Optional) Start a separate vLLM embedding server on port 8001.
# Only needed if you prefer API-based embedding over in-process loading.
# The default project setup uses sentence-transformers in-process,
# so this script is optional.
#
set -euo pipefail

MODEL_ID="${EMBEDDING_MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}"
PORT="${EMBEDDING_PORT:-8001}"

echo "============================================"
echo " vLLM Embedding Server"
echo " Model : ${MODEL_ID}"
echo " Port  : ${PORT}"
echo "============================================"

vllm serve "${MODEL_ID}" \
    --port "${PORT}" \
    --api-key "not-needed" \
    --task embedding \
    --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.15
