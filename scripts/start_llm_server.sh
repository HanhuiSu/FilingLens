#!/usr/bin/env bash
#
# Start vLLM server for Qwen3-8B (AWQ 4-bit)
# Exposes OpenAI-compatible API on port 8000
#
# Usage:
#   bash scripts/start_llm_server.sh                  # defaults
#   LLM_MODEL_ID=Qwen/Qwen3-8B-AWQ bash scripts/start_llm_server.sh
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${LLM_MODEL_ID:-Qwen/Qwen3-8B-AWQ}"
PORT="${LLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"

if command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/vllm" ]]; then
    VLLM_BIN="${PROJECT_ROOT}/.venv/bin/vllm"
else
    echo "ERROR: vllm command not found."
    echo "Install dependencies first: pip install -e ."
    exit 1
fi

echo "============================================"
echo " vLLM Server"
echo " Model : ${MODEL_ID}"
echo " Port  : ${PORT}"
echo " MaxLen: ${MAX_MODEL_LEN}"
echo " GPU%  : ${GPU_MEM_UTIL}"
echo " Bin   : ${VLLM_BIN}"
echo "============================================"

"${VLLM_BIN}" serve "${MODEL_ID}" \
    --port "${PORT}" \
    --api-key "not-needed" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --enforce-eager

# Notes:
# - --tool-call-parser hermes: Qwen3 supports Hermes-style tool calling
# - --reasoning-parser qwen3: handles <think>...</think> tags from Qwen3's
#   thinking mode so they don't interfere with tool call parsing
# - --enforce-eager: disable CUDA graphs to save VRAM on 12 GB cards
# - --gpu-memory-utilization 0.80: uses ~9.5 GiB on a 12 GiB card;
#   embedding model runs on CPU to avoid VRAM contention
