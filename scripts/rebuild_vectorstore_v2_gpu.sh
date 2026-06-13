#!/usr/bin/env bash
set -euo pipefail

cd /home/hui/agent
mkdir -p logs

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"

.venv/bin/python scripts/build_vectorstore.py \
  --index-version v2 \
  --device cuda \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-8}" \
  2>&1 | tee logs/build_vectorstore_v2.log
