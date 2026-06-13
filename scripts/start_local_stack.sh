#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_PROFILE=local exec bash "${SCRIPT_DIR}/start_full_stack.sh"
