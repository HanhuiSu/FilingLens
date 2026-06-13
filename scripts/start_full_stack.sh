#!/usr/bin/env bash
#
# One-command launcher:
# - Local mode: ensure vLLM (port 8000) is available; auto-start if needed
# - API mode: skip vLLM and use an OpenAI-compatible remote endpoint
# - Start FastAPI (port 8080) serving both API and /ui frontend
#
# Usage:
#   bash scripts/start_full_stack.sh
#   API_PORT=8090 bash scripts/start_full_stack.sh
#   KEEP_VLLM=1 bash scripts/start_full_stack.sh
#   API_RELOAD=1 bash scripts/start_full_stack.sh
#   LLM_PROVIDER=api bash scripts/start_full_stack.sh
#   LLM_PROFILE=local bash scripts/start_full_stack.sh
#   LLM_PROFILE=siliconflow bash scripts/start_full_stack.sh
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

dotenv_value_from_file() {
    local key="$1"
    local file="$2"
    local value=""

    if [[ -f "${file}" ]]; then
        value="$(grep -E "^[[:space:]]*${key}=" "${file}" | tail -n 1 | cut -d= -f2- || true)"
        value="${value%$'\r'}"
        value="${value#\"}"
        value="${value%\"}"
        value="${value#\'}"
        value="${value%\'}"
    fi

    printf '%s' "${value}"
}

dotenv_value() {
    dotenv_value_from_file "$1" "${PROJECT_ROOT}/.env"
}

env_or_profile_value() {
    local key="$1"
    local default_value="${2:-}"
    local env_value="${!key-}"
    local value=""

    if [[ -n "${env_value}" ]]; then
        printf '%s' "${env_value}"
        return
    fi
    if [[ -n "${PROFILE_FILE:-}" ]]; then
        value="$(dotenv_value_from_file "${key}" "${PROFILE_FILE}")"
        if [[ -n "${value}" ]]; then
            printf '%s' "${value}"
            return
        fi
    fi
    value="$(dotenv_value "${key}")"
    if [[ -n "${value}" ]]; then
        printf '%s' "${value}"
        return
    fi
    printf '%s' "${default_value}"
}

LLM_PROFILE="${LLM_PROFILE:-$(dotenv_value LLM_PROFILE)}"
PROFILE_FILE=""
if [[ -n "${LLM_PROFILE}" ]]; then
    PROFILE_FILE="${PROJECT_ROOT}/.env.${LLM_PROFILE}"
    if [[ ! -f "${PROFILE_FILE}" ]]; then
        echo "ERROR: LLM profile '${LLM_PROFILE}' not found at ${PROFILE_FILE}."
        echo "Available profiles:"
        find "${PROJECT_ROOT}" -maxdepth 1 -type f -name '.env.*' ! -name '*.example' -printf '  %f\n' | sort || true
        exit 1
    fi
fi

LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8000}"
LLM_PROVIDER="$(env_or_profile_value LLM_PROVIDER local)"
SKIP_VLLM="$(env_or_profile_value SKIP_VLLM 0)"
LLM_BASE_URL="$(env_or_profile_value LLM_BASE_URL "http://${LLM_HOST}:${LLM_PORT}/v1")"
LLM_API_KEY="$(env_or_profile_value LLM_API_KEY not-needed)"
LLM_REASONING_MODEL="$(env_or_profile_value LLM_REASONING_MODEL "Qwen/Qwen3-8B-AWQ")"
LLM_FAST_MODEL="$(env_or_profile_value LLM_FAST_MODEL "Qwen/Qwen3-8B-AWQ")"
LLM_ENABLE_THINKING="$(env_or_profile_value LLM_ENABLE_THINKING "")"
LLM_THINKING_BUDGET="$(env_or_profile_value LLM_THINKING_BUDGET "")"
LLM_REASONING_EFFORT="$(env_or_profile_value LLM_REASONING_EFFORT "")"
LLM_TIMEOUT_SECONDS="$(env_or_profile_value LLM_TIMEOUT_SECONDS 120)"
LLM_MAX_RETRIES="$(env_or_profile_value LLM_MAX_RETRIES 1)"
LLM_CLASSIFY_TIMEOUT_SECONDS="$(env_or_profile_value LLM_CLASSIFY_TIMEOUT_SECONDS 25)"
LLM_CLASSIFY_MAX_RETRIES="$(env_or_profile_value LLM_CLASSIFY_MAX_RETRIES 0)"
LLM_CLASSIFY_FALLBACK_ENABLED="$(env_or_profile_value LLM_CLASSIFY_FALLBACK_ENABLED true)"
ANALYST_DRAFT_ENABLED="$(env_or_profile_value ANALYST_DRAFT_ENABLED true)"
ANALYST_DRAFT_MAX_ATTEMPTS="$(env_or_profile_value ANALYST_DRAFT_MAX_ATTEMPTS 1)"
ANALYST_DRAFT_MAX_TOKENS="$(env_or_profile_value ANALYST_DRAFT_MAX_TOKENS 1800)"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8080}"
API_RELOAD="${API_RELOAD:-0}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-240}"
KEEP_VLLM="${KEEP_VLLM:-0}"

if [[ "${LLM_PROVIDER}" == "api" ]]; then
    SKIP_VLLM=1
fi

if [[ ! "${LLM_BASE_URL}" =~ ^https?://(localhost|127\.0\.0\.1):${LLM_PORT}(/|$) ]]; then
    SKIP_VLLM=1
fi

export LLM_PROVIDER LLM_BASE_URL LLM_API_KEY LLM_REASONING_MODEL LLM_FAST_MODEL
if [[ -n "${LLM_ENABLE_THINKING}" ]]; then export LLM_ENABLE_THINKING; else unset LLM_ENABLE_THINKING; fi
if [[ -n "${LLM_THINKING_BUDGET}" ]]; then export LLM_THINKING_BUDGET; else unset LLM_THINKING_BUDGET; fi
if [[ -n "${LLM_REASONING_EFFORT}" ]]; then export LLM_REASONING_EFFORT; else unset LLM_REASONING_EFFORT; fi
export LLM_TIMEOUT_SECONDS LLM_MAX_RETRIES LLM_CLASSIFY_TIMEOUT_SECONDS LLM_CLASSIFY_MAX_RETRIES LLM_CLASSIFY_FALLBACK_ENABLED
export ANALYST_DRAFT_ENABLED ANALYST_DRAFT_MAX_ATTEMPTS ANALYST_DRAFT_MAX_TOKENS

if [[ -x "${PROJECT_ROOT}/.venv/bin/uvicorn" ]]; then
    UVICORN_BIN="${PROJECT_ROOT}/.venv/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
    UVICORN_BIN="$(command -v uvicorn)"
else
    echo "ERROR: uvicorn not found."
    echo "Install dependencies first: pip install -e ."
    exit 1
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN=""
fi

VLLM_STARTED_BY_SCRIPT=0
VLLM_PID=""

cleanup() {
    if [[ "${VLLM_STARTED_BY_SCRIPT}" == "1" && -n "${VLLM_PID}" && "${KEEP_VLLM}" != "1" ]]; then
        echo
        echo "Stopping vLLM (pid=${VLLM_PID})..."
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

llm_ready() {
    curl -fsS \
        -H "Authorization: Bearer ${LLM_API_KEY}" \
        "http://${LLM_HOST}:${LLM_PORT}/v1/models" >/dev/null 2>&1
}

api_port_available() {
    if [[ -z "${PYTHON_BIN}" ]]; then
        return 0
    fi
    "${PYTHON_BIN}" - "${API_HOST}" "${API_PORT}" <<'PY'
import errno
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError as exc:
    if exc.errno in {errno.EADDRINUSE, errno.EACCES}:
        sys.exit(1)
    print(f"ERROR: unable to validate API port {host}:{port}: {exc}", file=sys.stderr)
    sys.exit(2)
finally:
    sock.close()
PY
}

echo "=================================================="
echo " Full Stack Launcher"
echo " Profile: ${LLM_PROFILE:-default}"
if [[ "${SKIP_VLLM}" == "1" ]]; then
    echo " LLM  : API endpoint ${LLM_BASE_URL}"
    echo " Model: ${LLM_REASONING_MODEL}"
else
    echo " vLLM : http://${LLM_HOST}:${LLM_PORT}"
    echo " Model: ${LLM_REASONING_MODEL}"
fi
echo " API  : http://127.0.0.1:${API_PORT}"
echo " UI   : http://127.0.0.1:${API_PORT}/ui/"
if [[ "${API_RELOAD}" == "1" ]]; then
    echo " Reload: FastAPI enabled for src/ and frontend/"
fi
echo "=================================================="

if [[ "${SKIP_VLLM}" == "1" ]]; then
    echo "Skipping local vLLM startup."
elif llm_ready; then
    echo "vLLM already running."
else
    echo "vLLM is not running. Starting now..."
    (
        cd "${PROJECT_ROOT}"
        bash scripts/start_llm_server.sh
    ) >"${LOG_DIR}/vllm.log" 2>&1 &
    VLLM_PID=$!
    VLLM_STARTED_BY_SCRIPT=1

    for ((i = 1; i <= START_TIMEOUT_SECONDS; i++)); do
        if llm_ready; then
            echo "vLLM is ready."
            break
        fi
        if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
            echo "ERROR: vLLM exited during startup."
            echo "Last logs (${LOG_DIR}/vllm.log):"
            tail -n 120 "${LOG_DIR}/vllm.log" || true
            exit 1
        fi
        if (( i % 15 == 0 )); then
            echo "Waiting for vLLM... ${i}s / ${START_TIMEOUT_SECONDS}s"
        fi
        sleep 1
    done

    if ! llm_ready; then
        echo "ERROR: timeout waiting for vLLM startup."
        echo "Last logs (${LOG_DIR}/vllm.log):"
        tail -n 120 "${LOG_DIR}/vllm.log" || true
        exit 1
    fi
fi

if ! api_port_available; then
    echo "ERROR: API port ${API_PORT} is already in use or unavailable."
    echo "If this is an existing Agent run, open: http://127.0.0.1:${API_PORT}/ui/"
    echo "Otherwise start this stack on another port:"
    echo "  API_PORT=8090 bash scripts/start_siliconflow_stack.sh"
    echo "To find the process using ${API_PORT} on Linux:"
    echo "  lsof -iTCP:${API_PORT} -sTCP:LISTEN -n -P"
    exit 1
fi

echo "Starting FastAPI..."
echo "Open in browser: http://127.0.0.1:${API_PORT}/ui/"
echo
cd "${PROJECT_ROOT}"
UVICORN_RELOAD_ARGS=()
if [[ "${API_RELOAD}" == "1" ]]; then
    UVICORN_RELOAD_ARGS=(--reload --reload-dir "${PROJECT_ROOT}/src" --reload-dir "${PROJECT_ROOT}/frontend")
fi
PYTHONPATH="${PROJECT_ROOT}" \
    "${UVICORN_BIN}" src.api.app:app --host "${API_HOST}" --port "${API_PORT}" "${UVICORN_RELOAD_ARGS[@]}"
