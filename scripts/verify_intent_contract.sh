#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

.venv/bin/pytest tests/test_intent_policy_paraphrases.py
.venv/bin/pytest tests/test_query_plan.py tests/test_evidence_planner.py tests/test_evidence_sufficiency.py
.venv/bin/pytest tests/test_trace_ui.py tests/test_api.py
