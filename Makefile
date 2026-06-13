PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff

.PHONY: lint test test-all smoke eval eval-planning report-eval eval-answer-smoke eval-answer retrieval-eval data-status

lint:
	$(RUFF) check .

test:
	$(PYTEST) -q

test-all:
	$(PYTEST) -q -rs

smoke:
	@echo "Running API smoke. Requires FastAPI at http://localhost:8080 and vLLM at http://localhost:8000/v1."
	@$(PYTHON) -c "import httpx; r=httpx.get('http://localhost:8080/health', timeout=5); r.raise_for_status(); print('API health OK:', r.json())" || (echo "ERROR: FastAPI is not reachable. Start it with: bash scripts/start_full_stack.sh"; exit 1)
	$(PYTHON) scripts/verify_e2e.py

eval:
	@echo "Running release eval. Requires running FastAPI + vLLM and populated local data."
	$(PYTHON) eval/run_phase5_release_eval.py

eval-planning:
	$(PYTHON) eval/run_methodology_intent_eval.py --mode planning

report-eval:
	$(PYTHON) eval/run_report_eval.py

eval-answer-smoke:
	$(PYTHON) eval/run_answer_benchmark.py --benchmark eval/answer_benchmark_v2.jsonl --limit 6

eval-answer:
	$(PYTHON) eval/run_answer_benchmark.py --benchmark eval/answer_benchmark_v2.jsonl

retrieval-eval:
	@echo "Running retrieval gold eval. Requires local embedding model cache and populated Chroma vectorstore."
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) eval/run_retrieval_eval.py --gold eval/retrieval_gold.jsonl

data-status:
	$(PYTHON) scripts/data_status.py
