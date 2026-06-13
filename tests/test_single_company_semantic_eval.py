import json
from pathlib import Path

from src.agent.query_understanding import build_query_understanding


EVAL_PATH = Path("eval/single_company_fundamental_semantic_eval.jsonl")


def _rows():
    with EVAL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def test_single_company_fundamental_semantic_eval_fixture_is_covered_by_query_understanding():
    rows = list(_rows())
    assert 20 <= len(rows) <= 40

    for row in rows:
        understanding = build_query_understanding(row["query"])
        tickers = [company.ticker for company in understanding.companies]

        assert tickers == row["expected_tickers"], row["id"]
        assert understanding.analysis_scope == row["expected_analysis_scope"], row["id"]
        assert understanding.safety_intent == row["expected_safety_intent"], row["id"]
        if "expected_methodology_intent" in row:
            assert understanding.methodology_intent == row["expected_methodology_intent"], row["id"]
        if row.get("expected_needs_clarification") is not None:
            assert understanding.needs_clarification is row["expected_needs_clarification"], row["id"]

        for dimension in row["expected_dimensions"]:
            assert dimension in understanding.requested_dimensions, row["id"]
        for dimension in row.get("must_not_dimensions", []):
            assert dimension not in understanding.requested_dimensions, row["id"]
