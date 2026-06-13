"""Unit tests for eval metrics — no LLM or DB needed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    answer_accuracy,
    citation_accuracy,
    citation_required_accuracy,
    correctness_summary,
    event_alignment_accuracy,
    event_numeric_accuracy,
    event_tool_success_rate,
    event_window_accuracy,
    latency_score,
    market_reaction_block_presence_rate,
    numeric_fact_accuracy,
    numerical_accuracy,
    period_accuracy,
    retrieval_hit_rate,
    tool_success_rate,
)


class TestAnswerAccuracy:

    def test_full_match(self):
        score = answer_accuracy("Apple revenue was 143.8 billion", "143.8 billion revenue")
        assert score >= 0.5

    def test_no_match(self):
        assert answer_accuracy("nothing relevant here", "Apple revenue growth") < 0.5

    def test_numeric_expected(self):
        assert answer_accuracy("Revenue: 143756000000", "143756000000") == 1.0

    def test_empty_answer(self):
        assert answer_accuracy("", "some answer") == 0.0

    def test_empty_expected(self):
        assert answer_accuracy("some answer", "") == 0.0


class TestNumericalAccuracy:

    def test_exact_match(self):
        assert numerical_accuracy("Revenue was 143756000000 USD", [143756000000]) == 1.0

    def test_scaled_billions(self):
        assert numerical_accuracy("Revenue was 143.8 billion", [143756000000]) == 1.0

    def test_percentage(self):
        assert numerical_accuracy("Margin was 35.37%", [0.3537]) == 1.0

    def test_no_match(self):
        assert numerical_accuracy("No numbers here", [100.0]) == 0.0

    def test_empty_key_numbers(self):
        assert numerical_accuracy("anything", []) == 1.0

    def test_multiple_numbers_partial(self):
        score = numerical_accuracy("Revenue 100 and margin 50%", [100.0, 200.0])
        assert score == 0.5

    def test_tolerance(self):
        assert numerical_accuracy("Value: 100.5", [100.0]) == 1.0


class TestRetrievalHitRate:

    def test_hit(self):
        docs = [{"ticker": "AAPL", "form_type": "10-K", "fiscal_period": "2024", "section": "ITEM_1"}]
        assert retrieval_hit_rate(docs, ["AAPL 10-K"]) == 1.0

    def test_miss(self):
        docs = [{"ticker": "MSFT", "form_type": "10-Q", "fiscal_period": "2024", "section": "ITEM_1"}]
        assert retrieval_hit_rate(docs, ["AAPL 10-K"]) == 0.0

    def test_empty_expected(self):
        assert retrieval_hit_rate([], []) == 1.0

    def test_partial(self):
        docs = [{"ticker": "AAPL", "form_type": "10-K", "fiscal_period": "", "section": ""}]
        assert retrieval_hit_rate(docs, ["AAPL 10-K", "MSFT 10-K"]) == 0.5


class TestCitationAccuracy:

    def test_hit(self):
        cites = [{"ticker": "TSLA", "form_type": "10-K", "fiscal_period": "2025"}]
        assert citation_accuracy(cites, ["TSLA 10-K"]) == 1.0

    def test_miss(self):
        cites = [{"ticker": "AAPL", "form_type": "10-Q", "fiscal_period": "2024"}]
        assert citation_accuracy(cites, ["TSLA 10-K"]) == 0.0

    def test_empty(self):
        assert citation_accuracy([], []) == 1.0


class TestToolSuccessRate:

    def test_all_ok(self):
        results = [{"tool": "a"}, {"tool": "b", "data": {}}]
        assert tool_success_rate(results) == 1.0

    def test_one_error(self):
        results = [{"tool": "a"}, {"tool": "b", "error": "fail"}]
        assert tool_success_rate(results) == 0.5

    def test_empty(self):
        assert tool_success_rate([]) == 1.0


class TestLatencyScore:

    def test_under_budget(self):
        assert latency_score(30.0, budget=60.0) == 1.0

    def test_at_budget(self):
        assert latency_score(60.0, budget=60.0) == 1.0

    def test_over_budget(self):
        score = latency_score(120.0, budget=60.0)
        assert 0.0 < score < 1.0

    def test_triple_budget(self):
        assert latency_score(180.0, budget=60.0) == 0.0


class TestEventMetrics:

    def test_event_alignment_required(self):
        score = event_alignment_accuracy(
            event_results=[
                {
                    "data": {
                        "events": [
                            {"event_type": "10Q"},
                            {"event_type": "10Q"},
                        ]
                    }
                }
            ],
            expected_event_type="10Q",
            expected_latest_n=4,
            required=True,
        )
        assert score == 1.0

    def test_event_alignment_missing_when_required(self):
        score = event_alignment_accuracy(
            event_results=[],
            expected_event_type="10Q",
            expected_latest_n=1,
            required=True,
        )
        assert score == 0.0

    def test_event_numeric_accuracy(self):
        output = {
            "market_reaction": {
                "events": [
                    {"return_1d": 0.02, "return_5d": 0.04},
                ]
            }
        }
        score = event_numeric_accuracy(output=output, key_numbers=[0.02, 0.04], required=True)
        assert score == 1.0

    def test_event_tool_success_rate(self):
        score = event_tool_success_rate(
            tool_results=[
                {"tool": "query_event_price_window", "data": {}},
                {"tool": "query_event_price_window", "error": "x"},
            ],
            required=True,
        )
        assert score == 0.5

    def test_market_reaction_block_presence(self):
        output = {"market_reaction": {"title": "Market Reaction", "events": []}}
        assert market_reaction_block_presence_rate(output, required=True) == 1.0


class TestStructuredNumericFactAccuracy:

    def _fact(self, **overrides):
        fact = {
            "ticker": "AAPL",
            "metric": "revenue",
            "period_type": "quarterly",
            "period_end": "2025-12-27",
            "expected_value": 143756000000,
            "unit": "USD",
            "tolerance_pct": 0.01,
        }
        fact.update(overrides)
        return fact

    def _trace(self, **overrides):
        row = {
            "ticker": "AAPL",
            "metric": "revenue",
            "period_type": "quarterly",
            "period_end": "2025-12-27",
            "value": 143756000000,
            "unit": "USD",
        }
        row.update(overrides)
        return {"numeric_evidence": [row]}

    def test_full_match(self):
        score, failures = numeric_fact_accuracy({}, self._trace(), [self._fact()])
        assert score == 1.0
        assert failures == []

    def test_wrong_value(self):
        score, failures = numeric_fact_accuracy({}, self._trace(value=100), [self._fact()])
        assert score == 0.0
        assert failures[0]["type"] == "wrong_value"

    def test_wrong_period(self):
        score, failures = numeric_fact_accuracy({}, self._trace(period_end="2024-12-28"), [self._fact()])
        assert score == 0.0
        assert failures[0]["type"] == "wrong_period"

    def test_missing_metric(self):
        score, failures = numeric_fact_accuracy({}, self._trace(metric="net_income"), [self._fact()])
        assert score == 0.0
        assert failures[0]["type"] == "missing_expected_fact"


class TestStructuredPeriodAccuracy:

    def test_numeric_period_match(self):
        fact = {
            "ticker": "MSFT",
            "metric": "revenue",
            "period_type": "quarterly",
            "period_end": "2025-12-31",
        }
        trace = {
            "numeric_evidence": [
                {
                    "ticker": "MSFT",
                    "metric": "revenue",
                    "period_type": "quarterly",
                    "period_end": "2025-12-31",
                }
            ]
        }
        score, failures = period_accuracy({}, trace, [fact], [])
        assert score == 1.0
        assert failures == []

    def test_numeric_period_wrong(self):
        fact = {
            "ticker": "MSFT",
            "metric": "revenue",
            "period_type": "quarterly",
            "period_end": "2025-12-31",
        }
        trace = {
            "numeric_evidence": [
                {
                    "ticker": "MSFT",
                    "metric": "revenue",
                    "period_type": "quarterly",
                    "period_end": "2024-12-31",
                }
            ]
        }
        score, failures = period_accuracy({}, trace, [fact], [])
        assert score == 0.0
        assert failures[0]["type"] == "wrong_period"

    def test_comparison_partial_period_match_not_full_credit(self):
        facts = [
            {"ticker": "AAPL", "metric": "revenue", "period_type": "quarterly", "period_end": "2025-12-27"},
            {"ticker": "MSFT", "metric": "revenue", "period_type": "quarterly", "period_end": "2025-12-31"},
        ]
        trace = {
            "numeric_evidence": [
                {"ticker": "AAPL", "metric": "revenue", "period_type": "quarterly", "period_end": "2025-12-27"},
                {"ticker": "MSFT", "metric": "revenue", "period_type": "quarterly", "period_end": "2024-12-31"},
            ]
        }
        score, failures = period_accuracy({}, trace, facts, [])
        assert score == 0.5
        assert failures[0]["type"] == "wrong_period"

    def test_event_date_match(self):
        expected_event = {
            "ticker": "AAPL",
            "event_type": "10Q",
            "event_date": "2025-08-01",
            "fiscal_period": "2025-06-28",
        }
        trace = {
            "event_results": [
                {
                    "data": {
                        "events": [
                            {
                                "ticker": "AAPL",
                                "event_type": "10Q",
                                "event_date": "2025-08-01",
                                "fiscal_period": "2025-06-28",
                            }
                        ]
                    }
                }
            ]
        }
        score, failures = period_accuracy({}, trace, [], [expected_event])
        assert score == 1.0
        assert failures == []


class TestCitationRequiredAccuracy:

    def test_required_but_empty(self):
        score, failures = citation_required_accuracy({}, [], requires_citation=True)
        assert score == 0.0
        assert failures[0]["type"] == "missing_citation"

    def test_section_wrong(self):
        expected = [
            {
                "source_kind": "document",
                "ticker": "TSLA",
                "form_type": "10-K",
                "section": "ITEM_1A",
                "fiscal_period": "2025",
                "must_include_terms": ["risk"],
            }
        ]
        trace = {
            "text_citations": [
                {
                    "source_kind": "document",
                    "ticker": "TSLA",
                    "form_type": "10-K",
                    "section": "ITEM_7",
                    "fiscal_period": "2025",
                    "text_snippet": "risk competition",
                }
            ]
        }
        score, failures = citation_required_accuracy(trace, expected, requires_citation=True)
        assert score == 0.0
        assert failures[0]["type"] == "retrieval_wrong_section"

    def test_terms_missing(self):
        expected = [
            {
                "source_kind": "document",
                "ticker": "TSLA",
                "form_type": "10-K",
                "section": "ITEM_1A",
                "fiscal_period": "2025",
                "must_include_terms": ["competition"],
            }
        ]
        trace = {
            "text_citations": [
                {
                    "source_kind": "document",
                    "ticker": "TSLA",
                    "form_type": "10-K",
                    "section": "ITEM_1A",
                    "fiscal_period": "2025",
                    "text_snippet": "supply chain risk",
                }
            ]
        }
        score, failures = citation_required_accuracy(trace, expected, requires_citation=True)
        assert score == 0.0
        assert failures[0]["type"] == "missing_citation"

    def test_match(self):
        expected = [
            {
                "source_kind": "document",
                "ticker": "TSLA",
                "form_type": "10-K",
                "section": "ITEM_1A",
                "fiscal_period": "2025",
                "must_include_terms": ["competition"],
            }
        ]
        trace = {
            "text_citations": [
                {
                    "source_kind": "document",
                    "ticker": "TSLA",
                    "form_type": "10-K",
                    "section": "ITEM_1A",
                    "fiscal_period": "2025",
                    "text_snippet": "competition and regulatory risk",
                }
            ]
        }
        score, failures = citation_required_accuracy(trace, expected, requires_citation=True)
        assert score == 1.0
        assert failures == []


class TestStructuredEventWindowAccuracy:

    def _event(self, **overrides):
        event = {
            "ticker": "AAPL",
            "event_type": "10Q",
            "event_date": "2025-08-01",
            "fiscal_period": "2025-06-28",
            "return_1d": 0.0123,
            "return_5d": -0.0184,
        }
        event.update(overrides)
        return event

    def _expected(self, **overrides):
        event = {
            "ticker": "AAPL",
            "event_type": "10Q",
            "event_date": "2025-08-01",
            "fiscal_period": "2025-06-28",
            "returns": {
                "return_1d": 0.0123,
                "return_5d": -0.0184,
            },
            "tolerance_abs": 0.0005,
        }
        event.update(overrides)
        return event

    def test_match(self):
        output = {"market_reaction": {"events": [self._event()]}}
        score, failures = event_window_accuracy(output, {}, [self._expected()])
        assert score == 1.0
        assert failures == []

    def test_wrong_event_date(self):
        output = {"market_reaction": {"events": [self._event(event_date="2025-07-31")]}}
        score, failures = event_window_accuracy(output, {}, [self._expected()])
        assert score == 0.0
        assert failures[0]["type"] == "wrong_event_date"

    def test_wrong_event_return(self):
        output = {"market_reaction": {"events": [self._event(return_5d=0.02)]}}
        score, failures = event_window_accuracy(output, {}, [self._expected()])
        assert score == 0.0
        assert failures[0]["type"] == "wrong_event_return"


class TestCorrectnessSummary:

    def test_failure_reason_drives_summary(self):
        summary = correctness_summary(
            {
                "answer_accuracy": 1.0,
                "numerical_accuracy": 1.0,
                "numeric_fact_accuracy": 0.0,
                "period_accuracy": 1.0,
                "citation_required_accuracy": 1.0,
                "event_window_accuracy": 1.0,
                "failure_reasons": [{"type": "wrong_value", "message": "bad"}],
            }
        )
        assert summary["correctness_pass"] is False
        assert summary["primary_failure_type"] == "wrong_value"


class TestBenchmarkLoad:

    def test_load_and_count(self):
        from eval.run_eval import load_benchmark
        items = load_benchmark(ROOT / "eval" / "benchmark.jsonl")
        assert len(items) >= 50
        assert all("question" in i for i in items)
        assert all("id" in i for i in items)

    def test_filter_category(self):
        from eval.run_eval import load_benchmark
        items = load_benchmark(ROOT / "eval" / "benchmark.jsonl", category="fact_qa")
        assert all(i["category"] == "fact_qa" for i in items)
        assert len(items) > 0

    def test_filter_ids(self):
        from eval.run_eval import load_benchmark
        items = load_benchmark(ROOT / "eval" / "benchmark.jsonl", ids={"fq01", "tr01"})
        assert len(items) == 2
        assert {i["id"] for i in items} == {"fq01", "tr01"}

    def test_limit(self):
        from eval.run_eval import load_benchmark
        items = load_benchmark(ROOT / "eval" / "benchmark.jsonl", limit=3)
        assert len(items) == 3

    def test_v2_schema_when_present(self):
        paths = [
            ROOT / "eval" / "benchmark.jsonl",
            ROOT / "docs" / "archive" / "baselines" / "pre_change_20260415" / "baseline_questions_25.jsonl",
            ROOT / "eval" / "benchmark_event_v11.jsonl",
        ]
        for path in paths:
            rows = [
                __import__("json").loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for row in rows:
                has_v2 = any(k in row for k in ("expected_facts", "expected_text_evidence", "expected_events"))
                if not has_v2:
                    continue
                assert row.get("expected_facts") or row.get("expected_text_evidence") or row.get("expected_events")
                if row.get("requires_citation") and row.get("category") != "event_reaction":
                    assert row.get("expected_text_evidence")
                if row.get("category") == "event_reaction" and row.get("returns_required", True):
                    assert row.get("expected_events")
