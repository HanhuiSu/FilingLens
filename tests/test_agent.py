"""Agent integration tests — Phase 4.

Tests the LangGraph agent workflow:
  - Graph compilation
  - Node function contracts
  - End-to-end (requires vLLM + DuckDB + ChromaDB)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Graph structure tests (no LLM / DB needed)
# ---------------------------------------------------------------------------


class TestGraphCompilation:

    def test_compile(self):
        from src.agent.graph import compile_agent
        app = compile_agent()
        assert app is not None
        assert hasattr(app, "invoke")

    def test_nodes_present(self):
        from src.agent.graph import build_graph
        g = build_graph()
        assert "classify" in g.nodes
        assert "execute_tools" in g.nodes
        assert "evaluate" in g.nodes
        assert "generate" in g.nodes

    def test_state_keys(self):
        from src.agent.state import AgentState
        keys = set(AgentState.__annotations__.keys())
        required = {
            "user_query", "task_type", "companies", "data_route",
            "tool_results", "retrieved_docs", "final_answer",
            "trace_id", "messages",
        }
        assert required.issubset(keys)


# ---------------------------------------------------------------------------
# Node function unit tests
# ---------------------------------------------------------------------------


class TestSelectTools:

    def test_documents_only(self):
        from src.agent.nodes import _select_tools
        tools = _select_tools("report_summary", "documents_only")
        assert tools == ["search_filings"]

    def test_structured_only_fact(self):
        from src.agent.nodes import _select_tools
        tools = _select_tools("fact_qa", "structured_only")
        assert tools == ["query_financial_data"]

    def test_structured_with_compute(self):
        from src.agent.nodes import _select_tools
        tools = _select_tools("trend_analysis", "structured_only")
        assert "query_financial_data" in tools
        assert "compute_metrics" in tools

    def test_hybrid(self):
        from src.agent.nodes import _select_tools
        tools = _select_tools("fact_qa", "hybrid")
        assert "search_filings" in tools
        assert "query_financial_data" in tools


class TestInferPeriodType:

    def test_quarterly_keywords(self):
        from src.agent.nodes import _infer_period_type
        assert _infer_period_type({"user_query": "Apple Q3 revenue"}) == "quarterly"
        assert _infer_period_type({"user_query": "最近几个季度"}) == "quarterly"

    def test_annual_keywords(self):
        from src.agent.nodes import _infer_period_type
        assert _infer_period_type({"user_query": "annual revenue"}) == "annual"
        assert _infer_period_type({"user_query": "年度营收"}) == "annual"

    def test_no_hint(self):
        from src.agent.nodes import _infer_period_type
        assert _infer_period_type({"user_query": "revenue trend"}) is None


class TestEvaluateEvidence:

    def test_docs_sufficient_for_documents_only(self):
        from src.agent.nodes import evaluate_evidence
        state = {
            "data_route": "documents_only",
            "tool_results": [],
            "retrieved_docs": [{"text": "x"}],
            "evidence_loop_count": 1,
        }
        out = evaluate_evidence(state)
        assert out["evidence_sufficient"] is True

    def test_insufficient_when_empty(self):
        from src.agent.nodes import evaluate_evidence
        state = {
            "data_route": "hybrid",
            "tool_results": [],
            "retrieved_docs": [],
            "evidence_loop_count": 0,
        }
        out = evaluate_evidence(state)
        assert out["evidence_sufficient"] is False

    def test_needs_tools_false_is_sufficient(self):
        from src.agent.nodes import evaluate_evidence
        state = {
            "needs_tools": False,
            "data_route": "hybrid",
            "tool_results": [],
            "retrieved_docs": [],
            "evidence_loop_count": 0,
        }
        out = evaluate_evidence(state)
        assert out["evidence_sufficient"] is True

    def test_max_loops_forces_sufficient(self):
        from src.agent.nodes import evaluate_evidence
        state = {
            "data_route": "hybrid",
            "tool_results": [],
            "retrieved_docs": [],
            "evidence_loop_count": 2,
        }
        out = evaluate_evidence(state)
        assert out["evidence_sufficient"] is True


class TestCheckEvidence:

    def test_sufficient_routes_generate(self):
        from src.agent.nodes import check_evidence
        assert check_evidence({"evidence_sufficient": True}) == "sufficient"

    def test_insufficient_routes_execute(self):
        from src.agent.nodes import check_evidence
        assert check_evidence({"evidence_sufficient": False}) == "insufficient"


class TestParseJson:

    def test_clean_json(self):
        from src.agent.nodes import _parse_json_response
        r = _parse_json_response('{"task_type": "fact_qa"}')
        assert r["task_type"] == "fact_qa"

    def test_json_with_think_tags(self):
        from src.agent.nodes import _parse_json_response
        r = _parse_json_response('<think>some reasoning</think>{"task_type":"trend_analysis"}')
        assert r["task_type"] == "trend_analysis"

    def test_json_with_surrounding_text(self):
        from src.agent.nodes import _parse_json_response
        r = _parse_json_response('Here is the result: {"x": 1} done.')
        assert r["x"] == 1

    def test_empty_returns_empty(self):
        from src.agent.nodes import _parse_json_response
        assert _parse_json_response("") == {}
        assert _parse_json_response("no json here") == {}


class TestConversationalAnswering:

    def test_meta_answer_short_circuits_without_citations(self):
        from src.agent import nodes

        out = nodes.generate_answer(
            {
                "trace_id": "trace-meta-1",
                "user_query": "你是谁？",
                "task_type": "fact_qa",
                "answer_mode": "meta",
                "safety_intent": "normal",
                "needs_tools": False,
                "selected_tools": [],
                "tool_results": [],
                "retrieved_docs": [],
            }
        )

        assert "财报" in out["final_answer"]
        assert out["citations"] == []
        assert out["output"]["answer_mode"] == "meta"
        assert out["output"]["needs_tools"] is False
        assert out["output"]["used_tools"] == []
        assert out["output"]["view"]["kind"] == "meta_response"
        assert out["synthesis_strategy"] == "conversational_short_circuit"
        assert out["why_tools_skipped"]

    def test_capabilities_meta_response_mentions_scope_and_limits(self):
        from src.agent import nodes

        out = nodes.generate_answer(
            {
                "trace_id": "trace-meta-2",
                "user_query": "你能做什么？",
                "task_type": "fact_qa",
                "answer_mode": "meta",
                "safety_intent": "normal",
                "needs_tools": False,
                "selected_tools": [],
                "tool_results": [],
                "retrieved_docs": [],
            }
        )

        assert out["output"]["view"]["kind"] == "meta_response"
        assert "SEC" in out["final_answer"]
        assert "结构化财务指标" in out["final_answer"]
        assert "投资建议" in out["final_answer"]
        assert out["output"]["numeric_evidence"] == []

    def test_clarification_answer_asks_question_without_tools(self):
        from src.agent import nodes

        out = nodes.generate_answer(
            {
                "trace_id": "trace-clarify-1",
                "user_query": "帮我分析一下",
                "task_type": "fact_qa",
                "answer_mode": "clarification",
                "safety_intent": "normal",
                "needs_clarification": True,
                "clarification_question": "你想分析哪家公司？",
                "needs_tools": False,
                "selected_tools": [],
                "tool_results": [],
                "retrieved_docs": [],
            }
        )

        assert out["final_answer"] == "你想分析哪家公司？"
        assert out["citations"] == []
        assert out["output"]["answer_mode"] == "clarification"
        assert out["output"]["needs_clarification"] is True
        assert out["output"]["view"]["kind"] == "clarification"
        assert out["output"]["view"]["example_questions"]

    def test_stock_price_prediction_refusal_does_not_fabricate_forecast(self):
        from src.agent import nodes

        out = nodes.generate_answer(
            {
                "trace_id": "trace-stock-predict",
                "user_query": "预测明天苹果股价",
                "task_type": "fact_qa",
                "answer_mode": "refusal_or_redirect",
                "safety_intent": "unsupported_or_out_of_scope",
                "needs_tools": False,
                "selected_tools": [],
                "tool_results": [],
                "retrieved_docs": [],
                "safety_limitations": [
                    {"code": "no_realtime_news_access", "severity": "high"},
                    {"code": "unsupported_price_prediction", "severity": "high"},
                ],
            }
        )

        assert out["output"]["view"]["kind"] == "refusal_or_redirect"
        assert out["final_answer_source"] == "unsupported_or_refusal"
        assert "明天会涨" not in out["final_answer"]
        assert out["output"]["used_tools"] == []
        codes = {item["code"] for item in out["output"]["limitations"]}
        assert "no_realtime_news_access" in codes
        assert "unsupported_price_prediction" in codes

    def test_insufficient_evidence_requirements_prevent_synthesis(self):
        from src.agent import nodes

        out = nodes.generate_answer(
            {
                "trace_id": "trace-req-insufficient",
                "user_query": "AAPL revenue",
                "task_type": "fact_qa",
                "answer_mode": "direct_fact",
                "safety_intent": "normal",
                "needs_tools": True,
                "companies": ["AAPL"],
                "selected_tools": ["query_financial_data"],
                "evidence_plan": {
                    "evidence_requirements": [
                        {
                            "requirement_id": "REQ-NUM-AAPL-FACT",
                            "requirement_type": "numeric",
                            "company": "AAPL",
                            "metrics": ["revenue"],
                            "required": True,
                            "min_results": 1,
                        }
                    ],
                    "rejected_requirements": [],
                },
                "evidence_sufficiency": {
                    "overall_status": "insufficient",
                    "satisfied_requirements": [],
                    "missing_requirements": ["REQ-NUM-AAPL-FACT"],
                    "can_synthesize": False,
                    "degradation_reason": "required_evidence_missing",
                },
                "tool_results": [
                    {
                        "tool": "query_financial_data",
                        "ticker": "AAPL",
                        "data": {
                            "ticker": "AAPL",
                            "financial_facts": [
                                {
                                    "requirement_id": "REQ-NUM-AAPL-FACT",
                                    "ticker": "AAPL",
                                    "metric": "revenue",
                                    "period_type": "annual",
                                    "period_end": "2025-12-31",
                                    "value": 100,
                                    "unit": "USD",
                                }
                            ],
                        },
                    }
                ],
                "retrieved_docs": [],
            }
        )

        assert "100" not in out["final_answer"]
        assert out["synthesis_strategy"] == "synthesis_degraded"
        codes = {item["code"] for item in out["output"]["limitations"]}
        assert "requirement_missing" in codes

    def test_partial_evidence_requirements_add_limitation_and_use_satisfied_evidence(self):
        from src.agent import nodes

        out = nodes.generate_answer(
            {
                "trace_id": "trace-req-partial",
                "user_query": "AAPL revenue and risk",
                "task_type": "fact_qa",
                "answer_mode": "direct_fact",
                "safety_intent": "normal",
                "needs_tools": True,
                "companies": ["AAPL"],
                "selected_tools": ["query_financial_data"],
                "period_query": {"period_type": "annual"},
                "evidence_plan": {
                    "evidence_requirements": [
                        {
                            "requirement_id": "REQ-NUM-AAPL-FACT",
                            "requirement_type": "numeric",
                            "company": "AAPL",
                            "metrics": ["revenue"],
                            "required": True,
                            "min_results": 1,
                        },
                        {
                            "requirement_id": "REQ-TEXT-AAPL-RISK",
                            "requirement_type": "text",
                            "company": "AAPL",
                            "required": True,
                            "min_results": 1,
                        },
                    ],
                    "rejected_requirements": [],
                },
                "evidence_sufficiency": {
                    "overall_status": "partial",
                    "satisfied_requirements": ["REQ-NUM-AAPL-FACT"],
                    "missing_requirements": ["REQ-TEXT-AAPL-RISK"],
                    "can_synthesize": True,
                    "degradation_reason": "text_evidence_missing",
                },
                "tool_results": [
                    {
                        "tool": "query_financial_data",
                        "ticker": "AAPL",
                        "data": {
                            "ticker": "AAPL",
                            "financial_facts": [
                                {
                                    "requirement_id": "REQ-NUM-AAPL-FACT",
                                    "ticker": "AAPL",
                                    "metric": "revenue",
                                    "period_type": "annual",
                                    "period_end": "2025-12-31",
                                    "value": 100,
                                    "unit": "USD",
                                }
                            ],
                        },
                    }
                ],
                "retrieved_docs": [
                    {
                        "requirement_id": "REQ-TEXT-AAPL-RISK",
                        "ticker": "AAPL",
                        "filing_id": "f1",
                        "form_type": "10-K",
                        "fiscal_period": "2025",
                        "section": "ITEM_1A",
                        "chunk_order": 1,
                        "text": "This text should not be used because the requirement is missing.",
                    }
                ],
            }
        )

        assert out["numeric_evidence"]
        assert out["text_evidence"] == []
        codes = {item["code"] for item in out["output"]["limitations"]}
        assert "requirement_missing" in codes

    def test_investment_advice_like_output_adds_non_advisory_boundary(self):
        from src.agent.nodes import _build_phase4_output

        output = _build_phase4_output(
            state={
                "trace_id": "trace-investment-1",
                "task_type": "company_comparison",
                "companies": ["AAPL", "AMZN"],
                "comparison_target": "AMZN",
                "answer_mode": "comparison_brief",
                "safety_intent": "investment_advice_like",
                "needs_tools": True,
                "selected_tools": ["query_financial_data"],
                "validated_tools": ["query_financial_data"],
                "analysis_plan": {
                    "analysis_dimensions": ["comparison"],
                    "needed_evidence": ["numeric"],
                    "validated_tools": ["query_financial_data"],
                },
                "requested_metrics": ["revenue"],
            },
            lang="zh",
            task_type="company_comparison",
            comparison_basis_label="same_period",
            period_query={"period_type": "latest"},
            numeric_claims=[{"sentence": "AAPL revenue is 100.", "evidence_ids": ["N1"]}],
            text_claims=[],
            numeric_evidence=[
                {
                    "evidence_id": "N1",
                    "ticker": "AAPL",
                    "metric": "revenue",
                    "period_type": "quarterly",
                    "period_end": "2025-12-31",
                    "value": 100,
                    "unit": "USD",
                }
            ],
            text_evidence=[],
            numeric_citations=[
                {
                    "evidence_id": "N1",
                    "ticker": "AAPL",
                    "metric": "revenue",
                    "period_type": "quarterly",
                    "period_end": "2025-12-31",
                }
            ],
            text_citations=[],
            unsupported_claims=[],
        )

        codes = {item["code"] for item in output["limitations"]}
        assert output["answer_mode"] == "comparison_brief"
        assert output["safety_intent"] == "investment_advice_like"
        assert "investment_advice_boundary" in codes
        assert output["analysis_plan_summary"]["validated_tools"] == ["query_financial_data"]


class TestTimeRangeGuardrails:

    def test_has_explicit_year(self):
        from src.agent.nodes import _has_explicit_year
        assert _has_explicit_year("Apple 2025 Q3 revenue")
        assert _has_explicit_year("Apple 2025年Q3营收")
        assert not _has_explicit_year("Apple 最近一个季度营收")

    def test_sanitize_drops_stale_range_for_recency(self):
        from src.agent.nodes import _sanitize_time_range_for_recency
        out = _sanitize_time_range_for_recency(
            user_query="Apple 最近一个季度营收是多少？",
            time_range={"start": "2023-07-01", "end": "2023-09-30"},
            today=date(2026, 4, 5),
        )
        assert out is None

    def test_sanitize_clears_recent_looking_range_for_recency(self):
        from src.agent.nodes import _sanitize_time_range_for_recency
        out = _sanitize_time_range_for_recency(
            user_query="Apple 最近一个季度营收是多少？",
            time_range={"start": "2025-10-01", "end": "2025-12-31"},
            today=date(2026, 4, 5),
        )
        assert out is None

    def test_sanitize_keeps_explicit_year(self):
        from src.agent.nodes import _sanitize_time_range_for_recency
        tr = {"start": "2023-07-01", "end": "2023-09-30"}
        out = _sanitize_time_range_for_recency(
            user_query="Apple 2023Q3 营收是多少？",
            time_range=tr,
            today=date(2026, 4, 5),
        )
        assert out == tr

    def test_execute_tools_retry_without_date_filter_for_recency(self, monkeypatch):
        from src.agent import nodes

        calls = []

        class DummyTool:
            @staticmethod
            def invoke(payload):
                calls.append(payload)
                # First call (with stale date range) returns empty.
                if payload.get("date_start") or payload.get("date_end"):
                    return {"ticker": "AAPL", "financial_facts": [], "price_data": []}
                # Retry without date filter returns data.
                return {
                    "ticker": "AAPL",
                    "financial_facts": [
                        {"ticker": "AAPL", "period_end": "2025-12-31", "metric": "revenue", "value": 1.0}
                    ],
                    "price_data": [],
                }

        monkeypatch.setattr(nodes, "query_financial_data", DummyTool())

        out = nodes.execute_tools(
            {
                "user_query": "Apple 最近一个季度营收是多少？",
                "task_type": "fact_qa",
                "companies": ["AAPL"],
                "comparison_target": None,
                "selected_tools": ["query_financial_data"],
                "time_range": {"start": "2023-07-01", "end": "2023-09-30"},
                "requested_metrics": ["revenue"],
                "tool_results": [],
                "retrieved_docs": [],
            }
        )

        assert len(calls) == 2
        assert calls[0]["date_start"] == "2023-07-01"
        assert calls[1]["date_start"] is None
        assert out["tool_results"][0]["data"]["financial_facts"]


class TestPeriodSemanticsPhase2:

    def test_normalize_period_query_for_chinese_quarter(self):
        from src.agent.nodes import _normalize_period_query

        pq = _normalize_period_query(
            user_query="Apple 2025年第一季度营收是多少？",
            raw=None,
            today=date(2026, 4, 15),
            task_type="fact_qa",
        )
        assert pq["period_type"] == "quarterly"
        assert pq["year"] == 2025
        assert pq["quarter"] == 1
        assert pq["year_basis"] == "fiscal"
        assert pq["needs_clarification"] is False

    def test_normalize_period_query_for_q_style(self):
        from src.agent.nodes import _normalize_period_query

        pq = _normalize_period_query(
            user_query="AAPL 2025Q1 revenue",
            raw=None,
            today=date(2026, 4, 15),
            task_type="fact_qa",
        )
        assert pq["period_type"] == "quarterly"
        assert pq["year"] == 2025
        assert pq["quarter"] == 1

    def test_normalize_period_query_for_last_year(self):
        from src.agent.nodes import _normalize_period_query

        pq = _normalize_period_query(
            user_query="Apple 去年的全年营收是多少？",
            raw=None,
            today=date(2026, 4, 15),
            task_type="fact_qa",
        )
        assert pq["period_type"] == "annual"
        assert pq["year"] == 2025
        assert pq["year_basis"] == "fiscal"

    def test_normalize_period_query_for_recent_quarters_default_4(self):
        from src.agent.nodes import _normalize_period_query

        pq = _normalize_period_query(
            user_query="分析 Apple 最近几个季度营收趋势",
            raw=None,
            today=date(2026, 4, 15),
            task_type="trend_analysis",
        )
        assert pq["period_type"] == "trailing"
        assert pq["trailing_n"] == 4

    def test_resolve_query_plan_flags_annual_missing_year(self):
        from src.agent.nodes import _resolve_query_plan

        out = _resolve_query_plan(
            {
                "user_query": "Apple 全年营收是多少？",
                "task_type": "fact_qa",
                "period_query": {
                    "period_type": "annual",
                    "year": None,
                    "year_basis": "fiscal",
                    "comparison_basis": "same_period",
                    "needs_clarification": False,
                    "is_explicit": False,
                },
            }
        )
        assert out["resolved_period_context"]["needs_clarification"] is True

    def test_resolve_query_plan_comparison_default_same_period(self):
        from src.agent.nodes import _resolve_query_plan

        out = _resolve_query_plan(
            {
                "user_query": "比较 Apple 和 Microsoft 最近一个季度营收",
                "task_type": "company_comparison",
                "period_query": {
                    "period_type": "latest",
                    "year_basis": "fiscal",
                    "comparison_basis": "same_period",
                },
            }
        )
        assert out["period_query"]["comparison_basis"] == "same_period"
        assert out["comparison_basis_label"] == "同季度"

    def test_generate_answer_rejects_period_mismatch(self, monkeypatch):
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"numeric_claims":[{"sentence":"AAPL revenue is 100 in 2025Q1.",'
                        '"evidence_ids":["N1"]}],"text_claims":[]}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())
        state = {
            "user_query": "Apple 2025Q1 营收是多少？",
            "task_type": "fact_qa",
            "period_query": {
                "period_type": "quarterly",
                "year": 2025,
                "quarter": 1,
                "year_basis": "fiscal",
                "comparison_basis": "same_period",
                "is_explicit": True,
                "needs_clarification": False,
                "clarification_reason": None,
            },
            "resolved_period_context": {
                "target_period_type": "quarterly",
                "comparison_basis": "same_period",
                "needs_clarification": False,
                "strict_period_match": True,
            },
            "comparison_basis_label": "",
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {
                                "ticker": "AAPL",
                                "metric": "revenue",
                                "period_type": "annual",
                                "period_end": "2025-09-30",
                                "value": 100.0,
                                "unit": "USD",
                                "fiscal_year": 2025,
                                "fiscal_quarter": 4,
                                "calendar_year": 2025,
                                "calendar_quarter": 3,
                            }
                        ],
                        "price_data": [],
                    },
                }
            ],
            "retrieved_docs": [],
        }

        out = nodes.generate_answer(state)
        assert "证据不足" in out["final_answer"] or "Time period is ambiguous" in out["final_answer"] or "时间口径不够明确" in out["final_answer"]
        assert out["unsupported_claims"]
        assert any(
            ("period_type_mismatch" in c.get("reason", ""))
            or ("claim_sentence_period_mismatch_with_evidence" in c.get("reason", ""))
            or ("no_numeric_citation_for_time_check" in c.get("reason", ""))
            for c in out["unsupported_claims"]
        )

    def test_execute_tools_same_period_no_fallback(self, monkeypatch):
        from src.agent import nodes

        class DummyQuery:
            @staticmethod
            def invoke(payload):
                ticker = payload["ticker"]
                if ticker == "AAPL":
                    facts = [
                        {
                            "ticker": "AAPL",
                            "metric": "revenue",
                            "period_type": "quarterly",
                            "period_end": "2025-12-31",
                            "value": 100.0,
                            "fiscal_year": 2026,
                            "fiscal_quarter": 1,
                            "calendar_year": 2025,
                            "calendar_quarter": 4,
                        }
                    ]
                else:
                    facts = [
                        {
                            "ticker": "MSFT",
                            "metric": "revenue",
                            "period_type": "quarterly",
                            "period_end": "2025-09-30",
                            "value": 90.0,
                            "fiscal_year": 2026,
                            "fiscal_quarter": 1,
                            "calendar_year": 2025,
                            "calendar_quarter": 3,
                        }
                    ]
                return {"ticker": ticker, "financial_facts": facts, "price_data": [], "period_context": {}}

        monkeypatch.setattr(nodes, "query_financial_data", DummyQuery())
        out = nodes.execute_tools(
            {
                "user_query": "比较 Apple 和 Microsoft 最近一个季度营收",
                "task_type": "company_comparison",
                "companies": ["AAPL", "MSFT"],
                "comparison_target": "MSFT",
                "selected_tools": ["query_financial_data"],
                "requested_metrics": ["revenue"],
                "tool_results": [],
                "retrieved_docs": [],
                "period_query": {
                    "period_type": "latest",
                    "year_basis": "fiscal",
                    "comparison_basis": "same_period",
                },
                "resolved_period_context": {
                    "target_period_type": "quarterly",
                    "comparison_basis": "same_period",
                    "needs_clarification": False,
                    "strict_period_match": True,
                },
            }
        )
        assert out["resolved_period_context"]["same_period_match"] is False
        assert out["resolved_period_context"]["common_periods"] == []

    def test_generate_answer_adds_annual_basis_line(self, monkeypatch):
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"numeric_claims":[{"sentence":"AAPL 2025财年营收为 100。",'
                        '"evidence_ids":["N1"]}],"text_claims":[]}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())
        state = {
            "user_query": "Apple 2025年全年营收是多少？",
            "task_type": "fact_qa",
            "period_query": {
                "period_type": "annual",
                "year": 2025,
                "quarter": None,
                "year_basis": "fiscal",
                "comparison_basis": "same_period",
                "is_explicit": True,
                "needs_clarification": False,
                "clarification_reason": None,
            },
            "resolved_period_context": {
                "target_period_type": "annual",
                "comparison_basis": "same_period",
                "needs_clarification": False,
                "strict_period_match": True,
            },
            "comparison_basis_label": "",
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {
                                "ticker": "AAPL",
                                "metric": "revenue",
                                "period_type": "annual",
                                "period_end": "2025-09-30",
                                "value": 100.0,
                                "unit": "USD",
                                "fiscal_year": 2025,
                                "fiscal_quarter": 4,
                                "calendar_year": 2025,
                                "calendar_quarter": 3,
                            }
                        ],
                        "price_data": [],
                    },
                }
            ],
            "retrieved_docs": [],
        }

        out = nodes.generate_answer(state)
        assert "时间口径：财年" in out["final_answer"]

    def test_execute_tools_clarifies_when_fiscal_calendar_unknown(self, monkeypatch):
        from src.agent import nodes

        class DummyQuery:
            @staticmethod
            def invoke(_payload):
                return {
                    "ticker": "AAPL",
                    "financial_facts": [],
                    "price_data": [],
                    "period_context": {"fiscal_year_end_month": None},
                }

        monkeypatch.setattr(nodes, "query_financial_data", DummyQuery())
        out = nodes.execute_tools(
            {
                "user_query": "Apple 2025Q1 营收是多少？",
                "task_type": "fact_qa",
                "companies": ["AAPL"],
                "comparison_target": None,
                "selected_tools": ["query_financial_data"],
                "requested_metrics": ["revenue"],
                "tool_results": [],
                "retrieved_docs": [],
                "period_query": {
                    "period_type": "quarterly",
                    "year": 2025,
                    "quarter": 1,
                    "year_basis": "fiscal",
                    "comparison_basis": "same_period",
                },
                "resolved_period_context": {
                    "target_period_type": "quarterly",
                    "comparison_basis": "same_period",
                    "needs_clarification": False,
                    "strict_period_match": True,
                },
            }
        )
        assert out["resolved_period_context"]["needs_clarification"] is True
        assert "fiscal_year_end_unknown_for_AAPL" in str(out["resolved_period_context"]["clarification_reason"])

    def test_generate_answer_uses_deterministic_numeric_claim_even_with_calendar_period_end(self, monkeypatch):
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"text_claims":[]}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())
        state = {
            "user_query": "Apple 2025Q1 营收是多少？",
            "task_type": "fact_qa",
            "period_query": {
                "period_type": "quarterly",
                "year": 2025,
                "quarter": 1,
                "year_basis": "fiscal",
                "comparison_basis": "same_period",
                "is_explicit": True,
                "needs_clarification": False,
                "clarification_reason": None,
            },
            "resolved_period_context": {
                "target_period_type": "quarterly",
                "comparison_basis": "same_period",
                "needs_clarification": False,
                "strict_period_match": True,
            },
            "comparison_basis_label": "",
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {
                                "ticker": "AAPL",
                                "metric": "revenue",
                                "period_type": "quarterly",
                                "period_end": "2024-12-31",
                                "value": 100.0,
                                "unit": "USD",
                                "fiscal_year": 2025,
                                "fiscal_quarter": 1,
                                "calendar_year": 2024,
                                "calendar_quarter": 4,
                            }
                        ],
                        "price_data": [],
                    },
                }
            ],
            "retrieved_docs": [],
        }

        out = nodes.generate_answer(state)
        assert out["numeric_citations"]
        assert not any(
            "claim_sentence_period_mismatch_with_evidence" in c.get("reason", "")
            for c in out["unsupported_claims"]
        )


class TestComparisonGuardrails:

    def test_classify_upgrades_multi_company_fact(self, monkeypatch):
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"task_type":"fact_qa","companies":["AAPL","amzn"],'
                        '"comparison_target":null,"time_range":null,'
                        '"requested_metrics":[],"data_route":"hybrid"}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())

        out = nodes.classify_and_extract({"user_query": "综合对比分析下apple和amzn"})
        assert out["task_type"] == "company_comparison"
        assert out["comparison_target"] == "AMZN"
        assert out["requested_metrics"] == ["revenue", "net_income"]
        assert "compute_metrics" in out["selected_tools"]

    def test_sample_docs_for_prompt_balances_tickers(self):
        from src.agent.nodes import _sample_docs_for_prompt

        docs = []
        for i in range(8):
            docs.append(
                {
                    "ticker": "AAPL",
                    "filing_id": f"a{i}",
                    "fiscal_period": "2025-12-31",
                    "section": "MIXED",
                    "chunk_order": i,
                }
            )
        for i in range(8):
            docs.append(
                {
                    "ticker": "AMZN",
                    "filing_id": f"m{i}",
                    "fiscal_period": "2025-12-31",
                    "section": "MIXED",
                    "chunk_order": i,
                }
            )

        sampled = _sample_docs_for_prompt(docs, max_chunks=8)
        sampled_tickers = {d["ticker"] for d in sampled}
        assert len(sampled) == 8
        assert {"AAPL", "AMZN"}.issubset(sampled_tickers)
        assert sampled[0]["ticker"] == "AAPL"
        assert sampled[1]["ticker"] == "AMZN"

    def test_format_fact_qa_answer_for_multi_company(self):
        from src.agent.nodes import _format_fact_qa_answer

        state = {
            "companies": ["AAPL", "AMZN"],
            "comparison_target": "AMZN",
            "requested_metrics": ["revenue"],
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {
                                "ticker": "AAPL",
                                "metric": "revenue",
                                "period_end": "2025-12-31",
                                "period_type": "quarterly",
                                "value": 143756000000.0,
                            }
                        ],
                        "price_data": [],
                    },
                },
                {
                    "tool": "query_financial_data",
                    "ticker": "AMZN",
                    "data": {
                        "ticker": "AMZN",
                        "financial_facts": [
                            {
                                "ticker": "AMZN",
                                "metric": "revenue",
                                "period_end": "2025-12-31",
                                "period_type": "quarterly",
                                "value": 213386000000.0,
                            }
                        ],
                        "price_data": [],
                    },
                },
            ],
        }
        citations = [
            {"ticker": "AAPL", "form_type": "10-Q", "fiscal_period": "2025-12-31", "section": "MIXED"},
            {"ticker": "AMZN", "form_type": "10-Q", "fiscal_period": "2025-12-31", "section": "MIXED"},
        ]

        answer = _format_fact_qa_answer(state, cleaned_answer="", lang="zh", citations=citations)
        assert "结论：" in answer
        assert "AAPL" in answer
        assert "AMZN" in answer
        assert "来源：" in answer


@pytest.mark.skipif(not Path(ROOT / "data" / "db" / "financial.duckdb").exists(), reason="DuckDB not populated")
class TestRecencyRegression:

    def test_real_db_latest_quarter_revenue_ignores_inferred_recency_range(self, monkeypatch):
        duckdb = pytest.importorskip("duckdb")
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"task_type":"fact_qa","companies":["AAPL"],'
                        '"comparison_target":null,'
                        '"time_range":{"start":"2023-07-01","end":"2023-09-30"},'
                        '"requested_metrics":["revenue"],'
                        '"data_route":"structured_only"}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())

        state = nodes.classify_and_extract({"user_query": "Apple 最近一个季度的营收是多少？"})
        assert state["time_range"] is None

        out = nodes.execute_tools(state)
        tr = next(
            item
            for item in out["tool_results"]
            if item.get("tool") == "query_financial_data" and item.get("ticker") == "AAPL"
        )
        facts = tr["data"]["financial_facts"]
        db_path = ROOT / "data" / "db" / "financial.duckdb"
        with duckdb.connect(str(db_path), read_only=True) as conn:
            expected_period_end, expected_value = conn.execute(
                """
                SELECT period_end, value
                FROM financial_facts
                WHERE ticker = 'AAPL'
                  AND metric = 'revenue'
                  AND period_type = 'quarterly'
                ORDER BY period_end DESC
                LIMIT 1
                """
            ).fetchone()
        expected_period_end = expected_period_end.isoformat() if hasattr(expected_period_end, "isoformat") else str(expected_period_end)

        assert facts
        assert facts[0]["metric"] == "revenue"
        assert facts[0]["period_type"] == "quarterly"
        assert facts[0]["period_end"] == expected_period_end
        assert facts[0]["value"] == pytest.approx(float(expected_value))


class TestComparisonAlignmentAndCitationBinding:

    def test_latest_comparable_pair_annual_same_year(self):
        from src.agent.nodes import _latest_comparable_pair

        rows = [
            {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2025-09-30", "value": 1},
            {"ticker": "AMZN", "metric": "revenue", "period_type": "annual", "period_end": "2025-12-31", "value": 2},
        ]
        pair = _latest_comparable_pair(rows, "AAPL", "AMZN", "revenue", "annual")
        assert pair is not None
        left, right = pair
        assert left["period_end"].startswith("2025")
        assert right["period_end"].startswith("2025")

    def test_latest_comparable_pair_quarterly_same_period_end(self):
        from src.agent.nodes import _latest_comparable_pair

        rows = [
            {"ticker": "AAPL", "metric": "net_income", "period_type": "quarterly", "period_end": "2025-12-31", "value": 1},
            {"ticker": "AMZN", "metric": "net_income", "period_type": "quarterly", "period_end": "2025-12-31", "value": 2},
        ]
        pair = _latest_comparable_pair(rows, "AAPL", "AMZN", "net_income", "quarterly")
        assert pair is not None
        left, right = pair
        assert left["period_end"] == right["period_end"] == "2025-12-31"

    def test_latest_comparable_pair_none_when_no_common_basis(self):
        from src.agent.nodes import _latest_comparable_pair

        rows = [
            {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2024-09-30", "value": 1},
            {"ticker": "AMZN", "metric": "revenue", "period_type": "annual", "period_end": "2025-12-31", "value": 2},
        ]
        pair = _latest_comparable_pair(rows, "AAPL", "AMZN", "revenue", "annual")
        assert pair is None

    def test_comparison_answer_requires_strict_basis(self):
        from src.agent.nodes import _compose_answer_payload

        state = {
            "task_type": "company_comparison",
            "companies": ["AAPL", "AMZN"],
            "comparison_target": "AMZN",
            "requested_metrics": ["revenue"],
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2024-09-30", "value": 100},
                        ],
                        "price_data": [],
                    },
                },
                {
                    "tool": "query_financial_data",
                    "ticker": "AMZN",
                    "data": {
                        "ticker": "AMZN",
                        "financial_facts": [
                            {"ticker": "AMZN", "metric": "revenue", "period_type": "annual", "period_end": "2025-12-31", "value": 200},
                        ],
                        "price_data": [],
                    },
                },
            ],
            "retrieved_docs": [],
        }
        payload = _compose_answer_payload(state, cleaned_answer="", lang="zh", document_citations=[])
        assert "无法形成严格同口径比较" in payload["answer"]

    def test_numeric_fact_answer_has_structured_sources(self):
        from src.agent.nodes import _compose_answer_payload

        state = {
            "task_type": "fact_qa",
            "companies": ["AAPL"],
            "comparison_target": None,
            "requested_metrics": ["revenue"],
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {"ticker": "AAPL", "metric": "revenue", "period_type": "quarterly", "period_end": "2025-12-31", "value": 143756000000},
                        ],
                        "price_data": [],
                    },
                }
            ],
            "retrieved_docs": [],
        }
        payload = _compose_answer_payload(
            state,
            cleaned_answer="",
            lang="zh",
            document_citations=[{"ticker": "AAPL", "form_type": "10-Q", "fiscal_period": "2025-12-31", "section": "ITEM_2"}],
        )
        assert payload["structured_sources"]
        assert all(c["source_kind"] == "structured" for c in payload["structured_sources"])
        assert "数据来源（结构化）" in payload["answer"]

    def test_execute_tools_applies_risk_section_allowlist(self, monkeypatch):
        from src.agent import nodes

        calls = []

        class DummySearch:
            @staticmethod
            def invoke(payload):
                calls.append(payload)
                return []

        monkeypatch.setattr(nodes, "search_filings", DummySearch())
        out = nodes.execute_tools(
            {
                "user_query": "apple有什么风险",
                "task_type": "report_summary",
                "companies": ["AAPL"],
                "comparison_target": None,
                "selected_tools": ["search_filings"],
                "time_range": None,
                "requested_metrics": [],
                "tool_results": [],
                "retrieved_docs": [],
            }
        )
        assert out["tool_results"]
        assert calls
        assert calls[0].get("section_allowlist") == ["ITEM_1A", "ITEM_7", "ITEM_2"]
        assert calls[0].get("strict_sections") is False

    def test_auto_compute_skips_non_comparable_comparison_rows(self, monkeypatch):
        from src.agent import nodes

        compute_calls = []

        class DummyCompute:
            @staticmethod
            def invoke(payload):
                compute_calls.append(payload)
                return {"ok": True}

        monkeypatch.setattr(nodes, "compute_metrics", DummyCompute())
        tool_results = [
            {
                "tool": "query_financial_data",
                "ticker": "AAPL",
                "data": {
                    "ticker": "AAPL",
                    "financial_facts": [
                        {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2024-09-30", "value": 100},
                        {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2023-09-30", "value": 90},
                    ],
                    "price_data": [],
                },
            },
            {
                "tool": "query_financial_data",
                "ticker": "AMZN",
                "data": {
                    "ticker": "AMZN",
                    "financial_facts": [
                        {"ticker": "AMZN", "metric": "revenue", "period_type": "annual", "period_end": "2025-12-31", "value": 200},
                        {"ticker": "AMZN", "metric": "revenue", "period_type": "annual", "period_end": "2022-12-31", "value": 150},
                    ],
                    "price_data": [],
                },
            },
        ]
        nodes._auto_compute(  # pylint: disable=protected-access
            state={"companies": ["AAPL", "AMZN"], "comparison_target": "AMZN"},
            tool_results=tool_results,
            task_type="company_comparison",
        )
        assert compute_calls == []


class TestEvidenceLayeringAndClaimValidation:

    def test_build_evidence_bundle_splits_numeric_and_text(self):
        from src.agent.nodes import _build_evidence_bundle

        state = {
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {
                                "ticker": "AAPL",
                                "metric": "revenue",
                                "period_type": "quarterly",
                                "period_end": "2025-12-31",
                                "value": 143756000000.0,
                                "unit": "USD",
                            }
                        ],
                        "price_data": [],
                    },
                },
                {
                    "tool": "compute_metrics",
                    "ticker": "AAPL",
                    "metric": "revenue",
                    "computation": "qoq",
                    "data": {
                        "results": [
                            {"period": "2025-12-31", "qoq_change": 0.12},
                        ]
                    },
                },
            ],
            "retrieved_docs": [
                {
                    "ticker": "AAPL",
                    "form_type": "10-Q",
                    "fiscal_period": "2025-12-31",
                    "section": "ITEM_2",
                    "chunk_order": 1,
                    "text": "Management discussed demand and margins.",
                }
            ],
        }
        out = _build_evidence_bundle(state)
        assert out["numeric_evidence"]
        assert out["text_evidence"]
        assert all(item["evidence_id"].startswith("N") for item in out["numeric_evidence"])
        assert all(item["evidence_id"].startswith("T") for item in out["text_evidence"])

    def test_validate_claims_rejects_numeric_claim_with_text_evidence(self):
        from src.agent.nodes import _validate_claims

        claims = [{"sentence": "Revenue is 100.", "evidence_ids": ["T1"]}]
        valid, unsupported = _validate_claims(claims, claim_type="numeric", evidence_map={"T1": {"evidence_id": "T1"}})
        assert valid == []
        assert unsupported
        assert unsupported[0]["reason"] == "numeric_claim_with_non_numeric_evidence"

    def test_validate_claims_rejects_estimation_word(self):
        from src.agent.nodes import _validate_claims

        claims = [{"sentence": "Revenue is estimated at 100.", "evidence_ids": ["N1"]}]
        valid, unsupported = _validate_claims(claims, claim_type="numeric", evidence_map={"N1": {"evidence_id": "N1"}})
        assert valid == []
        assert unsupported
        assert unsupported[0]["reason"] == "estimation_word_detected"

    def test_validate_numeric_claims_strict_rejects_value_mismatch(self):
        from src.agent.nodes import _validate_numeric_claims_strict

        claims = [
            {
                "sentence": "AAPL revenue (2025-12-31, quarterly) = 101 USD",
                "evidence_ids": ["N1"],
                "period_end": "2025-12-31",
                "value": 101.0,
                "unit": "USD",
            }
        ]
        numeric_map = {
            "N1": {
                "evidence_id": "N1",
                "ticker": "AAPL",
                "metric": "revenue",
                "period_type": "quarterly",
                "period_end": "2025-12-31",
                "value": 100.0,
                "unit": "USD",
            }
        }
        valid, unsupported = _validate_numeric_claims_strict(claims, numeric_map=numeric_map)
        assert valid == []
        assert unsupported
        assert unsupported[0]["reason"] == "numeric_claim_value_mismatch"

    def test_render_claim_answer_requires_numeric_claim_for_fact(self):
        from src.agent.nodes import _render_claim_answer

        answer = _render_claim_answer(
            lang="zh",
            task_type="fact_qa",
            numeric_claims=[],
            text_claims=[{"sentence": "管理层强调了风险。", "evidence_ids": ["T1"]}],
        )
        assert "证据不足" in answer

    def test_generate_answer_enforces_numeric_citations_for_numeric_claims(self, monkeypatch):
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"numeric_claims":[{"sentence":"AAPL revenue is 100 in 2025-12-31.",'
                        '"evidence_ids":["N1"]}],"text_claims":[]}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())
        state = {
            "user_query": "What is Apple revenue?",
            "task_type": "fact_qa",
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {
                                "ticker": "AAPL",
                                "metric": "revenue",
                                "period_type": "quarterly",
                                "period_end": "2025-12-31",
                                "value": 100.0,
                                "unit": "USD",
                            }
                        ],
                        "price_data": [],
                    },
                }
            ],
            "retrieved_docs": [
                {
                    "ticker": "AAPL",
                    "form_type": "10-Q",
                    "fiscal_period": "2025-12-31",
                    "section": "ITEM_2",
                    "chunk_order": 1,
                    "text": "Narrative text that should not be used for numeric claims.",
                }
            ],
        }

        out = nodes.generate_answer(state)
        assert "Insufficient evidence" not in out["final_answer"]
        assert out["numeric_citations"]
        assert all(c["source_kind"] == "structured" for c in out["numeric_citations"])
        assert all(c["source_kind"] != "document" for c in out["numeric_citations"])

    def test_execute_tools_skips_fact_text_when_structured_is_sufficient(self, monkeypatch):
        from src.agent import nodes

        search_calls = []

        class DummySearch:
            @staticmethod
            def invoke(payload):
                search_calls.append(payload)
                return []

        class DummyQuery:
            @staticmethod
            def invoke(_payload):
                return {
                    "ticker": "AAPL",
                    "financial_facts": [
                        {
                            "ticker": "AAPL",
                            "metric": "revenue",
                            "period_type": "quarterly",
                            "period_end": "2025-12-31",
                            "value": 100.0,
                            "unit": "USD",
                            "fiscal_year": 2025,
                            "fiscal_quarter": 1,
                        }
                    ],
                    "price_data": [],
                    "period_context": {"fiscal_year_end_month": 9},
                }

        monkeypatch.setattr(nodes, "search_filings", DummySearch())
        monkeypatch.setattr(nodes, "query_financial_data", DummyQuery())
        out = nodes.execute_tools(
            {
                "user_query": "Apple 最近一个季度营收是多少？",
                "task_type": "fact_qa",
                "data_route": "hybrid",
                "companies": ["AAPL"],
                "comparison_target": None,
                "selected_tools": ["query_financial_data", "search_filings"],
                "time_range": None,
                "requested_metrics": ["revenue"],
                "tool_results": [],
                "retrieved_docs": [],
                "period_query": {
                    "period_type": "latest",
                    "year": None,
                    "quarter": None,
                    "trailing_n": None,
                    "year_basis": "fiscal",
                    "comparison_basis": "same_period",
                },
                "resolved_period_context": {
                    "target_period_type": "quarterly",
                    "comparison_basis": "same_period",
                    "needs_clarification": False,
                    "strict_period_match": True,
                },
            }
        )
        assert search_calls == []
        assert out.get("retrieval_debug", {}).get("search_skipped")
        assert any(
            "structured_evidence_sufficient_for_fact_qa" in str(x.get("reason", ""))
            for x in out.get("retrieval_debug", {}).get("search_skipped", [])
        )

    def test_text_citation_caps_for_trend_and_summary(self):
        from src.agent.nodes import _apply_text_citation_policy

        base = []
        for i in range(8):
            base.append(
                {
                    "ticker": "AAPL",
                    "filing_id": "f1",
                    "section": "ITEM_2",
                    "fiscal_period": "2025-12-31",
                    "chunk_order": i,
                    "quality": "high",
                    "score": 0.9 - i * 0.01,
                    "evidence_id": f"T{i + 1}",
                }
            )

        trend_state = {"task_type": "trend_analysis", "retrieval_policy": {}}
        trend_out, _, _ = _apply_text_citation_policy(trend_state, base)
        assert len(trend_out) <= 3

        summary_state = {"task_type": "report_summary", "retrieval_policy": {}}
        summary_out, _, _ = _apply_text_citation_policy(summary_state, base)
        assert len(summary_out) <= 5


class TestPhase4OutputContract:

    def test_generate_answer_builds_fact_output(self, monkeypatch):
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"numeric_claims":[{"sentence":"AAPL revenue is 100 in 2025-12-31.",'
                        '"evidence_ids":["N1"]}],"text_claims":[]}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())
        state = {
            "trace_id": "trace-fact-1",
            "user_query": "AAPL 最近一个季度营收是多少？",
            "task_type": "fact_qa",
            "selected_tools": ["query_financial_data"],
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {
                                "ticker": "AAPL",
                                "metric": "revenue",
                                "period_type": "quarterly",
                                "period_end": "2025-12-31",
                                "value": 100.0,
                                "unit": "USD",
                            }
                        ],
                        "price_data": [],
                    },
                }
            ],
            "retrieved_docs": [],
            "period_query": {"period_type": "latest", "year_basis": "fiscal"},
            "resolved_period_context": {},
        }
        out = nodes.generate_answer(state)
        payload = out["output"]
        assert payload["protocol_version"] == "phase4.v1"
        assert payload["task_type"] == "fact_qa"
        assert payload["view"]["kind"] == "analytical_brief"
        assert payload["view"]["headline_metric"]
        assert len(payload["summary"]) <= 180
        assert len(payload["numeric_evidence"]) <= 2
        assert len(payload["text_evidence"]) <= 1

    def test_generate_answer_builds_comparison_view(self, monkeypatch):
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"numeric_claims":[{"sentence":"AAPL revenue 100 is higher than MSFT 80 in 2025-12-31.",'
                        '"evidence_ids":["N1","N2"]}],"text_claims":[]}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())
        state = {
            "trace_id": "trace-comp-1",
            "user_query": "比较 AAPL 和 MSFT 营收",
            "task_type": "company_comparison",
            "companies": ["AAPL"],
            "comparison_target": "MSFT",
            "comparison_basis_label": "同季度",
            "selected_tools": ["query_financial_data"],
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "financial_facts": [
                            {
                                "ticker": "AAPL",
                                "metric": "revenue",
                                "period_type": "quarterly",
                                "period_end": "2025-12-31",
                                "value": 100.0,
                                "unit": "USD",
                            }
                        ],
                        "price_data": [],
                    },
                },
                {
                    "tool": "query_financial_data",
                    "ticker": "MSFT",
                    "data": {
                        "ticker": "MSFT",
                        "financial_facts": [
                            {
                                "ticker": "MSFT",
                                "metric": "revenue",
                                "period_type": "quarterly",
                                "period_end": "2025-12-31",
                                "value": 80.0,
                                "unit": "USD",
                            }
                        ],
                        "price_data": [],
                    },
                },
            ],
            "retrieved_docs": [],
            "period_query": {"period_type": "latest", "comparison_basis": "same_period", "year_basis": "fiscal"},
            "resolved_period_context": {"same_period_match": True, "common_periods": ["2025-12-31"]},
        }
        out = nodes.generate_answer(state)
        payload = out["output"]
        assert payload["task_type"] == "company_comparison"
        assert payload["comparison_basis"] == "同季度"
        assert payload["view"]["kind"] == "comparison_brief"
        assert payload["view"]["comparison_basis_line"]
        assert payload["view"]["comparison_table"]["columns"]
        assert payload["view"]["comparison_table"]["rows"]

    def test_generate_answer_report_summary_has_limitations(self, monkeypatch):
        from src.agent import nodes

        class DummyLLM:
            @staticmethod
            def invoke(_messages):
                class Response:
                    content = (
                        '{"numeric_claims":[],"text_claims":[{"sentence":"Management highlighted demand risks.",'
                        '"evidence_ids":["T1"]}]}'
                    )

                return Response()

        monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())
        state = {
            "trace_id": "trace-summary-1",
            "user_query": "总结公司风险",
            "task_type": "report_summary",
            "selected_tools": ["search_filings"],
            "tool_results": [],
            "retrieved_docs": [
                {
                    "ticker": "AAPL",
                    "filing_id": "f1",
                    "form_type": "10-K",
                    "fiscal_period": "2025-09-30",
                    "section": "ITEM_1A",
                    "chunk_order": 1,
                    "text": "Demand volatility remains a risk factor.",
                }
            ],
            "period_query": {"period_type": "latest", "year_basis": "fiscal"},
            "resolved_period_context": {},
        }
        out = nodes.generate_answer(state)
        payload = out["output"]
        assert payload["view"]["kind"] == "analytical_brief"
        assert payload["view"]["risk_and_limits"]
        assert payload["limitations"]

    def test_build_phase4_output_enforces_trend_caps(self):
        from src.agent.nodes import _build_phase4_output

        numeric_evidence = []
        numeric_citations = []
        for i in range(10):
            evidence_id = f"N{i + 1}"
            row = {
                "evidence_id": evidence_id,
                "source_tool": "query_financial_data",
                "ticker": "AAPL",
                "metric": "revenue",
                "period_type": "quarterly",
                "period_end": f"2025-0{i % 4 + 1}-30",
                "value": float(i),
                "unit": "USD",
                "provenance": f"p{i}",
            }
            numeric_evidence.append(row)
            numeric_citations.append({"evidence_id": evidence_id})

        text_evidence = []
        text_citations = []
        for i in range(8):
            evidence_id = f"T{i + 1}"
            row = {
                "evidence_id": evidence_id,
                "ticker": "AAPL",
                "filing_id": "f1",
                "form_type": "10-Q",
                "fiscal_period": f"2025-0{i % 4 + 1}-30",
                "section": f"ITEM_{i + 1}",
                "chunk_order": i + 1,
                "text_snippet": "sample",
            }
            text_evidence.append(row)
            text_citations.append({"evidence_id": evidence_id})

        out = _build_phase4_output(
            state={
                "task_type": "trend_analysis",
                "trace_id": "trace-trend-1",
                "selected_tools": ["query_financial_data", "search_filings"],
                "companies": ["AAPL"],
            },
            lang="zh",
            task_type="trend_analysis",
            comparison_basis_label="",
            period_query={"period_type": "trailing", "trailing_n": 4},
            numeric_claims=[{"sentence": "Revenue moved from 1 to 2.", "evidence_ids": ["N1", "N2"]}],
            text_claims=[{"sentence": "Management discussed drivers.", "evidence_ids": ["T1"]}],
            numeric_evidence=numeric_evidence,
            text_evidence=text_evidence,
            numeric_citations=numeric_citations,
            text_citations=text_citations,
            unsupported_claims=[],
            period_error=None,
            comparison_text_unbalanced=False,
        )
        assert out["view"]["kind"] == "analytical_brief"
        assert len(out["numeric_evidence"]) <= 6
        assert len(out["text_evidence"]) <= 3

    def test_build_phase4_output_fact_headline_has_placeholder_when_missing_numeric(self):
        from src.agent.nodes import _build_phase4_output

        out = _build_phase4_output(
            state={
                "task_type": "fact_qa",
                "trace_id": "trace-fact-missing-headline",
                "selected_tools": ["query_financial_data"],
                "companies": ["AAPL"],
            },
            lang="zh",
            task_type="fact_qa",
            comparison_basis_label="",
            period_query={"period_type": "latest", "year_basis": "fiscal"},
            numeric_claims=[],
            text_claims=[],
            numeric_evidence=[],
            text_evidence=[],
            numeric_citations=[],
            text_citations=[],
            unsupported_claims=[],
            period_error="no_numeric_citation_for_time_check",
            comparison_text_unbalanced=False,
        )
        assert out["view"]["kind"] == "analytical_brief"
        assert isinstance(out["view"]["headline_metric"], dict)
        assert out["view"]["headline_metric"]
        assert out["view"]["headline_metric"]["value"] == "N/A"

    def test_build_phase4_output_comparison_table_has_placeholder_row_when_empty(self):
        from src.agent.nodes import _build_phase4_output

        out = _build_phase4_output(
            state={
                "task_type": "company_comparison",
                "trace_id": "trace-comp-placeholder",
                "selected_tools": ["query_financial_data"],
                "companies": ["AAPL", "MSFT"],
                "comparison_target": "MSFT",
            },
            lang="zh",
            task_type="company_comparison",
            comparison_basis_label="同季度",
            period_query={"period_type": "latest", "comparison_basis": "same_period", "year_basis": "fiscal"},
            numeric_claims=[],
            text_claims=[],
            numeric_evidence=[],
            text_evidence=[],
            numeric_citations=[],
            text_citations=[],
            unsupported_claims=[],
            period_error="no_common_period_for_same_period_comparison",
            comparison_text_unbalanced=False,
        )
        table = out["view"]["comparison_table"]
        assert table["columns"]
        assert table["rows"]
        assert table["rows"][0]["ticker"] == "N/A"

    def test_build_text_evidence_fills_unknown_period(self):
        from src.agent.nodes import _build_text_evidence

        docs = [
            {
                "ticker": "NVDA",
                "filing_id": "x1",
                "form_type": "10-K",
                "fiscal_period": "",
                "section": "ITEM_1A",
                "chunk_order": 1,
                "text": "Risk text",
            }
        ]
        out = _build_text_evidence(docs)
        assert out
        assert out[0]["fiscal_period"] == "UNKNOWN_PERIOD"


# ---------------------------------------------------------------------------
# End-to-end (needs running vLLM + data)
# ---------------------------------------------------------------------------

_VLLM_UP = False
try:
    import httpx
    r = httpx.get("http://localhost:8000/v1/models", headers={"Authorization": "Bearer not-needed"}, timeout=2)
    _VLLM_UP = r.status_code == 200
except Exception:
    pass

_DB_EXISTS = Path(ROOT / "data" / "db" / "financial.duckdb").exists()


@pytest.mark.skipif(not _VLLM_UP, reason="vLLM server not running")
@pytest.mark.skipif(not _DB_EXISTS, reason="DuckDB not populated")
@pytest.mark.e2e
@pytest.mark.requires_llm
class TestE2E:

    @staticmethod
    def _run(query: str) -> dict:
        from src.agent.graph import compile_agent
        return compile_agent().invoke({"user_query": query})

    def test_fact_qa(self):
        result = self._run("Apple 最近一个季度的营收是多少？")
        assert result.get("task_type") in ("fact_qa", "trend_analysis")
        assert result.get("final_answer")
        assert len(result.get("final_answer", "")) > 20

    def test_trend_analysis(self):
        result = self._run("分析 Microsoft 最近几个季度的 EPS 变化趋势")
        assert result.get("task_type") == "trend_analysis"
        assert len(result.get("tool_results", [])) > 0
        assert result.get("final_answer")
