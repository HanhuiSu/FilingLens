"""Unit tests for tool execution module boundaries."""

from __future__ import annotations

from typing import Any

from src.agent.evidence_planner import build_evidence_plan
from src.agent.tool_executor import execute_agent_tools, infer_period_type


class FakeTool:
    def __init__(self, response: Any):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> Any:
        self.calls.append(payload)
        return self.response


def test_execute_agent_tools_no_ticker_short_circuits():
    out = execute_agent_tools(
        {
            "user_query": "What was revenue?",
            "companies": [],
            "selected_tools": ["query_financial_data"],
            "task_type": "fact_qa",
        }
    )

    assert out["tool_results"][0]["tool"] == "_none"
    assert out["retrieved_docs"] == []


def test_execute_agent_tools_needs_tools_false_skips_tool_calls(monkeypatch):
    financial_tool = FakeTool({"financial_facts": []})
    monkeypatch.setattr("src.agent.tool_executor.query_financial_data", financial_tool)

    out = execute_agent_tools(
        {
            "user_query": "你是谁？",
            "companies": ["AAPL"],
            "selected_tools": ["query_financial_data"],
            "task_type": "fact_qa",
            "answer_mode": "meta",
            "needs_tools": False,
        }
    )

    assert financial_tool.calls == []
    assert out["tool_results"] == []
    assert out["retrieved_docs"] == []
    assert out["retrieval_debug"]["search_skipped"][0]["reason"] == "needs_tools_false"


def test_unsupported_prediction_needs_tools_false_skips_financial_and_search(monkeypatch):
    financial_tool = FakeTool({"financial_facts": []})
    search_tool = FakeTool([])
    monkeypatch.setattr("src.agent.tool_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.tool_executor.search_filings", search_tool)

    out = execute_agent_tools(
        {
            "user_query": "预测明天苹果股价",
            "companies": ["AAPL"],
            "selected_tools": ["query_financial_data", "search_filings"],
            "task_type": "fact_qa",
            "answer_mode": "refusal_or_redirect",
            "safety_intent": "unsupported_or_out_of_scope",
            "needs_tools": False,
        }
    )

    assert financial_tool.calls == []
    assert search_tool.calls == []
    assert out["selected_tools"] == []


def test_direct_fact_ignores_plan_extra_search_tool(monkeypatch):
    financial_tool = FakeTool({"financial_facts": [{"ticker": "AAPL", "metric": "revenue", "value": 100}]})
    search_tool = FakeTool([])
    monkeypatch.setattr("src.agent.tool_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.tool_executor.search_filings", search_tool)

    out = execute_agent_tools(
        {
            "user_query": "AAPL revenue",
            "companies": ["AAPL"],
            "selected_tools": ["query_financial_data"],
            "validated_tools": ["query_financial_data", "search_filings"],
            "analysis_plan": {"validated_tools": ["query_financial_data", "search_filings"]},
            "task_type": "fact_qa",
            "answer_mode": "direct_fact",
            "needs_tools": True,
        }
    )

    assert financial_tool.calls
    assert search_tool.calls == []
    assert out["selected_tools"] == ["query_financial_data"]


def test_cautious_outlook_forces_structured_compute_and_filing_search(monkeypatch):
    financial_tool = FakeTool(
        {
            "financial_facts": [
                {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2025-12-31", "value": 100},
                {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2024-12-31", "value": 90},
            ],
            "price_data": [],
        }
    )
    compute_tool = FakeTool({"metrics": []})
    search_tool = FakeTool([{"ticker": "AAPL", "section": "ITEM_7", "text": "MD&A outlook"}])
    monkeypatch.setattr("src.agent.tool_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.tool_executor.compute_metrics", compute_tool)
    monkeypatch.setattr("src.agent.tool_executor.search_filings", search_tool)

    out = execute_agent_tools(
        {
            "user_query": "你觉得今年苹果财报会怎么样？",
            "companies": ["AAPL"],
            "selected_tools": ["query_financial_data"],
            "validated_tools": ["query_financial_data", "compute_metrics", "search_filings"],
            "analysis_plan": {
                "validated_tools": ["query_financial_data", "compute_metrics", "search_filings"],
                "section_preferences": ["ITEM_7", "ITEM_1A", "ITEM_1", "ITEM_2"],
                "metric_requirements": ["revenue", "net_income", "operating_margin"],
            },
            "task_type": "fact_qa",
            "answer_mode": "cautious_outlook",
            "needs_tools": True,
            "requested_metrics": ["revenue"],
            "retrieval_policy": {"text_top_k": 1},
        }
    )

    assert set(["query_financial_data", "compute_metrics", "search_filings"]).issubset(out["selected_tools"])
    assert financial_tool.calls
    assert compute_tool.calls
    assert search_tool.calls
    assert search_tool.calls[0]["section_allowlist"] == ["ITEM_7", "ITEM_1A", "ITEM_1", "ITEM_2"]


def test_analytical_plan_guides_section_preferences(monkeypatch):
    search_tool = FakeTool(
        [
            {
                "ticker": "AAPL",
                "section": "ITEM_1A",
                "text": "Risk factors include competition, regulation, product demand uncertainty, and supply constraints that could affect operating results.",
                "supporting_snippet": "Risk factors include competition, regulation, product demand uncertainty, and supply constraints that could affect operating results.",
            }
        ]
    )
    monkeypatch.setattr("src.agent.tool_executor.search_filings", search_tool)

    out = execute_agent_tools(
        {
            "user_query": "Analyze Apple risks",
            "companies": ["AAPL"],
            "selected_tools": ["search_filings"],
            "validated_tools": ["search_filings"],
            "analysis_plan": {
                "validated_tools": ["search_filings"],
                "section_preferences": ["ITEM_1A", "ITEM_7"],
            },
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "needs_tools": True,
            "retrieval_policy": {"text_top_k": 1},
        }
    )

    assert out["retrieved_docs"]
    assert search_tool.calls[0]["section_allowlist"] == ["ITEM_1A"]


def test_investment_advice_like_comparison_forces_non_advisory_evidence_path(monkeypatch):
    financial_tool = FakeTool(
        {
            "financial_facts": [
                {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2025-12-31", "value": 100}
            ],
            "price_data": [],
        }
    )
    compute_tool = FakeTool({"metrics": []})
    search_tool = FakeTool([{"ticker": "AAPL", "section": "ITEM_7", "text": "MD&A"}])
    monkeypatch.setattr("src.agent.tool_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.tool_executor.compute_metrics", compute_tool)
    monkeypatch.setattr("src.agent.tool_executor.search_filings", search_tool)

    out = execute_agent_tools(
        {
            "user_query": "AAPL 和 AMZN 推荐哪个？",
            "companies": ["AAPL", "AMZN"],
            "comparison_target": "AMZN",
            "selected_tools": ["query_financial_data"],
            "validated_tools": ["query_financial_data", "compute_metrics", "search_filings"],
            "analysis_plan": {
                "validated_tools": ["query_financial_data", "compute_metrics", "search_filings"],
                "section_preferences": ["ITEM_7", "ITEM_1A", "ITEM_1", "ITEM_2"],
            },
            "task_type": "company_comparison",
            "answer_mode": "comparison_brief",
            "safety_intent": "investment_advice_like",
            "needs_tools": True,
            "requested_metrics": ["revenue"],
            "retrieval_policy": {"text_top_k": 4, "comparison_text_cap_per_company": 2},
        }
    )

    assert set(["query_financial_data", "compute_metrics", "search_filings"]).issubset(out["selected_tools"])
    assert len(financial_tool.calls) == 4
    assert search_tool.calls[0]["section_allowlist"] == ["ITEM_7", "ITEM_1A"]


def test_execute_agent_tools_runs_event_window_first(monkeypatch):
    event_tool = FakeTool(
        {
            "ticker": "AAPL",
            "events": [
                {
                    "ticker": "AAPL",
                    "event_type": "10Q",
                    "event_date": "2025-01-31",
                    "fiscal_period": "2024-12-31",
                    "returns": {"return_1d": 0.01},
                }
            ],
        }
    )
    monkeypatch.setattr("src.agent.tool_executor.query_event_price_window", event_tool)

    out = execute_agent_tools(
        {
            "user_query": "AAPL market reaction after 10-Q filing",
            "companies": ["AAPL"],
            "selected_tools": ["query_event_price_window"],
            "task_type": "trend_analysis",
            "event_intent": "required",
            "market_reaction_requested": True,
            "event_query": {"event_type": "10Q", "latest_n": 1, "window_days": [1]},
        }
    )

    assert event_tool.calls[0]["ticker"] == "AAPL"
    assert event_tool.calls[0]["window_days"] == [1]
    assert out["tool_results"][0]["tool"] == "query_event_price_window"
    assert out["retrieval_debug"]["event_calls"][0]["returned"] == 1


def test_infer_period_type_public_wrapper():
    assert infer_period_type({"user_query": "AAPL quarterly revenue"}) == "quarterly"
    assert infer_period_type({"user_query": "AAPL annual revenue"}) == "annual"


def test_requirement_driven_execution_binds_numeric_and_text_requirement_ids(monkeypatch):
    state = {
        "user_query": "Analyze Apple risks and revenue",
        "companies": ["AAPL"],
        "selected_tools": ["query_financial_data", "search_filings"],
        "validated_tools": ["query_financial_data", "search_filings"],
        "analysis_plan": {
            "companies": ["AAPL"],
            "metric_requirements": ["revenue"],
            "section_preferences": ["ITEM_1A", "ITEM_7"],
        },
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "methodology_intent": "risk_focused_analysis",
        "primary_dimension": "moat_and_competitive_risk",
        "query_understanding_summary": {
            "methodology_intent": "risk",
            "legacy_methodology_intent": "risk_focused_analysis",
            "intent_reasons": ["risk_family"],
        },
        "needs_tools": True,
        "requested_metrics": ["revenue"],
        "retrieval_policy": {"text_top_k": 1},
    }
    state["evidence_plan"] = build_evidence_plan(state).model_dump(exclude_none=True)
    financial_tool = FakeTool(
        {
            "financial_facts": [
                {"ticker": "AAPL", "metric": "revenue", "period_type": "annual", "period_end": "2025-12-31", "value": 100}
            ],
            "price_data": [],
        }
    )
    search_tool = FakeTool(
        [
            {"ticker": "AAPL", "filing_id": "f1", "section": "ITEM_1A", "text": "Risk factors include competition."},
            {"ticker": "AAPL", "filing_id": "f1", "section": "ITEM_7", "text": "Management discusses competitive pressure."},
        ]
    )
    monkeypatch.setattr("src.agent.tool_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.tool_executor.search_filings", search_tool)

    out = execute_agent_tools(state)

    fact_row = out["tool_results"][0]["data"]["financial_facts"][0]
    assert fact_row["requirement_id"] == "REQ-NUM-AAPL-LATEST"
    assert out["retrieved_docs"][0]["requirement_id"] == "REQ-TEXT-AAPL-MAJOR_RISKS_RISK"
    assert any(r["status"] == "satisfied" for r in out["evidence_collection_results"])
    assert out["evidence_sufficiency"]["overall_status"] == "sufficient"
    assert out["trace_summary"]["sufficiency_status"] == "sufficient"
    assert out["trace_summary"]["missing_requirements_count"] == 0
    assert out["requirement_status_map"]["REQ-TEXT-AAPL-MAJOR_RISKS_RISK"]["status"] == "satisfied"
    assert any(
        call["requirement_id"] == "REQ-TEXT-AAPL-MAJOR_RISKS_RISK"
        for call in out["retrieval_debug"]["requirement_calls"]
    )


def test_missing_text_requirement_records_retry_and_missing_status(monkeypatch):
    state = {
        "user_query": "Analyze Apple risks",
        "companies": ["AAPL"],
        "selected_tools": ["search_filings"],
        "validated_tools": ["search_filings"],
        "analysis_plan": {"companies": ["AAPL"], "section_preferences": ["ITEM_1A"]},
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "needs_tools": True,
        "retrieval_policy": {"text_top_k": 1},
    }
    state["evidence_plan"] = build_evidence_plan(state).model_dump(exclude_none=True)
    financial_tool = FakeTool({"financial_facts": [], "price_data": []})
    search_tool = FakeTool([])
    monkeypatch.setattr("src.agent.tool_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.tool_executor.search_filings", search_tool)

    out = execute_agent_tools(state)

    text_result = next(r for r in out["evidence_collection_results"] if r["evidence_type"] == "text")
    assert text_result["status"] == "missing"
    assert text_result["retry_count"] == 2
    assert text_result["failure_reason"] == "no_matching_evidence"
    assert out["evidence_sufficiency"]["overall_status"] == "insufficient"
    assert out["requirement_limitations"]
    assert out["trace_summary"]["missing_requirements_count"] >= 1


def test_event_requirement_binds_event_window_result(monkeypatch):
    state = {
        "user_query": "AAPL market reaction after filing",
        "companies": ["AAPL"],
        "selected_tools": ["query_event_price_window"],
        "task_type": "trend_analysis",
        "answer_mode": "direct_fact",
        "event_intent": "required",
        "market_reaction_requested": True,
        "event_query": {"event_type": "10Q", "latest_n": 1, "window_days": [1]},
        "needs_tools": True,
    }
    state["evidence_plan"] = {
        "user_query": state["user_query"],
        "task_type": "trend_analysis",
        "answer_mode": "analytical",
        "safety_intent": "normal",
        "analysis_goal": "event reaction",
        "evidence_requirements": [
            {
                "requirement_id": "REQ-EVENT-AAPL-FILING_REACTION",
                "requirement_type": "event",
                "company": "AAPL",
                "metrics": [],
                "section_preferences": [],
                "purpose": "event window",
                "required": True,
                "min_results": 1,
                "fallback_strategy": [],
            }
        ],
        "sufficiency_criteria": {"required_count": 1},
        "expected_synthesis_style": "analytical_brief",
        "rejected_requirements": [],
    }
    event_tool = FakeTool(
        {
            "ticker": "AAPL",
            "events": [
                {
                    "ticker": "AAPL",
                    "event_type": "10Q",
                    "event_date": "2025-01-31",
                    "fiscal_period": "2024-12-31",
                    "returns": {"return_1d": 0.01},
                }
            ],
        }
    )
    monkeypatch.setattr("src.agent.tool_executor.query_event_price_window", event_tool)

    out = execute_agent_tools(state)

    event = out["tool_results"][0]["data"]["events"][0]
    assert event["requirement_id"] == "REQ-EVENT-AAPL-FILING_REACTION"
    assert out["evidence_collection_results"][0]["status"] == "satisfied"
