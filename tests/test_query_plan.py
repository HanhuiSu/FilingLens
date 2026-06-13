"""Unit tests for query planning module boundaries."""

from __future__ import annotations

from datetime import date

from src.agent.query_plan import (
    build_classification_state,
    build_query_plan,
    detect_answer_mode,
    detect_event_intent,
    extract_tickers_fallback,
    select_tools,
)


def test_build_query_plan_returns_typed_model_with_dumpable_payload():
    plan = build_query_plan(
        {
            "user_query": "AAPL 2025 Q3 revenue",
            "task_type": "fact_qa",
            "period_query": {"period_type": "quarterly", "year": 2025, "quarter": 3},
            "selected_tools": ["query_financial_data"],
        }
    )

    payload = plan.model_dump(exclude_none=True)
    assert payload["period_query"]["period_type"] == "quarterly"
    assert payload["period_query"]["year"] == 2025
    assert payload["period_query"]["quarter"] == 3
    assert payload["selected_tools"] == ["query_financial_data"]
    assert payload["answer_mode"] == "direct_fact"
    assert payload["safety_intent"] == "normal"
    assert payload["needs_tools"] is True


def test_build_query_plan_prefers_query_understanding_summary_for_semantics():
    plan = build_query_plan(
        {
            "user_query": "neutral planning request",
            "task_type": "fact_qa",
            "query_understanding_summary": {
                "companies": [{"ticker": "AAPL", "canonical_name": "Apple Inc."}],
                "analysis_scope": "single_company",
                "methodology_intent": "risk",
                "legacy_methodology_intent": "risk_focused_analysis",
                "safety_intent": "normal",
                "legacy_safety_intent": "normal",
                "time_scope": {
                    "policy": "latest_available",
                    "period_scope": "latest annual + latest quarterly",
                    "is_explicit": False,
                },
            },
        }
    )

    assert plan.analysis_scope == "single_company"
    assert plan.methodology_intent == "risk_focused_analysis"
    assert plan.answer_mode == "risk_focused_analysis"
    assert plan.safety_intent == "normal"


def test_classification_state_parses_period_and_ticker_fallback():
    state_update = build_classification_state(
        user_query="Compare Apple and Microsoft revenue in 2025 Q3",
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="trace-1",
        today=date(2026, 4, 22),
    )

    assert state_update["task_type"] == "company_comparison"
    assert state_update["companies"][:2] == ["AAPL", "MSFT"]
    assert state_update["period_query"]["period_type"] == "quarterly"
    assert state_update["period_query"]["year"] == 2025
    assert state_update["period_query"]["quarter"] == 3
    assert state_update["evidence_plan"]["evidence_requirements"]
    assert state_update["rejected_requirements"] == []


def test_event_intent_and_tool_selection_are_explicit():
    assert detect_event_intent("AAPL stock return after earnings release", "fact_qa") == "required"

    tools = select_tools(
        "fact_qa",
        "hybrid",
        user_query="AAPL stock return after earnings release",
        event_intent="required",
    )
    assert "query_event_price_window" in tools
    assert "query_financial_data" in tools


def test_extract_tickers_fallback_maps_common_company_names():
    assert extract_tickers_fallback("Compare Apple, Microsoft, and Nvidia") == ["AAPL", "MSFT", "NVDA"]


def _classification(query: str, parsed: dict | None = None) -> dict:
    return build_classification_state(
        user_query=query,
        parsed=parsed or {"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="trace-conv-1",
        today=date(2026, 4, 22),
    )


def test_meta_questions_do_not_need_tools():
    for query in ("你是谁？", "你能做什么？"):
        state_update = _classification(query)
        assert state_update["task_type"] == "fact_qa"
        assert state_update["answer_mode"] == "meta"
        assert state_update["safety_intent"] == "normal"
        assert state_update["needs_tools"] is False
        assert state_update["selected_tools"] == []
        assert state_update["evidence_plan"]["evidence_requirements"] == []


def test_underspecified_analysis_asks_single_turn_clarification():
    state_update = _classification("帮我分析一下")

    assert state_update["answer_mode"] == "clarification"
    assert state_update["needs_clarification"] is True
    assert state_update["needs_tools"] is False
    assert state_update["clarification_question"]
    assert state_update["selected_tools"] == []


def test_investment_advice_like_comparison_is_reframed_not_blocked():
    state_update = _classification("AAPL 和 AMZN 推荐哪个？")

    assert state_update["task_type"] == "company_comparison"
    assert state_update["answer_mode"] == "comparison_brief"
    assert state_update["safety_intent"] == "investment_advice_like"
    assert state_update["needs_tools"] is True
    assert "query_financial_data" in state_update["selected_tools"]
    framework = state_update["selected_analysis_framework"]
    assert framework["framework_id"] == "fundamental_quality_analysis"
    assert {
        "revenue_quality",
        "profitability_quality",
        "moat_and_competitive_risk",
        "valuation_and_risk_boundary",
    } <= set(framework["active_dimension_ids"])
    assert state_update["trace_summary"]["analysis_framework_id"] == "fundamental_quality_analysis"
    assert "profitability_quality" in state_update["trace_summary"]["active_analysis_dimensions"]


def test_most_favored_comparison_is_also_investment_advice_like():
    state_update = _classification("aapple和amazon你最看好哪个")

    assert state_update["task_type"] == "company_comparison"
    assert state_update["answer_mode"] == "comparison_brief"
    assert state_update["safety_intent"] == "investment_advice_like"
    assert state_update["companies"][:2] == ["AAPL", "AMZN"]


def test_cautious_outlook_is_not_downgraded_to_direct_fact():
    state_update = _classification("你觉得今年苹果财报会怎么样？")

    assert state_update["task_type"] == "fact_qa"
    assert state_update["answer_mode"] == "cautious_outlook"
    assert state_update["answer_mode"] != "direct_fact"
    assert state_update["needs_tools"] is True
    assert state_update["companies"] == ["AAPL"]
    assert any(item["code"] == "forward_looking_caution" for item in state_update["safety_policy_reasons"])


def test_task_type_and_answer_mode_are_parallel_fields():
    state_update = _classification("AAPL 和 AMZN 推荐哪个？")

    assert "task_type" in state_update
    assert "answer_mode" in state_update
    assert state_update["task_type"] != state_update["answer_mode"]
    assert detect_answer_mode("你是谁？") == "meta"


def test_open_ended_company_problem_uses_analytical_filing_path():
    state_update = _classification("苹果现在最大的问题是什么？")

    assert state_update["task_type"] == "report_summary"
    assert state_update["answer_mode"] == "risk_focused_analysis"
    assert state_update["analysis_scope"] == "single_company"
    assert state_update["primary_dimension"] == "moat_and_competitive_risk"
    assert state_update["required_dimensions"] == ["moat_and_competitive_risk"]
    assert "business_model" in state_update["optional_dimensions"]
    assert state_update["needs_tools"] is True
    assert "search_filings" in state_update["validated_tools"]
    assert "ITEM_1A" in state_update["analysis_plan"]["section_preferences"]
    assert "ITEM_7" in state_update["analysis_plan"]["section_preferences"]


def test_risk_query_routes_to_risk_focused_mode():
    for query, ticker in [
        ("nvidia现在最大的问题是什么", "NVDA"),
        ("NVDA 最大风险是什么", "NVDA"),
        ("苹果目前最值得担心的点是什么", "AAPL"),
    ]:
        state_update = _classification(query)

        assert state_update["task_type"] == "report_summary"
        assert state_update["answer_mode"] == "risk_focused_analysis"
        assert state_update["analysis_scope"] == "single_company"
        assert state_update["primary_dimension"] == "moat_and_competitive_risk"
        assert state_update["required_dimensions"] == ["moat_and_competitive_risk"]
        assert state_update["optional_dimensions"] == ["business_model", "revenue_quality", "profitability_quality"]
        assert state_update["methodology_intent"] == "risk_focused_analysis"
        assert state_update["companies"] == [ticker]
        assert state_update["needs_clarification"] is False


def test_open_ended_single_company_routes_to_methodology_analysis():
    for query, ticker in [
        ("分析下 nvidia", "NVDA"),
        ("帮我看看苹果怎么样", "AAPL"),
        ("总结一下 amazon 的基本面", "AMZN"),
        ("analyze NVDA", "NVDA"),
        ("NVDA fundamentals", "NVDA"),
    ]:
        state_update = _classification(query)

        assert state_update["task_type"] == "report_summary"
        assert state_update["answer_mode"] == "analytical"
        assert state_update["analysis_scope"] == "single_company"
        assert state_update["companies"] == [ticker]
        assert state_update["needs_clarification"] is False
        assert state_update["needs_tools"] is True
        assert state_update["selected_analysis_framework"]["framework_id"] == "fundamental_quality_analysis"


def test_single_company_analysis_uses_latest_available_period():
    state_update = _classification("分析下 nvidia")

    assert state_update["time_policy"] == "latest_available"
    assert state_update["period_scope"] == "latest annual + latest quarterly"
    assert state_update["period_query"]["period_type"] == "latest"
    assert state_update["period_query"]["is_explicit"] is False
    assert state_update["period_query"]["needs_clarification"] is False
    assert state_update["analysis_plan"]["time_policy"] == "latest_available"
    assert state_update["evidence_plan"]["time_policy"] == "latest_available"


def test_single_company_analysis_activates_framework_dimensions():
    state_update = _classification("分析下 nvidia")

    assert state_update["selected_analysis_framework"]["active_dimension_ids"] == [
        "business_model",
        "revenue_quality",
        "profitability_quality",
        "cash_flow_quality",
        "balance_sheet_and_capital_intensity",
        "moat_and_competitive_risk",
        "valuation_and_risk_boundary",
    ]
    assert state_update["trace_summary"]["analysis_scope"] == "single_company"
    assert state_update["trace_summary"]["analysis_framework_id"] == "fundamental_quality_analysis"


def test_single_company_guardrail_keeps_clarification_and_direct_fact_paths():
    for query in ("分析一下", "这家公司怎么样", "帮我看看它"):
        state_update = _classification(query)

        assert state_update["answer_mode"] == "clarification"
        assert state_update["needs_clarification"] is True
        assert state_update["analysis_scope"] == ""
        assert state_update["needs_tools"] is False

    fact_state = _classification("2024Q2 NVDA 营收是多少？")
    assert fact_state["task_type"] == "fact_qa"
    assert fact_state["answer_mode"] == "direct_fact"
    assert fact_state["analysis_scope"] == ""
    assert fact_state["period_query"]["period_type"] == "quarterly"
    assert fact_state["period_query"]["year"] == 2024
    assert fact_state["period_query"]["quarter"] == 2


def test_tomorrow_stock_price_prediction_is_unsupported_without_tools():
    state_update = _classification("预测明天苹果股价")

    assert state_update["answer_mode"] == "refusal_or_redirect"
    assert state_update["safety_intent"] == "unsupported_or_out_of_scope"
    assert state_update["needs_tools"] is False
    assert state_update["selected_tools"] == []
    assert any(item["code"] == "no_realtime_news_access" for item in state_update["safety_limitations"])
    assert state_update["evidence_plan"]["evidence_requirements"] == []


def test_analysis_plan_rejects_unknown_metric_section_tool_and_ticker():
    state_update = _classification(
        "Analyze Apple risks and revenue",
        {
            "task_type": "report_summary",
            "companies": ["Apple"],
            "data_route": "hybrid",
            "analysis_plan": {
                "user_intent": "Analyze Apple",
                "companies": ["Apple", "FAKECO"],
                "metric_requirements": ["revenue", "free_cash_flow", "magic_metric"],
                "section_preferences": ["ITEM_7", "ITEM_99"],
                "proposed_tools": ["search_filings", "web_search"],
            },
        },
    )

    assert state_update["companies"] == ["AAPL"]
    assert state_update["analysis_plan_raw"]["metric_requirements"] == ["revenue", "free_cash_flow", "magic_metric"]
    assert state_update["analysis_plan"]["metric_requirements"] == ["revenue", "free_cash_flow", "net_income"]
    assert "ITEM_7" in state_update["analysis_plan"]["section_preferences"]
    assert "ITEM_99" not in state_update["analysis_plan"]["section_preferences"]
    assert "search_filings" in state_update["analysis_plan"]["validated_tools"]
    rejected = {(item["type"], item["reason"]) for item in state_update["rejected_plan_items"]}
    assert ("metric", "metric_not_allowed") in rejected
    assert ("section", "section_not_allowed") in rejected
    assert ("tool", "tool_not_allowed") in rejected
    assert ("ticker", "unknown_or_unsupported_ticker") in rejected


def test_analysis_plan_uses_existing_period_parser_not_raw_plan_period():
    state_update = _classification(
        "AAPL 2025 Q3 revenue",
        {
            "task_type": "fact_qa",
            "companies": ["AAPL"],
            "data_route": "structured_only",
            "analysis_plan": {
                "time_range": {"start": "1999-01-01", "end": "1999-12-31"},
                "metric_requirements": ["revenue"],
                "proposed_tools": ["query_financial_data"],
            },
        },
    )

    assert state_update["period_query"]["period_type"] == "quarterly"
    assert state_update["period_query"]["year"] == 2025
    assert state_update["period_query"]["quarter"] == 3
    assert state_update["analysis_plan"].get("time_range") == state_update["time_range"]


def test_analysis_plan_with_no_valid_ticker_downgrades_to_clarification():
    state_update = _classification(
        "Tell me revenue",
        {
            "task_type": "fact_qa",
            "companies": [],
            "data_route": "hybrid",
            "analysis_plan": {
                "companies": ["FAKECO"],
                "metric_requirements": ["revenue"],
                "proposed_tools": ["query_financial_data"],
            },
        },
    )

    assert state_update["answer_mode"] == "clarification"
    assert state_update["needs_tools"] is False
    assert state_update["validated_tools"] == []
    assert any(item["type"] == "ticker" for item in state_update["rejected_plan_items"])
