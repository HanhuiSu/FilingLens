from __future__ import annotations

from datetime import date

import pytest

from src.agent.query_plan import build_classification_state


def _state(query: str) -> dict:
    return build_classification_state(
        user_query=query,
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="test-intent-policy-paraphrases",
        today=date(2026, 4, 27),
    )


def test_causal_revenue_why_routes_to_revenue_analytical_single_company():
    state = _state("为什么nvidia的营收增长这么多")

    assert state["companies"] == ["NVDA"]
    assert state["task_type"] == "report_summary"
    assert state["answer_mode"] == "analytical"
    assert state["methodology_intent"] == "revenue_quality_analysis"
    assert state["analysis_scope"] == "single_company"
    assert state["canonical_intent"]["intent_family"] == "revenue"
    assert state["canonical_intent"]["analysis_scope"] == "single_company"
    assert state["evidence_policy_id"] == "single_company_revenue_v1"
    assert state["required_dimensions"] == ["revenue_quality"]


def test_aws_profit_importance_routes_to_segment_profitability():
    state = _state("AMZN 的 AWS 对整体利润有多重要？")

    assert state["companies"] == ["AMZN"]
    assert state["task_type"] == "report_summary"
    assert state["answer_mode"] == "analytical"
    assert state["canonical_intent"]["intent_family"] == "profitability"
    assert state["canonical_intent"]["analysis_scope"] == "single_company"
    assert state["canonical_intent"]["segment_focus"] == "AWS"
    assert state["required_dimensions"] == ["profitability_quality", "business_model"]
    assert state["evidence_policy_id"] == "single_company_composite_v1"

    requirements = state["evidence_plan"]["evidence_requirements"]
    aws_text = [req for req in requirements if req.get("retrieval_intent") == "aws_segment_profitability"]
    assert aws_text
    assert aws_text[0]["section_preferences"] == ["ITEM_7", "ITEM_2"]
    assert any("AWS operating income" in query for query in aws_text[0]["broadened_queries"])


RISK_PARAPHRASES = [
    "亚马逊下季度有什么风险？",
    "你分析一下下一个季度亚马逊的风险有什么？",
    "AMZN 接下来最需要担心什么？",
    "下个季度亚马逊风险点在哪里？",
    "亚马逊最大隐患是什么？",
    "亚马逊未来一个季度主要压力是什么？",
    "AMZN 近期有哪些经营风险？",
    "亚马逊接下来可能出什么问题？",
    "亚马逊后面最值得警惕的是什么？",
    "Amazon next quarter key risks?",
]


@pytest.mark.parametrize("query", RISK_PARAPHRASES)
def test_amzn_risk_paraphrases_share_canonical_intent_policy_and_scope(query: str):
    state = _state(query)

    assert state["companies"] == ["AMZN"]
    assert state["answer_mode"] == "risk_focused_analysis"
    assert state["canonical_intent"]["intent_family"] == "risk"
    assert state["canonical_intent"]["analysis_scope"] == "single_company"
    assert state["evidence_policy"]["policy_id"] == "single_company_risk_v1"
    assert state["evidence_policy_id"] == "single_company_risk_v1"
    assert state["required_dimensions"] == ["moat_and_competitive_risk"]

    requirements = state["evidence_plan"]["evidence_requirements"]
    core = [
        req
        for req in requirements
        if req.get("required", True) and req.get("requirement_scope") == "core"
    ]
    assert [req["requirement_id"] for req in core] == ["REQ-TEXT-AMZN-RISK_FACTORS"]
    assert core[0]["dimension_id"] == "moat_and_competitive_risk"
    assert core[0]["requirement_type"] == "text"
    assert "ITEM_1A" in core[0].get("primary_sections", []) + core[0].get("fallback_sections", [])

    by_id = {req["requirement_id"]: req for req in requirements}
    assert by_id["REQ-TEXT-AMZN-RISK_BUSINESS_MODEL"]["requirement_scope"] == "optional_context"
    assert by_id["REQ-TEXT-AMZN-RISK_BUSINESS_MODEL"]["required"] is False
    assert by_id["REQ-TEXT-AMZN-RISK_MDA"]["requirement_scope"] == "optional_context"
    assert by_id["REQ-TEXT-AMZN-RISK_MDA"]["required"] is False

    numeric_context = [
        req
        for req in requirements
        if req.get("requirement_type") in {"numeric", "calculation"}
        and req.get("metric") in {"revenue", "net_income", "net_margin"}
    ]
    assert numeric_context
    assert {req.get("requirement_scope") for req in numeric_context} <= {"optional_context", "diagnostic"}
    assert all(req.get("required") is False for req in numeric_context)


@pytest.mark.parametrize("query", ["AMZN overview", "amazon overview", "分析下 Amazon", "Amazon 公司概览"])
def test_amzn_overview_paraphrases_route_to_analytical_single_company_overview(query: str):
    state = _state(query)

    assert state["companies"] == ["AMZN"]
    assert state["task_type"] == "report_summary"
    assert state["answer_mode"] == "analytical"
    assert state["canonical_intent"]["intent_family"] == "overview"
    assert state["canonical_intent"]["analysis_scope"] == "single_company"
    assert state["evidence_policy_id"] == "single_company_overview_v1"
    assert state["required_dimensions"] == [
        "revenue_quality",
        "profitability_quality",
        "moat_and_competitive_risk",
    ]
    assert state["optional_dimensions"] == [
        "business_model",
        "cash_flow_quality",
        "balance_sheet_and_capital_intensity",
        "valuation_and_risk_boundary",
    ]
    core = [
        req
        for req in state["evidence_plan"]["evidence_requirements"]
        if req.get("required", True) and req.get("requirement_scope") == "core"
    ]
    assert len(core) >= 18


@pytest.mark.parametrize(
    ("query", "expected_family", "expected_mode"),
    [
        ("亚马逊营收是多少？", "overview", "direct_fact"),
        ("AMZN 最新净利润是多少？", "overview", "direct_fact"),
        ("亚马逊股价明天会涨吗？", "refusal", "refusal_or_redirect"),
        ("亚马逊现在便宜吗？", "valuation", "analytical"),
        ("分析亚马逊现金流质量", "cash_flow", "analytical"),
        ("分析亚马逊估值边界", "valuation", "analytical"),
    ],
)
def test_risk_terms_do_not_overroute_non_risk_questions(query: str, expected_family: str, expected_mode: str):
    state = _state(query)

    assert state["answer_mode"] == expected_mode
    assert state["canonical_intent"]["intent_family"] == expected_family
    assert state["answer_mode"] != "risk_focused_analysis"

    if expected_family == "valuation":
        assert state["methodology_intent"] == "valuation_boundary_analysis"
        assert state["required_dimensions"] == ["valuation_and_risk_boundary"]
    if expected_family == "cash_flow":
        assert state["methodology_intent"] == "cash_flow_quality_analysis"
        assert state["required_dimensions"] == ["cash_flow_quality"]
