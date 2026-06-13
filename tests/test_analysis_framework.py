"""Tests for methodology-v1 analysis framework selection."""

from __future__ import annotations

from dataclasses import fields

from src.agent.analysis_framework import (
    BALANCE_SHEET_AND_CAPITAL_INTENSITY,
    BUSINESS_MODEL,
    CASH_FLOW_QUALITY,
    MOAT_AND_COMPETITIVE_RISK,
    PROFITABILITY_QUALITY,
    REVENUE_QUALITY,
    VALUATION_AND_RISK_BOUNDARY,
    AnalysisDimension,
    get_fundamental_quality_analysis,
    select_analysis_framework,
)
from src.agent.types import AnalysisPlan, QueryPlan


def _active(query: object) -> set[str]:
    return set(select_analysis_framework(query).active_dimension_ids)


def test_fundamental_framework_has_seven_dimensions():
    dimensions = get_fundamental_quality_analysis()

    assert [field.name for field in fields(AnalysisDimension)] == [
        "id",
        "name",
        "description",
        "required_numeric_metrics",
        "optional_numeric_metrics",
        "required_text_sections",
        "optional_text_sections",
        "evidence_purpose",
        "missing_behavior",
        "allowed_claims",
        "forbidden_claims",
    ]
    assert [dimension.id for dimension in dimensions] == [
        BUSINESS_MODEL,
        REVENUE_QUALITY,
        PROFITABILITY_QUALITY,
        CASH_FLOW_QUALITY,
        BALANCE_SHEET_AND_CAPITAL_INTENSITY,
        MOAT_AND_COMPETITIVE_RISK,
        VALUATION_AND_RISK_BOUNDARY,
    ]


def test_dimension_guardrails_are_encoded():
    by_id = {dimension.id: dimension for dimension in get_fundamental_quality_analysis()}

    cash_flow = by_id[CASH_FLOW_QUALITY]
    assert {"operating_cash_flow", "free_cash_flow"} <= set(cash_flow.required_numeric_metrics)
    assert "do not make a cash-flow-quality conclusion" in cash_flow.missing_behavior

    valuation = by_id[VALUATION_AND_RISK_BOUNDARY]
    valuation_text = " ".join([valuation.missing_behavior, *valuation.forbidden_claims]).lower()
    assert "cheap" in valuation_text
    assert "expensive" in valuation_text
    assert "worth buying" in valuation_text

    moat = by_id[MOAT_AND_COMPETITIVE_RISK]
    assert "ITEM_1A" in moat.required_text_sections
    assert "do not make specific competitive-risk claims" in moat.missing_behavior

    profitability = by_id[PROFITABILITY_QUALITY]
    assert {"revenue", "net_income"} <= set(profitability.required_numeric_metrics)
    assert "net_margin" in profitability.optional_numeric_metrics

    business = by_id[BUSINESS_MODEL]
    assert "ITEM_1" in business.required_text_sections

    revenue = by_id[REVENUE_QUALITY]
    assert any("single revenue period" in claim for claim in revenue.forbidden_claims)


def test_selector_advice_like_activates_comparison_dimensions():
    active = _active({"user_query": "apple 和 amazon 更推荐哪个"})

    assert {
        REVENUE_QUALITY,
        PROFITABILITY_QUALITY,
        MOAT_AND_COMPETITIVE_RISK,
        VALUATION_AND_RISK_BOUNDARY,
    } <= active
    assert CASH_FLOW_QUALITY not in active
    assert BALANCE_SHEET_AND_CAPITAL_INTENSITY not in active


def test_selector_long_term_adds_cash_flow_and_balance_sheet():
    active = _active({"user_query": "AAPL 和 AMZN 长期更看好哪个"})

    assert CASH_FLOW_QUALITY in active
    assert BALANCE_SHEET_AND_CAPITAL_INTENSITY in active
    assert VALUATION_AND_RISK_BOUNDARY in active


def test_selector_risk_query_activates_risk_and_health_dimensions():
    active = _active({"user_query": "苹果现在最大问题和风险压力是什么？"})

    assert {
        BUSINESS_MODEL,
        MOAT_AND_COMPETITIVE_RISK,
        CASH_FLOW_QUALITY,
        BALANCE_SHEET_AND_CAPITAL_INTENSITY,
    } <= active


def test_selector_recent_performance_activates_revenue_and_profitability():
    active = _active({"user_query": "AAPL 最近财报营收和利润趋势怎么样？"})

    assert REVENUE_QUALITY in active
    assert PROFITABILITY_QUALITY in active


def test_selector_cash_flow_quality_adds_cash_flow_dimension():
    active = _active({"user_query": "AAPL 现金流质量和财务健康怎么样？"})

    assert CASH_FLOW_QUALITY in active


def test_selector_generic_analysis_activates_full_single_company_methodology_dimensions():
    selected = select_analysis_framework({"user_query": "帮我分析一下苹果这家公司怎么样"})
    active = set(selected.active_dimension_ids)

    assert {
        BUSINESS_MODEL,
        REVENUE_QUALITY,
        PROFITABILITY_QUALITY,
        CASH_FLOW_QUALITY,
        BALANCE_SHEET_AND_CAPITAL_INTENSITY,
        MOAT_AND_COMPETITIVE_RISK,
        VALUATION_AND_RISK_BOUNDARY,
    } <= active
    assert len(active) == 7
    assert selected.inactive_dimension_ids == []


def test_single_company_scope_activates_default_methodology_dimensions():
    selected = select_analysis_framework(
        {
            "user_query": "分析下 nvidia",
            "analysis_scope": "single_company",
            "analysis_plan": {"companies": ["NVDA"]},
        }
    )

    assert selected.framework_id == "fundamental_quality_analysis"
    assert selected.active_dimension_ids == [
        BUSINESS_MODEL,
        REVENUE_QUALITY,
        PROFITABILITY_QUALITY,
        CASH_FLOW_QUALITY,
        BALANCE_SHEET_AND_CAPITAL_INTENSITY,
        MOAT_AND_COMPETITIVE_RISK,
        VALUATION_AND_RISK_BOUNDARY,
    ]
    assert CASH_FLOW_QUALITY not in selected.inactive_dimension_ids
    assert BALANCE_SHEET_AND_CAPITAL_INTENSITY not in selected.inactive_dimension_ids
    assert selected.selection_reasons[0]["rule"] == "single_company_methodology_analysis"


def test_selector_accepts_query_plan_and_analysis_plan_fallback():
    plan = QueryPlan(
        analysis_plan=AnalysisPlan(user_intent="Analyze AAPL revenue and profit trend"),
    )

    active = _active(plan)

    assert REVENUE_QUALITY in active
    assert PROFITABILITY_QUALITY in active
