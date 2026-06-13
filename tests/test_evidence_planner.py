"""Tests for evidence-requirement planning and sufficiency."""

from __future__ import annotations

from src.agent.evidence_planner import (
    build_evidence_plan,
    evaluate_evidence_sufficiency,
    validate_evidence_requirement,
)
from src.agent.intent_policy import resolve_evidence_policy


def _state(**overrides):
    base = {
        "user_query": "AAPL revenue",
        "task_type": "fact_qa",
        "answer_mode": "direct_fact",
        "safety_intent": "normal",
        "needs_tools": True,
        "companies": ["AAPL"],
        "requested_metrics": ["revenue"],
        "period_query": {"period_type": "latest"},
        "analysis_plan": {
            "companies": ["AAPL"],
            "metric_requirements": ["revenue"],
            "section_preferences": ["ITEM_7", "ITEM_1A", "ITEM_1", "ITEM_2"],
            "answer_policy": {},
        },
    }
    base.update(overrides)
    return base


def _framework(*dimension_ids: str) -> dict:
    return {
        "framework_id": "fundamental_quality_analysis",
        "active_dimension_ids": list(dimension_ids),
        "dimensions": [
            {"id": dimension_id, "name": dimension_id.replace("_", " ").title(), "evidence_purpose": dimension_id}
            for dimension_id in dimension_ids
        ],
    }


def _overview_policy() -> dict:
    return resolve_evidence_policy(
        {"intent_family": "overview", "analysis_scope": "single_company", "requested_dimensions": []}
    ).model_dump(exclude_none=True)


def test_direct_fact_template_generates_required_numeric_requirement():
    plan = build_evidence_plan(_state()).model_dump(exclude_none=True)

    assert plan["task_type"] == "fact_qa"
    assert plan["evidence_requirements"][0]["requirement_type"] == "numeric"
    assert plan["evidence_requirements"][0]["company"] == "AAPL"
    assert plan["evidence_requirements"][0]["metrics"] == ["revenue"]
    assert plan["sufficiency_criteria"]["required_count"] == 1


def test_profit_decline_query_adds_recent_income_history_requirements():
    plan = build_evidence_plan(
        _state(
            user_query="为什么amazon的利润下降了",
            companies=["AMZN"],
            requested_metrics=["revenue"],
            analysis_plan={"companies": ["AMZN"], "metric_requirements": ["revenue"], "section_preferences": ["ITEM_7"]},
            selected_analysis_framework=_framework("profitability_quality"),
        )
    ).model_dump(exclude_none=True)

    history = [
        req
        for req in plan["evidence_requirements"]
        if req.get("evidence_role") == "profit_decline_premise_check"
    ]
    assert {req["metric"] for req in history} == {"net_income", "operating_income"}
    assert all(req["period_type"] == "ttm" for req in history)
    assert all(req["min_results"] == 2 for req in history)
    assert all(not req["required"] for req in history)


def test_segment_product_networking_query_uses_segment_driver_requirements():
    plan = build_evidence_plan(
        _state(
            user_query="为什么 NVDA 的网络业务增长这么快？",
            companies=["NVDA"],
            answer_mode="analytical",
            analysis_scope="single_company",
            selected_analysis_framework=_framework("revenue_quality", "business_model"),
            analysis_plan={
                "companies": ["NVDA"],
                "requested_dimensions": ["revenue_quality", "business_model"],
                "segment_or_product_scope": "networking",
                "canonical_intent": {"segment_or_product_scope": "networking"},
            },
        )
    ).model_dump(exclude_none=True)

    requirements = plan["evidence_requirements"]
    segment_reqs = [req for req in requirements if req.get("retrieval_intent") == "segment_product_driver"]
    assert len(segment_reqs) >= 5
    assert all(req.get("segment_or_product_scope") == "networking" for req in segment_reqs)
    assert any("InfiniBand" in " ".join(req.get("broadened_queries", [])) for req in segment_reqs)
    assert any("Spectrum-X" in " ".join(req.get("broadened_queries", [])) for req in segment_reqs)
    required_total_revenue = [
        req for req in requirements
        if req.get("requirement_type") == "numeric"
        and req.get("metric") == "revenue"
        and req.get("required")
    ]
    assert required_total_revenue == []


def test_trend_company_comparison_cautious_and_analytical_templates():
    trend = build_evidence_plan(
        _state(task_type="trend_analysis", answer_mode="analytical", requested_metrics=[])
    ).model_dump(exclude_none=True)
    comparison = build_evidence_plan(
        _state(
            user_query="AAPL vs AMZN",
            task_type="company_comparison",
            answer_mode="comparison_brief",
            companies=["AAPL", "AMZN"],
            comparison_target="AMZN",
            requested_metrics=[],
            analysis_plan={"companies": ["AAPL", "AMZN"], "section_preferences": ["ITEM_7", "ITEM_1A", "ITEM_1"]},
        )
    ).model_dump(exclude_none=True)
    outlook = build_evidence_plan(
        _state(user_query="你觉得今年苹果财报会怎么样？", answer_mode="cautious_outlook", requested_metrics=[])
    ).model_dump(exclude_none=True)
    analytical = build_evidence_plan(
        _state(user_query="苹果现在最大的问题是什么？", task_type="report_summary", answer_mode="analytical")
    ).model_dump(exclude_none=True)

    assert {"numeric", "calculation", "text"}.issubset({r["requirement_type"] for r in trend["evidence_requirements"]})
    assert len([r for r in comparison["evidence_requirements"] if r["requirement_type"] == "text"]) == 2
    assert {"REQ-TEXT-AAPL-MDA", "REQ-TEXT-AAPL-RISK"}.issubset(
        {r["requirement_id"] for r in outlook["evidence_requirements"]}
    )
    assert analytical["evidence_requirements"][0]["requirement_type"] == "text"


def test_investment_advice_like_uses_non_advisory_comparison_requirements():
    plan = build_evidence_plan(
        _state(
            user_query="AAPL 和 AMZN 推荐哪个？",
            task_type="company_comparison",
            answer_mode="comparison_brief",
            safety_intent="investment_advice_like",
            companies=["AAPL", "AMZN"],
            comparison_target="AMZN",
            requested_metrics=[],
            analysis_plan={"companies": ["AAPL", "AMZN"], "metric_requirements": ["revenue", "net_income"]},
        )
    ).model_dump(exclude_none=True)

    required = [r for r in plan["evidence_requirements"] if r["required"]]
    numeric_required = [r for r in required if r["requirement_type"] == "numeric"]
    text_required = [r for r in required if r["requirement_type"] == "text"]
    calc_optional = [r for r in plan["evidence_requirements"] if r["requirement_type"] == "calculation"]
    assert plan["safety_intent"] == "investment_advice_like"
    assert {r["company"] for r in numeric_required} == {"AAPL", "AMZN"}
    assert {tuple(r.get("metrics", [])) for r in numeric_required} == {("revenue",), ("net_income",)}
    assert {r["company"] for r in text_required} == {"AAPL", "AMZN"}
    assert all(r["section_preferences"] == ["ITEM_7", "ITEM_1A"] for r in text_required)
    assert all(r["primary_sections"] == ["ITEM_7", "ITEM_1A"] for r in text_required)
    assert all(r["fallback_sections"] == ["ITEM_1", "ITEM_2"] for r in text_required)
    assert {r["retrieval_intent"] for r in text_required} == {"comparison_context"}
    assert {r["requirement_id"] for r in calc_optional} == {
        "REQ-CALC-AAPL-OPERATING_MARGIN",
        "REQ-CALC-AAPL-GROWTH",
        "REQ-CALC-AMZN-OPERATING_MARGIN",
        "REQ-CALC-AMZN-GROWTH",
    }
    assert plan["expected_synthesis_style"] == "balanced_comparison"


def test_methodology_active_dimensions_generate_dimension_requirements():
    plan = build_evidence_plan(
        _state(
            user_query="apple 和 amazon 更推荐哪个",
            task_type="company_comparison",
            answer_mode="comparison_brief",
            safety_intent="investment_advice_like",
            companies=["AAPL", "AMZN"],
            comparison_target="AMZN",
            requested_metrics=[],
            analysis_plan={"companies": ["AAPL", "AMZN"]},
            selected_analysis_framework={
                "framework_id": "fundamental_quality_analysis",
                "active_dimension_ids": [
                    "revenue_quality",
                    "profitability_quality",
                    "moat_and_competitive_risk",
                    "valuation_and_risk_boundary",
                ],
                "dimensions": [
                    {"id": "revenue_quality", "name": "Revenue Quality", "evidence_purpose": "revenue evidence"},
                    {"id": "profitability_quality", "name": "Profitability Quality", "evidence_purpose": "profitability evidence"},
                    {"id": "moat_and_competitive_risk", "name": "Moat And Competitive Risk", "evidence_purpose": "risk evidence"},
                    {"id": "valuation_and_risk_boundary", "name": "Valuation And Risk Boundary", "evidence_purpose": "valuation boundary"},
                ],
            },
        )
    ).model_dump(exclude_none=True)

    reqs = plan["evidence_requirements"]
    assert reqs
    assert len(reqs) >= 30
    assert len(reqs) <= 36
    assert all(req["framework_id"] == "fundamental_quality_analysis" for req in reqs)
    numeric_reqs = [
        (req["company"], req["dimension_id"], req["requirement_type"], req.get("metric"))
        for req in reqs
        if req["requirement_type"] == "numeric"
    ]
    assert {
        ("AAPL", "revenue_quality", "numeric", "revenue"),
        ("AAPL", "profitability_quality", "numeric", "net_income"),
        ("AMZN", "revenue_quality", "numeric", "revenue"),
        ("AMZN", "profitability_quality", "numeric", "net_income"),
    } <= set(numeric_reqs)
    assert {
        (req["company"], req["dimension_id"], req["requirement_type"], req.get("metric"))
        for req in reqs
        if req["requirement_type"] == "calculation" and req["dimension_id"] == "profitability_quality"
    } == {
        ("AAPL", "profitability_quality", "calculation", "net_margin"),
        ("AMZN", "profitability_quality", "calculation", "net_margin"),
    }
    moat_text = [req for req in reqs if req["dimension_id"] == "moat_and_competitive_risk" and req["requirement_type"] == "text"]
    assert {req["company"] for req in moat_text} == {"AAPL", "AMZN"}
    assert all(req["retrieval_intent"] == "comparison_risk_context" for req in moat_text)
    assert all(req["retrieval_profile"] == "risk_summary" for req in moat_text)
    assert all(req["section_preferences"] == ["ITEM_1A", "ITEM_1", "BUSINESS"] for req in moat_text)
    assert all(req["primary_sections"] == ["ITEM_1A", "ITEM_1", "BUSINESS"] for req in moat_text)
    assert all(req["fallback_sections"] == ["ITEM_7", "MD&A"] for req in moat_text)
    assert all(
        {
            f"{req['company']} competition risk factors",
            f"{req['company']} business risks competitive pressure",
            f"{req['company']} risk factors competitive pressure",
        } <= set(req["broadened_queries"])
        for req in moat_text
    )
    valuation_reqs = [req for req in reqs if req["dimension_id"] == "valuation_and_risk_boundary"]
    assert {req.get("company") for req in valuation_reqs} == {"AAPL", "AMZN"}
    assert {"market_cap", "pe_ratio", "ps_ratio", "fcf_yield"} <= {req.get("metric") for req in valuation_reqs}
    assert {"price", "shares_outstanding", "revenue", "net_income", "free_cash_flow"} <= {
        req.get("metric") for req in valuation_reqs
    }
    assert not any(req.get("metric") in {"gross_margin", "operating_margin"} for req in reqs)
    assert any(req.get("metric") == "revenue_growth" for req in reqs)
    assert plan["rejected_requirements"] == []


def test_methodology_business_section_maps_business_to_item_1():
    plan = build_evidence_plan(
        _state(
            user_query="帮我分析一下苹果这家公司怎么样",
            task_type="report_summary",
            answer_mode="analytical",
            companies=["AAPL"],
            analysis_plan={"companies": ["AAPL"]},
            selected_analysis_framework={
                "framework_id": "fundamental_quality_analysis",
                "active_dimension_ids": ["business_model"],
                "dimensions": [{"id": "business_model", "name": "Business Model", "evidence_purpose": "business evidence"}],
            },
        )
    ).model_dump(exclude_none=True)

    req = plan["evidence_requirements"][0]
    assert req["dimension_id"] == "business_model"
    assert req["requirement_type"] == "text"
    assert req["section_preferences"] == ["ITEM_1"]
    assert "business model" in req["retrieval_query"]
    assert "BUSINESS" not in req["section_preferences"]


def test_single_company_evidence_plan_contains_numeric_and_text_requirements():
    policy = _overview_policy()
    plan = build_evidence_plan(
        _state(
            user_query="分析下 nvidia",
            task_type="report_summary",
            answer_mode="analytical",
            analysis_scope="single_company",
            time_policy="latest_available",
            period_scope="latest annual + latest quarterly",
            companies=["NVDA"],
            requested_metrics=["revenue", "net_income"],
            canonical_intent={"intent_family": "overview", "analysis_scope": "single_company", "requested_dimensions": []},
            evidence_policy=policy,
            evidence_policy_id=policy["policy_id"],
            required_dimensions=policy["required_dimensions"],
            optional_dimensions=policy["optional_dimensions"],
            analysis_plan={
                "companies": ["NVDA"],
                "analysis_scope": "single_company",
                "time_policy": "latest_available",
                "period_scope": "latest annual + latest quarterly",
                "metric_requirements": ["revenue", "net_income"],
                "evidence_policy": policy,
                "evidence_policy_id": policy["policy_id"],
                "required_dimensions": policy["required_dimensions"],
                "optional_dimensions": policy["optional_dimensions"],
            },
            selected_analysis_framework={
                "framework_id": "fundamental_quality_analysis",
                "active_dimension_ids": [
                    "business_model",
                    "revenue_quality",
                    "profitability_quality",
                    "cash_flow_quality",
                    "balance_sheet_and_capital_intensity",
                    "moat_and_competitive_risk",
                    "valuation_and_risk_boundary",
                ],
                "dimensions": [
                    {"id": "business_model", "name": "Business Model", "evidence_purpose": "business evidence"},
                    {"id": "revenue_quality", "name": "Revenue Quality", "evidence_purpose": "revenue evidence"},
                    {"id": "profitability_quality", "name": "Profitability Quality", "evidence_purpose": "profit evidence"},
                    {"id": "cash_flow_quality", "name": "Cash Flow Quality", "evidence_purpose": "cash evidence"},
                    {"id": "balance_sheet_and_capital_intensity", "name": "Balance Sheet And Capital Intensity", "evidence_purpose": "balance evidence"},
                    {"id": "moat_and_competitive_risk", "name": "Moat And Competitive Risk", "evidence_purpose": "risk evidence"},
                    {"id": "valuation_and_risk_boundary", "name": "Valuation Boundary", "evidence_purpose": "valuation boundary"},
                ],
            },
        )
    ).model_dump(exclude_none=True)

    reqs = plan["evidence_requirements"]
    assert plan["analysis_scope"] == "single_company"
    assert plan["time_policy"] == "latest_available"
    assert len(reqs) >= 34
    required = [req for req in reqs if req["required"]]
    assert len(required) >= 15
    numeric_required = [
        (req["dimension_id"], req["requirement_type"], req.get("metric"))
        for req in required
        if req["requirement_type"] in {"numeric", "calculation"}
    ]
    assert ("revenue_quality", "numeric", "revenue") in numeric_required
    assert ("profitability_quality", "numeric", "net_income") in numeric_required
    assert ("profitability_quality", "calculation", "net_margin") in numeric_required
    assert ("cash_flow_quality", "numeric", "operating_cash_flow") in numeric_required
    assert ("cash_flow_quality", "numeric", "free_cash_flow") in numeric_required
    assert ("balance_sheet_and_capital_intensity", "numeric", "cash_and_equivalents") in numeric_required
    assert ("balance_sheet_and_capital_intensity", "numeric", "total_debt") in numeric_required
    assert ("balance_sheet_and_capital_intensity", "numeric", "total_assets") in numeric_required
    assert ("balance_sheet_and_capital_intensity", "numeric", "total_liabilities") in numeric_required
    assert ("balance_sheet_and_capital_intensity", "numeric", "shareholders_equity") in numeric_required
    assert ("balance_sheet_and_capital_intensity", "numeric", "capital_expenditure") in numeric_required
    assert ("valuation_and_risk_boundary", "numeric", "price") in numeric_required
    assert ("valuation_and_risk_boundary", "numeric", "shares_outstanding") in numeric_required
    assert ("valuation_and_risk_boundary", "calculation", "market_cap") in numeric_required
    assert any(
        req["dimension_id"] == "revenue_quality"
        and req["requirement_type"] == "calculation"
        and req.get("metric") == "revenue_growth"
        and not req["required"]
        for req in reqs
    )
    assert any(
        req["dimension_id"] == "profitability_quality"
        and req["requirement_type"] == "numeric"
        and req.get("metric") == "net_income"
        and req.get("min_results") == 3
        and not req["required"]
        for req in reqs
    )
    assert any(
        req["dimension_id"] == "cash_flow_quality"
        and req["requirement_type"] == "calculation"
        and req.get("metric") == "free_cash_flow"
        and not req["required"]
        for req in reqs
    )
    assert any(
        req["dimension_id"] == "cash_flow_quality"
        and req["requirement_type"] == "calculation"
        and req.get("metric") == "fcf_margin"
        and not req["required"]
        for req in reqs
    )
    assert {
        req.get("metric")
        for req in reqs
        if req["requirement_type"] == "numeric" and req["dimension_id"] == "profitability_quality" and not req["required"]
    } == {"net_income", "gross_margin", "operating_margin", "eps"}

    business = next(req for req in reqs if req["requirement_id"] == "REQ-TEXT-NVDA-BUSINESS_MODEL")
    assert business["requirement_type"] == "text"
    assert business["required"] is False
    assert business["requirement_scope"] == "optional_context"
    assert business["section_preferences"] == ["ITEM_1", "BUSINESS"]
    assert business["fallback_sections"] == ["ITEM_7", "MD&A"]
    assert business["retrieval_intent"] == "single_company_business_model"
    assert "business overview products services revenue sources" in business["retrieval_query"]
    assert "AWS Prime marketplace fulfillment" not in business["retrieval_query"]
    assert {
        "NVDA Data Center Compute & Networking segment revenue",
        "NVDA networking InfiniBand Ethernet NVLink Spectrum-X",
        "NVDA products services customers markets",
    } <= set(business["broadened_queries"])
    assert business["retrieval_profile"] == "summary"

    risk = next(req for req in reqs if req["requirement_id"] == "REQ-TEXT-NVDA-RISK")
    assert risk["requirement_type"] == "text"
    assert risk["required"] is True
    assert risk["requirement_scope"] == "core"
    assert risk["section_preferences"] == ["ITEM_1A"]
    assert risk["fallback_sections"] == ["ITEM_7", "MD&A", "ITEM_1", "BUSINESS"]
    assert risk["retrieval_intent"] == "single_company_risk_context"
    assert "risk factors competition demand supply chain regulation customer concentration" in risk["retrieval_query"]
    assert {
        "NVDA competition risks",
        "NVDA demand supply chain risks",
        "NVDA regulatory customer concentration risks",
    } <= set(risk["broadened_queries"])
    assert risk["retrieval_profile"] == "risk_summary"

    competition = next(req for req in reqs if req["requirement_id"] == "REQ-TEXT-NVDA-COMPETITION")
    assert competition["dimension_id"] == "moat_and_competitive_risk"
    assert competition["section_preferences"] == ["ITEM_1", "ITEM_7", "ITEM_1A"]
    assert competition["fallback_sections"] == ["BUSINESS", "MD&A"]
    assert competition["retrieval_intent"] == "single_company_competition_context"
    assert {
        "NVDA competitive position",
        "NVDA market position products customers",
        "NVDA industry competition",
    } <= set(competition["broadened_queries"])

    mda = next(req for req in reqs if req["requirement_id"] == "REQ-METH-NVDA-PROFITABILITY_QUALITY_MDA_TEXT")
    assert mda["required"] is False
    assert mda["retrieval_intent"] == "single_company_operating_context"
    assert mda["section_preferences"] == ["ITEM_7", "MD&A"]

    valuation_metrics = {
        (req["requirement_type"], req.get("metric"), req["required"])
        for req in reqs
        if req["dimension_id"] == "valuation_and_risk_boundary"
    }
    assert ("numeric", "price", True) in valuation_metrics
    assert ("numeric", "shares_outstanding", True) in valuation_metrics
    assert ("numeric", "revenue", True) in valuation_metrics
    assert ("numeric", "net_income", True) in valuation_metrics
    assert ("numeric", "free_cash_flow", False) in valuation_metrics
    assert ("calculation", "market_cap", True) in valuation_metrics
    assert ("calculation", "pe_ratio", False) in valuation_metrics
    assert ("calculation", "ps_ratio", False) in valuation_metrics
    assert ("calculation", "fcf_yield", False) in valuation_metrics
    assert not any(
        req.get("fallback_strategy") == ["valuation_evidence_missing"]
        for req in reqs
        if req["dimension_id"] == "valuation_and_risk_boundary"
    )


def test_risk_focused_plan_prioritizes_risk_text():
    plan = build_evidence_plan(
        _state(
            user_query="nvidia现在最大的问题是什么",
            task_type="report_summary",
            answer_mode="risk_focused_analysis",
            analysis_scope="single_company",
            primary_dimension="moat_and_competitive_risk",
            required_dimensions=["moat_and_competitive_risk"],
            optional_dimensions=["business_model", "revenue_quality", "profitability_quality", "cash_flow_quality", "valuation_and_risk_boundary"],
            companies=["NVDA"],
            requested_metrics=["revenue", "net_income"],
            analysis_plan={
                "companies": ["NVDA"],
                "answer_mode": "risk_focused_analysis",
                "analysis_scope": "single_company",
                "primary_dimension": "moat_and_competitive_risk",
                "required_dimensions": ["moat_and_competitive_risk"],
                "optional_dimensions": ["business_model", "revenue_quality", "profitability_quality", "cash_flow_quality", "valuation_and_risk_boundary"],
            },
            selected_analysis_framework={
                "framework_id": "fundamental_quality_analysis",
                "active_dimension_ids": [
                    "business_model",
                    "moat_and_competitive_risk",
                    "revenue_quality",
                    "profitability_quality",
                ],
                "dimensions": [
                    {"id": "business_model", "name": "Business Model", "evidence_purpose": "business evidence"},
                    {"id": "moat_and_competitive_risk", "name": "Moat And Competitive Risk", "evidence_purpose": "risk evidence"},
                    {"id": "revenue_quality", "name": "Revenue Quality", "evidence_purpose": "revenue evidence"},
                    {"id": "profitability_quality", "name": "Profitability Quality", "evidence_purpose": "profit evidence"},
                ],
            },
        )
    ).model_dump(exclude_none=True)

    reqs = plan["evidence_requirements"]
    assert plan["answer_mode"] == "risk_focused_analysis"
    assert plan["primary_dimension"] == "moat_and_competitive_risk"
    assert 6 <= len(reqs) <= 10
    required_text = [req for req in reqs if req["requirement_type"] == "text" and req["required"]]
    optional_text = [req for req in reqs if req["requirement_type"] == "text" and not req["required"]]
    assert len(required_text) == 1
    assert {req["requirement_id"] for req in required_text} == {"REQ-TEXT-NVDA-RISK_FACTORS"}
    assert required_text[0]["requirement_scope"] == "core"
    assert {"REQ-TEXT-NVDA-RISK_BUSINESS_MODEL", "REQ-TEXT-NVDA-RISK_MDA"} <= {
        req["requirement_id"] for req in optional_text
    }
    assert {req["requirement_scope"] for req in optional_text} == {"optional_context"}
    assert plan["core_requirement_ids"] == ["REQ-TEXT-NVDA-RISK_FACTORS"]
    assert {"REQ-TEXT-NVDA-RISK_BUSINESS_MODEL", "REQ-TEXT-NVDA-RISK_MDA"} <= set(
        plan["optional_context_requirement_ids"]
    )
    risk = next(req for req in required_text if req["requirement_id"] == "REQ-TEXT-NVDA-RISK_FACTORS")
    assert risk["dimension_id"] == "moat_and_competitive_risk"
    assert risk["section_preferences"] == ["ITEM_1A"]
    assert risk["retrieval_intent"] == "risk_focused_risk_factors"
    assert "demand" in risk["retrieval_query"]
    mda = next(req for req in optional_text if req["requirement_id"] == "REQ-TEXT-NVDA-RISK_MDA")
    assert mda["section_preferences"] == ["ITEM_7", "MD&A"]
    assert "ITEM_2" in mda["fallback_sections"]
    assert not any(req.get("dimension_id") == "valuation_and_risk_boundary" for req in reqs)
    assert not any(req.get("metric") in {"operating_cash_flow", "free_cash_flow", "cash_and_equivalents", "total_debt"} for req in reqs)


def test_risk_focused_does_not_require_valuation():
    plan = build_evidence_plan(
        _state(
            user_query="NVDA 最大风险是什么",
            task_type="report_summary",
            answer_mode="risk_focused_analysis",
            analysis_scope="single_company",
            companies=["NVDA"],
            analysis_plan={"companies": ["NVDA"], "answer_mode": "risk_focused_analysis", "analysis_scope": "single_company"},
        )
    ).model_dump(exclude_none=True)

    assert not any(req.get("dimension_id") == "valuation_and_risk_boundary" for req in plan["evidence_requirements"])
    assert not any(req.get("metric") in {"price", "market_cap", "pe_ratio", "ps_ratio", "fcf_yield"} for req in plan["evidence_requirements"])


def test_methodology_metrics_are_allowed_not_rejected():
    req, rejected = validate_evidence_requirement(
        {
            "requirement_id": "REQ-METH-AAPL-CASH_FLOW_QUALITY-FCF",
            "requirement_type": "numeric",
            "company": "AAPL",
            "metric": "free_cash_flow",
            "metrics": ["free_cash_flow"],
            "period_type": "latest",
            "required": True,
            "min_results": 1,
            "framework_id": "fundamental_quality_analysis",
            "dimension_id": "cash_flow_quality",
            "dimension_name": "Cash Flow Quality",
            "analysis_purpose": "cash flow evidence",
        }
    )

    assert rejected == []
    assert req is not None
    assert req.metric == "free_cash_flow"
    assert req.dimension_id == "cash_flow_quality"


def test_open_ended_analytical_query_uses_intent_specific_text_requirements():
    plan = build_evidence_plan(
        _state(
            user_query="苹果现在最大的问题是什么？",
            task_type="report_summary",
            answer_mode="analytical",
            companies=["AAPL"],
            analysis_plan={"companies": ["AAPL"]},
        )
    ).model_dump(exclude_none=True)

    text_required = [r for r in plan["evidence_requirements"] if r["requirement_type"] == "text" and r["required"]]
    assert len(text_required) == 2
    assert {r["retrieval_intent"] for r in text_required} == {"biggest_problem"}
    assert {tuple(r["section_preferences"]) for r in text_required} == {("ITEM_1A",), ("ITEM_7",)}
    assert {tuple(r["fallback_sections"]) for r in text_required} == {
        ("ITEM_7", "ITEM_1", "ITEM_2"),
        ("ITEM_1A", "ITEM_1", "ITEM_2"),
    }


def test_analytical_text_intent_uses_structured_risk_state_not_raw_query():
    plan = build_evidence_plan(
        _state(
            user_query="neutral retrieval seed",
            task_type="report_summary",
            answer_mode="analytical",
            methodology_intent="risk_focused_analysis",
            analysis_scope="single_company",
            primary_dimension="moat_and_competitive_risk",
            requested_metrics=[],
            companies=["AAPL"],
            analysis_plan={
                "companies": ["AAPL"],
                "methodology_intent": "risk_focused_analysis",
                "analysis_scope": "single_company",
                "primary_dimension": "moat_and_competitive_risk",
            },
        )
    ).model_dump(exclude_none=True)

    text_required = [r for r in plan["evidence_requirements"] if r["requirement_type"] == "text" and r["required"]]
    assert {r["retrieval_intent"] for r in text_required} == {"major_risks"}
    assert all("neutral retrieval seed" in r["retrieval_query"] for r in text_required)


def test_structured_methodology_intents_bind_dimensions_without_raw_keywords():
    cases = [
        ("cash_flow_quality_analysis", "cash_flow_quality", {"operating_cash_flow", "free_cash_flow"}),
        ("balance_sheet_analysis", "balance_sheet_and_capital_intensity", {"cash_and_equivalents", "total_debt"}),
        ("valuation_boundary_analysis", "valuation_and_risk_boundary", {"price"}),
    ]
    for intent, dimension_id, expected_metrics in cases:
        plan = build_evidence_plan(
            _state(
                user_query="neutral retrieval seed",
                task_type="report_summary",
                answer_mode="analytical",
                methodology_intent=intent,
                analysis_scope="single_company",
                primary_dimension=dimension_id,
                required_dimensions=[dimension_id],
                optional_dimensions=[],
                requested_metrics=[],
                companies=["AAPL"],
                analysis_plan={
                    "companies": ["AAPL"],
                    "methodology_intent": intent,
                    "analysis_scope": "single_company",
                    "primary_dimension": dimension_id,
                    "required_dimensions": [dimension_id],
                    "optional_dimensions": [],
                },
                selected_analysis_framework=_framework(dimension_id),
            )
        ).model_dump(exclude_none=True)

        reqs = plan["evidence_requirements"]
        assert {req["dimension_id"] for req in reqs if req.get("dimension_id")} == {dimension_id}
        assert expected_metrics <= {req.get("metric") for req in reqs if req.get("metric")}


def test_comparison_risk_intent_reason_uses_risk_specific_text_requirements():
    plan = build_evidence_plan(
        _state(
            user_query="neutral comparison retrieval seed",
            task_type="company_comparison",
            answer_mode="comparison_brief",
            companies=["AAPL", "AMZN"],
            comparison_target="AMZN",
            analysis_plan={"companies": ["AAPL", "AMZN"]},
            query_understanding_summary={"intent_reasons": ["comparison_family", "comparison_risk_family"]},
        )
    ).model_dump(exclude_none=True)

    text_required = [r for r in plan["evidence_requirements"] if r["requirement_type"] == "text" and r["required"]]
    assert {r["company"] for r in text_required} == {"AAPL", "AMZN"}
    assert {r["retrieval_intent"] for r in text_required} == {"comparison_risk"}
    assert all(r["retrieval_profile"] == "risk_summary" for r in text_required)
    assert all("neutral comparison retrieval seed" in r["retrieval_query"] for r in text_required)


def test_meta_clarification_and_refusal_generate_no_tool_requirements():
    for answer_mode in ("meta", "clarification", "refusal_or_redirect"):
        plan = build_evidence_plan(_state(answer_mode=answer_mode, needs_tools=False)).model_dump(exclude_none=True)
        assert plan["evidence_requirements"] == []
        assert plan["sufficiency_criteria"]["required_count"] == 0


def test_intent_family_plans_bind_dimensions_to_requirements():
    from datetime import date

    from src.agent.query_plan import build_classification_state

    cases = [
        ("苹果现在贵不贵", "valuation_boundary_analysis", {"valuation_and_risk_boundary"}, {"price"}),
        ("苹果利润能不能变成现金", "cash_flow_quality_analysis", {"cash_flow_quality"}, {"operating_cash_flow", "free_cash_flow"}),
        ("苹果抗风险能力怎么样", "balance_sheet_analysis", {"balance_sheet_and_capital_intensity"}, {"cash_and_equivalents", "total_debt"}),
    ]
    for query, intent, dimensions, metrics in cases:
        state = build_classification_state(user_query=query, parsed={}, trace_id="planner-intent", today=date(2026, 4, 24))
        plan = state["evidence_plan"]
        reqs = plan["evidence_requirements"]

        assert plan["methodology_intent"] == intent
        assert dimensions == {req["dimension_id"] for req in reqs if req.get("dimension_id")}
        assert metrics <= {req.get("metric") for req in reqs if req.get("metric")}


def test_risk_focused_intent_family_does_not_over_require_full_methodology():
    from datetime import date

    from src.agent.query_plan import build_classification_state

    state = build_classification_state(
        user_query="nvidia 最大的问题是什么",
        parsed={},
        trace_id="planner-risk",
        today=date(2026, 4, 24),
    )
    reqs = state["evidence_plan"]["evidence_requirements"]

    assert state["methodology_intent"] == "risk_focused_analysis"
    assert 6 <= len(reqs) <= 10
    assert any(req["required"] and req["requirement_type"] == "text" for req in reqs)
    assert not any(req.get("dimension_id") == "valuation_and_risk_boundary" for req in reqs)
    assert not any(req.get("dimension_id") == "cash_flow_quality" for req in reqs)
    assert not any(req.get("dimension_id") == "balance_sheet_and_capital_intensity" for req in reqs)


def test_invalid_requirement_fields_are_rejected():
    req, rejected = validate_evidence_requirement(
        {
            "requirement_id": "REQ-BAD",
            "requirement_type": "web",
            "company": "FAKE",
            "metric": "free_cash_flow",
            "metrics": ["revenue", "magic_metric"],
            "period_type": "weekly",
            "section_preferences": ["ITEM_99"],
            "retrieval_query": "",
            "required": True,
            "min_results": 0,
        }
    )

    assert req is None
    reasons = {item["reason"] for item in rejected}
    assert "requirement_type_not_allowed" in reasons
    assert "unknown_or_unsupported_ticker" in reasons
    assert "metric_not_allowed" in reasons
    assert "section_not_allowed" in reasons
    assert "min_results_must_be_at_least_1" in reasons


def test_sufficiency_distinguishes_sufficient_partial_and_insufficient():
    plan = build_evidence_plan(
        _state(task_type="company_comparison", answer_mode="comparison_brief", companies=["AAPL", "AMZN"], comparison_target="AMZN")
    ).model_dump(exclude_none=True)
    req_ids = [r["requirement_id"] for r in plan["evidence_requirements"] if r["required"]]

    insufficient = evaluate_evidence_sufficiency(plan, []).model_dump()
    partial = evaluate_evidence_sufficiency(
        plan,
        [{"requirement_id": req_ids[0], "status": "satisfied"}],
    ).model_dump()
    sufficient = evaluate_evidence_sufficiency(
        plan,
        [{"requirement_id": rid, "status": "satisfied"} for rid in req_ids],
    ).model_dump()

    assert insufficient["overall_status"] == "insufficient"
    assert partial["overall_status"] == "partial"
    assert sufficient["overall_status"] == "sufficient"
