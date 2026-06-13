"""Tests for evidence sufficiency policy."""

from __future__ import annotations

from src.agent.evidence_sufficiency import (
    build_trace_summary,
    build_validated_collection_results,
    evaluate_evidence_sufficiency,
    finalize_evidence_accounting,
)
from src.agent.evidence_planner import build_requirements_from_research_plan
from src.agent.plan_validator import deterministic_causal_research_plan


def _req(rid: str, req_type: str, company: str = "AAPL", *, required: bool = True):
    return {
        "requirement_id": rid,
        "requirement_type": req_type,
        "company": company,
        "required": required,
        "min_results": 1,
    }


def _plan(requirements, **overrides):
    base = {
        "task_type": "fact_qa",
        "answer_mode": "direct_fact",
        "safety_intent": "normal",
        "evidence_requirements": requirements,
        "rejected_requirements": [],
    }
    base.update(overrides)
    return base


def _result(rid: str, status: str, evidence_type: str = "numeric"):
    return {"requirement_id": rid, "status": status, "evidence_type": evidence_type, "items": [{}] if status == "satisfied" else []}


def _causal_evidence_plan() -> dict:
    plan = deterministic_causal_research_plan(
        user_query="为什么 NVIDIA 的营收增长这么多",
        companies=["NVDA"],
    ).model_dump(exclude_none=True)
    return build_requirements_from_research_plan(
        {
            "user_query": "为什么 NVIDIA 的营收增长这么多",
            "task_type": "fact_qa",
            "answer_mode": "direct_fact",
            "safety_intent": "normal",
            "companies": ["NVDA"],
        },
        plan,
    ).model_dump(exclude_none=True)


def _req_by_role(plan: dict, role: str) -> dict:
    return next(req for req in plan["evidence_requirements"] if req.get("evidence_role") == role)


def test_required_missing_is_insufficient():
    out = evaluate_evidence_sufficiency(_plan([_req("REQ-NUM-AAPL", "numeric")]), []).model_dump()

    assert out["overall_status"] == "insufficient"
    assert out["missing_requirements"] == ["REQ-NUM-AAPL"]
    assert out["can_synthesize"] is False


def test_causal_single_period_revenue_only_makes_quantify_growth_partial():
    plan = _causal_evidence_plan()
    current = _req_by_role(plan, "current_revenue")

    out = evaluate_evidence_sufficiency(
        plan,
        [
            {
                "requirement_id": current["requirement_id"],
                "status": "satisfied",
                "evidence_type": "numeric",
                "items": [{"evidence_id": "N1", "metric": "revenue", "period_end": "2026-01-31"}],
            }
        ],
    ).model_dump(exclude_none=True)

    assert out["answer_part_status_by_id"]["quantify_growth"]["status"] == "partial"
    assert out["answer_part_status_by_id"]["quantify_growth"]["reason"] == "growth_calc_unavailable_or_segment_only"
    assert out["answer_part_status_by_id"]["quantify_growth"]["current_revenue"]["status"] == "satisfied"
    assert out["answer_part_status_by_id"]["quantify_growth"]["comparator_revenue"]["status"] == "missing"
    assert "quantify_growth" in out["partial_required_answer_parts"]
    assert out["answer_part_status_by_id"]["identify_growth_drivers"]["status"] == "missing_but_analyzable"
    assert "identify_growth_drivers" in out["missing_but_analyzable_answer_parts"]
    assert out["evidence_health"] == "degraded"
    assert out["answer_parts_clean_pass"] is False


def test_causal_total_growth_text_satisfies_quantify_growth():
    plan = _causal_evidence_plan()
    growth_text = _req_by_role(plan, "revenue_growth_text")

    out = evaluate_evidence_sufficiency(
        plan,
        [
            {
                "requirement_id": growth_text["requirement_id"],
                "status": "satisfied",
                "evidence_type": "text",
                "items": [
                    {
                        "evidence_id": "T1",
                        "supporting_snippet": "Total revenue increased 114% year over year as overall revenue growth was driven by demand.",
                        "section": "ITEM_7",
                    }
                ],
            }
        ],
    ).model_dump(exclude_none=True)

    assert out["answer_part_status_by_id"]["quantify_growth"]["status"] == "satisfied"


def test_causal_same_period_comparator_has_specific_partial_reason():
    plan = _causal_evidence_plan()
    current = _req_by_role(plan, "current_revenue")
    comparator = _req_by_role(plan, "comparator_revenue")
    calc = _req_by_role(plan, "revenue_growth_calculation")

    out = evaluate_evidence_sufficiency(
        plan,
        [
            {
                "requirement_id": current["requirement_id"],
                "status": "satisfied",
                "evidence_type": "numeric",
                "items": [{"evidence_id": "N1", "metric": "revenue", "period_end": "2026-01-31", "period_type": "annual"}],
            },
            {
                "requirement_id": comparator["requirement_id"],
                "status": "missing",
                "evidence_type": "numeric",
                "items": [],
                "failure_reason": "same_period_comparator",
                "quality_status": "same_period_comparator",
            },
            {
                "requirement_id": calc["requirement_id"],
                "status": "missing",
                "evidence_type": "calculation",
                "items": [],
                "failure_reason": "same_period_comparator",
                "quality_status": "same_period_comparator",
            },
        ],
    ).model_dump(exclude_none=True)

    quantify = out["answer_part_status_by_id"]["quantify_growth"]
    assert quantify["status"] == "partial"
    assert quantify["reason"] == "growth_calc_invalid_same_period"
    assert quantify["comparator_revenue"]["quality_status"] == "same_period_comparator"
    assert quantify["revenue_growth_calculation"]["quality_status"] == "same_period_comparator"


def test_causal_segment_driver_text_is_only_partial_driver_evidence():
    plan = _causal_evidence_plan()
    driver = _req_by_role(plan, "driver_text")

    out = evaluate_evidence_sufficiency(
        plan,
        [
            {
                "requirement_id": driver["requirement_id"],
                "status": "satisfied",
                "evidence_type": "text",
                "items": [
                    {
                        "evidence_id": "T1",
                        "supporting_snippet": "Data Center networking revenue grew 142% driven by NVLink and InfiniBand platforms.",
                        "section": "ITEM_7",
                    }
                ],
            }
        ],
    ).model_dump(exclude_none=True)

    status = out["answer_part_status_by_id"]["identify_growth_drivers"]
    assert status["status"] == "partial"
    assert status["reason"] == "only_segment_or_product_driver_evidence"
    assert "product_level_driver" in status["driver_levels"]
    assert "partial_required_answer_parts" in out
    assert "identify_growth_drivers" in out["partial_required_answer_parts"]


def test_company_comparison_requires_numeric_for_both_companies():
    plan = _plan(
        [_req("REQ-NUM-AAPL", "numeric", "AAPL"), _req("REQ-NUM-AMZN", "numeric", "AMZN")],
        task_type="company_comparison",
        answer_mode="comparison_brief",
    )

    partial = evaluate_evidence_sufficiency(plan, [_result("REQ-NUM-AAPL", "satisfied")]).model_dump()
    sufficient = evaluate_evidence_sufficiency(
        plan,
        [_result("REQ-NUM-AAPL", "satisfied"), _result("REQ-NUM-AMZN", "satisfied")],
    ).model_dump()

    assert partial["overall_status"] == "partial"
    assert partial["degradation_reason"] == "comparison_numeric_evidence_missing"
    assert sufficient["overall_status"] == "sufficient"


def test_investment_advice_like_can_synthesize_numeric_only_with_degradation():
    plan = _plan(
        [
            _req("REQ-NUM-AAPL", "numeric", "AAPL"),
            _req("REQ-NUM-AMZN", "numeric", "AMZN"),
            _req("REQ-TEXT-AAPL", "text", "AAPL"),
            _req("REQ-TEXT-AMZN", "text", "AMZN"),
        ],
        task_type="company_comparison",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [_result("REQ-NUM-AAPL", "satisfied"), _result("REQ-NUM-AMZN", "satisfied")],
    ).model_dump()

    assert out["overall_status"] == "partial"
    assert out["degradation_reason"] == "numeric_only_comparison"
    assert out["can_synthesize"] is True


def test_cautious_outlook_without_text_is_limited_outlook():
    plan = _plan(
        [
            _req("REQ-NUM-AAPL-OUTLOOK", "numeric", "AAPL"),
            _req("REQ-TEXT-AAPL-MDA", "text", "AAPL"),
            _req("REQ-TEXT-AAPL-RISK", "text", "AAPL"),
        ],
        answer_mode="cautious_outlook",
    )

    out = evaluate_evidence_sufficiency(plan, [_result("REQ-NUM-AAPL-OUTLOOK", "satisfied")]).model_dump()

    assert out["overall_status"] == "partial"
    assert out["degradation_reason"] == "limited_outlook"
    assert out["can_synthesize"] is True


def test_analytical_without_text_cannot_synthesize_strong_analysis():
    plan = _plan(
        [_req("REQ-TEXT-AAPL-ANALYSIS", "text", "AAPL"), _req("REQ-NUM-AAPL", "numeric", "AAPL", required=False)],
        task_type="report_summary",
        answer_mode="analytical",
    )

    out = evaluate_evidence_sufficiency(plan, [_result("REQ-NUM-AAPL", "satisfied")]).model_dump()

    assert out["overall_status"] == "insufficient"
    assert out["degradation_reason"] == "text_evidence_missing"
    assert out["can_synthesize"] is False


def test_analytical_with_partial_validated_text_can_synthesize_limited_analysis():
    plan = _plan(
        [
            _req("REQ-TEXT-AAPL-RISK", "text", "AAPL"),
            _req("REQ-TEXT-AAPL-MDA", "text", "AAPL"),
        ],
        task_type="report_summary",
        answer_mode="analytical",
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [_result("REQ-TEXT-AAPL-RISK", "satisfied", "text")],
    ).model_dump()

    assert out["overall_status"] == "partial"
    assert out["degradation_reason"] == "text_evidence_partial"
    assert out["can_synthesize"] is True


def test_rejected_required_requirement_degrades_result():
    plan = _plan(
        [_req("REQ-NUM-AAPL", "numeric", "AAPL")],
        rejected_requirements=[{"requirement_id": "REQ-BAD", "required": True, "reason": "metric_not_allowed"}],
    )

    out = evaluate_evidence_sufficiency(plan, []).model_dump()

    assert out["overall_status"] == "insufficient"
    assert out["degradation_reason"] == "rejected_required_requirement"
    assert out["rejected_requirements"]


def test_sufficient_result_has_no_requirement_missing_limitation():
    plan = _plan([_req("REQ-NUM-AAPL", "numeric")])

    out = evaluate_evidence_sufficiency(plan, [_result("REQ-NUM-AAPL", "satisfied")]).model_dump()

    assert out["overall_status"] == "sufficient"
    assert not any(item["code"] == "requirement_missing" for item in out["requirement_limitations"])


def test_imbalanced_company_evidence_adds_limitation():
    plan = _plan(
        [
            _req("REQ-NUM-AAPL", "numeric", "AAPL"),
            _req("REQ-NUM-AMZN", "numeric", "AMZN"),
            _req("REQ-TEXT-AAPL", "text", "AAPL"),
            _req("REQ-TEXT-AMZN", "text", "AMZN"),
        ],
        task_type="company_comparison",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            _result("REQ-NUM-AAPL", "satisfied"),
            _result("REQ-NUM-AMZN", "satisfied"),
            _result("REQ-TEXT-AAPL", "satisfied", "text"),
        ],
    ).model_dump()

    assert out["overall_status"] == "partial"
    assert out["degradation_reason"] == "numeric_only_comparison"
    assert any(item["code"] == "imbalanced_company_evidence" for item in out["requirement_limitations"])


def test_dependency_missing_calculation_is_reflected_in_requirement_limitations():
    plan = _plan(
        [
            _req("REQ-NUM-AAPL", "numeric", "AAPL"),
            _req("REQ-CALC-AAPL", "calculation", "AAPL", required=False),
        ],
        answer_mode="analytical",
        task_type="report_summary",
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            {
                "requirement_id": "REQ-CALC-AAPL",
                "status": "missing",
                "evidence_type": "calculation",
                "items": [],
                "failure_reason": "dependency_numeric_requirement_missing",
            },
        ],
    ).model_dump()

    calc_limitation = next(item for item in out["requirement_limitations"] if item.get("requirement_id") == "REQ-CALC-AAPL")
    assert calc_limitation["code"] == "requirement_missing"
    assert calc_limitation["failure_reason"] == "dependency_numeric_requirement_missing"


def test_dimension_sufficiency_statuses_and_hard_rules():
    plan = _plan(
        [
            {
                **_req("REQ-NUM-AAPL-CFO", "numeric", "AAPL"),
                "dimension_id": "cash_flow_quality",
                "dimension_name": "Cash Flow Quality",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-NUM-AAPL-FCF", "numeric", "AAPL"),
                "dimension_id": "cash_flow_quality",
                "dimension_name": "Cash Flow Quality",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-NUM-AAPL-REV", "numeric", "AAPL"),
                "dimension_id": "profitability_quality",
                "dimension_name": "Profitability Quality",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-CALC-AAPL-MARGIN", "calculation", "AAPL"),
                "dimension_id": "profitability_quality",
                "dimension_name": "Profitability Quality",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-NUM-AAPL-PRICE", "numeric", "AAPL"),
                "dimension_id": "valuation_and_risk_boundary",
                "dimension_name": "Valuation And Risk Boundary",
                "framework_id": "fundamental_quality_analysis",
            },
        ]
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            _result("REQ-NUM-AAPL-REV", "satisfied"),
            _result("REQ-NUM-AAPL-PRICE", "satisfied"),
        ],
    ).model_dump()

    dims = out["dimension_status_map"]
    assert dims["cash_flow_quality"]["status"] == "missing"
    assert dims["profitability_quality"]["status"] == "partial"
    assert dims["valuation_and_risk_boundary"]["status"] == "satisfied"
    assert "cash flow is strong" in dims["cash_flow_quality"]["forbidden_claims"]
    assert "based on net margin / net income evidence" in dims["profitability_quality"]["allowed_claims"]
    assert out["dimension_status_by_id"] == out["dimension_status_map"]
    assert out["covered_dimensions"] == ["valuation_and_risk_boundary"]
    assert out["covered_dimensions"] == out["satisfied_dimensions"]
    assert out["missing_dimensions"] == ["cash_flow_quality"]
    assert out["dimension_coverage_rate"] == 0.333333
    assert out["framework_sufficiency_status"] == "partial"


def test_single_company_sufficiency_allows_partial_methodology_answer():
    plan = _plan(
        [
            {
                **_req("REQ-BUS-NVDA", "text", "NVDA"),
                "dimension_id": "business_model",
                "dimension_name": "Business Model",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-REV-NVDA", "numeric", "NVDA"),
                "metric": "revenue",
                "metrics": ["revenue"],
                "dimension_id": "revenue_quality",
                "dimension_name": "Revenue Quality",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-NI-NVDA", "numeric", "NVDA"),
                "metric": "net_income",
                "metrics": ["net_income"],
                "dimension_id": "profitability_quality",
                "dimension_name": "Profitability Quality",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-MARGIN-NVDA", "calculation", "NVDA"),
                "metric": "net_margin",
                "metrics": ["net_margin"],
                "dimension_id": "profitability_quality",
                "dimension_name": "Profitability Quality",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-RISK-NVDA", "text", "NVDA"),
                "dimension_id": "moat_and_competitive_risk",
                "dimension_name": "Moat And Competitive Risk",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-VAL-NVDA", "calculation", None),
                "metric": "price",
                "metrics": ["price"],
                "dimension_id": "valuation_and_risk_boundary",
                "dimension_name": "Valuation Boundary",
                "framework_id": "fundamental_quality_analysis",
            },
        ],
        task_type="report_summary",
        answer_mode="analytical",
        analysis_scope="single_company",
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            _result("REQ-REV-NVDA", "satisfied"),
            _result("REQ-NI-NVDA", "satisfied"),
            _result("REQ-MARGIN-NVDA", "satisfied", "calculation"),
            {
                "requirement_id": "REQ-VAL-NVDA",
                "status": "missing",
                "evidence_type": "calculation",
                "items": [],
                "failure_reason": "valuation_evidence_missing",
            },
        ],
    ).model_dump()

    assert out["overall_status"] == "partial"
    assert out["can_synthesize"] is True
    assert out["degradation_reason"] == "valuation_evidence_missing"
    assert out["dimension_status_map"]["revenue_quality"]["status"] == "satisfied"
    assert out["dimension_status_map"]["revenue_quality"]["required_missing"] == []
    assert out["dimension_status_map"]["revenue_quality"]["enhanced_missing"] == []
    assert out["dimension_status_map"]["profitability_quality"]["status"] == "satisfied"
    assert out["dimension_status_map"]["business_model"]["status"] == "missing"
    assert out["dimension_status_map"]["moat_and_competitive_risk"]["status"] == "missing"
    assert out["dimension_status_map"]["valuation_and_risk_boundary"]["status"] == "missing"
    assert "profitability_quality" in out["covered_dimensions"]
    assert "profitability_quality" in out["covered_dimensions"]
    assert out["dimension_coverage_rate"] > 0


def test_single_company_core_numeric_missing_is_insufficient():
    plan = _plan(
        [
            {
                **_req("REQ-REV-NVDA", "numeric", "NVDA"),
                "metric": "revenue",
                "dimension_id": "revenue_quality",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-NI-NVDA", "numeric", "NVDA"),
                "metric": "net_income",
                "dimension_id": "profitability_quality",
                "framework_id": "fundamental_quality_analysis",
            },
        ],
        task_type="report_summary",
        answer_mode="analytical",
        analysis_scope="single_company",
    )

    out = evaluate_evidence_sufficiency(plan, []).model_dump()

    assert out["overall_status"] == "insufficient"
    assert out["can_synthesize"] is False
    assert out["degradation_reason"] == "core_numeric_evidence_missing"


def test_risk_focused_sufficiency_does_not_require_valuation():
    plan = _plan(
        [
            {
                **_req("REQ-TEXT-NVDA-RISK_BUSINESS_MODEL", "text", "NVDA"),
                "dimension_id": "business_model",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-TEXT-NVDA-RISK_FACTORS", "text", "NVDA"),
                "dimension_id": "moat_and_competitive_risk",
                "framework_id": "fundamental_quality_analysis",
            },
            {
                **_req("REQ-NUM-NVDA-REVENUE", "numeric", "NVDA", required=False),
                "metric": "revenue",
                "dimension_id": "revenue_quality",
                "framework_id": "fundamental_quality_analysis",
            },
        ],
        task_type="report_summary",
        answer_mode="risk_focused_analysis",
        analysis_scope="single_company",
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            _result("REQ-TEXT-NVDA-RISK_BUSINESS_MODEL", "satisfied", "text"),
            _result("REQ-TEXT-NVDA-RISK_FACTORS", "satisfied", "text"),
        ],
    ).model_dump()

    assert out["overall_status"] == "focused_sufficient"
    assert out["can_synthesize"] is True
    assert out["degradation_reason"] is None
    assert out["dimension_status_map"]["business_model"]["status"] == "satisfied"
    assert out["dimension_status_map"]["moat_and_competitive_risk"]["status"] == "satisfied"


def test_risk_focused_sufficiency_does_not_block_on_business_context_missing():
    plan = _plan(
        [
            {
                **_req("REQ-TEXT-NVDA-RISK_BUSINESS_MODEL", "text", "NVDA", required=False),
                "dimension_id": "business_model",
                "framework_id": "fundamental_quality_analysis",
                "requirement_scope": "optional_context",
            },
            {
                **_req("REQ-TEXT-NVDA-RISK_FACTORS", "text", "NVDA"),
                "dimension_id": "moat_and_competitive_risk",
                "framework_id": "fundamental_quality_analysis",
                "requirement_scope": "core",
            },
            {
                **_req("REQ-TEXT-NVDA-RISK_MDA", "text", "NVDA", required=False),
                "dimension_id": "moat_and_competitive_risk",
                "framework_id": "fundamental_quality_analysis",
                "requirement_scope": "optional_context",
            },
        ],
        task_type="report_summary",
        answer_mode="risk_focused_analysis",
        analysis_scope="single_company",
        required_dimensions=["moat_and_competitive_risk"],
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            _result("REQ-TEXT-NVDA-RISK_BUSINESS_MODEL", "missing", "text"),
            _result("REQ-TEXT-NVDA-RISK_FACTORS", "satisfied", "text"),
            _result("REQ-TEXT-NVDA-RISK_MDA", "missing", "text"),
        ],
    ).model_dump()

    assert out["overall_status"] == "focused_sufficient"
    assert out["can_synthesize"] is True
    assert out["degradation_reason"] is None
    assert out["required_text_satisfied_rate"] == 1.0
    assert out["missing_required_requirements_count"] == 0
    assert out["missing_optional_requirements_count"] == 2
    assert set(out["missing_optional_requirements"]) == {
        "REQ-TEXT-NVDA-RISK_BUSINESS_MODEL",
        "REQ-TEXT-NVDA-RISK_MDA",
    }
    assert "business_model" not in out["missing_dimensions"]
    assert out["requirement_limitations"] == []


def test_risk_focused_sufficiency_blocks_strong_risk_without_core_risk_factors():
    plan = _plan(
        [
            {
                **_req("REQ-TEXT-NVDA-RISK_BUSINESS_MODEL", "text", "NVDA", required=False),
                "dimension_id": "business_model",
                "framework_id": "fundamental_quality_analysis",
                "requirement_scope": "optional_context",
            },
            {
                **_req("REQ-TEXT-NVDA-RISK_FACTORS", "text", "NVDA"),
                "dimension_id": "moat_and_competitive_risk",
                "framework_id": "fundamental_quality_analysis",
                "requirement_scope": "core",
            },
            {
                **_req("REQ-TEXT-NVDA-RISK_MDA", "text", "NVDA", required=False),
                "dimension_id": "moat_and_competitive_risk",
                "framework_id": "fundamental_quality_analysis",
                "requirement_scope": "optional_context",
            },
        ],
        task_type="report_summary",
        answer_mode="risk_focused_analysis",
        analysis_scope="single_company",
        required_dimensions=["moat_and_competitive_risk"],
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            _result("REQ-TEXT-NVDA-RISK_BUSINESS_MODEL", "satisfied", "text"),
            _result("REQ-TEXT-NVDA-RISK_FACTORS", "missing", "text"),
            _result("REQ-TEXT-NVDA-RISK_MDA", "missing", "text"),
        ],
    ).model_dump()

    assert out["overall_status"] in {"partial", "insufficient"}
    assert out["overall_status"] != "focused_sufficient"
    assert out["can_synthesize"] is False
    assert out["missing_required_requirements_count"] == 1
    assert out["missing_required_requirements"] == ["REQ-TEXT-NVDA-RISK_FACTORS"]
    assert out["missing_optional_requirements"] == ["REQ-TEXT-NVDA-RISK_MDA"]
    assert out["dimension_status_map"]["moat_and_competitive_risk"]["status"] == "missing"


def test_single_company_explicit_dimensions_do_not_require_generic_core_numeric():
    plan = _plan(
        [
            {
                **_req("REQ-OCF", "numeric", "NVDA"),
                "metric": "operating_cash_flow",
                "dimension_id": "cash_flow_quality",
            },
            {
                **_req("REQ-FCF", "numeric", "NVDA"),
                "metric": "free_cash_flow",
                "dimension_id": "cash_flow_quality",
            },
            {
                **_req("REQ-CFO-NI", "calculation", "NVDA", required=False),
                "metric": "cfo_to_net_income",
                "dimension_id": "cash_flow_quality",
            },
            {
                **_req("REQ-RISK", "text", "NVDA"),
                "dimension_id": "moat_and_competitive_risk",
            },
            {
                **_req("REQ-PRICE", "numeric", "NVDA"),
                "metric": "price",
                "dimension_id": "valuation_and_risk_boundary",
            },
            {
                **_req("REQ-SHARES", "numeric", "NVDA"),
                "metric": "shares_outstanding",
                "dimension_id": "valuation_and_risk_boundary",
            },
            {
                **_req("REQ-MARKET-CAP", "calculation", "NVDA"),
                "metric": "market_cap",
                "dimension_id": "valuation_and_risk_boundary",
            },
            {
                **_req("REQ-PE", "calculation", "NVDA", required=False),
                "metric": "pe_ratio",
                "dimension_id": "valuation_and_risk_boundary",
            },
            {
                **_req("REQ-PS", "calculation", "NVDA", required=False),
                "metric": "ps_ratio",
                "dimension_id": "valuation_and_risk_boundary",
            },
        ],
        task_type="report_summary",
        answer_mode="analytical",
        analysis_scope="single_company",
        methodology_intent="risk_focused_analysis",
        required_dimensions=["cash_flow_quality", "valuation_and_risk_boundary", "moat_and_competitive_risk"],
    )
    results = [
        _result("REQ-OCF", "satisfied"),
        _result("REQ-FCF", "satisfied"),
        _result("REQ-CFO-NI", "satisfied", "calculation"),
        _result("REQ-RISK", "satisfied", "text"),
        _result("REQ-PRICE", "satisfied"),
        _result("REQ-SHARES", "satisfied"),
        _result("REQ-MARKET-CAP", "satisfied", "calculation"),
        _result("REQ-PE", "satisfied", "calculation"),
        _result("REQ-PS", "satisfied", "calculation"),
    ]

    out = evaluate_evidence_sufficiency(plan, results).model_dump()

    assert out["overall_status"] == "sufficient"
    assert out["can_synthesize"] is True
    assert out["degradation_reason"] is None
    assert out["dimension_status_map"]["cash_flow_quality"]["status"] == "satisfied"
    assert out["dimension_status_map"]["valuation_and_risk_boundary"]["status"] == "satisfied"
    assert out["dimension_status_map"]["moat_and_competitive_risk"]["status"] == "satisfied"


def test_single_company_overview_allows_limited_methodology_with_core_numeric_and_text():
    plan = _plan(
        [
            {
                **_req("REQ-NUM-NVDA-REVENUE", "numeric", "NVDA"),
                "metric": "revenue",
                "dimension_id": "revenue_quality",
            },
            {
                **_req("REQ-NUM-NVDA-NET_INCOME", "numeric", "NVDA"),
                "metric": "net_income",
                "dimension_id": "profitability_quality",
            },
            {
                **_req("REQ-CALC-NVDA-NET_MARGIN", "calculation", "NVDA"),
                "metric": "net_margin",
                "dimension_id": "profitability_quality",
            },
            {
                **_req("REQ-TEXT-NVDA-BUSINESS", "text", "NVDA"),
                "dimension_id": "business_model",
            },
            {
                **_req("REQ-CALC-VALUATION", "calculation", None),
                "metric": "price",
                "dimension_id": "valuation_and_risk_boundary",
                "fallback_strategy": ["valuation_evidence_missing"],
            },
        ],
        task_type="report_summary",
        answer_mode="analytical",
        analysis_scope="single_company",
        methodology_intent="single_company_overview",
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            _result("REQ-NUM-NVDA-REVENUE", "satisfied"),
            _result("REQ-NUM-NVDA-NET_INCOME", "satisfied"),
            _result("REQ-CALC-NVDA-NET_MARGIN", "satisfied"),
            _result("REQ-TEXT-NVDA-BUSINESS", "satisfied", "text"),
        ],
    ).model_dump()

    assert out["overall_status"] == "partial"
    assert out["can_synthesize"] is True
    assert out["degradation_reason"] == "valuation_evidence_missing"


def test_returned_numeric_without_final_validation_has_specific_reason():
    plan = _plan(
        [
            {
                **_req("REQ-VALUATION-PRICE", "numeric", "AMZN"),
                "metric": "price",
                "metrics": ["price"],
                "dimension_id": "valuation_and_risk_boundary",
            }
        ]
    )

    final_results = build_validated_collection_results(
        plan,
        [
            {
                "requirement_id": "REQ-VALUATION-PRICE",
                "status": "satisfied",
                "evidence_type": "numeric",
                "items": [
                    {
                        "evidence_id": "N1",
                        "metric": "adjusted_close",
                        "period_end": "2026-04-24",
                        "value": 100.0,
                        "source_requirement_id": "REQ-VALUATION-PRICE",
                    }
                ],
            }
        ],
        validated_numeric_evidence=[],
    )

    assert final_results[0]["status"] == "missing"
    assert final_results[0]["failure_reason"] == "valuation_price_quality_filter_rejected"
    assert final_results[0]["failure_reason"] != "no_validated_numeric_evidence"


def test_single_company_overview_business_model_gap_is_optional_when_risk_text_is_satisfied():
    plan = _plan(
        [
            {
                **_req("REQ-NUM-NVDA-REVENUE", "numeric", "NVDA"),
                "metric": "revenue",
                "dimension_id": "revenue_quality",
                "requirement_scope": "core",
            },
            {
                **_req("REQ-NUM-NVDA-NET_INCOME", "numeric", "NVDA"),
                "metric": "net_income",
                "dimension_id": "profitability_quality",
                "requirement_scope": "core",
            },
            {
                **_req("REQ-CALC-NVDA-NET_MARGIN", "calculation", "NVDA"),
                "metric": "net_margin",
                "dimension_id": "profitability_quality",
                "requirement_scope": "core",
            },
            {
                **_req("REQ-TEXT-NVDA-RISK", "text", "NVDA"),
                "dimension_id": "moat_and_competitive_risk",
                "requirement_scope": "core",
            },
            {
                **_req("REQ-TEXT-NVDA-BUSINESS", "text", "NVDA", required=False),
                "dimension_id": "business_model",
                "requirement_scope": "optional_context",
            },
        ],
        task_type="report_summary",
        answer_mode="analytical",
        analysis_scope="single_company",
        methodology_intent="single_company_overview",
        required_dimensions=["revenue_quality", "profitability_quality", "moat_and_competitive_risk"],
        optional_dimensions=["business_model"],
        evidence_policy_id="single_company_overview_v1",
    )

    out = evaluate_evidence_sufficiency(
        plan,
        [
            _result("REQ-NUM-NVDA-REVENUE", "satisfied"),
            _result("REQ-NUM-NVDA-NET_INCOME", "satisfied"),
            _result("REQ-CALC-NVDA-NET_MARGIN", "satisfied", "calculation"),
            _result("REQ-TEXT-NVDA-RISK", "satisfied", "text"),
        ],
    ).model_dump()

    assert out["overall_status"] == "sufficient"
    assert out["can_synthesize"] is True
    assert "REQ-TEXT-NVDA-BUSINESS" not in out["missing_required_requirements"]
    assert "REQ-TEXT-NVDA-BUSINESS" in out["missing_optional_requirements"]
    assert "business_model" not in out["missing_dimensions"]
    assert out["dimension_status_map"]["moat_and_competitive_risk"]["status"] == "satisfied"


def test_valuation_boundary_missing_can_render_boundary_without_cheap_expensive_claim():
    plan = _plan(
        [
            {
                **_req("REQ-CALC-VALUATION", "calculation", None),
                "metric": "price",
                "dimension_id": "valuation_and_risk_boundary",
                "fallback_strategy": ["valuation_evidence_missing"],
            },
        ],
        task_type="report_summary",
        answer_mode="analytical",
        analysis_scope="single_company",
        methodology_intent="valuation_boundary_analysis",
    )

    out = evaluate_evidence_sufficiency(plan, []).model_dump()

    assert out["overall_status"] == "partial"
    assert out["can_synthesize"] is True
    assert out["degradation_reason"] == "valuation_evidence_missing"


def test_finalized_text_requirement_uses_validated_bundle_not_raw_hits():
    plan = _plan(
        [
            _req("REQ-NUM-AAPL", "numeric", "AAPL"),
            _req("REQ-NUM-AMZN", "numeric", "AMZN"),
            _req("REQ-TEXT-AAPL", "text", "AAPL"),
            _req("REQ-TEXT-AMZN", "text", "AMZN"),
        ],
        task_type="company_comparison",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )
    raw_results = [
        _result("REQ-NUM-AAPL", "satisfied"),
        _result("REQ-NUM-AMZN", "satisfied"),
        _result("REQ-TEXT-AAPL", "satisfied", "text"),
        _result("REQ-TEXT-AMZN", "satisfied", "text"),
    ]

    finalized = finalize_evidence_accounting(
        plan,
        raw_results,
        validated_numeric_evidence=[
            {"requirement_id": "REQ-NUM-AAPL", "evidence_id": "N1"},
            {"requirement_id": "REQ-NUM-AMZN", "evidence_id": "N2"},
        ],
        validated_text_evidence=[],
        validation_failure_reasons={
            "REQ-TEXT-AAPL": "comparison_text_unbalanced",
            "REQ-TEXT-AMZN": "comparison_text_unbalanced",
        },
        synthesis_mode="limited_judgment",
    )

    assert finalized["evidence_sufficiency"]["overall_status"] == "partial"
    assert finalized["evidence_sufficiency"]["required_text_satisfied_rate"] == 0.0
    assert finalized["requirement_status_map"]["REQ-TEXT-AAPL"]["status"] == "missing"
    assert finalized["requirement_status_map"]["REQ-TEXT-AMZN"]["failure_reason"] == "comparison_text_unbalanced"
    assert finalized["trace_summary"]["final_synthesis_mode"] == "limited_judgment"


def test_trace_splits_required_optional_and_enhanced_missing_counts():
    plan = _plan(
        [
            {
                **_req("REQ-NUM-NVDA-CASH", "numeric", "NVDA"),
                "dimension_id": "balance_sheet_and_capital_intensity",
                "metric": "cash",
            },
            {
                **_req("REQ-NUM-NVDA-DEBT", "numeric", "NVDA"),
                "dimension_id": "balance_sheet_and_capital_intensity",
                "metric": "total_debt",
            },
            {
                **_req("REQ-NUM-NVDA-CAPEX", "numeric", "NVDA"),
                "dimension_id": "balance_sheet_and_capital_intensity",
                "metric": "capital_expenditure",
            },
            {
                **_req("REQ-NUM-NVDA-INVENTORY", "numeric", "NVDA", required=False),
                "dimension_id": "balance_sheet_and_capital_intensity",
                "metric": "inventory",
            },
        ],
        task_type="report_summary",
        answer_mode="analytical",
        analysis_scope="single_company",
        methodology_intent="balance_sheet_analysis",
    )
    results = [
        _result("REQ-NUM-NVDA-CASH", "satisfied"),
        _result("REQ-NUM-NVDA-DEBT", "satisfied"),
        _result("REQ-NUM-NVDA-CAPEX", "satisfied"),
    ]
    sufficiency = evaluate_evidence_sufficiency(plan, results).model_dump()
    trace = build_trace_summary(plan, results, sufficiency, synthesis_mode="methodology_single_company")

    assert trace["sufficiency_status"] == "sufficient"
    assert trace["missing_requirements_count"] == 0
    assert trace["total_missing_requirements_count"] == 1
    assert trace["missing_required_requirements_count"] == 0
    assert trace["missing_optional_requirements_count"] == 1
    assert trace["missing_enhanced_requirements_count"] >= 1
    assert trace["dimension_status_by_id"] == trace["dimension_status_map"]
    assert trace["covered_dimensions"] == trace["satisfied_dimensions"]
    assert trace["satisfied_dimensions"] == ["balance_sheet_and_capital_intensity"]
    assert trace["partial_dimensions"] == []
    assert trace["missing_dimensions"] == []
    balance_status = trace["dimension_status_by_id"]["balance_sheet_and_capital_intensity"]
    assert balance_status["status"] == "satisfied"
    assert balance_status["required_missing"] == []
    assert "inventory" in balance_status["enhanced_missing"]
    assert balance_status["limitations"]


def test_optional_duplicate_free_cash_flow_does_not_make_trace_partial():
    plan = _plan(
        [
            {
                **_req("REQ-OCF", "numeric", "NVDA"),
                "dimension_id": "cash_flow_quality",
                "metric": "operating_cash_flow",
            },
            {
                **_req("REQ-FCF", "numeric", "NVDA"),
                "dimension_id": "cash_flow_quality",
                "metric": "free_cash_flow",
            },
            {
                **_req("REQ-CFO-NI", "calculation", "NVDA", required=False),
                "dimension_id": "cash_flow_quality",
                "metric": "cfo_to_net_income",
            },
            {
                **_req("REQ-COMPUTED-FCF", "calculation", "NVDA", required=False),
                "dimension_id": "cash_flow_quality",
                "metric": "free_cash_flow",
            },
        ],
        task_type="report_summary",
        answer_mode="analytical",
        analysis_scope="single_company",
        methodology_intent="cash_flow_quality_analysis",
        required_dimensions=["cash_flow_quality"],
    )
    results = [
        _result("REQ-OCF", "satisfied"),
        _result("REQ-FCF", "satisfied"),
        _result("REQ-CFO-NI", "satisfied", "calculation"),
    ]

    sufficiency = evaluate_evidence_sufficiency(plan, results).model_dump()
    trace = build_trace_summary(plan, results, sufficiency, synthesis_mode="methodology_single_company")

    assert sufficiency["overall_status"] == "sufficient"
    assert sufficiency["degradation_reason"] is None
    assert trace["sufficiency_status"] == "sufficient"
    assert trace["missing_requirements_count"] == 0
    assert trace["total_missing_requirements_count"] == 1
    assert trace["missing_optional_requirements_count"] == 1
    assert trace["limitations_count"] == 0
