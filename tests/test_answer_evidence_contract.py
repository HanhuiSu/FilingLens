"""Unit tests for answer evidence contract checker."""

from __future__ import annotations

from copy import deepcopy

from src.agent.answer_contract import check_answer_contract, check_answer_evidence_contract


def _base_trace() -> dict:
    return {
        "final_answer": "市值为 $4.31T，P/E 为 100.36x，债务/权益约 7.02%。[N1][N2][N3]",
        "task_type": "report_summary",
        "analysis_scope": "single_company",
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "NVDA", "metric": "market_cap", "value": 4_311_463_935_165.4053, "unit": "USD"},
                {"evidence_id": "N2", "ticker": "NVDA", "metric": "pe_ratio", "value": 100.36, "unit": "ratio"},
                {"evidence_id": "N3", "ticker": "NVDA", "metric": "debt_to_equity", "value": 0.0702, "unit": "ratio"},
            ],
            "text_snippets": [{"evidence_id": "T1", "ticker": "NVDA", "dimension_id": "moat_and_competitive_risk"}],
            "dimension_status_map": {
                "valuation_and_risk_boundary": {
                    "status": "satisfied",
                    "supporting_evidence_ids": ["N1", "N2"],
                    "required_missing": [],
                    "enhanced_missing": [],
                },
                "balance_sheet_and_capital_intensity": {
                    "status": "satisfied",
                    "supporting_evidence_ids": ["N3"],
                    "required_missing": [],
                    "enhanced_missing": [],
                },
            },
        },
    }


def test_formatted_equivalent_numbers_and_citations_pass():
    result = check_answer_evidence_contract(_base_trace())

    assert result["passed"] is True
    assert result["metrics"]["numeric_grounding_rate"] == 1.0
    assert result["metrics"]["citation_validity_rate"] == 1.0


def test_material_claim_contract_ignores_renderer_headings():
    trace = _base_trace()
    trace["final_answer"] = "\n".join(
        [
            "Business Model And Revenue Sources",
            "Revenue Quality",
            "Profitability Quality",
            "Primary Risks",
            "证据边界",
            "市值为 $4.31T。[N1]",
        ]
    )

    result = check_answer_evidence_contract(trace)

    codes = [item.get("type") for item in result.get("violations", [])]
    assert "citation_free_material_claim" not in codes


def test_nvda_product_terms_are_not_company_token_leakage():
    answer = "Networking revenue grew 13% sequentially driven by XDR InfiniBand products, NVLink, and Ethernet for AI solutions.[T1]"
    state = {
        "final_answer": answer,
        "companies": ["NVDA"],
        "evidence_packet": {
            "text_evidence": [
                {
                    "evidence_id": "T1",
                    "ticker": "NVDA",
                    "claim": "Networking revenue grew 13% sequentially driven by XDR InfiniBand products, NVLink, and Ethernet for AI solutions.",
                    "supporting_snippet": "Networking revenue grew 13% sequentially driven by XDR InfiniBand products, NVLink, and Ethernet for AI solutions.",
                }
            ],
        },
    }

    result = check_answer_contract(answer, state)

    assert not any(item.code == "company_specific_token_leakage" for item in result.violations)


def test_numeric_grounding_accepts_scaled_display_units():
    trace = {
        "final_answer": "资本开支为 $151.00B，收入为 1815.19亿，P/E 为 32.31x，FCF yield 为 -0.08%。[N1][N2][N3][N4]",
        "output": {
            "numeric_evidence": [
                {"evidence_id": "N1", "ticker": "AMZN", "metric": "capital_expenditure", "value": 151_003_000_000, "unit": "USD"},
                {"evidence_id": "N2", "ticker": "AMZN", "metric": "revenue", "value": 181_519_000_000, "unit": "USD"},
                {"evidence_id": "N3", "ticker": "AMZN", "metric": "pe_ratio", "value": 32.31, "unit": "ratio"},
                {"evidence_id": "N4", "ticker": "AMZN", "metric": "fcf_yield", "value": -0.0008, "unit": "ratio"},
            ]
        },
    }

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is True
    assert result["metrics"]["numeric_grounding_rate"] == 1.0


def test_unsupported_numeric_fails():
    trace = _base_trace()
    trace["final_answer"] = "市值为 $9.99T。[N1]"

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert any(item["type"] == "unsupported_numeric" for item in result["violations"])


def test_missing_citation_fails():
    trace = _base_trace()
    trace["final_answer"] = "市值为 $4.31T。[N9]"

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert any(item["type"] == "invalid_citation" for item in result["violations"])


def test_missing_dimension_positive_claim_fails():
    trace = _base_trace()
    trace["final_answer"] = "NVDA 现金流强。"
    trace["evidence_packet"]["dimension_status_map"] = {"cash_flow_quality": {"status": "missing"}}

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert any(item["type"] == "dimension_status_violation" for item in result["violations"])


def test_evidence_summary_scope_overclaim_is_audit_warning_not_answer_violation():
    trace = _base_trace()
    trace["final_answer"] = "分部层面显示，Compute & Networking 增长与 AI 平台转型相关，不能完整代表总公司营收增长原因。[T1]"
    trace["evidence_packet"]["text_snippets"] = [
        {
            "evidence_id": "T1",
            "ticker": "NVDA",
            "claim": "NVIDIA的营收增长主要由加速计算和人工智能的平台转型驱动",
            "supporting_snippet": (
                "Compute & Networking revenue - The year over year increase was driven by "
                "platform shifts to accelerated computing and AI."
            ),
            "claim_scope": "segment",
            "allowed_claim_strength": "bounded_inference",
        }
    ]

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is True
    assert result["warnings"][0]["type"] == "evidence_summary_scope_overclaim"
    assert result["scope_overclaim_check"]["status"] == "passed"
    assert result["scope_overclaim_check"]["evidence_summary_warnings"][0]["code"] == "evidence_summary_scope_overclaim"


def test_partial_dimension_requires_limited_wording():
    trace = _base_trace()
    trace["final_answer"] = "NVDA 现金流质量很好。"
    trace["evidence_packet"]["dimension_status_map"] = {
        "cash_flow_quality": {"status": "partial", "supporting_evidence_ids": ["N1"]}
    }

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert any(item["type"] == "dimension_status_violation" for item in result["violations"])


def test_forbidden_valuation_advice_fails():
    trace = _base_trace()
    trace["final_answer"] = "NVDA 估值便宜，应该买。"

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert any(item["type"] == "forbidden_claim" for item in result["violations"])


def test_specific_caveat_warning_for_medium_confidence_or_enhanced_gap():
    trace = _base_trace()
    trace["final_answer"] = "债务/权益约 7.02%。[N3]"
    trace["evidence_packet"]["numeric_table"][2]["source_provider"] = "yfinance"
    trace["evidence_packet"]["numeric_table"][2]["confidence"] = "medium"
    trace["evidence_packet"]["dimension_status_map"]["balance_sheet_and_capital_intensity"]["enhanced_missing"] = [
        "capex_to_revenue"
    ]

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert any(
        item["type"] in {"missing_medium_confidence_source_caveat", "missing_growth_quantification_caveat"}
        for item in result["violations"]
    )


def test_internal_code_leakage_fails():
    trace = deepcopy(_base_trace())
    trace["final_answer"] = "REQ-NUM-NVDA dependency_numeric_requirement_missing"

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert result["metrics"]["raw_internal_leakage_count"] > 0


def test_contract_validation_is_post_hoc_and_does_not_mutate_trace():
    trace = deepcopy(_base_trace())
    trace["output"] = {"summary": "original summary", "view": {"kind": "methodology_single_company_brief"}}
    trace["synthesis"] = {"short_answer": "original synthesis"}
    trace["final_answer"] = "市值为 $9.99T。[N9]"
    before = deepcopy(trace)

    result = check_answer_evidence_contract(trace)

    assert {"passed", "violations", "metrics"} <= set(result)
    assert result["passed"] is False
    assert trace == before
    assert trace["final_answer"] == before["final_answer"]
    assert trace["output"] == before["output"]
    assert trace["synthesis"] == before["synthesis"]
    assert trace["evidence_packet"] == before["evidence_packet"]
