"""Unit tests for evidence construction and claim validation boundaries."""

from __future__ import annotations

from src.agent.evidence import (
    _build_deterministic_numeric_claims,
    _period_consistency_ok,
    build_evidence_bundle,
    collect_event_rows,
    collect_financial_rows,
    validate_claims,
    validate_numeric_claims_strict,
    validate_text_claims_enhanced,
)


def _financial_tool_result() -> dict:
    return {
        "tool": "query_financial_data",
        "ticker": "AAPL",
        "data": {
            "ticker": "AAPL",
            "financial_facts": [
                {
                    "requirement_id": "REQ-NUM-AAPL-FACT",
                    "ticker": "AAPL",
                    "metric": "revenue",
                    "period_type": "quarterly",
                    "period_end": "2025-09-27",
                    "value": 100.0,
                    "unit": "USD",
                }
            ],
        },
    }


def test_build_evidence_bundle_assigns_stable_numeric_and_text_ids():
    bundle = build_evidence_bundle(
        {
            "tool_results": [_financial_tool_result()],
            "retrieved_docs": [
                {
                    "ticker": "AAPL",
                    "filing_id": "f1",
                    "form_type": "10-Q",
                    "fiscal_period": "2025Q3",
                    "section": "ITEM_2",
                    "chunk_order": 1,
                    "text": "Revenue increased due to strong product demand.",
                    "requirement_id": "REQ-TEXT-AAPL-MDA",
                },
                {
                    "ticker": "AAPL",
                    "filing_id": "f1",
                    "form_type": "10-Q",
                    "fiscal_period": "2025Q3",
                    "section": "ITEM_2",
                    "chunk_order": 1,
                    "text": "Duplicate chunk should be deduped.",
                },
            ],
        }
    )

    assert bundle.numeric_evidence[0]["evidence_id"] == "N1"
    assert bundle.numeric_evidence[0]["requirement_id"] == "REQ-NUM-AAPL-FACT"
    assert bundle.text_evidence[0]["evidence_id"] == "T1"
    assert bundle.text_evidence[0]["requirement_id"] == "REQ-TEXT-AAPL-MDA"
    assert len(bundle.text_evidence) == 1


def test_risk_text_only_comparison_skips_numeric_period_gate_when_citable_text_exists():
    ok, reason = _period_consistency_ok(
        {
            "task_type": "company_comparison",
            "user_query": "AMZN 和 NVDA 哪个更危险？",
            "requested_dimensions": ["moat_and_competitive_risk"],
            "primary_dimension": "moat_and_competitive_risk",
            "period_query": {"period_type": "latest", "comparison_basis": "same_period"},
            "resolved_period_context": {"target_period_type": "annual"},
            "text_citations": [{"evidence_id": "T1"}],
        },
        numeric_citations=[],
    )

    assert ok is True
    assert reason is None


def test_profit_decline_fact_claims_keep_recent_income_history():
    bundle = build_evidence_bundle(
        {
            "tool_results": [
                {
                    "tool": "query_financial_data",
                    "ticker": "AMZN",
                    "data": {
                        "ticker": "AMZN",
                        "financial_facts": [
                            {
                                "requirement_id": "REQ-PROFIT-HISTORY",
                                "ticker": "AMZN",
                                "metric": "net_income",
                                "period_type": "quarterly",
                                "period_end": "2026-03-31",
                                "value": 20.0,
                                "unit": "USD",
                            },
                            {
                                "requirement_id": "REQ-PROFIT-HISTORY",
                                "ticker": "AMZN",
                                "metric": "net_income",
                                "period_type": "quarterly",
                                "period_end": "2025-12-31",
                                "value": 10.0,
                                "unit": "USD",
                            },
                            {
                                "requirement_id": "REQ-REVENUE",
                                "ticker": "AMZN",
                                "metric": "revenue",
                                "period_type": "quarterly",
                                "period_end": "2026-03-31",
                                "value": 100.0,
                                "unit": "USD",
                            },
                        ],
                    },
                }
            ]
        }
    )

    claims = _build_deterministic_numeric_claims(
        {
            "task_type": "fact_qa",
            "user_query": "为什么amazon的利润下降了",
            "requested_metrics": ["revenue"],
            "period_query": {"period_type": "latest"},
            "resolved_period_context": {},
        },
        list(bundle.numeric_evidence),
        "zh",
    )

    assert [claim["metric"] for claim in claims] == ["net_income", "net_income"]
    assert {claim["period_end"] for claim in claims} == {"2026-03-31", "2025-12-31"}


def test_text_evidence_preserves_supporting_snippet_and_score_breakdown():
    bundle = build_evidence_bundle(
        {
            "tool_results": [],
            "retrieved_docs": [
                {
                    "ticker": "AAPL",
                    "filing_id": "f1",
                    "form_type": "10-K",
                    "fiscal_period": "2025",
                    "section": "ITEM_1A",
                    "chunk_order": 7,
                    "text": "Generic text. Competition is a material risk factor.",
                    "supporting_snippet": "Competition is a material risk factor.",
                    "supporting_terms": ["competition", "risk"],
                    "score_breakdown": {"lexical_bonus": 0.08, "final_score": 0.91},
                }
            ],
        }
    )

    text = bundle.text_evidence[0]
    assert text["text_snippet"] == "Competition is a material risk factor."
    assert text["supporting_snippet"] == "Competition is a material risk factor."
    assert text["supporting_terms"] == ["competition", "risk"]
    assert text["score_breakdown"]["lexical_bonus"] == 0.08


def test_validate_text_claim_rejects_unsupported_snippet():
    valid, unsupported = validate_claims(
        [{"sentence": "Management highlighted cybersecurity risks.", "evidence_ids": ["T1"]}],
        claim_type="text",
        evidence_map={
            "T1": {
                "evidence_id": "T1",
                "text_snippet": "Revenue increased because services demand was strong.",
            }
        },
    )

    assert valid == []
    assert unsupported[0]["reason"] == "claim_not_supported_by_snippet"


def test_text_claim_validation_explains_failure_reason():
    valid, unsupported, warnings = validate_text_claims_enhanced(
        [
            {
                "claim": "Amazon faces cybersecurity risks.",
                "company": "AMZN",
                "citation_ref": "T1",
                "supporting_quote": "Revenue increased because services demand was strong.",
            },
            {
                "claim": "Apple faces a completely unrelated cybersecurity issue.",
                "company": "AAPL",
                "citation_ref": "BAD",
                "supporting_quote": "Revenue increased because services demand was strong.",
            },
        ],
        {
            "T1": {
                "evidence_id": "T1",
                "requirement_id": "REQ-TEXT-AAPL-MDA",
                "ticker": "AAPL",
                "section": "ITEM_7",
                "supporting_snippet": "Revenue increased because services demand was strong.",
                "supporting_terms": ["revenue", "services", "demand"],
            }
        },
    )

    assert valid == []
    assert warnings == []
    assert unsupported[0]["validation_error_code"] == "claim_uses_wrong_company"
    assert unsupported[0]["source_requirement_id"] == "REQ-TEXT-AAPL-MDA"
    assert unsupported[0]["source_snippet_id"] == "T1"
    assert unsupported[0]["citation_ref"] == "T1"
    assert unsupported[0]["claim_entities"]
    assert unsupported[0]["snippet_entities"]
    assert "cybersecurity" in unsupported[0]["missing_support_terms"]
    assert unsupported[1]["validation_error_code"] == "invalid_citation_ref"


def test_text_claim_generation_requires_citation_ref():
    valid, unsupported, _warnings = validate_text_claims_enhanced(
        [
            {
                "claim": "AAPL faces competition pressure.",
                "company": "AAPL",
                "supporting_quote": "Competition creates pressure.",
            }
        ],
        {
            "T1": {
                "evidence_id": "T1",
                "ticker": "AAPL",
                "section": "ITEM_1A",
                "supporting_snippet": "Competition creates pressure.",
            }
        },
    )

    assert valid == []
    assert unsupported[0]["validation_error_code"] == "missing_citation_ref"


def test_supported_text_claim_passes_with_soft_warning():
    valid, unsupported, warnings = validate_text_claims_enhanced(
        [
            {
                "claim": "AAPL has business risks.",
                "company": "AAPL",
                "claim_type": "risk_factor",
                "citation_ref": "T1",
                "supporting_quote": "Competition is a material risk factor.",
            }
        ],
        {
            "T1": {
                "evidence_id": "T1",
                "requirement_id": "REQ-TEXT-AAPL-RISK",
                "ticker": "AAPL",
                "section": "ITEM_1A",
                "supporting_snippet": "Competition is a material risk factor.",
                "supporting_terms": ["competition", "risk"],
            }
        },
    )

    assert unsupported == []
    assert valid[0]["evidence_ids"] == ["T1"]
    assert valid[0]["company"] == "AAPL"
    assert warnings[0]["validation_error_code"] == "claim_too_generic"


def test_single_company_text_claim_validation_allows_business_description():
    valid, unsupported, _warnings = validate_text_claims_enhanced(
        [
            {
                "claim": "NVDA 的业务主要围绕 GPU、数据中心和游戏等产品和市场展开。",
                "company": "NVDA",
                "citation_ref": "T1",
                "supporting_quote": "NVIDIA sells GPUs for data center, gaming, professional visualization and automotive markets.",
                "claim_type": "business_description",
            }
        ],
        {
            "T1": {
                "evidence_id": "T1",
                "requirement_id": "REQ-TEXT-NVDA-BUSINESS_MODEL",
                "dimension_id": "business_model",
                "ticker": "NVDA",
                "section": "ITEM_1",
                "supporting_snippet": "NVIDIA sells GPUs for data center, gaming, professional visualization and automotive markets.",
                "supporting_terms": ["gpu", "data", "center", "gaming", "automotive"],
            }
        },
        validation_context={
            "analysis_scope": "single_company",
            "requirement_dimension_map": {"REQ-TEXT-NVDA-BUSINESS_MODEL": "business_model"},
        },
    )

    assert unsupported == []
    assert valid[0]["dimension_id"] == "business_model"
    assert valid[0]["source_requirement_id"] == "REQ-TEXT-NVDA-BUSINESS_MODEL"


def test_single_company_text_claim_validation_allows_risk_summary():
    valid, unsupported, _warnings = validate_text_claims_enhanced(
        [
            {
                "claim": "NVDA 面临竞争、需求波动和供应链相关风险。",
                "company": "NVDA",
                "citation_ref": "T1",
                "supporting_quote": "Risk factors include intense competition, changes in demand, and supply chain constraints.",
                "claim_type": "competition_risk",
            }
        ],
        {
            "T1": {
                "evidence_id": "T1",
                "requirement_id": "REQ-TEXT-NVDA-RISK",
                "dimension_id": "moat_and_competitive_risk",
                "ticker": "NVDA",
                "section": "ITEM_1A",
                "supporting_snippet": "Risk factors include intense competition, changes in demand, and supply chain constraints.",
                "supporting_terms": ["competition", "demand", "supply", "chain"],
            }
        },
        validation_context={
            "analysis_scope": "single_company",
            "requirement_dimension_map": {"REQ-TEXT-NVDA-RISK": "moat_and_competitive_risk"},
        },
    )

    assert unsupported == []
    assert valid[0]["dimension_id"] == "moat_and_competitive_risk"


def test_single_company_text_claim_validation_preserves_hard_failures():
    valid, unsupported, _warnings = validate_text_claims_enhanced(
        [
            {
                "claim": "NVDA faces 20 regulatory investigations and investors should buy it.",
                "company": "NVDA",
                "citation_ref": "T1",
                "supporting_quote": "Competition and demand risks may affect the business.",
                "claim_type": "competition_risk",
            }
        ],
        {
            "T1": {
                "evidence_id": "T1",
                "requirement_id": "REQ-TEXT-NVDA-RISK",
                "dimension_id": "moat_and_competitive_risk",
                "ticker": "NVDA",
                "section": "ITEM_1A",
                "supporting_snippet": "Competition and demand risks may affect the business.",
            }
        },
        validation_context={
            "analysis_scope": "single_company",
            "requirement_dimension_map": {"REQ-TEXT-NVDA-RISK": "moat_and_competitive_risk"},
        },
    )

    assert valid == []
    assert unsupported[0]["validation_error_code"] == "claim_safety_violation"


def test_text_claim_validation_allows_seller_business_context():
    valid, unsupported, _warnings = validate_text_claims_enhanced(
        [
            {
                "claim": "AMZN 的中国卖家业务可能受到相关监管变化影响。",
                "company": "AMZN",
                "citation_ref": "T1",
                "supporting_quote": "中国卖家业务可能受到相关监管变化影响。",
                "claim_type": "regulatory_risk",
            }
        ],
        {
            "T1": {
                "evidence_id": "T1",
                "requirement_id": "REQ-TEXT-AMZN-RISK",
                "dimension_id": "moat_and_competitive_risk",
                "ticker": "AMZN",
                "section": "ITEM_1A",
                "supporting_snippet": "中国卖家业务可能受到相关监管变化影响。",
                "supporting_terms": ["regulation"],
            }
        },
        validation_context={
            "analysis_scope": "single_company",
            "requirement_dimension_map": {"REQ-TEXT-AMZN-RISK": "moat_and_competitive_risk"},
        },
    )

    assert unsupported == []
    assert valid[0]["citation_ref"] == "T1"


def test_unsupported_number_claim_fails():
    valid, unsupported, _warnings = validate_text_claims_enhanced(
        [
            {
                "claim": "AAPL faces 20 regulatory investigations.",
                "company": "AAPL",
                "citation_ref": "T1",
                "supporting_quote": "Regulatory investigations may affect the business.",
            }
        ],
        {
            "T1": {
                "evidence_id": "T1",
                "ticker": "AAPL",
                "section": "ITEM_1A",
                "supporting_snippet": "Regulatory investigations may affect the business.",
            }
        },
    )

    assert valid == []
    assert unsupported[0]["validation_error_code"] == "claim_contains_unsupported_number"


def test_validate_numeric_claims_rejects_wrong_period():
    numeric_map = {
        "N1": {
            "evidence_id": "N1",
            "ticker": "AAPL",
            "metric": "revenue",
            "period_type": "quarterly",
            "period_end": "2025-09-27",
            "value": 100.0,
            "unit": "USD",
        }
    }
    valid, unsupported = validate_numeric_claims_strict(
        [
            {
                "sentence": "AAPL revenue was 100.0 in 2024 Q3.",
                "evidence_ids": ["N1"],
                "period_end": "2024-09-28",
                "value": 100.0,
                "unit": "USD",
            }
        ],
        numeric_map,
    )

    assert valid == []
    assert unsupported[0]["reason"] == "numeric_claim_period_mismatch"


def test_collect_rows_preserves_structured_and_event_rows():
    financial = _financial_tool_result()
    event = {
        "tool": "query_event_price_window",
        "ticker": "AAPL",
        "data": {"ticker": "AAPL", "events": [{"event_date": "2025-01-31", "returns": {"return_1d": 0.01}}]},
    }

    assert collect_financial_rows([financial])[0]["metric"] == "revenue"
    assert collect_event_rows([event])[0]["ticker"] == "AAPL"
