"""Focused tests for bilingual output-language contracts."""

from __future__ import annotations

from src.agent.answer_relevance import bounded_causal_fallback_answer, judge_answer_relevance
from src.agent.answering import (
    _bounded_fcf_answer_if_needed,
    _bounded_scenario_risk_answer_if_needed,
    _bounded_valuation_comparison_answer_if_needed,
)
from src.agent.output_language import (
    EN_FORBIDDEN_CHINESE_TERMS,
    detect_output_language,
    display_theme,
    language_leakage_terms,
    repair_language_leakage,
)
from src.agent.rendering import render_risk_focused_analysis_brief


def _assert_no_english_leakage(text: str) -> None:
    leaked = language_leakage_terms(text, "en")
    assert leaked == []
    assert all(term not in text for term in EN_FORBIDDEN_CHINESE_TERMS)


def test_output_language_detection_explicit_and_default_cases():
    assert detect_output_language("What is driving NVIDIA's data center revenue growth?") == "en"
    assert detect_output_language("为什么 NVIDIA 的营收增长这么多？") == "zh"
    assert detect_output_language("Answer in English: 为什么 NVIDIA 的营收增长这么多？") == "en"
    assert detect_output_language("What is driving NVIDIA growth? 用中文回答") == "zh"
    assert detect_output_language("用中文回答, then in English") == "en"
    assert detect_output_language("in English, then 用中文") == "zh"
    assert detect_output_language("AAPL overview") == "zh"


def test_language_leakage_repair_and_relevance_gate_for_english():
    raw = "结论\nAAPL 和 NVDA's valuation risk.\n证据边界\n不能给买卖建议。"
    repaired = repair_language_leakage(raw, "en")

    assert "Conclusion" in repaired
    assert "AAPL and NVDA's" in repaired
    _assert_no_english_leakage(repaired)

    decision = judge_answer_relevance(raw, {"user_query": "Compare Apple and NVIDIA valuation risk.", "output_language": "en"})
    assert decision.route == "repair_answer"
    assert any(item["code"] == "language_leakage" for item in decision.deterministic_relevance_failures)


def test_risk_theme_display_is_language_localized():
    assert display_theme("product_demand_uncertainty", "en") == "new-product and demand uncertainty"
    assert display_theme("product_demand_uncertainty", "zh") == "新产品和需求不确定性"
    assert display_theme("fulfillment_inventory_capex_pressure", "en") == "fulfillment, inventory, and capex pressure"
    assert display_theme("fulfillment_inventory_capex_pressure", "zh") == "履约/库存/资本开支压力"


def test_english_risk_renderer_uses_theme_key_not_chinese_theme_name():
    rendered = render_risk_focused_analysis_brief(
        {
            "direct_judgment": "The main disclosed risk is demand uncertainty.",
            "top_risk": {
                "theme_key": "product_demand_uncertainty",
                "theme_name": "新产品和需求不确定性",
                "why_it_matters": "Validated risk text directly supports demand uncertainty.",
                "evidence_refs": ["T1"],
            },
            "risk_ranking": [
                {
                    "theme_key": "product_demand_uncertainty",
                    "theme_name": "新产品和需求不确定性",
                    "why_it_matters": "Validated risk text directly supports demand uncertainty.",
                    "evidence_refs": ["T1"],
                    "mechanism_support_level": "validated_text",
                }
            ],
            "filing_evidence": [
                {
                    "evidence_id": "T1",
                    "theme_key": "product_demand_uncertainty",
                    "theme_name": "新产品和需求不确定性",
                    "supporting_snippet": "Demand uncertainty may affect revenue.",
                    "mechanism_support_level": "validated_text",
                }
            ],
            "analysis_boundary": "Risk text does not quantify probability.",
        },
        lang="en",
    )

    assert "Risk Judgment" in rendered
    assert "new-product and demand uncertainty" in rendered
    _assert_no_english_leakage(rendered)


def test_english_valuation_comparison_renders_per_metric_judgments():
    rows = [
        {"evidence_id": "N1", "ticker": "AAPL", "metric": "pe_ratio", "value": 30.0, "display_value": "30.0x"},
        {"evidence_id": "N2", "ticker": "NVDA", "metric": "pe_ratio", "value": 60.0, "display_value": "60.0x"},
        {"evidence_id": "N3", "ticker": "AAPL", "metric": "ps_ratio", "value": 7.0, "display_value": "7.0x"},
        {"evidence_id": "N4", "ticker": "NVDA", "metric": "ps_ratio", "value": 25.0, "display_value": "25.0x"},
        {"evidence_id": "N5", "ticker": "AAPL", "metric": "fcf_yield", "value": 0.03, "display_value": "3.0%"},
        {"evidence_id": "N6", "ticker": "NVDA", "metric": "fcf_yield", "value": 0.01, "display_value": "1.0%"},
    ]
    answer = _bounded_valuation_comparison_answer_if_needed(
        "blocked",
        user_query="Compare Apple and NVIDIA valuation risk.",
        state={"task_type": "company_comparison", "companies": ["AAPL", "NVDA"]},
        numeric_evidence=rows,
        lang="en",
    )

    assert "Conclusion" in answer
    assert "Metric-by-Metric Comparison" in answer
    assert "- P/E:" in answer
    assert "- P/S:" in answer
    assert "- FCF yield:" in answer
    assert "AAPL 和 NVDA" not in answer
    _assert_no_english_leakage(answer)


def test_english_scenario_risk_includes_financial_transmission_path_without_chinese_labels():
    answer = _bounded_scenario_risk_answer_if_needed(
        "blocked",
        user_query="What is Microsoft's biggest financial risk if the economy slows next quarter?",
        text_evidence=[
            {
                "evidence_id": "T1",
                "ticker": "MSFT",
                "claim": "Macroeconomic conditions and customer spending may affect demand and revenue.",
            }
        ],
        lang="en",
    )

    assert "Risk Judgment" in answer
    assert "Verified Risk Text" in answer
    assert "Business-Model Inference" in answer
    assert "Financial Transmission Path" in answer
    assert "revenue" in answer and "margin" in answer and "FCF" in answer
    _assert_no_english_leakage(answer)


def test_english_cash_flow_pressure_answer_preserves_causal_path():
    rows = [
        {"evidence_id": "N1", "ticker": "AMZN", "metric": "operating_cash_flow", "value": 148_531_000_000, "display_value": "$148.53B"},
        {"evidence_id": "N2", "ticker": "AMZN", "metric": "free_cash_flow", "value": -2_472_000_000, "display_value": "$-2.47B"},
        {"evidence_id": "N3", "ticker": "AMZN", "metric": "capital_expenditure", "value": -151_003_000_000, "display_value": "$151.00B"},
    ]
    answer = _bounded_fcf_answer_if_needed(
        "blocked",
        user_query="Why is Amazon's free cash flow under pressure?",
        numeric_evidence=rows,
        lang="en",
    )

    assert "free cash flow" in answer.lower()
    assert "operating cash flow" in answer.lower()
    assert "capex" in answer.lower()
    assert "Conclusion" in answer
    assert "Verified Facts" in answer
    assert "Reasonable Inference" in answer
    assert "Data to Verify" in answer
    assert "Evidence Boundary" in answer
    _assert_no_english_leakage(answer)


def test_chinese_bounded_causal_fallback_keeps_chinese_labels():
    answer = bounded_causal_fallback_answer(
        {
            "user_query": "为什么 NVIDIA 的营收增长这么多？",
            "output_language": "zh",
            "companies": ["NVDA"],
            "numeric_evidence": [{"evidence_id": "N1", "metric": "revenue"}],
            "missing_but_analyzable_answer_parts": ["identify_growth_drivers"],
        }
    )

    assert "结论" in answer
    assert "已验证事实" in answer
    assert "合理推断" in answer
    assert "待验证数据" in answer
    assert "证据边界" in answer
    assert "Conclusion" not in answer
