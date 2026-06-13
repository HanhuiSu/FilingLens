"""Structured company-analysis report assembly tests."""

from __future__ import annotations

from src.agent.reporting import build_company_analysis_report, should_build_company_report


def _report_state(query: str = "分析 NVIDIA") -> dict:
    dimension_status = {
        "business_model": {"status": "satisfied", "supporting_evidence_ids": ["T1"]},
        "revenue_quality": {"status": "satisfied", "supporting_evidence_ids": ["N1"]},
        "profitability_quality": {"status": "partial", "supporting_evidence_ids": ["N2"], "enhanced_missing": ["gross_margin"]},
        "cash_flow_quality": {"status": "missing", "required_missing": ["free_cash_flow"]},
        "balance_sheet_and_capital_intensity": {"status": "satisfied", "supporting_evidence_ids": ["N3"]},
        "moat_and_competitive_risk": {"status": "satisfied", "supporting_evidence_ids": ["T2"]},
        "valuation_and_risk_boundary": {"status": "partial", "supporting_evidence_ids": ["N4"]},
    }
    return {
        "user_query": query,
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "methodology_intent": "overview",
        "companies": ["NVDA"],
        "final_answer": "NVDA 的基本面分析应基于收入、盈利、现金流、资产负债、风险和估值边界综合判断。",
        "resolved_period_context": {"label": "latest available filings"},
        "dimension_status_by_id": dimension_status,
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": 100.0, "unit": "USD"},
                {"evidence_id": "N2", "ticker": "NVDA", "metric": "net_income", "value": 20.0, "unit": "USD"},
                {"evidence_id": "N3", "ticker": "NVDA", "metric": "total_debt", "value": 5.0, "unit": "USD"},
                {"evidence_id": "N4", "ticker": "NVDA", "metric": "market_cap", "value": 200.0, "unit": "USD"},
            ],
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "NVDA", "dimension_id": "business_model"},
                {"evidence_id": "T2", "ticker": "NVDA", "dimension_id": "moat_and_competitive_risk"},
            ],
            "dimension_status_map": dimension_status,
        },
        "synthesis": {
            "short_answer": "NVDA 的基本面需要综合多维度证据判断。",
            "methodology_answer": {
                "analysis_scope": "single_company",
                "dimension_sections": [
                    {"dimension_id": "business_model", "summary": "业务概览由 filing 文本支持。", "evidence_refs": ["T1"]},
                    {"dimension_id": "revenue_quality", "summary": "收入质量有结构化收入证据支持。", "evidence_refs": ["N1"]},
                    {"dimension_id": "profitability_quality", "summary": "盈利质量有部分证据支持。", "evidence_refs": ["N2"]},
                    {"dimension_id": "balance_sheet_and_capital_intensity", "summary": "资产负债表安全性有债务证据支持。", "evidence_refs": ["N3"]},
                    {"dimension_id": "moat_and_competitive_risk", "summary": "风险因素由 filing 文本支持。", "evidence_refs": ["T2"]},
                    {"dimension_id": "valuation_and_risk_boundary", "summary": "估值只能作为边界指标，不提供目标价。", "evidence_refs": ["N4"]},
                ],
                "limitations": ["现金流证据不足。"],
            },
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }


def test_open_company_analysis_blocks_failed_report_draft():
    state = _report_state()

    report = build_company_analysis_report(state)

    assert report["contract_status"] == "passed"
    assert report["ticker"] == "NVDA"
    assert report["sections"]
    assert report["markdown"]
    assert report["overall_limitations"]


def test_narrow_analysis_does_not_build_full_report():
    state = _report_state("NVIDIA 现金流质量怎么样")

    assert should_build_company_report(state) is False
    assert build_company_analysis_report(state) == {}


def test_report_section_citations_stay_inside_section_refs():
    report = build_company_analysis_report(_report_state())

    assert report["contract_status"] == "passed"
    assert report["sections"]
    for section in report["sections"]:
        assert set(section["citations"]).issubset(set(section["key_evidence_ids"]))
    assert report["markdown"]


def test_report_section_builder_keeps_citations_inside_section_refs_when_validated():
    report = build_company_analysis_report({**_report_state(), "final_answer": "NVDA 的收入为 100 USD。[N1]"})

    if report["contract_status"] != "passed":
        return
    for section in report["sections"]:
        assert set(section["citations"]).issubset(set(section["key_evidence_ids"]))
    assert report["sections"][2]["citations"] == ["N1"]
    assert report["sections"][6]["citations"] == ["T2"]
