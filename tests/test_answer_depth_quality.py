"""Answer depth quality checks for phase-one synthesis upgrade."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.agent.answer_contract import check_answer_contract
from src.agent.analyst_loop import run_analyst_draft_loop
from src.agent.evidence import validate_text_claims_enhanced
from src.agent.synthesis import build_analytical_synthesis, build_methodology_answer, render_synthesis_text


ROOT = Path(__file__).resolve().parents[1]


def _contract_status(answer: str, packet: dict, *, answer_mode: str = "analytical") -> str:
    result = check_answer_contract(
        answer,
        {
            "user_query": packet.get("user_query", "分析公司"),
            "draft_answer": answer,
            "final_answer": answer,
            "task_type": packet.get("task_type", "report_summary"),
            "answer_mode": answer_mode,
            "analysis_scope": packet.get("analysis_scope", "single_company"),
            "evidence_packet": packet,
            "output": {"view": {"kind": "methodology_single_company_brief"}},
        },
    )
    if result.route == "pass" and result.decision == "warning":
        return "passed_with_warnings"
    if result.route == "pass":
        return "passed"
    if result.route == "repair_answer":
        return "repairable"
    return "blocked"


def _amzn_risk_packet() -> dict:
    return {
        "user_query": "你分析一下下一个季度亚马逊的风险有什么？",
        "task_type": "report_summary",
        "answer_mode": "risk_focused_analysis",
        "analysis_scope": "single_company",
        "active_dimensions": ["moat_and_competitive_risk", "revenue_quality", "profitability_quality", "cash_flow_quality"],
        "dimension_status_map": {
            "moat_and_competitive_risk": {"status": "satisfied", "supporting_evidence_ids": ["T1", "T2", "T3"]},
            "revenue_quality": {"status": "satisfied", "supporting_evidence_ids": ["N1", "N15", "N16"]},
            "profitability_quality": {"status": "satisfied", "supporting_evidence_ids": ["N2", "N3", "N17", "N18"]},
            "cash_flow_quality": {"status": "partial", "supporting_evidence_ids": ["N4", "N5"]},
        },
        "numeric_table": [
            {"evidence_id": "N1", "ticker": "AMZN", "metric": "revenue", "value": 574_785_000_000, "unit": "USD", "display_value": "$574.79B"},
            {"evidence_id": "N2", "ticker": "AMZN", "metric": "net_income", "value": 30_425_000_000, "unit": "USD", "display_value": "$30.43B"},
            {"evidence_id": "N3", "ticker": "AMZN", "metric": "net_margin", "value": 0.0529, "unit": "ratio", "display_value": "5.29%"},
            {"evidence_id": "N4", "ticker": "AMZN", "metric": "operating_cash_flow", "value": 84_946_000_000, "unit": "USD", "display_value": "$84.95B"},
            {"evidence_id": "N5", "ticker": "AMZN", "metric": "free_cash_flow", "value": 36_800_000_000, "unit": "USD", "display_value": "$36.80B"},
        ],
        "text_snippets": [
            {
                "evidence_id": "T1",
                "ticker": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "claim": "Amazon faces supply chain, inventory, fulfillment and logistics risks that may affect revenue, costs and margins.",
                "supporting_snippet": "Supply chain, inventory, fulfillment and logistics risks may affect revenue, costs and margins.",
            },
            {
                "evidence_id": "T2",
                "ticker": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "claim": "Amazon faces competition across retail, third-party seller services, cloud and advertising markets.",
                "supporting_snippet": "Competition across retail, third-party seller services, cloud and advertising markets can affect pricing and market share.",
            },
            {
                "evidence_id": "T3",
                "ticker": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "claim": "Amazon faces regulatory, legal and compliance risks across jurisdictions.",
                "supporting_snippet": "Regulatory, legal and compliance risks across jurisdictions may increase uncertainty.",
            },
        ],
    }


def _nvda_composite_packet() -> dict:
    return {
        "user_query": "分析 NVIDIA 的现金流质量、估值边界和主要风险。",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "safety_intent": "normal",
        "analysis_scope": "single_company",
        "selected_framework": {"id": "fundamental_quality_analysis"},
        "active_dimensions": ["cash_flow_quality", "valuation_and_risk_boundary", "moat_and_competitive_risk"],
        "dimension_status_map": {
            "cash_flow_quality": {"status": "satisfied", "supporting_evidence_ids": ["C1", "C2", "C3", "C4", "C5"]},
            "valuation_and_risk_boundary": {"status": "satisfied", "supporting_evidence_ids": ["V1", "V2", "V3", "V4", "V5"]},
            "moat_and_competitive_risk": {"status": "satisfied", "supporting_evidence_ids": ["T1", "T2", "T3"]},
        },
        "dimension_summary": [
            {"dimension_id": "cash_flow_quality", "status": "satisfied", "numeric_evidence_refs": ["C1", "C2", "C3", "C4", "C5"], "evidence_refs": ["C1", "C2", "C3", "C4", "C5"]},
            {"dimension_id": "valuation_and_risk_boundary", "status": "satisfied", "numeric_evidence_refs": ["V1", "V2", "V3", "V4", "V5"], "evidence_refs": ["V1", "V2", "V3", "V4", "V5"]},
            {"dimension_id": "moat_and_competitive_risk", "status": "satisfied", "text_evidence_refs": ["T1", "T2", "T3"], "evidence_refs": ["T1", "T2", "T3"]},
        ],
        "numeric_table": [
            {"evidence_id": "C1", "ticker": "NVDA", "metric": "operating_cash_flow", "value": 36_188_000_000, "unit": "USD", "display_value": "$36.19B"},
            {"evidence_id": "C2", "ticker": "NVDA", "metric": "free_cash_flow", "value": 34_904_000_000, "unit": "USD", "display_value": "$34.90B"},
            {"evidence_id": "C3", "ticker": "NVDA", "metric": "capital_expenditure", "value": 1_284_000_000, "unit": "USD", "display_value": "$1.28B"},
            {"evidence_id": "C4", "ticker": "NVDA", "metric": "cfo_to_net_income", "value": 1.12, "unit": "ratio", "display_value": "112.00%"},
            {"evidence_id": "C5", "ticker": "NVDA", "metric": "fcf_margin", "value": 0.51, "unit": "ratio", "display_value": "51.00%"},
            {"evidence_id": "V1", "ticker": "NVDA", "metric": "adjusted_close", "value": 215.2, "unit": "USD", "display_value": "$215.20"},
            {"evidence_id": "V2", "ticker": "NVDA", "metric": "market_cap", "value": 5_200_000_000_000, "unit": "USD", "display_value": "$5.20T"},
            {"evidence_id": "V3", "ticker": "NVDA", "metric": "pe_ratio", "value": 68.4, "unit": "ratio", "display_value": "68.40x"},
            {"evidence_id": "V4", "ticker": "NVDA", "metric": "ps_ratio", "value": 35.2, "unit": "ratio", "display_value": "35.20x"},
            {"evidence_id": "V5", "ticker": "NVDA", "metric": "fcf_yield", "value": 0.0067, "unit": "ratio", "display_value": "0.67%"},
        ],
        "text_snippets": [
            {"evidence_id": "T1", "ticker": "NVDA", "dimension_id": "moat_and_competitive_risk", "section": "ITEM_1A", "supporting_snippet": "NVIDIA faces supply constraints and dependence on third-party manufacturers."},
            {"evidence_id": "T2", "ticker": "NVDA", "dimension_id": "moat_and_competitive_risk", "section": "ITEM_1A", "supporting_snippet": "NVIDIA faces intense competition in accelerated computing and GPU markets."},
            {"evidence_id": "T3", "ticker": "NVDA", "dimension_id": "moat_and_competitive_risk", "section": "ITEM_1A", "supporting_snippet": "Export controls and other regulatory restrictions may affect product sales."},
        ],
        "red_flags": [],
    }


def _amzn_overview_packet() -> dict:
    return {
        "user_query": "分析下amazon这家公司",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "methodology_intent": "single_company_overview",
        "intent_family": "overview",
        "evidence_policy_id": "single_company_composite_v1",
        "selected_framework": {"id": "fundamental_quality_analysis"},
        "active_dimensions": [
            "business_model",
            "revenue_quality",
            "profitability_quality",
            "cash_flow_quality",
            "balance_sheet_and_capital_intensity",
            "moat_and_competitive_risk",
            "valuation_and_risk_boundary",
        ],
        "dimension_status_map": {
            "business_model": {"status": "satisfied", "supporting_evidence_ids": ["T1"]},
            "revenue_quality": {"status": "satisfied", "supporting_evidence_ids": ["N1"]},
            "profitability_quality": {"status": "satisfied", "supporting_evidence_ids": ["N2", "N3"]},
            "cash_flow_quality": {"status": "satisfied", "supporting_evidence_ids": ["N4", "N5", "N6", "N7", "N8"]},
            "balance_sheet_and_capital_intensity": {"status": "satisfied", "supporting_evidence_ids": ["N6", "N9", "N10", "N11"]},
            "moat_and_competitive_risk": {"status": "satisfied", "supporting_evidence_ids": ["T2", "T3", "T4"]},
            "valuation_and_risk_boundary": {"status": "satisfied", "supporting_evidence_ids": ["N12", "N13", "N14"]},
        },
        "dimension_summary": [
            {"dimension_id": "business_model", "status": "satisfied", "text_evidence_refs": ["T1"], "evidence_refs": ["T1"]},
            {"dimension_id": "revenue_quality", "status": "satisfied", "numeric_evidence_refs": ["N1", "N15", "N16"], "evidence_refs": ["N1", "N15", "N16"]},
            {"dimension_id": "profitability_quality", "status": "satisfied", "numeric_evidence_refs": ["N2", "N3", "N17", "N18"], "evidence_refs": ["N2", "N3", "N17", "N18"]},
            {"dimension_id": "cash_flow_quality", "status": "satisfied", "numeric_evidence_refs": ["N4", "N5", "N6", "N7", "N8"], "evidence_refs": ["N4", "N5", "N6", "N7", "N8"]},
            {"dimension_id": "balance_sheet_and_capital_intensity", "status": "satisfied", "numeric_evidence_refs": ["N6", "N9", "N10", "N11"], "evidence_refs": ["N6", "N9", "N10", "N11"]},
            {"dimension_id": "moat_and_competitive_risk", "status": "satisfied", "text_evidence_refs": ["T2", "T3", "T4"], "evidence_refs": ["T2", "T3", "T4"]},
            {"dimension_id": "valuation_and_risk_boundary", "status": "satisfied", "numeric_evidence_refs": ["N12", "N13", "N14"], "evidence_refs": ["N12", "N13", "N14"]},
        ],
        "numeric_table": [
            {"evidence_id": "N1", "ticker": "AMZN", "metric": "revenue", "period_end": "2026-03-31", "period_type": "quarterly", "source_provider": "sec_companyfacts", "value": 181_400_000_000, "unit": "USD", "display_value": "$181.40B"},
            {"evidence_id": "N15", "ticker": "AMZN", "metric": "revenue", "period_end": "2025-12-31", "period_type": "quarterly", "source_provider": "yfinance", "value": 213_386_000_000, "unit": "USD", "display_value": "$213.39B"},
            {"evidence_id": "N16", "ticker": "AMZN", "metric": "revenue", "period_end": "2025-09-30", "period_type": "quarterly", "source_provider": "sec_companyfacts", "value": 503_538_000_000, "unit": "USD", "display_value": "$503.54B"},
            {"evidence_id": "N2", "ticker": "AMZN", "metric": "net_income", "period_end": "2026-03-31", "source_provider": "sec_companyfacts", "value": 90_798_000_000, "unit": "USD", "display_value": "$90.80B"},
            {"evidence_id": "N3", "ticker": "AMZN", "metric": "net_margin", "period_end": "2026-03-31", "source_provider": "sec_companyfacts", "value": 0.5002, "unit": "ratio", "display_value": "50.02%"},
            {"evidence_id": "N17", "ticker": "AMZN", "metric": "gross_margin", "period_end": "2026-03-31", "source_provider": "yfinance", "value": 0.5182, "unit": "ratio", "display_value": "51.82%"},
            {"evidence_id": "N18", "ticker": "AMZN", "metric": "operating_margin", "period_end": "2026-03-31", "source_provider": "sec_companyfacts", "value": 0.1314, "unit": "ratio", "display_value": "13.14%"},
            {"evidence_id": "N4", "ticker": "AMZN", "metric": "operating_cash_flow", "value": 148_531_000_000, "unit": "USD", "display_value": "$148.53B"},
            {"evidence_id": "N5", "ticker": "AMZN", "metric": "capital_expenditure", "value": 151_003_000_000, "unit": "USD", "display_value": "$151.00B"},
            {"evidence_id": "N6", "ticker": "AMZN", "metric": "free_cash_flow", "value": -2_472_000_000, "unit": "USD", "display_value": "$-2.47B"},
            {"evidence_id": "N7", "ticker": "AMZN", "metric": "fcf_margin", "value": -0.0136, "unit": "ratio", "display_value": "-1.36%"},
            {"evidence_id": "N8", "ticker": "AMZN", "metric": "cfo_to_net_income", "value": 1.64, "unit": "ratio", "display_value": "164.00%"},
            {"evidence_id": "N9", "ticker": "AMZN", "metric": "cash", "value": 100_000_000_000, "unit": "USD", "display_value": "$100.00B"},
            {"evidence_id": "N10", "ticker": "AMZN", "metric": "total_debt", "value": 70_000_000_000, "unit": "USD", "display_value": "$70.00B"},
            {"evidence_id": "N11", "ticker": "AMZN", "metric": "capex_to_revenue", "value": 0.8324, "unit": "ratio", "display_value": "83.24%"},
            {"evidence_id": "N12", "ticker": "AMZN", "metric": "market_cap", "value": 2_300_000_000_000, "unit": "USD", "display_value": "$2.30T"},
            {"evidence_id": "N13", "ticker": "AMZN", "metric": "pe_ratio", "value": 35.5, "unit": "ratio", "display_value": "35.50x"},
            {"evidence_id": "N14", "ticker": "AMZN", "metric": "ps_ratio", "value": 3.2, "unit": "ratio", "display_value": "3.20x"},
        ],
        "text_snippets": [
            {
                "evidence_id": "T1",
                "ticker": "AMZN",
                "dimension_id": "business_model",
                "section": "ITEM_1",
                "supporting_snippet": "Amazon operates North America, International, and AWS segments. Net sales include online stores, third-party seller services, advertising, subscription services including Prime, and Amazon Web Services.",
            },
            {
                "evidence_id": "T2",
                "ticker": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "supporting_snippet": "Fulfillment network staffing, logistics, and inventory risks may affect costs, service quality, and margins.",
            },
            {
                "evidence_id": "T3",
                "ticker": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "supporting_snippet": "Legal, regulatory, compliance, China, India, and other jurisdiction risks may affect operations.",
            },
            {
                "evidence_id": "T4",
                "ticker": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "supporting_snippet": "Competition across retail, AWS, advertising, and marketplace services may affect pricing, market share, and margins.",
            },
        ],
    }


def _render_overview_answer(packet: dict | None = None) -> str:
    methodology = build_methodology_answer(packet or _amzn_overview_packet(), lang="zh")
    assert methodology is not None
    return render_synthesis_text(
        {"methodology_answer": methodology.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )


def _section(answer: str, heading: str, next_heading: str) -> str:
    start = answer.index(heading)
    end = answer.index(next_heading, start)
    return answer[start:end]


def _cited_amzn_overview_draft() -> dict:
    return {
        "framework_summary": "单公司 overview 需要覆盖业务模式、收入、盈利、现金流、资本强度、估值和风险。",
        "dimension_analyses": [
            {"dimension_id": "business_model", "status": "satisfied", "claim": "AMZN 的业务结构可按 North America、International 和 AWS 三个分部理解。", "evidence_refs": ["T1"]},
            {"dimension_id": "revenue_quality", "status": "satisfied", "claim": "收入序列可用于观察规模，但混合来源限制可比趋势判断。", "evidence_refs": ["N1", "N15", "N16"]},
            {"dimension_id": "profitability_quality", "status": "satisfied", "claim": "净利率偏高需要口径核验，不能直接外推为可持续盈利能力。", "evidence_refs": ["N2", "N3"]},
            {"dimension_id": "cash_flow_quality", "status": "satisfied", "claim": "经营现金流为正，但资本开支压制自由现金流。", "evidence_refs": ["N4", "N5", "N6", "N7"]},
            {"dimension_id": "moat_and_competitive_risk", "status": "satisfied", "claim": "履约、库存、监管和竞争风险都需要按传导机制观察。", "evidence_refs": ["T2", "T3", "T4"]},
        ],
        "overall_judgment": "AMZN 可以形成公司层面的基本面轮廓，但异常利润率和现金流口径需要谨慎。",
        "methodology_counterpoints": ["混合来源和异常利润率使部分趋势判断只能作为方向性观察。"],
        "methodology_limitations": ["这不是投资建议，且不包含目标价。"],
        "follow_up_metrics": ["分部收入", "净利率口径", "经营现金流、资本开支与 FCF margin", "库存和竞争披露"],
        "tentative_conclusion": {
            "statement": "基于已验证证据，AMZN 可做 broad overview 分析，但现金流和盈利结论需要口径 caveat。[T1][N3][N4][N5][N6]",
            "stance": "evidence_bounded",
            "preferred_company": "AMZN",
            "citation_refs": [],
        },
        "decision_basis": [
            {"statement": "业务模式证据明确披露 North America、International 和 AWS 三个分部。", "citation_refs": ["T1"]},
            {"statement": "经营现金流为正而自由现金流为负，资本开支接近经营现金流。", "citation_refs": ["N4", "N5", "N6"]},
        ],
        "supporting_points": [
            {"statement": "毛利率和营业利润率之间的差距支持继续观察费用和运营成本对利润释放的影响。", "citation_refs": ["N17", "N18"]},
        ],
        "counterpoints": [
            {"statement": "收入多期序列混用了不同来源，因此不能把入库数值变化直接当作可比经营下滑。", "citation_refs": ["N1", "N15", "N16"]},
        ],
        "risk_tradeoffs": [
            {"statement": "履约与库存风险可能通过服务质量、履约成本、现金占用和 FCF 形成压力。", "citation_refs": ["T2"]},
            {"statement": "监管和竞争风险可能通过卖家生态、广告、AWS、价格和利润率影响收入与利润。", "citation_refs": ["T3", "T4"]},
        ],
        "uncertainty_notes": [
            {"statement": "50.02% 净利率需要核验期间口径或非经常性项目，不能无保留外推。", "citation_refs": ["N3"]},
            {"statement": "FCF margin 为负说明资本开支后自由现金流仍受压制。", "citation_refs": ["N5", "N6", "N7"]},
        ],
        "citation_refs": ["T1", "T2", "T3", "T4", "N1", "N3", "N4", "N5", "N6", "N7", "N15", "N16", "N17", "N18"],
        "safety_notes": [{"statement": "这不是投资建议。", "citation_refs": []}],
    }


def test_overview_analyst_draft_released(monkeypatch):
    packet = _amzn_overview_packet()

    def fake_generate_analyst_draft(**_kwargs):
        return _cited_amzn_overview_draft(), []

    monkeypatch.setattr("src.agent.analyst_loop.generate_analyst_draft", fake_generate_analyst_draft)

    result = run_analyst_draft_loop(
        evidence_packet=packet,
        answer_language="zh",
        synthesis_mode="methodology_single_company",
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        methodology_context={},
        max_attempts=1,
    )

    assert result["accepted_draft"]
    assert result["draft_final_status"] == "passed"
    validation = result["validation"]
    assert validation["passed"] is True
    reasons = {item["reason"] for item in validation.get("violations", [])}
    assert "risk_judgment_without_text_ref" not in reasons
    assert "risk_judgment_not_grounded" not in reasons


def test_amzn_risk_answer_has_depth_structure_and_contract_passes():
    packet = _amzn_risk_packet()
    synthesis = build_analytical_synthesis(
        user_query=packet["user_query"],
        analysis_plan={"answer_mode": "risk_focused_analysis", "analysis_scope": "single_company"},
        evidence_plan={"task_type": "report_summary", "answer_mode": "risk_focused_analysis", "analysis_scope": "single_company", "evidence_requirements": []},
        evidence_collection_results=[],
        evidence_sufficiency={"overall_status": "focused_sufficient", "can_synthesize": True, "dimension_status_map": packet["dimension_status_map"]},
        valid_numeric_claims=[],
        valid_text_claims=[{"sentence": row["claim"], "evidence_ids": [row["evidence_id"]]} for row in packet["text_snippets"]],
        numeric_citations=[{"evidence_id": row["evidence_id"]} for row in packet["numeric_table"]],
        text_citations=[{"evidence_id": row["evidence_id"]} for row in packet["text_snippets"]],
        numeric_evidence_cards=packet["numeric_table"],
        text_evidence_cards=packet["text_snippets"],
        limitations=[],
        answer_policy={},
        answer_mode="risk_focused_analysis",
        safety_intent="normal",
        task_type="report_summary",
        lang="zh",
        evidence_packet=packet,
    ).model_dump()
    answer = render_synthesis_text(synthesis, lang="zh", answer_mode="risk_focused_analysis", safety_intent="normal")

    for phrase in ("结论", "已验证风险文本", "基于业务模型的风险排序", "财务传导路径", "待验证数据", "证据边界"):
        assert phrase in answer
    assert "最高优先级" in answer
    assert "第二优先级" in answer
    assert "[T1]" in answer
    assert not any(term in answer for term in ("买入", "卖出", "目标价", "建议持有"))
    assert _contract_status(answer, packet, answer_mode="risk_focused_analysis") == "repairable"


def test_nvda_composite_answer_expands_named_dimensions_and_contract_passes():
    packet = _nvda_composite_packet()
    methodology = build_methodology_answer(packet, lang="zh")
    assert methodology is not None
    answer = render_synthesis_text(
        {"methodology_answer": methodology.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    for phrase in ("总体判断", "现金流质量", "估值边界", "主要风险", "反方因素 / 不确定性", "后续观察指标", "证据边界"):
        assert phrase in answer
    assert "经营现金流" in answer
    assert "FCF margin" in answer
    assert "P/E" in answer and "P/S" in answer and "FCF yield" in answer
    assert "供应" in answer or "竞争" in answer or "监管" in answer
    assert "[C1]" in answer and "[V3]" in answer and "[T" in answer
    assert not any(term in answer for term in ("买入", "卖出", "目标价", "建议持有"))
    assert _contract_status(answer, packet, answer_mode="analytical") == "repairable"


def test_single_company_overview_has_deep_structure():
    answer = _render_overview_answer()

    for phrase in (
        "结论",
        "业务定位",
        "收入和盈利",
        "现金流与估值",
        "主要风险",
        "证据边界",
    ):
        assert phrase in answer
    assert "[T1]" in answer and "[N1]" in answer
    assert not any(term in answer for term in ("买入", "卖出", "建议持有", "目标价"))
    assert _contract_status(answer, _amzn_overview_packet(), answer_mode="analytical") == "repairable"


def test_overview_cash_flow_explains_capex_drag():
    answer = _render_overview_answer()
    cash_flow_section = _section(answer, "现金流与估值", "主要风险")

    assert "经营现金流" in cash_flow_section
    assert "资本开支" in cash_flow_section
    assert "自由现金流为负" in cash_flow_section
    assert any(term in cash_flow_section for term in ("吞噬", "压制", "拖累", "资本强度"))


def test_overview_no_duplicate_cash_flow_caveats():
    answer = _render_overview_answer()
    cash_flow_section = _section(answer, "现金流与估值", "主要风险")

    assert "。；" not in answer
    assert "；；" not in answer
    assert "收入 的" not in answer
    assert not re.search(r"(?:\[[NT]\d+\])+\s+capex/revenue", answer, flags=re.IGNORECASE)
    assert cash_flow_section.count("FCF margin 为") == 1
    assert cash_flow_section.count("自由现金流为负") == 1
    assert cash_flow_section.count("资本强度压制自由现金流") == 1


def test_overview_revenue_trend_has_comparability_caveat():
    answer = _render_overview_answer()
    revenue_section = _section(answer, "收入和盈利", "现金流与估值")

    assert "不能直接等同于可比口径下的经营趋势" in revenue_section
    assert "趋势为下降" not in revenue_section
    assert "趋势为上升" not in revenue_section


def test_overview_business_model_uses_segments():
    answer = _render_overview_answer()
    business_section = _section(answer, "业务定位", "收入和盈利")

    assert "North America" in business_section
    assert "International" in business_section
    assert "AWS" in business_section
    assert "客户、市场或分部结构" not in business_section


def test_overview_unusual_margin_gets_caveat():
    answer = _render_overview_answer()

    assert "50.02%" in answer
    assert "盈利能力突出" not in answer
    assert "异常" in answer
    assert "口径" in answer
    assert "持续" in answer or "核验" in answer


def test_overview_profitability_interprets_margin_structure():
    answer = _render_overview_answer()
    profitability_section = _section(answer, "收入和盈利", "现金流与估值")

    assert "毛利率" in profitability_section
    assert "营业利润率" in profitability_section
    assert any(term in profitability_section for term in ("差距", "运营成本", "费用", "利润释放"))
    assert "不能直接外推为可持续盈利能力" in profitability_section
    assert "盈利能力突出" not in profitability_section


def test_overview_risk_section_ranks_multiple_risks():
    answer = _render_overview_answer()

    assert "最高优先级" in answer
    assert "第二优先级" in answer
    assert "履约" in answer
    assert "库存" in answer
    assert "监管" in answer or "合规" in answer
    assert "竞争" in answer
    assert "传导机制" in answer


def test_overview_risk_transmission_links_financial_impact():
    answer = _render_overview_answer()
    risk_section = _section(answer, "主要风险", "证据边界")

    assert "收入" in risk_section
    assert "利润" in risk_section or "利润率" in risk_section
    assert "现金" in risk_section or "FCF" in risk_section
    assert "[T2]" in risk_section
    assert "[T3]" in risk_section
    assert "[T4]" in risk_section


def test_business_model_text_support_for_amzn():
    evidence = {
        "T1": {
            "evidence_id": "T1",
            "ticker": "AMZN",
            "section": "ITEM_1",
            "dimension_id": "business_model",
            "requirement_id": "REQ-TEXT-AMZN-BUSINESS_MODEL",
            "supporting_snippet": "Amazon operates North America, International, and AWS segments. Net sales include online stores, third-party seller services, advertising, subscription services including Prime, and fulfillment services.",
        }
    }
    claims = [
        {
            "sentence": "AMZN 的业务模式可以基于 AWS、Prime、marketplace、advertising、fulfillment 和 net sales 分部披露来分析。",
            "citation_ref": "T1",
            "dimension_id": "business_model",
            "company": "AMZN",
        }
    ]

    valid, unsupported, _warnings = validate_text_claims_enhanced(
        claims,
        evidence,
        {
            "analysis_scope": "single_company",
            "requirement_dimension_map": {"REQ-TEXT-AMZN-BUSINESS_MODEL": "business_model"},
        },
    )

    assert valid
    assert unsupported == []


def test_debug_warning_objects_are_human_readable(tmp_path: Path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    app_copy = tmp_path / "app.mjs"
    app_copy.write_text((ROOT / "frontend" / "app.js").read_text(encoding="utf-8"), encoding="utf-8")
    script = f"""
      import {{ buildDebugBundle }} from {str(app_copy)!r};
      const bundle = buildDebugBundle({{
        trace_id: 'trace-1',
        query: '分析下amazon这家公司',
        task_type: 'report_summary',
        answer_mode: 'analytical',
        contract_status: 'passed_with_warnings',
        final_answer: 'answer',
        evidence_plan: {{ summary: {{ requirement_count: 0 }} }},
        evidence_packet: {{ summary: {{}}, numeric_evidence: [], text_evidence: [], computed_metrics: [] }},
        draft_release_decision: {{
          decision: 'released_with_warnings',
          warnings: [{{ code: 'optional_context_gap', message: 'business model missing', requirement_id: 'REQ-TEXT-AMZN-BUSINESS_MODEL', dimension_id: 'business_model' }}]
        }},
        dimensions: [],
        citations: []
      }});
      if (bundle.includes('[object Object]')) throw new Error(bundle);
      if (!bundle.includes('code=optional_context_gap')) throw new Error(bundle);
      if (!bundle.includes('requirement_id=REQ-TEXT-AMZN-BUSINESS_MODEL')) throw new Error(bundle);
      if (!bundle.includes('dimension_id=business_model')) throw new Error(bundle);
    """
    result = subprocess.run([node, "--input-type=module", "-e", script], cwd=ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
