from __future__ import annotations

from src.agent.analyst_draft import (
    build_methodology_context,
    generate_analyst_draft,
)
from src.agent.draft_validation import validate_analyst_draft


class FakeLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _messages):
        class Response:
            def __init__(self, content: str):
                self.content = content

        return Response(self.content)


class CapturingLLM(FakeLLM):
    def __init__(self, content: str):
        super().__init__(content)
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return super().invoke(messages)


def _packet() -> dict:
    return {
        "user_query": "aapple和amazon你最看好哪个",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "safety_intent": "investment_advice_like",
        "numeric_table": [
            {"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue", "period_end": "2025-12-31", "value": 120.0},
            {"evidence_id": "N2", "ticker": "AAPL", "metric": "net_income", "period_end": "2025-12-31", "value": 32.0},
            {"evidence_id": "N3", "ticker": "AMZN", "metric": "revenue", "period_end": "2025-12-31", "value": 150.0},
            {"evidence_id": "N4", "ticker": "AMZN", "metric": "net_income", "period_end": "2025-12-31", "value": 22.0},
        ],
        "comparison_table": [],
        "text_snippets": [
            {"evidence_id": "T1", "ticker": "AAPL", "section": "ITEM_7", "text_snippet": "Margin discipline remained strong."},
            {"evidence_id": "T2", "ticker": "AMZN", "section": "ITEM_1A", "text_snippet": "Competition and reinvestment remain important."},
        ],
        "grouped_risk_themes": [],
        "grouped_business_themes": [],
        "provenance_notes": [],
        "missing_evidence_summary": {"overall_status": "sufficient"},
        "limitations": [],
        "citations": [{"evidence_id": "N1"}, {"evidence_id": "N2"}, {"evidence_id": "N3"}, {"evidence_id": "N4"}, {"evidence_id": "T1"}, {"evidence_id": "T2"}],
    }


def _packet_with_dimension(dimension_id: str, status: str) -> dict:
    packet = _packet()
    packet["answer_mode"] = "analytical"
    packet["task_type"] = "report_summary"
    packet["safety_intent"] = "normal"
    packet["dimension_sufficiency"] = {
        "dimension_status_map": {
            dimension_id: {
                "dimension_id": dimension_id,
                "status": status,
                "satisfied_requirements": [],
                "missing_requirements": ["REQ-MISSING"],
                "allowed_claims": [],
                "forbidden_claims": [],
                "limitation": "dimension evidence missing",
            }
        }
    }
    return packet


def _packet_with_dimension_summary() -> dict:
    packet = _packet_with_dimension("profitability_quality", "satisfied")
    packet["active_dimensions"] = ["profitability_quality", "cash_flow_quality"]
    packet["dimension_sufficiency"]["dimension_status_map"]["cash_flow_quality"] = {
        "dimension_id": "cash_flow_quality",
        "status": "missing",
        "satisfied_requirements": [],
        "missing_requirements": ["REQ-CFO"],
        "allowed_claims": [],
        "forbidden_claims": ["cash flow is strong"],
        "limitation": "当前缺少经营现金流/自由现金流证据，不能判断利润现金含量。",
    }
    packet["dimension_summary"] = [
        {
            "dimension_id": "profitability_quality",
            "status": "satisfied",
            "evidence_refs": ["N2"],
            "numeric_evidence_refs": ["N2"],
            "text_evidence_refs": [],
        },
        {
            "dimension_id": "cash_flow_quality",
            "status": "missing",
            "evidence_refs": [],
            "numeric_evidence_refs": [],
            "text_evidence_refs": [],
        },
    ]
    return packet


def _comparison_packet_with_dimensions() -> dict:
    packet = _packet()
    packet["selected_framework"] = {"framework_id": "fundamental_quality_analysis"}
    packet["active_dimensions"] = ["revenue_quality", "profitability_quality", "valuation_and_risk_boundary"]
    packet["dimension_sufficiency"] = {
        "dimension_status_map": {
            "revenue_quality": {"dimension_id": "revenue_quality", "status": "satisfied"},
            "profitability_quality": {"dimension_id": "profitability_quality", "status": "satisfied"},
            "valuation_and_risk_boundary": {
                "dimension_id": "valuation_and_risk_boundary",
                "status": "missing",
                "limitation": "当前缺少估值证据，不能判断价格是否便宜或昂贵。",
            },
        }
    }
    packet["dimension_status_map"] = packet["dimension_sufficiency"]["dimension_status_map"]
    packet["dimension_summary"] = [
        {
            "dimension_id": "revenue_quality",
            "status": "satisfied",
            "evidence_refs": ["N1", "N3"],
            "numeric_evidence_refs": ["N1", "N3"],
            "text_evidence_refs": [],
        },
        {
            "dimension_id": "profitability_quality",
            "status": "satisfied",
            "evidence_refs": ["N2", "N4"],
            "numeric_evidence_refs": ["N2", "N4"],
            "text_evidence_refs": [],
        },
        {
            "dimension_id": "valuation_and_risk_boundary",
            "status": "missing",
            "evidence_refs": [],
            "numeric_evidence_refs": [],
            "text_evidence_refs": [],
        },
    ]
    packet["red_flags"] = [{"severity": "medium", "category": "missing_evidence", "message": "Valuation evidence is missing."}]
    packet["missing_evidence_flags"] = list(packet["red_flags"])
    packet["allowed_claims"] = ["revenue scale comparison", "net income comparison"]
    packet["forbidden_claims"] = ["cheap", "expensive", "buy", "sell"]
    return packet


def _analytical_draft(statement: str, refs: list[str]) -> dict:
    return {
        "tentative_conclusion": {
            "statement": statement,
            "stance": "observation",
            "preferred_company": "",
            "citation_refs": refs,
        },
        "decision_basis": [{"statement": statement, "citation_refs": refs}],
        "supporting_points": [],
        "counterpoints": [],
        "risk_tradeoffs": [],
        "uncertainty_notes": [{"statement": "This is a limited evidence-based observation.", "citation_refs": refs}],
        "citation_refs": refs,
        "safety_notes": [],
    }


def test_generate_analyst_draft_invalid_json_falls_back(monkeypatch):
    monkeypatch.setattr("src.agent.analyst_draft._get_llm", lambda *args, **kwargs: FakeLLM("not json"))

    draft, issues = generate_analyst_draft(
        evidence_packet=_packet(),
        answer_language="English",
        synthesis_mode="balanced_comparison",
    )

    assert draft == {}
    assert any(item["reason"] == "analyst_draft_invalid_json" for item in issues)


def test_generate_analyst_draft_passes_explicit_methodology_context(monkeypatch):
    llm = CapturingLLM('{"tentative_conclusion":{},"decision_basis":[],"counterpoints":[],"citation_refs":[]}')
    monkeypatch.setattr("src.agent.analyst_draft._get_llm", lambda *args, **kwargs: llm)
    packet = _comparison_packet_with_dimensions()

    generate_analyst_draft(
        evidence_packet=packet,
        answer_language="English",
        synthesis_mode="limited_judgment",
        methodology_context=build_methodology_context(packet),
    )

    prompt = llm.messages[-1].content
    assert "## Methodology Context" in prompt
    assert '"selected_framework":"fundamental_quality_analysis"' in prompt
    assert '"active_dimensions"' in prompt
    assert '"valuation_and_risk_boundary"' in prompt
    assert '"forbidden_claims"' in prompt
    assert "core risk ranking" in prompt
    assert "risk transmission path" in prompt
    assert "revenue / profit / cash flow" in prompt
    assert "cash-flow quality, valuation boundary, and primary risks" in prompt
    assert "broad single-company overview" in prompt
    assert "cash flow and capex" in prompt
    assert "counterpoints, risk_tradeoffs, and uncertainty_notes" in prompt
    assert "delete it" in prompt
    assert "Risk and risk-transmission items should cite company-specific text evidence" in prompt


def test_dimension_missing_valuation_rejects_valuation_claims():
    result = validate_analyst_draft(
        draft=_analytical_draft("AAPL looks cheap on the current evidence.", ["N1"]),
        evidence_packet=_packet_with_dimension("valuation_and_risk_boundary", "missing"),
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    reasons = {item["reason"] for item in result["violations"]}
    assert "dimension_forbidden_valuation_claim" in reasons


def test_dimension_missing_cash_flow_rejects_cash_flow_quality_claims():
    result = validate_analyst_draft(
        draft=_analytical_draft("AAPL cash flow quality is strong.", ["N1"]),
        evidence_packet=_packet_with_dimension("cash_flow_quality", "missing"),
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    reasons = {item["reason"] for item in result["violations"]}
    assert "dimension_forbidden_cash_flow_quality_claim" in reasons


def test_dimension_missing_moat_rejects_specific_risk_claims():
    result = validate_analyst_draft(
        draft=_analytical_draft("Competition risk is severe for AAPL.", ["T1"]),
        evidence_packet=_packet_with_dimension("moat_and_competitive_risk", "missing"),
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    reasons = {item["reason"] for item in result["violations"]}
    assert "dimension_forbidden_specific_risk_claim" in reasons


def test_partial_profitability_requires_net_margin_or_net_income_framing():
    result = validate_analyst_draft(
        draft=_analytical_draft("AAPL has stronger profitability.", ["N2"]),
        evidence_packet=_packet_with_dimension("profitability_quality", "partial"),
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    reasons = {item["reason"] for item in result["violations"]}
    assert "dimension_partial_profitability_requires_net_margin_or_net_income" in reasons


def test_validate_analyst_draft_rejects_unknown_citation_and_invented_number():
    result = validate_analyst_draft(
        draft={
            "tentative_conclusion": {
                "statement": "AAPL looks stronger with 999 USD of profit.",
                "stance": "leans_toward_company",
                "preferred_company": "AAPL",
                "citation_refs": ["BAD"],
            },
            "decision_basis": [],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "citation_refs": ["BAD"],
            "safety_notes": [{"statement": "This is not investment advice.", "citation_refs": []}],
        },
        evidence_packet=_packet(),
        safety_policy={"answer_mode": "comparison_brief", "safety_intent": "investment_advice_like"},
        synthesis_mode="limited_judgment",
    ).model_dump()

    assert result["passed"] is False
    reasons = {item["reason"] for item in result["violations"]}
    assert "unknown_citation_ref" in reasons or "invented_number" in reasons


def test_validate_analyst_draft_accepts_scaled_chinese_units_and_directional_percentages():
    packet = _packet_with_dimension("cash_flow_quality", "satisfied")
    packet["active_dimensions"] = ["cash_flow_quality"]
    packet["dimension_summary"] = [
        {
            "dimension_id": "cash_flow_quality",
            "status": "satisfied",
            "evidence_refs": ["N1", "N2", "N3", "N4", "N5"],
            "numeric_evidence_refs": ["N1", "N2", "N3", "N4", "N5"],
            "text_evidence_refs": [],
        }
    ]
    packet["numeric_table"] = [
        {"evidence_id": "N1", "ticker": "AMZN", "metric": "revenue", "value": 181519000000, "unit": "USD"},
        {"evidence_id": "N2", "ticker": "AMZN", "metric": "revenue", "value": 213386000000, "unit": "USD"},
        {"evidence_id": "N3", "ticker": "AMZN", "metric": "revenue_growth", "value": -0.14934, "unit": "ratio", "display_value": "-14.93%"},
        {"evidence_id": "N4", "ticker": "AMZN", "metric": "free_cash_flow", "value": -2472000000, "unit": "USD", "display_value": "$-2.47B"},
        {"evidence_id": "N5", "ticker": "AMZN", "metric": "fcf_margin", "value": -0.013618, "unit": "ratio", "display_value": "-1.36%"},
    ]
    packet["citations"] = [{"evidence_id": f"N{i}"} for i in range(1, 6)]
    statement = "最新收入为1815.19亿美元，较上一季度下降约14.93%；自由现金流为负24.72亿美元，FCF margin 为-1.36%。"

    result = validate_analyst_draft(
        draft=_analytical_draft(statement, ["N1", "N2", "N3", "N4", "N5"]),
        evidence_packet=packet,
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    reasons = {item["reason"] for item in result["violations"]}
    assert "invented_number" not in reasons
    assert result["passed"] is True


def test_validate_comparison_draft_requires_counterpoint_and_non_advisory_note():
    result = validate_analyst_draft(
        draft={
            "tentative_conclusion": {
                "statement": "If profitability matters more, AAPL currently looks stronger.",
                "stance": "leans_toward_company",
                "preferred_company": "AAPL",
                "citation_refs": ["N2"],
            },
            "decision_basis": [{"statement": "AAPL net income is higher.", "citation_refs": ["N2", "N4"]}],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "citation_refs": ["N2", "N4"],
            "safety_notes": [],
        },
        evidence_packet=_packet(),
        safety_policy={"answer_mode": "comparison_brief", "safety_intent": "investment_advice_like"},
        synthesis_mode="limited_judgment",
    ).model_dump()

    assert result["passed"] is True
    assert result["final_status"] == "passed_with_warnings"
    reasons = {item["reason"] for item in result["warnings"]}
    assert "missing_counterpoint" in reasons
    assert "missing_non_advisory_note" in reasons


def test_validate_analytical_draft_requires_uncertainty_notes():
    packet = _packet()
    packet["answer_mode"] = "analytical"
    packet["task_type"] = "report_summary"
    result = validate_analyst_draft(
        draft={
            "tentative_conclusion": {
                "statement": "The main issue is competition pressure.",
                "stance": "main_issue",
                "preferred_company": "",
                "citation_refs": ["T2"],
            },
            "decision_basis": [{"statement": "Validated text highlights competition pressure.", "citation_refs": ["T2"]}],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "citation_refs": ["T2"],
            "safety_notes": [],
        },
        evidence_packet=packet,
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    assert result["passed"] is True
    assert result["final_status"] == "passed_with_warnings"
    reasons = {item["reason"] for item in result["warnings"]}
    assert "missing_uncertainty_notes" in reasons


def test_validate_cautious_outlook_blocks_forecast_and_projects_accepted_items():
    packet = _packet()
    packet["answer_mode"] = "cautious_outlook"
    result = validate_analyst_draft(
        draft={
            "tentative_conclusion": {
                "statement": "Revenue will rise tomorrow.",
                "stance": "cautious_observation",
                "preferred_company": "",
                "citation_refs": ["N1"],
            },
            "decision_basis": [{"statement": "Current disclosed data shows stable demand.", "citation_refs": ["T1"]}],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [{"statement": "This is only an observation from disclosed data.", "citation_refs": ["T1"]}],
            "citation_refs": ["N1", "T1"],
            "safety_notes": [],
        },
        evidence_packet=packet,
        safety_policy={"answer_mode": "cautious_outlook", "safety_intent": "normal"},
        synthesis_mode="cautious_outlook",
    ).model_dump()

    reasons = {item["reason"] for item in result["violations"]}
    assert "unsupported_forecast_wording" in reasons
    assert result["accepted_draft"] == {}


def test_validate_rejects_direct_stock_pick_language():
    result = validate_analyst_draft(
        draft={
            "tentative_conclusion": {
                "statement": "我建议买入 AAPL。",
                "stance": "leans_toward_company",
                "preferred_company": "AAPL",
                "citation_refs": ["N2"],
            },
            "decision_basis": [{"statement": "AAPL net income is higher.", "citation_refs": ["N2", "N4"]}],
            "supporting_points": [{"statement": "AAPL looks stronger.", "citation_refs": ["N2"]}],
            "counterpoints": [{"statement": "AMZN still has larger revenue scale.", "citation_refs": ["N3"]}],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "citation_refs": ["N2", "N3", "N4"],
            "safety_notes": [{"statement": "这不是投资建议。", "citation_refs": []}],
        },
        evidence_packet=_packet(),
        safety_policy={"answer_mode": "comparison_brief", "safety_intent": "investment_advice_like"},
        synthesis_mode="limited_judgment",
    ).model_dump()

    reasons = {item["reason"] for item in result["violations"]}
    assert "investment_advice_wording" in reasons


def test_validate_rejects_strong_risk_judgment_without_text_evidence():
    packet = _packet()
    packet["text_snippets"] = []
    packet["grouped_risk_themes"] = []
    result = validate_analyst_draft(
        draft={
            "tentative_conclusion": {
                "statement": "The main risk is competition pressure.",
                "stance": "main_issue",
                "preferred_company": "",
                "citation_refs": ["N1"],
            },
            "decision_basis": [{"statement": "Competition risk is severe.", "citation_refs": ["N1"]}],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [{"statement": "Risk pressure is rising.", "citation_refs": ["N1"]}],
            "uncertainty_notes": [],
            "citation_refs": ["N1"],
            "safety_notes": [],
        },
        evidence_packet=packet,
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="limited_analysis",
    ).model_dump()

    reasons = {item["reason"] for item in result["violations"]}
    assert "strong_risk_without_text_evidence" in reasons or "risk_judgment_without_text_ref" in reasons


def test_validate_merges_inline_citations_and_accepts_cited_risk_items():
    packet = _packet()
    result = validate_analyst_draft(
        draft={
            "tentative_conclusion": {
                "statement": "AMZN 的主要判断受竞争文本和收入规模共同约束。[T2][N3]",
                "stance": "evidence_bounded",
                "preferred_company": "AMZN",
                "citation_refs": [],
            },
            "decision_basis": [{"statement": "AMZN 收入规模仍是反方因素。[N3]", "citation_refs": []}],
            "supporting_points": [],
            "counterpoints": [{"statement": "收入证据支持规模反方因素。[N3]", "citation_refs": []}],
            "risk_tradeoffs": [{"statement": "竞争风险需要结合披露文本观察。", "citation_refs": ["T2"]}],
            "uncertainty_notes": [{"statement": "当前仍是基于已验证数据的有限判断。[N3]", "citation_refs": []}],
            "citation_refs": [],
            "safety_notes": [{"statement": "这不是投资建议。", "citation_refs": []}],
        },
        evidence_packet=packet,
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    assert result["passed"] is True
    accepted = result["accepted_draft"]
    assert accepted["tentative_conclusion"]["citation_refs"] == ["T2", "N3"]
    assert accepted["tentative_conclusion"]["statement"].endswith("共同约束。")
    reasons = {item["reason"] for item in result["violations"]}
    assert "risk_judgment_without_text_ref" not in reasons
    assert "risk_judgment_not_grounded" not in reasons


def test_validate_accepts_dual_track_dimension_analyses():
    packet = _packet_with_dimension_summary()

    result = validate_analyst_draft(
        draft={
            "framework_summary": "Use the fundamental quality framework.",
            "dimension_analyses": [
                {
                    "dimension_id": "profitability_quality",
                    "status": "satisfied",
                    "claim": "AAPL profitability is supported by net income evidence.",
                    "evidence_refs": ["N2"],
                }
            ],
            "overall_judgment": "AAPL profitability is supported by validated evidence.",
            "methodology_counterpoints": ["Cash flow evidence is still missing."],
            "methodology_limitations": ["Do not judge cash-flow quality."],
            "follow_up_metrics": ["operating cash flow"],
            "tentative_conclusion": {
                "statement": "AAPL profitability is supported by net income evidence.",
                "stance": "observation",
                "preferred_company": "",
                "citation_refs": ["N2"],
            },
            "decision_basis": [],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "citation_refs": ["N2"],
            "safety_notes": [],
        },
        evidence_packet=packet,
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    assert result["passed"] is True
    reasons = {item["reason"] for item in result["violations"]}
    assert "dimension_partial_profitability_requires_net_margin_or_net_income" not in reasons
    assert "inactive_dimension_analyzed" not in reasons
    assert result["accepted_draft"]["dimension_analyses"][0]["dimension_id"] == "profitability_quality"


def test_validate_comparison_requires_and_accepts_methodology_dimension_analyses():
    packet = _comparison_packet_with_dimensions()
    valid = validate_analyst_draft(
        draft={
            "framework_summary": "Use the fundamental quality framework.",
            "dimension_analyses": [
                {
                    "dimension_id": "revenue_quality",
                    "status": "satisfied",
                    "claim": "AMZN has stronger revenue scale.",
                    "evidence_refs": ["N3"],
                },
                {
                    "dimension_id": "profitability_quality",
                    "status": "satisfied",
                    "claim": "AAPL has stronger profitability based on net income evidence.",
                    "evidence_refs": ["N2"],
                },
            ],
            "overall_judgment": "If profitability matters more, AAPL currently looks stronger; AMZN has the revenue-scale counterpoint.",
            "methodology_counterpoints": ["AMZN has stronger revenue scale."],
            "methodology_limitations": ["Valuation evidence is missing."],
            "tentative_conclusion": {
                "statement": "If profitability matters more, AAPL currently looks stronger; AMZN has the revenue-scale counterpoint.",
                "stance": "leans_toward_company",
                "preferred_company": "AAPL",
                "citation_refs": ["N2", "N3"],
            },
            "decision_basis": [{"statement": "AAPL net income evidence supports the profitability view.", "citation_refs": ["N2"]}],
            "supporting_points": [],
            "counterpoints": [{"statement": "AMZN has stronger revenue scale.", "citation_refs": ["N3"]}],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "citation_refs": ["N2", "N3"],
            "safety_notes": [{"statement": "This is not investment advice.", "citation_refs": []}],
        },
        evidence_packet=packet,
        safety_policy={"answer_mode": "comparison_brief", "safety_intent": "investment_advice_like"},
        synthesis_mode="limited_judgment",
    ).model_dump()
    missing = validate_analyst_draft(
        draft={
            "tentative_conclusion": {
                "statement": "If profitability matters more, AAPL currently looks stronger.",
                "stance": "leans_toward_company",
                "preferred_company": "AAPL",
                "citation_refs": ["N2"],
            },
            "decision_basis": [{"statement": "AAPL net income evidence supports the profitability view.", "citation_refs": ["N2"]}],
            "counterpoints": [{"statement": "AMZN has stronger revenue scale.", "citation_refs": ["N3"]}],
            "safety_notes": [{"statement": "This is not investment advice.", "citation_refs": []}],
        },
        evidence_packet=packet,
        safety_policy={"answer_mode": "comparison_brief", "safety_intent": "investment_advice_like"},
        synthesis_mode="limited_judgment",
    ).model_dump()

    assert valid["passed"] is True
    assert [item["dimension_id"] for item in valid["accepted_draft"]["dimension_analyses"]] == [
        "revenue_quality",
        "profitability_quality",
    ]
    assert any(item["reason"] == "missing_dimension_analyses" for item in missing["violations"])


def test_validate_rejects_missing_dimension_analysis_and_wrong_dimension_refs():
    missing = validate_analyst_draft(
        draft={
            "dimension_analyses": [
                {
                    "dimension_id": "cash_flow_quality",
                    "status": "missing",
                    "claim": "AAPL cash flow quality is strong.",
                    "evidence_refs": ["N2"],
                }
            ],
            "tentative_conclusion": {"statement": "AAPL cash flow quality is strong.", "citation_refs": ["N2"]},
            "decision_basis": [],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [{"statement": "Limited evidence.", "citation_refs": ["N2"]}],
            "citation_refs": ["N2"],
            "safety_notes": [],
        },
        evidence_packet=_packet_with_dimension_summary(),
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()
    wrong_ref = validate_analyst_draft(
        draft={
            "dimension_analyses": [
                {
                    "dimension_id": "profitability_quality",
                    "status": "satisfied",
                    "claim": "AAPL profitability is supported by net income evidence.",
                    "evidence_refs": ["T1"],
                }
            ],
            "tentative_conclusion": {"statement": "AAPL profitability is supported by net income evidence.", "citation_refs": ["N2"]},
            "decision_basis": [],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [{"statement": "Limited evidence.", "citation_refs": ["N2"]}],
            "citation_refs": ["N2"],
            "safety_notes": [],
        },
        evidence_packet=_packet_with_dimension_summary(),
        safety_policy={"answer_mode": "analytical", "safety_intent": "normal"},
        synthesis_mode="validated_analysis",
    ).model_dump()

    assert any(item["reason"] == "dimension_forbidden_cash_flow_quality_claim" for item in missing["violations"])
    assert any(item["reason"] == "dimension_claim_refs_not_in_dimension" for item in wrong_ref["violations"])
