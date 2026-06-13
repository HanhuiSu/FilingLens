from datetime import date

from src.agent.query_plan import build_classification_state
from src.agent.rendering import _dedupe_text_cards
from src.agent.synthesis import build_analytical_synthesis, build_risk_focused_answer, render_synthesis_text


def _risk_packet(*, status: str = "satisfied", text: bool = True) -> dict:
    text_snippets = []
    if text:
        text_snippets.append(
            {
                "evidence_id": "T9",
                "ticker": "NVDA",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "claim": "NVDA 披露需求 / 供应链风险和新产品和需求不确定性，可能影响收入增长和利润率。",
                "supporting_snippet": "需求 / 供应链风险和新产品和需求不确定性 may affect revenue growth and margin.",
                "form_type": "10-K",
            }
        )
    return {
        "task_type": "report_summary",
        "answer_mode": "risk_focused_analysis",
        "analysis_scope": "single_company",
        "active_dimensions": ["business_model", "moat_and_competitive_risk"],
        "dimension_status_map": {
            "business_model": {"status": "satisfied"},
            "moat_and_competitive_risk": {"status": status},
        },
        "numeric_table": [
            {"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "display_value": "$60.90B"},
            {"evidence_id": "N2", "ticker": "NVDA", "metric": "net_income", "display_value": "$29.80B"},
            {"evidence_id": "N3", "ticker": "NVDA", "metric": "net_margin", "display_value": "48.93%"},
        ],
        "text_snippets": text_snippets,
    }


def test_satisfied_risk_dimension_forces_risk_focused_answer():
    packet = _risk_packet()
    synthesis = build_analytical_synthesis(
        user_query="nvidai最大的问题是什么",
        analysis_plan={"answer_mode": "risk_focused_analysis", "analysis_scope": "single_company"},
        evidence_plan={"task_type": "report_summary", "answer_mode": "risk_focused_analysis", "analysis_scope": "single_company"},
        evidence_collection_results=[],
        evidence_sufficiency={
            "overall_status": "focused_sufficient",
            "can_synthesize": True,
            "required_text_satisfied_rate": 1.0,
            "dimension_status_map": packet["dimension_status_map"],
        },
        valid_numeric_claims=[],
        valid_text_claims=[{"sentence": packet["text_snippets"][0]["claim"], "evidence_ids": ["T9"]}],
        numeric_citations=[],
        text_citations=[{"evidence_id": "T9"}],
        numeric_evidence_cards=[],
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

    assert synthesis["synthesis_mode"] == "risk_focused_analysis"
    assert "缺少足够的已验证风险文本证据" not in answer
    assert "不能判断 NVDA 最大的问题" not in answer
    assert "风险判断" in answer
    assert "[T9]" in answer


def test_risk_answer_fallback_theme_uses_validated_text_evidence():
    risk_answer = build_risk_focused_answer(_risk_packet(), lang="zh")

    assert risk_answer is not None
    payload = risk_answer.model_dump(exclude_none=True)
    assert payload["top_risk"]["evidence_refs"] == ["T9"]
    assert "需求" in payload["top_risk"]["theme_name"] or "供应链" in payload["top_risk"]["theme_name"]
    assert "缺少足够" not in payload["direct_judgment"]


def test_draft_risk_rendering_keeps_priority_and_observation_metrics():
    risk_answer = build_risk_focused_answer(_risk_packet(), lang="zh")
    assert risk_answer is not None
    synthesis = {
        "final_answer_source": "analyst_draft",
        "accepted_draft": {
            "tentative_conclusion": {
                "statement": "若 NVDA 面临需求和供应链波动，其增长可能承压。",
                "citation_refs": ["T9"],
            },
            "decision_basis": [
                {
                    "statement": "披露文本提到需求和供应链风险可能影响增长。",
                    "citation_refs": ["T9"],
                }
            ],
            "supporting_points": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
        },
        "risk_focused_answer": risk_answer.model_dump(exclude_none=True),
    }

    answer = render_synthesis_text(synthesis, lang="zh", answer_mode="risk_focused_analysis", safety_intent="normal")

    assert "最高优先级" in answer
    assert "关键观察指标" in answer
    assert "观察" in answer or "指标" in answer


def test_insufficient_risk_mode_requires_missing_or_absent_text():
    packet = _risk_packet(status="missing", text=False)
    synthesis = build_analytical_synthesis(
        user_query="nvidai最大的问题是什么",
        analysis_plan={"answer_mode": "risk_focused_analysis", "analysis_scope": "single_company"},
        evidence_plan={"task_type": "report_summary", "answer_mode": "risk_focused_analysis", "analysis_scope": "single_company"},
        evidence_collection_results=[],
        evidence_sufficiency={
            "overall_status": "insufficient",
            "can_synthesize": False,
            "required_text_satisfied_rate": 0.0,
            "dimension_status_map": packet["dimension_status_map"],
        },
        valid_numeric_claims=[],
        valid_text_claims=[],
        numeric_citations=[],
        text_citations=[],
        numeric_evidence_cards=[],
        text_evidence_cards=[],
        limitations=[],
        answer_policy={},
        answer_mode="risk_focused_analysis",
        safety_intent="normal",
        task_type="report_summary",
        lang="zh",
        evidence_packet=packet,
    ).model_dump()

    assert synthesis["synthesis_mode"] == "insufficient_risk_evidence"
    assert packet["dimension_status_map"]["moat_and_competitive_risk"]["status"] == "missing"


def test_risk_focused_active_dimensions_match_status_keys():
    state = build_classification_state(
        user_query="nvidai最大的问题是什么",
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="risk-consistency",
        today=date(2026, 4, 27),
    )

    assert state["selected_analysis_framework"]["active_dimension_ids"] == [
        "business_model",
        "moat_and_competitive_risk",
    ]
    assert state["trace_summary"]["active_analysis_dimensions"] == [
        "business_model",
        "moat_and_competitive_risk",
    ]
    assert state["trace_summary"]["supporting_context_dimensions"] == [
        "business_model",
        "revenue_quality",
        "profitability_quality",
    ]


def test_text_evidence_dedupes_same_risk_claim():
    rows = [
        {
            "evidence_id": "T1",
            "ticker": "NVDA",
            "form_type": "10-K",
            "section": "ITEM_1A",
            "chunk_order": 1,
            "claim": "NVDA faces demand and supply chain risks.",
        },
        {
            "evidence_id": "T2",
            "ticker": "NVDA",
            "form_type": "10-K",
            "section": "ITEM_1A",
            "chunk_order": 2,
            "claim": "NVDA faces demand and supply chain risks.",
        },
    ]

    assert len(_dedupe_text_cards(rows)) == 1
