from __future__ import annotations

from src.agent.plan_validator import deterministic_causal_research_plan
from src.agent.synthesis import build_analytical_synthesis, render_synthesis_text


def test_causal_gap_synthesis_renders_tiered_analysis_instead_of_refusal():
    plan = deterministic_causal_research_plan(
        user_query="为什么 NVIDIA 营收增长这么多",
        companies=["NVDA"],
    ).model_dump(exclude_none=True)
    evidence_plan = {
        "research_plan": plan,
        "required_answer_parts": plan["required_answer_parts"],
    }
    sufficiency = {
        "overall_status": "partial",
        "missing_but_analyzable_answer_parts": ["identify_growth_drivers"],
        "partial_required_answer_parts": ["quantify_growth"],
        "evidence_health": "degraded",
    }
    synthesis = build_analytical_synthesis(
        user_query="为什么 NVIDIA 营收增长这么多",
        analysis_plan={"companies": ["NVDA"]},
        evidence_plan=evidence_plan,
        evidence_collection_results=[],
        evidence_sufficiency=sufficiency,
        valid_numeric_claims=[{"sentence": "NVDA revenue was $68.127B.", "evidence_ids": ["N1"]}],
        valid_text_claims=[],
        numeric_citations=[{"evidence_id": "N1"}],
        text_citations=[],
        numeric_evidence_cards=[{"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": "68.127B"}],
        text_evidence_cards=[],
        limitations=[],
        answer_policy={},
        answer_mode="analytical",
        safety_intent="normal",
        task_type="report_summary",
        lang="zh",
    ).model_dump(exclude_none=True)
    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")

    assert synthesis["synthesis_mode"] == "causal_explanation_analytical_with_gaps"
    assert synthesis["short_answer"].startswith("当前不能确认 NVDA 营收增长的直接原因")
    assert synthesis["claim_tiers"]["hypothesis_to_verify"] >= 3
    assert "核心判断" in rendered
    assert "当前无法可靠计算总营收增长率" in rendered
    assert "NVDA revenue" not in rendered
    assert "已验证证据" in rendered
    assert "基于证据的合理推断" in rendered
    assert "待验证假设" in rendered
    assert "证据边界" in rendered
    assert "不能把未验证因素写成确定原因" in rendered
