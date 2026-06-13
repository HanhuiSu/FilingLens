"""Tests for the final answer relevance gate."""

from __future__ import annotations

from src.agent.answer_relevance import bounded_causal_fallback_answer, judge_answer_relevance
from src.agent.plan_validator import deterministic_causal_research_plan


def _state(*, missing_driver: bool = True, partial_parts: list[str] | None = None) -> dict:
    plan = deterministic_causal_research_plan(
        user_query="为什么 NVIDIA 的营收增长这么多",
        companies=["NVDA"],
    ).model_dump(exclude_none=True)
    return {
        "user_query": "为什么 NVIDIA 的营收增长这么多",
        "companies": ["NVDA"],
        "research_plan_used": plan,
        "required_answer_parts": plan["required_answer_parts"],
        "missing_required_answer_parts": [],
        "missing_but_analyzable_answer_parts": ["identify_growth_drivers"] if missing_driver else [],
        "partial_required_answer_parts": list(partial_parts or []),
        "numeric_evidence": [{"evidence_id": "N1", "metric": "revenue"}],
    }


def test_causal_answer_with_only_growth_number_fails_relevance_when_driver_missing():
    decision = judge_answer_relevance("NVIDIA revenue increased significantly [N1].", _state())

    assert decision.status == "failed"
    assert decision.route == "repair_answer"
    assert decision.action == "downgrade_to_bounded"
    assert decision.recommended_actions == ["downgrade_to_bounded"]
    assert all("do not generate new uncited analysis" in item for item in decision.repair_instructions)
    assert "identify_growth_drivers" in decision.missing_but_analyzable_answer_parts


def test_one_sentence_constraint_for_buy_question():
    state = {
        "user_query": "请用一句话回答：NVDA 是不是可以买？",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "canonical_intent": {"requested_dimensions": ["valuation_and_risk_boundary"]},
    }
    bad = "不能给买卖建议。当前证据只能做估值边界观察。"
    good = "不能给买卖建议；当前证据只能做有限估值边界观察。"

    bad_decision = judge_answer_relevance(bad, state)
    good_decision = judge_answer_relevance(good, state)

    assert bad_decision.route == "repair_answer"
    assert bad_decision.action == "downgrade_to_bounded"
    assert good_decision.route == "finalize"
    assert good_decision.action == "pass"
    assert "买入" not in good and "推荐买" not in good


def test_explicit_causal_boundary_releases_with_warnings_when_driver_missing():
    decision = judge_answer_relevance(
        "当前证据只能量化增长 [N1]，但原因证据不足，不能解释原因。",
        _state(),
    )

    assert decision.status == "failed"
    assert decision.route == "repair_answer"
    assert any(item["code"] == "causal_analysis_framework_missing" for item in decision.deterministic_relevance_failures)


def test_causal_answer_with_tiered_analysis_and_driver_gap_is_analytical_with_gaps():
    answer = "\n".join(
        [
            "核心判断",
            "可以分层分析，但不能把未验证因素写成确定原因。",
            "已验证证据",
            "- 已验证收入证据可作为增长量化的起点。[N1]",
            "基于证据的合理推断",
            "- 收入数字本身只能支持发生了增长，不能单独证明原因。",
            "待验证假设",
            "- 待验证假设：云厂商 AI capex 是否继续扩张。",
            "关键观察指标",
            "- 分部收入、ASP、出货量和递延收入。",
            "证据边界",
            "- 直接 driver text 不完整时，不能声称确定原因。",
        ]
    )
    decision = judge_answer_relevance(answer, _state())

    assert decision.status == "analytical_with_gaps"
    assert decision.decision == "warning"
    assert decision.route == "finalize"
    assert decision.missing_but_analyzable_answer_parts == ["identify_growth_drivers"]


def test_causal_answer_with_driver_text_citation_passes_relevance():
    decision = judge_answer_relevance(
        "Revenue growth was driven by data center demand according to filing text [T1].",
        _state(missing_driver=False),
    )

    assert decision.status == "passed"
    assert decision.route == "finalize"


def test_causal_answer_with_driver_and_partial_growth_gets_warning():
    decision = judge_answer_relevance(
        "总体驱动：Revenue growth was driven by demand for accelerated computing [T1]. 证据边界：总营收增长量化仍不完整。",
        _state(missing_driver=False, partial_parts=["quantify_growth"]),
    )

    assert decision.status == "passed_with_warnings"
    assert decision.decision == "warning"
    assert decision.route == "finalize"
    assert decision.partial_required_answer_parts == ["quantify_growth"]


def test_driver_claim_without_text_citation_fails():
    decision = judge_answer_relevance(
        "总体驱动：营收增长主要由 AI 需求驱动。",
        _state(missing_driver=False),
    )

    assert decision.status == "failed"
    assert decision.route == "repair_answer"
    assert any(item["code"] == "driver_claim_without_text_evidence" for item in decision.deterministic_relevance_failures)


def test_bounded_causal_fallback_says_it_can_quantify_but_not_explain():
    answer = bounded_causal_fallback_answer(_state())

    assert "结论" in answer
    assert "已验证事实" in answer
    assert "待验证假设" in answer
    assert "证据边界" in answer
    assert "[N1]" in answer


def test_relevance_rejects_boundary_only_fallback():
    decision = judge_answer_relevance(
        "证据边界：当前答案未可靠覆盖估值边界，只能列出边界。",
        {
            "user_query": "NVDA 估值贵不贵？",
            "canonical_intent": {"requested_dimensions": ["valuation_and_risk_boundary"]},
        },
    )

    assert decision.route == "repair_answer"
    assert any(item["code"] == "boundary_only_fallback_missing_analysis" for item in decision.deterministic_relevance_failures)


def test_relevance_accepts_layered_bounded_analysis():
    answer = "\n".join(
        [
            "结论",
            "有限判断：估值风险偏高，但不构成投资建议。[N1][N2]",
            "已验证事实",
            "- P/E 为 60x。[N1]",
            "- P/S 为 25x。[N2]",
            "合理推断",
            "- 合理推断：较高倍数意味着估值容错空间较低。[N1][N2]",
            "待验证假设",
            "- 待验证：增长率、FCF 转化和同业估值基准。",
            "证据边界",
            "- 不能给买卖建议或目标价。",
        ]
    )
    decision = judge_answer_relevance(
        answer,
        {
            "user_query": "NVDA 估值贵不贵？",
            "canonical_intent": {"requested_dimensions": ["valuation_and_risk_boundary"]},
        },
    )

    assert decision.route == "finalize"


def test_overview_numeric_validation_failure_cannot_clean_pass_as_unavailable_data():
    state = {
        "user_query": "AMZN overview",
        "research_plan_used": {
            "question_type": "overview",
            "required_answer_parts": [
                {"id": "overview", "description": "Company overview.", "required": True},
            ],
        },
        "required_answer_parts": [
            {"id": "overview", "description": "Company overview.", "required": True},
        ],
        "evidence_plan": {
            "evidence_requirements": [
                {
                    "requirement_id": "REQ-LEGACY-REV",
                    "requirement_type": "numeric",
                    "required": True,
                    "requirement_scope": "core",
                    "merged_from": ["legacy"],
                }
            ]
        },
        "evidence_validation_records": [
            {
                "requirement_id": "REQ-LEGACY-REV",
                "evidence_type": "numeric",
                "status": "missing",
                "tool_returned_count": 1,
                "validated_evidence_count": 0,
                "rejected_evidence_reason": "metric_mapping_failed",
            }
        ],
    }

    decision = judge_answer_relevance("AMZN overview：没有财务指标可用。", state)

    assert decision.status == "failed"
    assert decision.route == "repair_answer"
    assert any(
        item["code"] == "overview_numeric_validation_issue_misstated_as_unavailable"
        for item in decision.deterministic_relevance_failures
    )


def test_overview_numeric_validation_failure_releases_only_with_warning_when_not_misstated():
    state = {
        "user_query": "AMZN overview",
        "research_plan_used": {
            "question_type": "overview",
            "required_answer_parts": [
                {"id": "overview", "description": "Company overview.", "required": True},
            ],
        },
        "required_answer_parts": [
            {"id": "overview", "description": "Company overview.", "required": True},
        ],
        "evidence_plan": {
            "evidence_requirements": [
                {
                    "requirement_id": "REQ-LEGACY-REV",
                    "requirement_type": "numeric",
                    "required": True,
                    "requirement_scope": "core",
                    "merged_from": ["legacy", "research_plan"],
                }
            ]
        },
        "evidence_validation_records": [
            {
                "requirement_id": "REQ-LEGACY-REV",
                "evidence_type": "numeric",
                "status": "missing",
                "tool_returned_count": 1,
                "validated_evidence_count": 0,
                "rejected_evidence_reason": "numeric_validation_failed",
            }
        ],
    }

    decision = judge_answer_relevance("AMZN overview：本轮结构化财务数据未能通过验证。", state)

    assert decision.status == "passed_with_warnings"
    assert decision.decision == "warning"
    assert decision.route == "finalize"
