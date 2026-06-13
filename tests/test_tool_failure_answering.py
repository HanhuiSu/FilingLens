from __future__ import annotations

from src.agent.answer_relevance import bounded_causal_fallback_answer, judge_answer_relevance
from src.agent.plan_validator import deterministic_causal_research_plan


def test_tool_failure_fallback_outputs_verification_framework():
    plan = deterministic_causal_research_plan(
        user_query="为什么 NVIDIA 营收增长这么多",
        companies=["NVDA"],
    ).model_dump(exclude_none=True)
    state = {
        "user_query": "为什么 NVIDIA 营收增长这么多",
        "companies": ["NVDA"],
        "research_plan_used": plan,
        "required_answer_parts": plan["required_answer_parts"],
        "missing_but_analyzable_answer_parts": ["identify_growth_drivers"],
        "tool_error_context": [
            {
                "requirement_id": "REQ-RP-NVDA-GROWTH_DRIVER_TEXT",
                "kind": "tool_execution_error",
                "failure_reason": "timeout",
            }
        ],
    }

    answer = bounded_causal_fallback_answer(state)
    decision = judge_answer_relevance(answer, state)

    assert "检索退化" in answer or "工具" in answer
    assert "待验证假设" in answer
    assert "待验证数据" in answer
    assert decision.status == "analytical_with_gaps"
