"""Tests for Research Planner V1 JSON planning."""

from __future__ import annotations

from types import SimpleNamespace

from src.agent.research_planner import build_research_plan_raw, planner_mode


class _FakePlannerLlm:
    def invoke(self, _messages):
        return SimpleNamespace(
            content="""
            {
              "question_type": "causal_explanation",
              "user_goal": "Explain why NVIDIA revenue grew.",
              "companies": ["NVDA"],
              "time_scope": "latest",
              "required_answer_parts": [
                {"id": "quantify_growth", "description": "Quantify growth", "required": true},
                {"id": "identify_growth_drivers", "description": "Identify drivers", "required": true},
                {"id": "state_evidence_boundary", "description": "State boundary", "required": true}
              ],
              "evidence_requests": [
                {
                  "id": "revenue_growth_numeric",
                  "type": "numeric",
                  "scope": "core",
                  "company": "NVDA",
                  "metrics": ["revenue", "revenue_growth"],
                  "answer_part_ids": ["quantify_growth"]
                },
                {
                  "id": "growth_driver_text",
                  "type": "text",
                  "scope": "core",
                  "company": "NVDA",
                  "sections": ["ITEM_7", "ITEM_2"],
                  "queries": ["revenue growth drivers"],
                  "answer_part_ids": ["identify_growth_drivers"]
                }
              ],
              "fallback_answer_policy": "If driver text is missing, say causes cannot be explained.",
              "forbidden_claims": ["Do not infer causes from growth alone."]
            }
            """
        )


def test_planner_mode_defaults_to_expanded():
    assert planner_mode("not-a-mode") == "expanded"


def test_research_planner_parses_llm_json_without_live_model(monkeypatch):
    captured = {}

    def fake_get_llm(**kwargs):
        captured.update(kwargs)
        return _FakePlannerLlm()

    monkeypatch.setattr("src.agent.research_planner._get_llm", fake_get_llm)

    raw, trace = build_research_plan_raw(
        user_query="为什么 NVIDIA 的营收增长这么多",
        companies=["NVDA"],
        canonical_intent={"intent_family": "revenue"},
        query_understanding={},
    )

    assert trace["status"] == "parsed"
    assert trace["duration_ms"] >= 0
    assert captured["timeout"] == 25.0
    assert raw["question_type"] == "causal_explanation"
    assert {
        "quantify_growth",
        "identify_growth_drivers",
    } <= {part["id"] for part in raw["required_answer_parts"]}


def test_research_planner_timeout_records_fallback_reason(monkeypatch):
    def timeout_get_llm(**_kwargs):
        raise TimeoutError("planner timed out")

    monkeypatch.setattr("src.agent.research_planner._get_llm", timeout_get_llm)

    raw, trace = build_research_plan_raw(
        user_query="分析 Amazon",
        companies=["AMZN"],
        canonical_intent={"intent_family": "overview"},
        query_understanding={},
    )

    assert raw == {}
    assert trace["status"] == "llm_failed"
    assert trace["fallback_used"] is True
    assert trace["fallback_reason"] == "timeout"
    assert trace["duration_ms"] >= 0


def test_obvious_causal_query_uses_deterministic_skeleton_without_llm(monkeypatch):
    from src.agent.nodes import research_plan_node

    def fail_if_called(**_kwargs):
        raise AssertionError("LLM planner should not run for obvious causal query")

    monkeypatch.setattr("src.agent.nodes.build_research_plan_raw", fail_if_called)

    out = research_plan_node(
        {
            "user_query": "为什么 NVIDIA 的营收增长这么多",
            "companies": ["NVDA"],
            "needs_tools": True,
            "answer_mode": "direct_fact",
            "safety_intent": "normal",
            "evidence_plan": {"evidence_requirements": []},
            "canonical_intent": {"intent_family": "revenue"},
        }
    )

    assert out["research_plan_source"] == "deterministic_fallback"
    assert out["research_plan_fallback_reason"] == "validator_injected_causal"
    assert out["research_plan_duration_ms"] >= 0
    assert out["research_plan_used"]["question_type"] == "causal_explanation"


def test_overview_timeout_does_not_fall_back_to_causal_plan(monkeypatch):
    from src.agent.nodes import research_plan_node

    legacy_plan = {
        "evidence_requirements": [
            {
                "requirement_id": "REQ-LEGACY-AMZN-OVERVIEW",
                "requirement_type": "text",
                "company": "AMZN",
                "dimension_id": "business_model",
                "required": True,
                "requirement_scope": "core",
            }
        ],
        "core_requirement_ids": ["REQ-LEGACY-AMZN-OVERVIEW"],
    }
    monkeypatch.setattr("src.agent.nodes.planner_mode", lambda: "expanded")
    monkeypatch.setattr(
        "src.agent.nodes.build_research_plan_raw",
        lambda **_kwargs: ({}, {"source": "llm", "status": "llm_failed", "fallback_used": True, "fallback_reason": "timeout", "duration_ms": 25000}),
    )

    out = research_plan_node(
        {
            "user_query": "amazon overview",
            "companies": ["AMZN"],
            "needs_tools": True,
            "answer_mode": "analytical",
            "safety_intent": "normal",
            "evidence_plan": legacy_plan,
            "canonical_intent": {"intent_family": "overview", "analysis_scope": "single_company"},
        }
    )

    assert out["research_plan_fallback_reason"] == "timeout"
    assert out["research_plan_used"]["question_type"] == "unknown"
    assert out["research_plan_used"]["planner_source"] == "legacy_evidence_plan"
    assert out["plan_coverage_decision"]["strategy"] == "legacy_only"
    assert out["evidence_plan_used"]["source"] == "legacy_evidence_plan"
    assert [req["requirement_id"] for req in out["evidence_plan"]["evidence_requirements"]] == ["REQ-LEGACY-AMZN-OVERVIEW"]


def test_deterministic_intent_shortcuts_skip_llm_planner(monkeypatch):
    from src.agent.nodes import research_plan_node

    def fail_if_called(**_kwargs):
        raise AssertionError("LLM planner should not run for deterministic shortcut intents")

    monkeypatch.setattr("src.agent.nodes.planner_mode", lambda: "expanded")
    monkeypatch.setattr("src.agent.nodes.build_research_plan_raw", fail_if_called)

    cases = [
        {
            "user_query": "为什么amazon的利润下降了",
            "companies": ["AMZN"],
            "answer_mode": "direct_fact",
            "safety_intent": "normal",
        },
        {
            "user_query": "AMZN 现在值得买吗？",
            "companies": ["AMZN"],
            "answer_mode": "analytical",
            "safety_intent": "investment_advice_like",
        },
        {
            "user_query": "AMZN 和 NVDA 哪个更危险？",
            "companies": ["AMZN", "NVDA"],
            "answer_mode": "comparison_brief",
            "safety_intent": "normal",
            "task_type": "company_comparison",
        },
    ]
    for case in cases:
        out = research_plan_node(
            {
                "needs_tools": True,
                "task_type": case.get("task_type", "report_summary"),
                "evidence_plan": {"evidence_requirements": []},
                "canonical_intent": {},
                **case,
            }
        )

        assert out["research_plan_source"] == "deterministic_fallback"
        assert out["research_plan_fallback_reason"] == "deterministic_intent_shortcut"
        assert out["research_plan_duration_ms"] >= 0
        assert out["evidence_plan_used"]["strategy"] == "legacy_only"
