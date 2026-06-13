from datetime import date

from src.agent.entity_resolution import ResolvedCompany
from src.agent.evidence_sufficiency import evaluate_evidence_sufficiency
from src.agent.query_plan import build_classification_state, build_query_plan
from src.agent.query_understanding import QueryUnderstanding


def _state(query: str):
    return build_classification_state(
        user_query=query,
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="test-query-understanding",
        today=date(2026, 4, 27),
    )


def test_open_company_analysis_plan_derives_from_query_understanding():
    state = _state("分析下 nvidai")
    assert state["query_understanding_summary"]["methodology_intent"] == "overview"
    assert state["companies"] == ["NVDA"]
    assert state["task_type"] == "report_summary"
    assert state["answer_mode"] == "analytical"
    assert state["analysis_scope"] == "single_company"
    assert state["methodology_intent"] == "single_company_overview"
    assert state["time_policy"] == "latest_available"
    assert not state["needs_clarification"]


def test_risk_question_plan_uses_risk_focused_mode():
    state = _state("nvidai 最大的问题是什么")
    assert state["query_understanding_summary"]["methodology_intent"] == "risk"
    assert state["companies"] == ["NVDA"]
    assert state["answer_mode"] == "risk_focused_analysis"
    assert state["methodology_intent"] == "risk_focused_analysis"


def test_next_quarter_risk_question_uses_risk_focused_mode():
    state = _state("你分析一下下一个季度亚马逊的风险有什么？")
    assert state["query_understanding_summary"]["methodology_intent"] == "risk"
    assert state["companies"] == ["AMZN"]
    assert state["answer_mode"] == "risk_focused_analysis"
    assert state["methodology_intent"] == "risk_focused_analysis"
    assert state["required_dimensions"] == ["moat_and_competitive_risk"]
    assert state["canonical_intent"]["intent_family"] == "risk"
    assert state["evidence_policy_id"] == "single_company_risk_v1"


def test_chinese_risk_paraphrases_share_canonical_intent_and_answer_mode():
    queries = [
        "亚马逊下季度有什么风险",
        "AMZN 接下来最需要担心什么",
        "下个季度亚马逊风险点在哪里",
        "亚马逊最大隐患是什么",
    ]

    for query in queries:
        state = _state(query)
        assert state["companies"] == ["AMZN"]
        assert state["canonical_intent"]["intent_family"] == "risk"
        assert state["canonical_intent"]["analysis_scope"] == "single_company"
        assert state["answer_mode"] == "risk_focused_analysis"
        assert state["evidence_policy_id"] == "single_company_risk_v1"


def test_composite_single_company_query_keeps_explicit_dimensions():
    state = _state("分析英伟达的现金流质量、估值边界和主要风险")

    assert state["query_understanding_summary"]["requested_dimensions"] == [
        "cash_flow_quality",
        "valuation_and_risk_boundary",
        "moat_and_competitive_risk",
    ]
    assert state["companies"] == ["NVDA"]
    assert state["answer_mode"] == "analytical"
    assert state["analysis_scope"] == "single_company"
    assert state["evidence_policy_id"] == "single_company_composite_v1"
    assert state["required_dimensions"] == [
        "cash_flow_quality",
        "valuation_and_risk_boundary",
        "moat_and_competitive_risk",
    ]
    assert state["optional_dimensions"] == []
    assert state["selected_analysis_framework"]["active_dimension_ids"] == [
        "cash_flow_quality",
        "valuation_and_risk_boundary",
        "moat_and_competitive_risk",
    ]
    requirement_metrics = {
        (req["dimension_id"], req["requirement_type"], req.get("metric"), req["required"])
        for req in state["evidence_plan"]["evidence_requirements"]
    }
    core_dimensions = {
        req["dimension_id"]
        for req in state["evidence_plan"]["evidence_requirements"]
        if req.get("required", True) and req.get("requirement_scope") == "core"
    }
    assert core_dimensions == {
        "cash_flow_quality",
        "valuation_and_risk_boundary",
        "moat_and_competitive_risk",
    }
    assert ("cash_flow_quality", "numeric", "operating_cash_flow", True) in requirement_metrics
    assert ("cash_flow_quality", "numeric", "free_cash_flow", True) in requirement_metrics
    assert ("valuation_and_risk_boundary", "numeric", "price", True) in requirement_metrics
    assert ("valuation_and_risk_boundary", "calculation", "market_cap", True) in requirement_metrics
    risk_text_requirements = [
        req
        for req in state["evidence_plan"]["evidence_requirements"]
        if req["dimension_id"] == "moat_and_competitive_risk" and req["requirement_type"] == "text" and req["required"]
    ]
    assert risk_text_requirements
    assert any("ITEM_1A" in req.get("primary_sections", []) + req.get("fallback_sections", []) for req in risk_text_requirements)
    assert any(
        req["dimension_id"] == "moat_and_competitive_risk" and req["requirement_type"] == "text" and req["required"]
        for req in state["evidence_plan"]["evidence_requirements"]
    )
    collection_results = [
        {"requirement_id": req["requirement_id"], "status": "satisfied", "items": [{"evidence_id": req["requirement_id"]}]}
        for req in state["evidence_plan"]["evidence_requirements"]
        if req.get("required", True)
    ]
    sufficiency = evaluate_evidence_sufficiency(state["evidence_plan"], collection_results).model_dump()
    assert sufficiency["overall_status"] == "sufficient"
    required_ids = {
        req["requirement_id"]
        for req in state["evidence_plan"]["evidence_requirements"]
        if req.get("required", True)
    }
    assert not (set(sufficiency["missing_requirements"]) & required_ids)


def test_comparison_plan_preserves_investment_advice_boundary():
    state = _state("apple 和 amazon 更看好哪个")
    assert state["query_understanding_summary"]["analysis_scope"] == "comparison"
    assert state["companies"] == ["AAPL", "AMZN"]
    assert state["task_type"] == "company_comparison"
    assert state["answer_mode"] == "comparison_brief"
    assert state["safety_intent"] == "investment_advice_like"
    assert state["methodology_intent"] == "investment_advice_like"


def test_build_query_plan_uses_query_understanding_summary_when_present():
    state = _state("苹果现金流怎么样")
    plan = build_query_plan(state)
    assert plan.analysis_scope == "single_company"
    assert plan.methodology_intent == "cash_flow_quality_analysis"
    assert plan.answer_mode == "analytical"


def test_semantic_proposal_can_drive_final_planning_without_owning_tools():
    state = build_classification_state(
        user_query="苹果利润增长是不是有质量",
        parsed={
            "task_type": "fact_qa",
            "companies": [],
            "data_route": "hybrid",
            "query_understanding_proposal": {
                "company_mentions": ["Apple"],
                "analysis_scope": "single_company",
                "methodology_intent": "profitability",
                "requested_dimensions": ["profitability_quality", "cash_flow_quality"],
                "user_expectation": "diagnostic",
                "safety_intent": "normal",
                "time_scope": {
                    "policy": "latest_available",
                    "period_scope": "latest annual + latest quarterly",
                    "is_explicit": False,
                },
                "ambiguity": False,
                "needs_clarification": False,
                "confidence": 0.86,
                "reasons": ["Semantic parser maps profit quality wording to profitability analysis."],
            },
            "analysis_plan": {
                "proposed_tools": ["web_search"],
                "metric_requirements": ["magic_metric"],
            },
        },
        trace_id="semantic-proposal-plan",
        today=date(2026, 4, 27),
    )

    assert state["companies"] == ["AAPL"]
    assert state["query_understanding_summary"]["intent_source"] == "semantic_proposal_validated"
    assert state["task_type"] == "report_summary"
    assert state["answer_mode"] == "analytical"
    assert state["analysis_scope"] == "single_company"
    assert state["methodology_intent"] == "profitability_quality_analysis"
    assert state["required_dimensions"] == ["profitability_quality", "cash_flow_quality"]
    assert "search_filings" in state["validated_tools"]
    assert "query_financial_data" in state["validated_tools"]
    assert "web_search" not in state["validated_tools"]
    assert any(item["reason"] == "tool_not_allowed" for item in state["rejected_plan_items"])
    assert any(item["reason"] == "metric_not_allowed" for item in state["rejected_plan_items"])
    assert state["analysis_plan"]["requested_dimensions"] == ["profitability_quality", "cash_flow_quality"]
    assert state["analysis_plan"]["supporting_context_dimensions"] == []


def _understanding(
    *,
    tickers: list[str],
    methodology_intent: str,
    legacy_methodology_intent: str,
    analysis_scope: str,
    safety_intent: str = "normal",
    legacy_safety_intent: str = "normal",
    needs_clarification: bool = False,
) -> QueryUnderstanding:
    return QueryUnderstanding(
        raw_query="neutral planning request",
        normalized_query="neutral planning request",
        companies=[ResolvedCompany(ticker=ticker, canonical_name=ticker) for ticker in tickers],
        analysis_scope=analysis_scope,
        methodology_intent=methodology_intent,
        legacy_methodology_intent=legacy_methodology_intent,
        safety_intent=safety_intent,
        legacy_safety_intent=legacy_safety_intent,
        time_scope={
            "policy": "latest_available" if analysis_scope == "single_company" else "",
            "period_scope": "latest annual + latest quarterly" if analysis_scope == "single_company" else "",
            "is_explicit": False,
        },
        needs_clarification=needs_clarification,
        confidence=0.99,
        intent_source="test_structured_understanding",
    )


def test_risk_plan_prefers_query_understanding_over_raw_query(monkeypatch):
    monkeypatch.setattr(
        "src.agent.query_plan.build_query_understanding",
        lambda *_args, **_kwargs: _understanding(
            tickers=["AAPL"],
            methodology_intent="risk",
            legacy_methodology_intent="risk_focused_analysis",
            analysis_scope="single_company",
        ),
    )

    state = _state("neutral planning request")

    assert state["companies"] == ["AAPL"]
    assert state["task_type"] == "report_summary"
    assert state["answer_mode"] == "risk_focused_analysis"
    assert state["analysis_scope"] == "single_company"
    assert state["methodology_intent"] == "risk_focused_analysis"


def test_comparison_plan_prefers_query_understanding_over_raw_query(monkeypatch):
    monkeypatch.setattr(
        "src.agent.query_plan.build_query_understanding",
        lambda *_args, **_kwargs: _understanding(
            tickers=["AAPL", "AMZN"],
            methodology_intent="comparison",
            legacy_methodology_intent="company_comparison",
            analysis_scope="comparison",
            safety_intent="investment_advice_like",
            legacy_safety_intent="investment_advice_like",
        ),
    )

    state = _state("neutral planning request")

    assert state["companies"] == ["AAPL", "AMZN"]
    assert state["comparison_target"] == "AMZN"
    assert state["task_type"] == "company_comparison"
    assert state["answer_mode"] == "comparison_brief"
    assert state["analysis_scope"] == "comparison"
    assert state["safety_intent"] == "investment_advice_like"
    assert state["methodology_intent"] == "investment_advice_like"


def test_query_understanding_unknown_falls_back_to_legacy_raw_query_parsing(monkeypatch):
    monkeypatch.setattr(
        "src.agent.query_plan.build_query_understanding",
        lambda *_args, **_kwargs: _understanding(
            tickers=[],
            methodology_intent="none",
            legacy_methodology_intent="",
            analysis_scope="unknown",
        ),
    )

    state = _state("苹果现金流怎么样")

    assert state["companies"] == ["AAPL"]
    assert state["analysis_scope"] == "single_company"
    assert state["methodology_intent"] == "cash_flow_quality_analysis"
    assert state["answer_mode"] == "analytical"
