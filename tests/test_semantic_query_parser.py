from datetime import date

from src.agent import query_plan
from src.agent.semantic_query_parser import SemanticQueryParserResult, build_semantic_query_proposal


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeParserClient:
    def __init__(self, payload: str):
        self.payload = payload

    def invoke(self, _messages):
        return _FakeMessage(self.payload)


def test_semantic_query_parser_returns_json_proposal_without_live_llm():
    result = build_semantic_query_proposal(
        "Analyze NVIDIA's cash flow quality, valuation boundary, and key risks",
        llm_client=_FakeParserClient(
            """
            {
              "company_mentions": ["NVIDIA"],
              "analysis_scope": "single_company",
              "methodology_intent": "cash_flow",
              "requested_dimensions": [
                "cash_flow_quality",
                "valuation_and_risk_boundary",
                "moat_and_competitive_risk"
              ],
              "requested_metrics": ["operating_cash_flow", "free_cash_flow", "pe_ratio"],
              "user_expectation": "diagnostic",
              "safety_intent": "normal",
              "time_scope": {"policy": "latest_available", "period_scope": "latest annual + latest quarterly", "is_explicit": false},
              "confidence": 0.86,
              "ambiguity": false,
              "needs_clarification": false,
              "reasons": ["The query requests cash flow quality, valuation, and risks."]
            }
            """
        ),
        mode="validated",
    )

    assert result.ok is True
    assert result.mode == "validated"
    assert result.proposal["company_mentions"] == ["NVIDIA"]
    assert result.proposal["requested_dimensions"] == [
        "cash_flow_quality",
        "valuation_and_risk_boundary",
        "moat_and_competitive_risk",
    ]
    assert result.proposal["requested_metrics"] == ["operating_cash_flow", "free_cash_flow", "pe_ratio"]
    assert result.diagnostics["parse_status"] == "validated"


def test_semantic_query_parser_off_mode_does_not_call_llm():
    class RaisingClient:
        def invoke(self, _messages):
            raise AssertionError("off mode should not call the parser")

    result = build_semantic_query_proposal("Analyze NVIDIA", llm_client=RaisingClient(), mode="off")

    assert result.ok is False
    assert result.source == "disabled"
    assert result.diagnostics["parse_status"] == "disabled"


def test_semantic_query_parser_rejects_extra_fields_and_invalid_labels():
    result = build_semantic_query_proposal(
        "Analyze NVIDIA",
        llm_client=_FakeParserClient(
            """
            {
              "company_mentions": ["NVIDIA"],
              "analysis_scope": "single_company",
              "methodology_intent": "stock_pick",
              "requested_dimensions": ["cash_flow_quality", "made_up_dimension"],
              "requested_metrics": ["fcf", "made_up_metric"],
              "user_expectation": "diagnostic",
              "safety_intent": "normal",
              "confidence": 0.9,
              "ambiguity": false,
              "needs_clarification": false,
              "reasons": ["test"],
              "tool_name": "search_filings"
            }
            """
        ),
        mode="shadow",
    )

    assert result.ok is False
    assert result.error == "proposal_schema_invalid"
    assert result.proposal == {}
    reasons = {item["reason"] for item in result.warnings}
    assert "schema_extra_forbidden" in reasons
    assert "methodology_intent_not_allowed" in reasons
    assert "dimension_not_supported" in reasons
    assert "metric_not_supported" in reasons
    assert result.diagnostics["parse_status"] == "proposal_schema_invalid"


def test_semantic_query_parser_rejects_malformed_json():
    result = build_semantic_query_proposal(
        "Analyze NVIDIA",
        llm_client=_FakeParserClient("not json"),
        mode="validated",
    )

    assert result.ok is False
    assert result.error == "invalid_json_response"
    assert result.proposal == {}


def test_validated_semantic_parser_mode_injects_proposal(monkeypatch):
    def fake_semantic_parser(_query, *, mode=None, **_kwargs):
        return SemanticQueryParserResult(
            mode=mode or "validated",
            ok=True,
            source="test",
            proposal={
                "company_mentions": ["NVIDIA"],
                "analysis_scope": "single_company",
                "methodology_intent": "cash_flow",
                "requested_dimensions": [
                    "cash_flow_quality",
                    "valuation_and_risk_boundary",
                    "moat_and_competitive_risk",
                ],
                "requested_metrics": ["operating_cash_flow", "free_cash_flow", "pe_ratio"],
                "user_expectation": "diagnostic",
                "safety_intent": "normal",
                "confidence": 0.9,
                "ambiguity": False,
                "needs_clarification": False,
                "reasons": ["test proposal"],
            },
        )

    monkeypatch.setattr(query_plan.settings, "semantic_query_parser_mode", "validated")
    monkeypatch.setattr(query_plan, "build_semantic_query_proposal", fake_semantic_parser)

    state = query_plan.build_classification_state(
        user_query="unseen wording that only the parser can map",
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="semantic-parser-injection",
        today=date(2026, 4, 27),
    )

    assert state["semantic_parser_mode"] == "validated"
    assert state["semantic_parser"]["ok"] is True
    assert state["query_understanding_summary"]["intent_source"] == "semantic_proposal_validated"
    assert state["companies"] == ["NVDA"]
    assert state["required_dimensions"] == [
        "cash_flow_quality",
        "valuation_and_risk_boundary",
        "moat_and_competitive_risk",
    ]
    assert "operating_cash_flow" in state["requested_metrics"]
    assert "pe_ratio" in state["requested_metrics"]
    assert state["semantic_parser"]["disagreement"]["injected"] is True


def test_shadow_semantic_parser_mode_logs_without_injecting(monkeypatch):
    def fake_semantic_parser(_query, *, mode=None, **_kwargs):
        return SemanticQueryParserResult(
            mode=mode or "shadow",
            ok=True,
            source="test",
            proposal={
                "company_mentions": ["Apple"],
                "analysis_scope": "single_company",
                "methodology_intent": "cash_flow",
                "requested_dimensions": ["cash_flow_quality"],
                "requested_metrics": ["free_cash_flow"],
                "user_expectation": "diagnostic",
                "safety_intent": "normal",
                "confidence": 0.9,
                "ambiguity": False,
                "needs_clarification": False,
                "reasons": ["shadow proposal"],
            },
        )

    monkeypatch.setattr(query_plan.settings, "semantic_query_parser_mode", "shadow")
    monkeypatch.setattr(query_plan, "build_semantic_query_proposal", fake_semantic_parser)

    state = query_plan.build_classification_state(
        user_query="Apple opaque phrasing only parser understands",
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="semantic-parser-shadow",
        today=date(2026, 4, 27),
    )

    assert state["semantic_parser_mode"] == "shadow"
    assert state["semantic_parser"]["ok"] is True
    assert state["semantic_parser"]["disagreement"]["injected"] is False
    assert state["query_understanding_summary"]["intent_source"] == "fallback_rules"
    assert state["query_understanding_summary"]["requested_dimensions"] == []
    assert state["required_dimensions"] == []
