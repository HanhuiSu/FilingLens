from __future__ import annotations

from src.agent.analyst_loop import run_analyst_draft_loop


class FakeLLMSequence:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)

    def invoke(self, _messages):
        content = self.contents.pop(0) if self.contents else self.contents[-1]

        class Response:
            def __init__(self, content: str):
                self.content = content

        return Response(content)


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
        "grouped_risk_themes": [
            {"theme_code": "competition", "label": "Competition", "evidence_refs": ["T2"], "companies": ["AMZN"], "snippet_count": 1}
        ],
        "grouped_business_themes": [],
        "provenance_notes": [],
        "missing_evidence_summary": {"overall_status": "sufficient"},
        "limitations": [],
        "citations": [{"evidence_id": "N1"}, {"evidence_id": "N2"}, {"evidence_id": "N3"}, {"evidence_id": "N4"}, {"evidence_id": "T1"}, {"evidence_id": "T2"}],
    }


def test_revision_loop_accepts_soft_warnings_without_extra_revision(monkeypatch):
    llm = FakeLLMSequence(
        [
            '{"tentative_conclusion":{"statement":"If profitability matters more, AAPL currently looks stronger.","stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2"]},"decision_basis":[{"statement":"AAPL net income is higher.","citation_refs":["N2","N4"]}],"supporting_points":[],"counterpoints":[],"risk_tradeoffs":[],"uncertainty_notes":[],"citation_refs":["N2","N4"],"safety_notes":[]}',
            '{"tentative_conclusion":{"statement":"If profitability matters more, AAPL currently looks stronger, while AMZN is still stronger on revenue scale.","stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2","N3"]},"decision_basis":[{"statement":"AAPL net income is higher.","citation_refs":["N2","N4"]}],"supporting_points":[{"statement":"AAPL currently looks stronger on profitability quality.","citation_refs":["N2","N4"]}],"counterpoints":[{"statement":"AMZN still has larger revenue scale.","citation_refs":["N3"]}],"risk_tradeoffs":[],"uncertainty_notes":[],"citation_refs":["N2","N3","N4"],"safety_notes":[{"statement":"This is not investment advice.","citation_refs":[]}]}',
        ]
    )
    monkeypatch.setattr("src.agent.analyst_draft._get_llm", lambda *args, **kwargs: llm)

    out = run_analyst_draft_loop(
        evidence_packet=_packet(),
        answer_language="English",
        synthesis_mode="limited_judgment",
        safety_policy={"answer_mode": "comparison_brief", "safety_intent": "investment_advice_like"},
        comparison_judgment_frame={"preferred_company": "AAPL"},
        max_attempts=2,
    )

    assert out["draft_final_status"] == "passed"
    assert len(out["attempts"]) == 1
    assert out["validation"]["passed"] is True
    assert out["validation"]["final_status"] == "passed_with_warnings"


def test_revision_loop_falls_back_on_severe_boundary_violations(monkeypatch):
    llm = FakeLLMSequence(
        [
            '{"tentative_conclusion":{"statement":"You should buy AAPL because it will rise tomorrow.","stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2"]},"decision_basis":[{"statement":"AAPL net income is higher.","citation_refs":["N2","N4"]}],"supporting_points":[],"counterpoints":[],"risk_tradeoffs":[],"uncertainty_notes":[],"citation_refs":["N2","N4"],"safety_notes":[]}',
            '{"tentative_conclusion":{"statement":"You should still buy AAPL.","stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2"]},"decision_basis":[{"statement":"AAPL net income is higher.","citation_refs":["N2","N4"]}],"supporting_points":[],"counterpoints":[],"risk_tradeoffs":[],"uncertainty_notes":[],"citation_refs":["N2","N4"],"safety_notes":[]}',
        ]
    )
    monkeypatch.setattr("src.agent.analyst_draft._get_llm", lambda *args, **kwargs: llm)

    out = run_analyst_draft_loop(
        evidence_packet=_packet(),
        answer_language="English",
        synthesis_mode="limited_judgment",
        safety_policy={"answer_mode": "comparison_brief", "safety_intent": "investment_advice_like"},
        comparison_judgment_frame={"preferred_company": "AAPL"},
        max_attempts=2,
    )

    assert out["draft_final_status"] == "fallback"
    assert out["validation"]["passed"] is False
    assert len(out["attempts"]) == 2
