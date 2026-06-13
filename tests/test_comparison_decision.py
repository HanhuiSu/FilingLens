from __future__ import annotations

from src.agent.comparison_decision import build_comparison_judgment_frame


def test_comparison_judgment_frame_produces_conditional_preference():
    frame = build_comparison_judgment_frame(
        {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue", "period_end": "2025-12-31", "value": 120.0},
                {"evidence_id": "N2", "ticker": "AAPL", "metric": "net_income", "period_end": "2025-12-31", "value": 32.0},
                {"evidence_id": "N3", "ticker": "AMZN", "metric": "revenue", "period_end": "2025-12-31", "value": 150.0},
                {"evidence_id": "N4", "ticker": "AMZN", "metric": "net_income", "period_end": "2025-12-31", "value": 22.0},
            ],
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "AAPL", "section": "ITEM_7", "text_snippet": "Margin discipline remained strong."},
                {"evidence_id": "T2", "ticker": "AMZN", "section": "ITEM_1A", "text_snippet": "Competition and reinvestment remain important."},
            ],
            "grouped_risk_themes": [
                {"theme_code": "competition", "label": "Competition", "evidence_refs": ["T2"], "companies": ["AMZN"], "snippet_count": 1}
            ],
            "limitations": [{"code": "investment_advice_boundary"}],
            "missing_evidence_summary": {"overall_status": "partial", "degradation_reason": "numeric_only_comparison"},
        }
    ).model_dump()

    assert frame["preferred_company"] == "AAPL"
    assert frame["preference_type"] == "profitability"
    assert frame["profitability_winner"] == "AAPL"
    assert frame["scale_winner"] == "AMZN"
    assert frame["margin_winner"] == "AAPL"
    assert frame["growth_winner"] == "unavailable"
    assert frame["confidence_level"] in {"medium", "high"}
    assert "revenue" in frame["profitability_reason"]
    assert "net income" in frame["profitability_reason"]
    assert "net margin" in frame["profitability_reason"]
    assert "revenue scale" in frame["scale_reason"]
    assert frame["counterpoint"]
    assert frame["risk_tradeoff"]
    assert frame["evidence_basis"]
    assert "AAPL" in frame["rationale"]
    assert "AMZN" in frame["rationale"]


def test_comparison_judgment_frame_limits_risk_without_text_evidence():
    frame = build_comparison_judgment_frame(
        {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue", "period_end": "2025-12-31", "value": 120.0},
                {"evidence_id": "N2", "ticker": "AAPL", "metric": "net_income", "period_end": "2025-12-31", "value": 32.0},
                {"evidence_id": "N3", "ticker": "AMZN", "metric": "revenue", "period_end": "2025-12-31", "value": 150.0},
                {"evidence_id": "N4", "ticker": "AMZN", "metric": "net_income", "period_end": "2025-12-31", "value": 22.0},
            ],
            "text_snippets": [],
            "grouped_risk_themes": [],
            "limitations": [],
        }
    ).model_dump()

    assert frame["confidence_level"] == "low"
    assert "limited" in frame["risk_tradeoff"].lower()
    assert "Competition" not in frame["risk_tradeoff"]
    assert not any(item.get("dimension") == "risk" for item in frame["evidence_basis"])


def test_comparison_judgment_frame_uses_validated_text_risk_refs():
    frame = build_comparison_judgment_frame(
        {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue", "period_end": "2025-12-31", "value": 120.0},
                {"evidence_id": "N2", "ticker": "AAPL", "metric": "net_income", "period_end": "2025-12-31", "value": 32.0},
                {"evidence_id": "N3", "ticker": "AMZN", "metric": "revenue", "period_end": "2025-12-31", "value": 150.0},
                {"evidence_id": "N4", "ticker": "AMZN", "metric": "net_income", "period_end": "2025-12-31", "value": 22.0},
                {"evidence_id": "N5", "ticker": "AAPL", "metric": "net_margin", "period_end": "2025-12-31", "value": 0.2667, "unit": "ratio"},
                {"evidence_id": "N6", "ticker": "AMZN", "metric": "net_margin", "period_end": "2025-12-31", "value": 0.1467, "unit": "ratio"},
            ],
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "AAPL", "section": "ITEM_1A", "text_snippet": "Regulatory risks remain important."},
                {"evidence_id": "T2", "ticker": "AMZN", "section": "ITEM_1A", "text_snippet": "Competition remains important."},
            ],
            "grouped_risk_themes": [
                {"theme_code": "competition", "label": "Competition", "evidence_refs": ["T2"], "companies": ["AMZN"], "snippet_count": 1}
            ],
            "limitations": [],
        }
    ).model_dump()

    risk_basis = next(item for item in frame["evidence_basis"] if item["dimension"] == "risk")
    assert "Competition" in frame["risk_tradeoff"]
    assert risk_basis["evidence_refs"] == ["T2"]


def test_comparison_cash_flow_respects_requested_dimension():
    frame = build_comparison_judgment_frame(
        {
            "requested_dimensions": ["cash_flow_quality"],
            "active_dimensions": ["cash_flow_quality"],
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "MSFT", "metric": "operating_cash_flow", "period_end": "2025-12-31", "value": 120.0},
                {"evidence_id": "N2", "ticker": "MSFT", "metric": "free_cash_flow", "period_end": "2025-12-31", "value": 80.0},
                {"evidence_id": "N3", "ticker": "MSFT", "metric": "capital_expenditure", "period_end": "2025-12-31", "value": 40.0},
                {"evidence_id": "N4", "ticker": "MSFT", "metric": "fcf_margin", "period_end": "2025-12-31", "value": 0.25, "unit": "ratio"},
                {"evidence_id": "N5", "ticker": "AAPL", "metric": "operating_cash_flow", "period_end": "2025-12-31", "value": 90.0},
                {"evidence_id": "N6", "ticker": "AAPL", "metric": "free_cash_flow", "period_end": "2025-12-31", "value": 60.0},
                {"evidence_id": "N7", "ticker": "AAPL", "metric": "capital_expenditure", "period_end": "2025-12-31", "value": 30.0},
                {"evidence_id": "N8", "ticker": "AAPL", "metric": "fcf_margin", "period_end": "2025-12-31", "value": 0.18, "unit": "ratio"},
                {"evidence_id": "N9", "ticker": "AAPL", "metric": "net_margin", "period_end": "2025-12-31", "value": 0.4, "unit": "ratio"},
            ],
        }
    ).model_dump()

    assert frame["preference_type"] == "cash-flow quality"
    assert "free cash flow" in frame["rationale"].lower() or "operating cash flow" in frame["rationale"].lower()
    assert "net margin" not in frame["rationale"].lower()


def test_comparison_valuation_risk_respects_requested_dimension():
    frame = build_comparison_judgment_frame(
        {
            "requested_dimensions": ["valuation_and_risk_boundary"],
            "active_dimensions": ["valuation_and_risk_boundary"],
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "AAPL", "metric": "pe_ratio", "period_end": "2025-12-31", "value": 30.0, "unit": "ratio"},
                {"evidence_id": "N2", "ticker": "AAPL", "metric": "ps_ratio", "period_end": "2025-12-31", "value": 8.0, "unit": "ratio"},
                {"evidence_id": "N3", "ticker": "AAPL", "metric": "fcf_yield", "period_end": "2025-12-31", "value": 0.03, "unit": "ratio"},
                {"evidence_id": "N4", "ticker": "NVDA", "metric": "pe_ratio", "period_end": "2025-12-31", "value": 70.0, "unit": "ratio"},
                {"evidence_id": "N5", "ticker": "NVDA", "metric": "ps_ratio", "period_end": "2025-12-31", "value": 25.0, "unit": "ratio"},
                {"evidence_id": "N6", "ticker": "NVDA", "metric": "fcf_yield", "period_end": "2025-12-31", "value": 0.01, "unit": "ratio"},
            ],
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "AAPL", "text_snippet": "Demand risk."},
                {"evidence_id": "T2", "ticker": "NVDA", "text_snippet": "Regulatory risk."},
            ],
        }
    ).model_dump()

    assert frame["preference_type"] == "valuation risk"
    assert "p/e" in frame["rationale"].lower()
    assert "p/s" in frame["rationale"].lower()
    assert "ordinary risk-factor text" in frame["risk_tradeoff"].lower()
