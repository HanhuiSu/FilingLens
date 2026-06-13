"""Tests for methodology-v1 red flag diagnostics."""

from __future__ import annotations

from src.agent.red_flags import detect_red_flags, user_visible_red_flags


def _row(eid: str, ticker: str, metric: str, value: float, provider: str = "sec_companyfacts") -> dict:
    return {
        "evidence_id": eid,
        "ticker": ticker,
        "metric": metric,
        "value": value,
        "period_end": "2025-12-31",
        "source_provider": provider,
    }


def test_missing_evidence_red_flags_are_natural_language():
    flags = detect_red_flags(
        {"numeric_table": []},
        {
            "cash_flow_quality": {"status": "missing"},
            "valuation_and_risk_boundary": {"status": "missing"},
            "balance_sheet_and_capital_intensity": {"status": "missing"},
            "moat_and_competitive_risk": {"status": "missing"},
        },
    )

    messages = [flag.message for flag in flags]
    assert any("经营现金流/自由现金流" in message for message in messages)
    assert any("估值证据" in message for message in messages)
    assert any("现金/债务/资本开支" in message for message in messages)
    assert any("风险文本证据" in message for message in messages)
    assert "missing_cash_flow_evidence" not in "\n".join(messages)


def test_numeric_only_profitability_and_yfinance_flags():
    flags = detect_red_flags(
        {
            "numeric_table": [
                _row("N1", "AAPL", "revenue", 100),
                _row("N2", "AAPL", "net_income", 20),
                _row("N3", "AAPL", "net_margin", 0.2, provider="computed"),
                _row("N4", "AAPL", "adjusted_close", 10, provider="yfinance"),
            ]
        },
        {"profitability_quality": {"status": "partial"}},
    )

    ids = {flag.id for flag in flags}
    assert "numeric_only_profitability" in ids
    assert "yfinance_fallback_provider" in ids


def test_advantage_flags_use_thresholds():
    flags = detect_red_flags(
        {
            "numeric_table": [
                _row("N1", "AAPL", "revenue", 120),
                _row("N2", "AMZN", "revenue", 80),
                _row("N3", "AAPL", "net_margin", 0.25),
                _row("N4", "AMZN", "net_margin", 0.12),
            ]
        },
        {},
    )

    ids = {flag.id for flag in flags}
    assert "revenue_scale_advantage" in ids
    assert "profitability_margin_advantage" in ids
    visible = user_visible_red_flags(flags)
    assert all("id" not in item for item in visible)
    assert any("收入规模" in item["message"] for item in visible)
