from __future__ import annotations

from src.agent.metric_display import format_metric_value, metric_display_name, period_category


def test_valuation_multiples_are_not_percentages():
    assert format_metric_value("pe_ratio", 100.36, unit="ratio") == "100.36x"
    assert format_metric_value("ps_ratio", 63.29, unit="ratio") == "63.29x"
    assert "%" not in format_metric_value("pe_ratio", 100.36, unit="ratio")
    assert "%" not in format_metric_value("ps_ratio", 63.29, unit="ratio")


def test_currency_values_are_scaled_for_financial_statement_display():
    assert format_metric_value("market_cap", 4_311_463_935_165.4053, unit="USD") == "$4.31T"
    assert format_metric_value("net_debt", 435_000_000, unit="USD") == "$0.44B"


def test_yields_and_leverage_ratios_are_percentages():
    assert format_metric_value("fcf_yield", 0.0081, unit="ratio") == "0.81%"
    assert format_metric_value("debt_to_equity", 0.0702, unit="ratio") == "7.02%"


def test_share_price_and_labels_use_metric_semantics():
    assert format_metric_value("share_price", 142.75, unit="USD") == "$142.75"
    assert metric_display_name("cash_and_equivalents", "zh") == "现金及等价物"
    assert period_category("latest") == "point_in_time"
    assert period_category("quarterly") == "quarterly"
