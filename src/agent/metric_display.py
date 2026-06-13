"""User-facing financial metric display semantics.

Formatting is intentionally metric-driven. A ratio unit alone is not enough:
P/E and P/S are multiples, while margins and yields are percentages.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Literal

from src.agent.metric_availability import normalize_metric_name

UnitType = Literal["currency", "currency_per_share", "percentage", "multiple", "shares", "raw"]


@dataclass(frozen=True)
class MetricDisplaySemantics:
    metric: str
    display_name_zh: str
    display_name_en: str
    unit_type: UnitType
    precision: int = 2


_SEMANTICS: dict[str, MetricDisplaySemantics] = {
    "revenue": MetricDisplaySemantics("revenue", "收入", "Revenue", "currency"),
    "revenue_growth": MetricDisplaySemantics("revenue_growth", "收入增长", "Revenue Growth", "percentage"),
    "net_income": MetricDisplaySemantics("net_income", "净利润", "Net Income", "currency"),
    "gross_profit": MetricDisplaySemantics("gross_profit", "毛利润", "Gross Profit", "currency"),
    "operating_income": MetricDisplaySemantics("operating_income", "营业利润", "Operating Income", "currency"),
    "consolidated_operating_income": MetricDisplaySemantics(
        "consolidated_operating_income",
        "合并营业利润",
        "Consolidated Operating Income",
        "currency",
    ),
    "aws_operating_income": MetricDisplaySemantics(
        "aws_operating_income",
        "AWS营业利润",
        "AWS Operating Income",
        "currency",
    ),
    "aws_revenue": MetricDisplaySemantics("aws_revenue", "AWS收入", "AWS Revenue", "currency"),
    "gross_margin": MetricDisplaySemantics("gross_margin", "毛利率", "Gross Margin", "percentage"),
    "operating_margin": MetricDisplaySemantics("operating_margin", "营业利润率", "Operating Margin", "percentage"),
    "net_margin": MetricDisplaySemantics("net_margin", "净利率", "Net Margin", "percentage"),
    "operating_cash_flow": MetricDisplaySemantics(
        "operating_cash_flow",
        "经营现金流",
        "Operating Cash Flow",
        "currency",
    ),
    "free_cash_flow": MetricDisplaySemantics("free_cash_flow", "自由现金流", "Free Cash Flow", "currency"),
    "capital_expenditure": MetricDisplaySemantics("capital_expenditure", "资本开支", "Capital Expenditure", "currency"),
    "cash": MetricDisplaySemantics("cash", "现金及等价物", "Cash and Equivalents", "currency"),
    "total_debt": MetricDisplaySemantics("total_debt", "总债务", "Total Debt", "currency"),
    "net_debt": MetricDisplaySemantics("net_debt", "净债务", "Net Debt", "currency"),
    "total_assets": MetricDisplaySemantics("total_assets", "总资产", "Total Assets", "currency"),
    "total_liabilities": MetricDisplaySemantics("total_liabilities", "总负债", "Total Liabilities", "currency"),
    "shareholders_equity": MetricDisplaySemantics(
        "shareholders_equity",
        "股东权益",
        "Shareholders' Equity",
        "currency",
    ),
    "receivables": MetricDisplaySemantics("receivables", "应收款", "Receivables", "currency"),
    "inventory": MetricDisplaySemantics("inventory", "存货", "Inventory", "currency"),
    "share_price": MetricDisplaySemantics("share_price", "股价", "Share Price", "currency_per_share"),
    "market_cap": MetricDisplaySemantics("market_cap", "市值", "Market Cap", "currency"),
    "pe_ratio": MetricDisplaySemantics("pe_ratio", "P/E", "P/E", "multiple"),
    "ps_ratio": MetricDisplaySemantics("ps_ratio", "P/S", "P/S", "multiple"),
    "fcf_yield": MetricDisplaySemantics("fcf_yield", "FCF yield", "FCF Yield", "percentage"),
    "segment_profit_contribution": MetricDisplaySemantics(
        "segment_profit_contribution",
        "分部利润贡献率",
        "Segment Profit Contribution",
        "percentage",
    ),
    "debt_to_equity": MetricDisplaySemantics("debt_to_equity", "债务/权益", "Debt/Equity", "percentage"),
    "capex_to_revenue": MetricDisplaySemantics("capex_to_revenue", "资本开支/收入", "Capex/Revenue", "percentage"),
    "cash_conversion": MetricDisplaySemantics("cash_conversion", "现金转换率", "Cash Conversion", "percentage"),
    "cfo_to_net_income": MetricDisplaySemantics("cfo_to_net_income", "CFO/净利润", "CFO/Net Income", "percentage"),
    "fcf_margin": MetricDisplaySemantics("fcf_margin", "自由现金流率", "FCF Margin", "percentage"),
    "receivables_to_revenue": MetricDisplaySemantics(
        "receivables_to_revenue",
        "应收款/收入",
        "Receivables/Revenue",
        "percentage",
    ),
    "inventory_to_revenue": MetricDisplaySemantics(
        "inventory_to_revenue",
        "存货/收入",
        "Inventory/Revenue",
        "percentage",
    ),
    "shares_outstanding": MetricDisplaySemantics("shares_outstanding", "流通股数", "Shares Outstanding", "shares"),
}


def metric_semantics(metric: str) -> MetricDisplaySemantics:
    canonical = normalize_metric_name(str(metric or ""))
    if canonical in {"adjusted_close", "latest_close", "close", "price"}:
        canonical = "share_price"
    if canonical.startswith("post_return_"):
        return MetricDisplaySemantics(canonical, canonical, canonical, "percentage")
    if canonical.endswith("_margin") or canonical.endswith("_yield") or canonical.endswith("_rate"):
        return _SEMANTICS.get(canonical, MetricDisplaySemantics(canonical, canonical, canonical, "percentage"))
    return _SEMANTICS.get(
        canonical,
        MetricDisplaySemantics(canonical or str(metric or ""), str(metric or ""), str(metric or ""), "raw"),
    )


def metric_display_name(metric: str, lang: str = "zh") -> str:
    semantics = metric_semantics(metric)
    return semantics.display_name_zh if lang == "zh" else semantics.display_name_en


def period_category(period_type: Any) -> str:
    raw = str(period_type or "").strip().lower()
    if raw in {"annual", "fy", "year", "yearly"}:
        return "annual"
    if raw in {"quarterly", "quarter", "q"}:
        return "quarterly"
    if raw in {"ttm", "trailing", "trailing_twelve_months"}:
        return "ttm"
    if raw in {"daily", "latest", "as_of", "point_in_time", "computed"}:
        return "point_in_time"
    return raw or "point_in_time"


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_scaled_currency(value: float, currency: str, precision: int) -> str:
    sign = "-" if value < 0 else ""
    amount = abs(value)
    def scaled_text(denominator: int) -> str:
        scaled = (Decimal(str(amount)) / Decimal(denominator)).quantize(
            Decimal("1." + ("0" * precision)),
            rounding=ROUND_HALF_UP,
        )
        return f"{scaled:.{precision}f}"

    if amount >= 1_000_000_000_000:
        return f"{sign}{currency}{scaled_text(1_000_000_000_000)}T"
    if amount >= 100_000_000:
        return f"{sign}{currency}{scaled_text(1_000_000_000)}B"
    if amount >= 1_000_000:
        return f"{sign}{currency}{scaled_text(1_000_000)}M"
    return f"{sign}{currency}{amount:,.{precision}f}"


def _format_shares(value: float, precision: int) -> str:
    sign = "-" if value < 0 else ""
    amount = abs(value)
    if amount >= 1_000_000_000:
        return f"{sign}{amount / 1_000_000_000:.{precision}f}B shares"
    if amount >= 1_000_000:
        return f"{sign}{amount / 1_000_000:.{precision}f}M shares"
    return f"{sign}{amount:,.{precision}f} shares"


def format_metric_value(metric: str, value: Any, unit: str | None = None, currency: str = "$") -> str:
    semantics = metric_semantics(metric)
    numeric = _to_float(value)
    if numeric is None:
        return "N/A" if value in {None, ""} else str(value)
    precision = semantics.precision
    if semantics.unit_type == "currency":
        return _format_scaled_currency(numeric, currency, precision)
    if semantics.unit_type == "currency_per_share":
        return f"{currency}{numeric:,.{precision}f}"
    if semantics.unit_type == "percentage":
        return f"{numeric * 100:.{precision}f}%"
    if semantics.unit_type == "multiple":
        return f"{numeric:.{precision}f}x"
    if semantics.unit_type == "shares":
        return _format_shares(numeric, precision)
    normalized_unit = str(unit or "").lower()
    if normalized_unit in {"usd", "$"}:
        return _format_scaled_currency(numeric, currency, precision)
    return str(value)
