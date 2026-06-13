"""query_financial_data — look up structured financial facts and price history."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

import duckdb
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import settings
from src.db.schema import init_db

VALID_METRICS = {
    "revenue",
    "revenue_growth",
    "net_income",
    "eps",
    "gross_profit",
    "operating_income",
    "gross_margin",
    "operating_margin",
    "operating_cash_flow",
    "free_cash_flow",
    "capital_expenditure",
    "cash_and_equivalents",
    "short_term_debt",
    "long_term_debt",
    "total_debt",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "inventory",
    "receivables",
    "shares_outstanding",
    "cfo_to_net_income",
    "cash_conversion",
    "fcf_margin",
    "net_debt",
    "debt_to_equity",
    "capex_to_revenue",
    "receivables_to_revenue",
    "inventory_to_revenue",
    "market_cap",
    "pe_ratio",
    "ps_ratio",
    "fcf_yield",
}

PRICE_METRICS = {"open", "high", "low", "close", "adjusted_close", "volume"}
PRICE_ALIASES = {
    "price": "adjusted_close",
    "stock_price": "adjusted_close",
    "share_price": "adjusted_close",
}

PROVIDER_PRIORITY = {
    "sec_companyfacts": 0,
    "yfinance": 1,
}


class QueryFinancialDataInput(BaseModel):
    """Input schema for the query_financial_data tool."""

    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL")
    metrics: list[str] = Field(
        description=(
            "Financial metrics to query. "
            f"Fundamental: {sorted(VALID_METRICS)}. "
            f"Price: {sorted(PRICE_METRICS)}. "
            "Can mix both types."
        )
    )
    period_type: Optional[Literal["quarterly", "annual", "latest", "trailing"]] = Field(
        default=None,
        description=(
            "Time period mode for fundamentals. "
            "'quarterly'/'annual' = exact period type filtering; "
            "'latest' = latest row(s) under period filters; "
            "'trailing' = latest N rows under period filters."
        ),
    )
    target_period_type: Optional[Literal["quarterly", "annual"]] = Field(
        default=None,
        description="When period_type is latest/trailing, choose the underlying granularity.",
    )
    year: int | None = Field(
        default=None,
        description="Target year for period matching (fiscal or calendar based on year_basis).",
    )
    quarter: Optional[Literal[1, 2, 3, 4]] = Field(
        default=None,
        description="Target quarter index when querying quarterly data.",
    )
    trailing_n: int | None = Field(
        default=None,
        description="For period_type=trailing, number of latest periods to return (default 4, max 16).",
    )
    year_basis: Optional[Literal["fiscal", "calendar"]] = Field(
        default=None,
        description="Interpretation basis for year/quarter filtering.",
    )
    comparison_basis: Optional[Literal["same_period", "latest_fiscal_year", "calendar_year"]] = Field(
        default=None,
        description="Comparison basis marker, returned in period_context for traceability.",
    )
    strict_period_match: bool = Field(
        default=True,
        description="If true, do not relax period filters when no exact match exists.",
    )
    date_start: Optional[str] = Field(
        default=None,
        description="Only include data on or after this date (YYYY-MM-DD)",
    )
    date_end: Optional[str] = Field(
        default=None,
        description="Only include data on or before this date (YYYY-MM-DD)",
    )
    limit: int = Field(
        default=20,
        description="Maximum number of rows to return per metric for fundamentals, "
        "or total rows for price data (default 20, max 100)",
    )


def _normalize_metric(raw: str) -> str | None:
    """Map user-supplied metric name to canonical form (case-insensitive)."""
    lower = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if lower in VALID_METRICS:
        return lower
    _aliases: dict[str, str] = {
        "price_to_sales": "ps_ratio",
        "p_s_ratio": "ps_ratio",
        "total_revenue": "revenue",
        "revenues": "revenue",
        "netincome": "net_income",
        "net_profit": "net_income",
        "grossprofit": "gross_profit",
        "operatingincome": "operating_income",
        "operating_profit": "operating_income",
        "earnings_per_share": "eps",
        "grossmargin": "gross_margin",
        "operatingmargin": "operating_margin",
        "capex": "capital_expenditure",
        "capital_expenditures": "capital_expenditure",
        "cash": "cash_and_equivalents",
        "cash_equivalents": "cash_and_equivalents",
        "cash_and_cash_equivalents": "cash_and_equivalents",
        "debt": "total_debt",
        "short_debt": "short_term_debt",
        "current_debt": "short_term_debt",
        "long_debt": "long_term_debt",
        "assets": "total_assets",
        "liabilities": "total_liabilities",
        "equity": "shareholders_equity",
        "stockholders_equity": "shareholders_equity",
        "operating_cashflow": "operating_cash_flow",
        "operating_cash_flows": "operating_cash_flow",
        "ocf": "operating_cash_flow",
        "cfo": "operating_cash_flow",
        "fcf": "free_cash_flow",
        "free_cashflow": "free_cash_flow",
        "cfo_to_ni": "cfo_to_net_income",
        "cfo_net_income": "cfo_to_net_income",
        "cash_conversion_ratio": "cash_conversion",
        "netdebt": "net_debt",
        "debt_equity": "debt_to_equity",
        "debt_to_equity_ratio": "debt_to_equity",
        "capex_revenue": "capex_to_revenue",
        "receivables_revenue": "receivables_to_revenue",
        "inventory_revenue": "inventory_to_revenue",
        "accounts_receivable": "receivables",
        "receivables_net": "receivables",
        "shares": "shares_outstanding",
        "diluted_shares": "shares_outstanding",
        "price_to_earnings": "pe_ratio",
        "p_e_ratio": "pe_ratio",
        "free_cash_flow_yield": "fcf_yield",
    }
    return _aliases.get(lower)


def _normalize_price_metric(raw: str) -> str | None:
    lower = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if lower in PRICE_METRICS:
        return lower
    return PRICE_ALIASES.get(lower)


def _infer_fiscal_year_end_month(conn: duckdb.DuckDBPyConnection, ticker: str) -> int | None:
    row = conn.execute(
        """
        SELECT EXTRACT(MONTH FROM period_end)::INT AS m
        FROM financial_facts
        WHERE ticker = ? AND period_type = 'annual'
        ORDER BY period_end DESC
        LIMIT 1
        """,
        [ticker.upper()],
    ).fetchone()
    if not row:
        return None
    m = int(row[0])
    if 1 <= m <= 12:
        return m
    return None


def _annotate_period_fields(
    period_end: date,
    fiscal_year_end_month: int | None,
) -> dict[str, Any]:
    month = int(period_end.month)
    cal_year = int(period_end.year)
    cal_quarter = (month - 1) // 3 + 1
    fy: int | None = None
    fq: int | None = None
    if fiscal_year_end_month:
        fy = cal_year + (1 if month > fiscal_year_end_month else 0)
        fq = ((month - fiscal_year_end_month - 1) % 12) // 3 + 1
    return {
        "calendar_year": cal_year,
        "calendar_quarter": cal_quarter,
        "fiscal_year": fy,
        "fiscal_quarter": fq,
    }


def _relative_mismatch(sec_value: float, other_value: float, tolerance_pct: float) -> bool:
    denominator = max(abs(sec_value), 1e-9)
    return abs(sec_value - other_value) / denominator > tolerance_pct


def _values_conflict(metric: str, sec_value: Any, other_value: Any) -> bool:
    try:
        sec_num = float(sec_value)
        other_num = float(other_value)
    except (TypeError, ValueError):
        return False
    if metric in {
        "gross_margin",
        "operating_margin",
        "net_margin",
        "fcf_margin",
        "cfo_to_net_income",
        "cash_conversion",
        "debt_to_equity",
        "capex_to_revenue",
        "receivables_to_revenue",
        "inventory_to_revenue",
        "fcf_yield",
    }:
        return abs(sec_num - other_num) > 0.005
    if metric == "eps":
        return _relative_mismatch(sec_num, other_num, 0.02)
    return _relative_mismatch(sec_num, other_num, 0.01)


def _source_priority(row: dict[str, Any]) -> tuple[int, str, str]:
    provider = str(row.get("source_provider") or "yfinance")
    return (
        PROVIDER_PRIORITY.get(provider, 99),
        str(row.get("filing_date") or ""),
        str(row.get("source_tag") or ""),
    )


def _apply_source_priority_and_reconciliation(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    provider_counts: dict[str, int] = {}
    for row in rows:
        provider = str(row.get("source_provider") or "yfinance")
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        key = (
            str(row.get("ticker", "")),
            str(row.get("metric", "")),
            str(row.get("period_type", "")),
            str(row.get("period_end", "")),
        )
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    conflict_count = 0
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=_source_priority)
        picked = dict(ordered[0])
        sec_rows = [r for r in group_rows if str(r.get("source_provider") or "") == "sec_companyfacts"]
        yf_rows = [r for r in group_rows if str(r.get("source_provider") or "yfinance") == "yfinance"]
        if sec_rows and yf_rows:
            sec = sorted(sec_rows, key=_source_priority)[0]
            metric = str(sec.get("metric", ""))
            if any(_values_conflict(metric, sec.get("value"), yf.get("value")) for yf in yf_rows):
                conflict_count += 1
                picked["reconciliation_warning"] = "sec_yfinance_value_mismatch"
        selected.append(picked)

    source_summary = {
        "provider_counts": provider_counts,
        "sec_fact_count": sum(1 for row in selected if str(row.get("source_provider") or "") == "sec_companyfacts"),
        "fallback_yfinance_fact_count": sum(
            1 for row in selected if str(row.get("source_provider") or "yfinance") == "yfinance"
        ),
        "conflict_count": conflict_count,
        "conflict_rate": conflict_count / max(len(selected), 1),
        "uses_yfinance_fallback": any(str(row.get("source_provider") or "yfinance") == "yfinance" for row in selected),
        "has_reconciliation_warning": conflict_count > 0
        or any(str(row.get("reconciliation_warning") or "").strip() for row in selected),
    }
    return selected, source_summary


def _query_facts(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    metrics: list[str],
    period_type: str | None,
    target_period_type: str | None,
    year: int | None,
    quarter: int | None,
    trailing_n: int | None,
    year_basis: str | None,
    comparison_basis: str | None,
    strict_period_match: bool,
    date_start: str | None,
    date_end: str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_list = []
    for m in metrics:
        normalized = _normalize_metric(m)
        if normalized and normalized not in metric_list:
            metric_list.append(normalized)
    if not metric_list:
        return [], {
            "applied_filters": {},
            "year_basis": year_basis or "fiscal",
            "comparison_basis": comparison_basis or "same_period",
            "resolved_periods": [],
            "source_summary": {
                "provider_counts": {},
                "sec_fact_count": 0,
                "fallback_yfinance_fact_count": 0,
                "conflict_count": 0,
                "conflict_rate": 0.0,
                "uses_yfinance_fallback": False,
                "has_reconciliation_warning": False,
            },
            "notes": ["no_valid_metric"],
        }

    per_metric_limit = min(max(limit, 1), 100)
    trailing_n = min(max(trailing_n or 4, 1), 16)
    effective_year_basis = year_basis or "fiscal"
    mode = period_type or "any"
    effective_target_type = target_period_type
    if mode in {"latest", "trailing"}:
        effective_target_type = effective_target_type or "quarterly"
    elif mode in {"quarterly", "annual"}:
        effective_target_type = mode

    fiscal_year_end_month = _infer_fiscal_year_end_month(conn, ticker)
    notes: list[str] = []

    if (
        strict_period_match
        and effective_target_type == "annual"
        and effective_year_basis == "calendar"
        and fiscal_year_end_month is not None
        and fiscal_year_end_month != 12
    ):
        notes.append("calendar_year_not_strictly_mappable_for_non_dec_fiscal_year")
        return [], {
            "applied_filters": {
                "mode": mode,
                "target_period_type": effective_target_type,
                "year": year,
                "quarter": quarter,
                "trailing_n": trailing_n,
                "strict_period_match": strict_period_match,
            },
            "year_basis": effective_year_basis,
            "comparison_basis": comparison_basis or "same_period",
            "resolved_periods": [],
            "fiscal_year_end_month": fiscal_year_end_month,
            "source_summary": {
                "provider_counts": {},
                "sec_fact_count": 0,
                "fallback_yfinance_fact_count": 0,
                "conflict_count": 0,
                "conflict_rate": 0.0,
                "uses_yfinance_fallback": False,
                "has_reconciliation_warning": False,
            },
            "notes": notes,
        }

    all_results: list[dict[str, Any]] = []
    raw_provider_counts: dict[str, int] = {}
    raw_conflict_count = 0

    base_sql = (
        "SELECT ticker, period_end, period_type, metric, value, unit, filing_date, "
        "COALESCE(source_provider, 'yfinance') AS source_provider, source_url, source_filing_id, "
        "COALESCE(confidence, 'medium') AS confidence, "
        "COALESCE(extraction_method, 'api_statement') AS extraction_method, "
        "source_tag, reconciliation_warning "
        "FROM financial_facts WHERE ticker = ? AND metric = ?"
    )

    for metric_name in metric_list:
        params: list[Any] = [ticker.upper(), metric_name]
        sql = base_sql
        metric_target_type = None if metric_name == "shares_outstanding" and mode == "latest" else effective_target_type
        if mode in {"quarterly", "annual"}:
            sql += " AND period_type = ?"
            params.append(mode)
        elif mode in {"latest", "trailing"} and metric_target_type:
            sql += " AND period_type = ?"
            params.append(metric_target_type)
        if date_start:
            sql += " AND period_end >= CAST(? AS DATE)"
            params.append(date_start)
        if date_end:
            sql += " AND period_end <= CAST(? AS DATE)"
            params.append(date_end)
        sql += " ORDER BY period_end DESC LIMIT ?"
        params.append(per_metric_limit * 3)

        raw_rows = conn.execute(sql, params).fetchall()
        rows: list[dict[str, Any]] = []
        for r in raw_rows:
            period_end_dt: date = r[1]
            period_info = _annotate_period_fields(period_end_dt, fiscal_year_end_month)
            row = {
                "ticker": r[0],
                "period_end": str(period_end_dt),
                "period_type": r[2],
                "metric": r[3],
                "value": r[4],
                "unit": r[5],
                "filing_date": str(r[6]) if r[6] else "",
                "source_provider": r[7] or "yfinance",
                "source_url": r[8] or "",
                "source_filing_id": r[9] or "",
                "confidence": r[10] or "medium",
                "extraction_method": r[11] or "api_statement",
                "source_tag": r[12] or "",
                "reconciliation_warning": r[13] or "",
                **period_info,
            }

            if year is not None:
                key_year = row["fiscal_year"] if effective_year_basis == "fiscal" else row["calendar_year"]
                if key_year != year:
                    continue
            if quarter is not None:
                key_quarter = row["fiscal_quarter"] if effective_year_basis == "fiscal" else row["calendar_quarter"]
                if key_quarter != quarter:
                    continue
            rows.append(row)

        rows, metric_source_summary = _apply_source_priority_and_reconciliation(rows)
        raw_conflict_count += int(metric_source_summary.get("conflict_count", 0) or 0)
        for provider, count in (metric_source_summary.get("provider_counts", {}) or {}).items():
            raw_provider_counts[str(provider)] = raw_provider_counts.get(str(provider), 0) + int(count or 0)
        rows = sorted(rows, key=lambda x: x["period_end"], reverse=True)
        if mode == "latest":
            rows = rows[:1]
        elif mode == "trailing":
            rows = rows[:trailing_n]
        else:
            rows = rows[:per_metric_limit]
        all_results.extend(rows)

    all_results, source_summary = _apply_source_priority_and_reconciliation(all_results)
    source_summary["provider_counts"] = raw_provider_counts or source_summary.get("provider_counts", {})
    source_summary["conflict_count"] = raw_conflict_count
    source_summary["conflict_rate"] = raw_conflict_count / max(len(all_results), 1)
    source_summary["has_reconciliation_warning"] = bool(raw_conflict_count) or any(
        str(row.get("reconciliation_warning") or "").strip() for row in all_results
    )

    resolved_periods = sorted(
        {
            (
                str(r.get("ticker", "")),
                str(r.get("metric", "")),
                str(r.get("period_type", "")),
                str(r.get("period_end", "")),
            )
            for r in all_results
        },
        reverse=True,
    )
    if strict_period_match and (year is not None or quarter is not None) and not all_results:
        notes.append("no_exact_period_match")
    context = {
        "applied_filters": {
            "mode": mode,
            "target_period_type": effective_target_type,
            "year": year,
            "quarter": quarter,
            "trailing_n": trailing_n if mode == "trailing" else None,
            "date_start": date_start,
            "date_end": date_end,
            "strict_period_match": strict_period_match,
        },
        "year_basis": effective_year_basis,
        "comparison_basis": comparison_basis or "same_period",
        "resolved_periods": resolved_periods,
        "fiscal_year_end_month": fiscal_year_end_month,
        "source_summary": source_summary,
        "notes": notes,
    }
    return all_results, context


def _query_prices(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    metrics: list[str],
    date_start: str | None,
    date_end: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    price_cols = []
    for metric in metrics:
        normalized = _normalize_price_metric(metric)
        if normalized and normalized not in price_cols:
            price_cols.append(normalized)
    if not price_cols:
        return []

    cols_sql = ", ".join(price_cols)
    params: list[Any] = [ticker.upper()]
    sql = f"SELECT ticker, date, {cols_sql} FROM price_history WHERE ticker = ?"

    if date_start:
        sql += " AND date >= CAST(? AS DATE)"
        params.append(date_start)
    if date_end:
        sql += " AND date <= CAST(? AS DATE)"
        params.append(date_end)

    sql += " ORDER BY date DESC LIMIT ?"
    params.append(min(max(limit, 1), 100))

    rows = conn.execute(sql, params).fetchall()
    result: list[dict[str, Any]] = []
    for r in rows:
        entry: dict[str, Any] = {
            "ticker": r[0],
            "date": str(r[1]),
            "source_provider": "yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{ticker.upper()}/history",
            "source_filing_id": "",
            "confidence": "medium",
            "extraction_method": "api_price_history",
            "source_tag": "",
            "reconciliation_warning": "",
        }
        for idx, col in enumerate(price_cols):
            entry[col] = r[idx + 2]
        result.append(entry)
    return result


@tool("query_financial_data", args_schema=QueryFinancialDataInput)
def query_financial_data(
    ticker: str,
    metrics: list[str],
    period_type: str | None = None,
    target_period_type: str | None = None,
    year: int | None = None,
    quarter: int | None = None,
    trailing_n: int | None = None,
    year_basis: str | None = None,
    comparison_basis: str | None = None,
    strict_period_match: bool = True,
    date_start: str | None = None,
    date_end: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Query structured financial data for a company.

    Returns fundamental metrics (revenue, net_income, eps,
    gross_margin, operating_margin) from quarterly/annual filings,
    and/or daily price data (open, high, low, close, adjusted_close, volume).

    Use this tool when the question asks for specific numbers,
    financial metrics, or historical prices — NOT for narrative
    filing text (use search_filings for that).
    """
    conn = duckdb.connect(str(settings.duckdb_path))
    try:
        init_db(conn)
        facts, period_context = _query_facts(
            conn=conn,
            ticker=ticker,
            metrics=metrics,
            period_type=period_type,
            target_period_type=target_period_type,
            year=year,
            quarter=quarter,
            trailing_n=trailing_n,
            year_basis=year_basis,
            comparison_basis=comparison_basis,
            strict_period_match=strict_period_match,
            date_start=date_start,
            date_end=date_end,
            limit=limit,
        )
        prices = _query_prices(conn, ticker, metrics, date_start, date_end, limit)
    finally:
        conn.close()

    return {
        "ticker": ticker.upper(),
        "financial_facts": facts,
        "price_data": prices,
        "period_context": period_context,
    }
