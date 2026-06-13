#!/usr/bin/env python3
"""Load quarterly / annual financial metrics from yfinance into financial_facts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings

from src.db.queries import (
    FinancialFactRow,
    clear_financial_facts_for_ticker_provider,
    get_connection,
    insert_financial_facts_batch,
)


def _num(x) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _row_metrics(df: pd.DataFrame, row_name: str, col) -> float | None:
    if df is None or df.empty or row_name not in df.index:
        return None
    return _num(df.loc[row_name, col])


def _first_row_metric(df: pd.DataFrame, row_names: list[str], col) -> tuple[float | None, str | None]:
    for row_name in row_names:
        value = _row_metrics(df, row_name, col)
        if value is not None:
            return value, row_name
    return None, None


def _statement_periods(df: pd.DataFrame):
    if df is None or df.empty:
        return
    for col in df.columns:
        if not isinstance(col, pd.Timestamp):
            try:
                pe = pd.Timestamp(col).date()
            except Exception:
                continue
        else:
            pe = col.date()
        yield col, pe


def _facts_from_income_statement(
    df: pd.DataFrame,
    period_type: str,
    db_ticker: str,
    yf_symbol: str,
) -> list[FinancialFactRow]:
    if df is None or df.empty:
        return []
    rows: list[FinancialFactRow] = []
    for col, pe in _statement_periods(df):
        rev, rev_tag = _first_row_metric(df, ["Total Revenue", "Operating Revenue"], col)
        ni, ni_tag = _first_row_metric(df, ["Net Income", "Net Income Common Stockholders"], col)
        eps, eps_tag = _first_row_metric(df, ["Diluted EPS", "Basic EPS"], col)
        gp, gp_tag = _first_row_metric(df, ["Gross Profit"], col)
        op, op_tag = _first_row_metric(df, ["Operating Income", "Operating Income or Loss"], col)

        if rev is not None:
            rows.append(
                _yfinance_fact(db_ticker, pe, period_type, "revenue", rev, "USD", rev_tag or "Total Revenue", yf_symbol)
            )
        if ni is not None:
            rows.append(
                _yfinance_fact(db_ticker, pe, period_type, "net_income", ni, "USD", ni_tag or "Net Income", yf_symbol)
            )
        if eps is not None:
            rows.append(
                _yfinance_fact(db_ticker, pe, period_type, "eps", eps, "USD_per_share", eps_tag or "Diluted EPS", yf_symbol)
            )
        if gp is not None:
            rows.append(
                _yfinance_fact(db_ticker, pe, period_type, "gross_profit", gp, "USD", gp_tag or "Gross Profit", yf_symbol)
            )
        if op is not None:
            rows.append(
                _yfinance_fact(
                    db_ticker,
                    pe,
                    period_type,
                    "operating_income",
                    op,
                    "USD",
                    op_tag or "Operating Income",
                    yf_symbol,
                )
            )
        if gp is not None and rev not in (None, 0):
            rows.append(
                _yfinance_fact(
                    db_ticker,
                    pe,
                    period_type,
                    "gross_margin",
                    gp / rev if rev else None,
                    "ratio",
                    gp_tag or "Gross Profit",
                    yf_symbol,
                )
            )
        if op is not None and rev not in (None, 0):
            rows.append(
                _yfinance_fact(
                    db_ticker,
                    pe,
                    period_type,
                    "operating_margin",
                    op / rev if rev else None,
                    "ratio",
                    op_tag or "Operating Income",
                    yf_symbol,
                )
            )
    return rows


def _facts_from_statement(
    df: pd.DataFrame,
    period_type: str,
    db_ticker: str,
    yf_symbol: str,
) -> list[FinancialFactRow]:
    """Backward-compatible alias for income statement extraction."""
    return _facts_from_income_statement(df, period_type, db_ticker, yf_symbol)


def _facts_from_cashflow(
    df: pd.DataFrame,
    period_type: str,
    db_ticker: str,
    yf_symbol: str,
) -> list[FinancialFactRow]:
    rows: list[FinancialFactRow] = []
    for col, pe in _statement_periods(df):
        cfo, cfo_tag = _first_row_metric(
            df,
            ["Operating Cash Flow", "Total Cash From Operating Activities", "Net Cash Provided By Operating Activities"],
            col,
        )
        capex, capex_tag = _first_row_metric(
            df,
            ["Capital Expenditure", "Capital Expenditures", "Capital Expenditure Reported"],
            col,
        )
        fcf, fcf_tag = _first_row_metric(df, ["Free Cash Flow"], col)
        capex_outflow = abs(capex) if capex is not None else None
        if cfo is not None:
            rows.append(
                _yfinance_fact(
                    db_ticker,
                    pe,
                    period_type,
                    "operating_cash_flow",
                    cfo,
                    "USD",
                    cfo_tag or "Operating Cash Flow",
                    yf_symbol,
                )
            )
        if capex_outflow is not None:
            rows.append(
                _yfinance_fact(
                    db_ticker,
                    pe,
                    period_type,
                    "capital_expenditure",
                    capex_outflow,
                    "USD",
                    f"{capex_tag or 'Capital Expenditure'} (abs outflow)",
                    yf_symbol,
                )
            )
        if fcf is None and cfo is not None and capex_outflow is not None:
            fcf = cfo - capex_outflow
            fcf_tag = "Operating Cash Flow - abs(Capital Expenditure)"
        if fcf is not None:
            rows.append(
                _yfinance_fact(db_ticker, pe, period_type, "free_cash_flow", fcf, "USD", fcf_tag or "Free Cash Flow", yf_symbol)
            )
    return rows


def _facts_from_balance_sheet(
    df: pd.DataFrame,
    period_type: str,
    db_ticker: str,
    yf_symbol: str,
) -> list[FinancialFactRow]:
    rows: list[FinancialFactRow] = []
    for col, pe in _statement_periods(df):
        mappings = {
            "cash_and_equivalents": [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents And Short Term Investments",
            ],
            "short_term_debt": ["Current Debt", "Short Term Debt", "Short Long Term Debt"],
            "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
            "total_debt": ["Total Debt"],
            "total_assets": ["Total Assets"],
            "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
            "shareholders_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
            "receivables": ["Accounts Receivable", "Net Receivables"],
            "inventory": ["Inventory"],
        }
        values: dict[str, tuple[float, str]] = {}
        for metric, row_names in mappings.items():
            value, source_tag = _first_row_metric(df, row_names, col)
            if value is not None:
                values[metric] = (value, source_tag or row_names[0])
        if "total_debt" not in values:
            short_debt = values.get("short_term_debt", (0.0, ""))[0]
            long_debt = values.get("long_term_debt", (0.0, ""))[0]
            if short_debt or long_debt:
                values["total_debt"] = (short_debt + long_debt, "short_term_debt + long_term_debt")
        for metric, (value, source_tag) in values.items():
            rows.append(_yfinance_fact(db_ticker, pe, period_type, metric, value, "USD", source_tag, yf_symbol))
    return rows


def _facts_from_shares(ticker_obj: yf.Ticker, db_ticker: str, yf_symbol: str) -> list[FinancialFactRow]:
    rows: list[FinancialFactRow] = []
    shares_series = None
    try:
        shares_series = ticker_obj.get_shares_full()
    except Exception:
        shares_series = None
    if shares_series is not None and not shares_series.empty:
        for idx, value in shares_series.tail(12).items():
            shares = _num(value)
            if shares is None:
                continue
            rows.append(
                _yfinance_fact(
                    db_ticker,
                    pd.Timestamp(idx).date(),
                    "latest",
                    "shares_outstanding",
                    shares,
                    "shares",
                    "get_shares_full",
                    yf_symbol,
                )
            )
    if rows:
        return rows
    for attr in ("fast_info", "info"):
        try:
            payload = getattr(ticker_obj, attr) or {}
            shares = _num(payload.get("shares") or payload.get("sharesOutstanding"))
        except Exception:
            shares = None
        if shares is not None:
            rows.append(
                _yfinance_fact(
                    db_ticker,
                    pd.Timestamp.utcnow().date(),
                    "latest",
                    "shares_outstanding",
                    shares,
                    "shares",
                    f"{attr}.shares",
                    yf_symbol,
                )
            )
            break
    return rows


def _yfinance_fact(
    ticker: str,
    period_end,
    period_type: str,
    metric: str,
    value: float | None,
    unit: str,
    source_tag: str,
    yf_symbol: str,
) -> FinancialFactRow:
    return FinancialFactRow(
        ticker=ticker,
        period_end=period_end,
        period_type=period_type,
        metric=metric,
        value=value,
        unit=unit,
        filing_date=None,
        source_provider="yfinance",
        source_url=f"https://finance.yahoo.com/quote/{yf_symbol}/financials",
        source_filing_id=None,
        confidence="medium",
        extraction_method="api_statement",
        source_tag=source_tag,
    )


def load_ticker(conn, db_ticker: str) -> None:
    yf_sym = "GOOG" if db_ticker == "GOOGL" else db_ticker
    t = yf.Ticker(yf_sym)

    qdf = t.quarterly_income_stmt
    adf = t.income_stmt
    qcf = t.quarterly_cashflow
    acf = t.cashflow
    qbs = t.quarterly_balance_sheet
    abs_ = t.balance_sheet
    if qdf is None or qdf.empty:
        t2 = yf.Ticker("GOOGL" if yf_sym == "GOOG" else yf_sym)
        qdf = t2.quarterly_income_stmt
    if adf is None or adf.empty:
        t2 = yf.Ticker("GOOGL" if yf_sym == "GOOG" else yf_sym)
        adf = t2.income_stmt

    batch: list[FinancialFactRow] = []
    batch.extend(_facts_from_income_statement(qdf, "quarterly", db_ticker, yf_sym))
    batch.extend(_facts_from_income_statement(adf, "annual", db_ticker, yf_sym))
    batch.extend(_facts_from_cashflow(qcf, "quarterly", db_ticker, yf_sym))
    batch.extend(_facts_from_cashflow(acf, "annual", db_ticker, yf_sym))
    batch.extend(_facts_from_balance_sheet(qbs, "quarterly", db_ticker, yf_sym))
    batch.extend(_facts_from_balance_sheet(abs_, "annual", db_ticker, yf_sym))
    batch.extend(_facts_from_shares(t, db_ticker, yf_sym))
    if not batch:
        print(f"  {db_ticker}: 0 fact rows fetched — keeping existing data")
        return
    clear_financial_facts_for_ticker_provider(conn, db_ticker, "yfinance")
    insert_financial_facts_batch(conn, batch)
    print(f"  {db_ticker}: {len(batch)} fact rows")


def main() -> None:
    conn = get_connection()
    for ticker in settings.target_tickers:
        load_ticker(conn, ticker.upper())
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
