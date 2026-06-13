from __future__ import annotations

from datetime import date
import importlib

import pandas as pd

from scripts.load_financial_data import _facts_from_cashflow, _facts_from_income_statement
from src.db.queries import FinancialFactRow, insert_financial_facts_batch
from src.db.schema import init_db


def test_yfinance_hydration_rows_use_medium_fallback_provenance():
    income = pd.DataFrame(
        {
            pd.Timestamp("2026-01-31"): {
                "Total Revenue": 100.0,
                "Net Income": 30.0,
                "Gross Profit": 60.0,
                "Operating Income": 45.0,
            }
        }
    )

    rows = _facts_from_income_statement(income, "annual", "NVDA", "NVDA")
    by_metric = {row.metric: row for row in rows}

    assert by_metric["revenue"].source_provider == "yfinance"
    assert by_metric["revenue"].confidence == "medium"
    assert by_metric["revenue"].extraction_method == "api_statement"
    assert by_metric["gross_profit"].source_provider == "yfinance"


def test_yfinance_cashflow_hydration_normalizes_capex_and_fcf():
    cashflow = pd.DataFrame(
        {
            pd.Timestamp("2026-01-31"): {
                "Operating Cash Flow": 80.0,
                "Capital Expenditure": -25.0,
            }
        }
    )

    rows = _facts_from_cashflow(cashflow, "annual", "NVDA", "NVDA")
    by_metric = {row.metric: row for row in rows}

    assert by_metric["capital_expenditure"].value == 25.0
    assert by_metric["free_cash_flow"].value == 55.0
    assert by_metric["capital_expenditure"].source_tag.endswith("(abs outflow)")


def test_query_selection_prefers_sec_over_yfinance_fallback(tmp_path, monkeypatch):
    db_path = tmp_path / "precedence.duckdb"
    conn = init_db(db_path=db_path)
    try:
        insert_financial_facts_batch(
            conn,
            [
                FinancialFactRow(
                    "NVDA",
                    date(2026, 1, 31),
                    "annual",
                    "operating_cash_flow",
                    75.0,
                    "USD",
                    source_provider="yfinance",
                    confidence="medium",
                    extraction_method="api_statement",
                ),
                FinancialFactRow(
                    "NVDA",
                    date(2026, 1, 31),
                    "annual",
                    "operating_cash_flow",
                    80.0,
                    "USD",
                    source_provider="sec_companyfacts",
                    confidence="high",
                    extraction_method="xbrl_companyfacts",
                ),
            ],
        )
    finally:
        conn.close()

    qfd = importlib.import_module("src.tools.query_financial_data")

    class FakeSettings:
        duckdb_path = db_path

    monkeypatch.setattr(qfd, "settings", FakeSettings())
    result = qfd.query_financial_data.invoke(
        {"ticker": "NVDA", "metrics": ["operating_cash_flow"], "period_type": "annual", "limit": 5}
    )

    fact = result["financial_facts"][0]
    assert fact["source_provider"] == "sec_companyfacts"
    assert fact["value"] == 80.0
    assert result["period_context"]["source_summary"]["fallback_yfinance_fact_count"] == 0
