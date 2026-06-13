from __future__ import annotations

from datetime import date

from scripts.report_methodology_data_coverage import build_coverage_report, write_report
from src.db.queries import FinancialFactRow, PriceHistoryRow, insert_financial_facts_batch, insert_price_history_batch
from src.db.schema import init_db


def test_methodology_data_coverage_report_lists_available_and_missing_metrics(tmp_path):
    db_path = tmp_path / "coverage.duckdb"
    conn = init_db(db_path=db_path)
    try:
        insert_financial_facts_batch(
            conn,
            [
                FinancialFactRow("NVDA", date(2025, 1, 31), "annual", "revenue", 100.0, "USD", source_provider="sec_companyfacts", confidence="high", extraction_method="xbrl_companyfacts"),
                FinancialFactRow("NVDA", date(2025, 1, 31), "annual", "net_income", 40.0, "USD", source_provider="sec_companyfacts", confidence="high", extraction_method="xbrl_companyfacts"),
                FinancialFactRow("NVDA", date(2025, 1, 31), "annual", "operating_cash_flow", 50.0, "USD", source_provider="yfinance", confidence="medium", extraction_method="api_statement"),
                FinancialFactRow("NVDA", date(2025, 1, 31), "annual", "cash_and_equivalents", 20.0, "USD", source_provider="yfinance", confidence="medium", extraction_method="api_statement"),
            ],
        )
        insert_price_history_batch(
            conn,
            [
                PriceHistoryRow(
                    ticker="NVDA",
                    d=date(2026, 4, 2),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    adjusted_close=100.5,
                    volume=1_000_000,
                )
            ],
        )
    finally:
        conn.close()

    report = build_coverage_report(db_path=db_path, tickers=["NVDA"])

    nvda = report["companies"]["NVDA"]
    assert "revenue" in nvda["dimensions"]["revenue_quality"]["available_metrics"]
    assert "gross_profit" in nvda["dimensions"]["profitability_quality"]["missing_metrics"]
    assert "operating_cash_flow" in nvda["dimensions"]["cash_flow_quality"]["available_metrics"]
    assert "cash" in nvda["dimensions"]["balance_sheet_and_capital_intensity"]["available_metrics"]
    assert "share_price" in nvda["dimensions"]["valuation_and_risk_boundary"]["available_metrics"]
    assert nvda["dimensions"]["cash_flow_quality"]["provider_counts"]["yfinance"] == 1

    output = tmp_path / "methodology_data_coverage.json"
    write_report(report, output)
    assert output.exists()
