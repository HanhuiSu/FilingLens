from __future__ import annotations

from datetime import date

from scripts.report_financial_fact_reconciliation import build_markdown, build_reconciliation_report
from src.db.queries import FinancialFactRow, insert_financial_facts_batch
from src.db.schema import init_db


def test_financial_fact_reconciliation_reports_warning_and_preferred_source(tmp_path):
    db_path = tmp_path / "reconciliation.duckdb"
    conn = init_db(db_path=db_path)
    try:
        insert_financial_facts_batch(
            conn,
            [
                FinancialFactRow(
                    "NVDA",
                    date(2026, 1, 31),
                    "annual",
                    "revenue",
                    100.0,
                    "USD",
                    source_provider="sec_companyfacts",
                    confidence="high",
                    extraction_method="xbrl_companyfacts",
                    source_tag="Revenues",
                ),
                FinancialFactRow(
                    "NVDA",
                    date(2026, 1, 31),
                    "annual",
                    "revenue",
                    107.0,
                    "USD",
                    source_provider="yfinance",
                    confidence="medium",
                    extraction_method="api_statement",
                    source_tag="Total Revenue",
                ),
            ],
        )
    finally:
        conn.close()

    report = build_reconciliation_report(
        db_path=db_path,
        tickers=["NVDA"],
        metrics=["revenue"],
        write_warnings=True,
    )

    record = report["records"][0]
    assert record["preferred_source"] == "sec"
    assert record["warning_severity"] == "high"
    assert record["warning"] == "value_mismatch_gt_5pct"
    assert record["pct_diff"] == 0.07
    markdown = build_markdown(report)
    assert "Financial Fact Reconciliation" in markdown
    assert "value_mismatch_gt_5pct" in markdown

    conn = init_db(db_path=db_path)
    try:
        warnings = {
            row[0]
            for row in conn.execute(
                "SELECT reconciliation_warning FROM financial_facts WHERE ticker = 'NVDA'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert warnings == {"value_mismatch_gt_5pct"}


def test_financial_fact_reconciliation_flags_unit_mismatch(tmp_path):
    db_path = tmp_path / "unit.duckdb"
    conn = init_db(db_path=db_path)
    try:
        insert_financial_facts_batch(
            conn,
            [
                FinancialFactRow("AAPL", date(2025, 9, 27), "annual", "shares_outstanding", 10.0, "shares", source_provider="sec_companyfacts", confidence="high", extraction_method="xbrl_companyfacts"),
                FinancialFactRow("AAPL", date(2025, 9, 27), "annual", "shares_outstanding", 10.0, "USD", source_provider="yfinance", confidence="medium", extraction_method="api_statement"),
            ],
        )
    finally:
        conn.close()

    report = build_reconciliation_report(
        db_path=db_path,
        tickers=["AAPL"],
        metrics=["shares_outstanding"],
    )

    assert report["records"][0]["warning_severity"] == "high"
    assert report["records"][0]["warning"] == "unit_mismatch"
