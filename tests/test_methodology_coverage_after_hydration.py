from __future__ import annotations

from datetime import date
import json

from scripts.report_methodology_data_coverage import build_coverage_diff, build_coverage_report, write_coverage_diff, write_report
from src.db.queries import FinancialFactRow, PriceHistoryRow, insert_financial_facts_batch, insert_price_history_batch
from src.db.schema import init_db


def _insert_fact(conn, metric: str, value: float, *, provider: str = "sec_companyfacts") -> None:
    insert_financial_facts_batch(
        conn,
        [
            FinancialFactRow(
                "NVDA",
                date(2026, 1, 31),
                "annual",
                metric,
                value,
                "shares" if metric == "shares_outstanding" else "USD",
                source_provider=provider,
                confidence="high" if provider == "sec_companyfacts" else "medium",
                extraction_method="xbrl_companyfacts" if provider == "sec_companyfacts" else "api_statement",
            )
        ],
    )


def test_coverage_report_dimension_statuses_reflect_hydrated_metrics(tmp_path):
    db_path = tmp_path / "coverage.duckdb"
    conn = init_db(db_path=db_path)
    try:
        for metric, value in (
            ("revenue", 200.0),
            ("net_income", 50.0),
            ("gross_profit", 120.0),
            ("gross_margin", 0.6),
            ("operating_cash_flow", 70.0),
            ("free_cash_flow", 55.0),
            ("cash_conversion", 1.4),
            ("cash_and_equivalents", 30.0),
            ("total_debt", 10.0),
            ("total_assets", 300.0),
            ("total_liabilities", 90.0),
            ("shares_outstanding", 10.0),
            ("market_cap", 1000.0),
            ("pe_ratio", 20.0),
            ("ps_ratio", 5.0),
        ):
            _insert_fact(conn, metric, value)
        insert_price_history_batch(
            conn,
            [PriceHistoryRow("NVDA", date(2026, 4, 24), 100.0, 101.0, 99.0, 100.0, 100.0, 1000)],
        )
    finally:
        conn.close()

    report = build_coverage_report(db_path=db_path, tickers=["NVDA"])
    dims = report["companies"]["NVDA"]["dimensions"]

    assert dims["cash_flow_quality"]["status"] == "satisfied"
    assert dims["balance_sheet_and_capital_intensity"]["status"] == "satisfied"
    assert dims["valuation_and_risk_boundary"]["status"] == "satisfied"


def test_coverage_report_counts_programmatically_derivable_valuation_metrics(tmp_path):
    db_path = tmp_path / "coverage_derived.duckdb"
    conn = init_db(db_path=db_path)
    try:
        for metric, value in (
            ("revenue", 200.0),
            ("net_income", 50.0),
            ("free_cash_flow", 25.0),
            ("shares_outstanding", 10.0),
        ):
            _insert_fact(conn, metric, value)
        insert_price_history_batch(
            conn,
            [PriceHistoryRow("NVDA", date(2026, 4, 24), 100.0, 101.0, 99.0, 100.0, 100.0, 1000)],
        )
    finally:
        conn.close()

    report = build_coverage_report(db_path=db_path, tickers=["NVDA"])
    valuation = report["companies"]["NVDA"]["dimensions"]["valuation_and_risk_boundary"]

    assert valuation["status"] == "satisfied"
    assert {"market_cap", "pe_ratio", "ps_ratio", "fcf_yield"}.issubset(valuation["available_metrics"])
    assert valuation["metric_sources"]["market_cap"][0]["source_provider"] == "computed"
    assert valuation["metric_sources"]["pe_ratio"][0]["dependencies"]


def test_coverage_diff_shows_newly_available_metrics_and_remaining_gaps(tmp_path):
    before = {
        "companies": {
            "NVDA": {
                "dimensions": {
                    "cash_flow_quality": {
                        "status": "missing",
                        "available_metrics": [],
                        "missing_metrics": ["operating_cash_flow", "free_cash_flow"],
                    }
                }
            }
        }
    }
    after = {
        "companies": {
            "NVDA": {
                "dimensions": {
                    "cash_flow_quality": {
                        "status": "partial",
                        "available_metrics": ["operating_cash_flow"],
                        "missing_metrics": ["free_cash_flow", "cash_conversion"],
                    }
                }
            }
        }
    }

    markdown = build_coverage_diff(before, after)
    assert "missing -> partial" in markdown
    assert "operating_cash_flow" in markdown
    assert "free_cash_flow" in markdown
    assert "cash_conversion" in markdown

    before_path = tmp_path / "methodology_data_coverage_before.json"
    after_path = tmp_path / "methodology_data_coverage_after.json"
    diff_path = tmp_path / "methodology_data_coverage_diff.md"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")
    assert write_coverage_diff(before_path, after_path, diff_path)
    assert "Methodology Data Coverage Diff" in diff_path.read_text(encoding="utf-8")


def test_write_report_creates_snapshot_json(tmp_path):
    output = tmp_path / "methodology_data_coverage_after.json"
    write_report({"companies": {}}, output)
    assert output.exists()
