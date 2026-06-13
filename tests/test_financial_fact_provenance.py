"""Tests for structured financial-fact provenance and SEC companyfacts loading."""

from __future__ import annotations

from datetime import date
import importlib
from pathlib import Path

import duckdb
import pandas as pd

from src.db.queries import FinancialFactRow, get_connection, insert_financial_facts_batch
from src.db.schema import init_db


def test_init_db_migrates_old_financial_facts_schema(tmp_path: Path):
    db_path = tmp_path / "old.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE financial_facts (
            id BIGINT PRIMARY KEY,
            ticker VARCHAR NOT NULL,
            period_end DATE NOT NULL,
            period_type VARCHAR NOT NULL,
            metric VARCHAR NOT NULL,
            value DOUBLE,
            unit VARCHAR,
            filing_date DATE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO financial_facts
        (id, ticker, period_end, period_type, metric, value, unit, filing_date)
        VALUES (1, 'AAPL', DATE '2025-09-27', 'annual', 'revenue', 100.0, 'USD', NULL)
        """
    )

    init_db(conn)
    cols = {row[0] for row in conn.execute("DESCRIBE financial_facts").fetchall()}
    row = conn.execute(
        "SELECT source_provider, confidence, extraction_method FROM financial_facts WHERE id = 1"
    ).fetchone()
    conn.close()

    assert {
        "source_provider",
        "source_url",
        "source_filing_id",
        "confidence",
        "extraction_method",
        "source_tag",
        "reconciliation_warning",
    }.issubset(cols)
    assert row == ("yfinance", "medium", "api_statement")


def test_financial_fact_row_inserts_provenance_fields(tmp_path: Path):
    conn = get_connection(tmp_path / "facts.duckdb")
    insert_financial_facts_batch(
        conn,
        [
            FinancialFactRow(
                ticker="AAPL",
                period_end=date(2025, 9, 27),
                period_type="annual",
                metric="revenue",
                value=100.0,
                unit="USD",
                source_provider="sec_companyfacts",
                source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
                source_filing_id="0000320193-25-000079",
                confidence="high",
                extraction_method="xbrl_companyfacts",
                source_tag="RevenueFromContractWithCustomerExcludingAssessedTax",
            )
        ],
    )
    row = conn.execute(
        """
        SELECT source_provider, source_filing_id, confidence, extraction_method, source_tag
        FROM financial_facts
        """
    ).fetchone()
    conn.close()

    assert row == (
        "sec_companyfacts",
        "0000320193-25-000079",
        "high",
        "xbrl_companyfacts",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    )


def test_query_financial_data_prefers_sec_and_flags_reconciliation(tmp_path: Path, monkeypatch):
    conn = get_connection(tmp_path / "query.duckdb")
    rows = [
        FinancialFactRow(
            ticker="AAPL",
            period_end=date(2025, 9, 27),
            period_type="annual",
            metric="revenue",
            value=100.0,
            unit="USD",
            source_provider="yfinance",
            confidence="medium",
            extraction_method="api_statement",
            source_tag="Total Revenue",
        ),
        FinancialFactRow(
            ticker="AAPL",
            period_end=date(2025, 9, 27),
            period_type="annual",
            metric="revenue",
            value=110.0,
            unit="USD",
            source_provider="sec_companyfacts",
            source_filing_id="0000320193-25-000079",
            confidence="high",
            extraction_method="xbrl_companyfacts",
            source_tag="RevenueFromContractWithCustomerExcludingAssessedTax",
        ),
    ]
    insert_financial_facts_batch(conn, rows)
    conn.close()

    qfd = importlib.import_module("src.tools.query_financial_data")

    class FakeSettings:
        duckdb_path = tmp_path / "query.duckdb"

    monkeypatch.setattr(qfd, "settings", FakeSettings())
    result = qfd.query_financial_data.invoke(
        {"ticker": "AAPL", "metrics": ["revenue"], "period_type": "annual", "limit": 5}
    )

    facts = result["financial_facts"]
    summary = result["period_context"]["source_summary"]
    assert len(facts) == 1
    assert facts[0]["source_provider"] == "sec_companyfacts"
    assert facts[0]["value"] == 110.0
    assert facts[0]["reconciliation_warning"] == "sec_yfinance_value_mismatch"
    assert summary["sec_fact_count"] == 1
    assert summary["fallback_yfinance_fact_count"] == 0
    assert summary["conflict_count"] == 1


def test_load_financial_data_marks_yfinance_rows():
    from scripts.load_financial_data import _facts_from_statement

    df = pd.DataFrame(
        {
            pd.Timestamp("2025-09-27"): {
                "Total Revenue": 100.0,
                "Net Income": 20.0,
                "Diluted EPS": 2.0,
                "Gross Profit": 40.0,
                "Operating Income": 30.0,
            }
        }
    )

    rows = _facts_from_statement(df, "annual", "AAPL", "AAPL")
    by_metric = {row.metric: row for row in rows}

    assert by_metric["revenue"].source_provider == "yfinance"
    assert by_metric["revenue"].confidence == "medium"
    assert by_metric["revenue"].extraction_method == "api_statement"
    assert by_metric["revenue"].source_tag == "Total Revenue"


def test_sec_companyfacts_parser_tag_priority_and_derived_margin():
    from scripts.load_sec_companyfacts import extract_companyfacts_rows, normalize_cik

    mappings = {
        "revenue": {"unit": "USD", "tags": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"]},
        "net_income": {"unit": "USD", "tags": ["NetIncomeLoss"]},
        "eps": {"unit": "USD_per_share", "sec_units": ["USD/shares"], "tags": ["EarningsPerShareDiluted"]},
        "gross_profit": {"unit": "USD", "internal_only": True, "tags": ["GrossProfit"]},
        "gross_margin": {"unit": "ratio", "derived": {"numerator": "gross_profit", "denominator": "revenue"}},
    }
    payload = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "form": "10-K",
                                "end": "2025-09-27",
                                "filed": "2025-10-31",
                                "accn": "0000320193-25-000079",
                                "val": 100.0,
                            }
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "form": "10-K",
                                "end": "2025-09-27",
                                "filed": "2025-10-31",
                                "accn": "fallback",
                                "val": 999.0,
                            }
                        ]
                    }
                },
                "GrossProfit": {
                    "units": {
                        "USD": [
                            {
                                "form": "10-K",
                                "end": "2025-09-27",
                                "filed": "2025-10-31",
                                "accn": "0000320193-25-000079",
                                "val": 40.0,
                            }
                        ]
                    }
                },
            }
        }
    }

    rows = extract_companyfacts_rows("AAPL", "320193", payload, mappings)
    by_metric = {row.metric: row for row in rows}

    assert normalize_cik("320193") == "0000320193"
    assert by_metric["revenue"].value == 100.0
    assert by_metric["revenue"].source_tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert by_metric["revenue"].source_filing_id == "0000320193-25-000079"
    assert by_metric["gross_margin"].value == 0.4
    assert by_metric["gross_margin"].extraction_method == "xbrl_companyfacts_derived_ratio"


def test_api_models_accept_numeric_provenance_fields():
    from src.api.models import Citation, NumericEvidenceCard

    card = NumericEvidenceCard(
        evidence_id="N1",
        source_provider="yfinance",
        confidence="medium",
        extraction_method="api_statement",
        reconciliation_warning="sec_yfinance_value_mismatch",
    )
    citation = Citation(
        source="AAPL",
        filing_type="STRUCTURED_YFINANCE",
        period="2025-09-27",
        section="STRUCTURED",
        source_kind="structured",
        source_provider="yfinance",
        confidence="medium",
    )

    assert card.source_provider == "yfinance"
    assert citation.filing_type == "STRUCTURED_YFINANCE"
    assert citation.source_provider == "yfinance"
