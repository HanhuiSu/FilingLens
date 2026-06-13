"""DuckDB connection helpers and batched writes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import duckdb

from src.db.schema import init_db


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    from config import settings

    path = db_path or settings.duckdb_path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    init_db(conn)
    return conn


@dataclass
class CompanyRow:
    ticker: str
    company_name: str
    sector: str


def seed_companies(conn: duckdb.DuckDBPyConnection, rows: Sequence[CompanyRow]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO companies (ticker, company_name, sector)
        VALUES (?, ?, ?)
        """,
        [(r.ticker, r.company_name, r.sector) for r in rows],
    )


@dataclass
class FilingMetadataRow:
    filing_id: str
    ticker: str
    form_type: str
    fiscal_period: str | None
    filing_date: date | None
    source_url: str | None
    local_path: str
    processed_path: str | None = None


def upsert_filing_metadata(conn: duckdb.DuckDBPyConnection, rows: Sequence[FilingMetadataRow]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO filings_metadata (
            filing_id, ticker, form_type, fiscal_period, filing_date,
            source_url, local_path, processed_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.filing_id,
                r.ticker,
                r.form_type,
                r.fiscal_period,
                r.filing_date,
                r.source_url,
                r.local_path,
                r.processed_path,
            )
            for r in rows
        ],
    )


def update_filing_processed_path(
    conn: duckdb.DuckDBPyConnection,
    filing_id: str,
    processed_path: str,
    fiscal_period: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE filings_metadata
        SET processed_path = ?,
            fiscal_period = COALESCE(?, fiscal_period)
        WHERE filing_id = ?
        """,
        [processed_path, fiscal_period, filing_id],
    )


@dataclass
class FinancialFactRow:
    ticker: str
    period_end: date
    period_type: str
    metric: str
    value: float | None
    unit: str | None
    filing_date: date | None = None
    source_provider: str = "yfinance"
    source_url: str | None = None
    source_filing_id: str | None = None
    confidence: str = "medium"
    extraction_method: str = "api_statement"
    source_tag: str | None = None
    reconciliation_warning: str | None = None


def _next_financial_fact_ids(conn: duckdb.DuckDBPyConnection, n: int) -> list[int]:
    if n <= 0:
        return []
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM financial_facts").fetchone()
    start = int(row[0]) + 1
    return list(range(start, start + n))


def insert_financial_facts_batch(conn: duckdb.DuckDBPyConnection, rows: Sequence[FinancialFactRow]) -> None:
    if not rows:
        return
    ids = _next_financial_fact_ids(conn, len(rows))
    conn.executemany(
        """
        INSERT INTO financial_facts (
            id, ticker, period_end, period_type, metric, value, unit, filing_date,
            source_provider, source_url, source_filing_id, confidence,
            extraction_method, source_tag, reconciliation_warning
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                ids[i],
                r.ticker,
                r.period_end,
                r.period_type,
                r.metric,
                r.value,
                r.unit,
                r.filing_date,
                r.source_provider,
                r.source_url,
                r.source_filing_id,
                r.confidence,
                r.extraction_method,
                r.source_tag,
                r.reconciliation_warning,
            )
            for i, r in enumerate(rows)
        ],
    )


@dataclass
class PriceHistoryRow:
    ticker: str
    d: date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adjusted_close: float | None
    volume: float | None


def insert_price_history_batch(conn: duckdb.DuckDBPyConnection, rows: Sequence[PriceHistoryRow]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO price_history
        (ticker, date, open, high, low, close, adjusted_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (r.ticker, r.d, r.open, r.high, r.low, r.close, r.adjusted_close, r.volume)
            for r in rows
        ],
    )


@dataclass
class FilingChunkRow:
    chunk_id: str
    filing_id: str
    ticker: str
    section: str
    part: str | None
    section_instance: int | None
    quality: str | None
    chunk_text: str
    chunk_order: int


def insert_chunks_batch(conn: duckdb.DuckDBPyConnection, rows: Sequence[FilingChunkRow]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO filing_chunks
        (chunk_id, filing_id, ticker, section, part, section_instance, quality, chunk_text, chunk_order)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.chunk_id,
                r.filing_id,
                r.ticker,
                r.section,
                r.part,
                r.section_instance,
                r.quality,
                r.chunk_text,
                r.chunk_order,
            )
            for r in rows
        ],
    )


def delete_chunks_for_filing(conn: duckdb.DuckDBPyConnection, filing_id: str) -> None:
    conn.execute("DELETE FROM filing_chunks WHERE filing_id = ?", [filing_id])


def clear_financial_facts_for_ticker(conn: duckdb.DuckDBPyConnection, ticker: str) -> None:
    conn.execute("DELETE FROM financial_facts WHERE ticker = ?", [ticker])


def clear_financial_facts_for_ticker_provider(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    source_provider: str,
) -> None:
    conn.execute(
        "DELETE FROM financial_facts WHERE ticker = ? AND source_provider = ?",
        [ticker, source_provider],
    )


def clear_price_history_for_ticker(conn: duckdb.DuckDBPyConnection, ticker: str) -> None:
    conn.execute("DELETE FROM price_history WHERE ticker = ?", [ticker])


@dataclass
class FilingEventRow:
    ticker: str
    filing_id: str
    event_type: str
    event_date: date
    fiscal_period: str | None
    form_type: str
    filing_date: date | None
    trading_anchor_date: date | None
    anchor_rule: str = "filing_date_then_next_trading_day"
    has_price_data: bool = False


def upsert_filing_events(conn: duckdb.DuckDBPyConnection, rows: Sequence[FilingEventRow]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO filing_events
        (ticker, filing_id, event_type, event_date, fiscal_period, form_type, filing_date,
         trading_anchor_date, anchor_rule, has_price_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.ticker,
                r.filing_id,
                r.event_type,
                r.event_date,
                r.fiscal_period,
                r.form_type,
                r.filing_date,
                r.trading_anchor_date,
                r.anchor_rule,
                r.has_price_data,
            )
            for r in rows
        ],
    )


def clear_filing_events_for_ticker(conn: duckdb.DuckDBPyConnection, ticker: str) -> None:
    conn.execute("DELETE FROM filing_events WHERE ticker = ?", [ticker])


@dataclass
class EventPriceWindowRow:
    ticker: str
    filing_id: str
    event_date: date
    trading_anchor_date: date | None
    t_minus_1_date: date | None
    t_minus_1_close: float | None
    t_close_date: date | None
    t_close: float | None
    t_plus_1_date: date | None
    t_plus_1_close: float | None
    t_plus_3_date: date | None
    t_plus_3_close: float | None
    t_plus_5_date: date | None
    t_plus_5_close: float | None
    t_plus_10_date: date | None
    t_plus_10_close: float | None
    return_1d: float | None
    return_3d: float | None
    return_5d: float | None
    return_10d: float | None
    coverage_flag: str


def upsert_event_price_windows(conn: duckdb.DuckDBPyConnection, rows: Sequence[EventPriceWindowRow]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO event_price_windows
        (ticker, filing_id, event_date, trading_anchor_date,
         t_minus_1_date, t_minus_1_close, t_close_date, t_close,
         t_plus_1_date, t_plus_1_close, t_plus_3_date, t_plus_3_close,
         t_plus_5_date, t_plus_5_close, t_plus_10_date, t_plus_10_close,
         return_1d, return_3d, return_5d, return_10d, coverage_flag)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.ticker,
                r.filing_id,
                r.event_date,
                r.trading_anchor_date,
                r.t_minus_1_date,
                r.t_minus_1_close,
                r.t_close_date,
                r.t_close,
                r.t_plus_1_date,
                r.t_plus_1_close,
                r.t_plus_3_date,
                r.t_plus_3_close,
                r.t_plus_5_date,
                r.t_plus_5_close,
                r.t_plus_10_date,
                r.t_plus_10_close,
                r.return_1d,
                r.return_3d,
                r.return_5d,
                r.return_10d,
                r.coverage_flag,
            )
            for r in rows
        ],
    )


def clear_event_price_windows_for_ticker(conn: duckdb.DuckDBPyConnection, ticker: str) -> None:
    conn.execute("DELETE FROM event_price_windows WHERE ticker = ?", [ticker])
