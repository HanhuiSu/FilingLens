"""DuckDB schema definitions and database initialisation."""

from __future__ import annotations

from pathlib import Path

import duckdb

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS companies (
        ticker VARCHAR PRIMARY KEY,
        company_name VARCHAR NOT NULL,
        sector VARCHAR NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS filings_metadata (
        filing_id VARCHAR PRIMARY KEY,
        ticker VARCHAR NOT NULL,
        form_type VARCHAR NOT NULL,
        fiscal_period VARCHAR,
        filing_date DATE,
        source_url VARCHAR,
        local_path VARCHAR NOT NULL,
        processed_path VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS financial_facts (
        id BIGINT PRIMARY KEY,
        ticker VARCHAR NOT NULL,
        period_end DATE NOT NULL,
        period_type VARCHAR NOT NULL,
        metric VARCHAR NOT NULL,
        value DOUBLE,
        unit VARCHAR,
        filing_date DATE,
        source_provider VARCHAR DEFAULT 'yfinance',
        source_url VARCHAR,
        source_filing_id VARCHAR,
        confidence VARCHAR DEFAULT 'medium',
        extraction_method VARCHAR DEFAULT 'api_statement',
        source_tag VARCHAR,
        reconciliation_warning VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS price_history (
        ticker VARCHAR NOT NULL,
        date DATE NOT NULL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        adjusted_close DOUBLE,
        volume DOUBLE,
        PRIMARY KEY (ticker, date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS filing_chunks (
        chunk_id VARCHAR PRIMARY KEY,
        filing_id VARCHAR NOT NULL,
        ticker VARCHAR NOT NULL,
        section VARCHAR NOT NULL,
        part VARCHAR,
        section_instance INTEGER,
        quality VARCHAR,
        chunk_text VARCHAR NOT NULL,
        chunk_order INTEGER NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS filing_events (
        ticker VARCHAR NOT NULL,
        filing_id VARCHAR NOT NULL,
        event_type VARCHAR NOT NULL,
        event_date DATE NOT NULL,
        fiscal_period VARCHAR,
        form_type VARCHAR NOT NULL,
        filing_date DATE,
        trading_anchor_date DATE,
        anchor_rule VARCHAR NOT NULL,
        has_price_data BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (ticker, filing_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS event_price_windows (
        ticker VARCHAR NOT NULL,
        filing_id VARCHAR NOT NULL,
        event_date DATE NOT NULL,
        trading_anchor_date DATE,
        t_minus_1_date DATE,
        t_minus_1_close DOUBLE,
        t_close_date DATE,
        t_close DOUBLE,
        t_plus_1_date DATE,
        t_plus_1_close DOUBLE,
        t_plus_3_date DATE,
        t_plus_3_close DOUBLE,
        t_plus_5_date DATE,
        t_plus_5_close DOUBLE,
        t_plus_10_date DATE,
        t_plus_10_close DOUBLE,
        return_1d DOUBLE,
        return_3d DOUBLE,
        return_5d DOUBLE,
        return_10d DOUBLE,
        coverage_flag VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (ticker, filing_id)
    );
    """,
]


def init_db(conn: duckdb.DuckDBPyConnection | None = None, db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Create tables if missing. Returns an open connection (caller owns close if conn was None)."""
    if conn is None:
        path = db_path
        if path is None:
            from config import settings

            path = settings.duckdb_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(path))

    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)

    # Lightweight schema migrations for existing DB files.
    conn.execute("ALTER TABLE filing_chunks ADD COLUMN IF NOT EXISTS part VARCHAR")
    conn.execute("ALTER TABLE filing_chunks ADD COLUMN IF NOT EXISTS section_instance INTEGER")
    conn.execute("ALTER TABLE filing_chunks ADD COLUMN IF NOT EXISTS quality VARCHAR")
    conn.execute("ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS source_provider VARCHAR DEFAULT 'yfinance'")
    conn.execute("ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS source_url VARCHAR")
    conn.execute("ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS source_filing_id VARCHAR")
    conn.execute("ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS confidence VARCHAR DEFAULT 'medium'")
    conn.execute("ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS extraction_method VARCHAR DEFAULT 'api_statement'")
    conn.execute("ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS source_tag VARCHAR")
    conn.execute("ALTER TABLE financial_facts ADD COLUMN IF NOT EXISTS reconciliation_warning VARCHAR")
    conn.execute(
        """
        UPDATE financial_facts
        SET source_provider = COALESCE(source_provider, 'yfinance'),
            confidence = COALESCE(confidence, 'medium'),
            extraction_method = COALESCE(extraction_method, 'api_statement')
        """
    )

    return conn
