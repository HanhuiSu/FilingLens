"""DuckDB lexical fallback tests for filing text retrieval."""

from __future__ import annotations

import importlib

import duckdb

search_module = importlib.import_module("src.tools.search_filings")


def test_search_filings_lexical_fallback_returns_item_1a_shape(tmp_path, monkeypatch):
    db_path = tmp_path / "filings.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE filings_metadata (
                filing_id VARCHAR PRIMARY KEY,
                ticker VARCHAR NOT NULL,
                form_type VARCHAR NOT NULL,
                fiscal_period VARCHAR,
                filing_date DATE,
                source_url VARCHAR,
                local_path VARCHAR NOT NULL,
                processed_path VARCHAR
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE filing_chunks (
                chunk_id VARCHAR PRIMARY KEY,
                filing_id VARCHAR NOT NULL,
                ticker VARCHAR NOT NULL,
                section VARCHAR NOT NULL,
                part VARCHAR,
                section_instance INTEGER,
                quality VARCHAR,
                chunk_text VARCHAR NOT NULL,
                chunk_order INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO filings_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["nvda-2026-10k", "NVDA", "10-K", "2026", "2026-03-01", "", "/tmp/nvda.html", None],
        )
        conn.execute(
            "INSERT INTO filing_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "chunk-risk",
                "nvda-2026-10k",
                "NVDA",
                "ITEM_1A",
                "",
                1,
                "high",
                "Risk Factors include intense competition, supply chain constraints, and demand uncertainty.",
                33,
            ],
        )
        conn.execute(
            "INSERT INTO filing_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "chunk-business",
                "nvda-2026-10k",
                "NVDA",
                "ITEM_1",
                "",
                1,
                "high",
                "NVIDIA sells accelerated computing platforms.",
                10,
            ],
        )
    finally:
        conn.close()

    original_connect = duckdb.connect

    def connect_test_db(_path, read_only=False):
        return original_connect(str(db_path), read_only=read_only)

    monkeypatch.setattr(search_module.duckdb, "connect", connect_test_db)

    result = search_module.search_filings_lexical_fallback(
        ticker="NVDA",
        query="NVDA risk factors competition supply chain",
        top_k=2,
        section_allowlist=["ITEM_1A"],
        strict_sections=True,
        retrieval_profile="risk_summary",
        return_diagnostics=True,
    )

    assert isinstance(result, dict)
    assert result["diagnostics"]["backend"] == "duckdb_lexical"
    assert result["diagnostics"]["raw_hit_count"] == 2
    assert result["diagnostics"]["section_filtered_hit_count"] == 1
    item = result["items"][0]
    assert item["ticker"] == "NVDA"
    assert item["filing_id"] == "nvda-2026-10k"
    assert item["form_type"] == "10-K"
    assert item["section"] == "ITEM_1A"
    assert item["supporting_snippet"]
    assert {"competition", "supply"} & set(item["supporting_terms"])
    assert item["score_breakdown"]["backend"] == "duckdb_lexical"
