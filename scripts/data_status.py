#!/usr/bin/env python3
"""Print local data asset status without requiring vLLM or embedding models."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings


def _print_kv(label: str, value: Any) -> None:
    print(f"{label:<42} {value}")


def _table_names(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}


def _count_table(conn: duckdb.DuckDBPyConnection, tables: set[str], table: str) -> str:
    if table not in tables:
        return "missing"
    try:
        return str(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception as exc:
        return f"error: {exc}"


def _provider_distribution(conn: duckdb.DuckDBPyConnection, tables: set[str]) -> list[tuple[str, int]]:
    if "financial_facts" not in tables:
        return []
    try:
        rows = conn.execute(
            """
            SELECT COALESCE(NULLIF(source_provider, ''), 'unknown') AS provider, COUNT(*) AS count
            FROM financial_facts
            GROUP BY 1
            ORDER BY count DESC, provider
            """
        ).fetchall()
        return [(str(provider), int(count)) for provider, count in rows]
    except Exception as exc:
        return [(f"error: {exc}", 0)]


def _print_duckdb_status() -> None:
    db_path = settings.duckdb_path
    _print_kv("DuckDB path", db_path)
    _print_kv("DuckDB exists", db_path.exists())
    if not db_path.exists():
        print("WARNING: DuckDB file is missing. Run the data pipeline before integration/eval commands.")
        return

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception as exc:
        print(f"WARNING: Could not open DuckDB: {exc}")
        return

    try:
        tables = _table_names(conn)
        for table in (
            "companies",
            "filings_metadata",
            "filing_chunks",
            "financial_facts",
            "price_history",
            "filing_events",
            "event_price_windows",
        ):
            _print_kv(f"{table} count", _count_table(conn, tables, table))

        providers = _provider_distribution(conn, tables)
        if providers:
            print("financial_facts source_provider distribution")
            for provider, count in providers:
                print(f"  - {provider}: {count}")
        else:
            print("financial_facts source_provider distribution: unavailable")
    finally:
        conn.close()


def _collection_name(collection: Any) -> str:
    if isinstance(collection, str):
        return collection
    return str(getattr(collection, "name", collection))


def _print_chroma_status() -> None:
    chroma_path = settings.chroma_dir
    _print_kv("Chroma path", chroma_path)
    _print_kv("Chroma exists", chroma_path.exists())
    _print_kv("Current rag_index_version", settings.rag_index_version)
    _print_kv("Configured collection v1", settings.rag_collection_v1)
    _print_kv("Configured collection v2", settings.rag_collection_v2)
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        print("WARNING: Chroma vectorstore is missing or empty. Retrieval eval/search_filings will be skipped or fail.")
        return

    try:
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collections = client.list_collections()
    except Exception as exc:
        print(f"WARNING: Could not inspect Chroma vectorstore: {exc}")
        return

    if not collections:
        print("WARNING: No Chroma collections found.")
        return

    print("Chroma collections")
    for collection_ref in collections:
        name = _collection_name(collection_ref)
        try:
            collection = client.get_collection(name)
            count = collection.count()
        except Exception as exc:
            count = f"error: {exc}"
        print(f"  - {name}: {count}")


def main() -> None:
    print("FilingLens Data Status")
    print("=" * 42)
    _print_duckdb_status()
    print()
    _print_chroma_status()


if __name__ == "__main__":
    main()
