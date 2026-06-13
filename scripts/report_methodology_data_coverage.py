#!/usr/bin/env python3
"""Report methodology metric coverage from DuckDB financial_facts.

The script is intentionally read-only. It answers the practical question:
which dimensions can the methodology actually support for each ticker with
the facts currently loaded locally?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from src.agent.metric_availability import (
    DERIVED_METRIC_DEPENDENCIES,
    DIMENSION_CORE_METRICS,
    DIMENSION_ENHANCED_METRICS,
    MetricAvailabilityResolver,
    normalize_metric_name,
    raw_metric_names,
)


DIMENSION_METRICS: dict[str, list[str]] = {
    dimension_id: list(dict.fromkeys([*DIMENSION_CORE_METRICS.get(dimension_id, ()), *DIMENSION_ENHANCED_METRICS.get(dimension_id, ())]))
    for dimension_id in sorted(set(DIMENSION_CORE_METRICS) | set(DIMENSION_ENHANCED_METRICS))
}

SPOTLIGHT_TICKERS = ("NVDA", "AAPL", "AMZN", "MSFT")


def _metric_keys(metric: str) -> list[str]:
    return list(raw_metric_names(metric))


def _fact_rows(conn: duckdb.DuckDBPyConnection, ticker: str, metrics: list[str]) -> list[dict[str, Any]]:
    if not metrics:
        return []
    placeholders = ", ".join(["?"] * len(metrics))
    rows = conn.execute(
        f"""
        SELECT metric, period_end, period_type, COALESCE(source_provider, 'unknown') AS source_provider,
               COALESCE(confidence, '') AS confidence, COALESCE(extraction_method, '') AS extraction_method,
               COUNT(*) AS row_count
        FROM financial_facts
        WHERE ticker = ? AND metric IN ({placeholders})
        GROUP BY metric, period_end, period_type, source_provider, confidence, extraction_method
        ORDER BY period_end DESC
        """,
        [ticker.upper(), *metrics],
    ).fetchall()
    return [
        {
            "metric": row[0],
            "period_end": str(row[1]),
            "period_type": row[2],
            "source_provider": row[3],
            "confidence": row[4],
            "extraction_method": row[5],
            "row_count": int(row[6] or 0),
        }
        for row in rows
    ]


def _latest_price(conn: duckdb.DuckDBPyConnection, ticker: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT date, adjusted_close, close
        FROM price_history
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        [ticker.upper()],
    ).fetchone()
    if not row:
        return None
    value = row[1] if row[1] is not None else row[2]
    if value is None:
        return None
    return {
        "metric": "share_price",
        "period_end": str(row[0]),
        "period_type": "daily",
        "source_provider": "yfinance",
        "confidence": "medium",
        "extraction_method": "api_price_history",
        "row_count": 1,
    }


def _direct_metric_rows(conn: duckdb.DuckDBPyConnection, ticker: str, metric: str) -> list[dict[str, Any]]:
    if metric == "share_price":
        price = _latest_price(conn, ticker)
        return [price] if price else []
    return _fact_rows(conn, ticker, _metric_keys(metric))


def _coverage_metric_rows(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    metric: str,
    *,
    seen: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = _direct_metric_rows(conn, ticker, metric)
    if rows:
        return rows

    seen = set(seen or set())
    if metric in seen:
        return []
    seen.add(metric)
    dependency_metrics = DERIVED_METRIC_DEPENDENCIES.get(metric)
    if not dependency_metrics:
        return []

    dependencies: list[dict[str, Any]] = []
    latest_period = ""
    for dependency_metric in dependency_metrics:
        dependency_rows = _coverage_metric_rows(conn, ticker, dependency_metric, seen=seen)
        if not dependency_rows:
            return []
        primary = dict(dependency_rows[0])
        primary["dependency_metric"] = dependency_metric
        dependencies.append(primary)
        period = str(primary.get("period_end") or "")
        if period > latest_period:
            latest_period = period

    return [
        {
            "metric": metric,
            "period_end": latest_period,
            "period_type": "computed",
            "source_provider": "computed",
            "confidence": "derived",
            "extraction_method": "programmatic_dependency_check",
            "row_count": 1,
            "dependencies": dependencies,
        }
    ]


def _coverage_for_dimension(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    dimension_id: str,
    expected_metrics: list[str],
) -> dict[str, Any]:
    resolver = MetricAvailabilityResolver.from_duckdb(conn, ticker, expected_metrics)
    available: list[str] = []
    missing: list[str] = []
    provider_counts: Counter[str] = Counter()
    latest_periods: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    metric_sources: dict[str, list[dict[str, Any]]] = {}

    for metric in expected_metrics:
        canonical = normalize_metric_name(metric)
        availability = resolver.availability(canonical)
        if availability.available:
            rows = resolver.source_rows([canonical]).get(canonical, [])
            available.append(canonical)
            latest_periods[canonical] = str(rows[0].get("period_end") or "")
            row_counts[canonical] = sum(int(row.get("row_count", 0) or 0) for row in rows)
            metric_sources[canonical] = rows[:6]
            for row in rows:
                provider_counts[str(row.get("source_provider") or "unknown")] += int(row.get("row_count", 1) or 1)
        else:
            missing.append(canonical)

    status = _dimension_status(dimension_id, available)
    return {
        "dimension_id": dimension_id,
        "status": status,
        "available_metrics": available,
        "missing_metrics": missing,
        "coverage_rate": round(len(available) / max(len(expected_metrics), 1), 6),
        "provider_counts": dict(provider_counts),
        "latest_periods": latest_periods,
        "row_counts": row_counts,
        "metric_sources": metric_sources,
    }


def _dimension_status(dimension_id: str, available_metrics: list[str]) -> str:
    available = set(available_metrics)
    if dimension_id == "revenue_quality":
        return "satisfied" if "revenue" in available else "missing"
    if dimension_id == "profitability_quality":
        has_income = "net_income" in available
        margin_count = len({"gross_margin", "operating_margin", "net_margin"} & available)
        profit_detail_count = len({"gross_profit", "operating_income"} & available)
        if has_income and margin_count >= 2 and profit_detail_count >= 1:
            return "satisfied"
        if has_income or margin_count:
            return "partial"
        return "missing"
    if dimension_id == "cash_flow_quality":
        if {"operating_cash_flow", "free_cash_flow", "cash_conversion"}.issubset(available):
            return "satisfied"
        if {"operating_cash_flow", "free_cash_flow"} & available:
            return "partial"
        return "missing"
    if dimension_id == "balance_sheet_and_capital_intensity":
        if {"cash", "total_debt", "total_assets", "total_liabilities"}.issubset(available):
            return "satisfied"
        if {"cash", "total_debt", "total_assets", "total_liabilities", "shareholders_equity"} & available:
            return "partial"
        return "missing"
    if dimension_id == "valuation_and_risk_boundary":
        has_price = "share_price" in available
        has_market_cap = "market_cap" in available
        has_multiple = bool({"pe_ratio", "ps_ratio", "fcf_yield"} & available)
        if has_market_cap and {"pe_ratio", "ps_ratio"}.issubset(available):
            return "satisfied"
        if has_price and has_multiple:
            return "partial"
        if has_market_cap and has_multiple:
            return "partial"
        return "missing"
    return "satisfied" if available else "missing"


def build_coverage_report(
    *,
    db_path: Path | None = None,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    path = Path(db_path or settings.duckdb_path)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB file does not exist: {path}")
    conn = duckdb.connect(str(path), read_only=True)
    try:
        selected_tickers = [ticker.upper() for ticker in (tickers or settings.target_tickers)]
        companies: dict[str, Any] = {}
        for ticker in selected_tickers:
            dimension_reports = {
                dimension_id: _coverage_for_dimension(conn, ticker, dimension_id, metrics)
                for dimension_id, metrics in DIMENSION_METRICS.items()
            }
            available_count = sum(len(item["available_metrics"]) for item in dimension_reports.values())
            expected_count = sum(len(metrics) for metrics in DIMENSION_METRICS.values())
            companies[ticker] = {
                "ticker": ticker,
                "overall_metric_coverage_rate": round(available_count / max(expected_count, 1), 6),
                "dimensions": dimension_reports,
            }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db_path": str(path),
            "tickers": selected_tickers,
            "spotlight_tickers": {
                ticker: companies.get(ticker)
                for ticker in SPOTLIGHT_TICKERS
                if ticker in companies
            },
            "dimension_metric_requirements": DIMENSION_METRICS,
            "companies": companies,
        }
    finally:
        conn.close()


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _snapshot_path(snapshot: str, reports_dir: Path) -> Path:
    return reports_dir / f"methodology_data_coverage_{snapshot}.json"


def _status_value(report: dict[str, Any], ticker: str, dimension_id: str) -> str:
    return str(
        ((report.get("companies", {}) or {}).get(ticker, {}) or {})
        .get("dimensions", {})
        .get(dimension_id, {})
        .get("status", "missing")
    )


def _metrics_value(report: dict[str, Any], ticker: str, dimension_id: str, key: str) -> set[str]:
    return {
        str(item)
        for item in (
            ((report.get("companies", {}) or {}).get(ticker, {}) or {})
            .get("dimensions", {})
            .get(dimension_id, {})
            .get(key, [])
            or []
        )
        if str(item)
    }


def _diff_note(before_status: str, after_status: str, newly_available: set[str], still_missing: set[str]) -> str:
    if after_status != before_status:
        return f"{before_status} -> {after_status}"
    if newly_available:
        return "metrics improved without status change"
    if still_missing:
        return "still missing required metrics"
    return "unchanged"


def build_coverage_diff(before: dict[str, Any], after: dict[str, Any]) -> str:
    tickers = [
        ticker
        for ticker in SPOTLIGHT_TICKERS
        if ticker in (before.get("companies", {}) or {}) or ticker in (after.get("companies", {}) or {})
    ]
    if not tickers:
        tickers = sorted(set((before.get("companies", {}) or {}).keys()) | set((after.get("companies", {}) or {}).keys()))
    lines = [
        "# Methodology Data Coverage Diff",
        "",
        "| Ticker | Dimension | Before | After | Newly Available Metrics | Still Missing Metrics | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for ticker in tickers:
        for dimension_id in DIMENSION_METRICS:
            before_status = _status_value(before, ticker, dimension_id)
            after_status = _status_value(after, ticker, dimension_id)
            before_available = _metrics_value(before, ticker, dimension_id, "available_metrics")
            after_available = _metrics_value(after, ticker, dimension_id, "available_metrics")
            after_missing = _metrics_value(after, ticker, dimension_id, "missing_metrics")
            newly_available = after_available - before_available
            note = _diff_note(before_status, after_status, newly_available, after_missing)
            lines.append(
                "| {ticker} | {dimension} | {before_status} | {after_status} | {new_metrics} | {missing} | {note} |".format(
                    ticker=ticker,
                    dimension=dimension_id,
                    before_status=before_status,
                    after_status=after_status,
                    new_metrics=", ".join(sorted(newly_available)) or "-",
                    missing=", ".join(sorted(after_missing)) or "-",
                    note=note,
                )
            )
    return "\n".join(lines) + "\n"


def write_coverage_diff(before_path: Path, after_path: Path, output_path: Path) -> bool:
    if not before_path.exists() or not after_path.exists():
        return False
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_coverage_diff(before, after), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Report methodology data coverage from DuckDB.")
    parser.add_argument("--ticker", action="append", help="Ticker to include. Can be repeated. Defaults to target_tickers.")
    parser.add_argument("--db-path", type=Path, default=settings.duckdb_path, help="DuckDB path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "reports" / "methodology_data_coverage.json",
        help="JSON report path.",
    )
    parser.add_argument("--snapshot", choices=["before", "after"], help="Also write the named before/after snapshot.")
    parser.add_argument(
        "--diff-output",
        type=Path,
        default=settings.data_dir / "reports" / "methodology_data_coverage_diff.md",
        help="Markdown diff path written when --snapshot after and before snapshot exists.",
    )
    args = parser.parse_args()

    report = build_coverage_report(db_path=args.db_path, tickers=args.ticker)
    write_report(report, args.output)
    print(f"Wrote methodology data coverage report to {args.output}")
    if args.snapshot:
        snapshot_output = _snapshot_path(args.snapshot, args.output.parent)
        write_report(report, snapshot_output)
        print(f"Wrote {args.snapshot} coverage snapshot to {snapshot_output}")
        if args.snapshot == "after":
            before_path = _snapshot_path("before", args.output.parent)
            if write_coverage_diff(before_path, snapshot_output, args.diff_output):
                print(f"Wrote methodology data coverage diff to {args.diff_output}")


if __name__ == "__main__":
    main()
