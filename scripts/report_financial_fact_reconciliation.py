#!/usr/bin/env python3
"""Report SEC/yfinance financial fact reconciliation.

The report compares provider rows for the same ticker / period_type /
period_end / metric. It never changes fact values; when requested, it writes
only reconciliation_warning metadata back to financial_facts so downstream
evidence and trace can surface provider conflicts.
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
from src.db.schema import init_db


RECONCILIATION_METRICS = [
    "revenue",
    "net_income",
    "gross_profit",
    "operating_income",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "cash_and_equivalents",
    "cash",
    "total_debt",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "shares_outstanding",
]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_diff(sec_value: float | None, yf_value: float | None) -> float | None:
    if sec_value is None or yf_value is None:
        return None
    denominator = abs(sec_value)
    if denominator == 0:
        return 0.0 if yf_value == 0 else None
    return abs(sec_value - yf_value) / denominator


def _warning_for(sec_row: dict[str, Any], yf_row: dict[str, Any]) -> tuple[str | None, str | None]:
    warnings: list[str] = []
    severities: list[str] = []
    sec_unit = str(sec_row.get("unit") or "").strip()
    yf_unit = str(yf_row.get("unit") or "").strip()
    if sec_unit and yf_unit and sec_unit != yf_unit:
        warnings.append("unit_mismatch")
        severities.append("high")
    diff = _pct_diff(_safe_float(sec_row.get("value")), _safe_float(yf_row.get("value")))
    if diff is not None:
        if diff > 0.05:
            warnings.append("value_mismatch_gt_5pct")
            severities.append("high")
        elif diff > 0.02:
            warnings.append("value_mismatch_gt_2pct")
            severities.append("medium")
    if not warnings:
        return None, None
    severity = "high" if "high" in severities else "medium"
    return severity, ";".join(warnings)


def _rows(
    conn: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    placeholders_tickers = ", ".join(["?"] * len(tickers))
    placeholders_metrics = ", ".join(["?"] * len(metrics))
    rows = conn.execute(
        f"""
        SELECT ticker, period_end, period_type, metric, value, unit,
               COALESCE(source_provider, 'yfinance') AS source_provider,
               COALESCE(confidence, '') AS confidence,
               COALESCE(extraction_method, '') AS extraction_method,
               COALESCE(source_tag, '') AS source_tag,
               COALESCE(source_filing_id, '') AS source_filing_id,
               COALESCE(reconciliation_warning, '') AS reconciliation_warning
        FROM financial_facts
        WHERE ticker IN ({placeholders_tickers})
          AND metric IN ({placeholders_metrics})
          AND COALESCE(source_provider, 'yfinance') IN ('sec_companyfacts', 'yfinance')
        ORDER BY ticker, metric, period_end, period_type, source_provider
        """,
        [*tickers, *metrics],
    ).fetchall()
    return [
        {
            "ticker": str(row[0]).upper(),
            "period": str(row[1]),
            "period_end": row[1],
            "period_type": str(row[2]),
            "metric": str(row[3]),
            "value": _safe_float(row[4]),
            "unit": str(row[5] or ""),
            "source_provider": str(row[6]),
            "confidence": str(row[7] or ""),
            "extraction_method": str(row[8] or ""),
            "source_tag": str(row[9] or ""),
            "source_filing_id": str(row[10] or ""),
            "reconciliation_warning": str(row[11] or ""),
        }
        for row in rows
    ]


def _best_provider_row(rows: list[dict[str, Any]], provider: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if str(row.get("source_provider") or "") == provider]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            str(row.get("confidence") or ""),
            str(row.get("source_filing_id") or ""),
            str(row.get("source_tag") or ""),
        ),
        reverse=True,
    )[0]


def build_reconciliation_report(
    *,
    db_path: Path | None = None,
    tickers: list[str] | None = None,
    metrics: list[str] | None = None,
    write_warnings: bool = False,
) -> dict[str, Any]:
    selected_tickers = [ticker.upper() for ticker in (tickers or settings.target_tickers)]
    selected_metrics = list(metrics or RECONCILIATION_METRICS)
    conn = duckdb.connect(str(db_path or settings.duckdb_path))
    init_db(conn)
    try:
        rows = _rows(conn, tickers=selected_tickers, metrics=selected_metrics)
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            key = (
                str(row["ticker"]),
                str(row["period"]),
                str(row["period_type"]),
                str(row["metric"]),
            )
            grouped.setdefault(key, []).append(row)

        records: list[dict[str, Any]] = []
        warning_counts: Counter[str] = Counter()
        for (ticker, period, period_type, metric), group_rows in grouped.items():
            sec = _best_provider_row(group_rows, "sec_companyfacts")
            yf = _best_provider_row(group_rows, "yfinance")
            if not sec or not yf:
                continue
            sec_value = _safe_float(sec.get("value"))
            yf_value = _safe_float(yf.get("value"))
            abs_diff = abs(sec_value - yf_value) if sec_value is not None and yf_value is not None else None
            pct_diff = _pct_diff(sec_value, yf_value)
            severity, warning = _warning_for(sec, yf)
            if warning:
                warning_counts[severity or "warning"] += 1
            record = {
                "ticker": ticker,
                "period": period,
                "period_type": period_type,
                "metric": metric,
                "sec_value": sec_value,
                "yfinance_value": yf_value,
                "absolute_diff": abs_diff,
                "pct_diff": pct_diff,
                "sec_unit": sec.get("unit"),
                "yfinance_unit": yf.get("unit"),
                "preferred_source": "sec",
                "warning_severity": severity,
                "warning": warning,
                "sec_source_tag": sec.get("source_tag"),
                "yfinance_source_tag": yf.get("source_tag"),
            }
            records.append(record)
            if write_warnings:
                _write_warning(conn, record)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path or settings.duckdb_path),
            "tickers": selected_tickers,
            "metrics": selected_metrics,
            "summary": {
                "compared_count": len(records),
                "warning_count": sum(warning_counts.values()),
                "medium_warning_count": warning_counts.get("medium", 0),
                "high_warning_count": warning_counts.get("high", 0),
            },
            "records": records,
        }
    finally:
        conn.close()


def _write_warning(conn: duckdb.DuckDBPyConnection, record: dict[str, Any]) -> None:
    warning = record.get("warning")
    conn.execute(
        """
        UPDATE financial_facts
        SET reconciliation_warning = ?
        WHERE ticker = ?
          AND period_end = ?
          AND period_type = ?
          AND metric = ?
          AND COALESCE(source_provider, 'yfinance') IN ('sec_companyfacts', 'yfinance')
        """,
        [
            warning,
            record["ticker"],
            record["period"],
            record["period_type"],
            record["metric"],
        ],
    )


def write_json(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fmt_pct(value: Any) -> str:
    pct = _safe_float(value)
    if pct is None:
        return "-"
    return f"{pct * 100:.2f}%"


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Financial Fact Reconciliation",
        "",
        f"- Compared rows: {report.get('summary', {}).get('compared_count', 0)}",
        f"- Warnings: {report.get('summary', {}).get('warning_count', 0)}",
        "",
        "| Ticker | Period | Metric | SEC | yfinance | Diff | Warning | Preferred |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for record in report.get("records", []) or []:
        lines.append(
            "| {ticker} | {period} {period_type} | {metric} | {sec} | {yf} | {diff} | {warning} | {preferred} |".format(
                ticker=record.get("ticker", ""),
                period=record.get("period", ""),
                period_type=record.get("period_type", ""),
                metric=record.get("metric", ""),
                sec=record.get("sec_value"),
                yf=record.get("yfinance_value"),
                diff=_fmt_pct(record.get("pct_diff")),
                warning=record.get("warning") or "-",
                preferred=record.get("preferred_source", "sec"),
            )
        )
    return "\n".join(lines) + "\n"


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report SEC/yfinance financial fact reconciliation.")
    parser.add_argument("--ticker", action="append", help="Ticker to include. Can be repeated. Defaults to target_tickers.")
    parser.add_argument("--metric", action="append", help="Metric to compare. Can be repeated. Defaults to methodology metrics.")
    parser.add_argument("--db-path", type=Path, default=settings.duckdb_path, help="DuckDB path.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=settings.data_dir / "reports" / "financial_fact_reconciliation.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=settings.data_dir / "reports" / "financial_fact_reconciliation.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--no-write-warnings",
        action="store_true",
        help="Do not write reconciliation_warning metadata back to financial_facts.",
    )
    args = parser.parse_args()

    report = build_reconciliation_report(
        db_path=args.db_path,
        tickers=args.ticker,
        metrics=args.metric,
        write_warnings=not args.no_write_warnings,
    )
    write_json(report, args.json_output)
    write_markdown(report, args.md_output)
    print(f"Wrote financial fact reconciliation JSON to {args.json_output}")
    print(f"Wrote financial fact reconciliation Markdown to {args.md_output}")


if __name__ == "__main__":
    main()
