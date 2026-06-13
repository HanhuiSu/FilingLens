#!/usr/bin/env python3
"""Build filing_events from filings_metadata and price_history."""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.queries import (  # noqa: E402
    FilingEventRow,
    clear_filing_events_for_ticker,
    get_connection,
    upsert_filing_events,
)


def _normalize_form(form_type: str) -> tuple[str, str]:
    normalized = str(form_type or "").upper().replace(" ", "")
    if normalized == "10-Q":
        normalized = "10-Q"
    if normalized == "10-K":
        normalized = "10-K"
    event_type = "10Q" if normalized == "10-Q" else "10K"
    return normalized, event_type


def _load_trade_dates(conn, ticker: str) -> list[date]:
    rows = conn.execute(
        """
        SELECT date
        FROM price_history
        WHERE ticker = ?
        ORDER BY date
        """,
        [ticker.upper()],
    ).fetchall()
    return [r[0] for r in rows if isinstance(r[0], date)]


def _next_trade_date(trade_dates: list[date], target: date) -> date | None:
    if not trade_dates:
        return None
    idx = bisect.bisect_left(trade_dates, target)
    if idx >= len(trade_dates):
        return None
    return trade_dates[idx]


def main() -> None:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT filing_id, ticker, form_type, fiscal_period, filing_date
        FROM filings_metadata
        WHERE filing_date IS NOT NULL
          AND UPPER(REPLACE(form_type, ' ', '')) IN ('10-K', '10-Q')
        ORDER BY ticker, filing_date DESC
        """
    ).fetchall()

    by_ticker: dict[str, list[tuple[str, str, str | None, date]]] = defaultdict(list)
    for filing_id, ticker, form_type, fiscal_period, filing_date in rows:
        if not filing_id or not ticker or not isinstance(filing_date, date):
            continue
        by_ticker[str(ticker).upper()].append(
            (str(filing_id), str(form_type or ""), str(fiscal_period) if fiscal_period else None, filing_date)
        )

    total = 0
    for ticker, items in sorted(by_ticker.items()):
        trade_dates = _load_trade_dates(conn, ticker)
        clear_filing_events_for_ticker(conn, ticker)
        batch: list[FilingEventRow] = []
        for filing_id, form_type, fiscal_period, filing_date in items:
            normalized_form, event_type = _normalize_form(form_type)
            anchor_date = _next_trade_date(trade_dates, filing_date)
            batch.append(
                FilingEventRow(
                    ticker=ticker,
                    filing_id=filing_id,
                    event_type=event_type,
                    event_date=filing_date,
                    fiscal_period=fiscal_period,
                    form_type=normalized_form,
                    filing_date=filing_date,
                    trading_anchor_date=anchor_date,
                    anchor_rule="filing_date_then_next_trading_day",
                    has_price_data=anchor_date is not None,
                )
            )
        upsert_filing_events(conn, batch)
        total += len(batch)
        print(f"{ticker}: {len(batch)} filing events")

    conn.close()
    print(f"Done. upserted filing_events rows: {total}")


if __name__ == "__main__":
    main()
