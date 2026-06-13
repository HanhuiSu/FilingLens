#!/usr/bin/env python3
"""Build event_price_windows from filing_events and price_history."""

from __future__ import annotations

import bisect
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.queries import (  # noqa: E402
    EventPriceWindowRow,
    clear_event_price_windows_for_ticker,
    get_connection,
    upsert_event_price_windows,
)

WINDOW_OFFSETS = (1, 3, 5, 10)


def _load_price_series(conn, ticker: str) -> tuple[list[date], dict[date, float | None]]:
    rows = conn.execute(
        """
        SELECT date, adjusted_close, close
        FROM price_history
        WHERE ticker = ?
        ORDER BY date
        """,
        [ticker.upper()],
    ).fetchall()
    dates: list[date] = []
    closes: dict[date, float | None] = {}
    for d, adjusted_close, close in rows:
        if not isinstance(d, date):
            continue
        price = adjusted_close if adjusted_close is not None else close
        price_value = float(price) if price is not None else None
        dates.append(d)
        closes[d] = price_value
    return dates, closes


def _next_trade_idx(trade_dates: list[date], target: date) -> int | None:
    if not trade_dates:
        return None
    idx = bisect.bisect_left(trade_dates, target)
    if idx >= len(trade_dates):
        return None
    return idx


def _ret(base: float | None, future: float | None) -> float | None:
    if base is None or future is None or base == 0:
        return None
    return (future / base) - 1.0


def _coverage_flag(values: dict[int, float | None], anchor_price: float | None) -> str:
    if anchor_price is None:
        return "no_anchor_price"
    missing = [f"t+{n}" for n in WINDOW_OFFSETS if values.get(n) is None]
    if not missing:
        return "complete"
    return "partial_missing:" + ",".join(missing)


def main() -> None:
    conn = get_connection()
    events = conn.execute(
        """
        SELECT ticker, filing_id, event_date, trading_anchor_date
        FROM filing_events
        ORDER BY ticker, event_date DESC
        """
    ).fetchall()

    by_ticker: dict[str, list[tuple[str, date, date | None]]] = defaultdict(list)
    for ticker, filing_id, event_date, trading_anchor_date in events:
        if not ticker or not filing_id or not isinstance(event_date, date):
            continue
        by_ticker[str(ticker).upper()].append((str(filing_id), event_date, trading_anchor_date))

    total = 0
    for ticker, items in sorted(by_ticker.items()):
        trade_dates, closes = _load_price_series(conn, ticker)
        clear_event_price_windows_for_ticker(conn, ticker)
        rows: list[EventPriceWindowRow] = []
        for filing_id, event_date, trading_anchor_date in items:
            idx = _next_trade_idx(trade_dates, trading_anchor_date or event_date)
            if idx is None:
                rows.append(
                    EventPriceWindowRow(
                        ticker=ticker,
                        filing_id=filing_id,
                        event_date=event_date,
                        trading_anchor_date=None,
                        t_minus_1_date=None,
                        t_minus_1_close=None,
                        t_close_date=None,
                        t_close=None,
                        t_plus_1_date=None,
                        t_plus_1_close=None,
                        t_plus_3_date=None,
                        t_plus_3_close=None,
                        t_plus_5_date=None,
                        t_plus_5_close=None,
                        t_plus_10_date=None,
                        t_plus_10_close=None,
                        return_1d=None,
                        return_3d=None,
                        return_5d=None,
                        return_10d=None,
                        coverage_flag="no_trade_after_event",
                    )
                )
                continue

            anchor_date = trade_dates[idx]
            anchor_price = closes.get(anchor_date)
            t_minus_1_date = trade_dates[idx - 1] if idx - 1 >= 0 else None
            t_minus_1_close = closes.get(t_minus_1_date) if t_minus_1_date else None

            plus_dates: dict[int, date | None] = {}
            plus_close: dict[int, float | None] = {}
            for offset in WINDOW_OFFSETS:
                j = idx + offset
                if j < len(trade_dates):
                    d = trade_dates[j]
                    plus_dates[offset] = d
                    plus_close[offset] = closes.get(d)
                else:
                    plus_dates[offset] = None
                    plus_close[offset] = None

            rows.append(
                EventPriceWindowRow(
                    ticker=ticker,
                    filing_id=filing_id,
                    event_date=event_date,
                    trading_anchor_date=anchor_date,
                    t_minus_1_date=t_minus_1_date,
                    t_minus_1_close=t_minus_1_close,
                    t_close_date=anchor_date,
                    t_close=anchor_price,
                    t_plus_1_date=plus_dates[1],
                    t_plus_1_close=plus_close[1],
                    t_plus_3_date=plus_dates[3],
                    t_plus_3_close=plus_close[3],
                    t_plus_5_date=plus_dates[5],
                    t_plus_5_close=plus_close[5],
                    t_plus_10_date=plus_dates[10],
                    t_plus_10_close=plus_close[10],
                    return_1d=_ret(anchor_price, plus_close[1]),
                    return_3d=_ret(anchor_price, plus_close[3]),
                    return_5d=_ret(anchor_price, plus_close[5]),
                    return_10d=_ret(anchor_price, plus_close[10]),
                    coverage_flag=_coverage_flag(plus_close, anchor_price),
                )
            )

        upsert_event_price_windows(conn, rows)
        total += len(rows)
        print(f"{ticker}: {len(rows)} event windows")

    conn.close()
    print(f"Done. upserted event_price_windows rows: {total}")


if __name__ == "__main__":
    main()
