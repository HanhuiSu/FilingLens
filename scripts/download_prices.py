#!/usr/bin/env python3
"""Download daily OHLCV from yfinance into DuckDB price_history."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings

from src.db.queries import PriceHistoryRow, clear_price_history_for_ticker, get_connection, insert_price_history_batch


def fetch_prices(db_ticker: str, yf_symbol: str, start: date) -> list[PriceHistoryRow]:
    t = yf.Ticker(yf_symbol)
    hist = t.history(start=start.isoformat(), interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        return []
    rows: list[PriceHistoryRow] = []
    for idx, row in hist.iterrows():
        d = idx.date() if hasattr(idx, "date") else idx
        if not isinstance(d, date):
            d = d.to_pydatetime().date()
        adj = row.get("Adj Close")
        if adj is None or (hasattr(adj, "__float__") and str(adj) == "nan"):
            adj = float(row["Close"]) if "Close" in row and row["Close"] == row["Close"] else None
        else:
            adj = float(adj)
        rows.append(
            PriceHistoryRow(
                ticker=db_ticker.upper(),
                d=d,
                open=float(row["Open"]) if row["Open"] == row["Open"] else None,
                high=float(row["High"]) if row["High"] == row["High"] else None,
                low=float(row["Low"]) if row["Low"] == row["Low"] else None,
                close=float(row["Close"]) if row["Close"] == row["Close"] else None,
                adjusted_close=adj,
                volume=float(row["Volume"]) if row["Volume"] == row["Volume"] else None,
            )
        )
    return rows


def main() -> None:
    start = date.today() - timedelta(days=365 * settings.data_years + 30)
    conn = get_connection()
    for ticker in settings.target_tickers:
        db_t = ticker.upper()
        yf_sym = "GOOG" if db_t == "GOOGL" else db_t
        rows = fetch_prices(db_t, yf_sym, start)
        if not rows and db_t == "GOOGL":
            rows = fetch_prices(db_t, "GOOGL", start)
        if not rows:
            print(f"  No price data for {ticker} — keeping existing data")
            continue
        clear_price_history_for_ticker(conn, db_t)
        insert_price_history_batch(conn, rows)
        print(f"  {db_t}: {len(rows)} price rows")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
