"""query_event_price_window — structured filing-event price reaction lookup."""

from __future__ import annotations

from typing import Any, Literal, Optional

import duckdb
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import settings

WINDOW_SET = {1, 3, 5, 10}


class QueryEventPriceWindowInput(BaseModel):
    """Input schema for the query_event_price_window tool."""

    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL")
    event_type: Optional[Literal["10Q", "10K", "any"]] = Field(
        default="any",
        description="Event type filter: 10Q, 10K, or any.",
    )
    fiscal_period: str | None = Field(
        default=None,
        description="Optional fiscal period filter, e.g. 2025-12-31.",
    )
    event_date: str | None = Field(
        default=None,
        description="Optional exact filing event date (YYYY-MM-DD).",
    )
    latest_n: int | None = Field(
        default=4,
        description="Maximum number of latest events to return (1-20).",
    )
    window_days: Optional[list[Literal[1, 3, 5, 10]]] = Field(
        default=None,
        description="Requested reaction windows in trading days.",
    )
    sort_by: Optional[Literal["event_date", "return_abs"]] = Field(
        default="event_date",
        description="Sort events by event_date or absolute return magnitude.",
    )
    sort_order: Optional[Literal["asc", "desc"]] = Field(
        default="desc",
        description="Sort order for event_date sorting.",
    )


def _window_days_or_default(window_days: list[int] | None) -> list[int]:
    if not window_days:
        return [1, 3, 5, 10]
    out: list[int] = []
    for n in window_days:
        if int(n) in WINDOW_SET and int(n) not in out:
            out.append(int(n))
    return out or [1, 3, 5, 10]


def _serialize_event(row: dict[str, Any], window_days: list[int]) -> dict[str, Any]:
    prices: dict[str, Any] = {
        "t_minus_1": {"date": row.get("t_minus_1_date"), "close": row.get("t_minus_1_close")},
        "t_close": {"date": row.get("t_close_date"), "close": row.get("t_close")},
    }
    returns: dict[str, Any] = {}
    for n in window_days:
        prices[f"t_plus_{n}"] = {
            "date": row.get(f"t_plus_{n}_date"),
            "close": row.get(f"t_plus_{n}_close"),
        }
        returns[f"return_{n}d"] = row.get(f"return_{n}d")
    return {
        "ticker": row.get("ticker"),
        "filing_id": row.get("filing_id"),
        "event_type": row.get("event_type"),
        "form_type": row.get("form_type"),
        "fiscal_period": row.get("fiscal_period"),
        "event_date": row.get("event_date"),
        "trading_anchor_date": row.get("trading_anchor_date"),
        "anchor_rule": row.get("anchor_rule"),
        "coverage_flag": row.get("coverage_flag"),
        "prices": prices,
        "returns": returns,
    }


def _abs_return_score(row: dict[str, Any], window_days: list[int]) -> float:
    values: list[float] = []
    for n in window_days:
        raw = row.get(f"return_{n}d")
        try:
            if raw is not None:
                values.append(abs(float(raw)))
        except (TypeError, ValueError):
            continue
    return max(values) if values else -1.0


@tool("query_event_price_window", args_schema=QueryEventPriceWindowInput)
def query_event_price_window(
    ticker: str,
    event_type: str = "any",
    fiscal_period: str | None = None,
    event_date: str | None = None,
    latest_n: int | None = 4,
    window_days: list[int] | None = None,
    sort_by: str = "event_date",
    sort_order: str = "desc",
) -> dict[str, Any]:
    """Query precomputed post-filing price reaction windows from DuckDB.

    This tool is deterministic and reads only local structured event tables.
    """
    windows = _window_days_or_default(window_days)
    event_type = str(event_type or "any").upper()
    if event_type not in {"10Q", "10K", "ANY"}:
        event_type = "ANY"
    sort_by = str(sort_by or "event_date").lower()
    if sort_by not in {"event_date", "return_abs"}:
        sort_by = "event_date"
    sort_order = str(sort_order or "desc").lower()
    if sort_order not in {"asc", "desc"}:
        sort_order = "desc"
    top_n = min(max(int(latest_n or 4), 1), 20)

    conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        sql = """
        SELECT
            e.ticker,
            e.filing_id,
            e.event_type,
            e.form_type,
            e.fiscal_period,
            e.event_date,
            e.trading_anchor_date,
            e.anchor_rule,
            w.t_minus_1_date, w.t_minus_1_close,
            w.t_close_date, w.t_close,
            w.t_plus_1_date, w.t_plus_1_close,
            w.t_plus_3_date, w.t_plus_3_close,
            w.t_plus_5_date, w.t_plus_5_close,
            w.t_plus_10_date, w.t_plus_10_close,
            w.return_1d, w.return_3d, w.return_5d, w.return_10d,
            w.coverage_flag
        FROM filing_events e
        LEFT JOIN event_price_windows w
          ON e.ticker = w.ticker AND e.filing_id = w.filing_id
        WHERE e.ticker = ?
        """
        params: list[Any] = [ticker.upper()]
        if event_type != "ANY":
            sql += " AND e.event_type = ?"
            params.append(event_type)
        if fiscal_period:
            sql += " AND e.fiscal_period = ?"
            params.append(fiscal_period)
        if event_date:
            sql += " AND e.event_date = CAST(? AS DATE)"
            params.append(event_date)
        sql += " ORDER BY e.event_date DESC LIMIT 200"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    keys = [
        "ticker",
        "filing_id",
        "event_type",
        "form_type",
        "fiscal_period",
        "event_date",
        "trading_anchor_date",
        "anchor_rule",
        "t_minus_1_date",
        "t_minus_1_close",
        "t_close_date",
        "t_close",
        "t_plus_1_date",
        "t_plus_1_close",
        "t_plus_3_date",
        "t_plus_3_close",
        "t_plus_5_date",
        "t_plus_5_close",
        "t_plus_10_date",
        "t_plus_10_close",
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "coverage_flag",
    ]
    records: list[dict[str, Any]] = []
    for row in rows:
        item = {k: row[idx] for idx, k in enumerate(keys)}
        for k, v in list(item.items()):
            if hasattr(v, "isoformat"):
                item[k] = v.isoformat()
        records.append(item)

    if sort_by == "return_abs":
        records = sorted(records, key=lambda r: _abs_return_score(r, windows), reverse=True)
    else:
        reverse = sort_order == "desc"
        records = sorted(records, key=lambda r: str(r.get("event_date", "")), reverse=reverse)

    records = records[:top_n]
    events = [_serialize_event(r, windows) for r in records]
    return {
        "ticker": ticker.upper(),
        "events": events,
        "applied_filters": {
            "event_type": event_type.lower(),
            "fiscal_period": fiscal_period,
            "event_date": event_date,
            "latest_n": top_n,
            "window_days": windows,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
    }
