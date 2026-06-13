"""V1.1 event-window capability tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import importlib

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import init_db  # noqa: E402


def _prepare_event_db(path: Path) -> None:
    conn = duckdb.connect(str(path))
    init_db(conn)
    conn.execute(
        """
        INSERT INTO filing_events
        (ticker, filing_id, event_type, event_date, fiscal_period, form_type, filing_date,
         trading_anchor_date, anchor_rule, has_price_data)
        VALUES
        ('AAPL','f1','10Q',DATE '2025-01-31','2024-12-31','10-Q',DATE '2025-01-31',DATE '2025-01-31','filing_date_then_next_trading_day',TRUE),
        ('AAPL','f2','10Q',DATE '2024-11-01','2024-09-30','10-Q',DATE '2024-11-01',DATE '2024-11-01','filing_date_then_next_trading_day',TRUE)
        """
    )
    conn.execute(
        """
        INSERT INTO event_price_windows
        (ticker, filing_id, event_date, trading_anchor_date,
         t_minus_1_date, t_minus_1_close, t_close_date, t_close,
         t_plus_1_date, t_plus_1_close, t_plus_3_date, t_plus_3_close,
         t_plus_5_date, t_plus_5_close, t_plus_10_date, t_plus_10_close,
         return_1d, return_3d, return_5d, return_10d, coverage_flag)
        VALUES
        ('AAPL','f1',DATE '2025-01-31',DATE '2025-01-31',
         DATE '2025-01-30',190.0,DATE '2025-01-31',192.0,
         DATE '2025-02-03',195.0,DATE '2025-02-05',194.0,
         DATE '2025-02-07',198.0,DATE '2025-02-14',200.0,
         0.015625,0.010417,0.03125,0.041667,'complete'),
        ('AAPL','f2',DATE '2024-11-01',DATE '2024-11-01',
         DATE '2024-10-31',170.0,DATE '2024-11-01',168.0,
         DATE '2024-11-04',166.0,DATE '2024-11-06',169.0,
         DATE '2024-11-08',165.0,DATE '2024-11-15',162.0,
         -0.011905,0.005952,-0.017857,-0.035714,'complete')
        """
    )
    conn.close()


def test_query_event_price_window_tool(tmp_path, monkeypatch):
    tool_mod = importlib.import_module("src.tools.query_event_price_window")

    db_path = tmp_path / "event_tool.duckdb"
    _prepare_event_db(db_path)

    class DummySettings:
        duckdb_path = db_path

    monkeypatch.setattr(tool_mod, "settings", DummySettings())

    out = tool_mod.query_event_price_window.invoke(
        {
            "ticker": "AAPL",
            "event_type": "10Q",
            "latest_n": 1,
            "window_days": [1, 5],
            "sort_by": "event_date",
            "sort_order": "desc",
        }
    )
    assert out["ticker"] == "AAPL"
    assert len(out["events"]) == 1
    event = out["events"][0]
    assert event["event_type"] == "10Q"
    assert "return_1d" in event["returns"]
    assert "return_5d" in event["returns"]


def test_execute_tools_runs_event_query_first(monkeypatch):
    from src.agent import nodes

    call_order: list[str] = []

    class DummyEventTool:
        @staticmethod
        def invoke(_payload):
            call_order.append("event")
            return {
                "ticker": "AAPL",
                "events": [
                    {
                        "ticker": "AAPL",
                        "filing_id": "f1",
                        "event_type": "10Q",
                        "form_type": "10-Q",
                        "fiscal_period": "2024-12-31",
                        "event_date": "2025-01-31",
                        "trading_anchor_date": "2025-01-31",
                        "anchor_rule": "filing_date_then_next_trading_day",
                        "coverage_flag": "complete",
                        "returns": {"return_1d": 0.02, "return_5d": 0.04},
                        "prices": {},
                    }
                ],
                "applied_filters": {"latest_n": 1},
            }

    class DummyFinancialTool:
        @staticmethod
        def invoke(_payload):
            call_order.append("financial")
            return {"ticker": "AAPL", "financial_facts": [], "price_data": [], "period_context": {}}

    monkeypatch.setattr(nodes, "query_event_price_window", DummyEventTool())
    monkeypatch.setattr(nodes, "query_financial_data", DummyFinancialTool())

    out = nodes.execute_tools(
        {
            "user_query": "Apple 最近一次财报后股价反应如何？",
            "task_type": "trend_analysis",
            "data_route": "hybrid",
            "companies": ["AAPL"],
            "comparison_target": None,
            "selected_tools": ["query_event_price_window", "query_financial_data"],
            "requested_metrics": ["adjusted_close"],
            "time_range": None,
            "tool_results": [],
            "retrieved_docs": [],
            "market_reaction_requested": True,
            "event_query": {"latest_n": 1, "window_days": [1, 5], "event_type": "any"},
            "period_query": {"period_type": "latest", "year_basis": "fiscal", "comparison_basis": "same_period"},
            "resolved_period_context": {"target_period_type": "quarterly", "strict_period_match": True},
        }
    )
    assert call_order[0] == "event"
    assert out["event_results"]
    assert out["market_reaction_evidence"]
    assert any(tr.get("tool") == "query_event_price_window" for tr in out["tool_results"])


def test_generate_answer_emits_market_reaction_block():
    from src.agent import nodes

    state = {
        "trace_id": "trace-event-1",
        "user_query": "Apple 最近一次 10-Q 后股价 1 天和 5 天反应如何？",
        "task_type": "trend_analysis",
        "selected_tools": ["query_event_price_window"],
        "market_reaction_requested": True,
        "event_query": {"event_type": "10Q", "latest_n": 1, "window_days": [1, 5]},
        "event_results": [
            {
                "ticker": "AAPL",
                "data": {
                    "ticker": "AAPL",
                    "events": [
                        {
                            "ticker": "AAPL",
                            "filing_id": "f1",
                            "event_type": "10Q",
                            "form_type": "10-Q",
                            "fiscal_period": "2024-12-31",
                            "event_date": "2025-01-31",
                            "trading_anchor_date": "2025-01-31",
                            "anchor_rule": "filing_date_then_next_trading_day",
                            "coverage_flag": "complete",
                            "returns": {"return_1d": 0.015, "return_5d": 0.031},
                            "prices": {},
                        }
                    ],
                },
            }
        ],
        "market_reaction_evidence": [],
        "market_reaction_limitations": [],
        "period_query": {"period_type": "latest", "year_basis": "fiscal", "comparison_basis": "same_period"},
        "resolved_period_context": {
            "target_period_type": "quarterly",
            "comparison_basis": "same_period",
            "needs_clarification": False,
            "strict_period_match": True,
        },
        "comparison_basis_label": "",
        "tool_results": [
            {
                "tool": "query_event_price_window",
                "ticker": "AAPL",
                "data": {
                    "ticker": "AAPL",
                    "events": [
                        {
                            "ticker": "AAPL",
                            "filing_id": "f1",
                            "event_type": "10Q",
                            "form_type": "10-Q",
                            "fiscal_period": "2024-12-31",
                            "event_date": "2025-01-31",
                            "trading_anchor_date": "2025-01-31",
                            "anchor_rule": "filing_date_then_next_trading_day",
                            "coverage_flag": "complete",
                            "returns": {"return_1d": 0.015, "return_5d": 0.031},
                            "prices": {},
                        }
                    ],
                },
            }
        ],
        "retrieved_docs": [],
    }

    out = nodes.generate_answer(state)
    payload = out["output"]
    assert payload["task_type"] == "trend_analysis"
    assert "market_reaction" in payload
    assert payload["market_reaction"]["events"]
    assert payload["market_reaction"]["highlights"]
    assert any(c.get("metric") == "post_return_1d" for c in out["numeric_citations"])


def test_build_filing_events_trade_day_alignment():
    from scripts.build_filing_events import _next_trade_date

    trade_dates = [date(2025, 1, 30), date(2025, 1, 31), date(2025, 2, 3)]
    assert _next_trade_date(trade_dates, date(2025, 1, 31)) == date(2025, 1, 31)
    assert _next_trade_date(trade_dates, date(2025, 2, 1)) == date(2025, 2, 3)


def test_market_reaction_intent_detection_with_spaced_chinese_phrase():
    from src.agent.nodes import _is_market_reaction_query

    query = "Apple 最近一次 10-Q 后股价 1 天和 5 天反应如何？"
    assert _is_market_reaction_query(query) is True


def test_market_reaction_intent_detection_with_stock_and_return_words():
    from src.agent.nodes import _is_market_reaction_query

    query = "How did the stock price return after the latest 10-Q?"
    assert _is_market_reaction_query(query) is True
